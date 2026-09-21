import cv2
import os
import sys
import time
import math
from collections import defaultdict, deque
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "yolov8n.pt"
video_path = sys.argv[1] if len(sys.argv) > 1 else "street_walk.mp4"

YOLO_CONFIDENCE = 0.35
IMG_SIZE = 480
TRACKER = "bytetrack.yaml"

NORMAL_SPEECH_COOLDOWN = 3.0
EMERGENCY_SPEECH_COOLDOWN = 1.0

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_ENABLED = True
GEMINI_COOLDOWN = 4.0

IMPORTANT_OBJECTS = {"person", "bicycle", "motorcycle", "car", "bus", "truck"}
SECONDARY_OBJECTS = {"dog", "chair"}
ALL_OBJECTS = IMPORTANT_OBJECTS | SECONDARY_OBJECTS

AVERAGE_OBJECT_WIDTH_METERS = {
    "person": 0.45, "bicycle": 0.60, "motorcycle": 0.80,
    "car": 1.80, "bus": 2.50, "truck": 2.50, "dog": 0.40, "chair": 0.50,
}
APPROX_FOCAL_LENGTH_PIXELS = 700.0

def speak(text):
    print(f"\n[ASSISTANT AUDIO]: {text}\n")

# ============================================================
# GEMINI CLIENT INITIALIZATION
# ============================================================
gemini_client = None
if GEMINI_ENABLED:
    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            gemini_client = genai.Client(api_key=api_key)
            print("[SYSTEM] Gemini GenAI Engine: ENABLED")
        else:
            print("[SYSTEM] Gemini GenAI: DISABLED (GEMINI_API_KEY not set)")
    except Exception as e:
        print(f"[SYSTEM] Gemini setup error: {e}")

# ============================================================
# YOLO MODEL & VIDEO SOURCE
# ============================================================
print(f"[SYSTEM] Loading YOLOv8 and opening '{video_path}'...")
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video input '{video_path}'.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ============================================================
# STATE TRACKING
# ============================================================
previous_scene = []
last_speech_time = 0.0
last_emergency_time = 0.0
last_gemini_time = 0.0
last_message = ""
running_time = 0.0

track_history = defaultdict(lambda: deque(maxlen=8))
track_first_seen = {}
track_last_seen = {}

timeline_records = []

def format_time(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def center_of_box(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def calculate_distance(label, box_width):
    if label not in AVERAGE_OBJECT_WIDTH_METERS or box_width <= 5:
        return None
    real_w = AVERAGE_OBJECT_WIDTH_METERS[label]
    dist = (real_w * APPROX_FOCAL_LENGTH_PIXELS) / box_width
    return max(0.3, min(30.0, dist))

def format_distance(dist):
    if dist is None:
        return None
    if dist < 1.0:
        return "less than one meter"
    if dist < 10.0:
        return f"about {round(dist)} meters"
    return f"about {round(dist / 5) * 5} meters"

def get_zone(x_center):
    ratio = x_center / frame_width
    if ratio < 0.35:
        return "left"
    elif ratio > 0.65:
        return "right"
    return "center"

def calculate_motion(history):
    if len(history) < 3:
        return "moving ahead"
    old = history[0]
    new = history[-1]
    dx = new["x"] - old["x"]
    dw = new["width"] - old["width"]
    thresh = frame_width * 0.025

    h_move = "right" if dx > thresh else ("left" if dx < -thresh else None)
    v_move = "approaching" if dw > thresh else ("moving away" if dw < -thresh else None)

    if h_move and v_move:
        return f"moving toward your {h_move} and {v_move}"
    if h_move:
        return f"moving toward your {h_move}"
    if v_move:
        return v_move
    return "moving ahead"

def natural_count(label, count):
    if label == "person":
        return "one person" if count == 1 else (f"two people" if count == 2 else f"{count} people")
    return f"one {label}" if count == 1 else f"{count} {label}s"

def object_priority(obj):
    label = obj["label"]
    zone = obj["zone"]
    motion = obj["motion"]
    dist = obj["distance"]
    score = 50 if label in {"car", "motorcycle", "bus", "truck"} else (40 if label == "bicycle" else (25 if label == "person" else 5))
    if zone == "center":
        score += 30
    if "approaching" in motion:
        score += 30
    if dist and dist < 3.0:
        score += 40
    elif dist and dist < 6.0:
        score += 25
    return score

def extract_scene(result):
    objects = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0 or not boxes.is_track or boxes.id is None:
        return objects

    xyxy = boxes.xyxy.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    track_ids = boxes.id.int().cpu().tolist()

    for box, cls, conf, track_id in zip(xyxy, classes, confs, track_ids):
        label = model.names[int(cls)]
        if label not in ALL_OBJECTS:
            continue

        x1, y1, x2, y2 = box
        xc, yc = center_of_box(x1, y1, x2, y2)
        w, h = (x2 - x1), (y2 - y1)

        track_history[track_id].append({"x": xc, "y": yc, "width": w, "height": h})
        dist = calculate_distance(label, w)
        motion = calculate_motion(track_history[track_id])

        obj = {
            "track_id": int(track_id),
            "label": label,
            "confidence": float(conf),
            "zone": get_zone(xc),
            "motion": motion,
            "distance": dist,
            "distance_text": format_distance(dist),
        }
        obj["priority"] = object_priority(obj)
        objects.append(obj)
    return objects

def is_relevant(obj):
    label, zone, motion, dist = obj["label"], obj["zone"], obj["motion"], obj["distance"]
    if label in {"car", "motorcycle", "bus", "truck"}:
        return zone == "center" or "approaching" in motion or (dist and dist < 6.0)
    if label == "bicycle":
        return True
    if label == "person":
        return zone == "center" or "approaching" in motion or (dist and dist < 4.0)
    return False

def build_basic_description(important_objects):
    if not important_objects:
        return None
    top = important_objects[0]
    count = sum(1 for o in important_objects if o["label"] == top["label"] and o["zone"] == top["zone"])
    item_str = natural_count(top["label"], count)

    if top["zone"] == "center":
        dist_str = f" {top['distance_text']}" if top["distance_text"] else ""
        return f"Careful, {item_str} ahead of you{dist_str}, {top['motion']}."
    return f"{item_str.capitalize()} on your {top['zone']}, {top['motion']}."

def ask_gemini(frame, important_objects):
    global last_gemini_time
    if gemini_client is None:
        return None
    now = time.time()
    if now - last_gemini_time < GEMINI_COOLDOWN:
        return None
    last_gemini_time = now

    small_frame = cv2.resize(frame, (480, 360))
    success, encoded = cv2.imencode(".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if not success:
        return None

    prompt = f"""
You are the voice assistant for Sahayak Drishti smart cane.
The user is visually impaired. Summarize this immediate safety scene in 1 calm, concise sentence:
Objects: {[{'type': o['label'], 'zone': o['zone'], 'dist': o['distance_text'], 'motion': o['motion']} for o in important_objects]}
Never invent objects or numbers. Only describe what is in the structured data.
"""
    try:
        from google.genai import types
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=60)
        )
        if response.text and response.text.strip().upper() != "SILENT":
            return response.text.strip().replace("\n", " ")
    except Exception:
        pass
    return None

def is_emergency(objects):
    for obj in objects:
        if obj["zone"] == "center":
            if obj["label"] in {"car", "motorcycle", "bus", "truck"} and (obj["distance"] is None or obj["distance"] < 5.0):
                return True
            if obj["label"] == "person" and obj["distance"] and obj["distance"] < 2.0:
                return True
    return False

# ============================================================
# MAIN INFERENCE LOOP
# ============================================================
frame_count = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_sec = frame_count / fps

        # Skip every 2nd frame for CPU speed
        if frame_count % 2 != 0:
            continue

        percent = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
        sys.stdout.write(f"\rAnalyzing: {percent:3d}% ({format_time(current_sec)})")
        sys.stdout.flush()

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER,
            conf=YOLO_CONFIDENCE,
            imgsz=IMG_SIZE,
            verbose=False
        )

        objects = extract_scene(results[0])
        important = sorted([o for o in objects if is_relevant(o)], key=lambda x: x["priority"], reverse=True)

        if not important:
            continue

        emergency = is_emergency(important)
        now = time.time()

        allow_emergency = emergency and (now - last_emergency_time >= EMERGENCY_SPEECH_COOLDOWN)
        allow_normal = (not emergency) and (now - last_speech_time >= NORMAL_SPEECH_COOLDOWN)

        if allow_emergency or allow_normal:
            desc = ask_gemini(frame, important) or build_basic_description(important)
            if desc and desc != last_message:
                timestamp = format_time(current_sec)
                print(f"\r[{timestamp}] {'[EMERGENCY]' if emergency else '[ALERT]'} {desc}")
                last_message = desc
                last_speech_time = now
                if emergency:
                    last_emergency_time = now

                timeline_records.append({
                    "time": timestamp,
                    "hazard": desc,
                    "type": "EMERGENCY" if emergency else "INFO"
                })

finally:
    cap.release()

print("\n" + "=" * 60)
print("              ASSISTIVE RUNTIME SUMMARY")
print("=" * 60)
if timeline_records:
    for rec in timeline_records:
        print(f"[{rec['time']}] {rec['hazard']}")
else:
    print("Path was clear.")
print("=" * 60 + "\n")
