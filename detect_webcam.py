import cv2
import os
import sys
import math
import time
import logging
from collections import deque
from ultralytics import YOLO

# Silence Google GenAI SDK logging warnings
logging.getLogger("google_genai").setLevel(logging.ERROR)

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "yolov8n.pt"
video_path = sys.argv[1] if len(sys.argv) > 1 else "lib1.mp4"

IMG_SIZE = 384
CONFIDENCE = 0.45
FRAME_SKIP = 3

ALERT_COOLDOWN = 3.0
EMERGENCY_COOLDOWN = 1.2
GEMINI_COOLDOWN = 4.0

MAX_TRACK_DISTANCE = 110
MAX_TRACK_AGE = 0.8
MIN_CONFIRMATIONS = 2

# Walking Corridor (Center lane boundaries)
LEFT_BOUNDARY = 0.32
RIGHT_BOUNDARY = 0.68
HYSTERESIS_RATIO = 0.04

ALLOWED_OBSTACLES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "chair", "dining table"
}

# Height-based real-world measurements for stable distance estimation
AVERAGE_HEIGHTS = {
    "person": 1.70, "bicycle": 1.00, "motorcycle": 1.10,
    "car": 1.50, "bus": 3.00, "truck": 3.00, "dog": 0.50,
    "chair": 0.85, "dining table": 0.75
}
FOCAL_LENGTH_PIXELS = 650.0

# ============================================================
# GEMINI SETUP
# ============================================================
gemini_client = None
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=api_key)
        print("\n[SYSTEM] Gemini GenAI Engine: ENABLED\n")
    except Exception as e:
        print(f"\n[SYSTEM] Gemini setup warning: {e}\n")
else:
    print("\n[SYSTEM] Gemini GenAI: DISABLED (GEMINI_API_KEY not set)\n")

# ============================================================
# INITIALIZATION
# ============================================================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open '{video_path}'.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_duration_sec = total_frames / fps
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640

b1 = frame_w * LEFT_BOUNDARY
b2 = frame_w * RIGHT_BOUNDARY
hysteresis_margin = frame_w * HYSTERESIS_RATIO

tracks = {}
next_track_id = 1
last_alert_time = -10.0
last_emergency_time = -10.0
last_gemini_time = -10.0
last_alert_signature = ""
timeline_records = []
frame_count = 0

def format_time(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def estimate_distance_meters(label, box_h):
    if label not in AVERAGE_HEIGHTS or box_h <= 5:
        return None
    raw_dist = (AVERAGE_HEIGHTS[label] * FOCAL_LENGTH_PIXELS) / box_h
    raw_dist = max(0.5, min(20.0, raw_dist))
    if raw_dist < 1.8:
        return 1
    elif raw_dist < 3.8:
        return 3
    elif raw_dist < 6.5:
        return 5
    return round(raw_dist / 5.0) * 5

def get_zone(x, previous_zone=None):
    if previous_zone == "left":
        return "left" if x < (b1 + hysteresis_margin) else ("right" if x > b2 else "center")
    elif previous_zone == "center":
        if x < (b1 - hysteresis_margin):
            return "left"
        if x > (b2 + hysteresis_margin):
            return "right"
        return "center"
    elif previous_zone == "right":
        return "right" if x > (b2 - hysteresis_margin) else ("left" if x < b1 else "center")
    return "left" if x < b1 else ("right" if x > b2 else "center")

def ask_gemini(frame, prompt_context):
    global last_gemini_time
    if not gemini_client:
        return None
    now = time.time()
    if now - last_gemini_time < GEMINI_COOLDOWN:
        return None

    try:
        from google.genai import types
        small_frame = cv2.resize(frame, (320, 240))
        _, encoded = cv2.imencode(".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])

        prompt = f"""You are the voice of a smart cane for a visually impaired user.
Obstacle data: {prompt_context}
State what is directly blocking them or if they must turn. Keep it to 1 calm sentence under 12 words."""

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=30,
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.NONE
                    )
                )
            )
        )
        last_gemini_time = now
        if response.text:
            return response.text.strip().replace("\n", " ")
    except Exception:
        pass
    return None

def match_detections(detections, current_sec):
    global next_track_id
    matched = set()

    for det in detections:
        best_id, min_d = None, float("inf")
        for t_id, trk in tracks.items():
            if t_id in matched or trk["label"] != det["label"]:
                continue
            if current_sec - trk["last_seen"] > MAX_TRACK_AGE:
                continue
            d = distance(det["x"], det["y"], trk["x"], trk["y"])
            if d < min_d:
                min_d, best_id = d, t_id

        dist_m = estimate_distance_meters(det["label"], det["box_h"])
        x1 = det["x"] - det["box_w"] / 2
        x2 = det["x"] + det["box_w"] / 2

        if best_id is not None and min_d <= MAX_TRACK_DISTANCE:
            trk = tracks[best_id]
            trk["x"], trk["y"] = det["x"], det["y"]
            trk["x1"], trk["x2"] = x1, x2
            trk["box_h"], trk["box_w"] = det["box_h"], det["box_w"]
            trk["dist_m"] = dist_m
            trk["last_seen"] = current_sec
            trk["frames_seen"] += 1
            trk["zone"] = get_zone(det["x"], trk["zone"])
            matched.add(best_id)
        else:
            t_id = next_track_id
            next_track_id += 1
            tracks[t_id] = {
                "id": t_id,
                "label": det["label"],
                "x": det["x"],
                "y": det["y"],
                "x1": x1,
                "x2": x2,
                "box_h": det["box_h"],
                "box_w": det["box_w"],
                "dist_m": dist_m,
                "last_seen": current_sec,
                "frames_seen": 1,
                "zone": get_zone(det["x"])
            }
            matched.add(t_id)

    for t_id in [k for k, v in tracks.items() if current_sec - v["last_seen"] > MAX_TRACK_AGE]:
        del tracks[t_id]

print("=" * 68)
print(f" [SAHAYAK DRISHTI] High-Priority Safety Engine: '{video_path}'")
print("=" * 68 + "\n")

# ============================================================
# MAIN INFERENCE LOOP
# ============================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_sec = frame_count / fps

    if frame_count % FRAME_SKIP != 0:
        continue

    percent = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
    bar = "=" * (percent // 5) + ">" + " " * (20 - (percent // 5))
    sys.stdout.write(f"\rAnalyzing: [{bar}] {percent:3d}% ({format_time(current_sec)} / {format_time(total_duration_sec)})")
    sys.stdout.flush()

    results = model(frame, imgsz=IMG_SIZE, conf=CONFIDENCE, verbose=False)
    boxes = results[0].boxes

    detections = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label not in ALLOWED_OBSTACLES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "x": (x1 + x2) / 2.0,
                "y": (y1 + y2) / 2.0,
                "box_w": x2 - x1,
                "box_h": y2 - y1
            })

    match_detections(detections, current_sec)

    # 1. Filter background clutter (height >= 18% of frame)
    visible = [
        t for t in tracks.values()
        if (current_sec - t["last_seen"] <= 0.4 and t["frames_seen"] >= MIN_CONFIRMATIONS and (t["box_h"] / frame_h) >= 0.18)
    ]
    if not visible:
        continue

    # 2. Check if the corridor is blocked by seating or a group ahead
    center_items = [t for t in visible if t["zone"] == "center"]
    has_table = any(t["label"] in ["dining table", "chair"] for t in center_items)
    people_in_center = sum(1 for t in center_items if t["label"] == "person")
    is_corridor_blocked = (has_table and people_in_center >= 2) or (people_in_center >= 3)

    # 3. Detect items in the walking lane or side furniture sticking into it
    def is_walking_hazard(t):
        if t["zone"] == "center":
            return True
        if t["label"] in ["chair", "dining table"]:
            if (t["x2"] > b1 and t["x1"] < b2) and (t["box_h"] / frame_h >= 0.22):
                return True
        return False

    hazards = [t for t in visible if is_walking_hazard(t)]
    if not hazards and not is_corridor_blocked:
        continue

    # 4. Pick leading obstacle
    hazards.sort(key=lambda t: (t["zone"] == "center", t["box_h"]), reverse=True)
    lead = hazards[0] if hazards else center_items[0]

    dist_val = lead.get("dist_m")
    dist_str = f"about {dist_val} meters" if dist_val else "close to you"
    is_emergency = (lead["zone"] == "center" and ((lead["box_h"] / frame_h) >= 0.38 or (dist_val is not None and dist_val <= 1)))

    emergency_allowed = (current_sec - last_emergency_time >= EMERGENCY_COOLDOWN)
    cooldown_expired = (current_sec - last_alert_time >= ALERT_COOLDOWN)

    state_sig = f"{lead['label']}_{lead['zone']}_{dist_val}_{is_corridor_blocked}"

    should_announce = False
    if is_emergency and emergency_allowed:
        should_announce = True
        last_emergency_time = current_sec
    elif cooldown_expired and state_sig != last_alert_signature:
        should_announce = True

    if should_announce:
        timestamp = format_time(current_sec)
        location = "ahead of you" if lead["zone"] == "center" else f"on your {lead['zone']}"

        # PRIORITY 1: Hallway blocked ahead
        if is_corridor_blocked:
            msg = "[ALERT] Path blocked ahead by table and seated group. Turn left or right to go around."

        # PRIORITY 2: Protruding side chair
        elif lead["zone"] != "center" and lead["label"] == "chair":
            steer = "right" if lead["zone"] == "left" else "left"
            msg = f"[ALERT] Caution, chair sticking out on your {lead['zone']}, step {steer}."

        # PRIORITY 3: General Emergency / Close Obstacle
        elif is_emergency:
            msg = f"[EMERGENCY] Careful, {lead['label']} is very close {location}!"

        # PRIORITY 4: Normal Alert (Ask Gemini or use local fallback)
        else:
            gemini_txt = ask_gemini(frame, f"{lead['label']} {location}, {dist_str}")
            if gemini_txt:
                msg = f"[ALERT] {gemini_txt}"
            else:
                msg = f"[ALERT] Caution, {lead['label']} {location}, {dist_str}."

        sys.stdout.write(f"\r{' ' * 85}\r\n[{timestamp}] {msg}\n\n")
        sys.stdout.flush()

        last_alert_time = current_sec
        last_alert_signature = state_sig
        timeline_records.append({"time": timestamp, "message": msg})

cap.release()

print("\n\n" + "=" * 68)
print("              HIGH-PRIORITY ASSISTIVE SUMMARY")
print("=" * 68 + "\n")
if timeline_records:
    for rec in timeline_records:
        print(f" [{rec['time']}] {rec['message']}\n")
else:
    print(" Path remained clear throughout navigation.\n")
print("=" * 68 + "\n")
