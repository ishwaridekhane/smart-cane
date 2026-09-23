import cv2
import os
import sys
import math
import time
from collections import deque
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "yolov8n.pt"
video_path = sys.argv[1] if len(sys.argv) > 1 else "street_walk.mp4"

IMG_SIZE = 384
CONFIDENCE = 0.48
FRAME_SKIP = 3

ALERT_COOLDOWN = 3.5
EMERGENCY_COOLDOWN = 1.2
GEMINI_COOLDOWN = 4.0

MAX_TRACK_DISTANCE = 110
MAX_TRACK_AGE = 1.0
MIN_CONFIRMATIONS = 3
MOVEMENT_HISTORY = 6

LEFT_BOUNDARY = 0.35
RIGHT_BOUNDARY = 0.65
HYSTERESIS_RATIO = 0.05
NEAR_RATIO = 0.35

ALLOWED_OBSTACLES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "chair", "dining table"
}

AVERAGE_WIDTHS = {
    "person": 0.45, "bicycle": 0.60, "motorcycle": 0.80,
    "car": 1.80, "bus": 2.50, "truck": 2.50, "dog": 0.40, "chair": 0.50,
    "dining table": 0.90
}
APPROX_FOCAL_LENGTH_PIXELS = 700.0

# ============================================================
# GENAI SETUP (GEMINI)
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
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

b1 = frame_w * LEFT_BOUNDARY
b2 = frame_w * RIGHT_BOUNDARY
hysteresis_margin = frame_w * HYSTERESIS_RATIO

tracks = {}
next_track_id = 1
last_alert_time = -10.0
last_gemini_time = -10.0
last_alert_summary = ""
last_emergency_time = -10.0
timeline_records = []
frame_count = 0

def format_time(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def estimate_distance_meters(label, box_w):
    if label not in AVERAGE_WIDTHS or box_w <= 5:
        return None
    raw_dist = (AVERAGE_WIDTHS[label] * APPROX_FOCAL_LENGTH_PIXELS) / box_w
    raw_dist = max(0.5, min(25.0, raw_dist))
    if raw_dist < 1.5:
        return 1
    elif raw_dist < 3.5:
        return 3
    elif raw_dist < 6.0:
        return 5
    elif raw_dist < 9.0:
        return 8
    return round(raw_dist / 5.0) * 5

def get_zone(x, previous_zone):
    if previous_zone == "left":
        if x < b1 + hysteresis_margin:
            return "left"
        return "right" if x > b2 else "center"
    elif previous_zone == "center":
        if x < b1 - hysteresis_margin:
            return "left"
        if x > b2 + hysteresis_margin:
            return "right"
        return "center"
    elif previous_zone == "right":
        if x > b2 - hysteresis_margin:
            return "right"
        return "left" if x < b1 else "center"
    else:
        if x < b1:
            return "left"
        if x > b2:
            return "right"
        return "center"

def get_movement(history):
    if len(history) < 3:
        return "moving ahead"
    dx = history[-1][0] - history[0][0]
    dh = history[-1][1] - history[0][1]
    h_thresh = frame_w * 0.035

    h_move = "to the right" if dx > h_thresh else ("to the left" if dx < -h_thresh else None)
    approaching = dh > 0.08

    if approaching and h_move:
        return f"approaching from {h_move}"
    if approaching:
        return "approaching"
    if h_move:
        return f"shifting {h_move}"
    return "moving ahead"

def natural_phrase(label, count):
    if label == "person":
        return "one person" if count == 1 else ("two people" if count == 2 else f"{count} people")
    return f"one {label}" if count == 1 else f"{count} {label}s"

def ask_gemini(frame, detected_info):
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
Summarize the key obstacle in 1 concise, calm sentence based on these confirmed detections:
{detected_info}
Include distance in meters if available. Never invent details. Return only the spoken sentence."""

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=40)
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

        dist_m = estimate_distance_meters(det["label"], det["box_w"])

        if best_id is not None and min_d <= MAX_TRACK_DISTANCE:
            trk = tracks[best_id]
            trk["x"], trk["y"] = det["x"], det["y"]
            trk["x1"] = det["x"] - det["box_w"] / 2
            trk["x2"] = det["x"] + det["box_w"] / 2
            trk["box_h"], trk["box_w"] = det["box_h"], det["box_w"]
            trk["dist_m"] = dist_m
            trk["last_seen"] = current_sec
            trk["frames_seen"] += 1
            trk["history"].append((det["x"], det["box_h"] / frame_h))
            trk["zone"] = get_zone(det["x"], trk["zone"])
            trk["proximity"] = "near" if (det["box_h"] / frame_h) > NEAR_RATIO else "far"
            matched.add(best_id)
        else:
            t_id = next_track_id
            next_track_id += 1
            init_zone = get_zone(det["x"], None)
            tracks[t_id] = {
                "id": t_id,
                "label": det["label"],
                "x": det["x"],
                "y": det["y"],
                "x1": det["x"] - det["box_w"] / 2,
                "x2": det["x"] + det["box_w"] / 2,
                "box_h": det["box_h"],
                "box_w": det["box_w"],
                "dist_m": dist_m,
                "first_seen": current_sec,
                "last_seen": current_sec,
                "frames_seen": 1,
                "history": deque([(det["x"], det["box_h"] / frame_h)], maxlen=MOVEMENT_HISTORY),
                "zone": init_zone,
                "proximity": "near" if (det["box_h"] / frame_h) > NEAR_RATIO else "far"
            }
            matched.add(t_id)

    for t_id in [k for k, v in tracks.items() if current_sec - v["last_seen"] > MAX_TRACK_AGE]:
        del tracks[t_id]

print("=" * 68)
print(f" [SAHAYAK DRISHTI] Vision Engine Running: '{video_path}'")
print(f" [SYSTEM] Duration: {format_time(total_duration_sec)} | High-speed ARM Mode")
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

    # Keep only confirmed objects within roughly 4-5 meters (bounding box height >= 20% of frame)
    visible = [
        t for t in tracks.values() 
        if (current_sec - t["last_seen"] <= 0.4 and t["frames_seen"] >= MIN_CONFIRMATIONS and (t["box_h"] / frame_h) >= 0.20)
    ]
    if not visible:
        continue

    # 1. Check if hallway ahead is blocked by group and table
    center_items = [t for t in visible if t["zone"] == "center"]
    has_table = any(t["label"] in ["dining table", "chair"] for t in center_items)
    people_in_center = sum(1 for t in center_items if t["label"] == "person")
    is_hallway_blocked = (has_table and people_in_center >= 2) or (people_in_center >= 3)

    # 2. Check if side chairs stick into walking corridor
    def sticks_into_lane(trk):
        if trk["label"] not in ["chair", "dining table"]:
            return True
        if trk["zone"] == "center":
            return True
        # If chair is close and its edges cross into the central lane
        if trk["proximity"] == "near" and (trk.get("x2", 0) > b1 and trk.get("x1", 0) < b2):
            return True
        return False

    valid_hazards = [t for t in visible if sticks_into_lane(t)]
    if not valid_hazards and not is_hallway_blocked:
        continue

    valid_hazards.sort(key=lambda t: (t["zone"] == "center", t["box_h"]), reverse=True)
    lead = valid_hazards[0] if valid_hazards else center_items[0]

    emergency_now = (lead["zone"] == "center" and lead["proximity"] == "near")
    emergency_allowed = (current_sec - last_emergency_time >= EMERGENCY_COOLDOWN)
    cooldown_expired = (current_sec - last_alert_time >= ALERT_COOLDOWN)

    zone_members = [t for t in valid_hazards if t["label"] == lead["label"] and t["zone"] == lead["zone"]]
    count = len(zone_members)
    phrase = natural_phrase(lead["label"], count)
    movement = get_movement(lead["history"])
    dist_val = lead.get("dist_m")
    dist_str = f"about {dist_val} meters" if dist_val else ""

    state_signature = f"{phrase}_{lead['zone']}_{dist_val}_{is_hallway_blocked}"

    should_announce = False
    if emergency_now and emergency_allowed:
        should_announce = True
        last_emergency_time = current_sec
    elif cooldown_expired and state_signature != last_alert_summary:
        should_announce = True

    if should_announce:
        timestamp = format_time(current_sec)
        location = "ahead of you" if lead["zone"] == "center" else f"on your {lead['zone']}"

        detected_context = {
            "obstacle": phrase,
            "location": location,
            "distance": dist_str,
            "motion": movement,
            "hallway_blocked": is_hallway_blocked,
            "emergency": emergency_now
        }
        gemini_msg = ask_gemini(frame, detected_context)

        if gemini_msg:
            msg = f"[{'EMERGENCY' if emergency_now else 'ALERT'}] {gemini_msg}"
        else:
            dist_clause = f" {dist_str}" if dist_str else ""
            be_verb = "is" if count == 1 else "are"

            if is_hallway_blocked:
                msg = "[ALERT] Path blocked ahead by table and group. Turn left or right to go around."
            elif lead["zone"] != "center" and lead["label"] == "chair":
                steer_dir = "right" if lead["zone"] == "left" else "left"
                msg = f"[ALERT] Caution, chair sticking out on your {lead['zone']}, step {steer_dir}."
            elif emergency_now:
                msg = f"[EMERGENCY] Careful, {phrase} {be_verb} very close {location}{dist_clause}!"
            else:
                msg = f"[ALERT] Caution, {phrase} {be_verb} {location}{dist_clause}."

        # Extra line space around each alert for clarity
        sys.stdout.write(f"\r{' '*85}\r\n[{timestamp}] {msg}\n\n")
        sys.stdout.flush()

        last_alert_time = current_sec
        last_alert_summary = state_signature

        timeline_records.append({
            "time": timestamp,
            "message": msg
        })

cap.release()

print("\n\n" + "=" * 68)
print("                   ASSISTIVE SCENE SUMMARY")
print("=" * 68 + "\n")
if timeline_records:
    for rec in timeline_records:
        print(f" [{rec['time']}] {rec['message']}\n")
else:
    print(" Path remained clear throughout navigation.\n")
print("=" * 68 + "\n")
