import cv2
import sys
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video_path = sys.argv[1] if len(sys.argv) > 1 else "street_walk.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open '{video_path}'.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

b1 = frame_w * 0.35
b2 = frame_w * 0.65

ALLOWED_OBSTACLES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'dog', 'chair'}

frame_count = 0
last_broadcast_msg = ""
last_broadcast_time = -10.0
timeline_records = []

def format_time(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def get_zone(x_center):
    if x_center < b1:
        return "left"
    elif x_center > b2:
        return "right"
    return "center"

def pluralize(label, count):
    if count == 1:
        return f"1 {label.capitalize()}"
    if label == "person":
        return f"{count} People"
    return f"{count} {label.capitalize()}s"

print(f"\n[ASSISTIVE CANE] Active on '{video_path}'\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_sec = frame_count / fps

    # Check every 2nd frame for real-time responsiveness
    if frame_count % 2 != 0:
        continue

    results = model(frame, imgsz=480, conf=0.45, verbose=False)
    boxes = results[0].boxes

    # Aggregate counts per (label, zone, proximity)
    scene_counts = {}
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            label = model.names[int(box.cls[0])]
            if label not in ALLOWED_OBSTACLES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2.0
            box_h = y2 - y1

            proximity = "NEAR" if (box_h / frame_h) > 0.35 else "FAR"
            zone = get_zone(x_center)

            key = (label, zone, proximity)
            scene_counts[key] = scene_counts.get(key, 0) + 1

    # Select the most critical hazard in the frame (Prioritize NEAR > CENTER path)
    if scene_counts:
        sorted_hazards = sorted(
            scene_counts.items(),
            key=lambda item: (item[0][2] == "NEAR", item[0][1] == "center"),
            reverse=True
        )
        (top_label, top_zone, top_prox), count = sorted_hazards[0]

        phrase_count = pluralize(top_label, count)
        if top_prox == "NEAR":
            current_msg = f"[WARNING] Close hazard: {phrase_count} on {top_zone}!"
        else:
            current_msg = f"[ALERT] Ahead: {phrase_count} on {top_zone}."

        # Emit only if the message is different or 3 seconds have passed
        if current_msg != last_broadcast_msg or (current_sec - last_broadcast_time >= 3.0):
            print(f"[{format_time(current_sec)}] {current_msg}")
            last_broadcast_msg = current_msg
            last_broadcast_time = current_sec

            timeline_records.append({
                "hazard": phrase_count,
                "zone": top_zone.upper(),
                "time": format_time(current_sec)
            })

cap.release()

print("\n" + "=" * 46)
print("          ASSISTIVE OBSTACLE TIMELINE")
print("=" * 46)
if timeline_records:
    print(f"{'Time':<8} | {'Obstacle':<18} | {'Direction'}")
    print("-" * 46)
    for r in timeline_records:
        print(f"{r['time']:<8} | {r['hazard']:<18} | {r['zone']}")
else:
    print("No critical hazards detected.")
print("=" * 46 + "\n")
