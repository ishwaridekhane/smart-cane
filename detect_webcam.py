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

active_tracks = {}
next_id = 1
frame_count = 0
timeline_records = []

def format_time(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def get_zone(x_center):
    if x_center < b1:
        return "LEFT"
    elif x_center > b2:
        return "RIGHT"
    return "CENTER"

print(f"\n[ASSISTIVE CANE ENGINE] Active on '{video_path}'\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_sec = frame_count / fps

    # Skip every 2nd frame for CPU performance
    if frame_count % 2 != 0:
        continue

    results = model(frame, imgsz=480, conf=0.45, verbose=False)
    boxes = results[0].boxes

    detected_items = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            label = model.names[int(box.cls[0])]
            if label not in ALLOWED_OBSTACLES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2.0
            box_h = y2 - y1

            # Proximity estimation based on vertical frame ratio
            proximity = "NEAR" if (box_h / frame_h) > 0.35 else "FAR"
            detected_items.append((label, x_center, proximity))

    # Match and track
    matched = set()
    for label, x_center, proximity in detected_items:
        best_id = None
        min_dist = float('inf')

        for t_id, data in active_tracks.items():
            if t_id not in matched and data["label"] == label:
                dist = abs(data["last_x"] - x_center)
                if dist < 180:
                    min_dist = dist
                    best_id = t_id

        zone = get_zone(x_center)

        if best_id is not None:
            matched.add(best_id)
            active_tracks[best_id]["last_seen"] = current_sec
            active_tracks[best_id]["last_x"] = x_center

            # Alert if an object moves from FAR to NEAR
            if proximity == "NEAR" and active_tracks[best_id]["proximity"] == "FAR":
                active_tracks[best_id]["proximity"] = "NEAR"
                sys.stdout.write(f"\r[{format_time(current_sec)}] [WARNING] {label.capitalize()} approaching close on {zone.lower()}!\n")

            # Alert on major zone change with cooldown
            elif active_tracks[best_id]["pos"] != zone and (current_sec - active_tracks[best_id]["last_alert_time"] >= 2.5):
                active_tracks[best_id]["pos"] = zone
                active_tracks[best_id]["last_alert_time"] = current_sec
                sys.stdout.write(f"\r[{format_time(current_sec)}] [UPDATE] {label.capitalize()} shifted to {zone.lower()}.\n")
        else:
            active_tracks[next_id] = {
                "label": label,
                "pos": zone,
                "proximity": proximity,
                "start_time": current_sec,
                "last_seen": current_sec,
                "last_alert_time": current_sec,
                "last_x": x_center
            }
            # Immediate notification with distance context
            urgency = "Close hazard:" if proximity == "NEAR" else "Ahead:"
            sys.stdout.write(f"\r[{format_time(current_sec)}] [ALERT] {urgency} {label.capitalize()} on {zone.lower()}.\n")
            next_id += 1

    # Prune inactive tracks
    for t_id in list(active_tracks.keys()):
        if current_sec - active_tracks[t_id]["last_seen"] > 0.8:
            data = active_tracks.pop(t_id)
            timeline_records.append({
                "obstacle": data["label"].capitalize(),
                "position": data["pos"],
                "proximity": data["proximity"],
                "start": format_time(data["start_time"]),
                "end": format_time(data["last_seen"])
            })

for t_id, data in active_tracks.items():
    timeline_records.append({
        "obstacle": data["label"].capitalize(),
        "position": data["pos"],
        "proximity": data["proximity"],
        "start": format_time(data["start_time"]),
        "end": format_time(data["last_seen"])
    })

cap.release()

print("\n" + "=" * 54)
print("            ASSISTIVE NAVIGATION SUMMARY")
print("=" * 54)
if timeline_records:
    print(f"{'Hazard':<12} | {'Proximity':<10} | {'Zone':<8} | {'Window'}")
    print("-" * 54)
    for r in timeline_records:
        print(f"{r['obstacle']:<12} | {r['proximity']:<10} | {r['position']:<8} | {r['start']} - {r['end']}")
else:
    print("Path completely clear.")
print("=" * 54 + "\n")
