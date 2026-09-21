import cv2
import sys
from ultralytics import YOLO

# 1. Load model
model = YOLO("yolov8n.pt")
video_path = sys.argv[1] if len(sys.argv) > 1 else "street_walk.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open '{video_path}'.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_duration_sec = total_frames / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Dead-band margin prevents flicker between center and sides
b1 = width * 0.35
b2 = width * 0.65

ALLOWED_OBSTACLES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'chair', 'dog'}

active_tracks = {}
next_id = 1
frame_count = 0
timeline_records = []

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def get_zone(x_center, current_zone):
    """Hysteresis zone: requires crossing into the boundary to change zones."""
    if current_zone == "CENTER":
        if x_center < b1 - 25:
            return "LEFT"
        elif x_center > b2 + 25:
            return "RIGHT"
        return "CENTER"
    elif current_zone == "LEFT":
        return "CENTER" if x_center > b1 + 25 else "LEFT"
    elif current_zone == "RIGHT":
        return "CENTER" if x_center < b2 - 25 else "RIGHT"
    else:
        if x_center < b1:
            return "LEFT"
        elif x_center > b2:
            return "RIGHT"
        return "CENTER"

print(f"\n[SYSTEM] Assistive Vision Running on: '{video_path}'")
print(f"[SYSTEM] Duration: {format_time(total_duration_sec)} ({total_frames} frames)\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_sec = frame_count / fps

    # 4x speed optimization: skip 3 of every 4 frames
    if frame_count % 4 != 0:
        continue

    # Live progress bar
    percent = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
    bar = "=" * (percent // 5) + ">" + " " * (20 - (percent // 5))
    sys.stdout.write(f"\rAnalyzing: [{bar}] {percent:3d}% ({format_time(current_sec)} / {format_time(total_duration_sec)})")
    sys.stdout.flush()

    # imgsz=320 runs significantly faster on ARM CPU
    results = model(frame, imgsz=320, conf=0.55, verbose=False)
    boxes = results[0].boxes

    detected_items = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            label = model.names[int(box.cls[0])]
            if label not in ALLOWED_OBSTACLES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2.0
            detected_items.append((label, x_center))

    matched = set()
    for label, x_center in detected_items:
        best_id = None
        min_dist = float('inf')

        for t_id, data in active_tracks.items():
            if t_id not in matched and data["label"] == label:
                dist = abs(data["last_x"] - x_center)
                if dist < 160:
                    min_dist = dist
                    best_id = t_id

        if best_id is not None:
            matched.add(best_id)
            prev_zone = active_tracks[best_id]["pos"]
            new_zone = get_zone(x_center, prev_zone)

            active_tracks[best_id]["last_seen"] = current_sec
            active_tracks[best_id]["last_x"] = x_center
            active_tracks[best_id]["streak"] += 1

            # 1. Announce once after persistence filter
            if active_tracks[best_id]["streak"] == 2 and not active_tracks[best_id]["announced"]:
                active_tracks[best_id]["announced"] = True
                active_tracks[best_id]["pos"] = new_zone
                active_tracks[best_id]["last_alert_time"] = current_sec
                sys.stdout.write(f"\r{' '*70}\r[{format_time(current_sec)}] [ALERT] {label.capitalize()} ahead on the {new_zone.lower()}.\n")

            # 2. Update ONLY if position shifted AND at least 2.5 seconds have elapsed
            elif active_tracks[best_id]["announced"] and prev_zone != new_zone:
                if current_sec - active_tracks[best_id]["last_alert_time"] >= 2.5:
                    sys.stdout.write(f"\r{' '*70}\r[{format_time(current_sec)}] [UPDATE] {label.capitalize()} moved to {new_zone.lower()}.\n")
                    active_tracks[best_id]["pos"] = new_zone
                    active_tracks[best_id]["last_alert_time"] = current_sec
        else:
            initial_zone = get_zone(x_center, None)
            active_tracks[next_id] = {
                "label": label,
                "pos": initial_zone,
                "start_time": current_sec,
                "last_seen": current_sec,
                "last_alert_time": current_sec,
                "last_x": x_center,
                "streak": 1,
                "announced": False
            }
            next_id += 1

    # Remove objects cleared from sight (absent > 0.8 seconds)
    for t_id in list(active_tracks.keys()):
        if current_sec - active_tracks[t_id]["last_seen"] > 0.8:
            data = active_tracks.pop(t_id)
            if data["announced"]:
                timeline_records.append({
                    "obstacle": data["label"].capitalize(),
                    "position": data["pos"],
                    "start": format_time(data["start_time"]),
                    "end": format_time(data["last_seen"])
                })

for t_id, data in active_tracks.items():
    if data["announced"]:
        timeline_records.append({
            "obstacle": data["label"].capitalize(),
            "position": data["pos"],
            "start": format_time(data["start_time"]),
            "end": format_time(data["last_seen"])
        })

cap.release()

# Executive Clean Summary
print("\n" + "=" * 48)
print("           ASSISTIVE OBSTACLE SUMMARY")
print("=" * 48)
if timeline_records:
    print(f"{'Obstacle':<12} | {'Direction':<9} | {'Duration'}")
    print("-" * 48)
    for row in timeline_records:
        print(f"{row['obstacle']:<12} | {row['position']:<9} | {row['start']} to {row['end']}")
else:
    print("Path was clear throughout the walk.")
print("=" * 48 + "\n")
