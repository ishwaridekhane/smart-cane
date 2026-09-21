import cv2
import sys
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video_path = sys.argv[1] if len(sys.argv) > 1 else "street_walk.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open '{video_path}'.")
    sys.exit(1)

# Video metadata
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_duration_sec = total_frames / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

left_thresh = width / 3.0
right_thresh = (2.0 * width) / 3.0

ALLOWED_OBSTACLES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'chair', 'dog'}

active_tracks = {}
next_id = 1
frame_count = 0
timeline_records = []

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

print(f"\n[SYSTEM] Assistive Cane Vision Initialized: '{video_path}'")
print(f"[SYSTEM] Duration: {format_time(total_duration_sec)} ({total_frames} frames)\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_sec = frame_count / fps

    # Print live progress on the same terminal line every 5 frames
    if frame_count % 5 == 0 or frame_count == total_frames:
        percent = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
        bar = "=" * (percent // 5) + ">" + " " * (20 - (percent // 5))
        sys.stdout.write(f"\rAnalyzing: [{bar}] {percent:3d}% ({format_time(current_sec)} / {format_time(total_duration_sec)})")
        sys.stdout.flush()

    # Skip every 2nd frame for real-time CPU efficiency
    if frame_count % 2 != 0:
        continue

    results = model(frame, conf=0.55, verbose=False)
    boxes = results[0].boxes

    detected_items = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            label = model.names[int(box.cls[0])]
            if label not in ALLOWED_OBSTACLES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2.0

            if x_center < left_thresh:
                pos = "LEFT"
            elif x_center > right_thresh:
                pos = "RIGHT"
            else:
                pos = "CENTER"

            detected_items.append((label, pos, x_center))

    matched = set()
    for label, pos, x_center in detected_items:
        best_id = None
        min_dist = float('inf')

        for t_id, data in active_tracks.items():
            if t_id not in matched and data["label"] == label:
                dist = abs(data["last_x"] - x_center)
                if dist < 150:
                    min_dist = dist
                    best_id = t_id

        if best_id is not None:
            matched.add(best_id)
            prev_pos = active_tracks[best_id]["pos"]
            active_tracks[best_id]["last_seen"] = current_sec
            active_tracks[best_id]["last_x"] = x_center
            active_tracks[best_id]["streak"] += 1

            # 1. Announce once after 3 steady frames to avoid ghost alerts
            if active_tracks[best_id]["streak"] == 3 and not active_tracks[best_id]["announced"]:
                active_tracks[best_id]["announced"] = True
                active_tracks[best_id]["pos"] = pos
                sys.stdout.write(f"\r[{format_time(current_sec)}] [ALERT] {label.capitalize()} detected ahead ({pos.lower()}).\n")

            # 2. Update position only if a major shift occurs
            elif active_tracks[best_id]["announced"] and prev_pos != pos:
                sys.stdout.write(f"\r[{format_time(current_sec)}] [UPDATE] {label.capitalize()} shifted to {pos.lower()}.\n")
                active_tracks[best_id]["pos"] = pos
        else:
            active_tracks[next_id] = {
                "label": label,
                "pos": pos,
                "start_time": current_sec,
                "last_seen": current_sec,
                "last_x": x_center,
                "streak": 1,
                "announced": False
            }
            next_id += 1

    # Remove cleared obstacles
    for t_id in list(active_tracks.keys()):
        if current_sec - active_tracks[t_id]["last_seen"] > 0.4:
            data = active_tracks.pop(t_id)
            if data["announced"]:
                timeline_records.append({
                    "obstacle": data["label"].capitalize(),
                    "position": data["pos"],
                    "start": format_time(data["start_time"]),
                    "end": format_time(data["last_seen"])
                })

# Flush any lingering objects
for t_id, data in active_tracks.items():
    if data["announced"]:
        timeline_records.append({
            "obstacle": data["label"].capitalize(),
            "position": data["pos"],
            "start": format_time(data["start_time"]),
            "end": format_time(data["last_seen"])
        })

cap.release()

# Final Clean Structured Analysis
print("\n\n" + "=" * 50)
print("             NAVIGATION HAZARD SUMMARY")
print("=" * 50)
if timeline_records:
    print(f"{'Hazard':<14} | {'Direction':<10} | {'Active Window'}")
    print("-" * 50)
    for row in timeline_records:
        print(f"{row['obstacle']:<14} | {row['position']:<10} | {row['start']} - {row['end']}")
else:
    print("Clear path. No critical hazards detected.")
print("=" * 50 + "\n")
