import cv2
import sys
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Accept video argument or default to walk.mp4
video_path = sys.argv[1] if len(sys.argv) > 1 else "walk.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
left_threshold = width / 3
right_threshold = (2 * width) / 3

# Obstacle whitelist: strictly drops noise like kite, cell phone, sports ball
ALLOWED_OBSTACLES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'chair'}

active_tracks = {}      # track_id -> {"name": str, "start": int, "pos": str, "last_seen": int}
completed_events = []
frame_count = 0

# Map object class to sequential numbering (e.g., PERSON 1, PERSON 2)
class_counter = {}
id_to_label = {}

print(f"Tracking obstacles in '{video_path}' (conf=0.55)...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Frame-skip for speed on Pi
    if frame_count % 2 != 0:
        continue

    # Use YOLO's built-in tracker with confidence filter
    results = model.track(frame, persist=True, conf=0.55, verbose=False)
    boxes = results[0].boxes
    current_frame_track_ids = set()

    if boxes is not None and boxes.id is not None:
        for box, track_id_tensor in zip(boxes, boxes.id):
            track_id = int(track_id_tensor)
            cls_id = int(box.cls[0])
            raw_label = model.names[cls_id]

            if raw_label not in ALLOWED_OBSTACLES:
                continue

            # Assign sequential number to new unique instances
            if track_id not in id_to_label:
                class_counter[raw_label] = class_counter.get(raw_label, 0) + 1
                id_to_label[track_id] = f"{raw_label.upper()} {class_counter[raw_label]}"

            display_name = id_to_label[track_id]

            # Calculate position
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2.0
            if x_center < left_threshold:
                position = "LEFT"
            elif x_center > right_threshold:
                position = "RIGHT"
            else:
                position = "CENTER"

            current_frame_track_ids.add(track_id)

            if track_id not in active_tracks:
                active_tracks[track_id] = {
                    "name": display_name,
                    "start": frame_count,
                    "pos": position,
                    "last_seen": frame_count
                }
                print(f"--> [{display_name}] entered on the [{position}] at frame {frame_count}")
            else:
                active_tracks[track_id]["last_seen"] = frame_count
                # Update position if the object moved laterally
                if active_tracks[track_id]["pos"] != position:
                    active_tracks[track_id]["pos"] = position

    # Close tracks that left the frame (absent for more than 4 frames)
    for t_id in list(active_tracks.keys()):
        if t_id not in current_frame_track_ids and (frame_count - active_tracks[t_id]["last_seen"] > 4):
            data = active_tracks.pop(t_id)
            # Only record if visible for more than 4 frames (filters instant flickers)
            if data["last_seen"] - data["start"] >= 4:
                completed_events.append((data["name"], data["pos"], data["start"], data["last_seen"]))
                print(f"<-- [{data['name']}] on the [{data['pos']}] left at frame {data['last_seen']}")

# Flush remaining active tracks
for t_id, data in active_tracks.items():
    if data["last_seen"] - data["start"] >= 4:
        completed_events.append((data["name"], data["pos"], data["start"], data["last_seen"]))

cap.release()

print("\n" + "=" * 54)
print("             DETECTION & POSITION TIMELINE")
print("=" * 54)
for name, pos, start, end in completed_events:
    print(f"  {name:<16} [{pos:<6}] : Frame {start} to Frame {end}")
print("=" * 54)
