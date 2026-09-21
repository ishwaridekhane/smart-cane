import cv2
import sys
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

video_path = sys.argv[1] if len(sys.argv) > 1 else "selfie.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
left_thresh = width / 3
right_thresh = (2 * width) / 3

# 1. Strictly allow only physical obstacles a cane user must avoid
ALLOWED_OBSTACLES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'chair'}

# Tracking structures
active_tracks = {}      # track_id: {"label": str, "pos": str, "start": int, "last_seen": int, "streak": int, "announced": bool}
next_track_id = 1
completed_events = []
class_counts = {}
id_display_names = {}
frame_count = 0

print(f"Analyzing '{video_path}' ...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    # Conf=0.55 cuts out low-confidence hallucinations
    results = model(frame, conf=0.55, verbose=False)
    boxes = results[0].boxes

    detected_items = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

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

    # Match detections to existing active tracks
    matched_tracks = set()
    for label, pos, x_center in detected_items:
        best_id = None
        min_dist = float('inf')

        for t_id, data in active_tracks.items():
            if t_id not in matched_tracks and data["label"] == label:
                dist = abs(data.get("last_x", x_center) - x_center)
                if dist < 150:  # within spatial proximity
                    min_dist = dist
                    best_id = t_id

        if best_id is not None:
            matched_tracks.add(best_id)
            active_tracks[best_id]["last_seen"] = frame_count
            active_tracks[best_id]["last_x"] = x_center
            active_tracks[best_id]["pos"] = pos
            active_tracks[best_id]["streak"] += 1

            # Announce only after persisting for 3 checks (filters 1-frame glitches)
            if active_tracks[best_id]["streak"] >= 3 and not active_tracks[best_id]["announced"]:
                active_tracks[best_id]["announced"] = True
                name = id_display_names[best_id]
                print(f"--> [{name}] confirmed on the [{pos}] at frame {active_tracks[best_id]['start']}")
        else:
            # New potential obstacle
            t_id = next_track_id
            next_track_id += 1
            class_counts[label] = class_counts.get(label, 0) + 1
            id_display_names[t_id] = f"{label.upper()} {class_counts[label]}"

            active_tracks[t_id] = {
                "label": label,
                "pos": pos,
                "start": frame_count,
                "last_seen": frame_count,
                "last_x": x_center,
                "streak": 1,
                "announced": False
            }

    # Clear lost obstacles (absent for more than 4 frames)
    for t_id in list(active_tracks.keys()):
        if frame_count - active_tracks[t_id]["last_seen"] > 4:
            data = active_tracks.pop(t_id)
            if data["announced"]:
                name = id_display_names[t_id]
                completed_events.append((name, data["pos"], data["start"], data["last_seen"]))
                print(f"<-- [{name}] on the [{data['pos']}] cleared at frame {data['last_seen']}")

for t_id, data in active_tracks.items():
    if data["announced"]:
        name = id_display_names[t_id]
        completed_events.append((name, data["pos"], data["start"], data["last_seen"]))

cap.release()

print("\n" + "=" * 54)
print("             CONFIRMED OBSTACLE TIMELINE")
print("=" * 54)
for name, pos, start, end in completed_events:
    print(f"  {name:<16} [{pos:<6}] : Frame {start} to Frame {end}")
print("=" * 54)
