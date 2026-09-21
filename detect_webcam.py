import cv2
import sys
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video_path = sys.argv[1] if len(sys.argv) > 1 else "street_walk.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open '{video_path}'.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
left_thresh = width / 3
right_thresh = (2 * width) / 3

ALLOWED_OBSTACLES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'chair', 'dog'}

active_tracks = {}
next_id = 1
frame_count = 0

print("Smart Cane Assistant Active...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
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

            pos = "LEFT" if x_center < left_thresh else ("RIGHT" if x_center > right_thresh else "CENTER")
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
            active_tracks[best_id]["last_seen"] = frame_count
            active_tracks[best_id]["last_x"] = x_center
            active_tracks[best_id]["streak"] += 1

            # 1. Initial confirmation after 3 steady frames
            if active_tracks[best_id]["streak"] >= 3 and not active_tracks[best_id]["announced"]:
                active_tracks[best_id]["announced"] = True
                active_tracks[best_id]["pos"] = pos
                print(f">> ALERT: {label.capitalize()} detected ahead on the {pos.lower()}.")

            # 2. Movement change notification
            elif active_tracks[best_id]["announced"] and prev_pos != pos:
                print(f">> UPDATE: {label.capitalize()} moved from {prev_pos.lower()} to {pos.lower()}.")
                active_tracks[best_id]["pos"] = pos
        else:
            active_tracks[next_id] = {
                "label": label,
                "pos": pos,
                "last_seen": frame_count,
                "last_x": x_center,
                "streak": 1,
                "announced": False
            }
            next_id += 1

    # Announce path clear when obstacle leaves
    for t_id in list(active_tracks.keys()):
        if frame_count - active_tracks[t_id]["last_seen"] > 6:
            data = active_tracks.pop(t_id)
            if data["announced"]:
                print(f"<< CLEAR: Path on {data['pos'].lower()} is now clear.")

cap.release()
