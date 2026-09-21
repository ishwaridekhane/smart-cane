import cv2
import sys
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# If you pass a filename as an argument (e.g. python3 detect_webcam.py selfie.mp4), it uses it.
# Otherwise, it defaults to selfie.mp4.
video_path = sys.argv[1] if len(sys.argv) > 1 else "selfie.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'. Make sure it exists!")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
left_threshold = width / 3
right_threshold = (2 * width) / 3

active_events = {}
completed_events = []
frame_count = 0

print(f"Analyzing '{video_path}' for obstacles and positions (LEFT / CENTER / RIGHT)...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Check every 2nd frame for 2x speedup on Raspberry Pi CPU
    if frame_count % 2 != 0:
        continue

    results = model(frame, verbose=False)
    boxes = results[0].boxes
    current_frame_detections = {}

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Calculate horizontal center of the detected object
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2.0

            if x_center < left_threshold:
                position = "LEFT"
            elif x_center > right_threshold:
                position = "RIGHT"
            else:
                position = "CENTER"

            # Track unique object and its position
            key = (label, position)
            current_frame_detections[key] = True

    # Check for new objects entering
    for (label, pos) in current_frame_detections.keys():
        if (label, pos) not in active_events:
            active_events[(label, pos)] = frame_count
            print(f"--> [{label.upper()}] detected on the [{pos}] at frame {frame_count}")

    # Check for objects leaving
    for (label, pos) in list(active_events.keys()):
        if (label, pos) not in current_frame_detections:
            start_f = active_events.pop((label, pos))
            completed_events.append((label, pos, start_f, frame_count - 2))
            print(f"<-- [{label.upper()}] on the [{pos}] left at frame {frame_count}")

# Close any remaining active detections
for (label, pos), start_f in active_events.items():
    completed_events.append((label, pos, start_f, frame_count))

cap.release()

print("\n" + "=" * 50)
print("             DETECTION & POSITION TIMELINE")
print("=" * 50)
for label, pos, start, end in completed_events:
    print(f"  {label.upper():<12} [{pos:<6}] : Frame {start} to Frame {end}")
print("=" * 50)
