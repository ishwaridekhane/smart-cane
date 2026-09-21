import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("walk.mp4")

if not cap.isOpened():
    print("Error opening video")
    exit()

active_events = {}
completed_events = []
frame_count = 0

print("Scanning video for object appearance intervals...\n")

while cap.isOpened() and frame_count < 150:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    results = model(frame, verbose=False)
    detected_classes = set(model.names[int(c)] for c in results[0].boxes.cls)

    # Detect new objects appearing
    for label in detected_classes:
        if label not in active_events:
            active_events[label] = frame_count
            print(f"--> [{label.upper()}] entered at frame {frame_count}")

    # Detect objects leaving
    for label in list(active_events.keys()):
        if label not in detected_classes:
            start_f = active_events.pop(label)
            completed_events.append((label, start_f, frame_count - 2))
            print(f"<-- [{label.upper()}] left at frame {frame_count}")

# Close any still-active objects
for label, start_f in active_events.items():
    completed_events.append((label, start_f, frame_count))

cap.release()

# Final summary table
print("\n" + "=" * 40)
print("           DETECTION TIMELINE")
print("=" * 40)
for label, start, end in completed_events:
    print(f"  {label.upper():<12} : Frame {start} to Frame {end}")
print("=" * 40)
