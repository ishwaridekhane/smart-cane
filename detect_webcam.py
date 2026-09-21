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

print("Scanning frames 1 to 400...\n")

# Extended to frame 400 to capture cars (frame ~210) and bicycles (frame ~330)
while cap.isOpened() and frame_count < 400:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Check every 2nd frame for 2x speed
    if frame_count % 2 != 0:
        continue

    results = model(frame, verbose=False)
    detected_classes = set(model.names[int(c)] for c in results[0].boxes.cls)

    # Object appears
    for label in detected_classes:
        if label not in active_events:
            active_events[label] = frame_count
            print(f"--> [{label.upper()}] entered at frame {frame_count}")

    # Object leaves
    for label in list(active_events.keys()):
        if label not in detected_classes:
            start_f = active_events.pop(label)
            completed_events.append((label, start_f, frame_count - 2))
            print(f"<-- [{label.upper()}] left at frame {frame_count}")

# Close any still-visible objects
for label, start_f in active_events.items():
    completed_events.append((label, start_f, frame_count))

cap.release()

print("\n" + "=" * 42)
print("             DETECTION TIMELINE")
print("=" * 42)
for label, start, end in completed_events:
    print(f"  {label.upper():<14} : Frame {start} to Frame {end}")
print("=" * 42)
