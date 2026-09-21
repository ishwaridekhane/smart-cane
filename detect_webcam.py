import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("walk.mp4")

if not cap.isOpened():
    print("Error opening video")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

print("Fast processing started (max 2 minutes)...")

frame_count = 0
saved_count = 0

# Stop at frame 150 (covers the core scene in ~90 seconds)
while cap.isOpened() and frame_count < 150:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Skip every other frame: 2x speedup with full 640px accuracy
    if frame_count % 2 != 0:
        continue

    saved_count += 1
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

    labels = [model.names[int(c)] for c in results[0].boxes.cls]
    print(f"Processed frame {frame_count}/150 | Detected: {labels}")

cap.release()
out.release()
print(f"Done! Saved {saved_count} frames to output.avi.")
