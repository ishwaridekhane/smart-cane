import cv2
from ultralytics import YOLO

# Load the lightweight YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Open walk.mp4 video file instead of webcam
cap = cv2.VideoCapture("walk.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video dimensions and framerate
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0

# Initialize VideoWriter to save annotated frames to output.avi
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

print("Processing video and saving annotated frames to output.avi...")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Run YOLOv8 detection
    results = model(frame, verbose=False)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Save the annotated frame
    out.write(annotated_frame)

    # Print log in terminal every 15 frames
    if frame_count % 15 == 0:
        labels = [model.names[int(c)] for c in results[0].boxes.cls]
        print(f"Frame {frame_count}: Saved with detections {labels}")

cap.release()
out.release()
print(f"Finished! Output saved to output.avi ({frame_count} frames total).")
