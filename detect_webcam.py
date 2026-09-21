import cv2
from ultralytics import YOLO

# Load the lightweight YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Open laptop webcam (0 is usually the default built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' on the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 detection on the frame
    results = model(frame, stream=True)

    # Plot bounding boxes on the frame
    for r in results:
        annotated_frame = r.plot()

    # Display the live feed with detections
    cv2.imshow("Sahayak Drishti - Laptop Test Feed", annotated_frame)

    # Press 'q' key on keyboard to close window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()