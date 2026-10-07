import cv2
import sys
import time
import math
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolov8n.pt"
PHONE_URL = sys.argv[1] if len(sys.argv) > 1 else None

# FAST SETTINGS FOR RASPBERRY PI 4
YOLO_SIZE = 320
CONFIDENCE = 0.40
FRAME_SKIP = 4

ALERT_COOLDOWN = 2.0

LEFT_BOUNDARY = 0.33
RIGHT_BOUNDARY = 0.67

FOCAL_LENGTH_PIXELS = 350.0

# Objects useful for navigation
ALLOWED_OBJECTS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "cat",
    "chair",
    "couch",
    "dining table",
    "bench",
    "backpack",
    "suitcase",
    "bottle",
    "cup",
    "book",
    "laptop",
    "keyboard",
    "mouse",
    "cell phone",
    "remote",
    "tv",
    "potted plant",
}

# Approximate real-world heights
OBJECT_HEIGHTS = {
    "person": 1.70,
    "bicycle": 1.00,
    "motorcycle": 1.10,
    "car": 1.50,
    "bus": 3.00,
    "truck": 3.00,
    "dog": 0.50,
    "cat": 0.30,
    "chair": 0.85,
    "couch": 0.80,
    "dining table": 0.75,
    "bench": 0.50,
    "backpack": 0.45,
    "suitcase": 0.70,
    "bottle": 0.25,
    "cup": 0.12,
    "book": 0.25,
    "laptop": 0.25,
    "keyboard": 0.05,
    "mouse": 0.04,
    "cell phone": 0.15,
    "remote": 0.15,
    "tv": 0.70,
    "potted plant": 0.50,
}

# ============================================================
# FUNCTIONS
# ============================================================

def get_direction(x_center, frame_width):
    ratio = x_center / frame_width

    if ratio < LEFT_BOUNDARY:
        return "left"
    elif ratio > RIGHT_BOUNDARY:
        return "right"
    else:
        return "ahead"


def estimate_distance(object_name, box_height):
    if box_height <= 0:
        return None

    real_height = OBJECT_HEIGHTS.get(object_name)

    if real_height is None:
        return None

    distance = (real_height * FOCAL_LENGTH_PIXELS) / box_height

    # Keep the result sensible
    distance = max(0.5, min(distance, 10.0))

    return distance


def distance_text(distance):
    if distance is None:
        return ""

    if distance < 1.5:
        return "about 1 meter"
    elif distance < 2.5:
        return "about 2 meters"
    elif distance < 3.5:
        return "about 3 meters"
    elif distance < 5:
        return "about 4 meters"
    elif distance < 7:
        return "about 6 meters"
    else:
        return "about 10 meters"


def get_position_message(name, direction):
    if direction == "ahead":
        return f"{name} ahead."
    else:
        return f"{name} on your {direction}."


def get_walking_message(name, direction, distance):
    d_text = distance_text(distance)

    if d_text:
        if direction == "ahead":
            return f"{name} ahead, {d_text}."
        else:
            return f"{name} on your {direction}, {d_text}."

    return get_position_message(name, direction)


def is_dangerous(name, direction, distance):
    """
    Conservative walking-mode safety rule.
    """

    # Anything directly ahead and close
    if direction == "ahead" and distance is not None and distance <= 1.5:
        return True

    # People ahead are important even when slightly farther
    if name == "person" and direction == "ahead" and distance is not None:
        if distance <= 3.0:
            return True

    # Vehicles/bicycles ahead
    if name in {"bicycle", "motorcycle", "car", "bus", "truck"}:
        if direction == "ahead" and distance is not None and distance <= 3.0:
            return True

    return False


# ============================================================
# START
# ============================================================

print()
print("============================================================")
print("              SAHAYAK DRISHTI AI")
print("             FAST PHONE NAVIGATION")
print("============================================================")
print()

# Ask user mode
while True:
    mode = input("Are you sitting or walking?\n[s] Sitting\n[w] Walking\n> ").strip().lower()

    if mode in ("s", "w"):
        break

    print("Please enter s or w.")

if mode == "s":
    print("\nMode: SITTING")
else:
    print("\nMode: WALKING")

# Camera URL
if PHONE_URL is None:
    PHONE_URL = input("\nEnter phone camera URL:\n> ").strip()

print("\nLoading YOLOv8n...")
model = YOLO(MODEL_PATH)

print("Connecting to phone camera...")
cap = cv2.VideoCapture(PHONE_URL)

# Reduce OpenCV buffering
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("\nERROR: Could not open phone camera.")
    print("Check that IP Webcam is running.")
    sys.exit(1)

print("Camera connected.")
print("Fast mode started.")
print("Press Ctrl+C to stop.")
print()

# ============================================================
# STATE
# ============================================================

frame_count = 0

last_alert_time = 0

last_object = None
last_direction = None

# Used for simple movement estimation
previous_positions = {}

# Prevent "path clear" spam
path_blocked = False

# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("[WARNING] Camera frame unavailable.")
            time.sleep(0.2)
            continue

        frame_count += 1

        # ----------------------------------------------------
        # Skip frames to improve Raspberry Pi speed
        # ----------------------------------------------------

        if frame_count % FRAME_SKIP != 0:
            continue

        # ----------------------------------------------------
        # Resize BEFORE YOLO
        # Huge performance improvement
        # ----------------------------------------------------

        frame = cv2.resize(frame, (640, 360))

        frame_height, frame_width = frame.shape[:2]

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            imgsz=YOLO_SIZE,
            conf=CONFIDENCE,
            verbose=False,
            device="cpu"
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0])

                if confidence < CONFIDENCE:
                    continue

                class_id = int(box.cls[0])
                name = model.names[class_id]

                if name not in ALLOWED_OBJECTS:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                box_width = x2 - x1
                box_height = y2 - y1

                if box_height <= 5:
                    continue

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                direction = get_direction(center_x, frame_width)

                distance = estimate_distance(
                    name,
                    box_height
                )

                detections.append({
                    "name": name,
                    "confidence": confidence,
                    "x": center_x,
                    "y": center_y,
                    "height": box_height,
                    "direction": direction,
                    "distance": distance,
                })

        # ----------------------------------------------------
        # Nothing detected
        # ----------------------------------------------------

        if not detections:

            if path_blocked:

                now = time.time()

                if now - last_alert_time >= ALERT_COOLDOWN:

                    print("[INFO] Path is clear.")

                    last_alert_time = now
                    path_blocked = False

            continue

        # ----------------------------------------------------
        # Sort:
        # 1. Ahead first
        # 2. Larger object first
        # ----------------------------------------------------

        detections.sort(
            key=lambda d: (
                d["direction"] != "ahead",
                -d["height"]
            )
        )

        best = detections[0]

        name = best["name"]
        direction = best["direction"]
        distance = best["distance"]

        # ----------------------------------------------------
        # SIMPLE MOVEMENT ESTIMATION
        # ----------------------------------------------------

        movement = ""

        previous = previous_positions.get(name)

        if previous is not None:

            old_x, old_y, old_time = previous

            dx = best["x"] - old_x
            dy = best["y"] - old_y

            elapsed = time.time() - old_time

            if elapsed > 0:

                # Object moving toward camera:
                # bounding box becoming taller
                old_height = previous_positions.get(
                    name + "_height",
                    best["height"]
                )

                height_change = best["height"] - old_height

                if height_change > 12:
                    movement = "moving toward you"

                elif height_change < -12:
                    movement = "moving away"

                elif abs(dx) > 35:

                    if dx > 0:
                        movement = "moving left to right"
                    else:
                        movement = "moving right to left"

        previous_positions[name] = (
            best["x"],
            best["y"],
            time.time()
        )

        previous_positions[name + "_height"] = best["height"]

        # ----------------------------------------------------
        # CURRENT TIME
        # ----------------------------------------------------

        now = time.time()

        # ----------------------------------------------------
        # SITTING MODE
        # ----------------------------------------------------

        if mode == "s":

            # Only announce when object/direction changes
            changed = (
                name != last_object or
                direction != last_direction
            )

            if changed and now - last_alert_time >= ALERT_COOLDOWN:

                message = get_position_message(
                    name,
                    direction
                )

                print(f"[ALERT] {message}")

                last_alert_time = now

                last_object = name
                last_direction = direction

            continue

        # ----------------------------------------------------
        # WALKING MODE
        # ----------------------------------------------------

        dangerous = is_dangerous(
            name,
            direction,
            distance
        )

        # ----------------------------------------------------
        # EMERGENCY
        # ----------------------------------------------------

        if dangerous and now - last_alert_time >= ALERT_COOLDOWN:

            d_text = distance_text(distance)

            if d_text:
                message = (
                    f"Careful, {name} {direction}, "
                    f"{d_text}."
                )
            else:
                message = (
                    f"Careful, {name} {direction}."
                )

            print(f"[EMERGENCY] {message}")

            last_alert_time = now

            path_blocked = True

            last_object = name
            last_direction = direction

            continue

        # ----------------------------------------------------
        # NORMAL WALKING ALERT
        # ----------------------------------------------------

        changed = (
            name != last_object or
            direction != last_direction
        )

        if changed and now - last_alert_time >= ALERT_COOLDOWN:

            message = get_walking_message(
                name,
                direction,
                distance
            )

            if movement:
                message = message[:-1] + f", {movement}."

            print(f"[ALERT] {message}")

            last_alert_time = now

            last_object = name
            last_direction = direction

            if direction == "ahead":
                path_blocked = True

        # ----------------------------------------------------
        # CLEAN OLD POSITION DATA
        # ----------------------------------------------------

        if len(previous_positions) > 30:

            keys = list(previous_positions.keys())

            for key in keys[:-20]:
                del previous_positions[key]

except KeyboardInterrupt:

    print("\n")
    print("============================================================")
    print("             NAVIGATION STOPPED")
    print("============================================================")

finally:

    cap.release()
    cv2.destroyAllWindows()
