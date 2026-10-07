import cv2
import sys
import time
from ultralytics import YOLO

# ============================================================
# SAHAYAK DRISHTI AI
# FAST PHONE NAVIGATION - RASPBERRY PI 4
# ============================================================

MODEL_PATH = "yolov8n.pt"

PHONE_URL = sys.argv[1] if len(sys.argv) > 1 else None

# ------------------------------------------------------------
# SPEED SETTINGS
# ------------------------------------------------------------

YOLO_SIZE = 320
CONFIDENCE = 0.40
FRAME_SKIP = 4

ALERT_COOLDOWN = 2.0

# ------------------------------------------------------------
# CAMERA / POSITION
# ------------------------------------------------------------

LEFT_BOUNDARY = 0.33
RIGHT_BOUNDARY = 0.67

# ------------------------------------------------------------
# DISTANCE CALIBRATION
# ------------------------------------------------------------

CALIBRATION_DISTANCE = 1.0       # Put person 1 meter away
AVERAGE_PERSON_HEIGHT = 1.70     # meters

FOCAL_LENGTH_PIXELS = None

# ------------------------------------------------------------
# OBJECT HEIGHTS
# Approximate real-world heights
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# USEFUL COCO CLASSES
# ------------------------------------------------------------

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


# ============================================================
# POSITION
# ============================================================

def get_direction(center_x, frame_width):

    ratio = center_x / frame_width

    if ratio < LEFT_BOUNDARY:
        return "left"

    if ratio > RIGHT_BOUNDARY:
        return "right"

    return "ahead"


# ============================================================
# DISTANCE
# ============================================================

def calculate_focal_length(box_height):

    if box_height <= 0:
        return None

    return (
        box_height * CALIBRATION_DISTANCE
    ) / AVERAGE_PERSON_HEIGHT


def estimate_distance(object_name, box_height):

    global FOCAL_LENGTH_PIXELS

    if box_height <= 0:
        return None

    if FOCAL_LENGTH_PIXELS is None:
        return None

    real_height = OBJECT_HEIGHTS.get(object_name)

    if real_height is None:
        return None

    distance = (
        real_height * FOCAL_LENGTH_PIXELS
    ) / box_height

    # Don't report ridiculous values
    distance = max(0.10, min(distance, 10.0))

    return distance


def distance_text(distance):

    if distance is None:
        return ""

    if distance < 0.30:
        return "about 20 centimeters"

    elif distance < 0.70:
        return "about 50 centimeters"

    elif distance < 1.25:
        return "about 1 meter"

    elif distance < 1.75:
        return "about 1.5 meters"

    elif distance < 2.50:
        return "about 2 meters"

    elif distance < 3.50:
        return "about 3 meters"

    elif distance < 4.50:
        return "about 4 meters"

    elif distance < 6.50:
        return "about 6 meters"

    else:
        return "about 10 meters"


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_camera(model, cap):

    global FOCAL_LENGTH_PIXELS

    print()
    print("============================================================")
    print("                 CAMERA CALIBRATION")
    print("============================================================")
    print()
    print("Stand about 1 METER in front of the phone.")
    print("Make sure your FULL BODY is visible.")
    print()
    print("The program will automatically detect you.")
    print("Please stay still for a few seconds.")
    print()

    samples = []

    start_time = time.time()

    while time.time() - start_time < 12:

        ret, frame = cap.read()

        if not ret:
            continue

        # Keep processing lightweight
        frame = cv2.resize(frame, (640, 360))

        results = model.predict(
            source=frame,
            imgsz=YOLO_SIZE,
            conf=0.45,
            verbose=False,
            device="cpu"
        )

        best_height = None
        best_conf = 0

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])
                name = model.names[class_id]

                if name != "person":
                    continue

                confidence = float(box.conf[0])

                if confidence < best_conf:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                height = y2 - y1

                # Need a reasonably visible person
                if height < 50:
                    continue

                best_height = height
                best_conf = confidence

        if best_height is not None:

            samples.append(best_height)

            print(
                f"\rCalibrating... "
                f"{len(samples)}/10 samples detected",
                end="",
                flush=True
            )

        if len(samples) >= 10:
            break

    print()

    if len(samples) < 5:

        print()
        print("[WARNING] Could not calibrate camera.")
        print("Using safe fallback calibration.")
        print()

        # Fallback only
        FOCAL_LENGTH_PIXELS = 500.0

        return

    # Remove extreme values
    samples.sort()

    middle = samples[
        max(0, len(samples) // 4):
        min(len(samples), (len(samples) * 3) // 4)
    ]

    average_height = sum(middle) / len(middle)

    FOCAL_LENGTH_PIXELS = calculate_focal_length(
        average_height
    )

    print()
    print(
        f"[CALIBRATION] Person box height: "
        f"{average_height:.1f}px"
    )

    print(
        f"[CALIBRATION] Focal length: "
        f"{FOCAL_LENGTH_PIXELS:.1f}px"
    )

    print("[CALIBRATION] Distance system ready.")
    print()


# ============================================================
# SIMPLE MOVEMENT
# ============================================================

previous_objects = {}


def get_movement(name, x, y, height):

    old = previous_objects.get(name)

    previous_objects[name] = (
        x,
        y,
        height,
        time.time()
    )

    if old is None:
        return ""

    old_x, old_y, old_height, old_time = old

    # Bounding box getting significantly bigger
    if height - old_height > 15:
        return "moving toward you"

    # Bounding box getting significantly smaller
    if old_height - height > 15:
        return "moving away"

    dx = x - old_x

    if dx > 35:
        return "moving left to right"

    if dx < -35:
        return "moving right to left"

    return ""


# ============================================================
# SAFETY
# ============================================================

def is_dangerous(name, direction, distance):

    if distance is None:
        return False

    # Anything extremely close ahead
    if direction == "ahead" and distance <= 0.8:
        return True

    # Person ahead
    if (
        name == "person"
        and direction == "ahead"
        and distance <= 2.5
    ):
        return True

    # Vehicle / bicycle ahead
    if (
        name in {
            "bicycle",
            "motorcycle",
            "car",
            "bus",
            "truck"
        }
        and direction == "ahead"
        and distance <= 3.0
    ):
        return True

    return False


# ============================================================
# MAIN
# ============================================================

print()
print("============================================================")
print("                 SAHAYAK DRISHTI AI")
print("              FAST PHONE NAVIGATION")
print("============================================================")
print()

# ------------------------------------------------------------
# MODE
# ------------------------------------------------------------

while True:

    mode = input(
        "Are you sitting or walking?\n"
        "[s] Sitting\n"
        "[w] Walking\n"
        "> "
    ).strip().lower()

    if mode in ("s", "w"):
        break

    print("Please enter s or w.")


if mode == "s":
    print("\nMode: SITTING")

else:
    print("\nMode: WALKING")


# ------------------------------------------------------------
# PHONE URL
# ------------------------------------------------------------

if PHONE_URL is None:

    PHONE_URL = input(
        "\nEnter phone camera URL:\n> "
    ).strip()


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\nLoading YOLOv8n...")

model = YOLO(MODEL_PATH)


# ------------------------------------------------------------
# CAMERA
# ------------------------------------------------------------

print("Connecting to phone camera...")

cap = cv2.VideoCapture(PHONE_URL)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():

    print()
    print("[ERROR] Could not open phone camera.")
    print("Make sure IP Webcam is running.")
    sys.exit(1)


print("Camera connected.")


# ------------------------------------------------------------
# CALIBRATE
# ------------------------------------------------------------

calibrate_camera(
    model,
    cap
)


print("Fast navigation started.")
print("Press Ctrl+C to stop.")
print()


# ============================================================
# STATE
# ============================================================

frame_count = 0

last_alert_time = 0

last_object = None
last_direction = None

path_blocked = False


# ============================================================
# LOOP
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "[WARNING] Camera frame unavailable."
            )

            time.sleep(0.1)

            continue


        frame_count += 1


        # ----------------------------------------------------
        # FRAME SKIPPING
        # ----------------------------------------------------

        if frame_count % FRAME_SKIP != 0:
            continue


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        frame = cv2.resize(
            frame,
            (640, 360)
        )

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


        # ----------------------------------------------------
        # READ DETECTIONS
        # ----------------------------------------------------

        for result in results:

            if result.boxes is None:
                continue


            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )


                if confidence < CONFIDENCE:
                    continue


                class_id = int(
                    box.cls[0]
                )

                name = model.names[class_id]


                if name not in ALLOWED_OBJECTS:
                    continue


                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                box_height = y2 - y1


                if box_height < 8:
                    continue


                center_x = (
                    x1 + x2
                ) // 2


                center_y = (
                    y1 + y2
                ) // 2


                direction = get_direction(
                    center_x,
                    frame_width
                )


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
        # NOTHING DETECTED
        # ----------------------------------------------------

        if not detections:

            if path_blocked:

                now = time.time()

                if (
                    now - last_alert_time
                    >= ALERT_COOLDOWN
                ):

                    print(
                        "[INFO] Path is clear."
                    )

                    last_alert_time = now

                    path_blocked = False

            continue


        # ----------------------------------------------------
        # CHOOSE MOST IMPORTANT OBJECT
        #
        # Ahead > side
        # Larger object > smaller object
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

        height = best["height"]

        x = best["x"]

        y = best["y"]


        # ----------------------------------------------------
        # MOVEMENT
        # ----------------------------------------------------

        movement = get_movement(
            name,
            x,
            y,
            height
        )


        now = time.time()


        # ====================================================
        # SITTING MODE
        # ====================================================

        if mode == "s":

            changed = (
                name != last_object
                or direction != last_direction
            )


            if (
                changed
                and
                now - last_alert_time
                >= ALERT_COOLDOWN
            ):

                if direction == "ahead":

                    message = (
                        f"{name} ahead."
                    )

                else:

                    message = (
                        f"{name} "
                        f"on your {direction}."
                    )


                print(
                    f"[ALERT] {message}"
                )


                last_alert_time = now

                last_object = name

                last_direction = direction


            continue


        # ====================================================
        # WALKING MODE
        # ====================================================

        dangerous = is_dangerous(
            name,
            direction,
            distance
        )


        # ----------------------------------------------------
        # EMERGENCY
        # ----------------------------------------------------

        if (
            dangerous
            and
            now - last_alert_time
            >= ALERT_COOLDOWN
        ):

            d_text = distance_text(
                distance
            )


            if d_text:

                message = (
                    f"Careful, "
                    f"{name} "
                    f"{direction}, "
                    f"{d_text}."
                )

            else:

                message = (
                    f"Careful, "
                    f"{name} "
                    f"{direction}."
                )


            print(
                f"[EMERGENCY] {message}"
            )


            last_alert_time = now

            path_blocked = True

            last_object = name

            last_direction = direction

            continue


        # ----------------------------------------------------
        # NORMAL ALERT
        # ----------------------------------------------------

        changed = (
            name != last_object
            or direction != last_direction
        )


        if (
            changed
            and
            now - last_alert_time
            >= ALERT_COOLDOWN
        ):

            d_text = distance_text(
                distance
            )


            if direction == "ahead":

                message = (
                    f"{name} ahead"
                )

            else:

                message = (
                    f"{name} "
                    f"on your {direction}"
                )


            if d_text:

                message += (
                    f", {d_text}"
                )


            if movement:

                message += (
                    f", {movement}"
                )


            message += "."


            print(
                f"[ALERT] {message}"
            )


            last_alert_time = now

            last_object = name

            last_direction = direction


            if direction == "ahead":

                path_blocked = True


# ============================================================
# STOP
# ============================================================

except KeyboardInterrupt:

    print()
    print()
    print("============================================================")
    print("              NAVIGATION STOPPED")
    print("============================================================")


finally:

    cap.release()

    cv2.destroyAllWindows()
