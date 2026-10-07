import cv2
import time
import sys
from collections import defaultdict, deque
from ultralytics import YOLO


# ============================================================
# SAHAYAK DRISHTI AI
# PHONE CAMERA NAVIGATION
# ============================================================

MODEL_PATH = "yolov8n.pt"

DEFAULT_URL = "http://10.108.217.59:8080/video"

IMG_SIZE = 384
CONFIDENCE = 0.45
FRAME_SKIP = 3

MESSAGE_COOLDOWN = 2.0


# Objects useful for navigation
NAV_OBJECTS = {
    # People / animals
    "person",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",

    # Vehicles
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",

    # Road / outdoor
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",

    # Bags / personal items
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",

    # Sports
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",

    # Food
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",

    # Furniture / surroundings
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",

    # Electronics
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",

    # Other common objects
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}


# ============================================================
# MOVEMENT TRACKING
# ============================================================

tracks = defaultdict(lambda: deque(maxlen=6))


def get_movement(name, cx, cy):

    history = tracks[name]

    history.append((cx, cy))

    if len(history) < 4:
        return "stationary"

    old_x, old_y = history[0]
    new_x, new_y = history[-1]

    dx = new_x - old_x
    dy = new_y - old_y

    if abs(dx) < 12 and abs(dy) < 12:
        return "stationary"

    if abs(dx) > abs(dy):

        if dx > 0:
            return "moving right"

        return "moving left"

    if dy > 12:
        return "approaching"

    return "moving away"


# ============================================================
# DIRECTION
# ============================================================

def get_direction(cx, width):

    ratio = cx / width

    if ratio < 0.33:
        return "left"

    if ratio > 0.66:
        return "right"

    return "ahead"


# ============================================================
# APPROXIMATE DISTANCE
# ============================================================

def get_distance(box_height, frame_height, name):

    ratio = box_height / frame_height

    # Person
    if name == "person":

        if ratio >= 0.72:
            return "very close"

        if ratio >= 0.48:
            return "about 1 meter"

        if ratio >= 0.32:
            return "about 2 meters"

        if ratio >= 0.22:
            return "about 3 meters"

        if ratio >= 0.14:
            return "about 5 meters"

        return "far"

    # Other objects
    if ratio >= 0.55:
        return "very close"

    if ratio >= 0.35:
        return "about 1 meter"

    if ratio >= 0.23:
        return "about 2 meters"

    if ratio >= 0.15:
        return "about 3 meters"

    if ratio >= 0.10:
        return "about 5 meters"

    return "far"


# ============================================================
# NAVIGATION MESSAGE
# ============================================================

def navigation_message(detections):

    if not detections:
        return None, False

    # --------------------------------------------------------
    # PERSON AHEAD
    # --------------------------------------------------------

    people_ahead = [
        d for d in detections
        if d["name"] == "person"
        and d["direction"] == "ahead"
    ]

    if people_ahead:

        person = max(
            people_ahead,
            key=lambda d: d["box_height"]
        )

        distance = person["distance"]
        movement = person["movement"]

        if distance == "very close":

            return (
                "Person very close ahead. Be careful.",
                True
            )

        if movement == "approaching":

            return (
                f"Person ahead, {distance}, "
                "moving toward you. Be careful.",
                True
            )

        return (
            f"Person ahead, {distance}.",
            False
        )

    # --------------------------------------------------------
    # OTHER OBJECT AHEAD
    # --------------------------------------------------------

    objects_ahead = [
        d for d in detections
        if d["direction"] == "ahead"
    ]

    if objects_ahead:

        obj = max(
            objects_ahead,
            key=lambda d: d["box_height"]
        )

        if obj["distance"] == "very close":

            return (
                f"{obj['name'].capitalize()} very close ahead.",
                True
            )

        return (
            f"{obj['name'].capitalize()} ahead, "
            f"{obj['distance']}.",
            False
        )

    # --------------------------------------------------------
    # CLOSE OBJECT ON LEFT / RIGHT
    # --------------------------------------------------------

    side_objects = [
        d for d in detections
        if d["direction"] in ("left", "right")
        and d["distance"] in (
            "very close",
            "about 1 meter",
            "about 2 meters"
        )
    ]

    if side_objects:

        obj = max(
            side_objects,
            key=lambda d: d["box_height"]
        )

        return (
            f"{obj['name'].capitalize()} on your "
            f"{obj['direction']}, {obj['distance']}.",
            False
        )

    return None, False


# ============================================================
# MAIN
# ============================================================

def main():

    # Use URL from command line if provided
    video_url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else DEFAULT_URL
    )

    print()
    print("=" * 55)
    print("SAHAYAK DRISHTI AI")
    print("PHONE CAMERA NAVIGATION")
    print("=" * 55)
    print()

    # --------------------------------------------------------
    # MODE
    # --------------------------------------------------------

    while True:

        mode = input(
            "Are you sitting or walking?\n"
            "s = Sitting\n"
            "w = Walking\n"
            "> "
        ).strip().lower()

        if mode in ("s", "w"):
            break

        print("Please enter s or w.")

    print()

    if mode == "w":
        print("[MODE] Walking")
    else:
        print("[MODE] Sitting / Testing")

    print()

    # --------------------------------------------------------
    # LOAD YOLO
    # --------------------------------------------------------

    print("[YOLO] Loading model...")

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print("[YOLO ERROR]", e)
        return

    print("[YOLO] Model loaded.")
    print()

    # --------------------------------------------------------
    # CONNECT PHONE
    # --------------------------------------------------------

    print("[CAMERA] Connecting to phone...")

    cap = cv2.VideoCapture(video_url)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():

        print("[ERROR] Could not connect to phone camera.")
        return

    print("[CAMERA] Connected.")
    print()

    print("=" * 55)
    print("NAVIGATION STARTED")
    print("=" * 55)
    print()
    print("Press Ctrl+C to stop.")
    print()

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    frame_number = 0

    last_message = ""
    last_message_time = 0

    path_was_blocked = False

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print("[CAMERA] Frame lost. Waiting...")
                time.sleep(0.2)
                continue

            frame_number += 1

            # Process every 3rd frame
            if frame_number % FRAME_SKIP != 0:
                continue

            # ------------------------------------------------
            # Resize for faster processing
            # ------------------------------------------------

            frame = cv2.resize(
                frame,
                (640, 360)
            )

            height, width = frame.shape[:2]

            # ------------------------------------------------
            # YOLO DETECTION
            # ------------------------------------------------

            results = model(
                frame,
                imgsz=IMG_SIZE,
                conf=CONFIDENCE,
                verbose=False
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

                    # Ignore irrelevant COCO objects
                    if name not in NAV_OBJECTS:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    box_height = y2 - y1

                    # Ignore tiny detections
                    if box_height < 18:
                        continue

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    direction = get_direction(
                        center_x,
                        width
                    )

                    distance = get_distance(
                        box_height,
                        height,
                        name
                    )

                    movement = get_movement(
                        name,
                        center_x,
                        center_y
                    )

                    detections.append({
                        "name": name,
                        "confidence": confidence,
                        "box_height": box_height,
                        "direction": direction,
                        "distance": distance,
                        "movement": movement
                    })

            # ------------------------------------------------
            # PATH STATUS
            # ------------------------------------------------

            path_blocked = any(
                d["direction"] == "ahead"
                and d["distance"] != "far"
                for d in detections
            )

            # Only announce clear after a previous blockage
            if path_was_blocked and not path_blocked:

                now = time.time()

                if now - last_message_time > MESSAGE_COOLDOWN:

                    print("[INFO] Path is clear.")

                    last_message = "Path is clear."
                    last_message_time = now

            path_was_blocked = path_blocked

            # ------------------------------------------------
            # NAVIGATION
            # ------------------------------------------------

            message, emergency = navigation_message(
                detections
            )

            if message:

                now = time.time()

                # Prevent constant repetition
                if (
                    message != last_message
                    or now - last_message_time
                    > MESSAGE_COOLDOWN
                ):

                    if emergency:
                        print(
                            f"[EMERGENCY] {message}"
                        )
                    else:
                        print(
                            f"[INFO] {message}"
                        )

                    last_message = message
                    last_message_time = now

    except KeyboardInterrupt:

        print()
        print("[SYSTEM] Stopping navigation...")

    finally:

        cap.release()

        print("[SYSTEM] Camera released.")
        print("[SYSTEM] Navigation stopped.")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
