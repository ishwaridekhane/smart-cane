import cv2
import time
import sys
from collections import defaultdict, deque
from ultralytics import YOLO


# ============================================================
# SAHAYAK DRISHTI AI - PHONE NAVIGATION
# ============================================================

MODEL_PATH = "yolov8n.pt"
DEFAULT_URL = "http://10.108.217.59:8080/video"

IMG_SIZE = 384
CONFIDENCE = 0.45
FRAME_SKIP = 3
MESSAGE_COOLDOWN = 2.0

NAV_OBJECTS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "chair",
    "bench",
    "backpack",
    "suitcase",
    "dining table",
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
        return "moving right" if dx > 0 else "moving left"

    if dy > 12:
        return "approaching"

    return "moving away"


# ============================================================
# LEFT / AHEAD / RIGHT
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

def navigation_message(detections, walking):

    if not detections:
        return None, False

    # --------------------------------------------------------
    # PERSON AHEAD = HIGHEST PRIORITY
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

        if person["distance"] == "very close":
            return (
                "Person very close ahead. Be careful.",
                True
            )

        if person["movement"] == "approaching":
            return (
                f"Person ahead, {person['distance']}, "
                f"moving toward you. Be careful.",
                True
            )

        return (
            f"Person ahead, {person['distance']}.",
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
    # CLOSE SIDE OBJECT
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

    video_url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else DEFAULT_URL
    )

    print()
    print("=" * 55)
    print("SAHAYAK DRISHTI AI")
    print("PHONE NAVIGATION")
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

        print("Enter s or w.")

    walking = mode == "w"

    print()
    print("[MODE]", "Walking" if walking else "Sitting / Testing")
    print()

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print("[YOLO] Loading model...")

    model = YOLO(MODEL_PATH)

    print("[YOLO] Model loaded.")

    # --------------------------------------------------------
    # CAMERA
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

    frame_number = 0

    last_message = ""
    last_message_time = 0

    path_blocked = False

    # ========================================================
    # LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            print("[CAMERA] Frame lost.")
            time.sleep(0.2)
            continue

        frame_number += 1

        if frame_number % FRAME_SKIP != 0:
            continue

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        display = cv2.resize(frame, (640, 360))

        height, width = display.shape[:2]

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model(
            display,
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

                if name not in NAV_OBJECTS:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                box_height = y2 - y1

                if box_height < 18:
                    continue

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                direction = get_direction(
                    cx,
                    width
                )

                distance = get_distance(
                    box_height,
                    height,
                    name
                )

                movement = get_movement(
                    name,
                    cx,
                    cy
                )

                detections.append({
                    "name": name,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "box_height": box_height,
                    "direction": direction,
                    "distance": distance,
                    "movement": movement
                })

        # ----------------------------------------------------
        # PATH
        # ----------------------------------------------------

        blocked = any(
            d["direction"] == "ahead"
            and d["distance"] != "far"
            for d in detections
        )

        # Say clear ONLY after previously blocked.
        if path_blocked and not blocked:

            now = time.time()

            if now - last_message_time > MESSAGE_COOLDOWN:

                print("[INFO] Path is clear.")

                last_message = "Path is clear."
                last_message_time = now

        path_blocked = blocked

        # ----------------------------------------------------
        # NAVIGATION
        # ----------------------------------------------------

        message, emergency = navigation_message(
            detections,
            walking
        )

        if message:

            now = time.time()

            if (
                message != last_message
                or now - last_message_time > MESSAGE_COOLDOWN
            ):

                tag = (
                    "[EMERGENCY]"
                    if emergency
                    else "[INFO]"
                )

                print(f"{tag} {message}")

                last_message = message
                last_message_time = now

        # ----------------------------------------------------
        # DISPLAY ONLY NECESSARY INFORMATION
        # ----------------------------------------------------

        for d in detections:

            x1 = int(d["x1"])
            y1 = int(d["y1"])
            x2 = int(d["x2"])
            y2 = int(d["y2"])

            label = (
                f"{d['name']} | "
                f"{d['direction']} | "
                f"{d['distance']}"
            )

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                label,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1
            )

        # Only mode at the top.
        cv2.putText(
            display,
            "WALKING" if walking else "SITTING",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Sahayak Drishti",
            display
        )

        # ----------------------------------------------------
        # Q = EXIT
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("Navigation stopped.")


if __name__ == "__main__":
    main()
