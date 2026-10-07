import cv2
import math
import time
import sys
from collections import deque
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolov8n.pt"

# Phone URL is supplied when running the program
PHONE_URL = sys.argv[1] if len(sys.argv) > 1 else None

IMG_SIZE = 384
CONFIDENCE = 0.45
FRAME_SKIP = 3

ALERT_COOLDOWN = 3.0
EMERGENCY_COOLDOWN = 1.5

MAX_TRACK_DISTANCE = 110
MAX_TRACK_AGE = 0.8
MIN_CONFIRMATIONS = 2

MOVEMENT_HISTORY = 8
MIN_HORIZONTAL_MOVEMENT = 35
MIN_VERTICAL_MOVEMENT = 25
MIN_DISTANCE_CHANGE = 0.18

# Walking corridor
LEFT_BOUNDARY = 0.32
RIGHT_BOUNDARY = 0.68

# Objects useful for navigation
ALLOWED_OBSTACLES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "chair",
    "dining table",
    "bench",
    "backpack",
    "suitcase"
}

# Approximate real-world heights in meters
AVERAGE_HEIGHTS = {
    "person": 1.70,
    "bicycle": 1.00,
    "motorcycle": 1.10,
    "car": 1.50,
    "bus": 3.00,
    "truck": 3.00,
    "dog": 0.50,
    "chair": 0.85,
    "dining table": 0.75,
    "bench": 0.50,
    "backpack": 0.45,
    "suitcase": 0.70
}

FOCAL_LENGTH_PIXELS = 650.0


# ============================================================
# CHECK PHONE URL
# ============================================================

if not PHONE_URL:

    print("=" * 68)
    print("              SAHAYAK DRISHTI")
    print("           PHONE NAVIGATION MODE")
    print("=" * 68)

    print("\n[ERROR] Phone camera URL was not provided.")
    print("\nRun the program like this:")
    print(
        'python3 phone_navigation.py "http://PHONE_IP:8080/video"'
    )
    print("\nExample:")
    print(
        'python3 phone_navigation.py "http://10.108.217.59:8080/video"'
    )

    sys.exit(1)


# ============================================================
# INITIALIZATION
# ============================================================

print("=" * 68)
print("              SAHAYAK DRISHTI")
print("           PHONE NAVIGATION MODE")
print("=" * 68)

print(f"\n[SYSTEM] Phone stream: {PHONE_URL}")
print("[SYSTEM] Loading YOLOv8n...")

model = YOLO(MODEL_PATH)

print("[SYSTEM] Connecting to phone camera...")

cap = cv2.VideoCapture(PHONE_URL)

if not cap.isOpened():

    print("[ERROR] Cannot connect to phone camera.")
    print("[ERROR] Check the IP Webcam URL and Wi-Fi connection.")

    sys.exit(1)

print("[SYSTEM] Phone camera connected.")
print("[SYSTEM] YOLO navigation engine started.\n")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

frame_h = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
) or 1080

frame_w = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
) or 1920

b1 = frame_w * LEFT_BOUNDARY
b2 = frame_w * RIGHT_BOUNDARY

tracks = {}
next_track_id = 1

last_alert_time = -10.0
last_emergency_time = -10.0
last_alert_signature = ""

frame_count = 0


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def format_time(seconds):

    return (
        f"{int(seconds // 60):02d}:"
        f"{int(seconds % 60):02d}"
    )


def distance(x1, y1, x2, y2):

    return math.hypot(
        x2 - x1,
        y2 - y1
    )


def estimate_distance_meters(label, box_h):

    if label not in AVERAGE_HEIGHTS:
        return None

    if box_h <= 5:
        return None

    raw_distance = (
        AVERAGE_HEIGHTS[label]
        * FOCAL_LENGTH_PIXELS
    ) / box_h

    raw_distance = max(
        0.5,
        min(20.0, raw_distance)
    )

    if raw_distance < 1.8:
        return 1

    elif raw_distance < 3.8:
        return 3

    elif raw_distance < 6.5:
        return 5

    return round(
        raw_distance / 5.0
    ) * 5


def get_zone(x):

    if x < b1:
        return "left"

    elif x > b2:
        return "right"

    return "center"


# ============================================================
# TRACKING
# ============================================================

def match_detections(detections, current_sec):

    global next_track_id

    matched = set()

    for det in detections:

        best_id = None
        min_d = float("inf")

        for track_id, track in tracks.items():

            if track_id in matched:
                continue

            if track["label"] != det["label"]:
                continue

            if (
                current_sec
                - track["last_seen"]
                > MAX_TRACK_AGE
            ):
                continue

            d = distance(
                det["x"],
                det["y"],
                track["x"],
                track["y"]
            )

            if d < min_d:

                min_d = d
                best_id = track_id

        dist_m = estimate_distance_meters(
            det["label"],
            det["box_h"]
        )

        # ----------------------------------------------------
        # Existing track
        # ----------------------------------------------------

        if (
            best_id is not None
            and min_d <= MAX_TRACK_DISTANCE
        ):

            track = tracks[best_id]

            track["x"] = det["x"]
            track["y"] = det["y"]

            track["box_h"] = det["box_h"]
            track["box_w"] = det["box_w"]

            track["dist_m"] = dist_m

            track["last_seen"] = current_sec

            track["frames_seen"] += 1

            track["zone"] = get_zone(
                det["x"]
            )

            track["history"].append({

                "x": det["x"],

                "y": det["y"],

                "box_h": det["box_h"],

                "dist": dist_m
            })

            matched.add(best_id)

        # ----------------------------------------------------
        # New track
        # ----------------------------------------------------

        else:

            track_id = next_track_id

            next_track_id += 1

            tracks[track_id] = {

                "id": track_id,

                "label": det["label"],

                "x": det["x"],
                "y": det["y"],

                "box_h": det["box_h"],
                "box_w": det["box_w"],

                "dist_m": dist_m,

                "last_seen": current_sec,

                "frames_seen": 1,

                "zone": get_zone(
                    det["x"]
                ),

                "history": deque(
                    [{
                        "x": det["x"],
                        "y": det["y"],
                        "box_h": det["box_h"],
                        "dist": dist_m
                    }],
                    maxlen=MOVEMENT_HISTORY
                )
            }

            matched.add(track_id)

    # --------------------------------------------------------
    # Remove old tracks
    # --------------------------------------------------------

    old_tracks = [

        track_id

        for track_id, track
        in tracks.items()

        if (
            current_sec
            - track["last_seen"]
            > MAX_TRACK_AGE
        )
    ]

    for track_id in old_tracks:

        del tracks[track_id]


# ============================================================
# MOVEMENT
# ============================================================

def get_movement(track):

    history = track["history"]

    if len(history) < 4:

        return "stationary"

    first = history[0]
    last = history[-1]

    dx = last["x"] - first["x"]
    dy = last["y"] - first["y"]

    # --------------------------------------------------------
    # Horizontal movement
    # --------------------------------------------------------

    if abs(dx) >= MIN_HORIZONTAL_MOVEMENT:

        if dx > 0:

            return "left_to_right"

        return "right_to_left"

    # --------------------------------------------------------
    # Approaching / moving away
    # --------------------------------------------------------

    if (
        first["dist"] is not None
        and last["dist"] is not None
    ):

        change = (
            first["dist"]
            - last["dist"]
        )

        if change >= MIN_DISTANCE_CHANGE:

            if (
                abs(dy)
                >= MIN_VERTICAL_MOVEMENT

                or abs(
                    last["box_h"]
                    - first["box_h"]
                ) >= 10
            ):

                return "toward"

        elif change <= -MIN_DISTANCE_CHANGE:

            if (
                abs(dy)
                >= MIN_VERTICAL_MOVEMENT

                or abs(
                    last["box_h"]
                    - first["box_h"]
                ) >= 10
            ):

                return "away"

    return "stationary"


# ============================================================
# NAVIGATION MESSAGE
# ============================================================

def create_navigation_message(track):

    label = track["label"]

    zone = track["zone"]

    dist_m = track["dist_m"]

    movement = get_movement(track)

    if label == "dining table":

        label = "table"

    if dist_m is not None:

        distance_text = (
            f"about {dist_m} meters"
        )

    else:

        distance_text = "nearby"

    # --------------------------------------------------------
    # CENTER
    # --------------------------------------------------------

    if zone == "center":

        if movement == "toward":

            return (
                f"{label.capitalize()} ahead, "
                f"{distance_text}, moving toward you. "
                f"Be careful."
            )

        if movement == "left_to_right":

            return (
                f"{label.capitalize()} ahead, "
                f"{distance_text}, moving left to right. "
                f"Be careful."
            )

        if movement == "right_to_left":

            return (
                f"{label.capitalize()} ahead, "
                f"{distance_text}, moving right to left. "
                f"Be careful."
            )

        if (
            dist_m is not None
            and dist_m <= 1
        ):

            return (
                f"Careful, {label} ahead, "
                f"about 1 meter."
            )

        return (
            f"{label.capitalize()} ahead, "
            f"{distance_text}. "
            f"Move slightly left or right."
        )

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    if zone == "left":

        return (
            f"{label.capitalize()} on your left, "
            f"{distance_text}. "
            f"Stay slightly right."
        )

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    return (
        f"{label.capitalize()} on your right, "
        f"{distance_text}. "
        f"Stay slightly left."
    )


# ============================================================
# MAIN INFERENCE LOOP
# ============================================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:

        print(
            "\n[ERROR] Lost phone camera stream."
        )

        break

    frame_count += 1

    current_sec = frame_count / fps

    if frame_count % FRAME_SKIP != 0:

        continue

    # ========================================================
    # YOLO
    # ========================================================

    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )

    boxes = results[0].boxes

    detections = []

    if (
        boxes is not None
        and len(boxes) > 0
    ):

        for box in boxes:

            cls_id = int(
                box.cls[0]
            )

            label = model.names[
                cls_id
            ]

            if label not in ALLOWED_OBSTACLES:

                continue

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            detections.append({

                "label": label,

                "x": (
                    x1 + x2
                ) / 2,

                "y": (
                    y1 + y2
                ) / 2,

                "box_w": x2 - x1,

                "box_h": y2 - y1
            })

    # ========================================================
    # UPDATE TRACKS
    # ========================================================

    match_detections(
        detections,
        current_sec
    )

    # ========================================================
    # VISIBLE OBJECTS
    # ========================================================

    visible = [

        track

        for track in tracks.values()

        if (
            current_sec
            - track["last_seen"]
            <= 0.4

            and track["frames_seen"]
            >= MIN_CONFIRMATIONS
        )
    ]

    if not visible:

        continue

    # ========================================================
    # PRIORITIZE OBJECTS IN WALKING PATH
    # ========================================================

    visible.sort(

        key=lambda track: (

            track["zone"] == "center",

            track["dist_m"] is not None,

            -(track["dist_m"] or 999)
        ),

        reverse=True
    )

    lead = visible[0]

    # ========================================================
    # EMERGENCY
    # ========================================================

    emergency = (

        lead["zone"] == "center"

        and (

            lead["box_h"]
            / frame_h >= 0.38

            or (

                lead["dist_m"]
                is not None

                and lead["dist_m"] <= 1
            )
        )
    )

    emergency_allowed = (

        current_sec
        - last_emergency_time
        >= EMERGENCY_COOLDOWN
    )

    cooldown_expired = (

        current_sec
        - last_alert_time
        >= ALERT_COOLDOWN
    )

    # ========================================================
    # MESSAGE SIGNATURE
    # ========================================================

    movement = get_movement(lead)

    signature = (

        f"{lead['label']}_"
        f"{lead['zone']}_"
        f"{lead['dist_m']}_"
        f"{movement}"
    )

    should_alert = False

    # ========================================================
    # EMERGENCY ALERT
    # ========================================================

    if (
        emergency
        and emergency_allowed
    ):

        should_alert = True

        last_emergency_time = current_sec

    # ========================================================
    # NORMAL ALERT
    # ========================================================

    elif (
        cooldown_expired
        and signature
        != last_alert_signature
    ):

        should_alert = True

    # ========================================================
    # OUTPUT
    # ========================================================

    if should_alert:

        timestamp = format_time(
            current_sec
        )

        message = (
            create_navigation_message(
                lead
            )
        )

        if emergency:

            prefix = "[EMERGENCY]"

        else:

            prefix = "[NAVIGATION]"

        print(
            f"[{timestamp}] "
            f"{prefix} {message}"
        )

        last_alert_time = current_sec

        last_alert_signature = signature


# ============================================================
# CLEANUP
# ============================================================

cap.release()

print("\n" + "=" * 68)
print("             NAVIGATION TEST COMPLETE")
print("=" * 68)
