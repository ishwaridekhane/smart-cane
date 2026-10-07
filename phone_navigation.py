import cv2
import sys
import math
import time
from collections import deque
from ultralytics import YOLO

# ============================================================
# SAHAYAK DRISHTI - PHONE WEBCAM NAVIGATION
# ============================================================
# Camera source:
# Android phone -> IP Webcam -> Wi-Fi -> Raspberry Pi
#
# Modes:
#   Sitting -> identify useful objects only
#   Walking -> navigation + distance + movement + warnings
#
# Model:
#   yolov8n.pt
#
# IMPORTANT:
# Standard YOLOv8n COCO does NOT detect:
#   door, headphones, hand, face
# as separate classes.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolov8n.pt"

PHONE_URL = (
    sys.argv[1]
    if len(sys.argv) > 1
    else None
)

IMG_SIZE = 640

# Slightly lower than old 0.45, but confirmation prevents
# one weak detection from immediately becoming an alert.
CONFIDENCE = 0.35

FRAME_SKIP = 2

# Normal alerts
ALERT_COOLDOWN = 3.0

# Emergency alerts
EMERGENCY_COOLDOWN = 2.0

# Detection must survive several processed frames
MIN_CONFIRMATIONS = 3

# Tracking
MAX_TRACK_DISTANCE = 130
MAX_TRACK_AGE = 1.0

# Movement history
MOVEMENT_HISTORY = 8

MIN_HORIZONTAL_MOVEMENT = 40
MIN_VERTICAL_MOVEMENT = 30

# Distance change threshold
MIN_DISTANCE_CHANGE = 0.25

# Walking corridor
LEFT_BOUNDARY = 0.32
RIGHT_BOUNDARY = 0.68

# Objects useful for this project
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
    "umbrella",
    "handbag",
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
    "clock",
    "potted plant"
}

# Approximate real-world heights.
# These are only rough estimates.
AVERAGE_HEIGHTS = {
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
    "umbrella": 0.90,
    "handbag": 0.30,
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
    "clock": 0.30,
    "potted plant": 0.50
}

FOCAL_LENGTH_PIXELS = 650.0


# ============================================================
# PHONE URL CHECK
# ============================================================

if not PHONE_URL:

    print("=" * 68)
    print("              SAHAYAK DRISHTI")
    print("           PHONE NAVIGATION MODE")
    print("=" * 68)

    print("\n[ERROR] Phone camera URL was not provided.")

    print("\nRun:")
    print(
        'python3 phone_navigation.py '
        '"http://PHONE_IP:8080/video"'
    )

    print("\nExample:")
    print(
        'python3 phone_navigation.py '
        '"http://10.108.217.59:8080/video"'
    )

    sys.exit(1)


# ============================================================
# STARTUP
# ============================================================

print("=" * 68)
print("              SAHAYAK DRISHTI")
print("           PHONE NAVIGATION MODE")
print("=" * 68)

print("\nAre you sitting or walking?")
print("  s = Sitting")
print("  w = Walking")

while True:

    mode_input = input(
        "\nEnter s or w: "
    ).strip().lower()

    if mode_input == "s":
        MODE = "sitting"
        break

    if mode_input == "w":
        MODE = "walking"
        break

    print(
        "[ERROR] Please enter only s or w."
    )

print(
    f"\n[SYSTEM] "
    f"{MODE.capitalize()} mode selected."
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    f"\n[SYSTEM] Phone stream: {PHONE_URL}"
)

print(
    "[SYSTEM] Loading YOLOv8n..."
)

try:

    model = YOLO(MODEL_PATH)

except Exception as e:

    print(
        f"[ERROR] Could not load {MODEL_PATH}"
    )

    print(
        f"[ERROR] {e}"
    )

    sys.exit(1)


# ============================================================
# CONNECT PHONE CAMERA
# ============================================================

print(
    "[SYSTEM] Connecting to phone camera..."
)

cap = cv2.VideoCapture(
    PHONE_URL
)

if not cap.isOpened():

    print(
        "[ERROR] Cannot connect to phone camera."
    )

    print(
        "[ERROR] Check IP Webcam and Wi-Fi."
    )

    sys.exit(1)


# Try to reduce internal buffering.
cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

print(
    "[SYSTEM] Phone camera connected."
)

print(
    "[SYSTEM] YOLO navigation engine started.\n"
)


# ============================================================
# CAMERA INFORMATION
# ============================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 1 or math.isnan(fps):

    fps = 30.0

frame_w = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

frame_h = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

if frame_w <= 0:
    frame_w = 1920

if frame_h <= 0:
    frame_h = 1080

left_boundary_px = (
    frame_w * LEFT_BOUNDARY
)

right_boundary_px = (
    frame_w * RIGHT_BOUNDARY
)


# ============================================================
# TRACK STORAGE
# ============================================================

tracks = {}

next_track_id = 1

last_alert_time = -10.0

last_emergency_time = -10.0

last_alert_signature = ""

path_blocked = False

frame_count = 0

timeline_records = []


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def format_time(seconds):

    return (
        f"{int(seconds // 60):02d}:"
        f"{int(seconds % 60):02d}"
    )


def point_distance(
    x1,
    y1,
    x2,
    y2
):

    return math.hypot(
        x2 - x1,
        y2 - y1
    )


# ============================================================
# DISTANCE ESTIMATION
# ============================================================

def estimate_distance_meters(
    label,
    box_height
):

    if label not in AVERAGE_HEIGHTS:
        return None

    if box_height <= 10:
        return None

    real_height = (
        AVERAGE_HEIGHTS[label]
    )

    raw_distance = (
        real_height
        * FOCAL_LENGTH_PIXELS
        / box_height
    )

    # Do not trust extreme estimates.
    if raw_distance < 0.5:
        raw_distance = 0.5

    if raw_distance > 20:
        raw_distance = 20

    # Coarse distance buckets.
    if raw_distance < 1.5:
        return 1

    if raw_distance < 3.0:
        return 2

    if raw_distance < 4.5:
        return 4

    if raw_distance < 7.0:
        return 6

    return 10


# ============================================================
# ZONE
# ============================================================

def get_zone(x):

    if x < left_boundary_px:
        return "left"

    if x > right_boundary_px:
        return "right"

    return "center"


# ============================================================
# OBJECT NAME
# ============================================================

def object_name(label):

    if label == "dining table":
        return "table"

    if label == "cell phone":
        return "phone"

    return label


# ============================================================
# TRACKING
# ============================================================

def update_tracks(
    detections,
    current_time
):

    global next_track_id

    matched_tracks = set()

    # --------------------------------------------------------
    # MATCH NEW DETECTIONS TO EXISTING TRACKS
    # --------------------------------------------------------

    for detection in detections:

        best_track_id = None

        best_distance = float("inf")

        for track_id, track in tracks.items():

            if track_id in matched_tracks:
                continue

            if (
                track["label"]
                != detection["label"]
            ):
                continue

            if (
                current_time
                - track["last_seen"]
                > MAX_TRACK_AGE
            ):
                continue

            d = point_distance(
                detection["x"],
                detection["y"],
                track["x"],
                track["y"]
            )

            if d < best_distance:

                best_distance = d

                best_track_id = track_id

        estimated_distance = (
            estimate_distance_meters(
                detection["label"],
                detection["box_h"]
            )
        )

        # ----------------------------------------------------
        # EXISTING TRACK
        # ----------------------------------------------------

        if (
            best_track_id is not None
            and best_distance
            <= MAX_TRACK_DISTANCE
        ):

            track = tracks[
                best_track_id
            ]

            track["x"] = detection["x"]

            track["y"] = detection["y"]

            track["box_w"] = (
                detection["box_w"]
            )

            track["box_h"] = (
                detection["box_h"]
            )

            track["x1"] = detection["x1"]

            track["x2"] = detection["x2"]

            track["y1"] = detection["y1"]

            track["y2"] = detection["y2"]

            track["dist_m"] = (
                estimated_distance
            )

            track["confidence"] = (
                detection["confidence"]
            )

            track["last_seen"] = (
                current_time
            )

            track["frames_seen"] += 1

            track["zone"] = get_zone(
                detection["x"]
            )

            track["history"].append({

                "x": detection["x"],

                "y": detection["y"],

                "box_h":
                    detection["box_h"],

                "dist":
                    estimated_distance,

                "time":
                    current_time

            })

            matched_tracks.add(
                best_track_id
            )

        # ----------------------------------------------------
        # NEW TRACK
        # ----------------------------------------------------

        else:

            track_id = next_track_id

            next_track_id += 1

            tracks[track_id] = {

                "id":
                    track_id,

                "label":
                    detection["label"],

                "x":
                    detection["x"],

                "y":
                    detection["y"],

                "x1":
                    detection["x1"],

                "y1":
                    detection["y1"],

                "x2":
                    detection["x2"],

                "y2":
                    detection["y2"],

                "box_w":
                    detection["box_w"],

                "box_h":
                    detection["box_h"],

                "dist_m":
                    estimated_distance,

                "confidence":
                    detection["confidence"],

                "last_seen":
                    current_time,

                "frames_seen":
                    1,

                "zone":
                    get_zone(
                        detection["x"]
                    ),

                "history":
                    deque(
                        [{
                            "x":
                                detection["x"],

                            "y":
                                detection["y"],

                            "box_h":
                                detection["box_h"],

                            "dist":
                                estimated_distance,

                            "time":
                                current_time

                        }],
                        maxlen=
                            MOVEMENT_HISTORY
                    )
            }

            matched_tracks.add(
                track_id
            )

    # --------------------------------------------------------
    # REMOVE OLD TRACKS
    # --------------------------------------------------------

    expired = [

        track_id

        for track_id, track
        in tracks.items()

        if (
            current_time
            - track["last_seen"]
            > MAX_TRACK_AGE
        )
    ]

    for track_id in expired:

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

    dx = (
        last["x"]
        - first["x"]
    )

    dy = (
        last["y"]
        - first["y"]
    )

    # --------------------------------------------------------
    # LEFT -> RIGHT / RIGHT -> LEFT
    # --------------------------------------------------------

    if abs(dx) >= MIN_HORIZONTAL_MOVEMENT:

        if dx > 0:
            return "left_to_right"

        return "right_to_left"

    # --------------------------------------------------------
    # TOWARD / AWAY
    # --------------------------------------------------------

    if (
        first["dist"] is not None
        and last["dist"] is not None
    ):

        change = (
            first["dist"]
            - last["dist"]
        )

        if (
            change
            >= MIN_DISTANCE_CHANGE
        ):

            if (
                abs(dy)
                >= MIN_VERTICAL_MOVEMENT

                or abs(
                    last["box_h"]
                    - first["box_h"]
                )
                >= 15
            ):

                return "toward"

        if (
            change
            <= -MIN_DISTANCE_CHANGE
        ):

            if (
                abs(dy)
                >= MIN_VERTICAL_MOVEMENT

                or abs(
                    last["box_h"]
                    - first["box_h"]
                )
                >= 15
            ):

                return "away"

    return "stationary"


def movement_phrase(track):

    movement = get_movement(
        track
    )

    if movement == "left_to_right":

        return (
            "moving left to right "
            "across your path"
        )

    if movement == "right_to_left":

        return (
            "moving right to left "
            "across your path"
        )

    if movement == "toward":

        return (
            "moving toward you"
        )

    if movement == "away":

        return (
            "moving away from you"
        )

    return None


# ============================================================
# CREATE SITTING MESSAGE
# ============================================================

def create_sitting_message(track):

    name = object_name(
        track["label"]
    )

    zone = track["zone"]

    if zone == "center":

        return (
            f"{name.capitalize()} ahead."
        )

    if zone == "left":

        return (
            f"{name.capitalize()} "
            f"on your left."
        )

    return (
        f"{name.capitalize()} "
        f"on your right."
    )


# ============================================================
# CREATE WALKING MESSAGE
# ============================================================

def create_walking_message(track):

    name = object_name(
        track["label"]
    )

    zone = track["zone"]

    dist_m = track["dist_m"]

    movement = movement_phrase(
        track
    )

    # --------------------------------------------------------
    # DISTANCE TEXT
    # --------------------------------------------------------

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

        if movement:

            return (
                f"{name.capitalize()} ahead, "
                f"{distance_text}, "
                f"{movement}. "
                f"Be careful."
            )

        if (
            dist_m is not None
            and dist_m <= 1
        ):

            return (
                f"Careful, "
                f"{name} ahead, "
                f"about 1 meter."
            )

        return (
            f"{name.capitalize()} ahead, "
            f"{distance_text}. "
            f"Stay alert."
        )

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    if zone == "left":

        if movement:

            return (
                f"{name.capitalize()} "
                f"on your left, "
                f"{distance_text}, "
                f"{movement}."
            )

        return (
            f"{name.capitalize()} "
            f"on your left, "
            f"{distance_text}. "
            f"Stay slightly right."
        )

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    if movement:

        return (
            f"{name.capitalize()} "
            f"on your right, "
            f"{distance_text}, "
            f"{movement}."
        )

    return (
        f"{name.capitalize()} "
        f"on your right, "
        f"{distance_text}. "
        f"Stay slightly left."
    )


# ============================================================
# DETERMINE WHETHER OBJECT IS A WALKING HAZARD
# ============================================================

def is_walking_hazard(track):

    label = track["label"]

    zone = track["zone"]

    # Anything in the center is potentially relevant.
    if zone == "center":
        return True

    # Large physical obstacles on either side.
    if label in {
        "chair",
        "couch",
        "dining table",
        "bench"
    }:

        # Object must overlap the walking corridor.
        if (
            track["x2"]
            > left_boundary_px

            and track["x1"]
            < right_boundary_px
        ):

            return True

    # Moving people/vehicles near the corridor.
    if label in {
        "person",
        "bicycle",
        "motorcycle",
        "car",
        "bus",
        "truck"
    }:

        return True

    return False


# ============================================================
# EMERGENCY DECISION
# ============================================================
# IMPORTANT:
# We do NOT trigger emergency merely because a box is large.
#
# We require:
#   1. object is centered
#   2. object has been confirmed
#   3. object is close OR clearly approaching
# ============================================================

def is_emergency(track):

    if track["zone"] != "center":
        return False

    if (
        track["frames_seen"]
        < MIN_CONFIRMATIONS
    ):
        return False

    distance_m = track["dist_m"]

    movement = get_movement(
        track
    )

    # Very close confirmed object.
    if (
        distance_m is not None
        and distance_m <= 1
    ):

        return True

    # Confirmed object approaching.
    if (
        movement == "toward"
        and distance_m is not None
        and distance_m <= 3
    ):

        return True

    return False


# ============================================================
# MAIN LOOP
# ============================================================

print(
    "=" * 68
)

print(
    " [SAHAYAK DRISHTI] "
    "High-Priority Safety Engine: "
    f"Phone Camera ({MODE.upper()} MODE)"
)

print(
    "=" * 68
)

print()


try:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:

            print(
                "\n[ERROR] "
                "Lost phone camera stream."
            )

            break

        frame_count += 1

        current_time = (
            frame_count / fps
        )

        # ----------------------------------------------------
        # FRAME SKIPPING
        # ----------------------------------------------------

        if (
            frame_count
            % FRAME_SKIP
            != 0
        ):

            continue

        # ----------------------------------------------------
        # YOLO INFERENCE
        # ----------------------------------------------------

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

                confidence = float(
                    box.conf[0]
                )

                # Extra safety check.
                if (
                    confidence
                    < CONFIDENCE
                ):
                    continue

                class_id = int(
                    box.cls[0]
                )

                label = model.names[
                    class_id
                ]

                if (
                    label
                    not in ALLOWED_OBJECTS
                ):
                    continue

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                box_width = (
                    x2 - x1
                )

                box_height = (
                    y2 - y1
                )

                center_x = (
                    x1 + x2
                ) / 2

                center_y = (
                    y1 + y2
                ) / 2

                detections.append({

                    "label":
                        label,

                    "confidence":
                        confidence,

                    "x":
                        center_x,

                    "y":
                        center_y,

                    "x1":
                        x1,

                    "y1":
                        y1,

                    "x2":
                        x2,

                    "y2":
                        y2,

                    "box_w":
                        box_width,

                    "box_h":
                        box_height
                })

        # ----------------------------------------------------
        # UPDATE TRACKS
        # ----------------------------------------------------

        update_tracks(
            detections,
            current_time
        )

        # ----------------------------------------------------
        # ONLY USE RECENT + CONFIRMED OBJECTS
        # ----------------------------------------------------

        visible = [

            track

            for track
            in tracks.values()

            if (
                current_time
                - track["last_seen"]
                <= 0.45

                and track["frames_seen"]
                >= MIN_CONFIRMATIONS
            )
        ]

        # ----------------------------------------------------
        # SITTING MODE
        # ----------------------------------------------------

        if MODE == "sitting":

            if not visible:

                continue

            # Largest/closest useful object first.
            visible.sort(

                key=lambda track: (

                    track["box_h"],

                    track["confidence"]

                ),

                reverse=True
            )

            lead = visible[0]

            signature = (

                "SITTING_"

                + lead["label"]

                + "_"

                + lead["zone"]
            )

            cooldown_expired = (

                current_time
                - last_alert_time
                >= ALERT_COOLDOWN
            )

            if (
                cooldown_expired
                and signature
                != last_alert_signature
            ):

                timestamp = (
                    format_time(
                        current_time
                    )
                )

                message = (
                    create_sitting_message(
                        lead
                    )
                )

                print(
                    f"[{timestamp}] "
                    f"[ALERT] "
                    f"{message}"
                )

                last_alert_time = (
                    current_time
                )

                last_alert_signature = (
                    signature
                )

                timeline_records.append({

                    "time":
                        timestamp,

                    "message":
                        "[ALERT] "
                        + message

                })

            continue

        # ----------------------------------------------------
        # WALKING MODE
        # ----------------------------------------------------

        hazards = [

            track

            for track
            in visible

            if is_walking_hazard(
                track
            )
        ]

        # ----------------------------------------------------
        # PATH CLEAR
        # ----------------------------------------------------

        if not hazards:

            if path_blocked:

                timestamp = (
                    format_time(
                        current_time
                    )
                )

                message = (
                    "[INFO] "
                    "Path is clear."
                )

                print(
                    f"[{timestamp}] "
                    f"{message}"
                )

                timeline_records.append({

                    "time":
                        timestamp,

                    "message":
                        message

                })

                path_blocked = False

            continue

        # We have at least one hazard.
        path_blocked = True

        # ----------------------------------------------------
        # PRIORITIZE
        # ----------------------------------------------------

        hazards.sort(

            key=lambda track: (

                track["zone"]
                == "center",

                track["label"]
                == "person",

                track["dist_m"]
                is not None,

                -(track["dist_m"] or 999),

                track["box_h"]

            ),

            reverse=True
        )

        lead = hazards[0]

        # ----------------------------------------------------
        # EMERGENCY
        # ----------------------------------------------------

        emergency = is_emergency(
            lead
        )

        emergency_allowed = (

            current_time
            - last_emergency_time
            >= EMERGENCY_COOLDOWN
        )

        normal_cooldown_expired = (

            current_time
            - last_alert_time
            >= ALERT_COOLDOWN
        )

        # ----------------------------------------------------
        # ALERT SIGNATURE
        # ----------------------------------------------------

        movement = get_movement(
            lead
        )

        signature = (

            "WALKING_"

            + lead["label"]

            + "_"

            + lead["zone"]

            + "_"

            + str(lead["dist_m"])

            + "_"

            + movement
        )

        should_alert = False

        # ----------------------------------------------------
        # EMERGENCY ALERT
        # ----------------------------------------------------

        if (
            emergency
            and emergency_allowed
        ):

            should_alert = True

            last_emergency_time = (
                current_time
            )

        # ----------------------------------------------------
        # NORMAL ALERT
        # ----------------------------------------------------

        elif (
            normal_cooldown_expired
            and signature
            != last_alert_signature
        ):

            should_alert = True

        # ----------------------------------------------------
        # PRINT ALERT
        # ----------------------------------------------------

        if should_alert:

            timestamp = (
                format_time(
                    current_time
                )
            )

            message = (
                create_walking_message(
                    lead
                )
            )

            if emergency:

                prefix = "[EMERGENCY]"

            else:

                prefix = "[ALERT]"

            print(
                f"[{timestamp}] "
                f"{prefix} "
                f"{message}"
            )

            last_alert_time = (
                current_time
            )

            last_alert_signature = (
                signature
            )

            timeline_records.append({

                "time":
                    timestamp,

                "message":
                    prefix
                    + " "
                    + message

            })


# ============================================================
# STOP WITH CTRL+C
# ============================================================

except KeyboardInterrupt:

    print(
        "\n\n[SYSTEM] "
        "Navigation stopped by user."
    )


# ============================================================
# CLEANUP
# ============================================================

finally:

    cap.release()


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 68
)

print(
    "          HIGH-PRIORITY ASSISTIVE SUMMARY"
)

print(
    "=" * 68
)

if timeline_records:

    for record in timeline_records:

        print(
            f"[{record['time']}] "
            f"{record['message']}"
        )

else:

    print(
        "No alerts recorded."
    )

print(
    "=" * 68
)
