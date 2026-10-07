import cv2
import sys
import math
import time
from collections import deque
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolov8n.pt"

# Phone IP Webcam URL
PHONE_URL = sys.argv[1] if len(sys.argv) > 1 else None

IMG_SIZE = 384
CONFIDENCE = 0.45
FRAME_SKIP = 3

ALERT_COOLDOWN = 3.0
EMERGENCY_COOLDOWN = 1.5

MAX_TRACK_DISTANCE = 110
MAX_TRACK_AGE = 0.8
MIN_CONFIRMATIONS = 2

# Movement detection
MOVEMENT_HISTORY = 8
MIN_HORIZONTAL_MOVEMENT = 35
MIN_VERTICAL_MOVEMENT = 25
MIN_DISTANCE_CHANGE = 0.18

# Walking corridor
LEFT_BOUNDARY = 0.32
RIGHT_BOUNDARY = 0.68
HYSTERESIS_RATIO = 0.04

# ============================================================
# YOLO OBJECTS
# ============================================================
# These are useful COCO objects for navigation/environment
# awareness. Standard yolov8n.pt cannot detect doors/headphones.

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

# ============================================================
# APPROXIMATE OBJECT HEIGHTS
# ============================================================

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
# USER MODE
# ============================================================

print("=" * 68)
print("              SAHAYAK DRISHTI")
print("           PHONE NAVIGATION MODE")
print("=" * 68)

print("\nAre you sitting or walking?")
print("  s = Sitting")
print("  w = Walking")

while True:

    user_mode = input("\nEnter s or w: ").strip().lower()

    if user_mode in ["s", "w"]:
        break

    print("[ERROR] Please enter only s or w.")

if user_mode == "s":
    MODE = "sitting"
    print("\n[SYSTEM] Sitting mode selected.")
else:
    MODE = "walking"
    print("\n[SYSTEM] Walking mode selected.")


# ============================================================
# INITIALIZATION
# ============================================================

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

hysteresis_margin = frame_w * HYSTERESIS_RATIO


# ============================================================
# TRACKING VARIABLES
# ============================================================

tracks = {}
next_track_id = 1

last_alert_time = -10.0
last_emergency_time = -10.0
last_alert_signature = ""

path_is_clear = False

timeline_records = []

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


def get_zone(x, previous_zone=None):

    if previous_zone == "left":

        return (
            "left"
            if x < (b1 + hysteresis_margin)
            else (
                "right"
                if x > b2
                else "center"
            )
        )

    elif previous_zone == "center":

        if x < (b1 - hysteresis_margin):
            return "left"

        if x > (b2 + hysteresis_margin):
            return "right"

        return "center"

    elif previous_zone == "right":

        return (
            "right"
            if x > (b2 - hysteresis_margin)
            else (
                "left"
                if x < b1
                else "center"
            )
        )

    return (
        "left"
        if x < b1
        else (
            "right"
            if x > b2
            else "center"
        )
    )


# ============================================================
# TRACKING
# ============================================================

def match_detections(detections, current_sec):

    global next_track_id

    matched = set()

    for det in detections:

        best_id = None
        min_d = float("inf")

        for t_id, trk in tracks.items():

            if t_id in matched:
                continue

            if trk["label"] != det["label"]:
                continue

            if (
                current_sec
                - trk["last_seen"]
                > MAX_TRACK_AGE
            ):
                continue

            d = distance(
                det["x"],
                det["y"],
                trk["x"],
                trk["y"]
            )

            if d < min_d:

                min_d = d
                best_id = t_id

        dist_m = estimate_distance_meters(
            det["label"],
            det["box_h"]
        )

        x1 = (
            det["x"]
            - det["box_w"] / 2
        )

        x2 = (
            det["x"]
            + det["box_w"] / 2
        )

        # ----------------------------------------------------
        # EXISTING TRACK
        # ----------------------------------------------------

        if (
            best_id is not None
            and min_d <= MAX_TRACK_DISTANCE
        ):

            trk = tracks[best_id]

            trk["x"] = det["x"]
            trk["y"] = det["y"]

            trk["x1"] = x1
            trk["x2"] = x2

            trk["box_h"] = det["box_h"]
            trk["box_w"] = det["box_w"]

            trk["dist_m"] = dist_m

            trk["last_seen"] = current_sec

            trk["frames_seen"] += 1

            trk["zone"] = get_zone(
                det["x"],
                trk["zone"]
            )

            trk["history"].append({

                "x": det["x"],
                "y": det["y"],
                "box_h": det["box_h"],
                "dist": dist_m,
                "time": current_sec

            })

            matched.add(best_id)

        # ----------------------------------------------------
        # NEW TRACK
        # ----------------------------------------------------

        else:

            t_id = next_track_id
            next_track_id += 1

            initial_zone = get_zone(
                det["x"]
            )

            tracks[t_id] = {

                "id": t_id,

                "label": det["label"],

                "x": det["x"],
                "y": det["y"],

                "x1": x1,
                "x2": x2,

                "box_h": det["box_h"],
                "box_w": det["box_w"],

                "dist_m": dist_m,

                "last_seen": current_sec,

                "frames_seen": 1,

                "zone": initial_zone,

                "history": deque(
                    [{
                        "x": det["x"],
                        "y": det["y"],
                        "box_h": det["box_h"],
                        "dist": dist_m,
                        "time": current_sec
                    }],
                    maxlen=MOVEMENT_HISTORY
                )
            }

            matched.add(t_id)

    # --------------------------------------------------------
    # REMOVE OLD TRACKS
    # --------------------------------------------------------

    for t_id in [

        k

        for k, v in tracks.items()

        if (
            current_sec
            - v["last_seen"]
            > MAX_TRACK_AGE
        )

    ]:

        del tracks[t_id]


# ============================================================
# MOVEMENT ANALYSIS
# ============================================================

def get_movement(trk):

    history = trk["history"]

    if len(history) < 4:
        return "stationary"

    first = history[0]
    last = history[-1]

    dx = last["x"] - first["x"]
    dy = last["y"] - first["y"]

    # Horizontal movement

    if abs(dx) >= MIN_HORIZONTAL_MOVEMENT:

        if dx > 0:
            return "left_to_right"

        return "right_to_left"

    # Toward / away

    if (
        first["dist"] is not None
        and last["dist"] is not None
    ):

        distance_change = (
            first["dist"]
            - last["dist"]
        )

        if distance_change >= MIN_DISTANCE_CHANGE:

            if (
                abs(dy)
                >= MIN_VERTICAL_MOVEMENT

                or abs(
                    last["box_h"]
                    - first["box_h"]
                ) >= 10
            ):

                return "toward"

        elif distance_change <= -MIN_DISTANCE_CHANGE:

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


def movement_phrase(trk):

    movement = get_movement(trk)

    if movement == "right_to_left":
        return "moving right to left across your path"

    if movement == "left_to_right":
        return "moving left to right across your path"

    if movement == "toward":
        return "moving toward you"

    if movement == "away":
        return "moving away from you"

    return None


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
# SITTING MODE MESSAGE
# ============================================================

def create_sitting_message(trk):

    name = object_name(
        trk["label"]
    )

    zone = trk["zone"]

    dist_m = trk["dist_m"]

    if zone == "center":
        location = "ahead"
    else:
        location = f"on your {zone}"

    if dist_m is not None:

        return (
            f"{name.capitalize()} "
            f"{location}, about "
            f"{dist_m} meters."
        )

    return (
        f"{name.capitalize()} "
        f"{location}."
    )


# ============================================================
# WALKING MODE MESSAGE
# ============================================================

def create_walking_message(trk):

    name = object_name(
        trk["label"]
    )

    zone = trk["zone"]

    dist_m = trk["dist_m"]

    movement_text = movement_phrase(
        trk
    )

    if dist_m is not None:

        dist_text = (
            f"about {dist_m} meters"
        )

    else:

        dist_text = "nearby"

    # --------------------------------------------------------
    # CENTER
    # --------------------------------------------------------

    if zone == "center":

        if movement_text:

            return (
                f"{name.capitalize()} ahead, "
                f"{dist_text}, "
                f"{movement_text}. "
                f"Be careful."
            )

        if (
            dist_m is not None
            and dist_m <= 1
        ):

            return (
                f"Careful, {name} ahead, "
                f"about 1 meter."
            )

        return (
            f"{name.capitalize()} ahead, "
            f"{dist_text}. "
            f"Stay alert."
        )

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    if zone == "left":

        if movement_text:

            return (
                f"{name.capitalize()} "
                f"on your left, "
                f"{dist_text}, "
                f"{movement_text}."
            )

        return (
            f"{name.capitalize()} "
            f"on your left, "
            f"{dist_text}. "
            f"Stay slightly right."
        )

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    if movement_text:

        return (
            f"{name.capitalize()} "
            f"on your right, "
            f"{dist_text}, "
            f"{movement_text}."
        )

    return (
        f"{name.capitalize()} "
        f"on your right, "
        f"{dist_text}. "
        f"Stay slightly left."
    )


# ============================================================
# DISPLAY HEADER
# ============================================================

print("=" * 68)

print(
    " [SAHAYAK DRISHTI] High-Priority Safety Engine: "
    f"Phone Camera ({MODE.upper()} MODE)"
)

print("=" * 68 + "\n")


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

    current_sec = (
        frame_count / fps
    )

    if frame_count % FRAME_SKIP != 0:
        continue

    # ========================================================
    # DISPLAY PROGRESS
    # ========================================================

    sys.stdout.write(
        f"\rAnalyzing live phone camera..."
    )

    sys.stdout.flush()

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

            if label not in ALLOWED_OBJECTS:
                continue

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            detections.append({

                "label": label,

                "x": (
                    x1 + x2
                ) / 2.0,

                "y": (
                    y1 + y2
                ) / 2.0,

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

        t

        for t in tracks.values()

        if (
            current_sec
            - t["last_seen"]
            <= 0.4

            and t["frames_seen"]
            >= MIN_CONFIRMATIONS
        )
    ]

    if not visible:
        continue

    # ========================================================
    # SITTING MODE
    # ========================================================

    if MODE == "sitting":

        # Nearest/largest useful object first

        visible.sort(

            key=lambda t: (

                t["dist_m"] is not None,

                -(t["dist_m"] or 999),

                t["box_h"]

            ),

            reverse=True
        )

        lead = visible[0]

        dist_val = lead["dist_m"]

        signature = (
            f"sitting_"
            f"{lead['label']}_"
            f"{lead['zone']}_"
            f"{dist_val}"
        )

        cooldown_expired = (
            current_sec
            - last_alert_time
            >= ALERT_COOLDOWN
        )

        if (
            cooldown_expired
            and signature
            != last_alert_signature
        ):

            timestamp = format_time(
                current_sec
            )

            message = (
                create_sitting_message(
                    lead
                )
            )

            print(
                f"\r{' ' * 100}\r"
                f"[{timestamp}] "
                f"[ALERT] {message}\n"
            )

            last_alert_time = current_sec

            last_alert_signature = signature

            timeline_records.append({

                "time": timestamp,

                "message":
                    f"[ALERT] {message}"

            })

        continue

    # ========================================================
    # WALKING MODE
    # ========================================================

    # Center objects are higher priority

    visible.sort(

        key=lambda t: (

            t["zone"] == "center",

            t["dist_m"] is not None,

            -(t["dist_m"] or 999),

            t["box_h"]

        ),

        reverse=True
    )

    lead = visible[0]

    # ========================================================
    # WALKING HAZARDS
    # ========================================================

    hazards = [

        t

        for t in visible

        if t["zone"] == "center"

        or (
            t["label"]
            in ["chair", "dining table", "bench", "couch"]

            and t["x2"] > b1
            and t["x1"] < b2
        )
    ]

    # ========================================================
    # PATH CLEAR
    # ========================================================

    if not hazards:

        if not path_is_clear:

            timestamp = format_time(
                current_sec
            )

            msg = (
                "[INFO] Path is clear."
            )

            sys.stdout.write(
                f"\r{' ' * 100}\r\n"
                f"[{timestamp}] {msg}\n\n"
            )

            sys.stdout.flush()

            timeline_records.append({

                "time": timestamp,

                "message": msg

            })

            path_is_clear = True

        continue

    else:

        path_is_clear = False

    # ========================================================
    # PRIORITIZE HAZARDS
    # ========================================================

    hazards.sort(

        key=lambda t: (

            t["zone"] == "center",

            t["dist_m"] is not None,

            -(t["dist_m"] or 999),

            t["box_h"]

        ),

        reverse=True
    )

    lead = hazards[0]

    # ========================================================
    # DISTANCE
    # ========================================================

    dist_val = lead["dist_m"]

    # ========================================================
    # MOVEMENT
    # ========================================================

    movement = get_movement(
        lead
    )

    movement_text = movement_phrase(
        lead
    )

    # ========================================================
    # EMERGENCY
    # ========================================================

    is_emergency = (

        lead["zone"] == "center"

        and (

            (
                lead["box_h"]
                / frame_h
            ) >= 0.38

            or (

                dist_val is not None

                and dist_val <= 1

            )
        )
    )

    # Approaching object becomes stronger warning

    if (

        lead["zone"] == "center"

        and movement == "toward"

        and dist_val is not None

        and dist_val <= 3
    ):

        is_emergency = True

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
    # SIGNATURE
    # ========================================================

    signature = (

        f"walking_"
        f"{lead['label']}_"
        f"{lead['zone']}_"
        f"{dist_val}_"
        f"{movement}"
    )

    should_announce = False

    # ========================================================
    # EMERGENCY
    # ========================================================

    if (

        is_emergency

        and emergency_allowed
    ):

        should_announce = True

        last_emergency_time = (
            current_sec
        )

    # ========================================================
    # NORMAL
    # ========================================================

    elif (

        cooldown_expired

        and signature
        != last_alert_signature
    ):

        should_announce = True

    # ========================================================
    # OUTPUT
    # ========================================================

    if should_announce:

        timestamp = format_time(
            current_sec
        )

        message = (
            create_walking_message(
                lead
            )
        )

        if is_emergency:

            prefix = "[EMERGENCY]"

        else:

            prefix = "[ALERT]"

        sys.stdout.write(
            f"\r{' ' * 100}\r\n"
            f"[{timestamp}] "
            f"{prefix} {message}\n\n"
        )

        sys.stdout.flush()

        last_alert_time = (
            current_sec
        )

        last_alert_signature = (
            signature
        )

        timeline_records.append({

            "time": timestamp,

            "message":
                f"{prefix} {message}"

        })


# ============================================================
# CLEANUP
# ============================================================

cap.release()

print(
    "\n\n"
    + "=" * 68
)

print(
    "              HIGH-PRIORITY ASSISTIVE SUMMARY"
)

print(
    "=" * 68
    + "\n"
)

if timeline_records:

    for rec in timeline_records:

        print(
            f" [{rec['time']}] "
            f"{rec['message']}\n"
        )

else:

    print(
        " No navigation alerts recorded.\n"
    )

print(
    "=" * 68
    + "\n"
)
