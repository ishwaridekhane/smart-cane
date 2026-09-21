
import cv2
import sys
import math
import time
from collections import deque
from ultralytics import YOLO

# Optional voice output
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolov8n.pt"

video_path = sys.argv[1] if len(sys.argv) > 1 else "street_walk.mp4"

# YOLO
IMG_SIZE = 480

# Lower than before so small/brief objects such as bicycles
# have a better chance of being detected.
CONFIDENCE = 0.30

# Raspberry Pi CPU optimization
FRAME_SKIP = 2

# Global minimum time between normal announcements
ALERT_COOLDOWN = 3.0

# Emergency can interrupt the cooldown
EMERGENCY_COOLDOWN = 1.0

# Simple tracking
MAX_TRACK_DISTANCE = 120
MAX_TRACK_AGE = 1.5

# A detection must normally be seen this many times
# before becoming a stable object.
MIN_CONFIRMATIONS = 2

# Objects that are detected only briefly can still be announced
# if YOLO is reasonably confident.
TRANSIENT_CONFIDENCE = 0.55

# Proximity
NEAR_RATIO = 0.35

# Direction boundaries
LEFT_BOUNDARY = 0.35
RIGHT_BOUNDARY = 0.65

# Dead-band around boundaries
HYSTERESIS_RATIO = 0.05

# Recent movement history
MOVEMENT_HISTORY = 6

# Only describe objects that are relevant to the user.
ALLOWED_OBSTACLES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "chair"
}


# ============================================================
# MODEL
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open '{video_path}'.")
    sys.exit(1)


fps = cap.get(cv2.CAP_PROP_FPS)

if not fps or fps <= 0:
    fps = 30.0

frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


# ============================================================
# BOUNDARIES
# ============================================================

b1 = frame_w * LEFT_BOUNDARY
b2 = frame_w * RIGHT_BOUNDARY

hysteresis_margin = frame_w * HYSTERESIS_RATIO


# ============================================================
# OPTIONAL TEXT TO SPEECH
# ============================================================

speech_engine = None

if TTS_AVAILABLE:

    try:
        speech_engine = pyttsx3.init()

        speech_engine.setProperty("rate", 165)

    except Exception:
        speech_engine = None


def speak(message):
    """
    Speak the assistant message if pyttsx3 is available.
    Console output always happens.
    """

    print(message)

    if speech_engine is not None:

        try:
            speech_engine.say(message)
            speech_engine.runAndWait()

        except Exception:
            pass


# ============================================================
# TRACK STORAGE
# ============================================================

tracks = {}

next_track_id = 1


# ============================================================
# ALERT STATE
# ============================================================

last_alert_time = -10.0
last_alert_message = ""

last_emergency_time = -10.0

timeline_records = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_time(seconds):

    return (
        f"{int(seconds // 60):02d}:"
        f"{int(seconds % 60):02d}"
    )


def distance(x1, y1, x2, y2):

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def get_zone(track):

    x = track["x"]

    previous_zone = track["zone"]


    # --------------------------------------------------------
    # First state
    # --------------------------------------------------------

    if previous_zone is None:

        if x < b1:
            return "left"

        if x > b2:
            return "right"

        return "center"


    # --------------------------------------------------------
    # Hysteresis
    # --------------------------------------------------------

    if previous_zone == "left":

        if x < b1 + hysteresis_margin:
            return "left"

        if x > b2:
            return "right"

        return "center"


    if previous_zone == "center":

        if x < b1 - hysteresis_margin:
            return "left"

        if x > b2 + hysteresis_margin:
            return "right"

        return "center"


    # Previous = right

    if x > b2 - hysteresis_margin:
        return "right"

    if x < b1:
        return "left"

    return "center"


def get_movement(track):

    history = track["history"]

    if len(history) < 3:
        return "moving straight ahead"


    old_x, old_h = history[0]
    new_x, new_h = history[-1]


    dx = new_x - old_x
    dh = new_h - old_h


    # Horizontal movement
    horizontal_threshold = frame_w * 0.025


    if dx > horizontal_threshold:

        horizontal = "toward your right"

    elif dx < -horizontal_threshold:

        horizontal = "toward your left"

    else:

        horizontal = None


    # Object getting larger = approaching
    if dh > 0.08:

        depth = "approaching"

    elif dh < -0.08:

        depth = "moving away"

    else:

        depth = None


    if horizontal and depth:

        return f"{depth}, moving {horizontal}"

    if horizontal:

        return f"moving {horizontal}"

    if depth:

        return depth

    return "moving straight ahead"


def get_proximity(box_height):

    ratio = box_height / frame_h

    if ratio > NEAR_RATIO:
        return "near"

    return "far"


def object_name(label, count):

    if label == "person":

        if count == 1:
            return "person"

        return "people"

    if count == 1:
        return label

    return label + "s"


def describe_count(label, count):

    if label == "person":

        if count == 1:
            return "one person"

        if count == 2:
            return "two people"

        return f"{count} people"


    if count == 1:
        return f"one {label}"

    return f"{count} {label}s"


def build_object_description(track_group):

    """
    Create natural language describing a group of similar objects.
    """

    if not track_group:
        return ""


    label = track_group[0]["label"]

    count = len(track_group)

    count_text = describe_count(label, count)


    # Use the most important/nearest track
    best_track = sorted(
        track_group,
        key=lambda t: (
            t["proximity"] == "near",
            t["box_height"]
        ),
        reverse=True
    )[0]


    zone = best_track["zone"]

    movement = get_movement(best_track)


    # --------------------------------------------------------
    # Location wording
    # --------------------------------------------------------

    if zone == "center":

        location = "ahead of you"

    elif zone == "left":

        location = "on your left"

    else:

        location = "on your right"


    # --------------------------------------------------------
    # Proximity
    # --------------------------------------------------------

    if best_track["proximity"] == "near":

        if zone == "center":

            return (
                f"Be careful. {count_text} "
                f"is very close ahead of you, "
                f"{movement}."
            )

        return (
            f"Be careful. {count_text} "
            f"is close {location}, "
            f"{movement}."
        )


    # --------------------------------------------------------
    # Normal scene description
    # --------------------------------------------------------

    if zone == "center":

        return (
            f"There {'is' if count == 1 else 'are'} "
            f"{count_text} {location}, "
            f"{movement}."
        )


    return (
        f"There {'is' if count == 1 else 'are'} "
        f"{count_text} {location}, "
        f"{movement}."
    )


# ============================================================
# TRACK MATCHING
# ============================================================

def match_detections(detections, current_sec):

    global next_track_id

    matched_track_ids = set()


    # --------------------------------------------------------
    # Match each detection to nearest existing track
    # --------------------------------------------------------

    for detection in detections:

        label = detection["label"]

        x = detection["x"]
        y = detection["y"]


        best_id = None
        best_distance = float("inf")


        for track_id, track in tracks.items():

            if track_id in matched_track_ids:
                continue

            if track["label"] != label:
                continue


            age = current_sec - track["last_seen"]

            if age > MAX_TRACK_AGE:
                continue


            d = distance(
                x,
                y,
                track["x"],
                track["y"]
            )


            if d < best_distance:

                best_distance = d
                best_id = track_id


        # ----------------------------------------------------
        # Existing object
        # ----------------------------------------------------

        if (
            best_id is not None
            and best_distance <= MAX_TRACK_DISTANCE
        ):

            track = tracks[best_id]


            track["x"] = x
            track["y"] = y

            track["box_height"] = detection["box_height"]

            track["confidence"] = detection["confidence"]

            track["last_seen"] = current_sec

            track["frames_seen"] += 1


            track["history"].append(
                (
                    x,
                    detection["box_height"]
                )
            )


            track["zone"] = get_zone(track)

            track["proximity"] = get_proximity(
                detection["box_height"]
            )


            matched_track_ids.add(best_id)


        # ----------------------------------------------------
        # New object
        # ----------------------------------------------------

        else:

            track_id = next_track_id

            next_track_id += 1


            new_track = {

                "id": track_id,

                "label": label,

                "x": x,

                "y": y,

                "box_height":
                    detection["box_height"],

                "confidence":
                    detection["confidence"],

                "first_seen":
                    current_sec,

                "last_seen":
                    current_sec,

                "frames_seen": 1,

                "history":
                    deque(
                        [
                            (
                                x,
                                detection["box_height"]
                            )
                        ],
                        maxlen=MOVEMENT_HISTORY
                    ),

                "zone": None,

                "proximity":
                    get_proximity(
                        detection["box_height"]
                    ),

                "announced": False
            }


            new_track["zone"] = get_zone(
                new_track
            )


            tracks[track_id] = new_track

            matched_track_ids.add(track_id)


    # --------------------------------------------------------
    # Delete old tracks
    # --------------------------------------------------------

    expired_ids = []

    for track_id, track in tracks.items():

        if (
            current_sec - track["last_seen"]
            > MAX_TRACK_AGE
        ):

            expired_ids.append(track_id)


    for track_id in expired_ids:

        del tracks[track_id]


# ============================================================
# FIND IMPORTANT NEW OBJECTS
# ============================================================

def find_new_relevant_tracks(current_sec):

    new_objects = []


    for track in tracks.values():

        age = current_sec - track["last_seen"]

        if age > 0.3:
            continue


        # Stable detection
        confirmed = (
            track["frames_seen"]
            >= MIN_CONFIRMATIONS
        )


        # Brief but high-confidence detection
        transient = (
            track["confidence"]
            >= TRANSIENT_CONFIDENCE
        )


        if not track["announced"] and (
            confirmed or transient
        ):

            new_objects.append(track)


    return new_objects


# ============================================================
# FIND CURRENT SCENE
# ============================================================

def get_current_scene(current_sec):

    visible = []


    for track in tracks.values():

        age = current_sec - track["last_seen"]


        if age <= 0.5:

            visible.append(track)


    return visible


# ============================================================
# CREATE ASSISTANT MESSAGE
# ============================================================

def create_assistant_message(new_objects, visible_objects):

    # --------------------------------------------------------
    # NEW OBJECT HAS PRIORITY
    # --------------------------------------------------------

    if new_objects:

        # Group new objects by label
        grouped = {}

        for track in new_objects:

            grouped.setdefault(
                track["label"],
                []
            ).append(track)


        descriptions = []

        for group in grouped.values():

            descriptions.append(
                build_object_description(group)
            )


        if len(descriptions) == 1:

            return descriptions[0]


        return " ".join(descriptions)


    # --------------------------------------------------------
    # Otherwise describe the important current scene
    # --------------------------------------------------------

    if not visible_objects:

        return None


    # Find nearest object
    important = sorted(
        visible_objects,
        key=lambda t: (
            t["proximity"] == "near",
            t["zone"] == "center",
            t["box_height"]
        ),
        reverse=True
    )


    best = important[0]


    # Only create a description if movement is meaningful
    return build_object_description(
        [best]
    )


# ============================================================
# START
# ============================================================

print()
print("=" * 60)
print("          SAHAYAK DRISHTI - ASSISTIVE VISION")
print("=" * 60)

print(f"Video: {video_path}")
print(f"YOLO confidence: {CONFIDENCE}")
print(f"Frame skip: {FRAME_SKIP}")
print(f"Alert cooldown: {ALERT_COOLDOWN}s")

if speech_engine is not None:
    print("Voice assistant: ENABLED")
else:
    print("Voice assistant: console only")

print("=" * 60)
print()


# ============================================================
# MAIN LOOP
# ============================================================

frame_count = 0


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break


    frame_count += 1

    current_sec = frame_count / fps


    # --------------------------------------------------------
    # Frame skipping
    # --------------------------------------------------------

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


    if boxes is not None and len(boxes) > 0:

        for box in boxes:

            label = model.names[
                int(box.cls[0])
            ]


            if label not in ALLOWED_OBSTACLES:
                continue


            confidence = float(
                box.conf[0]
            )


            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )


            x_center = (
                x1 + x2
            ) / 2.0


            y_center = (
                y1 + y2
            ) / 2.0


            box_height = y2 - y1


            detections.append({

                "label": label,

                "x": x_center,

                "y": y_center,

                "box_height":
                    box_height,

                "confidence":
                    confidence
            })


    # ========================================================
    # UPDATE TRACKS
    # ========================================================

    match_detections(
        detections,
        current_sec
    )


    # ========================================================
    # GET SCENE
    # ========================================================

    visible_objects = get_current_scene(
        current_sec
    )


    # ========================================================
    # NEW OBJECTS
    # ========================================================

    new_objects = find_new_relevant_tracks(
        current_sec
    )


    # ========================================================
    # EMERGENCY CHECK
    # ========================================================

    emergency_objects = [

        track

        for track in visible_objects

        if (
            track["zone"] == "center"
            and track["proximity"] == "near"
        )
    ]


    emergency_now = len(
        emergency_objects
    ) > 0


    emergency_allowed = (
        current_sec - last_emergency_time
        >= EMERGENCY_COOLDOWN
    )


    # ========================================================
    # BUILD MESSAGE
    # ========================================================

    message = create_assistant_message(
        new_objects,
        visible_objects
    )


    # ========================================================
    # ALERT DECISION
    # ========================================================

    normal_cooldown_expired = (
        current_sec - last_alert_time
        >= ALERT_COOLDOWN
    )


    should_alert = False


    # --------------------------------------------------------
    # Emergency
    # --------------------------------------------------------

    if (
        emergency_now
        and emergency_allowed
    ):

        should_alert = True

        last_emergency_time = current_sec


    # --------------------------------------------------------
    # Normal new object / scene change
    # --------------------------------------------------------

    elif (
        message is not None
        and normal_cooldown_expired
        and message != last_alert_message
    ):

        should_alert = True


    # ========================================================
    # ANNOUNCE
    # ========================================================

    if should_alert:

        timestamp = format_time(
            current_sec
        )


        full_message = (
            f"[{timestamp}] "
            f"{message}"
        )


        speak(full_message)


        last_alert_time = current_sec

        last_alert_message = message


        # Mark new objects as announced
        for track in new_objects:

            track["announced"] = True


        timeline_records.append({

            "time": timestamp,

            "message": message,

            "emergency":
                emergency_now

        })


# ============================================================
# CLEANUP
# ============================================================

cap.release()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("                 ASSISTIVE SCENE TIMELINE")
print("=" * 70)


if timeline_records:

    for record in timeline_records:

        emergency_text = (
            " [EMERGENCY]"
            if record["emergency"]
            else ""
        )


        print(
            f"[{record['time']}]"
            f"{emergency_text} "
            f"{record['message']}"
        )


else:

    print("No alerts generated.")


print("=" * 70)
print()

