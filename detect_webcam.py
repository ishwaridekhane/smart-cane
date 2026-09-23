import cv2
import os
import sys
import math
import time
from collections import deque
from ultralytics import YOLO

# ============================================================
# SAHAYAK DRISHTI - ASSISTIVE WALKING VISION ENGINE
#
# Main goals:
#   - Stay quiet when nothing important is happening.
#   - Say "path is clear" only once when the path becomes clear.
#   - Focus mainly on the walking path.
#   - Group people instead of announcing every person separately.
#   - Detect objects moving into the user's path.
#   - Detect objects crossing from left/right.
#   - Avoid repeated irritating announcements.
#   - Keep local YOLO logic independent from Gemini.
#
# Compatible with:
#   Windows development
#   Raspberry Pi 4 deployment
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolov8n.pt"

# Webcam by default.
# Example:
#   python detect_webcam.py
#
# Video:
#   python detect_webcam.py street_walk.mp4
#
video_source = sys.argv[1] if len(sys.argv) > 1 else 0


# ------------------------------------------------------------
# YOLO PERFORMANCE
# ------------------------------------------------------------

IMG_SIZE = 384

CONFIDENCE = 0.45

# Raspberry Pi 4:
# Start with 3.
#
# If the Pi is very fast, try 2.
# If the Pi struggles badly, try 4.
FRAME_SKIP = 3


# ------------------------------------------------------------
# TRACKING
# ------------------------------------------------------------

MAX_TRACK_DISTANCE = 120

MAX_TRACK_AGE = 1.2

MIN_CONFIRMATIONS = 3

MOVEMENT_HISTORY = 8


# ------------------------------------------------------------
# ANNOUNCEMENT TIMING
# ------------------------------------------------------------

# Normal messages should not repeat quickly.
NORMAL_REPEAT_COOLDOWN = 4.0

# Very close obstacle can interrupt normal silence.
EMERGENCY_COOLDOWN = 1.5

# Gemini should not be called constantly.
GEMINI_COOLDOWN = 5.0


# ------------------------------------------------------------
# WALKING CORRIDOR
# ------------------------------------------------------------

LEFT_BOUNDARY = 0.33

RIGHT_BOUNDARY = 0.67

# Prevent an object from jumping between zones
# because of tiny detection changes.
HYSTERESIS_RATIO = 0.04


# ------------------------------------------------------------
# OBJECT SIZE / DISTANCE
# ------------------------------------------------------------

PATH_RELEVANCE_HEIGHT = 0.12

NEAR_RATIO = 0.32

CRITICAL_RATIO = 0.45


# ------------------------------------------------------------
# YOLO COCO OBJECTS WE USE
# ------------------------------------------------------------

ALLOWED_OBSTACLES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "chair",
    "dining table"
}


# ------------------------------------------------------------
# APPROXIMATE OBJECT WIDTHS
#
# Used only for rough distance estimation.
# They are NOT accurate measurements.
# ------------------------------------------------------------

AVERAGE_WIDTHS = {
    "person": 0.45,
    "bicycle": 0.60,
    "motorcycle": 0.80,
    "car": 1.80,
    "bus": 2.50,
    "truck": 2.50,
    "dog": 0.40,
    "chair": 0.50,
    "dining table": 0.90
}

APPROX_FOCAL_LENGTH_PIXELS = 700.0


# ============================================================
# GEMINI SETUP
# ============================================================

gemini_client = None

api_key = os.getenv("GEMINI_API_KEY")

if api_key:

    try:

        from google import genai

        gemini_client = genai.Client(
            api_key=api_key
        )

        print("\n[SYSTEM] Gemini GenAI: ENABLED\n")

    except Exception as e:

        print(
            f"\n[SYSTEM] Gemini setup warning: {e}\n"
        )

else:

    print(
        "\n[SYSTEM] Gemini GenAI: DISABLED "
        "(GEMINI_API_KEY not set)\n"
    )


# ============================================================
# INITIALIZATION
# ============================================================

try:

    model = YOLO(MODEL_PATH)

except Exception as e:

    print(
        f"\nERROR: Could not load YOLO model: {e}"
    )

    sys.exit(1)


cap = cv2.VideoCapture(video_source)

if not cap.isOpened():

    print(
        f"\nERROR: Cannot open camera/video: "
        f"{video_source}"
    )

    sys.exit(1)


# ------------------------------------------------------------
# CAMERA / VIDEO INFORMATION
# ------------------------------------------------------------

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:

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

    frame_w = 640


if frame_h <= 0:

    frame_h = 480


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


# ============================================================
# GLOBAL STATE
# ============================================================

tracks = {}

next_track_id = 1

last_announcement_time = -999.0

last_emergency_time = -999.0

last_gemini_time = -999.0

# ------------------------------------------------------------
# PATH STATE
# ------------------------------------------------------------

path_was_clear = True

clear_announced = False


# ------------------------------------------------------------
# OBJECT ANNOUNCEMENT MEMORY
# ------------------------------------------------------------

announced_objects = {}


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

timeline_records = []

frame_count = 0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_time(seconds):

    minutes = int(
        seconds // 60
    )

    secs = int(
        seconds % 60
    )

    return f"{minutes:02d}:{secs:02d}"


def distance_pixels(
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
# ROUGH DISTANCE ESTIMATION
# ============================================================

def estimate_distance_meters(
    label,
    box_w
):

    if label not in AVERAGE_WIDTHS:

        return None

    if box_w <= 5:

        return None

    raw_distance = (
        AVERAGE_WIDTHS[label]
        * APPROX_FOCAL_LENGTH_PIXELS
    ) / box_w

    raw_distance = max(
        0.5,
        min(
            25.0,
            raw_distance
        )
    )

    if raw_distance < 1.5:

        return 1

    elif raw_distance < 3.5:

        return 3

    elif raw_distance < 6.0:

        return 5

    elif raw_distance < 9.0:

        return 8

    else:

        return round(
            raw_distance / 5.0
        ) * 5


# ============================================================
# ZONE DETECTION
# ============================================================

def get_zone(
    x,
    previous_zone=None
):

    left_boundary = (
        frame_w * LEFT_BOUNDARY
    )

    right_boundary = (
        frame_w * RIGHT_BOUNDARY
    )

    hysteresis_margin = (
        frame_w * HYSTERESIS_RATIO
    )


    # --------------------------------------------------------
    # Previously LEFT
    # --------------------------------------------------------

    if previous_zone == "left":

        if x < (
            left_boundary
            + hysteresis_margin
        ):

            return "left"

        if x > right_boundary:

            return "right"

        return "center"


    # --------------------------------------------------------
    # Previously RIGHT
    # --------------------------------------------------------

    if previous_zone == "right":

        if x > (
            right_boundary
            - hysteresis_margin
        ):

            return "right"

        if x < left_boundary:

            return "left"

        return "center"


    # --------------------------------------------------------
    # Previously CENTER
    # --------------------------------------------------------

    if previous_zone == "center":

        if x < (
            left_boundary
            - hysteresis_margin
        ):

            return "left"

        if x > (
            right_boundary
            + hysteresis_margin
        ):

            return "right"

        return "center"


    # --------------------------------------------------------
    # NEW OBJECT
    # --------------------------------------------------------

    if x < left_boundary:

        return "left"

    if x > right_boundary:

        return "right"

    return "center"


# ============================================================
# DISTANCE LEVEL
# ============================================================

def get_distance_level(track):

    height_ratio = (
        track["box_h"]
        / frame_h
    )

    if height_ratio >= CRITICAL_RATIO:

        return "critical"

    if height_ratio >= NEAR_RATIO:

        return "near"

    if height_ratio >= 0.20:

        return "medium"

    return "far"


# ============================================================
# MOVEMENT ANALYSIS
# ============================================================

def get_movement(track):

    history = track["history"]

    if len(history) < 4:

        return "stable"


    old_x, old_height = history[0]

    new_x, new_height = history[-1]


    horizontal_threshold = (
        frame_w * 0.04
    )

    approaching_threshold = 0.07


    dx = new_x - old_x

    dh = new_height - old_height


    moving_right = (
        dx > horizontal_threshold
    )

    moving_left = (
        dx < -horizontal_threshold
    )

    approaching = (
        dh > approaching_threshold
    )


    if approaching:

        if moving_right:

            return "approaching from the left"

        if moving_left:

            return "approaching from the right"

        return "approaching"


    if moving_right:

        return "moving toward the right"


    if moving_left:

        return "moving toward the left"


    return "stable"


# ============================================================
# CROSSING DIRECTION
# ============================================================

def get_crossing_direction(track):

    history = track["history"]

    if len(history) < 5:

        return None


    old_x = history[0][0]

    new_x = history[-1][0]


    movement = (
        new_x - old_x
    )

    threshold = (
        frame_w * 0.08
    )


    if movement > threshold:

        return "left to right"


    if movement < -threshold:

        return "right to left"


    return None


# ============================================================
# NATURAL OBJECT PHRASES
# ============================================================

def object_phrase(
    label,
    count=1
):

    if label == "person":

        if count == 1:

            return "one person"

        if count == 2:

            return "two people"

        if count == 3:

            return "three people"

        return (
            f"a group of {count} people"
        )


    if count == 1:

        return f"one {label}"


    return f"{count} {label}s"


# ============================================================
# TRACKING
# ============================================================

def match_detections(
    detections,
    current_sec
):

    global next_track_id

    matched_tracks = set()


    for detection in detections:

        best_id = None

        best_distance = float("inf")


        # ----------------------------------------------------
        # Find closest existing track
        # ----------------------------------------------------

        for track_id, track in tracks.items():

            if track_id in matched_tracks:

                continue


            if track["label"] != detection["label"]:

                continue


            if (
                current_sec
                - track["last_seen"]
                > MAX_TRACK_AGE
            ):

                continue


            d = distance_pixels(

                detection["x"],
                detection["y"],

                track["x"],
                track["y"]
            )


            if d < best_distance:

                best_distance = d

                best_id = track_id


        distance_m = estimate_distance_meters(

            detection["label"],

            detection["box_w"]
        )


        # ----------------------------------------------------
        # Update existing track
        # ----------------------------------------------------

        if (
            best_id is not None
            and best_distance
            <= MAX_TRACK_DISTANCE
        ):

            track = tracks[best_id]


            track["x"] = (
                detection["x"]
            )

            track["y"] = (
                detection["y"]
            )


            track["box_w"] = (
                detection["box_w"]
            )

            track["box_h"] = (
                detection["box_h"]
            )


            track["x1"] = (
                detection["x"]
                - detection["box_w"] / 2
            )

            track["x2"] = (
                detection["x"]
                + detection["box_w"] / 2
            )


            track["dist_m"] = (
                distance_m
            )


            track["last_seen"] = (
                current_sec
            )


            track["frames_seen"] += 1


            track["history"].append(

                (
                    detection["x"],

                    detection["box_h"]
                    / frame_h
                )
            )


            track["zone"] = get_zone(

                detection["x"],

                track["zone"]
            )


            matched_tracks.add(
                best_id
            )


        # ----------------------------------------------------
        # Create new track
        # ----------------------------------------------------

        else:

            track_id = next_track_id

            next_track_id += 1


            initial_zone = get_zone(

                detection["x"]
            )


            tracks[track_id] = {

                "id": track_id,

                "label": detection["label"],

                "x": detection["x"],

                "y": detection["y"],

                "box_w": detection["box_w"],

                "box_h": detection["box_h"],

                "x1": (
                    detection["x"]
                    - detection["box_w"] / 2
                ),

                "x2": (
                    detection["x"]
                    + detection["box_w"] / 2
                ),

                "dist_m": distance_m,

                "first_seen": current_sec,

                "last_seen": current_sec,

                "frames_seen": 1,

                "history": deque(

                    [
                        (
                            detection["x"],

                            detection["box_h"]
                            / frame_h
                        )
                    ],

                    maxlen=MOVEMENT_HISTORY
                ),

                "zone": initial_zone
            }


            matched_tracks.add(
                track_id
            )


    # --------------------------------------------------------
    # Remove expired tracks
    # --------------------------------------------------------

    expired_tracks = [

        track_id

        for track_id, track
        in tracks.items()

        if (
            current_sec
            - track["last_seen"]
            > MAX_TRACK_AGE
        )
    ]


    for track_id in expired_tracks:

        del tracks[track_id]


# ============================================================
# GET VISIBLE CONFIRMED OBJECTS
# ============================================================

def get_visible_tracks(
    current_sec
):

    visible = []


    for track in tracks.values():

        if (
            current_sec
            - track["last_seen"]
            > 0.5
        ):

            continue


        if (
            track["frames_seen"]
            < MIN_CONFIRMATIONS
        ):

            continue


        height_ratio = (
            track["box_h"]
            / frame_h
        )


        if (
            height_ratio
            < PATH_RELEVANCE_HEIGHT
        ):

            continue


        visible.append(track)


    return visible


# ============================================================
# DOES OBJECT BLOCK WALKING PATH?
#
# Important:
# A chair/table on the side may still stick into the
# walking corridor.
# ============================================================

def object_blocks_path(track):

    zone = track["zone"]

    height_ratio = (
        track["box_h"]
        / frame_h
    )


    # --------------------------------------------------------
    # Anything in center is relevant.
    # --------------------------------------------------------

    if zone == "center":

        return True


    # --------------------------------------------------------
    # Check if side object physically extends into
    # the central walking corridor.
    # --------------------------------------------------------

    left_boundary = (
        frame_w * LEFT_BOUNDARY
    )

    right_boundary = (
        frame_w * RIGHT_BOUNDARY
    )


    x1 = track.get(
        "x1",
        0
    )

    x2 = track.get(
        "x2",
        0
    )


    if (
        x2 > left_boundary
        and x1 < right_boundary
        and height_ratio
        >= PATH_RELEVANCE_HEIGHT
    ):

        return True


    return False


# ============================================================
# FIND OBJECTS THAT MATTER TO WALKING
# ============================================================

def find_path_obstacles(
    visible
):

    return [

        track

        for track in visible

        if object_blocks_path(track)

    ]


# ============================================================
# DETECT OBJECT ENTERING WALKING PATH
# ============================================================

def is_entering_path(track):

    history = track["history"]


    if len(history) < 5:

        return False


    old_x = history[0][0]

    new_x = history[-1][0]


    old_zone = get_zone(
        old_x
    )

    new_zone = track["zone"]


    # Object moved from side into center.
    if (
        old_zone != "center"
        and new_zone == "center"
    ):

        if (
            track["box_h"]
            / frame_h
            >= 0.14
        ):

            return True


    return False


# ============================================================
# CROSSING MESSAGE
# ============================================================

def crossing_message(track):

    direction = get_crossing_direction(
        track
    )


    label = track["label"]


    if direction == "left to right":

        return (
            f"{object_phrase(label)} "
            f"is crossing from your left to right"
        )


    if direction == "right to left":

        return (
            f"{object_phrase(label)} "
            f"is crossing from your right to left"
        )


    return (
        f"{object_phrase(label)} "
        f"is moving across your path"
    )


# ============================================================
# BUILD LOCAL ALERT
#
# This is the important navigation logic.
# Gemini does NOT control this logic.
# ============================================================

def build_alert(
    obstacles
):

    if not obstacles:

        return None


    # --------------------------------------------------------
    # 1. OBJECT ENTERING PATH
    # --------------------------------------------------------

    entering = [

        track

        for track in obstacles

        if is_entering_path(track)

    ]


    if entering:

        entering.sort(

            key=lambda x:
            x["box_h"],

            reverse=True
        )


        lead = entering[0]


        return (
            crossing_message(lead),
            lead
        )


    # --------------------------------------------------------
    # 2. VERY CLOSE CENTER OBJECT
    # --------------------------------------------------------

    critical = [

        track

        for track in obstacles

        if (
            track["zone"]
            == "center"

            and get_distance_level(track)
            == "critical"
        )
    ]


    if critical:

        critical.sort(

            key=lambda x:
            x["box_h"],

            reverse=True
        )


        lead = critical[0]


        return (

            f"Careful, "
            f"{object_phrase(lead['label'])} "
            f"is very close ahead of you",

            lead
        )


    # --------------------------------------------------------
    # 3. GROUP OF PEOPLE
    # --------------------------------------------------------

    people = [

        track

        for track in obstacles

        if (
            track["label"]
            == "person"

            and track["zone"]
            == "center"
        )
    ]


    if len(people) >= 3:

        lead = max(

            people,

            key=lambda x:
            x["box_h"]
        )


        return (

            f"A group of "
            f"{len(people)} people "
            f"is ahead of you",

            lead
        )


    # --------------------------------------------------------
    # 4. SINGLE / TWO PEOPLE
    # --------------------------------------------------------

    if people:

        people.sort(

            key=lambda x:
            x["box_h"],

            reverse=True
        )


        lead = people[0]


        if len(people) == 2:

            return (
                "Two people are ahead of you",
                lead
            )


        return (
            "One person is ahead of you",
            lead
        )


    # --------------------------------------------------------
    # 5. OTHER CENTER OBJECT
    # --------------------------------------------------------

    center_objects = [

        track

        for track in obstacles

        if track["zone"] == "center"
    ]


    if center_objects:

        center_objects.sort(

            key=lambda x:
            x["box_h"],

            reverse=True
        )


        lead = center_objects[0]


        label = lead["label"]


        level = get_distance_level(
            lead
        )


        if level == "near":

            return (

                f"{object_phrase(label)} "
                f"is ahead of you",

                lead
            )


        return (

            f"There is "
            f"{object_phrase(label)} "
            f"ahead of you",

            lead
        )


    # --------------------------------------------------------
    # 6. SIDE OBJECT ONLY IF CLOSE
    #
    # We DO NOT constantly announce side objects.
    # --------------------------------------------------------

    side_objects = [

        track

        for track in obstacles

        if (
            track["zone"] != "center"

            and get_distance_level(track)
            in ["near", "critical"]
        )
    ]


    if side_objects:

        side_objects.sort(

            key=lambda x:
            x["box_h"],

            reverse=True
        )


        lead = side_objects[0]


        return (

            f"{object_phrase(lead['label'])} "
            f"is close on your "
            f"{lead['zone']}",

            lead
        )


    return None


# ============================================================
# SHOULD THIS OBJECT BE ANNOUNCED AGAIN?
# ============================================================

def should_repeat(track):

    object_id = track["id"]


    previous = announced_objects.get(
        object_id
    )


    # New object.
    if previous is None:

        return True


    previous_zone = previous[
        "zone"
    ]

    previous_level = previous[
        "level"
    ]


    current_level = get_distance_level(
        track
    )


    # --------------------------------------------------------
    # Became significantly closer.
    # --------------------------------------------------------

    if (
        previous_level
        != current_level

        and current_level
        in ["near", "critical"]
    ):

        return True


    # --------------------------------------------------------
    # Moved into center path.
    # --------------------------------------------------------

    if (
        previous_zone != "center"

        and track["zone"]
        == "center"
    ):

        return True


    # --------------------------------------------------------
    # Otherwise remain quiet.
    # --------------------------------------------------------

    return False


# ============================================================
# REMEMBER ANNOUNCEMENT
# ============================================================

def remember_announcement(track):

    announced_objects[
        track["id"]
    ] = {

        "zone":
            track["zone"],

        "level":
            get_distance_level(track),

        "time":
            time.time()
    }


# ============================================================
# GEMINI SCENE DESCRIPTION
#
# Gemini is OPTIONAL.
#
# Python decides WHEN something is important.
# Gemini only helps phrase the scene naturally.
# ============================================================

def ask_gemini(
    frame,
    scene_data
):

    global last_gemini_time


    if gemini_client is None:

        return None


    now = time.time()


    if (
        now
        - last_gemini_time
        < GEMINI_COOLDOWN
    ):

        return None


    try:

        from google.genai import types


        small_frame = cv2.resize(

            frame,

            (320, 240)
        )


        success, encoded = cv2.imencode(

            ".jpg",

            small_frame,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                60
            ]
        )


        if not success:

            return None


        prompt = f"""
You are the visual assistant of a smart walking cane
for a blind person.

Your job is NOT to narrate everything visible.

Speak only about what matters for walking.

CONFIRMED LOCAL DETECTION INFORMATION:
{scene_data}

STRICT RULES:

1. Give ONE short, natural spoken sentence.

2. Be calm and useful.

3. Do not repeatedly say left/right.

4. Prioritize objects in the walking path.

5. Prioritize objects entering or crossing the walking path.

6. If several people are together, describe them as a group.

7. If an object is safely away from the walking path,
   do not mention it.

8. If an object is moving toward the user, mention that
   only when the movement is supported by the information.

9. Never invent an object.

10. Never invent stairs.

11. Never invent an elevator or lift.

12. Never invent a door.

13. Never invent a curb.

14. Never invent a pothole.

15. Never invent road conditions.

16. Never invent distance.

17. Never give a navigation instruction unless the
    confirmed information clearly supports it.

18. Do not mention AI.

19. Do not mention YOLO.

20. Do not mention confidence scores.

21. Maximum 18 words.

22. Return ONLY the sentence to speak.

The user should feel as if another person is quietly
describing only the important things around them.
"""


        response = gemini_client.models.generate_content(

            model="gemini-2.5-flash",

            contents=[

                types.Part.from_bytes(

                    data=encoded.tobytes(),

                    mime_type="image/jpeg"
                ),

                prompt
            ],

            config=types.GenerateContentConfig(

                temperature=0.1,

                max_output_tokens=50
            )
        )


        last_gemini_time = now


        if response.text:

            return (

                response.text
                .strip()
                .replace("\n", " ")
            )


    except Exception as e:

        print(
            f"\n[GEMINI WARNING] {e}"
        )


    return None


# ============================================================
# START MESSAGE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "[SAHAYAK DRISHTI] "
    "Assistive Vision Engine"
)

print(
    "[SYSTEM] YOLOv8n active"
)

print(
    "[SYSTEM] Frame skip:",
    FRAME_SKIP
)

print(
    "[SYSTEM] Goal: quiet, human-like walking assistance"
)

print(
    "[SYSTEM] Camera/video:",
    video_source
)

print(
    "=" * 70
    + "\n"
)


# ============================================================
# MAIN LOOP
# ============================================================

while cap.isOpened():

    ret, frame = cap.read()


    if not ret:

        break


    frame_count += 1


    current_sec = (
        frame_count / fps
    )


    # --------------------------------------------------------
    # Skip frames for performance.
    # --------------------------------------------------------

    if (
        frame_count
        % FRAME_SKIP
        != 0
    ):

        continue


    # ========================================================
    # YOLO DETECTION
    # ========================================================

    try:

        results = model(

            frame,

            imgsz=IMG_SIZE,

            conf=CONFIDENCE,

            verbose=False
        )

    except Exception as e:

        print(
            f"\n[YOLO WARNING] {e}"
        )

        continue


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


            if (
                label
                not in ALLOWED_OBSTACLES
            ):

                continue


            x1, y1, x2, y2 = (

                box.xyxy[0]
                .tolist()
            )


            box_w = (
                x2 - x1
            )

            box_h = (
                y2 - y1
            )


            detections.append({

                "label":
                    label,

                "x":
                    (x1 + x2) / 2.0,

                "y":
                    (y1 + y2) / 2.0,

                "box_w":
                    box_w,

                "box_h":
                    box_h
            })


    # ========================================================
    # UPDATE TRACKS
    # ========================================================

    match_detections(

        detections,

        current_sec
    )


    # ========================================================
    # GET CONFIRMED OBJECTS
    # ========================================================

    visible = get_visible_tracks(

        current_sec
    )


    # ========================================================
    # FIND WALKING-PATH OBSTACLES
    # ========================================================

    obstacles = find_path_obstacles(

        visible
    )


    # ========================================================
    # PATH CLEAR LOGIC
    # ========================================================

    path_clear = (
        len(obstacles) == 0
    )


    # --------------------------------------------------------
    # PATH IS CLEAR
    # --------------------------------------------------------

    if path_clear:


        # Only announce when transitioning from
        # obstacle -> clear.
        #
        # Do NOT announce repeatedly.

        if (
            not path_was_clear
            and not clear_announced
        ):

            msg = (
                "The path ahead is clear."
            )


            timestamp = format_time(

                current_sec
            )


            print(
                f"\n[{timestamp}] "
                f"[INFO] {msg}\n"
            )


            timeline_records.append({

                "time":
                    timestamp,

                "message":
                    msg
            })


            clear_announced = True


            last_announcement_time = (
                current_sec
            )


        path_was_clear = True


        continue


    # --------------------------------------------------------
    # PATH IS NOT CLEAR
    # --------------------------------------------------------

    path_was_clear = False

    clear_announced = False


    # ========================================================
    # BUILD ALERT
    # ========================================================

    alert_result = build_alert(

        obstacles
    )


    if alert_result is None:

        continue


    base_message, lead = (
        alert_result
    )


    # ========================================================
    # CURRENT DISTANCE LEVEL
    # ========================================================

    current_level = (
        get_distance_level(
            lead
        )
    )


    # ========================================================
    # EMERGENCY STATE
    # ========================================================

    emergency = (

        lead["zone"]
        == "center"

        and current_level
        == "critical"
    )


    # IMPORTANT:
    # Keep emergency timing based on time.time()
    # consistently.

    now = time.time()


    # ========================================================
    # EMERGENCY COOLDOWN
    # ========================================================

    if emergency:

        if (
            now
            - last_emergency_time
            < EMERGENCY_COOLDOWN
        ):

            continue


    # ========================================================
    # NORMAL COOLDOWN
    # ========================================================

    else:

        if (
            current_sec
            - last_announcement_time
            < NORMAL_REPEAT_COOLDOWN
        ):

            continue


    # ========================================================
    # DO NOT REPEAT THE SAME OBJECT
    # UNLESS ITS STATE CHANGED
    # ========================================================

    if not emergency:

        if not should_repeat(
            lead
        ):

            continue


    # ========================================================
    # GEMINI CONTEXT
    # ========================================================

    scene_data = {

        "main_object":
            lead["label"],

        "location":
            lead["zone"],

        "distance_level":
            current_level,

        "estimated_distance_m":
            lead.get("dist_m"),

        "movement":
            get_movement(lead),

        "crossing_direction":
            get_crossing_direction(
                lead
            ),

        "entering_walking_path":
            is_entering_path(
                lead
            ),

        "number_of_path_objects":
            len(obstacles)
    }


    # ========================================================
    # OPTIONAL GEMINI
    # ========================================================

    gemini_message = ask_gemini(

        frame,

        scene_data
    )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    if gemini_message:

        msg = gemini_message

    else:

        msg = base_message


    # ========================================================
    # PREFIX
    # ========================================================

    if emergency:

        prefix = "[URGENT]"

    else:

        prefix = "[INFO]"


    # ========================================================
    # OUTPUT
    # ========================================================

    timestamp = format_time(

        current_sec
    )


    print(
        f"\n[{timestamp}] "
        f"{prefix} "
        f"{msg}\n"
    )


    # ========================================================
    # UPDATE STATE
    # ========================================================

    last_announcement_time = (
        current_sec
    )


    if emergency:

        last_emergency_time = now


    remember_announcement(
        lead
    )


    timeline_records.append({

        "time":
            timestamp,

        "message":
            msg
    })


# ============================================================
# CLEANUP
# ============================================================

cap.release()


print(
    "\n"
    + "=" * 70
)

print(
    "ASSISTIVE SCENE SUMMARY"
)

print(
    "=" * 70
)


if timeline_records:

    for record in timeline_records:

        print(

            f"[{record['time']}] "
            f"{record['message']}"
        )

else:

    print(
        "No significant obstacles detected."
    )


print(
    "=" * 70
)
