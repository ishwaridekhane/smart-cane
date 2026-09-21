
import cv2
import sys
from collections import deque
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolov8n.pt"

video_path = sys.argv[1] if len(sys.argv) > 1 else "street_walk.mp4"

IMG_SIZE = 480
CONFIDENCE = 0.45

# Process every 2nd frame to reduce Raspberry Pi CPU load
FRAME_SKIP = 2

# Keep approximately 0.75 seconds of recent detections
SMOOTHING_SECONDS = 0.75

# Minimum time between normal alerts
ALERT_COOLDOWN = 3.0

# Directional hysteresis
HYSTERESIS_RATIO = 0.05


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
# LOAD MODEL AND VIDEO
# ============================================================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open '{video_path}'.")
    sys.exit(1)


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)

if not fps or fps <= 0:
    fps = 30.0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


# ============================================================
# DIRECTION BOUNDARIES
# ============================================================

b1 = frame_w * 0.35
b2 = frame_w * 0.65

hysteresis_margin = frame_w * HYSTERESIS_RATIO


# ============================================================
# TEMPORAL SMOOTHING BUFFER
# ============================================================

processed_fps = max(fps / FRAME_SKIP, 1)

buffer_size = max(
    3,
    int(SMOOTHING_SECONDS * processed_fps)
)

detection_buffer = deque(maxlen=buffer_size)


# ============================================================
# ALERT STATE
# ============================================================

frame_count = 0

last_alert_message = ""
last_alert_time = -10.0

timeline_records = []

previous_positions = {}

emergency_active = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_time(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def pluralize(label, count):

    if count == 1:
        return f"1 {label.capitalize()}"

    if label == "person":
        return f"{count} People"

    return f"{count} {label.capitalize()}s"


def get_zone_with_hysteresis(label, x_center):

    previous = previous_positions.get(label)

    # First detection of this object type
    if previous is None:

        if x_center < b1:
            zone = "left"

        elif x_center > b2:
            zone = "right"

        else:
            zone = "center"

        previous_positions[label] = {
            "x": x_center,
            "zone": zone
        }

        return zone


    previous_zone = previous["zone"]


    # --------------------------------------------------------
    # Previously LEFT
    # --------------------------------------------------------

    if previous_zone == "left":

        if x_center < b1 + hysteresis_margin:
            zone = "left"

        elif x_center > b2:
            zone = "right"

        else:
            zone = "center"


    # --------------------------------------------------------
    # Previously CENTER
    # --------------------------------------------------------

    elif previous_zone == "center":

        if x_center < b1 - hysteresis_margin:
            zone = "left"

        elif x_center > b2 + hysteresis_margin:
            zone = "right"

        else:
            zone = "center"


    # --------------------------------------------------------
    # Previously RIGHT
    # --------------------------------------------------------

    else:

        if x_center > b2 - hysteresis_margin:
            zone = "right"

        elif x_center < b1:
            zone = "left"

        else:
            zone = "center"


    previous_positions[label] = {
        "x": x_center,
        "zone": zone
    }

    return zone


def build_smoothed_scene(buffer):

    if not buffer:
        return {}

    combined = {}

    for scene in buffer:

        for key, count in scene.items():

            if key not in combined:
                combined[key] = []

            combined[key].append(count)


    smoothed_scene = {}

    for key, values in combined.items():

        sorted_values = sorted(values)

        middle = len(sorted_values) // 2

        if len(sorted_values) % 2 == 1:
            median_count = sorted_values[middle]

        else:
            median_count = (
                sorted_values[middle - 1]
                + sorted_values[middle]
            ) / 2


        median_count = int(round(median_count))

        if median_count > 0:
            smoothed_scene[key] = median_count


    return smoothed_scene


def choose_top_hazard(scene_counts):

    if not scene_counts:
        return None

    sorted_hazards = sorted(
        scene_counts.items(),
        key=lambda item: (
            item[0][2] == "NEAR",
            item[0][1] == "center",
            item[1]
        ),
        reverse=True
    )

    (label, zone, proximity), count = sorted_hazards[0]

    return label, zone, proximity, count


def is_emergency(label, zone, proximity):

    # NEAR + CENTER = immediate hazard
    return proximity == "NEAR" and zone == "center"


# ============================================================
# START
# ============================================================

print()
print(f"[ASSISTIVE CANE] Active on '{video_path}'")

print(
    f"[SYSTEM] Temporal smoothing: "
    f"{SMOOTHING_SECONDS:.2f}s | "
    f"Cooldown: {ALERT_COOLDOWN:.1f}s | "
    f"Frame skip: {FRAME_SKIP}"
)

print()


# ============================================================
# MAIN LOOP
# ============================================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break


    frame_count += 1

    current_sec = frame_count / fps


    # --------------------------------------------------------
    # Process every 2nd frame
    # --------------------------------------------------------

    if frame_count % FRAME_SKIP != 0:
        continue


    # ========================================================
    # YOLO DETECTION
    # ========================================================

    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )

    boxes = results[0].boxes


    # ========================================================
    # CURRENT FRAME DETECTIONS
    # ========================================================

    scene_counts = {}


    if boxes is not None and len(boxes) > 0:

        for box in boxes:

            label = model.names[int(box.cls[0])]


            if label not in ALLOWED_OBSTACLES:
                continue


            x1, y1, x2, y2 = box.xyxy[0].tolist()


            x_center = (x1 + x2) / 2.0

            box_h = y2 - y1


            # ------------------------------------------------
            # Estimate proximity
            # ------------------------------------------------

            if (box_h / frame_h) > 0.35:
                proximity = "NEAR"
            else:
                proximity = "FAR"


            # ------------------------------------------------
            # Direction with hysteresis
            # ------------------------------------------------

            zone = get_zone_with_hysteresis(
                label,
                x_center
            )


            key = (
                label,
                zone,
                proximity
            )


            scene_counts[key] = (
                scene_counts.get(key, 0) + 1
            )


    # ========================================================
    # ADD FRAME TO SMOOTHING BUFFER
    # ========================================================

    detection_buffer.append(scene_counts)


    # ========================================================
    # SMOOTH DETECTIONS
    # ========================================================

    smoothed_scene = build_smoothed_scene(
        detection_buffer
    )


    # ========================================================
    # SELECT IMPORTANT HAZARD
    # ========================================================

    top_hazard = choose_top_hazard(
        smoothed_scene
    )


    if top_hazard is None:
        continue


    (
        top_label,
        top_zone,
        top_proximity,
        count
    ) = top_hazard


    phrase_count = pluralize(
        top_label,
        count
    )


    # ========================================================
    # CREATE ALERT MESSAGE
    # ========================================================

    if top_proximity == "NEAR":

        current_msg = (
            f"[WARNING] Close hazard: "
            f"{phrase_count} on {top_zone}!"
        )

    else:

        current_msg = (
            f"[ALERT] Ahead: "
            f"{phrase_count} on {top_zone}."
        )


    # ========================================================
    # CHECK EMERGENCY
    # ========================================================

    current_emergency = is_emergency(
        top_label,
        top_zone,
        top_proximity
    )


    # Only trigger emergency when it changes
    # from FALSE -> TRUE
    new_emergency = (
        current_emergency
        and not emergency_active
    )


    emergency_active = current_emergency


    # ========================================================
    # GLOBAL COOLDOWN
    # ========================================================

    cooldown_expired = (
        current_sec - last_alert_time
        >= ALERT_COOLDOWN
    )


    # ========================================================
    # DECIDE WHETHER TO ALERT
    # ========================================================

    should_alert = False


    # New immediate emergency
    if new_emergency:

        should_alert = True


    # Normal alert:
    # BOTH conditions must be true.
    elif (
        current_msg != last_alert_message
        and cooldown_expired
    ):

        should_alert = True


    # ========================================================
    # PRINT ALERT
    # ========================================================

    if should_alert:

        timestamp = format_time(current_sec)

        print(
            f"[{timestamp}] {current_msg}"
        )


        last_alert_message = current_msg

        last_alert_time = current_sec


        timeline_records.append({
            "hazard": phrase_count,
            "zone": top_zone.upper(),
            "time": timestamp,
            "type": (
                "EMERGENCY"
                if new_emergency
                else "ALERT"
            )
        })


# ============================================================
# CLEANUP
# ============================================================

cap.release()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("              ASSISTIVE OBSTACLE TIMELINE")
print("=" * 60)


if timeline_records:

    print(
        f"{'Time':<8} | "
        f"{'Obstacle':<20} | "
        f"{'Direction':<10} | "
        f"{'Type'}"
    )

    print("-" * 60)


    for record in timeline_records:

        print(
            f"{record['time']:<8} | "
            f"{record['hazard']:<20} | "
            f"{record['zone']:<10} | "
            f"{record['type']}"
        )


else:

    print("No alerts generated.")


print("=" * 60)
print()
```
