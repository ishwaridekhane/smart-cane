import cv2
import time
import math
import os
import sys
from collections import defaultdict, deque

from ultralytics import YOLO

# ============================================================
# SAHAYAK DRISHTI AI - PHONE NAVIGATION
# Camera: Android IP Webcam
# Detection: YOLOv8n
# GenAI: Gemini
# ============================================================


# -----------------------------
# CONFIGURATION
# -----------------------------

MODEL_PATH = "yolov8n.pt"

# Your phone IP Webcam URL
DEFAULT_VIDEO_URL = "http://10.108.217.59:8080/video"

# YOLO settings
IMG_SIZE = 320
CONFIDENCE = 0.50

# Process every Nth frame
FRAME_SKIP = 4

# Gemini
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_COOLDOWN = 5.0

# Distance calibration
CALIBRATION_DISTANCE = 1.0       # meters
CALIBRATION_SAMPLES = 10

# Fallback focal length if calibration fails
DEFAULT_FOCAL_LENGTH = 500.0

# Assumed real-world heights for rough distance estimation
# Person distance is handled separately.
OBJECT_HEIGHTS = {
    "person": 1.70,
    "bicycle": 1.10,
    "car": 1.50,
    "motorcycle": 1.10,
    "bus": 3.00,
    "truck": 3.00,
    "dog": 0.60,
    "chair": 0.90,
    "bench": 0.50,
    "backpack": 0.45,
    "suitcase": 0.70,
    "dining table": 0.75,
    "laptop": 0.25,
    "cell phone": 0.15,
    "bottle": 0.25,
    "cup": 0.12,
    "book": 0.25,
    "keyboard": 0.05,
}

# Objects important for navigation
NAVIGATION_OBJECTS = {
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
# GEMINI SETUP
# ============================================================

gemini_client = None

try:
    from google import genai

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("[GEMINI] API key found.")
        print(f"[GEMINI] Model: {GEMINI_MODEL}")
    else:
        print("[GEMINI] GEMINI_API_KEY not found.")
        print("[GEMINI] Running without Gemini.")

except Exception as e:
    print("[GEMINI] SDK not available.")
    print("[GEMINI]", e)


last_gemini_time = 0.0


# ============================================================
# DISTANCE CALIBRATION
# ============================================================

def calibrate_camera(model, cap):
    """
    Ask the user to stand approximately 1 meter away.
    YOLO detects the person's bounding-box height.
    We calculate an approximate focal length.
    """

    print()
    print("=" * 60)
    print("DISTANCE CALIBRATION")
    print("=" * 60)
    print()
    print("Stand about 1 meter in front of the phone.")
    print("Try to show your FULL BODY.")
    print("Keep the phone facing you.")
    print()
    print("Press ENTER when ready.")
    input()

    print()
    print("Calibrating...")
    print("Please stay still for a few seconds.")
    print()

    samples = []

    start_time = time.time()

    while len(samples) < CALIBRATION_SAMPLES:
        ret, frame = cap.read()

        if not ret:
            continue

        small = cv2.resize(frame, (640, 360))

        results = model(
            small,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            verbose=False
        )

        best_height = 0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence < CONFIDENCE:
                    continue

                name = model.names[cls_id]

                if name != "person":
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                height = y2 - y1

                if height > best_height:
                    best_height = height

        if best_height > 30:
            samples.append(best_height)

            print(
                f"\rSamples: {len(samples)}/{CALIBRATION_SAMPLES}",
                end=""
            )

        if time.time() - start_time > 30:
            break

    print()
    print()

    if len(samples) >= 5:

        average_height = sum(samples) / len(samples)

        # F = pixel_height * real_distance / real_height
        focal_length = (
            average_height
            * CALIBRATION_DISTANCE
            / OBJECT_HEIGHTS["person"]
        )

        print(
            f"[CALIBRATION] Average person box height: "
            f"{average_height:.1f}px"
        )

        print(
            f"[CALIBRATION] Estimated focal length: "
            f"{focal_length:.1f}"
        )

        print("[CALIBRATION] Successful.")
        print()

        return focal_length

    print("[CALIBRATION] Could not detect person enough times.")
    print(
        f"[CALIBRATION] Using fallback focal length: "
        f"{DEFAULT_FOCAL_LENGTH}"
    )
    print()

    return DEFAULT_FOCAL_LENGTH


# ============================================================
# DISTANCE
# ============================================================

def estimate_distance(box_height, object_name, focal_length):
    """
    Monocular approximate distance.

    distance = real_height * focal_length / pixel_height
    """

    if box_height <= 1:
        return None

    real_height = OBJECT_HEIGHTS.get(
        object_name,
        0.70
    )

    distance = (
        real_height
        * focal_length
        / box_height
    )

    return max(0.2, min(distance, 20.0))


def natural_distance(distance, object_name, box_height, frame_height):
    """
    Convert numerical distance into natural walking language.
    """

    # Person is extremely close.
    if object_name == "person":

        if (
            distance is not None
            and distance < 0.65
        ):
            return "very close"

        if (
            box_height / frame_height
            > 0.72
        ):
            return "very close"

    if distance is None:
        return "nearby"

    if distance < 0.8:
        return "very close"

    if distance < 1.3:
        return "about 1 meter"

    if distance < 1.8:
        return "about 1.5 meters"

    if distance < 2.5:
        return "about 2 meters"

    if distance < 3.5:
        return "about 3 meters"

    if distance < 4.5:
        return "about 4 meters"

    if distance < 6.0:
        return "about 5 meters"

    if distance < 8.0:
        return "about 7 meters"

    return "far ahead"


# ============================================================
# POSITION
# ============================================================

def get_position(center_x, frame_width):
    ratio = center_x / frame_width

    if ratio < 0.33:
        return "left"

    if ratio > 0.66:
        return "right"

    return "ahead"


# ============================================================
# MOVEMENT TRACKING
# ============================================================

tracks = defaultdict(
    lambda: deque(maxlen=5)
)


def movement_direction(object_name, center_x, center_y):
    """
    Very simple movement estimation based on
    recent center positions.

    This is not full object tracking.
    """

    history = tracks[object_name]

    history.append(
        (
            center_x,
            center_y,
            time.time()
        )
    )

    if len(history) < 3:
        return "moving"

    old_x, old_y, old_t = history[0]
    new_x, new_y, new_t = history[-1]

    dx = new_x - old_x
    dy = new_y - old_y

    # Ignore tiny movements.
    if abs(dx) < 8 and abs(dy) < 8:
        return "stationary"

    if abs(dx) > abs(dy):

        if dx > 0:
            return "moving right"

        return "moving left"

    if dy > 0:
        return "approaching"

    return "moving away"


# ============================================================
# GEMINI CONTEXTUAL REASONING
# ============================================================

def ask_gemini(scene_description, walking_mode):
    global last_gemini_time

    if gemini_client is None:
        return None

    now = time.time()

    # Don't call Gemini constantly.
    if now - last_gemini_time < GEMINI_COOLDOWN:
        return None

    last_gemini_time = now

    mode_text = "walking" if walking_mode else "sitting/testing"

    prompt = f"""
You are the contextual AI assistant for an accessibility smart cane.

The fast local YOLO detector has already detected these objects:

{scene_description}

User mode:
{mode_text}

Give ONE very short, natural voice instruction.

Rules:
- Never invent an object.
- Never invent a direction.
- Use only the detected information.
- Keep it under 12 words.
- Say "ahead", "left", or "right".
- Use natural walking language.
- For danger, be direct.
- Do not mention YOLO, AI, bounding boxes, pixels, confidence,
  calibration, or computer vision.
- Do not use centimeters.
"""

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        text = response.text.strip()

        if text:
            return text

    except Exception as e:
        print(f"[GEMINI ERROR] {e}")

    return None


# ============================================================
# LOCAL SAFETY DECISION
# ============================================================

def local_navigation_message(
    detections,
    walking_mode
):

    if not detections:
        return None, False

    # Sort nearest first.
    detections = sorted(
        detections,
        key=lambda x: (
            x["distance"]
            if x["distance"] is not None
            else 99
        )
    )

    # --------------------------------------------------------
    # PEOPLE
    # --------------------------------------------------------

    people = [
        d for d in detections
        if d["name"] == "person"
    ]

    center_people = [
        p for p in people
        if p["position"] == "ahead"
    ]

    # Multiple people ahead
    if len(center_people) >= 2:

        nearest = center_people[0]

        distance_text = natural_distance(
            nearest["distance"],
            "person",
            nearest["box_height"],
            nearest["frame_height"]
        )

        if walking_mode:

            return (
                f"{len(center_people)} people ahead, "
                f"{distance_text}. Be careful.",
                True
            )

        return (
            f"{len(center_people)} people ahead, "
            f"{distance_text}.",
            True
        )

    # One person ahead
    if center_people:

        person = center_people[0]

        distance_text = natural_distance(
            person["distance"],
            "person",
            person["box_height"],
            person["frame_height"]
        )

        movement = person["movement"]

        # Very close
        if distance_text == "very close":

            return (
                "Person very close ahead. Be careful.",
                True
            )

        # Approaching
        if movement == "approaching":

            return (
                f"Person ahead, {distance_text}, "
                f"moving toward you. Be careful.",
                True
            )

        return (
            f"Person ahead, {distance_text}.",
            False
        )

    # --------------------------------------------------------
    # OTHER CENTER OBJECTS
    # --------------------------------------------------------

    center_objects = [
        d for d in detections
        if d["position"] == "ahead"
        and d["name"] != "person"
    ]

    if center_objects:

        obj = center_objects[0]

        distance_text = natural_distance(
            obj["distance"],
            obj["name"],
            obj["box_height"],
            obj["frame_height"]
        )

        if walking_mode:

            return (
                f"{obj['name'].capitalize()} ahead, "
                f"{distance_text}. Be careful.",
                True
            )

        return (
            f"{obj['name'].capitalize()} ahead, "
            f"{distance_text}.",
            True
        )

    # --------------------------------------------------------
    # CLOSE SIDE OBJECT
    # --------------------------------------------------------

    for obj in detections:

        if obj["position"] not in ("left", "right"):
            continue

        if obj["distance"] is None:
            continue

        if obj["distance"] < 2.5:

            distance_text = natural_distance(
                obj["distance"],
                obj["name"],
                obj["box_height"],
                obj["frame_height"]
            )

            return (
                f"{obj['name'].capitalize()} on your "
                f"{obj['position']}, {distance_text}.",
                False
            )

    return None, False


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # VIDEO URL
    # --------------------------------------------------------

    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    else:
        video_url = DEFAULT_VIDEO_URL

    print()
    print("=" * 60)
    print("SAHAYAK DRISHTI AI")
    print("PHONE CAMERA NAVIGATION")
    print("=" * 60)
    print()
    print(f"Camera: {video_url}")
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

    walking_mode = mode == "w"

    print()

    if walking_mode:
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
    # OPEN PHONE CAMERA
    # --------------------------------------------------------

    print("[CAMERA] Connecting to phone...")

    cap = cv2.VideoCapture(video_url)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():

        print()
        print("[ERROR] Could not open phone camera.")
        print()
        print("Check:")
        print("1. IP Webcam is running.")
        print("2. Phone and Raspberry Pi are on same Wi-Fi.")
        print("3. IP address is correct.")
        print()
        return

    print("[CAMERA] Connected.")
    print()

    # --------------------------------------------------------
    # TEST FIRST FRAME
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("[ERROR] Camera opened but no frame received.")
        cap.release()
        return

    print(
        f"[CAMERA] Frame size: "
        f"{frame.shape[1]}x{frame.shape[0]}"
    )

    print()

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    focal_length = calibrate_camera(
        model,
        cap
    )

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    frame_count = 0

    last_message = ""
    last_message_time = 0

    MESSAGE_COOLDOWN = 2.0

    path_blocked = False

    fps_start = time.time()
    fps_frames = 0

    print("=" * 60)
    print("NAVIGATION STARTED")
    print("=" * 60)
    print()
    print("Press Q to stop.")
    print()

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:

            print("[CAMERA] Frame lost. Reconnecting...")

            time.sleep(0.5)

            cap.release()

            cap = cv2.VideoCapture(video_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            continue

        frame_count += 1

        # ----------------------------------------------------
        # FRAME SKIP
        # ----------------------------------------------------

        if frame_count % FRAME_SKIP != 0:
            continue

        frame_height, frame_width = frame.shape[:2]

        # Resize for faster YOLO inference.
        display_frame = cv2.resize(
            frame,
            (640, 360)
        )

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model(
            display_frame,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            verbose=False
        )

        detections = []

        # ----------------------------------------------------
        # PROCESS DETECTIONS
        # ----------------------------------------------------

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0])

                if confidence < CONFIDENCE:
                    continue

                cls_id = int(box.cls[0])

                name = model.names[cls_id]

                # Only navigation-relevant objects
                if name not in NAVIGATION_OBJECTS:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                box_width = x2 - x1
                box_height = y2 - y1

                if box_height <= 5:
                    continue

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                position = get_position(
                    center_x,
                    640
                )

                distance = estimate_distance(
                    box_height,
                    name,
                    focal_length
                )

                movement = movement_direction(
                    name,
                    center_x,
                    center_y
                )

                detections.append({
                    "name": name,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "box_height": box_height,
                    "center_x": center_x,
                    "center_y": center_y,
                    "position": position,
                    "distance": distance,
                    "movement": movement,
                    "frame_height": 360,
                })

        # ----------------------------------------------------
        # PATH STATE
        # ----------------------------------------------------

        blocking_objects = []

        for d in detections:

            if d["position"] == "ahead":

                # Person/object is considered blocking
                if (
                    d["distance"] is not None
                    and d["distance"] < 3.0
                ):
                    blocking_objects.append(d)

        currently_blocked = len(blocking_objects) > 0

        # Only announce CLEAR when transitioning
        # from blocked -> clear.
        if path_blocked and not currently_blocked:

            now = time.time()

            if now - last_message_time > MESSAGE_COOLDOWN:

                print(
                    f"[{time.strftime('%M:%S')}] "
                    "[INFO] Path is clear."
                )

                last_message = "Path is clear."
                last_message_time = now

        path_blocked = currently_blocked

        # ----------------------------------------------------
        # LOCAL SAFETY MESSAGE
        # ----------------------------------------------------

        local_message, is_hazard = local_navigation_message(
            detections,
            walking_mode
        )

        # ----------------------------------------------------
        # GEMINI SCENE DESCRIPTION
        # ----------------------------------------------------

        gemini_message = None

        if detections:

            scene_lines = []

            for d in detections[:5]:

                distance_text = natural_distance(
                    d["distance"],
                    d["name"],
                    d["box_height"],
                    d["frame_height"]
                )

                scene_lines.append(
                    f"{d['name']} "
                    f"{d['position']} "
                    f"{distance_text}, "
                    f"{d['movement']}"
                )

            scene_description = "\n".join(
                scene_lines
            )

            # Gemini is contextual only.
            # Local emergency warning has priority.
            if not is_hazard:

                gemini_message = ask_gemini(
                    scene_description,
                    walking_mode
                )

        # ----------------------------------------------------
        # FINAL MESSAGE
        # ----------------------------------------------------

        message = None

        # Local safety logic gets priority.
        if local_message:

            message = local_message

        # Gemini can improve contextual wording
        # when local system doesn't have a stronger warning.
        elif gemini_message:

            message = gemini_message

        # ----------------------------------------------------
        # PRINT MESSAGE WITH COOLDOWN
        # ----------------------------------------------------

        if message:

            now = time.time()

            # Don't repeat identical message constantly.
            if (
                message != last_message
                or now - last_message_time > MESSAGE_COOLDOWN
            ):

                tag = "[WARNING]" if is_hazard else "[INFO]"

                print(
                    f"[{time.strftime('%M:%S')}] "
                    f"{tag} {message}"
                )

                last_message = message
                last_message_time = now

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        for d in detections:

            x1 = int(d["x1"])
            y1 = int(d["y1"])
            x2 = int(d["x2"])
            y2 = int(d["y2"])

            label = (
                f"{d['name']} "
                f"{d['position']}"
            )

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # Mode
        mode_text = (
            "WALKING"
            if walking_mode
            else "SITTING / TEST"
        )

        cv2.putText(
            display_frame,
            f"Mode: {mode_text}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Gemini status
        gemini_status = (
            "Gemini: ON"
            if gemini_client
            else "Gemini: OFF"
        )

        cv2.putText(
            display_frame,
            gemini_status,
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Sahayak Drishti - Phone Navigation",
            display_frame
        )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        fps_frames += 1

        elapsed = time.time() - fps_start

        if elapsed >= 5:

            fps = fps_frames / elapsed

            print(
                f"[SYSTEM] Processing FPS: {fps:.1f}"
            )

            fps_start = time.time()
            fps_frames = 0

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("NAVIGATION STOPPED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
