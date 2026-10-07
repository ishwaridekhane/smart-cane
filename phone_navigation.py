import cv2
import time
import os
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

# Gemini
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_COOLDOWN = 5


# Objects useful for navigation
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
# GEMINI
# ============================================================

gemini_client = None
last_gemini_call = 0

try:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        gemini_client = genai.Client(api_key=api_key)
        print("[GEMINI] API key found.")
        print(f"[GEMINI] Model: {GEMINI_MODEL}")
    else:
        print("[GEMINI] API key not found. Gemini disabled.")

except Exception as e:
    print("[GEMINI] Could not initialize:", e)


# ============================================================
# TRACKING
# ============================================================

tracks = defaultdict(lambda: deque(maxlen=6))


def get_movement(name, cx, cy):

    history = tracks[name]

    history.append((cx, cy, time.time()))

    if len(history) < 4:
        return "stationary"

    old_x, old_y, _ = history[0]
    new_x, new_y, _ = history[-1]

    dx = new_x - old_x
    dy = new_y - old_y

    if abs(dx) < 12 and abs(dy) < 12:
        return "stationary"

    if abs(dx) > abs(dy):

        if dx > 0:
            return "moving right"

        return "moving left"

    if dy > 10:
        return "approaching"

    if dy < -10:
        return "moving away"

    return "moving"


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
# DISTANCE
# ============================================================

def estimate_distance_band(box_height, frame_height, name):

    ratio = box_height / frame_height

    # For people, a large bounding box means the person
    # is physically close to the camera.
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

    # General objects
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
# GEMINI
# ============================================================

def ask_gemini(detections, walking):

    global last_gemini_call

    if gemini_client is None:
        return None

    now = time.time()

    if now - last_gemini_call < GEMINI_COOLDOWN:
        return None

    last_gemini_call = now

    scene = []

    for d in detections[:6]:

        scene.append(
            f"{d['name']} on {d['direction']}, "
            f"{d['distance']}, {d['movement']}"
        )

    scene_text = "\n".join(scene)

    mode = "walking" if walking else "sitting/testing"

    prompt = f"""
You are the contextual AI for an accessibility smart cane.

Detected scene:
{scene_text}

User mode: {mode}

Give ONE short voice instruction for the user.

Rules:
- Use ONLY detected objects.
- Never invent anything.
- Never change the detected direction.
- Keep it under 12 words.
- Use simple natural language.
- Say ahead, left, or right.
- Do not mention AI, YOLO, pixels, bounding boxes,
  confidence, or computer vision.
"""

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        answer = response.text.strip()

        if answer:
            return answer

    except Exception as e:
        print("[GEMINI ERROR]", e)

    return None


# ============================================================
# LOCAL SAFETY LOGIC
# ============================================================

def create_message(detections, walking):

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

        person = min(
            people_ahead,
            key=lambda x: x["box_height"],
            reverse=True
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
                f"Person ahead, {distance}, moving toward you. "
                f"Be careful.",
                True
            )

        return (
            f"Person ahead, {distance}.",
            False
        )

    # --------------------------------------------------------
    # MULTIPLE PEOPLE
    # --------------------------------------------------------

    if len(people_ahead) >= 2:

        return (
            f"{len(people_ahead)} people ahead. Be careful.",
            True
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
            key=lambda x: x["box_height"]
        )

        return (
            f"{obj['name'].capitalize()} ahead, "
            f"{obj['distance']}.",
            True
        )

    # --------------------------------------------------------
    # CLOSE OBJECT ON SIDE
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
            key=lambda x: x["box_height"]
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
    print("=" * 60)
    print("SAHAYAK DRISHTI AI")
    print("PHONE CAMERA NAVIGATION")
    print("=" * 60)
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

    walking = mode == "w"

    print()
    print(
        "[MODE]",
        "Walking" if walking else "Sitting / Testing"
    )
    print()

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print("[YOLO] Loading model...")

    model = YOLO(MODEL_PATH)

    print("[YOLO] Model loaded.")
    print()

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

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    frame_number = 0

    last_message = ""
    last_message_time = 0

    MESSAGE_COOLDOWN = 2.0

    path_was_blocked = False

    print("=" * 60)
    print("NAVIGATION STARTED")
    print("=" * 60)
    print()
    print("Press Q to stop.")
    print()

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

        display = cv2.resize(
            frame,
            (640, 360)
        )

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

                distance = estimate_distance_band(
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
        # SAFETY MESSAGE
        # ----------------------------------------------------

        local_message, emergency = create_message(
            detections,
            walking
        )

        # ----------------------------------------------------
        # PATH STATE
        # ----------------------------------------------------

        blocked = any(
            d["direction"] == "ahead"
            and d["distance"] != "far"
            for d in detections
        )

        # Only say "Path is clear" after it was blocked.
        if path_was_blocked and not blocked:

            now = time.time()

            if now - last_message_time > MESSAGE_COOLDOWN:

                print(
                    f"[{time.strftime('%M:%S')}] "
                    "[INFO] Path is clear."
                )

                last_message = "Path is clear."
                last_message_time = now

        path_was_blocked = blocked

        # ----------------------------------------------------
        # GEMINI
        # ----------------------------------------------------

        gemini_message = None

        if detections and not emergency:

            gemini_message = ask_gemini(
                detections,
                walking
            )

        # ----------------------------------------------------
        # FINAL MESSAGE
        # ----------------------------------------------------

        message = local_message

        if message is None:
            message = gemini_message

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
                f"{d['direction']} "
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
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1
            )

        # Mode
        cv2.putText(
            display,
            "WALKING" if walking else "SITTING / TEST",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Gemini
        cv2.putText(
            display,
            "Gemini: ON" if gemini_client else "Gemini: OFF",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "Sahayak Drishti - Phone Navigation",
            display
        )

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("Navigation stopped.")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
