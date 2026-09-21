import cv2
import json
import queue
import threading
import subprocess
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from ultralytics import YOLO


# =========================================================
# 1. TEXT-TO-SPEECH
# =========================================================

def speak(text):

    print(f">> Speaking: {text}")

    # Windows built-in speech
    powershell_command = f'''
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.Rate = 0
$speaker.Volume = 100
$speaker.Speak("{text}")
$speaker.Dispose()
'''

    try:

        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                powershell_command
            ],
            check=True
        )

        print(">> TTS finished")

    except Exception as e:

        print(f"[TTS Error]: {e}")


# =========================================================
# 2. VOSK VOICE RECOGNITION
# =========================================================

audio_q = queue.Queue()


def audio_callback(indata, frames, time, status):

    audio_q.put(bytes(indata))


print(">> Loading Vosk model...")

vosk_model = Model("model")

recognizer = KaldiRecognizer(
    vosk_model,
    16000
)

stream = sd.RawInputStream(
    samplerate=16000,
    blocksize=8000,
    dtype='int16',
    channels=1,
    callback=audio_callback
)

stream.start()

print(">> Vosk model loaded")


# =========================================================
# 3. YOLO
# =========================================================

print(">> Loading YOLO model...")

yolo_model = YOLO(
    "yolov8n.pt"
)

print(">> YOLO model loaded")


# =========================================================
# 4. CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print(
        "Error: Could not access camera."
    )

    stream.stop()
    stream.close()

    exit()


# =========================================================
# 5. STARTUP
# =========================================================

print("=" * 55)

print(
    "Sahayak Drishti: Voice-Activated Smart Cane Ready!"
)

print(
    "Ask questions like: 'What is ahead?', 'Look', 'Describe'"
)

print(
    "Press 'q' in the camera window to quit."
)

print("=" * 55)

print()


# Speak only once at startup

speak(
    "Hello, I am Sahayak Drishti. How can I help you?"
)


# =========================================================
# 6. VOICE TRIGGERS
# =========================================================

TRIGGER_WORDS = [
    "ahead",
    "front",
    "detect",
    "describe",
    "look",
    "see",
    "show"
]


# =========================================================
# 7. MAIN LOOP
# =========================================================

try:

    while True:

        # -------------------------------------------------
        # Get camera frame
        # -------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print(
                "Error: Could not read camera frame."
            )

            break


        # -------------------------------------------------
        # Check voice input
        # -------------------------------------------------

        trigger_detected = False

        while not audio_q.empty():

            data = audio_q.get()

            if recognizer.AcceptWaveform(data):

                res = json.loads(
                    recognizer.Result()
                )

                spoken_text = res.get(
                    "text",
                    ""
                )

                if spoken_text:

                    print(
                        f'[Heard]: "{spoken_text}"'
                    )

                    spoken_lower = (
                        spoken_text.lower()
                    )

                    if any(
                        word in spoken_lower
                        for word in TRIGGER_WORDS
                    ):

                        trigger_detected = True


        # -------------------------------------------------
        # Run YOLO only when user asks
        # -------------------------------------------------

        if trigger_detected:

            print(
                ">> Trigger recognized! "
                "Analyzing scene..."
            )

            results = yolo_model(
                frame,
                verbose=False
            )

            print(
                ">> YOLO detection finished"
            )


            # Camera dimensions

            height, width, _ = frame.shape

            print(
                f">> Frame size: {width} x {height}"
            )

            print(
                ">> Processing detections..."
            )


            detected_items = []


            # -------------------------------------------------
            # Process detections
            # -------------------------------------------------

            for r in results:

                for box in r.boxes:

                    confidence = float(
                        box.conf[0]
                    )

                    # Ignore weak detections

                    if confidence < 0.5:
                        continue


                    # Object name

                    cls_id = int(
                        box.cls[0]
                    )

                    cls_name = (
                        yolo_model.names[cls_id]
                    )


                    # Bounding box

                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )


                    # Object center

                    center_x = (
                        (x1 + x2) / 2
                    )


                    # -------------------------------------------------
                    # LEFT OR RIGHT ONLY
                    # -------------------------------------------------

                    if center_x < width / 2:

                        pos = "on your left"

                        direction = "LEFT"

                    else:

                        pos = "on your right"

                        direction = "RIGHT"


                    print(
                        f">> Detected: "
                        f"{cls_name} - {direction}"
                    )


                    detected_items.append(
                        f"{cls_name} {pos}"
                    )


            print(
                ">> Detection processing finished"
            )


            # -------------------------------------------------
            # Create answer
            # -------------------------------------------------

            if detected_items:

                unique_items = list(
                    dict.fromkeys(
                        detected_items
                    )
                )

                message = (
                    "I see "
                    + ", and ".join(
                        unique_items
                    )
                )

            else:

                message = (
                    "I do not see anything"
                )


            print(
                f">> Final answer: {message}"
            )


            # Speak answer

            speak(message)


            print(
                ">> Returned to waiting mode"
            )


        # -------------------------------------------------
        # Show camera
        # -------------------------------------------------

        cv2.imshow(
            "Sahayak Drishti Feed",
            frame
        )


        # Press Q to quit

        if (
            cv2.waitKey(1) & 0xFF
            == ord('q')
        ):

            break


finally:

    print(
        "Sahayak Drishti stopped."
    )

    stream.stop()
    stream.close()

    cap.release()

    cv2.destroyAllWindows()