import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.23
DROWSY_TIME = 1.5

# ----------------------------
# Momentum Risk Engine Settings
# ----------------------------
risk_score = 0.0

BLINK_RISK = 2
DROWSY_RISK = 25

DECAY_RATE = 0.92   # decay multiplier per second
RISK_THRESHOLD = 50

eye_closed_start = None
last_time = time.time()

# ----------------------------
def compute_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

# ----------------------------
# FaceLandmarker Setup
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    # Risk decay over time
    risk_score *= (DECAY_RATE ** dt)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]
        h, w, _ = frame.shape

        left_eye_pts = np.array(
            [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in LEFT_EYE]
        )
        right_eye_pts = np.array(
            [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in RIGHT_EYE]
        )

        avg_ear = (compute_ear(left_eye_pts) + compute_ear(right_eye_pts)) / 2.0

        # Display EAR
        cv2.putText(frame, f"EAR: {avg_ear:.2f}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # ----------------------------
        # Eye Closure Event Detection
        if avg_ear < EAR_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = time.time()

            closed_duration = time.time() - eye_closed_start

            # Add blink risk gradually
            risk_score += BLINK_RISK * dt

            if closed_duration > DROWSY_TIME:
                cv2.putText(frame, "DROWSINESS EVENT!",
                            (30, 140), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3)

        else:
            # Eyes reopened
            if eye_closed_start is not None:
                closed_duration = time.time() - eye_closed_start

                # If long closure → major risk event
                if closed_duration > DROWSY_TIME:
                    risk_score += DROWSY_RISK

                eye_closed_start = None

        # ----------------------------
        # Display Risk Score
        cv2.putText(frame, f"Momentum Risk: {risk_score:.1f}",
                    (30, 220), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 0), 3)

        # Threshold Alert
        if risk_score > RISK_THRESHOLD:
            cv2.putText(frame, "HIGH RISK WARNING!",
                        (30, 280), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 5)

    cv2.imshow("Momentum Risk Engine v1", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
