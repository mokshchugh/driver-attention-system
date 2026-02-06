import cv2
import numpy as np
import time
import psycopg2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ultralytics import YOLO

# ----------------------------
# Models
# ----------------------------
FACE_MODEL = "models/face_landmarker.task"
YOLO_MODEL = "yolov8n.pt"

# Load YOLO
yolo = YOLO(YOLO_MODEL)

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Thresholds
EAR_THRESHOLD = 0.23
DROWSY_TIME = 1.5

# Risk Engine Parameters
risk_score = 0.0

BLINK_RISK = 2
DROWSY_RISK = 25
PHONE_RISK = 20

DECAY_RATE = 0.92
RISK_THRESHOLD = 60

eye_closed_start = None
last_time = time.time()

# ----------------------------
# PostgreSQL Connection
# ----------------------------
conn = psycopg2.connect(
    dbname="driver_attention_db",
    user="moksh"
)
cursor = conn.cursor()

def log_event(score, event):
    cursor.execute(
        "INSERT INTO risk_logs (risk_score, event_type) VALUES (%s, %s)",
        (score, event)
    )
    conn.commit()

# ----------------------------
def compute_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

# ----------------------------
# FaceLandmarker Setup
# ----------------------------
base_options = python.BaseOptions(model_asset_path=FACE_MODEL)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# ----------------------------
# Webcam Start
# ----------------------------
cap = cv2.VideoCapture(0)

print("Final Attention Engine Running... Press ESC to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------------
    # Time + Decay
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    risk_score *= (DECAY_RATE ** dt)

    # ----------------------------
    # YOLO Phone Detection
    phone_detected = False

    results = yolo(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = yolo.names[cls]

            if label == "cell phone":
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 0, 255), 3)
                cv2.putText(frame, "PHONE!",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

    if phone_detected:
        risk_score += PHONE_RISK * dt
        log_event(risk_score, "PHONE_DISTRACTION")

    # ----------------------------
    # Face + EAR Detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]
        h, w, _ = frame.shape

        left_eye_pts = np.array(
            [(face_landmarks[i].x * w, face_landmarks[i].y * h)
             for i in LEFT_EYE]
        )
        right_eye_pts = np.array(
            [(face_landmarks[i].x * w, face_landmarks[i].y * h)
             for i in RIGHT_EYE]
        )

        avg_ear = (compute_ear(left_eye_pts) +
                   compute_ear(right_eye_pts)) / 2.0

        cv2.putText(frame, f"EAR: {avg_ear:.2f}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Eye closure logic
        if avg_ear < EAR_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = time.time()

            risk_score += BLINK_RISK * dt

        else:
            if eye_closed_start is not None:
                closed_duration = time.time() - eye_closed_start

                if closed_duration > DROWSY_TIME:
                    risk_score += DROWSY_RISK
                    log_event(risk_score, "DROWSINESS_EVENT")

                eye_closed_start = None

    # ----------------------------
    # Display Risk Score
    cv2.putText(frame, f"Momentum Risk: {risk_score:.1f}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (255, 255, 0), 3)

    # Threshold Alert
    if risk_score > RISK_THRESHOLD:
        cv2.putText(frame, "HIGH RISK!",
                    (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), 5)
        log_event(risk_score, "HIGH_RISK_WARNING")

    cv2.imshow("Driver Attention Momentum Engine", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()

print("Session Ended. Logs saved.")
