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

# Time thresholds
DROWSY_TIME = 1.5   # eyes closed > 1.5s = drowsy

# Blink state variables
eye_closed_start = None
drowsy_events = 0

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

        left_ear = compute_ear(left_eye_pts)
        right_ear = compute_ear(right_eye_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        # Display EAR
        cv2.putText(frame, f"EAR: {avg_ear:.2f}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # ----------------------------
        # Eye Closure Tracking
        if avg_ear < EAR_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = time.time()

            closed_duration = time.time() - eye_closed_start

            cv2.putText(frame, f"Eyes Closed: {closed_duration:.2f}s",
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

            # Drowsiness Event
            if closed_duration > DROWSY_TIME:
                cv2.putText(frame, "DROWSINESS ALERT!",
                            (30, 160), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 4)

        else:
            # Eyes reopened
            if eye_closed_start is not None:
                closed_duration = time.time() - eye_closed_start

                # If it was a long closure, count as drowsy event
                if closed_duration > DROWSY_TIME:
                    drowsy_events += 1

                eye_closed_start = None

        # Show event count
        cv2.putText(frame, f"Drowsy Events: {drowsy_events}",
                    (30, 220), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
