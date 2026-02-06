import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR threshold (tweak later)
EAR_THRESHOLD = 0.22

# ----------------------------
# EAR Calculation Function
# ----------------------------
def compute_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])

    ear = (A + B) / (2.0 * C)
    return ear

# ----------------------------
# MediaPipe FaceLandmarker Setup
# ----------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# ----------------------------
# Webcam Start
# ----------------------------
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

        # Extract eye points
        left_eye_pts = np.array(
            [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in LEFT_EYE]
        )

        right_eye_pts = np.array(
            [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in RIGHT_EYE]
        )

        # Compute EAR for both eyes
        left_ear = compute_ear(left_eye_pts)
        right_ear = compute_ear(right_eye_pts)

        avg_ear = (left_ear + right_ear) / 2.0

        # Display EAR value
        cv2.putText(frame, f"EAR: {avg_ear:.2f}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Blink detection
        if avg_ear < EAR_THRESHOLD:
            cv2.putText(frame, "BLINK / EYES CLOSED!",
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

        # Draw eye landmarks
        for (x, y) in left_eye_pts:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1)

        for (x, y) in right_eye_pts:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1)

    cv2.imshow("Blink Detection (EAR)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
