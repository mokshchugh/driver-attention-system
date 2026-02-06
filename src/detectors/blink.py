import numpy as np
from config import EAR_THRESHOLD

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def compute_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)


def get_eye_points(face_landmarks, w, h):
	left_eye_pts = np.array(
		[(face_landmarks[i].x * w, face_landmarks[i].y * h)
		 for i in LEFT_EYE]
	)

	right_eye_pts = np.array(
		[(face_landmarks[i].x * w, face_landmarks[i].y * h)
		 for i in RIGHT_EYE]
	)

	return left_eye_pts, right_eye_pts


def get_avg_ear(face_landmarks, w, h):
	left_eye_pts, right_eye_pts = get_eye_points(face_landmarks, w, h)

	return (compute_ear(left_eye_pts) +
			compute_ear(right_eye_pts)) / 2.0


def eyes_closed(avg_ear):
    return avg_ear < EAR_THRESHOLD
