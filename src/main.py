import cv2

from detectors.face_landmarks import FaceLandmarkDetector
from detectors.blink import get_avg_ear, eyes_closed, get_eye_points
from detectors.phone import PhoneDetector

from risk.momentum import MomentumRiskEngine
from db.logger import DBLogger

from config import DROWSY_TIME


def main():
	face_detector = FaceLandmarkDetector()
	phone_detector = PhoneDetector()
	risk_engine = MomentumRiskEngine()
	logger = DBLogger()

	cap = cv2.VideoCapture(0)

	print("Driver Attention Momentum System Running (ESC to quit)")

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		dt = risk_engine.update_decay()

		# ---- Phone Detection ----
		phone_detected, phone_box = phone_detector.detect(frame)

		if phone_detected:
			# Draw bounding box
			x1, y1, x2, y2 = phone_box
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

			# Add momentum risk
			risk_engine.add_phone_risk(dt)

			# Show feedback on screen
			cv2.putText(
				frame,
				"PHONE DETECTED!",
				(30, 200),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.2,
				(0, 0, 255),
				3
			)

			# Log only once every few seconds (cooldown)
			if not hasattr(main, "last_phone_log"):
				main.last_phone_log = 0

			if (cv2.getTickCount() - main.last_phone_log) / cv2.getTickFrequency() > 3:
				logger.log(risk_engine.risk_score, "PHONE_DISTRACTION_EVENT")
				main.last_phone_log = cv2.getTickCount()

		# ---- Face + EAR ----
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		result = face_detector.detect(rgb)

		if result.face_landmarks:
			face_landmarks = result.face_landmarks[0]
			h, w, _ = frame.shape

			avg_ear = get_avg_ear(face_landmarks, w, h)

			if eyes_closed(avg_ear):
				risk_engine.add_blink_risk(dt)

			else:
				if risk_engine.eye_closed_start is not None:
					risk_engine.add_drowsy_event()
					logger.log(risk_engine.risk_score, "DROWSINESS_EVENT")
					risk_engine.eye_closed_start = None

			# Draw eye landmark dots
			left_eye_pts, right_eye_pts = get_eye_points(face_landmarks, w, h)

			for (x, y) in left_eye_pts:
				cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)

			for (x, y) in right_eye_pts:
				cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)

		# ---- Display Risk ----
		cv2.putText(
			frame,
			f"Risk: {risk_engine.risk_score:.1f}",
			(30, 80),
			cv2.FONT_HERSHEY_SIMPLEX,
			1.3,
			(255, 255, 0),
			3
		)

		if risk_engine.high_risk():
			cv2.putText(
				frame,
				"HIGH RISK!",
				(30, 150),
				cv2.FONT_HERSHEY_SIMPLEX,
				2,
				(0, 0, 255),
				5
			)
			logger.log(risk_engine.risk_score, "HIGH_RISK_WARNING")

		# ---- Show Window ----
		cv2.imshow("Driver Attention Momentum System", frame)

		if cv2.waitKey(1) & 0xFF == 27:
			break

	cap.release()
	cv2.destroyAllWindows()
	logger.close()


if __name__ == "__main__":
	main()
