import cv2

from detectors.face_landmarks import FaceLandmarkDetector
from detectors.blink import get_avg_ear, eyes_closed, get_eye_points
from detectors.phone import PhoneDetector
from detectors.headpose import HeadPoseDetector

from risk.momentum import MomentumRiskEngine
from db.logger import DBLogger


def main():
	face_detector = FaceLandmarkDetector()
	phone_detector = PhoneDetector()
	headpose_detector = HeadPoseDetector()

	risk_engine = MomentumRiskEngine()
	logger = DBLogger()

	cap = cv2.VideoCapture(0)

	print("Driver Attention Momentum System Running (ESC to quit)")

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		dt = risk_engine.update_decay()

		# ==================================================
		# PHONE DETECTION (YOLO)
		# ==================================================
		phone_detected, phone_box = phone_detector.detect(frame)

		if phone_detected:
			# Draw bounding box
			x1, y1, x2, y2 = phone_box
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

			# Add momentum risk
			risk_engine.add_phone_risk(dt)

			# Display warning
			cv2.putText(
				frame,
				"PHONE DETECTED!",
				(30, 200),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.2,
				(0, 0, 255),
				3
			)

			# Cooldown logging (once every 3 seconds)
			if not hasattr(main, "last_phone_log"):
				main.last_phone_log = 0

			if (cv2.getTickCount() - main.last_phone_log) / cv2.getTickFrequency() > 3:
				logger.log(risk_engine.risk_score, "PHONE_DISTRACTION_EVENT")
				main.last_phone_log = cv2.getTickCount()

		# ==================================================
		# FACE + EYES + HEADPOSE
		# ==================================================
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		result = face_detector.detect(rgb)

		if result.face_landmarks:
			face_landmarks = result.face_landmarks[0]
			h, w, _ = frame.shape

			# ------------------------------------------
			# FULL FACE DOTS (Green)
			# ------------------------------------------
			for lm in face_landmarks:
				x = int(lm.x * w)
				y = int(lm.y * h)
				cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

			# ------------------------------------------
			# EYE DOTS (Yellow)
			# ------------------------------------------
			left_eye_pts, right_eye_pts = get_eye_points(face_landmarks, w, h)

			for (x, y) in left_eye_pts:
				cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)

			for (x, y) in right_eye_pts:
				cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)

			# ------------------------------------------
			# EAR Calculation
			# ------------------------------------------
			avg_ear = get_avg_ear(face_landmarks, w, h)

			cv2.putText(
				frame,
				f"EAR: {avg_ear:.2f}",
				(30, 50),
				cv2.FONT_HERSHEY_SIMPLEX,
				1,
				(255, 255, 255),
				2
			)

			# ------------------------------------------
			# HEAD POSE (Yaw)
			# ------------------------------------------
			if result.facial_transformation_matrixes:
				matrix = result.facial_transformation_matrixes[0]
				yaw = headpose_detector.get_yaw(matrix)

				cv2.putText(
					frame,
					f"Yaw: {yaw:.1f}",
					(30, 250),
					cv2.FONT_HERSHEY_SIMPLEX,
					1,
					(255, 255, 255),
					2
				)

				if headpose_detector.looking_away(yaw):
					cv2.putText(
						frame,
						"LOOKING AWAY!",
						(30, 300),
						cv2.FONT_HERSHEY_SIMPLEX,
						1.5,
						(0, 0, 255),
						4
					)

					# Add distraction risk
					risk_engine.risk_score += 10 * dt

					# Cooldown logging (once every 4 seconds)
					if not hasattr(main, "last_gaze_log"):
						main.last_gaze_log = 0

					if (cv2.getTickCount() - main.last_gaze_log) / cv2.getTickFrequency() > 4:
						logger.log(risk_engine.risk_score, "GAZE_AWAY_EVENT")
						main.last_gaze_log = cv2.getTickCount()

			# ------------------------------------------
			# EYE CLOSURE DURATION PENALTIES
			# ------------------------------------------
			if eyes_closed(avg_ear):
				if risk_engine.eye_closed_start is None:
					risk_engine.eye_closed_start = cv2.getTickCount()

				duration = (
					(cv2.getTickCount() - risk_engine.eye_closed_start)
					/ cv2.getTickFrequency()
				)

				risk_engine.add_blink_risk(dt)

				cv2.putText(
					frame,
					f"Eyes Closed: {duration:.1f}s",
					(30, 350),
					cv2.FONT_HERSHEY_SIMPLEX,
					1,
					(0, 0, 255),
					2
				)

				# Drowsy event after 1.5s
				if duration > 1.5 and not risk_engine.drowsy_flag:
					risk_engine.add_drowsy_event()
					logger.log(risk_engine.risk_score, "DROWSINESS_EVENT")
					risk_engine.drowsy_flag = True

				# Critical microsleep after 5s
				if duration > 5 and not risk_engine.critical_flag:
					risk_engine.add_critical_event()
					logger.log(risk_engine.risk_score, "MICROSLEEP_EVENT")
					risk_engine.critical_flag = True

			else:
				# Reset when eyes reopen
				risk_engine.eye_closed_start = None
				risk_engine.drowsy_flag = False
				risk_engine.critical_flag = False

		# ==================================================
		# DISPLAY MOMENTUM RISK SCORE
		# ==================================================
		cv2.putText(
			frame,
			f"Momentum Risk: {risk_engine.risk_score:.1f}",
			(30, 120),
			cv2.FONT_HERSHEY_SIMPLEX,
			1.3,
			(255, 255, 0),
			3
		)

		if risk_engine.high_risk():
			cv2.putText(
				frame,
				"HIGH RISK WARNING!",
				(30, 170),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.7,
				(0, 0, 255),
				5
			)

			if not hasattr(main, "last_highrisk_log"):
				main.last_highrisk_log = 0

			if (cv2.getTickCount() - main.last_highrisk_log) / cv2.getTickFrequency() > 5:
				logger.log(risk_engine.risk_score, "HIGH_RISK_WARNING")
				main.last_highrisk_log = cv2.getTickCount()

		# ==================================================
		# SHOW WINDOW
		# ==================================================
		cv2.imshow("Driver Attention Momentum System", frame)

		if cv2.waitKey(1) & 0xFF == 27:
			break

	cap.release()
	cv2.destroyAllWindows()
	logger.close()


if __name__ == "__main__":
	main()
