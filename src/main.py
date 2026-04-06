import argparse
import getpass
import sys

import cv2

from db.accounts import get_account_by_email, verify_password
from db.drivers import get_driver
from detectors.face_landmarks import FaceLandmarkDetector
from detectors.blink import get_avg_ear, eyes_closed, get_eye_points
from detectors.phone import PhoneDetector
from detectors.headpose import HeadPoseDetector

from risk.momentum import MomentumRiskEngine
from db.logger import DBLogger
from services.session_manager import SessionManager


# ─────────────────────────────────────────────
# NEW: terminal login — resolves driver_id from DB
# ─────────────────────────────────────────────

def prompt_login() -> int:
    """
    Prompt for email + password in the terminal.
    Verifies against the accounts table and returns the bound driver_id.
    Exits with a clear message on failure.
    """
    print("=== Driver Attention System – Login ===")
    email    = input("Email: ").strip().lower()
    password = getpass.getpass("Password: ")

    account = get_account_by_email(email)

    if account is None:
        print("Error: no account found for that email.")
        sys.exit(1)

    if not verify_password(password, account["password_hash"]):
        print("Error: incorrect password.")
        sys.exit(1)

    driver_id = account["driver_id"]
    if driver_id is None:
        print("Error: account has no linked driver profile.")
        sys.exit(1)

    print(f"Logged in as {account['name']} (driver #{driver_id})\n")
    return driver_id


# ─────────────────────────────────────────────
# EXISTING: main loop (driver_id source changed, nothing else)
# ─────────────────────────────────────────────

def main(driver_id: int):
    driver = get_driver(driver_id)

    if driver is None:
        raise ValueError(f"Driver {driver_id} does not exist.")

    face_detector    = FaceLandmarkDetector()
    phone_detector   = PhoneDetector()
    headpose_detector = HeadPoseDetector()

    risk_engine     = MomentumRiskEngine()
    session_manager = SessionManager(driver_id)
    session_manager.start_session()

    logger = DBLogger(driver_id=driver_id, session_id=session_manager.session_id)

    cap = cv2.VideoCapture(0)

    print(f"Driver Attention Momentum System Running for {driver['name']} (ESC to quit)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dt = risk_engine.update_decay()

            # ── PHONE DETECTION (YOLO) ─────────────────────
            phone_detected, phone_box = phone_detector.detect(frame)

            if phone_detected:
                x1, y1, x2, y2 = phone_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                risk_engine.add_phone_risk(dt)
                cv2.putText(frame, "PHONE DETECTED!", (30, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                if not hasattr(main, "last_phone_log"):
                    main.last_phone_log = 0
                if (cv2.getTickCount() - main.last_phone_log) / cv2.getTickFrequency() > 3:
                    logger.log(risk_engine.risk_score, "PHONE_DISTRACTION_EVENT")
                    main.last_phone_log = cv2.getTickCount()

            # ── FACE + EYES + HEADPOSE ─────────────────────
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_detector.detect(rgb)

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                h, w, _ = frame.shape

                for lm in face_landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

                left_eye_pts, right_eye_pts = get_eye_points(face_landmarks, w, h)
                for (x, y) in left_eye_pts + right_eye_pts:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)

                avg_ear = get_avg_ear(face_landmarks, w, h)
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if result.facial_transformation_matrixes:
                    yaw = headpose_detector.get_yaw(result.facial_transformation_matrixes[0])
                    cv2.putText(frame, f"Yaw: {yaw:.1f}", (30, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    if headpose_detector.looking_away(yaw):
                        cv2.putText(frame, "LOOKING AWAY!", (30, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        risk_engine.risk_score += 10 * dt

                        if not hasattr(main, "last_gaze_log"):
                            main.last_gaze_log = 0
                        if (cv2.getTickCount() - main.last_gaze_log) / cv2.getTickFrequency() > 4:
                            logger.log(risk_engine.risk_score, "GAZE_AWAY_EVENT")
                            main.last_gaze_log = cv2.getTickCount()

                if eyes_closed(avg_ear):
                    if risk_engine.eye_closed_start is None:
                        risk_engine.eye_closed_start = cv2.getTickCount()

                    duration = (cv2.getTickCount() - risk_engine.eye_closed_start) / cv2.getTickFrequency()
                    risk_engine.add_blink_risk(dt)

                    cv2.putText(frame, f"Eyes Closed: {duration:.1f}s", (30, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if duration > 1.5 and not risk_engine.drowsy_flag:
                        risk_engine.add_drowsy_event()
                        logger.log(risk_engine.risk_score, "DROWSINESS_EVENT")
                        risk_engine.drowsy_flag = True

                    if duration > 5 and not risk_engine.critical_flag:
                        risk_engine.add_critical_event()
                        logger.log(risk_engine.risk_score, "MICROSLEEP_EVENT")
                        risk_engine.critical_flag = True

                else:
                    risk_engine.eye_closed_start = None
                    risk_engine.drowsy_flag      = False
                    risk_engine.critical_flag    = False

            # ── RISK DISPLAY ───────────────────────────────
            cv2.putText(frame, f"Momentum Risk: {risk_engine.risk_score:.1f}", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 3)

            if risk_engine.high_risk():
                cv2.putText(frame, "HIGH RISK WARNING!", (30, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 5)

                if not hasattr(main, "last_highrisk_log"):
                    main.last_highrisk_log = 0
                if (cv2.getTickCount() - main.last_highrisk_log) / cv2.getTickFrequency() > 5:
                    logger.log(risk_engine.risk_score, "HIGH_RISK_WARNING")
                    main.last_highrisk_log = cv2.getTickCount()

            session_manager.update_max_risk(risk_engine.risk_score)

            cv2.imshow("Driver Attention Momentum System", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        session_manager.close_session()
        logger.close()


# ─────────────────────────────────────────────
# MODIFIED: --driver-id removed; login prompt resolves it
# ─────────────────────────────────────────────

if __name__ == "__main__":
    driver_id = prompt_login()
    main(driver_id=driver_id)
