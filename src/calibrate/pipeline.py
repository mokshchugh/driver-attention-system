import time

import cv2
import numpy as np

from config import CALIBRATION_DURATION_SECONDS

from detectors.blink import get_avg_ear
from detectors.face_landmarks import FaceLandmarkDetector
from detectors.headpose import HeadPoseDetector

from db.drivers import (
    update_baseline_local,
    update_baseline_neon,
)


def run_calibration_pipeline(
    driver_id,
    duration_seconds=CALIBRATION_DURATION_SECONDS,
):

    face_detector = FaceLandmarkDetector()
    headpose_detector = HeadPoseDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to access camera for calibration."
        )

    ear_samples = []
    yaw_samples = []

    started_at = time.time()

    try:
        while time.time() - started_at < duration_seconds:

            ret, frame = cap.read()

            if not ret:
                continue

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            result = face_detector.detect(rgb)

            if result.face_landmarks:

                face_landmarks = result.face_landmarks[0]

                h, w, _ = frame.shape

                # EAR
                avg_ear = get_avg_ear(
                    face_landmarks,
                    w,
                    h,
                )

                ear_samples.append(avg_ear)

                # YAW
                if result.facial_transformation_matrixes:

                    yaw = headpose_detector.get_yaw(
                        result.facial_transformation_matrixes[0]
                    )

                    yaw_samples.append(yaw)

            remaining = max(
                0,
                int(
                    duration_seconds
                    - (time.time() - started_at)
                ),
            )

            cv2.putText(
                frame,
                f"Calibration in progress: {remaining}s",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "Look straight ahead and keep eyes naturally open",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Driver Baseline Calibration",
                frame,
            )

            # ESC to quit
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not ear_samples or not yaw_samples:
        raise RuntimeError(
            "Calibration failed because no valid face data was captured."
        )

    average_ear = float(np.mean(ear_samples))
    average_yaw = float(np.mean(yaw_samples))
 
    # Update local cache
    local_driver = update_baseline_local(
        driver_id,
        average_ear,
        average_yaw,
    )

    if local_driver is None:
        raise RuntimeError(
            f"Driver {driver_id} not found locally."
        )

    # Sync to Neon
    neon_driver = update_baseline_neon(
        driver_id,
        average_ear,
        average_yaw,
    )

    if neon_driver is None:
        raise RuntimeError(
            f"Driver {driver_id} not found in Neon."
        )

    return {
        "driver_id": driver_id,
        "baseline_ear": average_ear,
        "baseline_yaw": average_yaw,
    }
