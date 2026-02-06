import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import FACE_MODEL_PATH


class FaceLandmarkDetector:
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path=FACE_MODEL_PATH
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_rgb):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        return self.detector.detect(mp_image)
