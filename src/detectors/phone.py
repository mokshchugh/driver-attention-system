from ultralytics import YOLO
from config import YOLO_MODEL_PATH


class PhoneDetector:
	def __init__(self):
		self.model = YOLO(YOLO_MODEL_PATH)

	def detect(self, frame):
		results = self.model(frame, verbose=False)

		for r in results:
			for box in r.boxes:
				cls = int(box.cls[0])
				label = self.model.names[cls]

				if label == "cell phone":
					x1, y1, x2, y2 = map(int, box.xyxy[0])
					return True, (x1, y1, x2, y2)

		return False, None
