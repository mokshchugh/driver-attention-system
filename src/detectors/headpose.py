import numpy as np
from config import YAW_DEVIATION_THRESHOLD

class HeadPoseDetector:
	def __init__(self, yaw_threshold=YAW_DEVIATION_THRESHOLD):
		self.yaw_threshold = yaw_threshold

	def get_yaw(self, transformation_matrix):
		mat = np.array(transformation_matrix).reshape(4, 4)

		# Approx yaw estimation from rotation matrix
		yaw = np.arctan2(mat[0, 2], mat[2, 2])
		yaw_deg = np.degrees(yaw)

		return yaw_deg

	def looking_away(self, yaw_deg, baseline_yaw):
		deviation = abs(yaw_deg - baseline_yaw)
		return deviation > self.yaw_threshold
