import time
from config import (
	BLINK_RISK, DROWSY_RISK, PHONE_RISK,
	DECAY_RATE, RISK_THRESHOLD
)

CRITICAL_RISK = 60


class MomentumRiskEngine:
	def __init__(self):
		self.risk_score = 0.0
		self.eye_closed_start = None
		self.last_time = time.time()

		# Flags for one-time penalties
		self.drowsy_flag = False
		self.critical_flag = False

	def update_decay(self):
		now = time.time()
		dt = now - self.last_time
		self.last_time = now

		self.risk_score *= (DECAY_RATE ** dt)
		return dt

	def add_blink_risk(self, dt):
		self.risk_score += BLINK_RISK * dt

	def add_phone_risk(self, dt):
		self.risk_score += PHONE_RISK * dt

	def add_drowsy_event(self):
		self.risk_score += DROWSY_RISK

	def add_critical_event(self):
		self.risk_score += CRITICAL_RISK

	def high_risk(self):
		return self.risk_score > RISK_THRESHOLD
