# ============================
# CONFIGURATION FILE
# ============================

# Model Paths
FACE_MODEL_PATH = "models/face_landmarker.task"
YOLO_MODEL_PATH = "models/yolov8n.pt"

# Eye Detection
EAR_THRESHOLD = 0.22
DROWSY_TIME = 1.5

# Momentum Risk Parameters
BLINK_RISK = 2
DROWSY_RISK = 25
PHONE_RISK = 20
CRITICAL_RISK = 60

DECAY_RATE = 0.92
RISK_THRESHOLD = 60

# Database
DB_NAME = "driver_attention_db"
DB_USER = "moksh"
