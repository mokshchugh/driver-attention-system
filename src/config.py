from dotenv import load_dotenv
import os

# ============================
# CONFIGURATION FILE
# ============================

# Model Paths
FACE_MODEL_PATH = "models/face_landmarker.task"
YOLO_MODEL_PATH = "models/yolov8n.pt"

# Baseline defaults (used if no calibration exists)
DEFAULT_BASELINE_EAR = 0.28
DEFAULT_BASELINE_YAW = 0.0

# Detection sensitivity
EAR_SENSITIVITY = 0.75
YAW_DEVIATION_THRESHOLD = 20

# Eye Detection
DROWSY_TIME = 1.5

# Momentum Risk Parameters
BLINK_RISK = 2
DROWSY_RISK = 25
PHONE_RISK = 20
CRITICAL_RISK = 60

DECAY_RATE = 0.92
RISK_THRESHOLD = 60

# Calibration
CALIBRATION_DURATION_SECONDS = 15

# Database

# Walks up from src/ to find .env at project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Local Postgres
LOCAL_PG_HOST     = os.getenv("LOCAL_PG_HOST", "localhost")
LOCAL_PG_PORT     = int(os.getenv("LOCAL_PG_PORT", 5432))
LOCAL_PG_DB       = os.getenv("LOCAL_PG_DB")
LOCAL_PG_USER     = os.getenv("LOCAL_PG_USER")
LOCAL_PG_PASSWORD = os.getenv("LOCAL_PG_PASSWORD")

# Neon Postgres
NEON_HOST         = os.getenv("NEON_HOST")
NEON_PORT         = int(os.getenv("NEON_PORT", 5432))
NEON_DB           = os.getenv("NEON_DB")
NEON_USER         = os.getenv("NEON_USER")
NEON_PASSWORD     = os.getenv("NEON_PASSWORD")


# Sanity check on startup

# Detect environment
STREAMLIT_CLOUD = os.getenv("STREAMLIT_CLOUD", "false").lower() == "true"

# Base required vars (always needed)
_required = {
    "NEON_HOST": NEON_HOST,
    "NEON_DB": NEON_DB,
    "NEON_USER": NEON_USER,
    "NEON_PASSWORD": NEON_PASSWORD,
}

# Local DB only required outside Streamlit Cloud
if not STREAMLIT_CLOUD:
    _required.update({
        "LOCAL_PG_DB": LOCAL_PG_DB,
        "LOCAL_PG_USER": LOCAL_PG_USER,
        "LOCAL_PG_PASSWORD": LOCAL_PG_PASSWORD,
    })

_missing = [k for k, v in _required.items() if not v]

if _missing:
    raise EnvironmentError(
        f"Missing required env vars: {', '.join(_missing)}"
    )
