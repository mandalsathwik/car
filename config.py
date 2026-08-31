import os

# Model Paths
VEHICLE_MODEL_PATH = "yolo11n.pt"
PLATE_MODEL_PATH = "models/license_plate.pt"

# Detection Parameters
CONF_THRESHOLD = 0.20
IOU_THRESHOLD = 0.45
IMG_SIZE = 640  # 640x640 optimal resolution for high FPS real-time performance

# Target Vehicle Categories (to be matched dynamically against model.names)
TARGET_VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck", "bicycle"]

# Plate Detection & OCR Parameters
PLATE_CONF_THRESHOLD = 0.40
OCR_INTERVAL = 5     # Run OCR every N frames per active vehicle track
TEMPORAL_VOTING_WINDOW = 10  # Number of OCR observations to keep for voting

# Tracker Settings
TRACKER_CONFIG = "bytetrack.yaml"

# Input / Output Settings
VIDEO_SOURCE = "test.video/4.mp4"
OUTPUT_CSV = "vehicle_events.csv"
SHOW_DISPLAY = True
MAX_DISPLAY_WIDTH = 1280

# Environment check
HAS_PLATE_MODEL = os.path.exists(PLATE_MODEL_PATH)
