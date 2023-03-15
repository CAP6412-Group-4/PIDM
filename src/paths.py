
from pathlib import Path

BASE_DIR = Path().parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = BASE_DIR / "input"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

DEEP_FASHION_DIR = DATA_DIR / "deepfashion_256x256"
TARGET_EDITS = DEEP_FASHION_DIR / "target_edits"
TARGET_MASK = DEEP_FASHION_DIR / "target_mask"
TARGET_POSE = DEEP_FASHION_DIR / "target_pose"

FASHION_ECOMMERCE_IMAGES_DIR = DATA_DIR / "fashion e-commerce images"
