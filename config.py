"""Central configuration. Change values here, not inside other files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "chest_xray"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "images"
MODEL_PATH = MODEL_DIR / "pneumonia_mobilenetv2.keras"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]   # alphabetical => label 0 / 1
IMG_SIZE = (224, 224)                   # MobileNetV2 default input
BATCH_SIZE = 32
SEED = 42
VAL_SPLIT = 0.2                         # carved out of train (official val is only 16 images)

EPOCHS_HEAD = 5                         # phase 1: train only the new classifier head
EPOCHS_FINE = 10                        # phase 2: fine-tune top of the backbone
LR_HEAD = 1e-3
LR_FINE = 1e-5
FINE_TUNE_FROM = 100                    # unfreeze MobileNetV2 layers from this index onward
THRESHOLD = 0.5                         # P(pneumonia) >= threshold => PNEUMONIA

for d in (MODEL_DIR, OUTPUT_DIR, IMAGE_DIR):
    d.mkdir(parents=True, exist_ok=True)
