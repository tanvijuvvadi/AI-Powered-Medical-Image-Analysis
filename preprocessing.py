"""OpenCV/NumPy helpers for single images (used by predict, API and demos).

Training uses tf.data pipelines (see data_loader.py); this module guarantees
that inference applies the SAME steps: RGB, 224x224, float32 in [0, 255].
Normalisation to [-1, 1] happens inside the model (the 'preprocess' layer).
"""
import cv2
import numpy as np

from src.config import IMG_SIZE


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def decode_gray(raw_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Uploaded file is not a valid image")
    return img


def apply_clahe(gray: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast enhancement. Used for visual demo only (not in the trained pipeline)."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)


def to_model_input(gray: np.ndarray) -> np.ndarray:
    """Grayscale uint8 -> (1, 224, 224, 3) float32 in [0, 255]."""
    resized = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype("float32")
    return rgb[np.newaxis, ...]
