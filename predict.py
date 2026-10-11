"""Single-image inference with optional Grad-CAM figure."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import CLASS_NAMES, MODEL_PATH, OUTPUT_DIR, THRESHOLD
from src.gradcam import make_gradcam, overlay_heatmap
from src.preprocessing import load_gray, to_model_input

_model = None


def get_model(path=MODEL_PATH):
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(path)
    return _model


def predict_gray(gray, model=None):
    model = model or get_model()
    prob = float(model.predict(to_model_input(gray), verbose=0)[0][0])
    label = CLASS_NAMES[int(prob >= THRESHOLD)]
    return label, prob


def predict_file(path, with_gradcam=True):
    gray = load_gray(path)
    model = get_model()
    label, prob = predict_gray(gray, model)
    print(f"{Path(path).name}: {label} (P(pneumonia) = {prob:.3f})")
    if with_gradcam:
        cam = make_gradcam(model, to_model_input(gray))
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Input X-ray")
        axes[1].imshow(overlay_heatmap(gray, cam)); axes[1].set_title(f"Grad-CAM: {label} ({prob:.2f})")
        for a in axes: a.axis("off")
        out = OUTPUT_DIR / f"prediction_{Path(path).stem}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Saved {out}")
    return label, prob
