"""Grad-CAM: highlights the image regions that drove the prediction (explainability)."""
import cv2
import numpy as np
import tensorflow as tf

from src.model import get_backbone


def make_gradcam(model, batch: np.ndarray) -> np.ndarray:
    """batch: (1,224,224,3) float in [0,255]. Returns heatmap in [0,1], shape (7,7)."""
    base = get_backbone(model)
    conv_model = tf.keras.Model(base.input, base.get_layer("out_relu").output)
    x = model.get_layer("preprocess")(tf.convert_to_tensor(batch))
    with tf.GradientTape() as tape:
        conv_out = conv_model(x)
        tape.watch(conv_out)
        h = model.get_layer("gap")(conv_out)
        score = model.get_layer("pred")(h)[:, 0]        # P(pneumonia)
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8)


def overlay_heatmap(gray: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Returns an RGB image: X-ray with the heatmap blended on top."""
    base = cv2.cvtColor(cv2.resize(gray, (224, 224)), cv2.COLOR_GRAY2RGB)
    hm = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (224, 224))), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 1 - alpha, hm, alpha, 0)
