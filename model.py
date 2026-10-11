"""Model definitions: MobileNetV2 transfer learning (default) and a small baseline CNN."""
import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import FINE_TUNE_FROM, IMG_SIZE, LR_FINE, LR_HEAD


def _augmentation():
    # Mild, anatomically plausible augmentation. No horizontal flip: the heart sits
    # on the left side of the image, so flipping creates unrealistic anatomy.
    return models.Sequential([
        layers.RandomRotation(0.04),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")


def _compile(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")])
    return model


def build_transfer_model(pretrained: bool = True):
    """Input: float images in [0,255]. Output: P(pneumonia)."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False,
        weights="imagenet" if pretrained else None)
    base.trainable = False                       # phase 1: backbone frozen

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = _augmentation()(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1, name="preprocess")(x)   # -> [-1, 1]
    x = base(x, training=False)                  # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)
    return _compile(models.Model(inputs, outputs, name="pneumonia_mobilenetv2"), LR_HEAD)


def get_backbone(model):
    return next(l for l in model.layers if isinstance(l, tf.keras.Model)
                and l.name.startswith("mobilenet"))


def unfreeze_for_finetuning(model):
    """Phase 2: unfreeze the top of the backbone, recompile with a tiny learning rate."""
    base = get_backbone(model)
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_FROM]:
        layer.trainable = False
    return _compile(model, LR_FINE)


def build_baseline_cnn():
    """Small CNN from scratch (for comparison; expect lower performance)."""
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = _augmentation()(inputs)
    x = layers.Rescaling(1 / 255.0, name="preprocess")(x)
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)
    return _compile(models.Model(inputs, outputs, name="baseline_cnn"), 1e-3)
