"""Builds train / validation / test tf.data pipelines and class weights."""
from pathlib import Path

import tensorflow as tf

from src.config import (BATCH_SIZE, CLASS_NAMES, DATA_DIR, IMG_SIZE, SEED, VAL_SPLIT)


def _check(data_dir: Path):
    if not (data_dir / "train").exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. See data/README.md for download steps.")


def load_datasets(data_dir: Path = DATA_DIR):
    """Returns train_ds, val_ds, test_ds. Test set is NOT shuffled (needed for metrics)."""
    _check(data_dir)
    common = dict(class_names=CLASS_NAMES, color_mode="rgb", label_mode="binary",
                  image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train", validation_split=VAL_SPLIT, subset="training", seed=SEED, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train", validation_split=VAL_SPLIT, subset="validation", seed=SEED, **common)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test", shuffle=False, **common)
    auto = tf.data.AUTOTUNE
    return train_ds.prefetch(auto), val_ds.prefetch(auto), test_ds.prefetch(auto)


def count_images(split: str, data_dir: Path = DATA_DIR) -> dict:
    exts = {".jpeg", ".jpg", ".png"}
    return {c: sum(1 for p in (data_dir / split / c).iterdir() if p.suffix.lower() in exts)
            for c in CLASS_NAMES}


def compute_class_weights(data_dir: Path = DATA_DIR) -> dict:
    """The dataset has ~3x more PNEUMONIA than NORMAL; weights stop the model
    from simply predicting 'PNEUMONIA' for everything."""
    counts = count_images("train", data_dir)
    total = sum(counts.values())
    return {i: total / (len(CLASS_NAMES) * counts[c]) for i, c in enumerate(CLASS_NAMES)}
