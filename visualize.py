"""All plots. Every function saves a PNG into outputs/ (and optionally images/)."""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # works without a display (servers, Colab, CI)
import matplotlib.pyplot as plt
import numpy as np

from src.config import CLASS_NAMES, DATA_DIR, OUTPUT_DIR
from src.preprocessing import apply_clahe, load_gray


def _save(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_class_distribution(counts_by_split: dict):
    fig, ax = plt.subplots(figsize=(6, 4))
    splits = list(counts_by_split)
    w = 0.35
    for i, c in enumerate(CLASS_NAMES):
        ax.bar(np.arange(len(splits)) + i * w, [counts_by_split[s][c] for s in splits], w, label=c)
    ax.set_xticks(np.arange(len(splits)) + w / 2, splits)
    ax.set_title("Class distribution"); ax.set_ylabel("Images"); ax.legend()
    return _save(fig, "01_class_distribution.png")


def plot_sample_images(n_per_class: int = 4, seed: int = 42):
    rng = random.Random(seed)
    fig, axes = plt.subplots(2, n_per_class, figsize=(3 * n_per_class, 6.5))
    for r, cls in enumerate(CLASS_NAMES):
        files = sorted((DATA_DIR / "train" / cls).glob("*.jp*g"))
        for c, f in enumerate(rng.sample(files, n_per_class)):
            axes[r, c].imshow(load_gray(f), cmap="gray"); axes[r, c].axis("off")
            axes[r, c].set_title(cls, fontsize=10)
    fig.suptitle("Sample chest X-rays from the training set")
    return _save(fig, "02_sample_images.png")


def plot_preprocessing_steps(image_path):
    import cv2
    gray = load_gray(image_path)
    steps = [("Original", gray), ("Resized 224x224", cv2.resize(gray, (224, 224))),
             ("CLAHE (demo)", apply_clahe(cv2.resize(gray, (224, 224))))]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (t, im) in zip(axes, steps):
        ax.imshow(im, cmap="gray"); ax.set_title(t); ax.axis("off")
    return _save(fig, "03_preprocessing_steps.png")


def plot_history(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        ax.plot(history[key], label="train"); ax.plot(history["val_" + key], label="validation")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
    return _save(fig, "04_training_curves.png")


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], CLASS_NAMES); ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test set)")
    return _save(fig, "05_confusion_matrix.png")


def plot_roc(fpr, tpr, auc):
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}"); ax.plot([0, 1], [0, 1], "--", c="gray")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend()
    return _save(fig, "06_roc_curve.png")


def plot_prediction_grid(images, y_true, y_prob, threshold=0.5, n=12):
    """Predicted-vs-actual grid. Green title = correct, red = wrong."""
    idx = np.random.default_rng(0).choice(len(images), min(n, len(images)), replace=False)
    cols = 4; rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.3 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), idx):
        pred = int(y_prob[i] >= threshold)
        ax.imshow(images[i].astype("uint8"))
        ax.set_title(f"Actual: {CLASS_NAMES[int(y_true[i])]}\nPred: {CLASS_NAMES[pred]} ({y_prob[i]:.2f})",
                     fontsize=9, color="green" if pred == int(y_true[i]) else "red")
    fig.suptitle("Predicted vs actual (test set)")
    return _save(fig, "07_prediction_grid.png")
