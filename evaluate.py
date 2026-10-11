"""Evaluates on the held-out test set and saves metrics + figures."""
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

from src.config import CLASS_NAMES, MODEL_PATH, OUTPUT_DIR, THRESHOLD
from src.data_loader import load_datasets
from src.visualize import plot_confusion_matrix, plot_prediction_grid, plot_roc


def evaluate(model_path=MODEL_PATH):
    model = tf.keras.models.load_model(model_path)
    _, _, test_ds = load_datasets()

    # test_ds is NOT shuffled, so predictions line up with labels.
    images = np.concatenate([x.numpy() for x, _ in test_ds])
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).ravel().astype(int)
    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob >= THRESHOLD).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall_sensitivity": recall_score(y_true, y_pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "threshold": THRESHOLD,
        "n_test": int(len(y_true)),
    }
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print(json.dumps(metrics, indent=2))
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_confusion_matrix(cm)
    plot_roc(fpr, tpr, auc)
    plot_prediction_grid(images, y_true, y_prob, THRESHOLD)
    return metrics
