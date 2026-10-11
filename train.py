"""Two-phase training: (1) head only, (2) fine-tune top backbone layers."""
import json

import tensorflow as tf

from src.config import (EPOCHS_FINE, EPOCHS_HEAD, MODEL_PATH, OUTPUT_DIR, SEED)
from src.data_loader import compute_class_weights, load_datasets
from src.model import build_baseline_cnn, build_transfer_model, unfreeze_for_finetuning
from src.visualize import plot_history


def _callbacks(tag):
    return [
        tf.keras.callbacks.CSVLogger(OUTPUT_DIR / f"training_log_{tag}.csv"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
    ]


def train(baseline: bool = False):
    tf.keras.utils.set_random_seed(SEED)
    train_ds, val_ds, _ = load_datasets()
    cw = compute_class_weights()
    print("Class weights:", cw)

    history = {}
    def merge(h):
        for k, v in h.history.items():
            history.setdefault(k, []).extend(v)

    if baseline:
        model = build_baseline_cnn()
        merge(model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD + EPOCHS_FINE,
                        class_weight=cw, callbacks=_callbacks("baseline")))
        out = MODEL_PATH.with_name("baseline_cnn.keras")
    else:
        model = build_transfer_model()
        print("\n=== Phase 1: training classifier head ===")
        merge(model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
                        class_weight=cw, callbacks=_callbacks("phase1")))
        print("\n=== Phase 2: fine-tuning top of MobileNetV2 ===")
        model = unfreeze_for_finetuning(model)
        merge(model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
                        class_weight=cw, callbacks=_callbacks("phase2")))
        out = MODEL_PATH

    model.summary()
    model.save(out)
    (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history)
    print(f"\nModel saved to {out}")
    return model
