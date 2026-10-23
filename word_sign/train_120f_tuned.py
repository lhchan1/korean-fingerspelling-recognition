#!/usr/bin/env python3
"""Run isolated 120-frame regularization experiments; never overwrite baseline."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import experiment_120_frames as experiment


VARIANTS = {
    "regularized_a": {
        "units": 64, "noise": 0.01, "dropout": 0.5,
        "learning_rate": 0.0005, "max_epochs": 120, "patience": 12,
        "reduce_lr_patience": 4,
    },
    "compact_b": {
        "units": 48, "noise": 0.01, "dropout": 0.4,
        "learning_rate": 0.0005, "max_epochs": 120, "patience": 12,
        "reduce_lr_patience": 4,
    },
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def train(variant: str) -> None:
    config = VARIANTS[variant]
    output = experiment.EXPERIMENT / f"seed_44_{variant}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"기존 실험을 덮어쓰지 않습니다: {output}")
    features_dir = experiment.COMPLETE_FEATURES
    split = experiment.read_split()
    arrays = experiment.load_120f_arrays(split, features_dir)

    import tensorflow as tf
    import train_word_lstm as base

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    base.set_seed(44)
    labels = json.loads((experiment.BASELINE / "labels.json").read_text(encoding="utf-8"))
    x_train, y_train = arrays["train"]
    x_val, y_val = arrays["val"]
    x_test, y_test = arrays["test"]
    counts = Counter(y_train.tolist())
    weights = {class_id: len(y_train) / (len(labels) * count)
               for class_id, count in counts.items()}
    model = tf.keras.Sequential([
        tf.keras.layers.Input((120, 146), name="landmarks"),
        tf.keras.layers.GaussianNoise(config["noise"]),
        tf.keras.layers.LSTM(config["units"], name="lstm"),
        tf.keras.layers.Dropout(config["dropout"]),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(labels), activation="softmax", name="probabilities"),
    ], name=f"word_sign_120f_{variant}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    output.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output / "best.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["patience"],
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=config["reduce_lr_patience"], min_lr=1e-5),
        tf.keras.callbacks.CSVLogger(output / "history.csv"),
    ]
    write_json(output / "config.json", {
        "variant": variant, "input_shape": [120, 146], "parameters": model.count_params(),
        "same_split_as": str(experiment.BASELINE_SPLIT),
        "feature_source": str(features_dir), "seed": 44, "batch_size": 8, **config,
    })
    print(f"START {variant}; train={x_train.shape}, val={x_val.shape}, test={x_test.shape}", flush=True)
    model.summary()
    started = time.monotonic()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=config["max_epochs"], batch_size=8, callbacks=callbacks,
                        class_weight=weights, shuffle=True, verbose=2)
    best_epoch = int(np.argmin(history.history["val_loss"]))
    best_model = tf.keras.models.load_model(output / "best.keras")
    val_loss, val_acc = best_model.evaluate(x_val, y_val, batch_size=8, verbose=0)
    test_loss, test_acc = best_model.evaluate(x_test, y_test, batch_size=8, verbose=0)
    probs = best_model.predict(x_test, batch_size=8, verbose=0)
    predicted = probs.argmax(axis=1)
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for actual, guess in zip(y_test, predicted):
        matrix[int(actual), int(guess)] += 1
    f1_scores = []
    for i in range(len(labels)):
        tp = int(matrix[i, i])
        support = int(matrix[i, :].sum())
        predicted_count = int(matrix[:, i].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    baseline = json.loads((experiment.EXPERIMENT / "seed_44" / "metrics.json").read_text(encoding="utf-8"))
    metrics = {
        "variant": variant, "parameters": best_model.count_params(),
        "train_count": len(y_train), "val_count": len(y_val), "test_count": len(y_test),
        "epochs_trained": len(history.history["loss"]), "best_epoch": best_epoch + 1,
        "train_accuracy_at_best_epoch": float(history.history["accuracy"][best_epoch]),
        "train_loss_at_best_epoch": float(history.history["loss"][best_epoch]),
        "val_accuracy": float(val_acc), "val_loss": float(val_loss),
        "test_accuracy": float(test_acc), "test_loss": float(test_loss),
        "test_correct": int(np.trace(matrix)), "macro_f1": float(np.mean(f1_scores)),
        "baseline_120f_test_accuracy": baseline["test_accuracy"],
        "baseline_60f_test_accuracy": baseline["baseline_v10_seed44_test_accuracy"],
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "confusion_matrix.json", {"labels": labels, "counts": matrix.tolist()})
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "actual", "predicted", "confidence"])
        test_rows = [row for row in split if row["split"] == "test"]
        for row, actual, guess, probability in zip(test_rows, y_test, predicted, probs):
            writer.writerow([row["sample_id"], labels[int(actual)], labels[int(guess)],
                             float(np.max(probability))])
    print("RESULT", json.dumps(metrics, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="120프레임 모델 정규화/조기종료 실험")
    parser.add_argument("--variant", choices=VARIANTS, default="regularized_a")
    args = parser.parse_args()
    train(args.variant)


if __name__ == "__main__":
    main()
