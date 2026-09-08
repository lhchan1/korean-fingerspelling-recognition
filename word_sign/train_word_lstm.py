#!/usr/bin/env python3
"""Train a compact LSTM for isolated Korean sign-word classification."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


META_COLUMNS = {
    "sample_id", "signer_id", "label_index", "label", "frame_order",
    "source_frame", "timestamp_ms",
}


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="양손+Pose 단어 수어 LSTM 학습")
    parser.add_argument("--features", type=Path, default=base / "word_features" / "landmarks.csv")
    parser.add_argument("--quality", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=base / "training" / "word_lstm_v1")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--val-per-class", type=int, default=1)
    parser.add_argument("--test-per-class", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-review", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_samples(path: Path, sequence_length: int):
    with path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError("특징 CSV 헤더가 없습니다.")
        feature_columns = [name for name in reader.fieldnames if name not in META_COLUMNS]
        if len(feature_columns) != 146:
            raise ValueError(f"프레임당 특징은 146개여야 합니다: {len(feature_columns)}")
        grouped = defaultdict(list)
        for row in reader:
            grouped[row["sample_id"]].append(row)

    samples = []
    for sample_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["frame_order"]))
        if len(rows) != sequence_length:
            raise ValueError(f"{sample_id}: {len(rows)}프레임 (필요: {sequence_length})")
        sequence = np.asarray(
            [[float(row[name]) for name in feature_columns] for row in rows],
            dtype=np.float32,
        )
        if not np.isfinite(sequence).all():
            raise ValueError(f"{sample_id}: NaN/Inf 특징이 있습니다.")
        first = rows[0]
        samples.append({
            "sample_id": sample_id,
            "signer_id": first["signer_id"],
            "label_index": int(first["label_index"]),
            "label": first["label"],
            "sequence": sequence,
        })
    return samples, feature_columns


def split_samples(samples, labels, val_count, test_count, seed):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["label"]].append(sample)
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    counts = {}
    split_counts = {}
    for label in labels:
        items = sorted(grouped[label], key=lambda x: x["sample_id"])
        if len(items) <= val_count + test_count:
            raise ValueError(f"{label} 데이터가 분할하기에 부족합니다: {len(items)}")
        rng.shuffle(items)
        split["test"].extend(items[:test_count])
        split["val"].extend(items[test_count:test_count + val_count])
        split["train"].extend(items[test_count + val_count:])
        counts[label] = len(items)
        split_counts[label] = {
            "train": len(items) - val_count - test_count,
            "val": val_count,
            "test": test_count,
        }
    return split, counts, split_counts


def to_arrays(items, label_to_id):
    x = np.stack([item["sequence"] for item in items]).astype(np.float32)
    y = np.asarray([label_to_id[item["label"]] for item in items], dtype=np.int32)
    return x, y


def make_model(sequence_length: int, feature_count: int, class_count: int, lr: float):
    model = tf.keras.Sequential([
        tf.keras.layers.Input((sequence_length, feature_count), name="landmarks"),
        tf.keras.layers.GaussianNoise(0.005),
        tf.keras.layers.LSTM(64, name="lstm"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(class_count, activation="softmax", name="probabilities"),
    ], name="word_sign_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def write_csv(path: Path, header, rows) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not args.features.exists():
        raise FileNotFoundError(f"특징 CSV가 없습니다: {args.features}")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError("출력 폴더가 비어 있지 않습니다. --overwrite를 사용하세요.")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    samples, feature_columns = load_samples(args.features, args.sequence_length)
    quality_path = args.quality or args.features.with_name("quality_report.csv")
    excluded = []
    if quality_path.exists() and not args.include_review:
        with quality_path.open(newline="", encoding="utf-8-sig") as fp:
            excluded = [r["sample_id"] for r in csv.DictReader(fp) if r["result"] != "ok"]
        excluded_set = set(excluded)
        samples = [sample for sample in samples if sample["sample_id"] not in excluded_set]

    labels = sorted(
        {sample["label"] for sample in samples},
        key=lambda label: min(s["label_index"] for s in samples if s["label"] == label),
    )
    if len(labels) < 2:
        raise ValueError("학습에는 최소 2개 클래스가 필요합니다.")
    label_to_id = {label: index for index, label in enumerate(labels)}
    split, counts, split_counts = split_samples(
        samples, labels, args.val_per_class, args.test_per_class, args.seed
    )
    x_train, y_train = to_arrays(split["train"], label_to_id)
    x_val, y_val = to_arrays(split["val"], label_to_id)
    x_test, y_test = to_arrays(split["test"], label_to_id)

    train_counts = Counter(y_train.tolist())
    class_weights = {
        class_id: len(y_train) / (len(labels) * count)
        for class_id, count in train_counts.items()
    }
    model = make_model(args.sequence_length, len(feature_columns), len(labels), args.learning_rate)
    model_path = args.output / "word_sign_lstm.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(4, args.patience // 4), min_lr=1e-5
        ),
    ]
    print(f"라벨: {labels}")
    print(f"영상 수: {counts}; 분할: {split_counts}")
    print(f"입력: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")
    model.summary()
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs,
        batch_size=args.batch_size, callbacks=callbacks, class_weight=class_weights,
        shuffle=True, verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for actual, predicted in zip(y_test, predictions):
        matrix[int(actual), int(predicted)] += 1
    model.save(model_path)

    (args.output / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = {
        "sequence_length": args.sequence_length,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "normalization": "shoulder midpoint origin; divide xyz by 3D shoulder distance",
        "labels": labels,
        "label_to_id": label_to_id,
        "videos_per_class": counts,
        "split_per_class": split_counts,
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "quality_excluded_samples": excluded,
        "seed": args.seed,
    }
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_videos": len(y_test),
        "epochs_trained": len(history.history["loss"]),
        "warning": "Single-signer random video split; this is not signer-independent accuracy.",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(
        args.output / "split.csv",
        ["split", "sample_id", "signer_id", "label", "frames"],
        ([name, x["sample_id"], x["signer_id"], x["label"], len(x["sequence"])]
         for name in ("train", "val", "test") for x in split[name]),
    )
    write_csv(
        args.output / "history.csv",
        ["epoch", *history.history.keys()],
        ([epoch + 1, *[history.history[k][epoch] for k in history.history]]
         for epoch in range(len(history.history["loss"]))),
    )
    write_csv(
        args.output / "confusion_matrix.csv",
        ["actual\\predicted", *labels],
        ([label, *row.tolist()] for label, row in zip(labels, matrix)),
    )
    write_csv(
        args.output / "test_predictions.csv",
        ["sample_id", "actual", "predicted", *[f"prob_{x}" for x in labels]],
        ([item["sample_id"], labels[int(actual)], labels[int(predicted)], *probs.tolist()]
         for item, actual, predicted, probs in zip(split["test"], y_test, predictions, probabilities)),
    )
    print("테스트 혼동행렬 (행=실제, 열=예측)")
    print(matrix)
    print(f"테스트 정확도: {test_accuracy:.1%} ({len(y_test)}개 영상)")
    print(f"저장 위치: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
