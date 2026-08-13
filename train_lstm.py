#!/usr/bin/env python3
"""Train and evaluate a small LSTM from extracted hand-landmark sequences."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


LANDMARK_COUNT = 21
FEATURE_COLUMNS = [
    f"norm_{axis}{idx}" for idx in range(LANDMARK_COUNT) for axis in "xyz"
]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="MediaPipe 특징으로 지문자 LSTM 학습")
    parser.add_argument(
        "--features",
        type=Path,
        default=project_dir / "features" / "landmarks.csv",
    )
    parser.add_argument(
        "--quality",
        type=Path,
        default=None,
        help="품질 보고서. 기본값은 landmarks.csv 옆 quality_report.csv",
    )
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="quality_report에서 review/error인 영상도 포함",
    )
    parser.add_argument(
        "--min-detection-rate",
        type=float,
        default=0.5,
        help="이 검출률 미만 영상은 학습에서 제외 (기본 0.5)",
    )
    parser.add_argument(
        "--output", type=Path, default=project_dir / "training" / "lstm_v6"
    )
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument(
        "--per-class",
        type=int,
        default=0,
        help="클래스당 최대 영상 수. 0이면 각 클래스의 모든 영상 사용",
    )
    parser.add_argument("--val-per-class", type=int, default=1)
    parser.add_argument("--test-per-class", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
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


def load_sequences(path: Path, selected: set[str] | None):
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            if selected is None or row["label"] in selected:
                grouped[row["sample_id"]].append(row)

    samples: list[dict[str, object]] = []
    for sample_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["frame_index"]))
        sequence = np.asarray(
            [[float(row[column]) for column in FEATURE_COLUMNS] for row in rows],
            dtype=np.float32,
        )
        if len(sequence) == 0 or not np.isfinite(sequence).all():
            print(f"[WARN] 잘못된 특징 영상 제외: {sample_id}", file=sys.stderr)
            continue
        first = rows[0]
        samples.append(
            {
                "sample_id": sample_id,
                "label": first["label"],
                "label_index": int(first["label_index"]),
                "take": int(first["take"]),
                "signer_id": first["signer_id"],
                "sequence": sequence,
            }
        )
    return samples


def resample(sequence: np.ndarray, length: int) -> np.ndarray:
    if len(sequence) == length:
        return sequence
    positions = np.linspace(0, len(sequence) - 1, length)
    left = np.floor(positions).astype(int)
    right = np.ceil(positions).astype(int)
    weights = (positions - left).astype(np.float32)[:, None]
    return sequence[left] * (1.0 - weights) + sequence[right] * weights


def split_dataset(samples, labels, per_class, val_count, test_count, seed):
    by_label = defaultdict(list)
    for sample in samples:
        by_label[sample["label"]].append(sample)
    counts = {label: len(by_label[label]) for label in labels}
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"데이터가 없는 라벨이 있습니다: {counts}")
    if per_class and per_class > min(counts.values()):
        raise ValueError(f"--per-class={per_class}이 최소 클래스 영상 수보다 큽니다: {counts}")
    minimum_used = per_class or min(counts.values())
    if minimum_used <= val_count + test_count:
        raise ValueError("클래스당 영상 수가 검증+테스트 영상 수보다 커야 합니다.")

    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    used_counts = {}
    split_counts = {}
    for label in labels:
        candidates = sorted(
            by_label[label], key=lambda item: (item["take"], item["sample_id"])
        )
        # Keep balancing deterministic but avoid always selecting only early takes.
        rng.shuffle(candidates)
        chosen = candidates[:per_class] if per_class else candidates
        used_counts[label] = len(chosen)
        split["test"].extend(chosen[:test_count])
        split["val"].extend(chosen[test_count : test_count + val_count])
        split["train"].extend(chosen[test_count + val_count :])
        split_counts[label] = {
            "train": len(chosen) - val_count - test_count,
            "val": val_count,
            "test": test_count,
        }
    return split, counts, used_counts, split_counts


def arrays(items, label_to_id, sequence_length):
    x = np.stack([resample(item["sequence"], sequence_length) for item in items])
    y = np.asarray([label_to_id[item["label"]] for item in items], dtype=np.int32)
    return x, y


def make_model(sequence_length: int, feature_count: int, class_count: int, lr: float):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, feature_count), name="landmarks"),
            tf.keras.layers.LSTM(64, name="lstm"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(class_count, activation="softmax", name="probabilities"),
        ],
        name="fingerspelling_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.int32)
    for true_id, pred_id in zip(y_true, y_pred):
        matrix[int(true_id), int(pred_id)] += 1
    return matrix


def write_split(path: Path, split) -> None:
    fields = ["split", "sample_id", "signer_id", "label", "take", "frames"]
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for split_name in ("train", "val", "test"):
            for item in split[split_name]:
                writer.writerow(
                    {
                        "split": split_name,
                        "sample_id": item["sample_id"],
                        "signer_id": item["signer_id"],
                        "label": item["label"],
                        "take": item["take"],
                        "frames": len(item["sequence"]),
                    }
                )


def main() -> int:
    args = parse_args()
    if args.sequence_length < 2:
        raise ValueError("--sequence-length는 2 이상이어야 합니다.")
    if not args.features.exists():
        raise FileNotFoundError(f"특징 CSV가 없습니다: {args.features}")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError("출력 폴더가 비어 있지 않습니다. --overwrite를 사용하세요.")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    selected = set(args.labels) if args.labels else None
    samples = load_sequences(args.features, selected)
    quality_path = args.quality or args.features.with_name("quality_report.csv")
    excluded_quality: list[str] = []
    if quality_path.exists() and not args.include_review:
        with quality_path.open("r", newline="", encoding="utf-8-sig") as fp:
            excluded_quality = []
            for row in csv.DictReader(fp):
                rate = float(row.get("detection_rate") or 0.0)
                if row.get("result") == "error" or rate < args.min_detection_rate:
                    excluded_quality.append(row["sample_id"])
        excluded_set = set(excluded_quality)
        samples = [sample for sample in samples if sample["sample_id"] not in excluded_set]
        if excluded_quality:
            print(f"품질 기준으로 제외: {excluded_quality}")
    if not samples:
        raise ValueError("학습할 시퀀스가 없습니다.")
    labels = sorted({sample["label"] for sample in samples}, key=lambda label: min(
        sample["label_index"] for sample in samples if sample["label"] == label
    ))
    if len(labels) < 2:
        raise ValueError("최소 2개 라벨이 필요합니다.")
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    split, original_counts, used_counts, split_counts = split_dataset(
        samples,
        labels,
        args.per_class,
        args.val_per_class,
        args.test_per_class,
        args.seed,
    )
    x_train, y_train = arrays(split["train"], label_to_id, args.sequence_length)
    x_val, y_val = arrays(split["val"], label_to_id, args.sequence_length)
    x_test, y_test = arrays(split["test"], label_to_id, args.sequence_length)

    train_counts = Counter(y_train.tolist())
    class_weights = {
        class_id: len(y_train) / (len(labels) * count)
        for class_id, count in train_counts.items()
    }

    model = make_model(args.sequence_length, len(FEATURE_COLUMNS), len(labels), args.learning_rate)
    best_path = args.output / "fingerspelling_lstm.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            best_path, monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(3, args.patience // 3), min_lr=1e-5
        ),
    ]
    print(f"라벨: {labels}")
    print(f"원본 영상 수: {original_counts}; 사용 영상 수: {used_counts}")
    print(f"입력: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")
    print(f"클래스 가중치: {class_weights}")
    model.summary()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
        shuffle=True,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y_test, predictions, len(labels))

    model.save(best_path)
    (args.output / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = {
        "sequence_length": args.sequence_length,
        "feature_columns": FEATURE_COLUMNS,
        "normalization": "wrist origin; divide xyz by max wrist-to-MCP 3D distance",
        "seed": args.seed,
        "labels": labels,
        "label_to_id": label_to_id,
        "used_videos_per_class": used_counts,
        "split_per_class": split_counts,
        "class_weights": {str(key): value for key, value in class_weights.items()},
        "quality_excluded_samples": excluded_quality,
        "min_detection_rate": args.min_detection_rate,
    }
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs_trained": len(history.history["loss"]),
        "warning": "Single-signer pilot result; not a signer-independent evaluation.",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_split(args.output / "split.csv", split)

    with (args.output / "history.csv").open("w", newline="", encoding="utf-8-sig") as fp:
        keys = list(history.history)
        writer = csv.DictWriter(fp, fieldnames=["epoch"] + keys)
        writer.writeheader()
        for epoch in range(len(history.history[keys[0]])):
            row = {"epoch": epoch + 1}
            row.update({key: history.history[key][epoch] for key in keys})
            writer.writerow(row)

    with (args.output / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fp:
        writer = csv.writer(fp)
        writer.writerow(["actual\\predicted"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row.tolist())

    with (args.output / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fp:
        writer = csv.writer(fp)
        writer.writerow(["sample_id", "actual", "predicted", *[f"prob_{x}" for x in labels]])
        for item, true_id, pred_id, probs in zip(split["test"], y_test, predictions, probabilities):
            writer.writerow(
                [item["sample_id"], labels[int(true_id)], labels[int(pred_id)], *probs.tolist()]
            )

    print("\n테스트 혼동행렬 (행=실제, 열=예측)")
    print(matrix)
    print(f"테스트 정확도: {test_accuracy:.1%} ({len(y_test)}개 영상)")
    print(f"저장 위치: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
