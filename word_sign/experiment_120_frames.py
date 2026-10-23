#!/usr/bin/env python3
"""Prepare and train a paired 120-frame experiment against v10 seed 44.

This script never changes the 60-frame features, v10 model, or deployed web app.
Run ``prepare``, then extract features with extract_word_landmarks.py, then ``train``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "word_dataset"
BASELINE = ROOT / "training" / "word_lstm_v10_belly_seed_sweep_20261018"
BASELINE_SPLIT = BASELINE / "seed_44" / "split.csv"
BASELINE_FEATURES = ROOT / "word_features_belly_signer_split_20261018" / "landmarks.csv"
EXPERIMENT = ROOT / "training" / "word_lstm_120f_seed44_experiment"
FEATURES = ROOT / "word_features_120f_seed44_experiment"
RESUME_PART = ROOT / "word_features_120f_seed44_resume_part2"
COMPLETE_FEATURES = ROOT / "word_features_120f_seed44_complete"
FRAMES = 120
FEATURE_COUNT = 146
SEED = 44
META_COLUMNS = {
    "sample_id", "signer_id", "label_index", "label", "frame_order",
    "source_frame", "timestamp_ms",
}


def read_split() -> list[dict[str, str]]:
    with BASELINE_SPLIT.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1574 or Counter(row["split"] for row in rows) != {
        "train": 1260, "val": 157, "test": 157,
    }:
        raise ValueError("v10 seed 44 분할이 예상한 1,574/1,260/157/157 구성이 아닙니다.")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("v10 분할에 중복 sample_id가 있습니다.")
    if {row["signer_id"] for row in rows} - {"S001", "S002", "S003", "S004"}:
        raise ValueError("v10 분할에 S005 또는 알 수 없는 촬영자가 포함되어 있습니다.")
    return rows


def prepare() -> None:
    """Write only a new metadata subset; feature extraction is a separate user step."""
    target = EXPERIMENT / "metadata_120f.csv"
    if target.exists():
        raise FileExistsError(f"기존 파일을 덮어쓰지 않습니다: {target}")
    split = read_split()
    wanted = {row["sample_id"]: row for row in split}
    source = DATASET / "metadata.csv"
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if not fieldnames or "status" not in fieldnames:
            raise ValueError("원본 metadata.csv의 헤더를 확인하세요.")
        rows = {}
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id in wanted:
                if sample_id in rows:
                    raise ValueError(f"metadata.csv의 중복 sample_id: {sample_id}")
                rows[sample_id] = row
    missing = wanted.keys() - rows.keys()
    if missing:
        raise ValueError(f"원본 metadata.csv에 없는 영상 {len(missing)}개: {sorted(missing)[:5]}")
    for sample_id, row in rows.items():
        reference = wanted[sample_id]
        if (row["signer_id"], row["label"]) != (reference["signer_id"], reference["label"]):
            raise ValueError(f"v10 분할과 촬영자/라벨 불일치: {sample_id}")
        if row["status"] != "accepted":
            raise ValueError(f"원본 metadata 상태가 accepted가 아님: {sample_id}")
        if not (DATASET / row["file_path"]).is_file():
            raise FileNotFoundError(f"원본 영상이 없음: {DATASET / row['file_path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for reference in split:
            writer.writerow(rows[reference["sample_id"]])
    print(f"준비 완료: {target} ({len(rows)}개 영상; S005 제외)")
    print("다음: README_120_FRAMES.md의 특징 추출 명령을 직접 실행하세요.")


def feature_columns(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        names = next(csv.reader(stream))
    columns = [name for name in names if name not in META_COLUMNS]
    if len(columns) != FEATURE_COUNT:
        raise ValueError(f"특징 열이 {FEATURE_COUNT}개가 아닙니다: {len(columns)}")
    return columns


def load_120f_arrays(split: list[dict[str, str]], features_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Stream CSV rows directly into preallocated arrays to limit Mac RAM use."""
    source = features_dir / "landmarks.csv"
    if not source.is_file():
        raise FileNotFoundError(f"120프레임 특징 파일이 없습니다: {source}")
    expected_columns = feature_columns(BASELINE_FEATURES)
    actual_columns = feature_columns(source)
    if actual_columns != expected_columns:
        raise ValueError("120프레임 특징 순서가 v10 60프레임 특징 순서와 다릅니다.")
    labels = json.loads((BASELINE / "labels.json").read_text(encoding="utf-8"))
    label_to_id = {label: index for index, label in enumerate(labels)}
    by_kind = {kind: [row for row in split if row["split"] == kind]
               for kind in ("train", "val", "test")}
    arrays = {}
    locations = {}
    for kind, rows in by_kind.items():
        x = np.empty((len(rows), FRAMES, FEATURE_COUNT), dtype=np.float32)
        y = np.asarray([label_to_id[row["label"]] for row in rows], dtype=np.int32)
        arrays[kind] = (x, y)
        for index, row in enumerate(rows):
            locations[row["sample_id"]] = (kind, index, row)
    counts = Counter()
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id not in locations:
                raise ValueError(f"v10 분할에 없는 특징 영상: {sample_id}")
            kind, index, reference = locations[sample_id]
            order = counts[sample_id]
            if order >= FRAMES or int(row["frame_order"]) != order:
                raise ValueError(f"프레임 개수/순서 오류: {sample_id}, {row['frame_order']}")
            if (row["signer_id"], row["label"]) != (reference["signer_id"], reference["label"]):
                raise ValueError(f"v10 분할과 특징 라벨 불일치: {sample_id}")
            values = np.asarray([float(row[name]) for name in actual_columns], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"NaN/Inf 특징값: {sample_id}, frame={order}")
            arrays[kind][0][index, order] = values
            counts[sample_id] += 1
    incomplete = [sample_id for sample_id in locations if counts[sample_id] != FRAMES]
    if incomplete:
        raise ValueError(f"120프레임이 아닌 영상 {len(incomplete)}개: {incomplete[:5]}")
    return arrays


def train(features_dir: Path) -> None:
    """Train only after the user has completed 120-frame feature extraction."""
    output = EXPERIMENT / "seed_44"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"기존 실험을 덮어쓰지 않습니다: {output}")
    split = read_split()
    arrays = load_120f_arrays(split, features_dir)

    import tensorflow as tf  # noqa: WPS433 - import only for the train command
    import train_word_lstm as base

    try:
        tf.config.set_visible_devices([], "GPU")  # Match the previous v10 CPU experiment.
    except RuntimeError:
        pass
    base.set_seed(SEED)
    labels = json.loads((BASELINE / "labels.json").read_text(encoding="utf-8"))
    x_train, y_train = arrays["train"]
    x_val, y_val = arrays["val"]
    x_test, y_test = arrays["test"]
    frequencies = Counter(y_train.tolist())
    weights = {class_id: len(y_train) / (len(labels) * count)
               for class_id, count in frequencies.items()}
    model = base.make_model(FRAMES, FEATURE_COUNT, len(labels), 0.001)
    output.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output / "best.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        tf.keras.callbacks.CSVLogger(output / "history.csv"),
    ]
    print(f"학습 입력: {x_train.shape}; 검증: {x_val.shape}; 테스트: {x_test.shape}")
    model.summary()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=180,
                        batch_size=8, callbacks=callbacks, class_weight=weights,
                        shuffle=True, verbose=2)
    best = int(np.argmin(history.history["val_loss"]))
    best_model = tf.keras.models.load_model(output / "best.keras")
    val_loss, val_acc = best_model.evaluate(x_val, y_val, batch_size=8, verbose=0)
    test_loss, test_acc = best_model.evaluate(x_test, y_test, batch_size=8, verbose=0)
    probabilities = best_model.predict(x_test, batch_size=8, verbose=0)
    predictions = probabilities.argmax(axis=1)
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for actual, predicted in zip(y_test, predictions):
        matrix[int(actual), int(predicted)] += 1
    f1_scores = []
    for class_id in range(len(labels)):
        tp = int(matrix[class_id, class_id])
        actual_count = int(matrix[class_id, :].sum())
        predicted_count = int(matrix[:, class_id].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / actual_count if actual_count else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    baseline_metrics = json.loads(
        (BASELINE / "seed_44" / "metrics.json").read_text(encoding="utf-8")
    )
    metrics = {
        "experiment": "120f_vs_v10_60f_seed44", "input_shape": [FRAMES, FEATURE_COUNT],
        "feature_source": str(features_dir.resolve()),
        "parameters": best_model.count_params(),
        "split_counts": dict(Counter(row["split"] for row in split)),
        "best_epoch": best + 1, "epochs_trained": len(history.history["loss"]),
        "train_accuracy_at_best_epoch": float(history.history["accuracy"][best]),
        "train_loss_at_best_epoch": float(history.history["loss"][best]),
        "val_accuracy": float(val_acc), "val_loss": float(val_loss),
        "test_accuracy": float(test_acc), "test_loss": float(test_loss),
        "test_correct": int(np.trace(matrix)), "macro_f1": float(np.mean(f1_scores)),
        "baseline_v10_seed44_test_accuracy": baseline_metrics["test_accuracy"],
        "baseline_v10_seed44_test_loss": baseline_metrics["test_loss"],
        "warning": "Same fixed split, but 120-frame extraction may alter landmark detection quality. "
                   "This is an offline test, not a mobile 30-FPS runtime test.",
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "config.json").write_text(
        json.dumps({"input_shape": [FRAMES, FEATURE_COUNT], "labels": labels,
                    "architecture": "GaussianNoise0.005 -> LSTM64 -> Dropout0.3 -> Dense32 -> Dense24",
                    "seed": SEED, "learning_rate": 0.001, "batch_size": 8,
                    "epochs_max": 180, "patience": 20,
                    "split_source": str(BASELINE_SPLIT)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "confusion_matrix.json").write_text(
        json.dumps({"labels": labels, "counts": matrix.tolist()}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "actual", "predicted", "confidence"])
        test_rows = [row for row in split if row["split"] == "test"]
        for row, actual, predicted, probs in zip(test_rows, y_test, predictions, probabilities):
            writer.writerow([row["sample_id"], labels[int(actual)], labels[int(predicted)],
                             float(np.max(probs))])
    print(f"120프레임 Test: {test_acc:.2%}, loss={test_loss:.4f}; "
          f"v10 60프레임 Test: {baseline_metrics['test_accuracy']:.2%}")
    print(f"결과: {output}")


def _read_resume_state() -> tuple[list[dict[str, str]], list[str], int, int]:
    """Validate the interrupted files without changing them."""
    metadata_path = EXPERIMENT / "metadata_120f.csv"
    with metadata_path.open(newline="", encoding="utf-8-sig") as stream:
        metadata_reader = csv.DictReader(stream)
        metadata_fields = metadata_reader.fieldnames
        if not metadata_fields:
            raise ValueError("120프레임 실험 메타데이터 헤더가 없습니다.")
        metadata = list(metadata_reader)
    if len(metadata) != len(read_split()):
        raise ValueError("실험 메타데이터 영상 수가 v10 분할과 다릅니다.")
    with (FEATURES / "quality_report.csv").open(newline="", encoding="utf-8-sig") as stream:
        completed = list(csv.DictReader(stream))
    if len(completed) >= len(metadata):
        raise ValueError("모든 영상이 이미 완료되어 이어하기가 필요하지 않습니다.")
    for index, row in enumerate(completed):
        if row["sample_id"] != metadata[index]["sample_id"]:
            raise ValueError(f"품질 보고서 순서가 메타데이터와 다릅니다: {index}")
    original_path = FEATURES / "landmarks.csv"
    with original_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != len(META_COLUMNS) + FEATURE_COUNT:
            raise ValueError("중단된 특징 CSV 헤더의 열 수가 다릅니다.")
        count = 0
        for row in reader:
            video_index, frame_index = divmod(count, FRAMES)
            if video_index > len(completed):
                raise ValueError("품질 보고서보다 완료 영상이 둘 이상 앞서 있습니다.")
            if len(row) != len(header) or row[0] != metadata[video_index]["sample_id"]:
                raise ValueError(f"중단된 특징 CSV의 영상/열 순서가 잘못되었습니다: {count}")
            if int(row[4]) != frame_index:
                raise ValueError(f"중단된 특징 CSV의 프레임 순서가 잘못되었습니다: {count}")
            count += 1
    if not len(completed) * FRAMES <= count <= (len(completed) + 1) * FRAMES:
        raise ValueError("품질 보고서와 특징 CSV의 완료 영상 수가 맞지 않습니다.")
    partial = count - len(completed) * FRAMES
    return metadata, metadata_fields, len(completed), partial


def resume_prepare() -> None:
    metadata, fields, completed, partial = _read_resume_state()
    target = EXPERIMENT / "metadata_remaining_120f.csv"
    if target.exists():
        raise FileExistsError(f"기존 이어하기 목록을 덮어쓰지 않습니다: {target}")
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata[completed:])
    print(f"완료: {completed}/{len(metadata)}개 영상")
    print(f"부분 기록: 다음 영상 {metadata[completed]['sample_id']}의 {partial}/{FRAMES}프레임")
    print(f"재추출 대상: {len(metadata) - completed}개 (부분 기록 영상 포함)")
    print(f"이어하기 목록: {target}")


def _validate_part2(expected_ids: list[str]) -> tuple[list[str], list[str]]:
    landmarks_path = RESUME_PART / "landmarks.csv"
    quality_path = RESUME_PART / "quality_report.csv"
    with landmarks_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        landmarks_header = next(reader)
        total = 0
        for count, row in enumerate(reader):
            index, order = divmod(count, FRAMES)
            if (index >= len(expected_ids) or len(row) != len(landmarks_header)
                    or row[0] != expected_ids[index] or int(row[4]) != order):
                raise ValueError(f"이어하기 특징 CSV 영상/프레임/열 오류: 행 {count + 2}")
            total += 1
        if total != len(expected_ids) * FRAMES:
            raise ValueError("이어하기 특징 CSV의 총 프레임 수가 부족합니다.")
    with quality_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        quality_header = reader.fieldnames
        quality_rows = list(reader)
    if not quality_header or [row["sample_id"] for row in quality_rows] != expected_ids:
        raise ValueError("이어하기 품질 보고서가 남은 영상 목록과 다릅니다.")
    return landmarks_header, quality_header


def resume_merge() -> None:
    metadata, _, completed, _ = _read_resume_state()
    remaining_ids = [row["sample_id"] for row in metadata[completed:]]
    part2_header, quality_header = _validate_part2(remaining_ids)
    if COMPLETE_FEATURES.exists():
        raise FileExistsError(f"기존 통합 결과를 덮어쓰지 않습니다: {COMPLETE_FEATURES}")
    COMPLETE_FEATURES.mkdir()
    with (FEATURES / "landmarks.csv").open(newline="", encoding="utf-8-sig") as original, \
         (RESUME_PART / "landmarks.csv").open(newline="", encoding="utf-8-sig") as rest, \
         (COMPLETE_FEATURES / "landmarks.csv").open("w", newline="", encoding="utf-8-sig") as target:
        old_reader, new_reader, writer = csv.reader(original), csv.reader(rest), csv.writer(target)
        old_header = next(old_reader)
        if old_header != next(new_reader) or old_header != part2_header:
            raise ValueError("기존 특징과 이어하기 특징의 열 순서가 다릅니다.")
        writer.writerow(old_header)
        for _ in range(completed * FRAMES):
            writer.writerow(next(old_reader))
        writer.writerows(new_reader)
    with (FEATURES / "quality_report.csv").open(newline="", encoding="utf-8-sig") as original, \
         (RESUME_PART / "quality_report.csv").open(newline="", encoding="utf-8-sig") as rest, \
         (COMPLETE_FEATURES / "quality_report.csv").open("w", newline="", encoding="utf-8-sig") as target:
        old_reader, new_reader, writer = csv.reader(original), csv.reader(rest), csv.writer(target)
        old_header = next(old_reader)
        if old_header != next(new_reader) or old_header != quality_header:
            raise ValueError("기존 품질 보고서와 이어하기 보고서의 열 순서가 다릅니다.")
        writer.writerow(old_header)
        for _ in range(completed):
            writer.writerow(next(old_reader))
        writer.writerows(new_reader)
    print(f"통합 완료: {COMPLETE_FEATURES}")
    print(f"기존 {completed}개 + 재추출 {len(remaining_ids)}개 = {len(metadata)}개")


def main() -> None:
    parser = argparse.ArgumentParser(description="v10과 짝지은 120프레임 오프라인 실험")
    parser.add_argument("command", choices=("prepare", "resume-prepare", "resume-merge", "train"))
    parser.add_argument("--features", type=Path, default=FEATURES,
                        help="train에 사용할 특징 폴더; 중단 후 통합 결과는 complete 폴더를 지정")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "resume-prepare":
        resume_prepare()
    elif args.command == "resume-merge":
        resume_merge()
    else:
        train(args.features)


if __name__ == "__main__":
    main()
