#!/usr/bin/env python3
"""Signer-level cross-validation and final evaluation for word-sign LSTM."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from train_word_lstm import load_samples, make_model, set_seed, to_arrays, write_csv


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="촬영자 독립 교차검증 및 최종 LSTM 학습")
    parser.add_argument("--mode", choices=("cv", "final", "all"), default="cv")
    parser.add_argument("--features", type=Path, default=base / "word_features" / "landmarks.csv")
    parser.add_argument("--quality", type=Path, default=base / "word_features" / "quality_report.csv")
    parser.add_argument("--output", type=Path, default=base / "training" / "word_lstm_loso")
    parser.add_argument("--cv-signers", nargs="+", default=["S001", "S002", "S003"])
    parser.add_argument("--final-test-signer", default="S004")
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-exclude", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_filtered_samples(args):
    samples, feature_columns = load_samples(args.features, args.sequence_length)
    excluded = []
    if args.quality.exists() and not args.include_exclude:
        with args.quality.open(newline="", encoding="utf-8-sig") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames and "quality_grade" in reader.fieldnames:
                excluded = [row["sample_id"] for row in reader if row["quality_grade"] == "exclude"]
            else:
                excluded = [row["sample_id"] for row in reader if row["result"] != "ok"]
        excluded_set = set(excluded)
        samples = [sample for sample in samples if sample["sample_id"] not in excluded_set]
    return samples, feature_columns, excluded


def validate_setup(samples, cv_signers, test_signer, labels):
    cv_signers = list(dict.fromkeys(cv_signers))
    if len(cv_signers) < 3:
        raise ValueError("촬영자 교차검증에는 최소 3명의 --cv-signers가 필요합니다.")
    if test_signer in cv_signers:
        raise ValueError("최종 테스트 촬영자는 교차검증 촬영자와 겹칠 수 없습니다.")
    known = {sample["signer_id"] for sample in samples}
    unknown = set(cv_signers + [test_signer]) - known
    if unknown:
        raise ValueError(f"특징 CSV에 없는 촬영자입니다: {sorted(unknown)}")
    for signer in cv_signers + [test_signer]:
        signer_labels = {sample["label"] for sample in samples if sample["signer_id"] == signer}
        missing = [label for label in labels if label not in signer_labels]
        if missing:
            raise ValueError(f"{signer}에 없는 클래스: {missing}")
    return cv_signers


def class_weights(items, label_to_id):
    counts = Counter(label_to_id[item["label"]] for item in items)
    return {
        class_id: len(items) / (len(label_to_id) * count)
        for class_id, count in counts.items()
    }


def evaluate(model, items, label_to_id, labels):
    x, y = to_arrays(items, label_to_id)
    loss, accuracy = model.evaluate(x, y, verbose=0)
    probabilities = model.predict(x, verbose=0)
    predictions = probabilities.argmax(axis=1)
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for actual, predicted in zip(y, predictions):
        matrix[int(actual), int(predicted)] += 1

    per_class = []
    for class_id, label in enumerate(labels):
        tp = int(matrix[class_id, class_id])
        support = int(matrix[class_id, :].sum())
        predicted_count = int(matrix[:, class_id].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append((label, precision, recall, f1, support))
    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean([row[1] for row in per_class])),
        "macro_recall": float(np.mean([row[2] for row in per_class])),
        "macro_f1": float(np.mean([row[3] for row in per_class])),
        "videos": len(items),
    }
    prediction_rows = [
        [item["sample_id"], item["signer_id"], labels[int(actual)], labels[int(predicted)],
         *probs.tolist()]
        for item, actual, predicted, probs in zip(items, y, predictions, probabilities)
    ]
    return metrics, matrix, per_class, prediction_rows


def save_evaluation(output, labels, metrics, matrix, per_class, prediction_rows):
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "confusion_matrix.csv", ["actual\\predicted", *labels],
              ([label, *row.tolist()] for label, row in zip(labels, matrix)))
    write_csv(output / "per_class_metrics.csv", ["label", "precision", "recall", "f1", "support"],
              per_class)
    write_csv(output / "predictions.csv",
              ["sample_id", "signer_id", "actual", "predicted", *[f"prob_{x}" for x in labels]],
              prediction_rows)
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def train_fold(args, fold_number, val_signer, train_items, val_items, labels, label_to_id, output):
    set_seed(args.seed + fold_number)
    x_train, y_train = to_arrays(train_items, label_to_id)
    x_val, y_val = to_arrays(val_items, label_to_id)
    model = make_model(args.sequence_length, x_train.shape[2], len(labels), args.learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(4, args.patience // 4), min_lr=1e-5
        ),
    ]
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs,
        batch_size=args.batch_size, callbacks=callbacks,
        class_weight=class_weights(train_items, label_to_id), shuffle=True, verbose=2,
    )
    fold_dir = output / f"fold_{val_signer}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model.save(fold_dir / "word_sign_lstm.keras")
    write_csv(
        fold_dir / "history.csv", ["epoch", *history.history.keys()],
        ([epoch + 1, *[history.history[key][epoch] for key in history.history]]
         for epoch in range(len(history.history["loss"]))),
    )
    metrics, matrix, per_class, prediction_rows = evaluate(model, val_items, label_to_id, labels)
    best_index = int(np.argmin(history.history["val_loss"]))
    metrics.update({
        "fold": fold_number,
        "validation_signer": val_signer,
        "training_signers": sorted({item["signer_id"] for item in train_items}),
        "train_videos": len(train_items),
        "best_epoch": best_index + 1,
        "best_val_loss": float(history.history["val_loss"][best_index]),
        "epochs_trained": len(history.history["loss"]),
    })
    save_evaluation(fold_dir, labels, metrics, matrix, per_class, prediction_rows)
    return metrics


def run_cv(args, samples, labels, label_to_id, output):
    summaries = []
    for fold_number, val_signer in enumerate(args.cv_signers, 1):
        train_signers = set(args.cv_signers) - {val_signer}
        train_items = [item for item in samples if item["signer_id"] in train_signers]
        val_items = [item for item in samples if item["signer_id"] == val_signer]
        print(f"\n=== Fold {fold_number}: train={sorted(train_signers)}, val={val_signer} ===")
        summaries.append(train_fold(
            args, fold_number, val_signer, train_items, val_items, labels, label_to_id, output
        ))
    recommended_epoch = max(1, int(round(statistics.median(row["best_epoch"] for row in summaries))))
    summary = {
        "cv_signers": args.cv_signers,
        "folds": summaries,
        "mean_accuracy": float(np.mean([row["accuracy"] for row in summaries])),
        "std_accuracy": float(np.std([row["accuracy"] for row in summaries])),
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in summaries])),
        "std_macro_f1": float(np.std([row["macro_f1"] for row in summaries])),
        "recommended_final_epochs": recommended_epoch,
    }
    (output / "cv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(
        output / "cv_summary.csv",
        ["fold", "validation_signer", "training_signers", "accuracy", "macro_f1",
         "best_epoch", "epochs_trained", "train_videos", "videos"],
        ([row["fold"], row["validation_signer"], "+".join(row["training_signers"]),
          row["accuracy"], row["macro_f1"], row["best_epoch"], row["epochs_trained"],
          row["train_videos"], row["videos"]] for row in summaries),
    )
    print(f"교차검증 평균 정확도: {summary['mean_accuracy']:.1%}")
    print(f"교차검증 평균 Macro F1: {summary['mean_macro_f1']:.1%}")
    print(f"권장 최종 epoch: {recommended_epoch}")
    return summary


def run_final(args, samples, feature_count, labels, label_to_id, excluded, output, final_epochs):
    train_items = [item for item in samples if item["signer_id"] in set(args.cv_signers)]
    test_items = [item for item in samples if item["signer_id"] == args.final_test_signer]
    set_seed(args.seed)
    x_train, y_train = to_arrays(train_items, label_to_id)
    model = make_model(args.sequence_length, feature_count, len(labels), args.learning_rate)
    print(f"\n=== Final: train={args.cv_signers}, test={args.final_test_signer}, epochs={final_epochs} ===")
    history = model.fit(
        x_train, y_train, epochs=final_epochs, batch_size=args.batch_size,
        class_weight=class_weights(train_items, label_to_id), shuffle=True, verbose=2,
    )
    final_dir = output / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save(final_dir / "word_sign_lstm.keras")
    write_csv(
        final_dir / "history.csv", ["epoch", *history.history.keys()],
        ([epoch + 1, *[history.history[key][epoch] for key in history.history]]
         for epoch in range(len(history.history["loss"]))),
    )
    metrics, matrix, per_class, prediction_rows = evaluate(model, test_items, label_to_id, labels)
    metrics.update({
        "split_strategy": "train_all_cv_signers_test_unseen_signer",
        "training_signers": args.cv_signers,
        "test_signer": args.final_test_signer,
        "train_videos": len(train_items),
        "final_epochs": final_epochs,
        "parameter_count": model.count_params(),
    })
    save_evaluation(final_dir, labels, metrics, matrix, per_class, prediction_rows)
    write_csv(
        final_dir / "split.csv", ["split", "sample_id", "signer_id", "label", "frames"],
        ([name, item["sample_id"], item["signer_id"], item["label"], len(item["sequence"])]
         for name, items in (("train", train_items), ("test", test_items)) for item in items),
    )
    (final_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = {
        "sequence_length": args.sequence_length,
        "feature_count": feature_count,
        "labels": labels,
        "label_to_id": label_to_id,
        "training_signers": args.cv_signers,
        "test_signer": args.final_test_signer,
        "train_videos": len(train_items),
        "test_videos": len(test_items),
        "final_epochs": final_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "quality_excluded_samples": excluded,
        "parameter_count": model.count_params(),
    }
    (final_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"최종 S004 정확도: {metrics['accuracy']:.1%}")
    print(f"최종 S004 Macro F1: {metrics['macro_f1']:.1%}")
    return metrics


def main() -> int:
    args = parse_args()
    if not args.features.exists():
        raise FileNotFoundError(f"특징 CSV가 없습니다: {args.features}")
    if args.mode == "final" and not args.final_epochs:
        raise ValueError("final 모드에는 --final-epochs가 필요합니다.")
    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            raise FileExistsError("출력 폴더가 비어 있지 않습니다. --overwrite를 사용하세요.")
    args.output.mkdir(parents=True, exist_ok=True)

    samples, feature_columns, excluded = load_filtered_samples(args)
    labels = sorted(
        {sample["label"] for sample in samples},
        key=lambda label: min(s["label_index"] for s in samples if s["label"] == label),
    )
    label_to_id = {label: index for index, label in enumerate(labels)}
    args.cv_signers = validate_setup(
        samples, args.cv_signers, args.final_test_signer, labels
    )
    selected = set(args.cv_signers + [args.final_test_signer])
    samples = [sample for sample in samples if sample["signer_id"] in selected]
    print(f"클래스: {len(labels)}개; 제외 영상: {len(excluded)}개")
    print(f"교차검증 촬영자: {args.cv_signers}; 최종 테스트 촬영자: {args.final_test_signer}")

    summary = None
    if args.mode in {"cv", "all"}:
        summary = run_cv(args, samples, labels, label_to_id, args.output)
    if args.mode in {"final", "all"}:
        final_epochs = args.final_epochs or summary["recommended_final_epochs"]
        run_final(
            args, samples, len(feature_columns), labels, label_to_id, excluded,
            args.output, final_epochs,
        )
    print(f"저장 위치: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
