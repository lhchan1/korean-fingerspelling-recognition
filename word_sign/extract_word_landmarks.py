#!/usr/bin/env python3
"""Extract two-hand and upper-body pose sequences for sign-word videos."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from contextlib import ExitStack
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


HAND_POINTS = 21
POSE_INDICES = (11, 12, 13, 14, 15, 16)  # shoulders, elbows, wrists
TARGET_FRAMES = 45


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="단어 영상에서 양손+상체 특징 추출")
    parser.add_argument("--dataset", type=Path, default=base / "word_dataset")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--hand-model", type=Path,
                        default=base.parent / "models" / "hand_landmarker.task")
    parser.add_argument("--pose-model", type=Path,
                        default=base.parent / "models" / "pose_landmarker_lite.task")
    parser.add_argument("--output", type=Path, default=base / "word_features")
    parser.add_argument("--frames", type=int, default=TARGET_FRAMES)
    parser.add_argument("--cpu", action="store_true",
                        help="macOS Metal 충돌을 피하도록 MediaPipe CPU delegate 사용")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def base_options(path: Path, use_cpu: bool):
    options = {"model_asset_path": str(path)}
    if use_cpu:
        options["delegate"] = mp.tasks.BaseOptions.Delegate.CPU
    return mp.tasks.BaseOptions(**options)


def hand_landmarker(path: Path, use_cpu: bool = False):
    return mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options(path, use_cpu),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def pose_landmarker(path: Path, use_cpu: bool = False):
    return mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options(path, use_cpu),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def frame_indices(total: int, target: int) -> list[int]:
    if total <= 0: return []
    return np.rint(np.linspace(0, total - 1, target)).astype(int).tolist()


def handedness_name(category) -> str:
    return str(getattr(category, "category_name", "") or getattr(category, "display_name", ""))


def vector(point, center: np.ndarray, scale: float) -> list[float]:
    return [(float(point.x) - center[0]) / scale,
            (float(point.y) - center[1]) / scale,
            (float(point.z) - center[2]) / scale]


def features_for_frame(hand_result, pose_result) -> tuple[list[float], bool, bool, bool]:
    pose_ok = bool(pose_result.pose_landmarks)
    if pose_ok:
        pose = pose_result.pose_landmarks[0]
        left_shoulder, right_shoulder = pose[11], pose[12]
        center = np.asarray([(left_shoulder.x + right_shoulder.x) / 2,
                             (left_shoulder.y + right_shoulder.y) / 2,
                             (left_shoulder.z + right_shoulder.z) / 2], dtype=np.float32)
        scale = math.sqrt((left_shoulder.x-right_shoulder.x)**2 +
                          (left_shoulder.y-right_shoulder.y)**2 +
                          (left_shoulder.z-right_shoulder.z)**2)
        if scale < 1e-8: scale = 1.0
        pose_values = [value for idx in POSE_INDICES for value in vector(pose[idx], center, scale)]
    else:
        center, scale = np.zeros(3, dtype=np.float32), 1.0
        pose_values = [0.0] * (len(POSE_INDICES) * 3)

    hands = {"Left": None, "Right": None}
    for landmarks, categories in zip(hand_result.hand_landmarks, hand_result.handedness):
        if categories:
            name = handedness_name(categories[0]).title()
            if name in hands and hands[name] is None:
                hands[name] = landmarks
    values = []
    for name in ("Left", "Right"):
        landmarks = hands[name]
        # Body-relative normalization is undefined without the shoulders.
        # Treat the hand as missing instead of mixing absolute image coordinates
        # with shoulder-normalized coordinates.
        if landmarks is None or not pose_ok:
            values.extend([0.0] * (HAND_POINTS * 3))
        else:
            values.extend(v for point in landmarks for v in vector(point, center, scale))
    left_ok = pose_ok and hands["Left"] is not None
    right_ok = pose_ok and hands["Right"] is not None
    values.extend(pose_values)
    values.extend([float(left_ok), float(right_ok)])
    if len(values) != 146:
        raise AssertionError(f"특징 길이는 146이어야 합니다: {len(values)}")
    return values, left_ok, right_ok, pose_ok


def fields() -> list[str]:
    xyz = [f"{side}_{axis}{i}" for side in ("left", "right")
           for i in range(HAND_POINTS) for axis in "xyz"]
    pose = [f"pose_{idx}_{axis}" for idx in POSE_INDICES for axis in "xyz"]
    return ["sample_id", "signer_id", "label_index", "label", "frame_order",
            "source_frame", "timestamp_ms"] + xyz + pose + ["left_detected", "right_detected"]


def main() -> int:
    args = parse_args()
    metadata = (args.metadata or args.dataset / "metadata.csv").resolve()
    for path in (metadata, args.hand_model, args.pose_model):
        if not path.exists():
            raise FileNotFoundError(f"필요한 파일이 없습니다: {path}")
    if args.frames < 2: raise ValueError("--frames는 2 이상이어야 합니다.")
    args.output.mkdir(parents=True, exist_ok=True)
    landmarks_path, quality_path = args.output / "landmarks.csv", args.output / "quality_report.csv"
    if not args.overwrite and (landmarks_path.exists() or quality_path.exists()):
        raise FileExistsError("출력 파일이 있습니다. --overwrite를 사용하세요.")
    with metadata.open(newline="", encoding="utf-8-sig") as fp:
        rows = [r for r in csv.DictReader(fp) if r.get("status", "accepted") == "accepted"]

    with landmarks_path.open("w", newline="", encoding="utf-8-sig") as lf, \
         quality_path.open("w", newline="", encoding="utf-8-sig") as qf, ExitStack() as stack:
        writer = csv.DictWriter(lf, fieldnames=fields()); writer.writeheader()
        qfields = ["sample_id", "signer_id", "label_index", "label", "file_path",
                   "duration_sec", "sampled_frames", "left_rate", "right_rate",
                   "active_hand_rate", "pose_rate", "quality_grade", "result"]
        qwriter = csv.DictWriter(qf, fieldnames=qfields); qwriter.writeheader()
        hands = stack.enter_context(hand_landmarker(args.hand_model, args.cpu))
        pose = stack.enter_context(pose_landmarker(args.pose_model, args.cpu))
        for n, row in enumerate(rows, 1):
            path = args.dataset / row["file_path"]
            cap = cv2.VideoCapture(str(path)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not cap.isOpened() or total <= 0 or fps <= 0:
                print(f"[WARN] 읽을 수 없는 영상: {path}", file=sys.stderr); cap.release(); continue
            left_count = right_count = pose_count = 0
            for order, source_idx in enumerate(frame_indices(total, args.frames)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, source_idx); ok, bgr = cap.read()
                if not ok: continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hr, pr = hands.detect(image), pose.detect(image)
                values, left_ok, right_ok, pose_ok = features_for_frame(hr, pr)
                left_count += left_ok; right_count += right_ok; pose_count += pose_ok
                base = {"sample_id": row["sample_id"], "signer_id": row["signer_id"],
                        "label_index": row["label_index"], "label": row["label"],
                        "frame_order": order, "source_frame": source_idx,
                        "timestamp_ms": round(source_idx / fps * 1000, 3)}
                writer.writerow(dict(zip(fields(), list(base.values()) + values)))
            cap.release()
            denom = args.frames
            label = row["label"]
            pose_rate = pose_count / denom
            active_hand_rate = max(left_count, right_count) / denom
            # IDLE may correctly contain no visible hands. OTHER may also be a
            # non-sign body movement, so pose visibility is its core requirement.
            if label in {"IDLE", "OTHER"}:
                quality_ok = pose_rate >= .8
                quality_grade = "usable" if quality_ok else "exclude"
            else:
                quality_ok = pose_rate >= .8 and active_hand_rate >= .4
                if pose_rate < .8 or active_hand_rate < .2:
                    quality_grade = "exclude"
                elif active_hand_rate < .4:
                    quality_grade = "caution"
                else:
                    quality_grade = "usable"
            qwriter.writerow({"sample_id": row["sample_id"], "signer_id": row["signer_id"],
                              "label_index": row["label_index"], "label": row["label"],
                              "file_path": row["file_path"], "duration_sec": row.get("duration_sec", ""),
                              "sampled_frames": denom, "left_rate": round(left_count/denom, 4),
                              "right_rate": round(right_count/denom, 4),
                              "active_hand_rate": round(active_hand_rate, 4),
                              "pose_rate": round(pose_count/denom, 4), "quality_grade": quality_grade,
                              "result": "ok" if quality_ok else "review"})
            print(f"[{n}/{len(rows)}] {row['sample_id']}")
    print(f"Saved: {landmarks_path}\nSaved: {quality_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr); raise SystemExit(1)
