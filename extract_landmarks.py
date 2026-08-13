#!/usr/bin/env python3
"""Extract MediaPipe hand landmarks from accepted fingerspelling videos."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import mediapipe as mp


LANDMARK_COUNT = 21
MCP_INDICES = (5, 9, 13, 17)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="영상에서 MediaPipe 손 특징 추출")
    parser.add_argument("--dataset", type=Path, default=project_dir / "dataset")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--model",
        type=Path,
        default=project_dir / "models" / "hand_landmarker.task",
        help="MediaPipe Hand Landmarker .task 모델",
    )
    parser.add_argument(
        "--output", type=Path, default=project_dir / "features",
        help="landmarks.csv와 quality_report.csv 저장 폴더",
    )
    parser.add_argument("--start-sec", type=float, default=1.0)
    parser.add_argument("--end-sec", type=float, default=3.0)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="특정 라벨만 처리 (예: --labels ㄱ ㄴ ㄷ)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def feature_fields() -> list[str]:
    base = [
        "sample_id", "signer_id", "session_id", "label_index", "label", "take",
        "file_path", "frame_index", "timestamp_ms", "handedness", "handedness_score",
    ]
    raw = [f"raw_{axis}{idx}" for idx in range(LANDMARK_COUNT) for axis in "xyz"]
    normalized = [f"norm_{axis}{idx}" for idx in range(LANDMARK_COUNT) for axis in "xyz"]
    return base + raw + normalized


def normalize_landmarks(landmarks) -> tuple[list[float], float]:
    wrist = landmarks[0]
    centered = [
        (point.x - wrist.x, point.y - wrist.y, point.z - wrist.z)
        for point in landmarks
    ]
    # Wrist-to-MCP distance is stable and less affected by fingertip poses.
    scale = max(
        math.sqrt(centered[idx][0] ** 2 + centered[idx][1] ** 2 + centered[idx][2] ** 2)
        for idx in MCP_INDICES
    )
    if scale < 1e-8:
        scale = 1.0
    values: list[float] = []
    for x, y, z in centered:
        values.extend((x / scale, y / scale, z / scale))
    return values, scale


def make_landmarker(args: argparse.Namespace):
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        # Frames are sampled independently from many short files. IMAGE mode
        # avoids carrying tracker state between videos and lets one task be reused.
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def load_metadata(path: Path, labels: set[str] | None) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))
    rows = [row for row in rows if row.get("status", "accepted") == "accepted"]
    if labels is not None:
        rows = [row for row in rows if row.get("label") in labels]
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in seen:
            print(f"[WARN] 중복/빈 sample_id 제외: {sample_id!r}", file=sys.stderr)
            continue
        seen.add(sample_id)
        unique.append(row)
    return unique


def main() -> int:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step은 1 이상이어야 합니다.")
    if args.start_sec < 0 or args.end_sec <= args.start_sec:
        raise ValueError("시간 범위는 0 <= start-sec < end-sec 이어야 합니다.")
    if not args.model.exists():
        raise FileNotFoundError(
            f"MediaPipe 모델이 없습니다: {args.model}\n"
            "README의 'MediaPipe 준비' 명령으로 hand_landmarker.task를 내려받으세요."
        )

    dataset = args.dataset.resolve()
    metadata = (args.metadata or dataset / "metadata.csv").resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    landmarks_path = output / "landmarks.csv"
    quality_path = output / "quality_report.csv"
    if not args.overwrite and (landmarks_path.exists() or quality_path.exists()):
        raise FileExistsError("출력 파일이 이미 있습니다. 다시 만들려면 --overwrite를 사용하세요.")

    rows = load_metadata(metadata, set(args.labels) if args.labels else None)
    if not rows:
        raise ValueError("처리할 accepted 영상이 없습니다.")

    quality_fields = [
        "sample_id", "label", "file_path", "video_frames", "video_fps",
        "sampled_frames", "detected_frames", "detection_rate", "result", "message",
    ]
    feature_fp = landmarks_path.open("w", newline="", encoding="utf-8-sig")
    quality_fp = quality_path.open("w", newline="", encoding="utf-8-sig")
    feature_writer = csv.DictWriter(feature_fp, fieldnames=feature_fields())
    quality_writer = csv.DictWriter(quality_fp, fieldnames=quality_fields)
    feature_writer.writeheader()
    quality_writer.writeheader()

    total_detected = 0
    total_sampled = 0
    shared_landmarker = make_landmarker(args)
    try:
        for number, metadata_row in enumerate(rows, start=1):
            video_path = dataset / metadata_row["file_path"]
            sampled = 0
            detected = 0
            result_name = "ok"
            message = ""
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not video_path.exists() or not cap.isOpened() or fps <= 0:
                result_name = "error"
                message = "video_open_failed"
            else:
                start_frame = max(0, round(args.start_sec * fps))
                end_frame = min(frame_count, round(args.end_sec * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                with nullcontext(shared_landmarker) as landmarker:
                    frame_index = start_frame
                    while frame_index < end_frame:
                        ok, bgr = cap.read()
                        if not ok:
                            message = "early_end_of_video"
                            break
                        if (frame_index - start_frame) % args.frame_step != 0:
                            frame_index += 1
                            continue
                        sampled += 1
                        timestamp_ms = int(round(frame_index * 1000.0 / fps))
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        detection = landmarker.detect(mp_image)
                        if detection.hand_landmarks:
                            detected += 1
                            hand = detection.hand_landmarks[0]
                            handedness = detection.handedness[0][0]
                            raw_values: list[float] = []
                            for point in hand:
                                raw_values.extend((point.x, point.y, point.z))
                            normalized_values, _ = normalize_landmarks(hand)
                            output_row: dict[str, object] = {
                                key: metadata_row.get(key, "")
                                for key in (
                                    "sample_id", "signer_id", "session_id", "label_index",
                                    "label", "take", "file_path",
                                )
                            }
                            output_row.update(
                                frame_index=frame_index,
                                timestamp_ms=timestamp_ms,
                                handedness=handedness.category_name,
                                handedness_score=round(handedness.score, 6),
                            )
                            for idx, value in enumerate(raw_values):
                                point_idx, axis_idx = divmod(idx, 3)
                                output_row[f"raw_{'xyz'[axis_idx]}{point_idx}"] = round(value, 8)
                            for idx, value in enumerate(normalized_values):
                                point_idx, axis_idx = divmod(idx, 3)
                                output_row[f"norm_{'xyz'[axis_idx]}{point_idx}"] = round(value, 8)
                            feature_writer.writerow(output_row)
                        frame_index += 1
            cap.release()
            rate = detected / sampled if sampled else 0.0
            if result_name == "ok" and rate < 0.8:
                result_name = "review"
                message = message or "detection_rate_below_0.8"
            quality_writer.writerow(
                {
                    "sample_id": metadata_row["sample_id"],
                    "label": metadata_row["label"],
                    "file_path": metadata_row["file_path"],
                    "video_frames": frame_count,
                    "video_fps": round(fps, 3),
                    "sampled_frames": sampled,
                    "detected_frames": detected,
                    "detection_rate": round(rate, 4),
                    "result": result_name,
                    "message": message,
                }
            )
            feature_fp.flush()
            quality_fp.flush()
            total_sampled += sampled
            total_detected += detected
            print(
                f"[{number:02d}/{len(rows):02d}] {metadata_row['label']} "
                f"{metadata_row['sample_id']}: {detected}/{sampled} ({rate:.1%}) {result_name}"
            )
    finally:
        shared_landmarker.close()
        feature_fp.close()
        quality_fp.close()

    overall = total_detected / total_sampled if total_sampled else 0.0
    print(f"\n완료: {len(rows)}개 영상, 전체 검출률 {overall:.1%}")
    print(f"특징: {landmarks_path}")
    print(f"품질: {quality_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
