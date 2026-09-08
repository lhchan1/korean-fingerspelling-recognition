#!/usr/bin/env python3
"""Record fixed-duration Korean sign-word clips from a webcam."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


CSV_FIELDS = [
    "sample_id", "signer_id", "session_id", "label_index", "label", "take",
    "file_path", "recorded_at", "duration_sec", "fps", "width", "height",
    "camera_index", "saved_mirrored", "status",
]


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="한국 수어 단어 영상 수집기")
    parser.add_argument("--signer", required=True, help="촬영자 ID (예: S001)")
    parser.add_argument("--session", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--labels", type=Path, default=base / "word_labels.txt")
    parser.add_argument("--output", type=Path, default=base / "word_dataset")
    parser.add_argument("--save-mirrored", action="store_true")
    return parser.parse_args()


def load_labels(path: Path) -> list[str]:
    labels = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("라벨 파일이 비었거나 중복 라벨이 있습니다.")
    return labels


def safe_component(value: str, name: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or any(c in value for c in "/\\\0"):
        raise ValueError(f"올바르지 않은 {name}: {value!r}")
    return value


def append_metadata(path: Path, row: dict[str, object]) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def rewrite_status(path: Path, sample_id: str, new_path: str) -> None:
    with path.open(newline="", encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))
    for row in rows:
        if row["sample_id"] == sample_id:
            row["file_path"] = new_path
            row["status"] = "rejected"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def next_take(directory: Path, prefix: str) -> int:
    takes = []
    for path in directory.glob(f"{prefix}_*.mp4"):
        try:
            takes.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    return max(takes, default=0) + 1


def draw_hud(frame, lines: list[str], color=(255, 255, 255)) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (24, 34 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2, cv2.LINE_AA)


def recording_instruction(label: str) -> str:
    if label == "IDLE":
        return "Stay naturally still; do not perform a sign"
    if label == "OTHER":
        return "Perform one non-target gesture naturally"
    return "Perform one complete sign naturally"


def main() -> int:
    args = parse_args()
    args.signer = safe_component(args.signer, "signer")
    args.session = safe_component(args.session, "session")
    if args.duration <= 0 or args.fps <= 0 or args.countdown < 0:
        raise ValueError("duration/fps는 양수이고 countdown은 0 이상이어야 합니다.")
    labels = load_labels(args.labels)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "metadata.csv"

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. macOS 카메라 권한을 확인하세요.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    label_idx, state, started = 0, "READY", 0.0
    writer = None
    current_path = None
    current_take = 0
    frames_written = 0
    last_saved = None
    window = "Korean Sign Word Capture"
    print("SPACE=record, N/Enter/Right=next, P/Left=previous, R=retake, Q/Esc=quit")

    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                raise RuntimeError("카메라 프레임을 읽지 못했습니다.")
            now = time.monotonic()
            preview = cv2.flip(raw, 1)
            label = labels[label_idx]
            # word_labels.txt is intentionally zero-based: 00_IDLE, 01_OTHER, ...
            number = label_idx
            label_dir = output / args.signer / f"{number:02d}_{label}"
            prefix = f"{args.signer}_{args.session}_{number:02d}"

            if state == "COUNTDOWN":
                remaining = args.countdown - (now - started)
                if remaining <= 0:
                    label_dir.mkdir(parents=True, exist_ok=True)
                    current_take = next_take(label_dir, prefix)
                    current_path = label_dir / f"{prefix}_{current_take:03d}.mp4"
                    writer = cv2.VideoWriter(str(current_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"영상 파일을 만들 수 없습니다: {current_path}")
                    frames_written, state, started = 0, "RECORDING", now
                else:
                    draw_hud(preview, [f"GET READY: {max(1, int(remaining)+1)}",
                                      f"Class {number:02d} ({label_idx + 1}/{len(labels)}): {label}",
                                      "Keep both hands, shoulders and elbows visible"], (0, 220, 255))

            if state == "RECORDING":
                writer.write(cv2.flip(raw, 1) if args.save_mirrored else raw)
                frames_written += 1
                elapsed = now - started
                draw_hud(preview, [f"RECORDING {elapsed:.1f}/{args.duration:.1f} sec",
                                  f"Class {number:02d} ({label_idx + 1}/{len(labels)}): {label} | Take {current_take}",
                                  recording_instruction(label)], (50, 80, 255))
                if elapsed >= args.duration:
                    writer.release(); writer = None
                    relative = current_path.relative_to(output).as_posix()
                    append_metadata(metadata, {
                        "sample_id": current_path.stem, "signer_id": args.signer,
                        "session_id": args.session, "label_index": number, "label": label,
                        "take": current_take, "file_path": relative,
                        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "duration_sec": round(frames_written / args.fps, 3), "fps": args.fps,
                        "width": width, "height": height, "camera_index": args.camera,
                        "saved_mirrored": args.save_mirrored, "status": "accepted",
                    })
                    last_saved, state = current_path, "READY"
                    print(f"Saved: {current_path}")

            if state == "READY":
                take = next_take(label_dir, prefix)
                draw_hud(preview, [f"READY | Class {number:02d} ({label_idx + 1}/{len(labels)}): {label}",
                                  f"Next take {take} | SPACE record | N next | P previous",
                                  f"Signer {args.signer} | {args.duration:g} sec | hands + upper body"])

            cv2.imshow(window, preview)
            key = cv2.waitKeyEx(1); ascii_key = key & 0xFF if key >= 0 else -1
            if ascii_key in (ord("q"), ord("Q"), 27): break
            if state != "READY": continue
            if ascii_key == ord(" "):
                state, started = "COUNTDOWN", now
            elif ascii_key in (ord("n"), ord("N"), 13) or key in (63235, 65363, 2555904):
                label_idx = min(label_idx + 1, len(labels) - 1)
            elif ascii_key in (ord("p"), ord("P")) or key in (63234, 65361, 2424832):
                label_idx = max(label_idx - 1, 0)
            elif ascii_key in (ord("r"), ord("R")) and last_saved and last_saved.exists():
                rejected = output / "rejected" / args.signer / last_saved.parent.name / last_saved.name
                rejected.parent.mkdir(parents=True, exist_ok=True)
                last_saved.rename(rejected)
                rewrite_status(metadata, last_saved.stem, rejected.relative_to(output).as_posix())
                print(f"Rejected: {rejected}"); last_saved = None
    finally:
        if writer is not None: writer.release()
        cap.release(); cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
