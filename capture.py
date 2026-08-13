#!/usr/bin/env python3
"""Mac webcam recorder for a Korean fingerspelling dataset."""

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
    "sample_id",
    "signer_id",
    "session_id",
    "label_index",
    "label",
    "take",
    "file_path",
    "recorded_at",
    "duration_sec",
    "fps",
    "width",
    "height",
    "camera_index",
    "saved_mirrored",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="한국어 지문자 웹캠 영상 수집기")
    parser.add_argument("--signer", required=True, help="촬영자 ID (예: S001)")
    parser.add_argument(
        "--session",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="촬영 세션 ID",
    )
    parser.add_argument("--camera", type=int, default=0, help="카메라 번호")
    parser.add_argument("--duration", type=float, default=4.0, help="영상 길이(초)")
    parser.add_argument("--countdown", type=int, default=3, help="촬영 전 카운트다운(초)")
    parser.add_argument("--fps", type=float, default=30.0, help="저장 FPS")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).with_name("labels.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("dataset"),
    )
    parser.add_argument(
        "--save-mirrored",
        action="store_true",
        help="좌우 반전된 영상을 저장 (미리보기는 항상 거울 모드)",
    )
    return parser.parse_args()


def load_labels(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"라벨 파일을 찾을 수 없습니다: {path}")
    labels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not labels:
        raise ValueError("라벨 파일이 비어 있습니다.")
    if len(labels) != len(set(labels)):
        raise ValueError("라벨 파일에 중복된 항목이 있습니다.")
    return labels


def safe_component(value: str, field: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."}:
        raise ValueError(f"{field} 값이 올바르지 않습니다.")
    if any(ch in value for ch in ("/", "\\", "\0")):
        raise ValueError(f"{field}에는 경로 구분자를 사용할 수 없습니다.")
    return value


def append_metadata(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def mark_rejected(csv_path: Path, sample_id: str, new_path: str) -> None:
    """Mark one metadata row rejected while preserving an audit trail."""
    if not csv_path.exists():
        return
    with csv_path.open("r", newline="", encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))
    for row in rows:
        if row.get("sample_id") == sample_id:
            row["file_path"] = new_path
            row["status"] = "rejected"
    temp_path = csv_path.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, csv_path)


def next_take(label_dir: Path, prefix: str) -> int:
    takes: list[int] = []
    for path in label_dir.glob(f"{prefix}_*.mp4"):
        try:
            takes.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(takes, default=0) + 1


def draw_hud(frame, lines: list[str], color=(255, 255, 255)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 145), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (24, 32 + idx * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


def open_camera(args: argparse.Namespace):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise RuntimeError(
            "카메라를 열 수 없습니다. macOS 시스템 설정 > 개인정보 보호 및 보안 > "
            "카메라에서 Terminal(또는 사용 중인 앱)의 접근을 허용하세요."
        )
    return cap


def main() -> int:
    args = parse_args()
    args.signer = safe_component(args.signer, "signer")
    args.session = safe_component(args.session, "session")
    if args.duration <= 0 or args.fps <= 0 or args.countdown < 0:
        raise ValueError("duration/fps는 양수이고 countdown은 0 이상이어야 합니다.")

    labels = load_labels(args.labels)
    output_root = args.output.resolve()
    metadata_path = output_root / "metadata.csv"
    cap = open_camera(args)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    label_idx = 0
    state = "READY"
    state_started = 0.0
    writer = None
    current_path: Path | None = None
    current_take = 0
    frames_written = 0
    last_saved: Path | None = None

    window = "Korean Fingerspelling Capture"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Controls: SPACE=record, N/Enter/Right=next, P/Left=previous, R=retake, Q/Esc=quit")
    print(f"Current label [{label_idx + 1}/{len(labels)}]: {labels[label_idx]}")

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                print("카메라 프레임을 읽지 못했습니다.", file=sys.stderr)
                break

            now = time.monotonic()
            preview = cv2.flip(raw_frame, 1)

            if state == "COUNTDOWN":
                remaining = args.countdown - (now - state_started)
                if remaining <= 0:
                    label_number = label_idx + 1
                    label_dir = output_root / args.signer / args.session / f"{label_number:02d}_{labels[label_idx]}"
                    label_dir.mkdir(parents=True, exist_ok=True)
                    prefix = f"{args.signer}_{args.session}_{label_number:02d}"
                    current_take = next_take(label_dir, prefix)
                    current_path = label_dir / f"{prefix}_{current_take:03d}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(current_path), fourcc, args.fps, (actual_width, actual_height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"영상 파일을 만들 수 없습니다: {current_path}")
                    frames_written = 0
                    state = "RECORDING"
                    state_started = now
                else:
                    draw_hud(
                        preview,
                        [
                            f"GET READY: {max(1, int(remaining) + 1)}",
                            f"Label {label_idx + 1}/{len(labels)} (see terminal)",
                            "Keep your hand and wrist visible",
                        ],
                        (0, 220, 255),
                    )

            if state == "RECORDING":
                save_frame = cv2.flip(raw_frame, 1) if args.save_mirrored else raw_frame
                writer.write(save_frame)
                frames_written += 1
                elapsed = now - state_started
                draw_hud(
                    preview,
                    [
                        f"RECORDING  {elapsed:.1f}/{args.duration:.1f} sec",
                        f"Label {label_idx + 1}/{len(labels)} | Take {current_take}",
                        "Move the hand angle slightly while holding the shape",
                    ],
                    (50, 80, 255),
                )
                if elapsed >= args.duration:
                    writer.release()
                    writer = None
                    recorded_at = datetime.now().astimezone().isoformat(timespec="seconds")
                    relative_path = current_path.relative_to(output_root)
                    sample_id = current_path.stem
                    append_metadata(
                        metadata_path,
                        {
                            "sample_id": sample_id,
                            "signer_id": args.signer,
                            "session_id": args.session,
                            "label_index": label_number,
                            "label": labels[label_idx],
                            "take": current_take,
                            "file_path": relative_path.as_posix(),
                            "recorded_at": recorded_at,
                            "duration_sec": round(frames_written / args.fps, 3),
                            "fps": args.fps,
                            "width": actual_width,
                            "height": actual_height,
                            "camera_index": args.camera,
                            "saved_mirrored": args.save_mirrored,
                            "status": "accepted",
                        },
                    )
                    last_saved = current_path
                    print(f"Saved: {current_path}")
                    state = "READY"

            if state == "READY":
                label_number = label_idx + 1
                label_dir = output_root / args.signer / args.session / f"{label_number:02d}_{labels[label_idx]}"
                prefix = f"{args.signer}_{args.session}_{label_number:02d}"
                take = next_take(label_dir, prefix)
                draw_hud(
                    preview,
                    [
                        f"READY | Label {label_idx + 1}/{len(labels)} | Next take {take}",
                        "SPACE record | N/ENTER next | P previous | R redo | Q quit",
                        f"Signer {args.signer} | Session {args.session}",
                    ],
                )

            cv2.imshow(window, preview)
            # waitKeyEx preserves platform-specific arrow-key codes.  The low
            # byte is still used for ordinary ASCII keys.
            key_full = cv2.waitKeyEx(1)
            key_ascii = key_full & 0xFF if key_full >= 0 else -1
            if key_ascii in (ord("q"), ord("Q"), 27):
                break
            if state != "READY":
                continue
            if key_ascii == ord(" "):
                state = "COUNTDOWN"
                state_started = now
            elif key_ascii in (ord("n"), ord("N"), ord("d"), ord("D"), 13) or key_full in (
                63235, 65363, 2555904
            ):
                label_idx = (label_idx + 1) % len(labels)
                last_saved = None
                print(f"Current label [{label_idx + 1}/{len(labels)}]: {labels[label_idx]}")
            elif key_ascii in (ord("p"), ord("P"), ord("a"), ord("A"), 8) or key_full in (
                63234, 65361, 2424832
            ):
                label_idx = (label_idx - 1) % len(labels)
                last_saved = None
                print(f"Current label [{label_idx + 1}/{len(labels)}]: {labels[label_idx]}")
            elif key_ascii in (ord("r"), ord("R")):
                if last_saved and last_saved.exists():
                    # Keep the old clip recoverable instead of deleting it.
                    rejected = output_root / "rejected"
                    rejected.mkdir(parents=True, exist_ok=True)
                    destination = rejected / last_saved.name
                    suffix = 1
                    while destination.exists():
                        destination = rejected / f"{last_saved.stem}_{suffix}{last_saved.suffix}"
                        suffix += 1
                    os.replace(last_saved, destination)
                    mark_rejected(
                        metadata_path,
                        last_saved.stem,
                        destination.relative_to(output_root).as_posix(),
                    )
                    print(f"Moved rejected clip to: {destination}")
                    last_saved = None
                    state = "COUNTDOWN"
                    state_started = now
                else:
                    print("현재 라벨에서 바로 전에 저장한 영상이 없습니다.")
    finally:
        if writer is not None:
            writer.release()
            if current_path and current_path.exists():
                current_path.unlink()
        cap.release()
        cv2.destroyAllWindows()

    print(f"Dataset: {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
