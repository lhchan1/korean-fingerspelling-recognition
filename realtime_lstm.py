#!/usr/bin/env python3
"""Real-time webcam inference for the trained fingerspelling LSTM."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fingerspelling_matplotlib")
)

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


LANDMARK_COUNT = 21
MCP_INDICES = (5, 9, 13, 17)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

# Dataset folder/metadata numbering (the 1~36 chart order).
LABEL_NUMBERS = {
    label: number
    for number, label in enumerate(
        "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅢㅚㅟ",
        start=1,
    )
}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="LSTM 지문자 실시간 웹캠 테스트")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--model",
        type=Path,
        default=project_dir / "training" / "lstm_v10_no_double" / "fingerspelling_lstm.keras",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_dir / "training" / "lstm_v10_no_double" / "config.json",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=project_dir / "training" / "lstm_v10_no_double" / "labels.json",
    )
    parser.add_argument(
        "--landmarker-model",
        type=Path,
        default=project_dir / "models" / "hand_landmarker.task",
    )
    parser.add_argument("--sample-fps", type=float, default=6.0)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror-preview", action="store_true")
    return parser.parse_args()


def normalize_landmarks(landmarks) -> np.ndarray:
    """Exactly the same normalization used by extract_landmarks.py."""
    wrist = landmarks[0]
    centered = np.asarray(
        [
            (point.x - wrist.x, point.y - wrist.y, point.z - wrist.z)
            for point in landmarks
        ],
        dtype=np.float32,
    )
    scale = max(float(np.linalg.norm(centered[idx])) for idx in MCP_INDICES)
    if scale < 1e-8:
        scale = 1.0
    return (centered / scale).reshape(-1)


def create_landmarker(model_path: Path):
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def find_korean_font(size: int):
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def put_korean_text(frame, text: str, position, font, color=(255, 255, 255)):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_hand(frame, landmarks, mirrored: bool) -> None:
    height, width = frame.shape[:2]
    points = []
    for point in landmarks:
        x = 1.0 - point.x if mirrored else point.x
        points.append((int(x * width), int(point.y * height)))
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (50, 220, 50), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, (30, 80, 255), -1, cv2.LINE_AA)


def open_camera(args: argparse.Namespace):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(
            "카메라를 열 수 없습니다. macOS 시스템 설정 > 개인정보 보호 및 보안 > "
            "카메라에서 Terminal 또는 실행 앱의 권한을 확인하세요."
        )
    return cap


def main() -> int:
    args = parse_args()
    for path in (args.model, args.config, args.labels, args.landmarker_model):
        if not path.exists():
            raise FileNotFoundError(f"필요한 파일이 없습니다: {path}")
    if args.sample_fps <= 0 or not 0.0 <= args.threshold <= 1.0:
        raise ValueError("sample-fps는 양수이고 threshold는 0~1이어야 합니다.")
    if args.smoothing_window < 1 or not 1 <= args.min_votes <= args.smoothing_window:
        raise ValueError("min-votes는 1 이상 smoothing-window 이하여야 합니다.")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    sequence_length = int(config["sequence_length"])
    expected_features = len(config["feature_columns"])
    if expected_features != LANDMARK_COUNT * 3:
        raise ValueError(f"지원하지 않는 특징 수입니다: {expected_features}")

    model = tf.keras.models.load_model(args.model)
    if tuple(model.input_shape[1:]) != (sequence_length, expected_features):
        raise ValueError(
            f"모델 입력 {model.input_shape}과 설정 {(sequence_length, expected_features)}이 다릅니다."
        )

    cap = open_camera(args)
    mirror_preview = not args.no_mirror_preview
    sequence = deque(maxlen=sequence_length)
    recent_predictions = deque(maxlen=args.smoothing_window)
    last_sample_time = 0.0
    last_hand_seen = 0.0
    last_landmarks = None
    probabilities = np.zeros(len(labels), dtype=np.float32)
    display_label = "손을 보여주세요"
    display_number = None
    display_confidence = 0.0
    font_large = find_korean_font(52)
    font_small = find_korean_font(24)
    window = "Fingerspelling LSTM Realtime"

    unknown_labels = [label for label in labels if label not in LABEL_NUMBERS]
    if unknown_labels:
        raise ValueError(f"라벨 번호표에 없는 모델 라벨입니다: {unknown_labels}")
    print("Labels:", [f"{LABEL_NUMBERS[label]}번 {label}" for label in labels])
    print("약 2초 동안 같은 지문자 자세를 유지하세요. Q 또는 Esc로 종료합니다.")
    print("주의: NONE 클래스가 없어 손이 검출되면 학습된 지문자 중 하나로 분류합니다.")

    start_time = time.monotonic()
    try:
        with create_landmarker(args.landmarker_model) as landmarker:
            while True:
                ok, raw_frame = cap.read()
                if not ok:
                    raise RuntimeError("카메라 프레임을 읽지 못했습니다.")
                now = time.monotonic()
                if now - last_sample_time >= 1.0 / args.sample_fps:
                    timestamp_ms = int((now - start_time) * 1000)
                    rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    last_sample_time = now

                    if result.hand_landmarks:
                        last_landmarks = result.hand_landmarks[0]
                        last_hand_seen = now
                        sequence.append(normalize_landmarks(last_landmarks))
                        if len(sequence) == sequence_length:
                            model_input = np.asarray(sequence, dtype=np.float32)[None, ...]
                            probabilities = model.predict(model_input, verbose=0)[0]
                            pred_id = int(np.argmax(probabilities))
                            confidence = float(probabilities[pred_id])
                            recent_predictions.append(pred_id if confidence >= args.threshold else -1)
                            votes = Counter(recent_predictions)
                            winner, vote_count = votes.most_common(1)[0]
                            if winner >= 0 and vote_count >= args.min_votes:
                                display_label = labels[winner]
                                display_number = LABEL_NUMBERS[display_label]
                                display_confidence = float(probabilities[winner])
                            else:
                                display_label = "판단 중"
                                display_number = None
                                display_confidence = confidence
                        else:
                            display_label = f"수집 중 {len(sequence)}/{sequence_length}"
                            display_number = None
                            display_confidence = 0.0
                    elif now - last_hand_seen > 0.5:
                        sequence.clear()
                        recent_predictions.clear()
                        last_landmarks = None
                        display_label = "손을 보여주세요"
                        display_number = None
                        display_confidence = 0.0

                preview = cv2.flip(raw_frame, 1) if mirror_preview else raw_frame.copy()
                if last_landmarks is not None and now - last_hand_seen <= 0.5:
                    draw_hand(preview, last_landmarks, mirror_preview)
                overlay = preview.copy()
                cv2.rectangle(overlay, (0, 0), (preview.shape[1], 145), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.58, preview, 0.42, 0, preview)
                confidence_text = (
                    f"confidence {display_confidence:.1%}" if display_confidence > 0 else ""
                )
                result_text = (
                    f"지문자: {display_label}   라벨 번호: {display_number}번"
                    if display_number is not None
                    else display_label
                )
                preview = put_korean_text(preview, result_text, (24, 14), font_large)
                preview = put_korean_text(
                    preview,
                    f"{confidence_text}   buffer {len(sequence)}/{sequence_length}",
                    (26, 92),
                    font_small,
                    (200, 220, 255),
                )
                cv2.putText(
                    preview,
                    "Q / ESC: quit | Perform one sign for about 3 seconds",
                    (24, preview.shape[0] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window, preview)
                key = cv2.waitKeyEx(1)
                key_ascii = key & 0xFF if key >= 0 else -1
                if key_ascii in (ord("q"), ord("Q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
