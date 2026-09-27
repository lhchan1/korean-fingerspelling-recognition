#!/usr/bin/env python3
"""Real-time webcam test for the isolated sign-word LSTM."""

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
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "word_sign_matplotlib"))

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from extract_word_landmarks import features_for_frame


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
    (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
POSE_CONNECTIONS = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
SUPPRESSED_LABELS = {"IDLE", "OTHER"}


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    default_training = base / "training" / "word_lstm_v3_medical_24class"
    parser = argparse.ArgumentParser(description="단어 수어 LSTM 실시간 웹캠 테스트")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=default_training / "word_sign_lstm.keras")
    parser.add_argument("--config", type=Path, default=default_training / "config.json")
    parser.add_argument("--labels", type=Path, default=default_training / "labels.json")
    parser.add_argument("--hand-model", type=Path, default=base.parent / "models" / "hand_landmarker.task")
    parser.add_argument("--pose-model", type=Path, default=base.parent / "models" / "pose_landmarker_lite.task")
    parser.add_argument("--sample-fps", type=float, default=15.0)
    parser.add_argument("--predict-interval", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror-preview", action="store_true")
    return parser.parse_args()


def create_hand_landmarker(path: Path):
    return mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def create_pose_landmarker(path: Path):
    return mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def open_camera(args: argparse.Namespace):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. macOS 카메라 권한을 확인하세요.")
    return cap


def find_korean_font(size: int):
    for path in (
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def put_texts(frame, texts):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for text, position, font, color in texts:
        draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def image_point(point, width: int, height: int, mirrored: bool):
    x = 1.0 - float(point.x) if mirrored else float(point.x)
    return int(x * width), int(float(point.y) * height)


def draw_landmarks(frame, hand_result, pose_result, mirrored: bool) -> None:
    height, width = frame.shape[:2]
    for hand in hand_result.hand_landmarks:
        points = [image_point(point, width, height, mirrored) for point in hand]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (50, 220, 70), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 4, (30, 90, 255), -1, cv2.LINE_AA)
    if pose_result.pose_landmarks:
        pose = pose_result.pose_landmarks[0]
        points = {index: image_point(pose[index], width, height, mirrored) for index in range(11, 17)}
        for start, end in POSE_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 180, 40), 3, cv2.LINE_AA)
        for point in points.values():
            cv2.circle(frame, point, 5, (255, 210, 40), -1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    for path in (args.model, args.config, args.labels, args.hand_model, args.pose_model):
        if not path.exists():
            raise FileNotFoundError(f"필요한 파일이 없습니다: {path}")
    if args.sample_fps <= 0 or args.predict_interval <= 0:
        raise ValueError("sample-fps와 predict-interval은 양수여야 합니다.")
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold는 0~1이어야 합니다.")
    if args.smoothing_window < 1 or not 1 <= args.min_votes <= args.smoothing_window:
        raise ValueError("min-votes는 1 이상 smoothing-window 이하여야 합니다.")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    sequence_length = int(config["sequence_length"])
    feature_count = int(config["feature_count"])
    if feature_count != 146:
        raise ValueError(f"지원하지 않는 특징 수입니다: {feature_count}")
    model = tf.keras.models.load_model(args.model)
    if tuple(model.input_shape[1:]) != (sequence_length, feature_count):
        raise ValueError(f"모델 입력 {model.input_shape}과 설정 {(sequence_length, feature_count)}이 다릅니다.")

    cap = open_camera(args)
    mirror = not args.no_mirror_preview
    sequence = deque(maxlen=sequence_length)
    recent = deque(maxlen=args.smoothing_window)
    recognized = deque(maxlen=6)
    probabilities = np.zeros(len(labels), dtype=np.float32)
    display_label = "버퍼 수집 중"
    display_confidence = 0.0
    stable_label = None
    armed = True
    last_sample = 0.0
    last_prediction = 0.0
    latest_hands = None
    latest_pose = None
    start = time.monotonic()
    font_large = find_korean_font(48)
    font_medium = find_korean_font(28)
    font_small = find_korean_font(21)
    window = "Word Sign LSTM Realtime"

    print("Labels:", labels)
    print("약 4초 버퍼를 사용합니다. 단어 사이에 IDLE 자세를 넣으세요.")
    print("Q/Esc=종료, C=인식 기록 삭제, R=버퍼 초기화")
    try:
        with create_hand_landmarker(args.hand_model) as hands, create_pose_landmarker(args.pose_model) as pose:
            while True:
                ok, raw = cap.read()
                if not ok:
                    raise RuntimeError("카메라 프레임을 읽지 못했습니다.")
                now = time.monotonic()
                if now - last_sample >= 1.0 / args.sample_fps:
                    timestamp_ms = int((now - start) * 1000)
                    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    latest_hands = hands.detect_for_video(image, timestamp_ms)
                    latest_pose = pose.detect_for_video(image, timestamp_ms)
                    values, _, _, _ = features_for_frame(latest_hands, latest_pose)
                    sequence.append(values)
                    last_sample = now

                if len(sequence) == sequence_length and now - last_prediction >= args.predict_interval:
                    probabilities = model.predict(np.asarray(sequence, dtype=np.float32)[None, ...], verbose=0)[0]
                    prediction = int(np.argmax(probabilities))
                    confidence = float(probabilities[prediction])
                    recent.append(prediction if confidence >= args.threshold else -1)
                    winner, votes = Counter(recent).most_common(1)[0]
                    if winner >= 0 and votes >= args.min_votes:
                        stable_label = labels[winner]
                        display_label = stable_label
                        display_confidence = float(probabilities[winner])
                        if stable_label in SUPPRESSED_LABELS:
                            armed = True
                        elif armed:
                            recognized.append(stable_label)
                            armed = False
                            print(f"Recognized: {stable_label} ({display_confidence:.1%})")
                    else:
                        stable_label = None
                        display_label = "판단 중"
                        display_confidence = confidence
                    last_prediction = now

                preview = cv2.flip(raw, 1) if mirror else raw.copy()
                if latest_hands is not None and latest_pose is not None:
                    draw_landmarks(preview, latest_hands, latest_pose, mirror)
                overlay = preview.copy()
                cv2.rectangle(overlay, (0, 0), (preview.shape[1], 185), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.62, preview, 0.38, 0, preview)
                progress = min(1.0, len(sequence) / sequence_length)
                cv2.rectangle(preview, (25, 145), (425, 163), (70, 70, 70), -1)
                cv2.rectangle(preview, (25, 145), (25 + int(400 * progress), 163), (70, 210, 100), -1)
                state_color = (255, 220, 120) if stable_label in SUPPRESSED_LABELS else (255, 255, 255)
                texts = [
                    (f"현재 상태: {display_label}", (24, 12), font_large, state_color),
                    (f"신뢰도 {display_confidence:.1%}   버퍼 {len(sequence)}/{sequence_length}", (26, 78), font_medium, (200, 225, 255)),
                    ("인식 결과: " + ("  ".join(recognized) if recognized else "없음"), (26, 110), font_small, (210, 255, 210)),
                ]
                preview = put_texts(preview, texts)
                cv2.putText(preview, "Q: quit | C: clear result | R: reset buffer", (24, preview.shape[0] - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(window, preview)
                key = cv2.waitKeyEx(1)
                ascii_key = key & 0xFF if key >= 0 else -1
                if ascii_key in (ord("q"), ord("Q"), 27):
                    break
                if ascii_key in (ord("c"), ord("C")):
                    recognized.clear()
                if ascii_key in (ord("r"), ord("R")):
                    sequence.clear(); recent.clear(); armed = True
                    display_label = "버퍼 수집 중"; display_confidence = 0.0
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
