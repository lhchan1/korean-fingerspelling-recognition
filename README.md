# 한국어 지문자 영상 수집기

MacBook 웹캠으로 지문자 영상을 촬영하고 `촬영자/세션/라벨` 단위로 저장합니다.
기본 라벨은 기본 자음 14종, 모음 17종과 `NONE`입니다.

## 설치

Python 3.10 이상을 권장합니다.

```bash
cd fingerspelling_capture
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 실행

```bash
python3 capture.py --signer S001
```

4초 대신 5초씩 촬영하려면:

```bash
python3 capture.py --signer S001 --duration 5
```

macOS가 카메라 권한을 요청하면 허용합니다. 열리지 않을 경우 `시스템 설정 >
개인정보 보호 및 보안 > 카메라`에서 Terminal 또는 Codex의 접근을 허용합니다.

## 키 조작

- `Space`: 3초 카운트다운 후 촬영
- `N`, `D`, `Enter` 또는 오른쪽 방향키: 다음 라벨
- `P`, `A`, `Backspace` 또는 왼쪽 방향키: 이전 라벨
- `R`: 방금 영상을 `dataset/rejected`로 옮기고 재촬영
- `Q` 또는 `Esc`: 종료

OpenCV 창은 한글 표시가 불안정하므로 현재 글자는 실행한 터미널에 출력됩니다.
키가 반응하지 않으면 먼저 카메라 미리보기 창을 클릭하고, macOS 입력 소스를
영문으로 바꿉니다. 한글 입력기 상태에서도 `Enter`로 다음 라벨로 이동할 수 있습니다.

## 저장 결과

```text
dataset/
├── metadata.csv
└── S001/
    └── 20260807_120000/
        ├── 00_ㄱ/
        │   ├── S001_20260807_120000_00_001.mp4
        │   └── S001_20260807_120000_00_002.mp4
        └── 01_ㄴ/
```

`metadata.csv`의 `status`는 정상 영상이면 `accepted`, `R`로 재촬영한 영상이면
`rejected`입니다. 학습 데이터 생성 시 `accepted` 행만 사용합니다.

미리보기는 거울처럼 좌우 반전되지만, 기본 저장 영상은 원본 방향입니다. Android
앱 전처리와 동일하게 반전 영상을 저장해야 하는 경우에만 `--save-mirrored`를
사용합니다.

라벨을 추가하거나 순서를 변경하려면 `labels.txt`를 편집합니다. 동적 지문자나
별도의 중립 동작을 추가할 때도 같은 방식으로 확장할 수 있습니다.

## 촬영 권장 방식

1. 중립 자세에서 시작합니다.
2. 지문자 자세를 만들고 2초 정도 유지합니다.
3. 손 모양은 유지하면서 각도를 좌우·상하로 조금씩 바꿉니다.
4. 손 전체와 손목이 항상 화면에 들어오게 합니다.
5. 같은 글자를 한 세션에서 10회 정도 촬영합니다.

학습·검증·테스트 분할은 추출된 프레임이 아니라 `signer_id` 기준으로 해야 합니다.

## MediaPipe 특징 추출

MediaPipe와 공식 Hand Landmarker 모델을 준비합니다.

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
mkdir -p models
curl -L \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task \
  -o models/hand_landmarker.task
```

현재 파일럿 라벨만 처리합니다.

```bash
python3 extract_landmarks.py --labels ㄱ ㄴ ㄷ
```

기본적으로 영상의 1~3초 구간에서 5프레임마다 특징을 추출합니다. 결과는 다음과
같습니다.

```text
features/
├── landmarks.csv       # 원본 좌표와 손목·손 크기 기준 정규화 좌표
└── quality_report.csv  # 영상별 검출 프레임 수와 검출률
```

기존 결과를 다시 생성할 때는 다음과 같이 실행합니다.

```bash
python3 extract_landmarks.py --labels ㄱ ㄴ ㄷ --overwrite
```

`quality_report.csv`에서 `result=review`인 영상은 검출률이 80% 미만이므로 직접
확인하거나 다시 촬영합니다. 같은 영상에서 나온 프레임은 이후 데이터 분할 시
반드시 같은 그룹에 유지해야 합니다.

## LSTM 학습

현재 `ㄱ`, `ㄴ`, `ㄷ` 파일럿 모델을 학습합니다.

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 train_lstm.py --labels ㄱ ㄴ ㄷ
```

각 클래스에서 검증 1개와 테스트 2개를 제외한 모든 영상을 학습에 사용합니다.
클래스별 영상 수 차이는 클래스 가중치로 보정하며, 동일 영상에서 나온 프레임은
항상 같은 분할에 들어갑니다.

```text
training/lstm_v3/
├── fingerspelling_lstm.keras
├── labels.json
├── config.json
├── metrics.json
├── split.csv
├── history.csv
├── confusion_matrix.csv
└── test_predictions.csv
```

기존 결과를 덮어쓰려면:

```bash
python3 train_lstm.py --labels ㄱ ㄴ ㄷ --overwrite
```

현재 결과는 촬영자 한 명의 파일럿 평가이므로 다른 사용자에 대한 일반화 성능으로
해석하면 안 됩니다.

## 실시간 웹캠 LSTM 테스트

```bash
source .venv/bin/activate
python3 realtime_lstm.py
```

손 랜드마크 12개가 약 2초 동안 모이면 예측을 시작합니다. 같은 지문자 자세를
유지하고 `Q` 또는 `Esc`로 종료합니다. 미리보기만 거울 모드이며 모델 입력에는
학습 영상과 동일한 원본 방향을 사용합니다.

신뢰도 기준을 조정하려면:

```bash
python3 realtime_lstm.py --threshold 0.75
```

현재 모델에는 `NONE`이 없기 때문에 손이 검출되면 항상 `ㄱ`, `ㄴ`, `ㄷ` 중
하나를 선택합니다. 실제 사용용 모델에는 이후 `NONE` 데이터를 추가해야 합니다.
