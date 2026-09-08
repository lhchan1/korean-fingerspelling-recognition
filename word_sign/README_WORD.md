# 단어 수어 데이터 수집 베이스

현재 클래스는 `IDLE`, `OTHER`, `나`, `가다`, `학교`, `먹다`, `아프다` 7종이며 1280×720, 30FPS, 4초로 저장한다.
원본 분석 좌표를 보존하기 위해 저장 영상은 기본적으로 미러링하지 않는다. 화면 미리보기만 거울 모드다.

## 1. 촬영

기존 프로젝트 가상환경을 활성화한 뒤 실행한다.

```bash
source .venv/bin/activate
python word_sign/capture_words.py --signer S001 --duration 4
```

조작키: `Space` 촬영, `N/Enter/→` 다음 단어, `P/←` 이전 단어, `R` 직전 영상 제외, `Q/Esc` 종료.

촬영 중에는 양손, 양쪽 어깨, 팔꿈치가 모두 화면 안에 있어야 한다. 자연스러운 속도로 한 단어 동작을 한 번만 수행한다.

## 2. 폴더

```text
word_dataset/
├── metadata.csv
└── S001/
    ├── 00_IDLE/
    ├── 01_OTHER/
    ├── 02_나/
    ├── 03_가다/
    ├── 04_학교/
    ├── 05_먹다/
    └── 06_아프다/
```

새 촬영자는 `--signer S002`처럼 실행하면 자동으로 동일한 폴더가 만들어진다.

팀원에게 촬영을 요청할 때는 [클래스별 대표 영상 및 촬영 가이드](examples/README.md)를 먼저 확인하도록 안내한다. `examples/`의 영상만 GitHub에 포함하고 `word_dataset/` 원본 영상은 업로드하지 않는다.

## 3. 양손 + Pose 특징 추출

`models/pose_landmarker_lite.task`가 필요하다. 손 모델은 기존 `models/hand_landmarker.task`를 재사용한다.

```bash
python word_sign/extract_word_landmarks.py --overwrite
```

4초 영상은 `--frames 60`으로 처리하며 프레임당 146개 특징이다.

```bash
python word_sign/extract_word_landmarks.py --frames 60 --overwrite
```

- 왼손: 21×3 = 63
- 오른손: 21×3 = 63
- 상체 포즈(어깨·팔꿈치·손목): 6×3 = 18
- 왼손/오른손 검출 마스크: 2

모든 좌표는 양쪽 어깨 중점을 원점으로 하고 어깨 사이 3D 거리로 나눈다. 이 방식은 손이 몸의 어느 위치에서 움직이는지 보존한다.

## 권장 데이터 수

한 사람이 클래스당 40~50개를 모두 찍기보다 여러 촬영자가 나누는 것이 좋다. 초기 검증은 촬영자 4~5명, 사람당 단어별 10개를 권장한다. 모델 평가 시에는 한 촬영자의 영상을 통째로 테스트셋으로 분리한다.

## IDLE과 OTHER 촬영

- `00_IDLE`: 4초 동안 특정 수어를 하지 않고 자연스럽게 대기한다. 손을 내린 자세, 준비 자세, 손을 편하게 든 자세 등을 다양하게 촬영한다.
- `01_OTHER`: `나`, `가다`, `학교`, `먹다`, `아프다`가 아닌 동작 하나를 4초 안에 수행한다. 손 흔들기, 얼굴 만지기, 옷 정리, 물건 집기, 의미 없는 손동작 등을 다양하게 섞는다.

`OTHER`에서 같은 동작만 반복하지 않는다. 실시간 출력에서는 모델 결과가 `IDLE` 또는 `OTHER`이면 단어로 추가하지 않는다.

## 4. 단어 LSTM 학습

특징 CSV에 존재하는 클래스를 라벨 번호 순서대로 학습한다. `먹다`, `아프다`를 촬영하고 특징을 다시 추출하면 총 7개 클래스가 된다.

```bash
python word_sign/extract_word_landmarks.py --frames 60 --overwrite
python word_sign/train_word_lstm.py --output word_sign/training/word_lstm_v2
```

현재 `word_lstm_v1`은 신규 클래스 추가 전의 5클래스 모델이다. 새 데이터 학습 결과는 기존 모델을 보존하도록 `word_lstm_v2/`에 저장한다. 한 촬영자의 영상을 무작위로 나눈 정확도는 새로운 사람에 대한 성능을 의미하지 않는다.

## 5. 실시간 테스트

```bash
python word_sign/realtime_word_lstm.py
```

처음 약 4초는 버퍼를 채운다. 단어 클래스는 인식 결과에 추가하고 `IDLE`, `OTHER`는 상태로만 표시한다. 같은 단어를 다시 하면 중복 추가하지 않도록 단어 사이에 잠시 `IDLE` 자세를 취한다.

- `Q` 또는 `Esc`: 종료
- `C`: 누적 인식 결과 삭제
- `R`: 4초 버퍼 초기화
