# 약국·병원 단어 수어 데이터 수집

약국 복약 안내와 병원 문진 상황을 위한 총 24개 클래스(`IDLE`, `OTHER`, 의료 단어 22개)를 수집한다. 영상은 1280×720, 30FPS, 4초로 저장한다. 미리보기만 거울 모드이며, 분석 좌표 보존을 위해 저장 영상은 좌우 반전하지 않는다.

## 1. 클래스와 폴더

```text
word_dataset/S001/
├── 00_IDLE/       ├── 01_OTHER/     ├── 02_나/
├── 03_머리/       ├── 04_배/        ├── 05_목/
├── 06_아프다/     ├── 07_어지럽다/  ├── 08_기침/
├── 09_있다/       ├── 10_가래/      ├── 11_메스껍다/
├── 12_지금/       ├── 13_어제/      ├── 14_심하다/
├── 15_가슴/       ├── 16_열/        ├── 17_허리/
├── 18_약/         ├── 19_콧물/      ├── 20_설사/
├── 21_없다/       ├── 22_구토/      └── 23_조금/
```

`목`은 신체 부위인 목/목구멍, `열`은 발열을 뜻한다. `메스껍다`는 구역감, `구토`는 실제 토하는 증상으로 구분한다.

기존 영상 중 새 분류에 해당하는 `IDLE`, `OTHER`, `나`, `아프다` 총 222개는 유지했다. 이전 실험 클래스의 영상과 기존 특징 파일은 `backups/medical_24class_migration_20260914_173605/`에 보관한다.

## 2. 촬영

```bash
source .venv/bin/activate
python word_sign/capture_words.py --signer S001 --duration 4
```

새 촬영자는 `S002`, `S003`처럼 고유 ID를 사용한다.

- `Space`: 3초 카운트다운 후 촬영
- `N`, `Enter`, `→`: 다음 클래스
- `P`, `←`: 이전 클래스
- `R`: 직전 영상을 제외하고 재촬영
- `Q`, `Esc`: 종료

### 4초 촬영 흐름

```text
0.0~0.5초  자연스러운 준비 자세
0.5~2.5초  해당 수어를 자연스러운 속도로 한 번 수행
2.5~4.0초  동작을 끝내고 자연스럽게 대기
```

동작이 짧아도 반복하거나 인위적으로 느리게 늘이지 않는다. 수어의 시작과 끝이 영상 안에 들어오게 하고, 양쪽 어깨·팔꿈치·사용하는 손이 프레임 밖으로 잘리지 않게 촬영한다. 역광을 피하고 손과 배경이 구분되는 옷과 장소를 사용한다. 클래스마다 촬영자 1인당 10개를 권장하며 거리, 속도, 시작 위치에 자연스러운 차이를 둔다.

### IDLE과 OTHER

- `00_IDLE`: 특정 수어 없이 손을 내리거나 준비 자세로 자연스럽게 대기한다.
- `01_OTHER`: 22개 의료 단어가 아닌 행동이나 수어를 한 번 수행한다. 얼굴 만지기, 옷 정리, 물건 집기, 손 흔들기 등을 다양하게 구성한다.

## 3. 특징 추출과 학습

`models/hand_landmarker.task`와 `models/pose_landmarker_lite.task`가 필요하다. 4초 영상을 60프레임으로 샘플링하고 프레임당 146개 특징(양손 126, 상체 Pose 18, 손 검출 마스크 2)을 만든다.

```bash
python word_sign/extract_word_landmarks.py --frames 60 --overwrite
python word_sign/train_word_lstm.py --output word_sign/training/medical_word_lstm_v1
```

촬영자 한 명의 영상을 통째로 테스트셋에 분리해 새 사람에 대한 성능을 확인한다. 라벨 체계가 변경됐으므로 이전 12클래스 모델은 새 실시간 인식에 그대로 사용하지 않는다.

## 4. 실시간 테스트

```bash
python word_sign/realtime_word_lstm.py
```

`IDLE`, `OTHER`는 문장에 추가하지 않는다. `C`는 누적 결과 삭제, `R`은 4초 버퍼 초기화, `Q`/`Esc`는 종료다.

## 5. GitHub 업로드

원본 `word_dataset/` 영상은 GitHub에 올리지 않는다. 검수한 클래스별 대표 영상 1개씩 총 24개만 `word_sign/examples/`에 복사한다. 촬영자 데이터 병합이 끝나기 전에는 특징 추출과 학습을 실행하지 않는다. 세부 절차는 [`examples/README.md`](examples/README.md)를 따른다.

```bash
git add .gitignore word_sign/README_WORD.md word_sign/examples \
  word_sign/word_labels.txt word_sign/capture_words.py \
  word_sign/word_dataset/metadata.csv
git diff --cached --name-only | grep -E '\.(mp4|mov|avi)$'
git status --short
```

출력된 모든 영상 경로가 `word_sign/examples/`로 시작하는지 확인한 뒤 커밋한다.
