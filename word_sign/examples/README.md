# 약국·병원 수어 촬영 및 대표 영상 공유 가이드

이 폴더는 팀원별 촬영 구도, 길이, 동작 시작 시점과 수어 표현을 맞추기 위한 참고 자료다. 학습용 원본 영상 전체는 GitHub에 올리지 않고, 검수한 클래스별 대표 영상 1개씩 총 24개만 공유한다.

## 클래스별 대표 영상

| 번호 | 클래스 | 대표 영상 |
|---:|---|---|
| 00 | IDLE | [영상 보기](00_IDLE/example_00_IDLE.mp4) |
| 01 | OTHER | [영상 보기](01_OTHER/example_01_OTHER.mp4) |
| 02 | 나 | [영상 보기](02_나/example_02_나.mp4) |
| 03 | 머리 | [영상 보기](03_머리/example_03_머리.mp4) |
| 04 | 배 | [영상 보기](04_배/example_04_배.mp4) |
| 05 | 목 | [영상 보기](05_목/example_05_목.mp4) |
| 06 | 아프다 | [영상 보기](06_아프다/example_06_아프다.mp4) |
| 07 | 어지럽다 | [영상 보기](07_어지럽다/example_07_어지럽다.mp4) |
| 08 | 기침 | [영상 보기](08_기침/example_08_기침.mp4) |
| 09 | 있다 | [영상 보기](09_있다/example_09_있다.mp4) |
| 10 | 가래 | [영상 보기](10_가래/example_10_가래.mp4) |
| 11 | 메스껍다 | [영상 보기](11_메스껍다/example_11_메스껍다.mp4) |
| 12 | 지금 | [영상 보기](12_지금/example_12_지금.mp4) |
| 13 | 어제 | [영상 보기](13_어제/example_13_어제.mp4) |
| 14 | 심하다 | [영상 보기](14_심하다/example_14_심하다.mp4) |
| 15 | 가슴 | [영상 보기](15_가슴/example_15_가슴.mp4) |
| 16 | 열 | [영상 보기](16_열/example_16_열.mp4) |
| 17 | 허리 | [영상 보기](17_허리/example_17_허리.mp4) |
| 18 | 약 | [영상 보기](18_약/example_18_약.mp4) |
| 19 | 콧물 | [영상 보기](19_콧물/example_19_콧물.mp4) |
| 20 | 설사 | [영상 보기](20_설사/example_20_설사.mp4) |
| 21 | 없다 | [영상 보기](21_없다/example_21_없다.mp4) |
| 22 | 구토 | [영상 보기](22_구토/example_22_구토.mp4) |
| 23 | 조금 | [영상 보기](23_조금/example_23_조금.mp4) |

대표 영상은 동작을 확인하는 기준이다. 모든 영상을 대표 영상과 똑같은 속도와 위치로 복제하지 말고, 의미가 바뀌지 않는 범위에서 자연스러운 개인차를 포함한다.

`목`은 신체 부위, `열`은 발열, `메스껍다`는 구역감, `구토`는 실제 토하는 증상이다.

## 팀원 촬영 방법

```bash
source .venv/bin/activate
python word_sign/capture_words.py --signer S005 --duration 4
```

각 팀원은 겹치지 않는 촬영자 ID를 사용한다. 현재 사용한 ID는 `S001`~`S004`이므로 다음 촬영자는 `S005`부터 사용한다.

- 1280×720, 30FPS, 4초
- 0.0~0.5초: 자연스러운 준비 자세
- 0.5~2.5초: 수어를 자연스러운 속도로 한 번 수행
- 2.5~4.0초: 동작을 마치고 대기
- 양쪽 어깨, 팔꿈치와 사용하는 손을 화면 안에 유지
- 클래스당 촬영자 1인 기준 10~12개 권장
- `OTHER`는 목표 의료 단어가 아닌 서로 다른 행동 10~15개
- 잘못 촬영한 영상은 파일명을 직접 바꾸지 말고 즉시 `R`로 제외

## 팀원 데이터 병합 순서

모든 팀원의 촬영이 끝난 뒤 다음 순서로 처리한다.

1. 팀원에게 `word_dataset/S005/`처럼 본인의 촬영자 폴더 전체를 받는다.
2. 촬영자 ID가 기존 폴더와 겹치지 않는지 확인한다.
3. 중앙 프로젝트의 `word_sign/word_dataset/` 아래에 촬영자 폴더를 복사한다.
4. 팀원이 보낸 `metadata.csv`를 중앙 CSV에 그대로 덮어쓰지 않는다.
5. 전체 영상 기준으로 `metadata.csv`를 다시 병합·검증한다.
6. 폴더, 파일명, 라벨 번호, 영상 길이와 재생 여부를 확인한다.
7. 모든 촬영자의 데이터가 확정된 뒤 특징 추출을 한 번 실행한다.

```bash
python word_sign/extract_word_landmarks.py --frames 60 --overwrite
```

팀원 데이터가 아직 추가되는 중이라면 특징 추출과 학습을 실행하지 않는다.

## 대표 영상 관리

GitHub에는 아래 구조의 영상 24개만 둔다.

```text
word_sign/examples/
├── 00_IDLE/example_00_IDLE.mp4
├── 01_OTHER/example_01_OTHER.mp4
├── 02_나/example_02_나.mp4
├── ...
└── 23_조금/example_23_조금.mp4
```

대표 영상을 교체할 때는 원본을 이동하지 않고 복사한다. 목적지 파일명은 `example_<번호>_<클래스>.mp4` 형식을 유지한다. 대표 영상도 수어 동작이 정확하고 손·어깨·팔꿈치가 잘리지 않은 것을 선택한다.

## GitHub 업로드

원본 `word_dataset/`, 전체 특징 CSV, 백업 파일은 GitHub에 올리지 않는다. 문서와 대표 영상 24개만 스테이징한다.

```bash
git add word_sign/README_WORD.md \
  word_sign/examples/README.md \
  word_sign/examples \
  word_sign/word_labels.txt \
  word_sign/capture_words.py \
  word_sign/word_dataset/metadata.csv
```

현재 대표 영상이 정확히 24개인지 확인한다.

```bash
find word_sign/examples -type f | grep -E '\.(mp4|mov|avi)$' | wc -l
```

결과가 `24`여야 한다. 이어서 스테이징된 영상 변경 경로를 확인한다. 기존 4개 대표 영상은 이미 Git에 있으므로 새로 추가되는 영상 수만 세면 24보다 작을 수 있고, 이전 클래스 영상 삭제도 함께 표시될 수 있다.

```bash
git diff --cached --name-status | grep -E '\.(mp4|mov|avi)$'
git diff --cached --name-only | grep -E '\.(mp4|mov|avi)$' | grep -v '^word_sign/examples/'
git status --short
```

두 번째 명령은 아무것도 출력되지 않아야 한다. 다른 경로가 보이면 커밋하지 말고 해당 파일만 스테이징에서 제외한다.

```bash
git restore --staged 잘못_추가된_영상_경로
```

확인이 끝나면 커밋하고 원격 저장소에 올린다.

```bash
git commit -m "docs: add 24 medical sign examples"
git push
```

GitHub는 단일 파일 100MB를 초과하면 일반 업로드를 거부한다. 대표 영상은 짧은 4초 영상만 사용하고, 원본 전체 데이터셋은 별도 공유 저장소를 사용한다.
