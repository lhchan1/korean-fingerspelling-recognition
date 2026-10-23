# 120프레임 모델 오프라인 비교 실험

목적: v10 seed 44의 **동일한 원본 영상 1,574개, 동일한 학습/검증/테스트 영상 ID, 동일한 모델 구조와 학습 설정**을 사용하고 입력 길이를 60→120프레임으로 바꿔 비교한다. S005는 기존 v10과 똑같이 제외한다. 기존 60프레임 특징/모델과 배포 웹은 수정하지 않는다.

## VS Code 터미널에서 직접 실행

아래 명령은 프로젝트 루트에서 실행한다. 특징 추출과 학습은 시간이 오래 걸리므로 각각 완료됐는지 확인한 뒤 다음 단계로 간다.

```bash
cd /Users/heechan/visualcodeworkspace/Capstone/fingerspelling_capture
source .venv/bin/activate
python word_sign/experiment_120_frames.py prepare
```

이 명령은 v10 seed 44의 `split.csv`에 있는 1,574개 ID만 뽑아 별도 메타데이터를 만든다. 원본 영상이나 기존 메타데이터를 변경하지 않는다.

```bash
python word_sign/extract_word_landmarks.py \
  --dataset word_sign/word_dataset \
  --metadata word_sign/training/word_lstm_120f_seed44_experiment/metadata_120f.csv \
  --output word_sign/word_features_120f_seed44_experiment \
  --frames 120 \
  --cpu
```

완료되면 아래 두 숫자를 확인한다. 첫 줄은 헤더 포함 **188,881행**(1,574×120+1), 둘째 줄은 헤더 포함 **1,575행**이어야 한다.

```bash
wc -l word_sign/word_features_120f_seed44_experiment/landmarks.csv
wc -l word_sign/word_features_120f_seed44_experiment/quality_report.csv
```

그다음에만 모델을 학습한다.

```bash
python word_sign/experiment_120_frames.py train
```

학습 후 결과를 본다.

```bash
python -m json.tool word_sign/training/word_lstm_120f_seed44_experiment/seed_44/metrics.json
python -m json.tool word_sign/training/word_lstm_v10_belly_seed_sweep_20261018/seed_44/metrics.json
```

120프레임 결과물은 `word_sign/training/word_lstm_120f_seed44_experiment/seed_44/`의 `best.keras`, `metrics.json`, `history.csv`, `confusion_matrix.json`, `test_predictions.csv`에 저장된다. 60프레임 기준 모델은 기존 v10 seed 44 파일을 그대로 참조한다.

## 해석할 때 주의

- 같은 영상과 같은 split을 사용하지만, 120프레임에서 MediaPipe 검출 품질이 달라질 수 있고 `--cpu` 실행 환경도 과거 추출 환경과 다를 수 있다. 따라서 점수 차이를 프레임 수만의 인과 효과로 단정하지 않는다. `quality_report.csv`도 같이 확인한다. 짝 비교를 위해 60프레임에서 채택한 1,574개를 120프레임에서도 전부 사용하며, 새 품질 판정으로 임의로 빼지 않는다.
- 두 모델 모두 테스트셋을 이미 모델 선택에 사용한 탐색적 실험이다. 점수가 높아도 새 촬영자/새 영상에서 추가 검증이 필요하다.
- `best.keras`는 학습용 모델이다. **이 실험은 웹 배포를 바꾸지 않는다.** 웹 적용 전에는 브라우저가 4초 동안 손·상체 특징을 실제로 약 30fps로 안정적으로 추출하는지 확인해야 한다. 기존 약 15fps 수집을 그대로 120프레임으로 늘리면 약 8초 입력이 되어 학습의 4초 영상과 달라진다.
- 특징 추출이나 학습이 중단되면 결과가 부분적으로 남을 수 있다. 이 스크립트는 기존 출력을 자동으로 덮어쓰거나 이어 쓰지 않는다. 파일 상태를 확인한 뒤 재실행 여부를 결정한다.

## 2026-10-21 중단 지점에서 이어하기

현재 원본 추출 결과에는 **772개 영상의 120프레임이 완료**되어 있고, 다음 영상 `S003_20260920_182058_15_002`는 **62/120프레임만 기록**되어 있다. 따라서 중단된 한 영상은 처음부터 다시 추출하고, 완료된 772개는 그대로 재사용한다. 원본 `word_features_120f_seed44_experiment/`는 수정하거나 삭제하지 않는다.

집에서 VS Code 터미널에 아래 순서대로 입력한다.

```bash
cd /Users/heechan/visualcodeworkspace/Capstone/fingerspelling_capture
source .venv/bin/activate
python word_sign/experiment_120_frames.py resume-prepare
```

위 명령은 기존 CSV를 검사하고 **나머지 802개 영상**의 메타데이터만 새 파일로 만든다. 출력에 `완료: 772/1574`, `부분 기록: ... 62/120프레임`, `재추출 대상: 802개`가 나타나는지 확인한다. 숫자가 다르면 추가 실행 전에 현재 파일 상태를 다시 확인한다.

```bash
python word_sign/extract_word_landmarks.py \
  --dataset word_sign/word_dataset \
  --metadata word_sign/training/word_lstm_120f_seed44_experiment/metadata_remaining_120f.csv \
  --output word_sign/word_features_120f_seed44_resume_part2 \
  --frames 120 \
  --cpu
```

출력이 `[802/802]`까지 끝나면 줄 수를 확인한다. 헤더를 포함해 특징 **96,241행**(802×120+1), 품질 보고서 **803행**이어야 한다.

```bash
wc -l word_sign/word_features_120f_seed44_resume_part2/landmarks.csv
wc -l word_sign/word_features_120f_seed44_resume_part2/quality_report.csv
```

이어하기 결과를 검증한 뒤, 기존 완료분과 **새 폴더로 통합**한다. 부분 기록 62행은 통합하지 않고 새로 추출한 120행으로 대체한다.

```bash
python word_sign/experiment_120_frames.py resume-merge
wc -l word_sign/word_features_120f_seed44_complete/landmarks.csv
wc -l word_sign/word_features_120f_seed44_complete/quality_report.csv
```

통합 결과는 헤더 포함 각각 **188,881행**, **1,575행**이어야 한다. 그다음에만 모델 학습을 시작한다.

```bash
python word_sign/experiment_120_frames.py train \
  --features word_sign/word_features_120f_seed44_complete
```

중요: 기존 `word_features_120f_seed44_experiment`에 `--overwrite`를 사용하거나 처음부터 다시 추출하지 않는다. 이어하기가 다시 중단되면 임의로 CSV를 합치지 말고 완료 영상 수를 먼저 확인한다.
