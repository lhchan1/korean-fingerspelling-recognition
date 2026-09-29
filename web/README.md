# Capstone Web

브라우저 카메라와 온디바이스 수어 인식을 검증하기 위한 독립 웹 프로젝트입니다.
기존 Python 학습 프로젝트와 데이터셋은 이 폴더에 복사하지 않습니다.

현재 배포 모델은 `word_lstm_v7_demo_randomsplit`입니다. 원본은
`word_sign/training/word_lstm_v7_demo_randomsplit/word_sign_lstm.keras`이며,
24클래스·56,888 파라미터 모델을 TensorFlow.js 가중치로 변환해 사용합니다.

## 현재 구현

- 전면 카메라 시작
- 전면/후면 카메라 전환
- 카메라 종료 및 리소스 해제
- 권한 거부와 카메라 오류 안내
- 모바일 반응형 화면
- 전면 카메라 미리보기만 좌우 반전
- MediaPipe Hand Landmarker로 최대 두 손 실시간 검출
- 손 관절과 연결선 오버레이 및 처리 FPS 표시
- 어깨·팔꿈치·손목 Pose 실시간 검출
- 학습 코드와 동일한 왼손 63 + 오른손 63 + Pose 18 + 마스크 2 특징 생성
- 15FPS로 60프레임 특징 버퍼 구성
- 24클래스 TensorFlow.js 모델 실시간 추론 및 상위 3개 결과 표시
- 신뢰도 75% 이상 예측을 최근 5회 중 3회 다수결로 안정화
- IDLE/OTHER 전환 기반 동일 동작 중복 방지
- 확정 단어 누적, 마지막 단어 취소, 전체 지우기

카메라 영상은 서버로 전송하지 않습니다.

## 로컬 실행

Python 3가 설치된 환경에서 다음 명령을 실행합니다.

```bash
cd ~/visualcodeworkspace/capstone-web
python3 dev_server.py
```

브라우저에서 <http://localhost:5173>을 열고 `카메라 시작`을 누릅니다.
브라우저의 단계별 MediaPipe 로그가 페이지 하단과 실행한 터미널에 동시에 표시됩니다.

> 카메라 API는 `localhost` 또는 HTTPS 환경에서만 사용할 수 있습니다.

## 디렉터리 구조

```text
capstone-web/
├── public/
│   └── models/
│       └── word-sign/    # 변환된 TensorFlow.js 모델을 둘 위치
├── src/
│   ├── camera.js         # 카메라 생명주기와 오류 처리
│   ├── main.js           # 화면 상태와 이벤트 연결
│   ├── recognition.js    # 예측 안정화와 단어 확정
│   └── style.css
├── index.html
└── README.md
```

## 다음 작업

1. 실제 수어 영상으로 임계값과 안정화 횟수 조정
2. 모바일 성능과 정확도 검증
3. 확정된 단어만 문장 생성 API로 전송
