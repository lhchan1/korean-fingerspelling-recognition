# 브라우저 성능 로그 (browser-timing-v1)

카메라 시작 후 로그의 `성능 측정 활성화`에서 모델 버전, TF.js 버전, backend,
브라우저 환경을 먼저 기록합니다. MediaPipe CPU/GPU 환경은 기존 초기화 로그에 나옵니다.
서로 다른 모델/기기/브라우저의 수치를 한 평균으로 합치지 않습니다.

MediaPipe와 모델은 약 5초마다 요약하고, 멈추거나 추론을 초기화할 때 남은 표본을 출력합니다.
각 요약의 samples는 해당 구간 표본 수이며 mean/median/p95/min/max의 단위는 ms입니다.
카메라 시작 시 최초 5개 특징 추출, 추론 버퍼 초기화 후 최초 3개 모델 추론을
워밍업 표본으로 제외합니다. 모델 준비 시 별도의 0 입력 워밍업도 결과 읽기까지 완료합니다.

## MediaPipe 처리 시간 요약

- handMs: Hand Landmarker detectForVideo 호출부터 반환까지
- poseMs: Pose Landmarker detectForVideo 호출부터 반환까지
- landmarkMs: 손 및 상체 호출의 합산 경과시간
- featureMs: 반환 좌표를 146차원 정규화 특징으로 변환하는 시간
- 그리기, 카메라 프레임 대기, LSTM 추론, 로그 출력 시간은 제외

## 단어 모델 추론 시간 요약

- inputPrepMs: 60프레임 배열 평탄화와 입력 tensor 생성
- modelMs: predict 호출부터 dataSync 결과 읽기 완료까지 (GPU 결과 대기 포함)
- totalMs: 입력 준비, 모델 실행, 배열 변환 및 tidy 정리까지. 순위 정렬/화면 출력 제외
- 카메라 수집, 250ms 호출 간격, IDLE 대기, 모델 다운로드를 포함하지 않음

## 서버 요청–응답 시간 측정

한 요청마다 sessionId와 wordCount를 포함해 출력합니다.

- connectMs: WebSocket 생성 직전부터 open 이벤트까지
- sendToResponseMs: send 직전부터 같은 세션의 유효한 result 메시지 수신까지
- requestResponseMs: WebSocket 생성 직전부터 유효한 result 수신까지 (연결 포함)
- elapsedMs: 요청 시작부터 검증 및 완료 처리 진입까지, 오류 요청에도 기록
- serverGenerationMs: null. 서버 내부 계산 시간을 브라우저에서 추정하지 않음
- status:error는 성공 요청 평균에서 제외. 잘못된 세션 결과는 성공으로 세지 않음

단어 입력/IDLE 대기 및 결과 화면 렌더링 시간은 요청–응답에 포함되지 않습니다.
sendToResponseMs는 네트워크, 서버 대기/처리, 브라우저 이벤트 처리 지연을 포함하므로
순수 네트워크 RTT나 KoBART 생성시간이라고 표기하지 않습니다.

## 발표용 측정

동일 기기·브라우저·모델을 유지하고 페이지를 전면에 둡니다. 화면 로그를 지운 뒤
30초 이상 수어를 수행하고 로그를 복사합니다. 로그 지우기는 통계 상태를 초기화하지
않으므로 깨끗한 새 세션이 필요하면 카메라를 정지/재시작합니다.
서버는 동일한 입력 단어 배열로 여러 번 측정하고 각 요청의 표본 수와 환경을 기록합니다.
구간 평균을 합칠 때는 samples로 가중 평균합니다. 중앙값/p95는 구간 값의 평균으로
전체 분포를 복원할 수 없습니다. 페이지가 가려지면 기존 추론 일시정지 동작을 유지합니다.

자동 검증 (실제 카메라/LLM 서버를 사용하지 않는 모의시계 테스트):
`node --test web/dev/timing_test.mjs`
