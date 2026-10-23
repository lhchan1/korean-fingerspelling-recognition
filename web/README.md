# Capstone Web

## 현재 설정 (2026-10-24 확인)

성능 측정이 포함된 테스트 웹입니다. 공개 주소: https://korean-fingerspelling-recognition.pages.dev/ . 60프레임을 모두 모은 후 추론하며, 단어 확정 후에는 안정적인 IDLE/OTHER가 있어야 다음 단어를 받습니다. 시연본의 조기 추론·직접 단어 전환 규칙과 다릅니다.

브라우저 카메라와 온디바이스 수어 인식을 검증하기 위한 독립 웹 프로젝트입니다.
기존 Python 학습 프로젝트와 데이터셋은 이 폴더에 복사하지 않습니다.

현재 모델은 `word-lstm-v10-belly-seed44`입니다.
원본은 `word_sign/training/word_lstm_v10_belly_seed_sweep_20261018/seed_44/best.keras`이며,
60×146 입력·24클래스·56,888 파라미터입니다. 웹에서는 `model.json`과
`word-lstm-v10-seed44.weights.bin`을 사용합니다.

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
- 확정 단어가 있는 상태에서 IDLE이 3초 유지되면 문장을 자동 확정
- 고정 WSS 주소의 KoBART FastAPI 서버로 확정 단어 배열만 자동 전송
- 연결·처리 중 중복 요청 방지 및 전체 요청 30초 제한
- 성공 응답의 `session_id`, `type`, `sentence` 검증 후 생성 문장 표시
- 성공 시 전송한 단어만 제거하고 처리 중 새로 인식한 단어는 보존
- 실패 시 단어를 유지하며 같은 IDLE 구간에서 자동 재시도하지 않음
- `IDLE`, `OTHER`를 제외한 22개 모델 단어 중 서로 다른 3개를 배열로 보내는 통신 테스트 버튼
- 직접 입력한 문장을 공백 기준으로 나눠 문자열 배열로 보내는 통신 테스트 입력창
- 모바일에서는 MediaPipe CPU delegate와 640×480 카메라를 사용해 WebGL 호환성을 확보
- MediaPipe 프레임 오류가 발생해도 검출 루프를 유지하고 화면 로그에 원인 표시

카메라 영상·이미지·랜드마크 좌표·146차원 특징 벡터는 서버로 전송하지 않습니다.
서버에는 `{ session_id, words }` 형식의 텍스트 JSON만 전송합니다.

KoBART WebSocket 주소는 `src/kobart-client.js`의
`KOBART_WEBSOCKET_URL` 상수에서 변경할 수 있습니다.

## 실행 구조

배포 서버에는 HTML/CSS/JavaScript와 함께 TensorFlow.js v10 가중치,
MediaPipe `.task`, WASM 파일을 올립니다. 사용자가 처음 접속하면 브라우저가
이 정적 파일들을 내려받고, 카메라 영상과 146차원 특징 추론은 사용자 기기
안에서 처리합니다. 현재 웹은 카메라 영상이나 랜드마크를 서버로 전송하지 않습니다.

## 로컬 개발 실행

Python 3가 설치된 환경에서 다음 명령을 실행합니다.

```bash
cd /Users/heechan/visualcodeworkspace/korean-fingerspelling-recognition-deploy/web
python3 dev_server.py
```

브라우저에서 <http://localhost:5173>을 열고 `카메라 시작`을 누릅니다.
브라우저의 단계별 MediaPipe 로그가 페이지 하단과 실행한 터미널에 동시에 표시됩니다.
`/api/log` 전송은 `localhost`, `127.0.0.1`, `::1`에서만 활성화됩니다.
공개 배포 환경에서는 화면과 브라우저 콘솔에만 로그가 남습니다.

> 카메라 API는 `localhost` 또는 HTTPS 환경에서만 사용할 수 있습니다.

개발 전용 테스트 페이지는 `dev/`에 있습니다.

- <http://localhost:5173/dev/model_test.html>
- <http://localhost:5173/dev/sequence_test.html>
- <http://localhost:5173/dev/recognition_test.html>

## Cloudflare Pages 공개 배포

이 폴더를 GitHub 저장소의 `web/`에 넣은 다음 Cloudflare Pages에서 저장소를
연결합니다.

```text
Production branch: main
Root directory: web
Build command: 비움 (필요하면 exit 0)
Build output directory: .
```

배포 후 발급되는 `https://<project>.pages.dev` 주소에서 카메라를 테스트합니다.
`_headers`는 카메라 권한을 같은 출처로 제한하고, 보안 헤더·WASM MIME 타입·
버전된 v10 가중치의 장기 캐시를 설정합니다. `model.json`, `config.json`,
`labels.json`은 재배포 시 갱신되도록 캐시하지 않습니다.
TensorFlow.js 4.22 번들이 런타임에 동적 함수를 생성하므로 `script-src`에는
`'unsafe-eval'`이 포함됩니다. 스크립트는 같은 출처와 Cloudflare Insights에 한정하며, 연결 대상은 `_headers`의 CSP에서 확인합니다.

`_headers`의 CSP `connect-src`에는
`wss://sign-kobart.duckdns.org`만 KoBART 연결 대상으로 허용합니다.
배포 전에 비밀정보 검사를 다시 실행하고, API 키는 절대로 이 정적 폴더에
저장하지 않습니다.

## 디렉터리 구조

```text
capstone-web/
├── _headers             # Cloudflare 보안·캐시·WASM 헤더
├── dev/                 # 개발 전용 테스트 페이지
├── public/
│   └── models/
│       └── word-sign/    # v10 TensorFlow.js 모델과 가중치
├── src/
│   ├── camera.js         # 카메라 생명주기와 오류 처리
│   ├── kobart-client.js  # KoBART WebSocket 요청·응답 검증
│   ├── main.js           # 화면 상태와 이벤트 연결
│   ├── recognition.js    # 예측 안정화와 단어 확정
│   └── style.css
├── index.html
└── README.md
```

## KoBART 서버 연결 동작

확정 단어가 하나 이상인 상태에서 안정적인 `IDLE` 예측이 3초 유지되면 자동으로
한 번 요청합니다. 문장 완성 버튼은 사용하지 않습니다. 성공했을 때만 그 요청에
포함된 단어를 제거하며, 요청 처리 중 추가된 단어는 다음 문장용으로 남깁니다.
실패한 요청은 같은 IDLE 구간에서 자동 반복하지 않고 새 수어 동작 후 다음 IDLE을
기다립니다. VM이 꺼져 있거나 시작 직후라면 연결 실패할 수 있으므로 서버 모델
로딩과 DNS 갱신이 끝난 뒤 다시 동작을 입력합니다.

---

## 개발 전용 테스트

이 폴더의 HTML 파일은 로컬 모델·시퀀스·안정화 검증용입니다.
Cloudflare Pages의 일반 사용자 링크로 안내하지 않습니다.

- `/dev/model_test.html`
- `/dev/sequence_test.html`
- `/dev/recognition_test.html`


---

## 브라우저 성능 로그 (browser-timing-v1)

카메라 시작 후 로그의 `성능 측정 활성화`에서 모델 버전, TF.js 버전, backend,
브라우저 환경을 먼저 기록합니다. MediaPipe CPU/GPU 환경은 기존 초기화 로그에 나옵니다.
서로 다른 모델/기기/브라우저의 수치를 한 평균으로 합치지 않습니다.

MediaPipe와 모델은 약 5초마다 요약하고, 멈추거나 추론을 초기화할 때 남은 표본을 출력합니다.
각 요약의 samples는 해당 구간 표본 수이며 mean/median/p95/min/max의 단위는 ms입니다.
카메라 시작 시 최초 5개 특징 추출, 추론 버퍼 초기화 후 최초 3개 모델 추론을
워밍업 표본으로 제외합니다. 모델 준비 시 별도의 0 입력 워밍업도 결과 읽기까지 완료합니다.

### MediaPipe 처리 시간 요약

- handMs: Hand Landmarker detectForVideo 호출부터 반환까지
- poseMs: Pose Landmarker detectForVideo 호출부터 반환까지
- landmarkMs: 손 및 상체 호출의 합산 경과시간
- featureMs: 반환 좌표를 146차원 정규화 특징으로 변환하는 시간
- 그리기, 카메라 프레임 대기, LSTM 추론, 로그 출력 시간은 제외

### 단어 모델 추론 시간 요약

- inputPrepMs: 60프레임 배열 평탄화와 입력 tensor 생성
- modelMs: predict 호출부터 dataSync 결과 읽기 완료까지 (GPU 결과 대기 포함)
- totalMs: 입력 준비, 모델 실행, 배열 변환 및 tidy 정리까지. 순위 정렬/화면 출력 제외
- 카메라 수집, 250ms 호출 간격, IDLE 대기, 모델 다운로드를 포함하지 않음

### 서버 요청–응답 시간 측정

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

### 발표용 측정

동일 기기·브라우저·모델을 유지하고 페이지를 전면에 둡니다. 화면 로그를 지운 뒤
30초 이상 수어를 수행하고 로그를 복사합니다. 로그 지우기는 통계 상태를 초기화하지
않으므로 깨끗한 새 세션이 필요하면 카메라를 정지/재시작합니다.
서버는 동일한 입력 단어 배열로 여러 번 측정하고 각 요청의 표본 수와 환경을 기록합니다.
구간 평균을 합칠 때는 samples로 가중 평균합니다. 중앙값/p95는 구간 값의 평균으로
전체 분포를 복원할 수 없습니다. 페이지가 가려지면 기존 추론 일시정지 동작을 유지합니다.

자동 검증 (저장소 루트에서 실행, 실제 카메라/LLM 서버를 사용하지 않는 모의시계 테스트):
`node --test web/dev/timing_test.mjs`

---

## 시스템과 통신 계약 (기존 read.md 통합)

Cloudflare Pages는 정적 웹·MediaPipe·TensorFlow.js 모델을 전달합니다. 사용자 브라우저가 카메라와 단어 추론을 처리하고, KoBART FastAPI 서버는 확정된 단어 배열로 문장을 만듭니다.

요청 예시:

```json
{"session_id":"web-고유ID","words":["나","아프다","머리"]}
```

응답 예시:

```json
{"type":"result","session_id":"web-고유ID","input_words":["나","아프다","머리"],"sentence":"저는 머리가 아픕니다."}
```

카메라 영상·이미지·랜드마크·146차원 특징·전체 확률은 KoBART에 전송하지 않습니다. 요청별 session_id를 검증하고, 성공 시 전송한 단어만 제거합니다. 처리 중 인식한 단어는 보존합니다. IDLE/OTHER는 서버 전송에서 제외합니다.

## 과거 통신 검증 기록과 한계

아래는 기존 read.md의 기록을 보존한 것입니다. 당시 모델은 v7이었으며, 수치와 성공 사례를 이번 점검에서 서버에 재요청해 재측정하지 않았습니다. 현재 서버 가동 상태나 현 모델 성능을 보장하지 않습니다.

### 7. 통신 검증 결과

Cloudflare에 배포된 웹과 KoBART 서버 사이의 WSS 연결 및 왕복 통신을 확인했습니다.

| 전송 단어 | 서버 반환 문장 | 결과 |
|---|---|:---:|
| `어제, 기침, 콧물, 열, 가래, 있다` | `어제 기침, 콧물, 열, 가래가 있었습니다.` | 성공 |
| `어지럽다, 없다, 조금` | `조금 어지럽습니다.` | 성공 |
| `나, 아프다, 머리` | `저는 머리가 아픕니다.` | 성공 |

서버 로그에서도 WebSocket 연결, JSON 수신, KoBART 추론 및 결과 반환을 확인했습니다. 예시 요청의 서버 추론 시간은 약 0.286~0.663초였습니다.

웹 모델에서는 다음 항목을 확인했습니다.

- TensorFlow.js 모델 로드 성공
- 입력 형태 `[batch, 60, 146]` 확인
- 출력 형태 `[batch, 24]` 확인
- 60프레임 시퀀스 및 다중 단어 배열 전송 확인

---

### 8. 현재 한계와 향후 과제

- 스마트폰 성능에 따라 MediaPipe와 TensorFlow.js 처리 속도가 달라질 수 있습니다.
- 좌표 기반 모델이지만 MediaPipe 검출은 조명, 손 가림, 촬영 거리와 각도의 영향을 받을 수 있습니다.
- 모델의 실제 성능은 학습에 포함되지 않은 촬영자와 환경으로 추가 검증해야 합니다.
- 짧거나 문맥이 부족한 단어 배열에서는 KoBART 문장 품질이 낮아질 수 있습니다.
- 협업 서버의 VM이 종료되거나 모델을 로딩 중이면 문장 생성 연결이 불가능합니다.
- 다중 사용자 환경을 위한 서버 부하 및 동시 요청 테스트가 추가로 필요합니다.

향후에는 모바일 기기별 성능 측정, 촬영자 독립 정확도 평가, KoBART 생성 품질 개선, 서버 상태 확인 및 동시 사용자 부하 테스트를 진행할 예정입니다.

---
