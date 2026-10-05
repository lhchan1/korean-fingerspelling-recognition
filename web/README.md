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
- 확정 단어가 있는 상태에서 IDLE이 3초 유지되면 문장을 자동 확정
- 고정 WSS 주소의 KoBART FastAPI 서버로 확정 단어 배열만 자동 전송
- 연결·처리 중 중복 요청 방지 및 전체 요청 30초 제한
- 성공 응답의 `session_id`, `type`, `sentence` 검증 후 생성 문장 표시
- 성공 시 전송한 단어만 제거하고 처리 중 새로 인식한 단어는 보존
- 실패 시 단어를 유지하며 같은 IDLE 구간에서 자동 재시도하지 않음
- 모바일에서는 MediaPipe CPU delegate와 640×480 카메라를 사용해 WebGL 호환성을 확보
- MediaPipe 프레임 오류가 발생해도 검출 루프를 유지하고 화면 로그에 원인 표시

카메라 영상·이미지·랜드마크 좌표·146차원 특징 벡터는 서버로 전송하지 않습니다.
서버에는 `{ session_id, words }` 형식의 텍스트 JSON만 전송합니다.

KoBART WebSocket 주소는 `src/kobart-client.js`의
`KOBART_WEBSOCKET_URL` 상수에서 변경할 수 있습니다.

## 실행 구조

배포 서버에는 HTML/CSS/JavaScript와 함께 TensorFlow.js v7 가중치,
MediaPipe `.task`, WASM 파일을 올립니다. 사용자가 처음 접속하면 브라우저가
이 정적 파일들을 내려받고, 카메라 영상과 146차원 특징 추론은 사용자 기기
안에서 처리합니다. 현재 웹은 카메라 영상이나 랜드마크를 서버로 전송하지 않습니다.

## 로컬 개발 실행

Python 3가 설치된 환경에서 다음 명령을 실행합니다.

```bash
cd ~/visualcodeworkspace/capstone-web
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
버전된 v7 가중치의 장기 캐시를 설정합니다. `model.json`, `config.json`,
`labels.json`은 재배포 시 갱신되도록 캐시하지 않습니다.
TensorFlow.js 4.22 번들이 런타임에 동적 함수를 생성하므로 `script-src`에는
`'unsafe-eval'`이 포함됩니다. 외부 스크립트 출처는 허용하지 않고 `'self'`만 유지합니다.

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
│       └── word-sign/    # v7 TensorFlow.js 모델과 가중치
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
