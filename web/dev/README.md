# 개발 전용 테스트

이 폴더의 HTML 파일은 로컬 모델·시퀀스·안정화 검증용입니다.
Cloudflare Pages의 일반 사용자 링크로 안내하지 않습니다.

- `/dev/model_test.html`
- `/dev/sequence_test.html`
- `/dev/recognition_test.html`

연속 단어 누적·IDLE 종료·가변 길이 WebSocket payload 회귀 테스트는
`node dev/recognition_sequence_test.mjs`로 실행합니다. 실제 수어 인식률 검증은
카메라 또는 별도 녹화 데이터로 진행해야 합니다.

