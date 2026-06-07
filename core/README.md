# core — 공통 라이브러리
파이프라인·서빙 공유. 외부 의존(LLM/임베딩/벡터스토어)은 여기서 캡슐화.

- `config/` 설정(.env)  `models/` pydantic 스키마
- `metadata/` corpus_index·OpenAlex/Crossref 클라이언트
- `citation/` 인용 overlap 검증
