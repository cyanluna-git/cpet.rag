# serving — 질의 API (FastAPI + LlamaIndex)

질의 경로: 번역(KO→EN) → 하이브리드(BM25+벡터) → Rerank → Bedrock Claude(Strict Citation) → 인용 검증 → 역번역(EN→KO).

- `retrieval/` query translation · hybrid · rerank
- `generation/` Bedrock Claude · strict citation 프롬프트
- `app/` `router/` FastAPI (Router→Service→Repository)
