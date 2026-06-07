# 아키텍처

상세 설계·근거는 **[RAG_IMPLEMENTATION_PLAN.md](RAG_IMPLEMENTATION_PLAN.md)** (v2), 결정 요약은 **[decisions/0001-tech-stack.md](decisions/0001-tech-stack.md)**.

```
[ 인입 — 이벤트·증분 ]
 PDF → S3(raw) ─event→ Lambda → Batch(GPU g5)
    Docling(+Gemini VLM 폴백) → Late Chunking 임베딩(Jina-v3) → 메타결합
    → LanceDB@S3 (papers·chunks·vec·FTS) + parsed md/ (processed dedup)

[ 서빙 — API ]
 사용자(웹/Claude Code) → API GW → ECS Fargate(FastAPI+LlamaIndex)
    KO→EN 번역(용어보호) → 하이브리드(BM25+벡터 top50) → Rerank top8
    → Bedrock Claude(Strict Citation) → 인용 overlap 검증 → EN→KO

 인증 Cognito · 관측 CloudWatch · 평가 RAGAS(배치)
 Obsidian: parsed md/ → 로컬 읽기전용, 메모는 Git
```

## 코드 ↔ 인프라 매핑
| 코드 | AWS |
|------|-----|
| `ingestion/` | Lambda(트리거) + Batch(GPU) |
| `serving/` | ECS Fargate + API GW/ALB |
| `core/` (LLM·임베딩·스토어 클라이언트) | Bedrock · Jina · LanceDB@S3 |
| `infra/terraform` | 전체 프로비저닝 |
| `eval/` | RAGAS 배치 (CI) |
