# ADR 0001 — RAG 기술 스택

- 상태: 채택 (2026-06-07)
- 근거: `docs/RAG_IMPLEMENTATION_PLAN.md` (v2) + Google Deep Research 리뷰(`docs/reviews/google_deep_research.md`)

## Context
운동생리·CPET 학술 PDF(794편, 증가) 대상 AWS 서버형 멀티유저 RAG. 한국어 질의→영문 논문. 인용 정확·증분 인입·저비용 필요.

## Decision
| 영역 | 채택 | 비고 |
|------|------|------|
| 파싱 | **Docling** + Gemini Flash VLM 폴백 | 복잡 표/수식만 VLM |
| 청킹 | **Late Chunking** | adaptive·early 폐기 |
| 임베딩 | **Jina-embeddings-v3** 단일 | late chunking 네이티브·MRL (SPECTER2 제거) |
| 벡터스토어 | **LanceDB on S3** | append-only·동시쓰기 안전. 대안 OpenSearch/pgvector |
| 검색 | Query Translation 샌드위치 + 하이브리드(BM25+벡터) + Rerank | 한↔영, Cohere/Qwen3 rerank |
| 생성/인용 | **Bedrock Claude + Strict Citation** | overlap 후처리 검증 |
| 평가 | **RAGAS** | Faithfulness·Context P/R |
| 오케스트레이션 | **LlamaIndex** | 커스텀 스크립트 대체 |
| 배포 | AWS: S3·Lambda·Batch(GPU)·ECS Fargate·Bedrock·Cognito | IaC Terraform/CDK |

## Consequences
- (+) 동시성·손상 위험 제거, 인용 신뢰성, 증분 저비용, 표준 프레임워크.
- (−) AWS 의존·초기 셋업 비용. 개발은 Colab PoC로 시작 후 이관.
- 거부: sqlite-vec(Drive 공유), SPECTER2 이원화, adaptive chunking, qmd 커스터마이즈.
