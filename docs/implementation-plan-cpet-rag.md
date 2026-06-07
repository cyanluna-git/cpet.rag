# cpet.rag — Implementation Plan

> 기반: SRS(`docs/srs-cpet-rag.md`) + 계획서 v2. Rolling Wave 실행(스켈레톤 카드 → JIT refine).

## Epic 구조
| Epic | 범위 | 디렉토리 |
|------|------|----------|
| **foundation** | 설정·스키마·메타·벡터스토어 인터페이스 | `core/` |
| **ingestion** | 파싱·청킹·임베딩·적재·파이프라인 | `ingestion/`, `notebooks/` |
| **retrieval** | 번역·하이브리드·리랭크 | `serving/retrieval/` |
| **generation** | 생성·인용·역번역 | `serving/generation/`, `core/citation/` |
| **serving** | API·오케스트레이션 | `serving/app/` |
| **eval** | RAGAS·평가셋 | `eval/` |
| **infra** | AWS·IaC·배포 | `infra/` |
| **frontend** | 챗봇 UI(선택) | `frontend/` |

## Phase 순서 (Rolling Wave)
```
P0 Foundation+Baseline → P1 Ingestion(Colab PoC) → P2 Retrieval+Generation
→ P3 Serving+Eval(+PoC E2E) → P4 AWS 운영화(+AWS E2E) → P5 Frontend(선택)
```

## 스토리(카드) 정의 — 31개

### Phase 0 — Foundation & Baseline
- **T1** [L2] core 패키지 골격+설정 — pyproject 의존성 확정, `core/config`(.env), import 구조
- **T2** [L1] 도메인 스키마 — `core/models`: Paper·Chunk·RetrievedChunk·Query req/res
- **T3** [L2] 메타데이터 로더 — `core/metadata`: corpus_index→Paper, OpenAlex/Crossref 보강
- **T4** [L2] 벡터스토어 인터페이스+LanceDB — 추상 VectorStore + LanceDB(로컬→S3)
- **T5** [L1] Colab PoC 노트북 골격 — deps·스토리지 마운트
- **T6** [L2] RAGAS 평가셋 — 복잡 논문 10편 + QA 20~30문항

### Phase 1 — Ingestion (Colab PoC)
- **T7** [L2] PDF 파싱(Docling) — PDF→markdown+구조 JSON
- **T8** [L2] VLM 폴백 파싱 — 복잡 표/수식 Gemini Flash
- **T9** [L2] Late Chunking 임베딩 — 구조경계+메타 프리펜드+Jina-v3
- **T10** [L2] 벡터스토어 적재 — chunks/vec/FTS upsert + processed dedup
- **T11** [L3] IngestionPipeline 통합(LlamaIndex) — 증분, 샘플 10편 E2E
- **T12** [L2] Obsidian Vault 생성 — md frontmatter + 인용 wikilink(OpenAlex)

### Phase 2 — Retrieval & Generation
- **T13** [L2] Query Translation 샌드위치 — KO→EN + 의학용어 보호 사전
- **T14** [L2] 하이브리드 검색 — BM25∪벡터 top50 병합 + 메타필터
- **T15** [L2] Reranker 통합 — Cohere/Qwen3 → top8
- **T16** [L2] 생성+Strict Citation — Bedrock Claude + 엄격 인용 프롬프트
- **T17** [L2] 인용 overlap 검증 — `core/citation`: 인용↔원문, 미검증 차단
- **T18** [L1] EN→KO 역번역 + 용어 복원

### Phase 3 — Serving & Eval
- **T19** [L3] 질의 오케스트레이션 — 번역→하이브리드→리랭크→생성→검증→역번역 연결
- **T20** [L2] FastAPI 서빙 — /query(스트리밍), Router→Service→Repo
- **T21** [L2] RAGAS 평가 루프 — Faithfulness·Context P/R + hit@k 회귀리포트
- **T22** [L2] 전체 794편 배치 인입 — Colab GPU, 증분 전체 적재
- **T23** [L3] **E2E 검증(PoC)** — 질의→인용 답변 정확성 + RAGAS 통과 ⟵E2E

### Phase 4 — AWS 운영화
- **T24** [L2] Terraform 인프라 — S3·IAM·네트워킹·환경분리
- **T25** [L2] 인입 인프라 — Lambda(S3 event)→AWS Batch(GPU)
- **T26** [L2] 서빙 인프라 — ECS Fargate + API GW/ALB + Dockerfile
- **T27** [L2] 인증(Cognito) — 사용자 풀 + API 보호
- **T28** [L2] 프로덕션 LLM/임베딩 — Bedrock Claude/Rerank + Jina 엔드포인트
- **T29** [L1] 관측(CloudWatch) — 로그·메트릭·알람
- **T30** [L3] **E2E 검증(AWS)** — 배포 환경 인입→질의→인용 전구간 ⟵E2E

### Phase 5 — Optional
- **T31** [L2] Frontend 챗봇 UI — 질의 UI(3110)+Cognito

## 의존성 그래프 (핵심)
```
T1 ─┬─ T2 ─ T3 ─┐
    └─ T4 ───────┼─ T7 ─ T8 ─ T9 ─ T10 ─ T11 ─┬─ T12
T5 ─────────────┘                              │
T6 ───────────────────────────(평가 기준선)────┤
T11 ─ T13 ─ T14 ─ T15 ─ T16 ─ T17 ─ T18 ─ T19 ─ T20 ─┬─ T21
                                                      └─ T22 ─ T23(PoC E2E)
T23 ─ T24 ─┬─ T25 ─┐
           ├─ T26 ─┼─ T27 ─ T28 ─ T29 ─ T30(AWS E2E)
           └───────┘
T20 ─ T31(선택)
```
- T7~T11는 T4(스토어)·T3(메타) 확정 후. T13~T19는 T11(인덱스) 후. T24+는 T23(PoC 통과) 후.

## 규모 개략
8 epic · 31 스토리 · E2E 2개(PoC/AWS) · L1×5 / L2×20 / L3×6.
