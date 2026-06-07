# cpet.rag — Rolling Wave Execution Order

> 31개 스켈레톤 카드. JIT refine: 카드 N 구현·검증 후 N+1 refine.
> 보드: https://cyanlunakanban.vercel.app/?project=cpet.rag

## Phase 0 — Foundation & Baseline
`#3108 → #3110 → #3111 → #3112 → #3113 → #3114`
- #3108 core 골격+설정 · #3110 스키마 · #3111 메타로더 · #3112 벡터스토어+LanceDB · #3113 Colab 노트북 · #3114 RAGAS 평가셋

## Phase 1 — Ingestion (Colab PoC)
`#3115 → #3116 → #3117 → #3118 → #3119 → #3120`
- #3115 Docling 파싱 · #3116 VLM 폴백 · #3117 Late Chunking 임베딩 · #3118 적재 · #3119 IngestionPipeline 통합 · #3120 Obsidian Vault

## Phase 2 — Retrieval & Generation
`#3121 → #3122 → #3123 → #3124 → #3125 → #3126`
- #3121 Query Translation · #3122 하이브리드 검색 · #3123 Reranker · #3124 생성+Strict Citation · #3125 인용검증 · #3126 역번역

## Phase 3 — Serving & Eval
`#3127 → #3128 → #3129 → #3130 → #3131(E2E)`
- #3127 질의 오케스트레이션 · #3128 FastAPI · #3129 RAGAS 루프 · #3130 794편 배치 · **#3131 E2E(PoC)**

## Phase 4 — AWS 운영화
`#3132 → #3133 → #3134 → #3135 → #3136 → #3137 → #3138(E2E)`
- #3132 Terraform · #3133 인입 인프라 · #3134 서빙 인프라 · #3135 Cognito · #3136 프로덕션 LLM/임베딩 · #3137 CloudWatch · **#3138 E2E(AWS)**

## Phase 5 — Optional
`#3139` — Frontend 챗봇 UI

## 의존성 노트
- #3115~#3119는 #3112(스토어)·#3111(메타) 확정 후
- #3121~#3127는 #3119(인덱스) 후 / #3124는 #3123(top-k) 후 / #3125는 #3124 후
- #3132~ (AWS)는 **#3131(PoC E2E) 통과 후** 착수
- #3139(UI)는 #3128(API) 후

## Rolling Wave 진행 규칙
각 카드: **Refine(N)**(이전 카드 실제 구현 확인 후 AC·엣지케이스·파일경로 구체화) → **Impl(N)**(`/kanban-run #N`) → **Verify(N)**(diff·테스트·다음 카드 영향 메모) → N+1.
