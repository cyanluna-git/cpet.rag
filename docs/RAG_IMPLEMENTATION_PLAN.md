# Sinico 학술 RAG 시스템 — 구현 계획서 v2 (AWS 서버형)

> v2 개정 2026-06-07 · 기반: v1 계획서 + Google Deep Research 리뷰 + **AWS 서버형 서비스로 목표 전환**
> 대상 코퍼스: 학술 논문 794편(지속 증가) · 운동생리/스포츠의학 · 한국어 질의 → 영문 논문
> 목표: **AWS에 배포되는 멀티유저 학술 RAG 서비스** (인용 정확·지속 인입·확장 가능)

---

## 0. v1 → v2 변경 요약 (Deep Research 리뷰 반영 + AWS 전환)

| # | 영역 | v1 | **v2 (변경)** | 근거 |
|---|------|----|--------------|------|
| 1 | 배포 형태 | 로컬+Drive 공유 | **AWS 서버형 서비스(S3/ECS/Bedrock)** | 사용자 지시 · Drive 위 SQLite 동시쓰기는 손상 위험 |
| 2 | 벡터스토어 | sqlite-vec(Drive) | **LanceDB on S3**(주) / OpenSearch·pgvector(대안) | SQLite 파일잠금·동기화 손상 → 오브젝트스토리지 서버리스 |
| 3 | 청킹 | 구조기반(early) + adaptive(v2 보류) | **Late Chunking 즉시 전면 도입**, adaptive 폐기 | 2026 표준, Context Cliff 해소, semantic보다 빠름 |
| 4 | 임베딩 | Qwen3 + SPECTER2 이원화 | **Jina-embeddings-v3 단일화**(late chunking 네이티브·task adapter·MRL) | 모델 이원화는 GPU·복잡도 낭비 |
| 5 | 다국어 | 다국어 임베딩 잠재공간 의존 | **Query Translation 샌드위치**(KO→EN 검색→EN 추론→KO 답변, 용어보호) | 의학 전문용어 cross-lingual 정렬 불완전 |
| 6 | 오케스트레이션 | 순수 Python 스크립트/qmd | **LlamaIndex** (Docling·LanceDB·Jina late chunking·RAGAS 네이티브) | 커스텀 기술부채↓ |
| 7 | 평가 | 수동 hit@k | **RAGAS**(Faithfulness·Context Precision/Recall) | LLM-as-judge 자동·민감한 회귀추적 |
| 8 | 인용 | "출처 달아라" 지시 | **Strict Citation 프롬프트 + overlap 후처리 검증** | 인용 환각 최대 57% 리스크 차단 |
| 9 | PDF 파싱 | Docling + Nougat 폴백 | **Docling + Gemini Flash VLM 폴백** | 복잡 표/수식 시각추론, 초저비용 |

**유지(리뷰가 인정한 강점)**: ① 풍부한 메타데이터 프리펜드(저비용 contextual) ② 무거운 인입은 GPU 배치 / 질의는 경량 분리 ③ Obsidian 지식그래프.

---

## 1. 목표 & 제약

- **서비스 형태**: AWS 배포 멀티유저(초기 3인→확장) RAG 챗봇 + API
- **입력**: 학술 PDF(표·수식·2열), 지속 추가 → **이벤트 기반 증분 인입**
- **질의**: 한국어 질문 → 영문 논문 근거 → **출처(논문·페이지) 인용** 답변
- **자산**: 이미 보유한 메타(DOI·전체저자·연도·저널·OpenAlex ID) → 하이브리드 필터·인용·그래프에 재사용
- **GPU**: 개발/PoC는 Colab, **운영 인입은 AWS Batch GPU(g5)**
- **LLM**: Amazon Bedrock(Claude) 우선(엔터프라이즈·동일 생태계)

---

## 2. AWS 레퍼런스 아키텍처

```
[ 인입 INGESTION — 이벤트 기반·증분 ]
 PDF → S3(raw/)  ──(S3 event)──▶ Lambda ──▶ AWS Batch (GPU g5)
                                              1) Docling 파싱 (+ Gemini Flash VLM 폴백)
                                              2) Late Chunking 임베딩 (Jina-v3, GPU)
                                              3) 메타 결합 (corpus_index/OpenAlex)
                                              └─▶ S3(parsed md/·artifacts) + 벡터스토어 적재
                                                   (processed 상태 → 신규분만)

[ 벡터스토어 ]  LanceDB on S3  (대안: OpenSearch Serverless / pgvector-Aurora)
               papers·chunks·embedding·FTS(BM25)·메타

[ 서빙 SERVING — API ]
 사용자(웹UI / Claude Code) ─▶ API Gateway/ALB ─▶ ECS Fargate (FastAPI + LlamaIndex)
       1) Query Translation  KO→EN (의학용어 사전 보호)
       2) 하이브리드 검색  BM25 ∪ 벡터(top50) → 병합
       3) Rerank  Cohere Rerank(Bedrock) / Qwen3-Reranker → top8
       4) 생성  Bedrock Claude (Strict Citation) → 인용 overlap 검증
       5) EN→KO 번역 + 보호용어 복원
 인증: Cognito · 관측: CloudWatch · 평가: RAGAS 배치(CI)

[ Obsidian Vault ]  parsed md/ 를 S3→로컬 읽기전용 동기화(그래프뷰), 사람 메모는 Git
```

| 계층 | 컴포넌트 | AWS 서비스 |
|------|----------|-----------|
| 원본/산출물 | PDF, parsed md, artifacts | **S3** |
| 인입 트리거 | 신규 PDF 감지 | S3 Event → **Lambda** |
| 인입 연산 | 파싱·임베딩(GPU) | **AWS Batch** (g5 GPU) / SageMaker |
| 벡터스토어 | 청크·벡터·BM25·메타 | **LanceDB@S3** / OpenSearch / Aurora-pgvector |
| API 서빙 | FastAPI+LlamaIndex | **ECS Fargate** + ALB/API GW |
| LLM/Rerank | 생성·재정렬·번역 | **Bedrock**(Claude, Cohere Rerank) |
| 인증 | 멀티유저 | **Cognito** |
| 관측/평가 | 로그·메트릭·RAGAS | **CloudWatch** + Batch |
| IaC | 인프라 정의 | **Terraform / AWS CDK** |

---

## 3. 기술 스택 결정 (리뷰 반영)

### 3.1 PDF 파싱 — Docling(주) + **Gemini Flash VLM 폴백**
- Docling: 레이아웃·읽기순서·표·수식 보존 MD, Apache-2.0, 빠름 → 1차 파서(AWS Batch GPU).
- 복잡 표/수식/파싱오류 페이지 → **해당 페이지 이미지를 Gemini Flash(VLM) API** 로 보내 구조화 추출 (Nougat 로컬 의존 대신, 초저비용·고정확).
- 산출: `s3://.../parsed/{doi}.md` + 구조 JSON(섹션/표/그림/페이지).

### 3.2 청킹 — **Late Chunking** (즉시·전면)
- 전통 early chunking의 Context Cliff 해소. 흐름:
  1. 긴 컨텍스트 임베딩 모델에 **문서/섹션 전체** 통과 → 토큰 임베딩(전역맥락 반영)
  2. Docling 구조 경계(섹션/단락)로 분할
  3. 경계별 토큰 임베딩 **mean-pooling** → 청크 벡터
- 임베딩 텍스트엔 **메타 프리펜드** 유지: `[{제목}·{제1저자} {연도}·{저널}·§{섹션}]` → 저비용 contextual 효과.
- **Adaptive Chunking(다중전략 평가)은 폐기** (오버헤드 대비 실효 낮음).

### 3.3 임베딩 — **Jina-embeddings-v3 단일화**
- 8k+ 컨텍스트·89개 언어·**late chunking 네이티브**·task LoRA 어댑터(retrieval/STS)·**MRL 차원축소**(1024→256).
- 단일 모델로 **본문 청크 검색 + 논문 유사도 그래프** 모두 처리 → SPECTER2 제거.
- 배포: SageMaker 엔드포인트 또는 Batch 내 로드(GPU). (Jina API도 가능)
- 대안: Qwen3-Embedding-4B(다국어 강). **평가셋으로 최종 확정**(MTEB 1위≠우리 데이터 1위).

### 3.4 벡터스토어 — **LanceDB on S3**(주) · 대안 표
| 옵션 | 장점 | 단점 | 적합 |
|------|------|------|------|
| **LanceDB@S3 (권장)** | 서버리스·오브젝트스토리지 동시 atomic write·버전관리·FTS 내장·최저비용 | 초대규모 동시질의 시 앱단 캐싱 필요 | 비용효율·증분 인입 |
| OpenSearch Serverless | 하이브리드(BM25+kNN) 네이티브·관리형·확장 | 최소비용 높음 | 대규모·고동시성 |
| pgvector (Aurora Serverless v2) | 메타+벡터+FTS 단일 SQL·ACID | 초대벡터 인덱스 한계 | SQL 친숙·정밀 메타조인 |
- 결론: **LanceDB@S3로 시작**, 트래픽·규모 커지면 OpenSearch로 승격. (3인 동시쓰기 손상 문제는 S3 append-only로 원천 해결)

### 3.5 검색 — **Query Translation 샌드위치 + 하이브리드 + Rerank**
1. **KO→EN 번역**(Bedrock Claude/전용 번역) + **의학용어 Entity 보호 사전**
2. EN 질의로 **BM25 + 벡터 각 top50** → 병합·중복제거 (영-영 매칭이 최고 recall)
3. **Rerank**(Cohere Rerank on Bedrock / Qwen3-Reranker-4B) → top8
4. 메타 필터(연도·저자·저널)는 corpus_index 조인
5. EN 추론 답변 → **EN→KO 번역 + 보호용어 복원**

### 3.6 생성 & 인용 — Bedrock Claude + **Strict Citation**
- 시스템 프롬프트 강제: "사용한 청크를 `[파일:start-end]` 또는 `[논문:페이지]`로 정확히 매핑 인용하지 않으면 **답변 거부**".
- **후처리 overlap 검증**: LLM이 단 인용 태그가 실제 검색 청크 원문에 존재하는지 확인 → 미검증 인용 차단(환각 인용 57% 리스크).

### 3.7 오케스트레이션 — **LlamaIndex**
- IngestionPipeline(증분·dedup), Docling Reader, **late_chunking**, LanceDB VectorStore, 메타필터, RAGAS 통합 — 커스텀 스크립트 대체.

---

## 4. 인입 파이프라인 (이벤트 기반·증분)
```
S3:raw/new.pdf ─event▶ Lambda ─submit▶ Batch(GPU)
  doc_hash/DOI가 processed에 없으면:
   Docling 파싱(+VLM 폴백) → md/ → Late Chunking 임베딩(Jina-v3)
   → LanceDB upsert(papers/chunks/vec/FTS) + 메타 결합 → processed 기록
```
- 신규 N편만 처리(비용↓). 임베딩 모델 버전 컬럼 → 교체 시에만 전체 재임베딩.
- `add_paper.py`(기존) → S3 업로드로 연결.

## 5. 서빙 파이프라인 (런타임)
FastAPI(LlamaIndex) on Fargate: `질의 → 번역 → 하이브리드 → rerank → Bedrock Claude(Strict Citation) → overlap검증 → 역번역`. 스트리밍 응답, 세션/이력은 별도(DynamoDB) — **벡터스토어와 분리**(읽기 일관성).

## 6. 평가 — RAGAS (CI 통합)
- 복잡 표/수식 포함 샘플 10편 + 전문가 QA 20~30문항 → **Baseline**.
- 지표: **Faithfulness · Context Precision · Context Recall** + Retrieval hit@k.
- 파이프라인 변경(임베딩/청크/리랭커) 시 자동 회귀측정 → 수치 근거로 튜닝.

## 7. AWS 인프라 / 비용 개략
- IaC: **Terraform/CDK**. 환경 분리(dev/prod), S3 버전닝, IAM 최소권한.
- 비용 성격: S3(소액) + Fargate(상시 소형) + Bedrock(토큰당) + Batch GPU(**인입 시에만** 과금) + LanceDB(스토리지=S3). 상시 GPU/관리형 벡터DB 구독 불필요 → **저비용 운영**.
- 대안 빠른길: **Bedrock Knowledge Bases**(완전관리 RAG)도 가능하나 **청킹(late chunking) 제어 약함** → 본 계획은 제어형(LlamaIndex) 우선.

## 8. 보안 / 멀티유저
- **Cognito** 사용자 풀(3인→확장), API Gateway 인증. S3/Bedrock IAM 역할 최소권한. PDF 저작권 고려해 **비공개 버킷**.

## 9. Obsidian 연동
- `parsed md/`(S3) → 각자 로컬 읽기전용 동기화 → 그래프뷰·메모. **사람 메모는 Git/Obsidian Sync**(벡터스토어와 분리, 손상위험 0).

## 10. 로드맵 (Dev → Prod)

| Phase | 기간 | 내용 |
|-------|------|------|
| **0 셋업·Baseline** | 1~3d | S3 버킷, LlamaIndex 골격, **RAGAS 평가셋 20~30문항**(복잡 논문 10편) |
| **1 파싱+Late Chunking** | 4~7d | Docling(+Gemini VLM 폴백) → Jina-v3 late chunking → LanceDB@S3 적재 (Colab GPU PoC) |
| **2 번역+검색+인용** | 8~11d | Translation 샌드위치 + 하이브리드 + Rerank + **Strict Citation** 질의 API |
| **3 E2E 평가·통합** | 12~14d | RAGAS 루프(Faithfulness/Precision) 튜닝 + Obsidian 검증 → 통과 시 **794편 전체 배치** |
| **4 AWS 운영화** | 이후 | Batch 이벤트 인입·Fargate API·Cognito·Terraform·CloudWatch, 증분 자동화 |

> 개발 가속: Phase 0~3은 **Colab GPU + 로컬**로 빠르게 검증 → Phase 4에서 **AWS로 이관**(동일 LlamaIndex 코드, 스토어만 S3).

## 11. 리스크 & 대응
| 리스크 | 대응 |
|--------|------|
| Late chunking 모델 컨텍스트 한계(초장문) | 섹션 단위 분할 후 적용 |
| Gemini VLM API 비용/한도 | 폴백 페이지에만 선택 적용, 캐시 |
| Bedrock Claude 리전/쿼터 | 리전 선택·쿼터 상향, Anthropic API 폴백 |
| 번역 단계 의학용어 오역 | Entity 보호 사전·역번역 검증 |
| LanceDB 동시질의 스케일 | 앱 캐시·OpenSearch 승격 경로 |
| Colab→AWS 이관 격차 | LlamaIndex 추상화로 스토어만 교체 |

---
### 참고 자료
- Google Deep Research 리뷰: `docs/google_deep_research.md`
- Late Chunking(Jina): https://jina.ai/news/late-chunking-in-long-context-embedding-models/
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- LanceDB on S3: https://lancedb.github.io/lancedb/ · RAGAS: https://docs.ragas.io
- Jina-embeddings-v3: https://huggingface.co/jinaai/jina-embeddings-v3
- LlamaIndex: https://docs.llamaindex.ai · Amazon Bedrock: https://aws.amazon.com/bedrock/
