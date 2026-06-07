# cpet.rag

운동생리·스포츠의학·CPET 분야 **학술 문헌 RAG 서비스**.
학술 PDF 코퍼스를 임베딩해, 한국어 질의로 영문 논문 근거를 검색하고 **출처 인용과 함께** 답변하는 멀티유저 RAG 시스템 (AWS 서버형).

> `cpet.db`(CPET 데이터 분석 플랫폼)의 **지식/문헌 레이어** — 두 프로젝트가 하나의 도메인 제품군을 이룹니다.

## 무엇

- **입력**: 학술 논문 PDF(표·수식 포함), 지속 추가 → 이벤트 기반 증분 인입
- **질의**: 한국어 질문 → 영문 논문 검색 → 출처(논문·페이지) 인용 답변
- **자산**: DOI·전체저자·연도·저널·OpenAlex ID 메타데이터 (`data/corpus_index.csv`)

## 아키텍처 (요약)

```
인입:  PDF→S3 →(event)→ Lambda → Batch(GPU): Docling(+VLM) → Late Chunking 임베딩(Jina-v3) → LanceDB@S3
서빙:  사용자 → API GW → Fargate(FastAPI+LlamaIndex): 번역(KO→EN) → 하이브리드 검색 → Rerank → Bedrock Claude(Strict Citation) → 역번역
평가:  RAGAS (Faithfulness·Context P/R)   인증: Cognito   관측: CloudWatch
```

전체 설계는 **[docs/RAG_IMPLEMENTATION_PLAN.md](docs/RAG_IMPLEMENTATION_PLAN.md)** (v2) 참고.

## 폴더 구조

| 디렉토리 | 역할 |
|----------|------|
| `ingestion/` | 인입 파이프라인 — `parse`(Docling+VLM) · `chunk`(Late Chunking) · `embed`(Jina-v3) · `load`(LanceDB) |
| `serving/` | API — `retrieval`(번역·하이브리드·rerank) · `generation`(Bedrock Claude·인용) · `app`/`router`(FastAPI) |
| `core/` | 공통 — `config` · `models`(schemas) · `metadata`(corpus_index·OpenAlex) · `citation`(overlap 검증) |
| `eval/` | RAGAS 평가 · QA 셋 |
| `infra/` | Terraform/CDK (S3·ECS·Batch·Bedrock·Cognito) · Docker |
| `notebooks/` | Colab GPU PoC |
| `frontend/` | 챗봇 웹 UI (선택) |
| `data/` | `corpus_index.csv` 등 (PDF 원본은 S3) |
| `scripts/` | `add_paper.py` 등 운영 |
| `docs/` | 계획서 · 아키텍처 · ADR · 리뷰 |

## 상태

🟡 **Phase 0 (셋업)** — 계획 확정, 스캐폴딩 완료. 다음: LlamaIndex+LanceDB 골격 + RAGAS 평가셋.
로드맵은 계획서 §10 참고 (Colab PoC → AWS 운영화).

## 개발 환경

- Python 3.11+ · (인입 GPU: Colab/AWS Batch) · AWS(S3·ECS·Bedrock)
- `cp .env.example .env` 후 값 채우기 → `docker compose up`(로컬 dev)

## 코퍼스 출처

시드: Hargreaves & Spriet (2020) 인용 네트워크 + dausin2026 레퍼런스. 현재 794편(증가 중).
PDF 원본은 별도 스토리지(S3 이관 예정), 메타는 `data/corpus_index.csv`.
