# cpet.rag — 프로젝트 규칙

운동생리·CPET 학술 문헌 RAG 서비스 (AWS 서버형). 상위 `~/dev/CLAUDE.md`(글로벌) 규칙을 따르되, 충돌 시 이 문서가 우선.

## 아키텍처 원칙
- **레이어 분리**: Router → Service → Repository. Router/Controller에 비즈니스 로직 금지.
- **파이프라인 분리**: `ingestion/`(인입, 배치·GPU) ↔ `serving/`(질의 API) 는 독립. 공통 로직은 `core/`.
- **벡터스토어 추상화**: LanceDB@S3 기본. 직접 의존 금지 — `core/` 인터페이스 경유 (OpenSearch/pgvector 교체 대비).
- **LLM/임베딩 클라이언트는 `core/`에 캡슐화** (Bedrock·Jina·번역). 호출부에 모델 ID 하드코딩 금지.

## 핵심 설계 결정 (docs/RAG_IMPLEMENTATION_PLAN.md v2)
- 파싱 **Docling + Gemini VLM 폴백** · 청킹 **Late Chunking**(adaptive 금지)
- 임베딩 **Jina-embeddings-v3 단일** · 검색 **Query Translation 샌드위치 + 하이브리드 + Rerank**
- 생성 **Bedrock Claude + Strict Citation**(overlap 후처리 검증 필수) · 평가 **RAGAS**
- 오케스트레이션 **LlamaIndex**

## 코드 스타일
- Python: type hints 필수, Black, snake_case 함수 / PascalCase 클래스 / UPPER_SNAKE 상수
- 환경변수는 `.env.example`에 반영, 하드코딩 금지 (AWS 키·버킷·모델 ID 등)
- 의존성 추가 시 lockfile 갱신

## 안전 가드
- **인용 정확도**: 생성 답변의 인용 태그는 반드시 검색 청크 원문과 overlap 검증 통과해야 노출 (환각 인용 차단)
- PDF는 저작권 자산 — 비공개 S3 버킷, 커밋 금지 (`data/`에 원본 PDF 두지 않음)
- `.env`·AWS 자격증명·corpus PDF 커밋 금지

## 포트 (로컬 dev)
- Frontend 3110 / Backend(API) 8110  (cpet.db 3100/8100 인접)

## 코퍼스 운영
- 메타 단일 진실원천: `data/corpus_index.csv` (DOI=중복키)
- 신규 논문: `scripts/add_paper.py <DOI> --by <이름>` → 메타·중복관리 → 인입 파이프라인이 증분 처리
