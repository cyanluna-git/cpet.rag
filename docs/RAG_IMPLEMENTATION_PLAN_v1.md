# Sinico 논문 코퍼스 — RAG 챗봇 구현 계획서

> 작성 2026-06-06 · 대상: `sinico.papers/pdf/` 794편(계속 증가) · GPU: Google Colab
> 목표: 모아진 + 앞으로 추가할 학술 PDF를 지속적으로 임베딩해 **인용 가능한 RAG 챗봇** 구축

---

## 0. 목표 & 제약

**목표**
- 794편(증가 중) 학술 PDF에 대해 자연어 질문 → **출처(논문·페이지) 인용**과 함께 답변
- 두 갈래 산출물:
  - `md/` → **Obsidian Vault** (그래프 뷰·메모·논문 간 링크 = 사람이 탐색)
  - `db/knowledge.db` → **벡터 검색** (Claude Code / 챗봇이 질의)
- 새 논문(`add_paper.py`로 추가)을 **증분(incremental)** 으로 계속 인덱싱

**제약 / 강점**
- 로컬은 GPU 없음 → **무거운 작업(파싱·임베딩)은 Colab GPU**, 질의·서빙은 로컬
- ⭐ **이미 풍부한 메타데이터 보유**(`corpus_index.csv` + `_archive/manifest.json`: DOI·제목·전체저자·연도·저널·OpenAlex ID) → 하이브리드 검색·인용·그래프에 그대로 활용 (큰 이점)
- 한글 질의 + 영문 논문 → **다국어 임베딩** 필요

---

## 1. 아키텍처 개요

```
 pdf/*.pdf  +  corpus_index.csv(메타)
        │
        ▼  ┌─────────── Colab GPU (배치, 증분) ───────────┐
   ingest.py
        │   1) 파싱   Docling → 구조보존 Markdown + JSON
        │   2) 청킹   구조기반 + 메타컨텍스트 프리펜드
        │   3) 임베딩 Qwen3-Embedding (GPU)
        ▼  └──────────────────────────────────────────────┘
   ┌────────────────────┬─────────────────────────────┐
   ▼                    ▼                             ▼
 md/ (Obsidian)    db/knowledge.db              (선택) papers.graph
 - frontmatter      - chunks + vector(sqlite-vec) - 인용 네트워크
 - 본문/요약        - BM25(FTS5) 하이브리드        (OpenAlex refs)
 - [[wikilink]]     - 메타 조인(DOI/저자/연도)
        │                    │
        └──── 그래프뷰 ──────┘
                             ▼
                  질의: 하이브리드(BM25+벡터) → 리랭크 → Claude 답변(+인용)
                             ▲
                     Claude Code (MCP) / 챗봇 UI  ← 로컬, GPU 불필요
```

핵심: **무거운 ingest는 Colab, 가벼운 query는 로컬.** `knowledge.db`(단일 SQLite 파일)를 Drive에 두면 양쪽에서 공유.

---

## 2. 기술 스택 결정 (근거 포함)

### 2.1 PDF 파싱 → **Docling** (주) + 필요시 Nougat(수식)
학술 PDF는 2열·표·수식·각주 때문에 일반 파서가 약함. 비교연구에서 일반 파서(PyMuPDF 등)는 "Scientific" 범주에서 모두 고전, 학습기반(Nougat) 우세 ([arXiv 2410.09871](https://arxiv.org/abs/2410.09871)).
- **Docling**(IBM): 레이아웃·읽기순서·표 셀·수식 위치·이미지를 통합 표현 → **Markdown 변환** 품질 좋고 AI 파이프라인용으로 설계. 참고 글의 기본 파서이기도 함 ([Firecrawl 2026](https://www.firecrawl.dev/blog/best-pdf-parsers)).
- 수식 비중 큰 논문만 **Nougat**(+Mathpix) 보조 ([Nougat 논문](https://arxiv.org/pdf/2308.13418)).
- **GROBID**(섹션·참고문헌 TEI)는 우리에겐 메타데이터가 이미 있어 선택사항.
- GPU 권장(레이아웃 모델). → **Colab에서 배치 실행.**

### 2.2 청킹 → 구조기반 + **메타데이터 컨텍스트**(저비용 Contextual Retrieval), 추후 Adaptive
참고한 [Adaptive Chunking](https://discuss.pytorch.kr/t/adaptive-chunking-rag/10478)의 핵심: **문서별 최적 청킹을 자동 선택**(재귀/페이지/LLM정규식/의미 4종 → RC·BI·ICC 등 5지표로 평가 → 최고점 선택). 청크 품질 작은 개선이 파이프라인 거치며 8~10pp로 증폭.
- **v1(실용)**: Docling Markdown의 **구조 경계(섹션/단락/표)** 를 존중하는 청킹, 목표 500~1000토큰 + 약간 overlap.
- ⭐ **Contextual Retrieval**(Anthropic): 청크 앞에 짧은 맥락을 붙여 임베딩하면 검색 실패율 **−35%, BM25 결합 −49%, 리랭크까지 −67%** ([Anthropic](https://www.anthropic.com/news/contextual-retrieval)). 우리는 **LLM 호출 없이** 이미 가진 메타로 저비용 구현:
  `[{제목} · {제1저자} {연도} · {저널} · §{섹션헤딩}]\n{청크본문}` 을 임베딩 텍스트로.
  (예산 되면 Claude Haiku로 청크별 1줄 컨텍스트 생성해 품질↑)
- **v2(고도화)**: 참고 논문의 다중전략+지표 평가를 도입(섹션 많은 리뷰논문 vs 짧은 레터 등 문서별 최적화).

### 2.3 임베딩 모델 → **Qwen3-Embedding**(주, Colab GPU) + **SPECTER2**(논문단위 보조)
- **Qwen3-Embedding-0.6B/4B**: MTEB 최상위·다국어(한↔영) 강함, 오픈소스, GPU 보유 시 최적 ([Milvus 2026](https://milvus.io/blog/choose-embedding-model-rag-2026.md)). 4B 권장(품질/속도 균형), 차원 1024.
- **SPECTER2**(Allen AI, 인용기반 논문 임베딩): title+abstract로 **논문 간 유사도** → Obsidian 관련논문 링크/그래프에 사용. 단 chunk-level RAG엔 한계(연구의제 수준에서 약함) → **본문 청크 검색엔 Qwen3, 논문 추천/그래프엔 SPECTER2** 로 역할 분리 ([SPECTER2](https://huggingface.co/allenai/specter2)).
- ⚠️ MTEB 1위가 우리 데이터 1위는 아님 → **소규모 평가셋으로 검증 후 확정**(§8).

### 2.4 벡터 DB → **sqlite-vec**(`knowledge.db`) — 기존 `qmd`와 일관
요청 구조의 `db/knowledge.db`(단일 SQLite)에 가장 부합. 임베디드·SQL 필터·Claude Code에서 직접 질의 가능 ([sqlite-vec/LanceDB 비교](https://shaharia.com/blog/choosing-embeddable-vector-database-go-application/)). "벡터DB 선택보다 청킹·검색 파이프라인이 훨씬 중요"([Encore](https://encore.dev/articles/best-vector-databases)).
- **옵션 A (권장)**: `sqlite-vec` 직접 구축. `chunks`(텍스트·메타·embedding) + FTS5(BM25). 단일 `.db` 파일을 Drive 공유.
  - ⚠️ 리스크: 로컬 `qmd`에서 `vec0` 모듈 로드 실패 경험 → sqlite-vec 확장 빌드/로딩 주의. **메모리: qmd embed/vec는 Node로 실행해야 동작**([[project_qmd_embed_node]]).
- **옵션 B**: 기존 **`qmd` 활용** — `md/`를 qmd 컬렉션에 넣으면 lex/vec/hyde 검색 + **Claude Code MCP 통합이 이미 됨**. 가장 빠른 길이나 청킹/하이브리드 커스터마이즈 제약.
- **옵션 C**: **LanceDB**(디스크 컬럼나, 메모리 초과 대비) — 수만 청크 이상으로 커지면 전환.
- 결론: **A로 구축하되, 빠른 PoC는 B(qmd)로 동시 진행**해 비교.

### 2.5 검색 → 하이브리드 + 리랭커 (가장 큰 효과)
- **BM25(FTS5) + 벡터** 동시 검색 후 병합 → **크로스인코더 리랭크**.
- 리랭커: `bge-reranker-v2-m3` 또는 `Qwen3-Reranker`(다국어). 10배 over-retrieve(예: 50) → 리랭크 → top-k(예: 8).
- 효과: 컨텍스추얼+BM25+리랭크 조합이 검색실패 **−67%** ([Anthropic](https://www.anthropic.com/news/contextual-retrieval)).
- **메타 필터**: "2023년 이후", "특정 저자/저널" 등은 `corpus_index.csv` 조인으로 정밀 필터.

### 2.6 Obsidian Vault (`md/`)
- 논문 1편 = md 노트 1개. **frontmatter**(doi, authors, year, journal, source, tags) + 본문(또는 구조요약) + **인용 wikilink**.
- 링크 소스: OpenAlex `referenced_works`(우리 manifest의 openalex_id로 조회) → 코퍼스 내 인용관계를 `[[논문]]`으로 → **그래프 뷰**에서 연구 맥락 시각화.
- 사람은 Obsidian에서 탐색·메모, 챗봇은 db에서 질의 — **같은 청크 ID로 연결**.

---

## 3. 파이프라인 상세 (`ingest.py`)

```
입력: pdf/<file>.pdf, corpus_index.csv(메타)
상태: processed 테이블(file_hash, doi) → 이미 처리분 skip (증분)

1. 메타 매칭   파일명→DOI→manifest(전체저자·제목·연도·저널·openalex_id)
2. 파싱        Docling(pdf) → markdown + 구조 JSON(섹션/표/그림)
3. md 생성     md/<doi-slug>.md (frontmatter + 본문 + [[인용링크]])  → Obsidian
4. 청킹        구조경계 기반 + 메타컨텍스트 프리펜드 → chunks[]
5. 임베딩      Qwen3-Embedding(GPU) → vectors[]   (배치)
6. 적재        knowledge.db:
                 papers(doi, title, authors, year, journal, openalex_id, md_path)
                 chunks(id, doi, section, text, ctx_text, page)
                 vec_chunks(chunk_id, embedding)         ← sqlite-vec
                 chunks_fts(text)                         ← FTS5/BM25
7. 그래프(선택) OpenAlex refs → links 테이블 → Obsidian wikilink 갱신
```

**질의 경로(런타임, 로컬)**
```
질문 → (메타필터 파싱) → BM25 top50 ∪ 벡터 top50 → 병합·중복제거
      → 리랭커 top8 → 프롬프트(청크+출처) → Claude 답변 + [DOI/제목/페이지] 인용
```

---

## 4. Colab GPU 분담 전략

| 작업 | 위치 | 이유 |
|------|------|------|
| Docling 파싱, Qwen3 임베딩, 리랭커 | **Colab GPU** | 모델 추론 무거움 |
| `knowledge.db`·`md/` 생성/저장 | Colab → **Drive 기록** | 결과물 공유 |
| 질의·리랭크(소량)·답변 | **로컬/Claude Code** | DB만 있으면 GPU 불필요 |

- Colab 노트북이 **Drive 마운트**(`sinico.papers/`) → `pdf/` 읽고 `md/`·`db/knowledge.db` 씀.
- 임베딩만 GPU 필요하면 로컬에서 작은 모델로도 가능하나, 794편+증가분은 Colab 배치가 효율적.
- 노트북은 `_tools/colab_ingest.ipynb`로 저장 → 새 논문 쌓이면 재실행(증분).

---

## 5. 증분 처리 (계속 추가되는 구조)

- `add_paper.py`가 `pdf/` + `corpus_index.csv`에 추가 → ingest는 **`processed` 테이블에 없는 파일만** 처리.
- 키: 파일 해시 + DOI. 재실행 시 신규 N편만 파싱·임베딩.
- 주기: 수동(Colab 재실행) 또는 "N편 쌓이면" 트리거. 비용 낮음(신규분만).
- 임베딩 모델 교체 시에만 전체 재임베딩(버전 컬럼 관리).

---

## 6. 디렉토리 구조 (제안)

```
sinico.papers/
├── pdf/                         ← 원본 (그대로)
├── index/corpus_index.csv       ← 메타 (그대로)
├── rag/
│   ├── md/                      ← Obsidian Vault (논문별 노트)
│   ├── db/knowledge.db          ← sqlite-vec + FTS5 (공유)
│   ├── eval/qa_set.jsonl        ← 평가셋
│   └── config.yaml              ← 모델·청크파라미터
└── _tools/
    ├── ingest.py                ← 파싱·청킹·임베딩·적재
    ├── query.py / rag_mcp.py    ← 질의(로컬, Claude Code 연동)
    └── colab_ingest.ipynb       ← GPU 배치 노트북
```

---

## 7. 단계별 실행 계획

| Phase | 산출물 | 핵심 |
|-------|--------|------|
| **0. 셋업** | Colab 노트북, deps, Drive 마운트, **평가셋 20문항** | 작게 5~10편으로 E2E 먼저 |
| **1. 파싱** | `md/` 초안 + 구조 JSON | Docling 품질 확인(표·수식) |
| **2. Vault** | Obsidian frontmatter + 인용 wikilink | 그래프 뷰 동작 |
| **3. 임베드/DB** | `knowledge.db`(청크+벡터+BM25) | sqlite-vec 적재·하이브리드 질의 |
| **4. 검색+서빙** | `query.py` 하이브리드+리랭크, Claude Code 연동 | 인용 포함 답변 |
| **5. 증분+고도화** | processed 증분, Adaptive 청킹, eval 루프 | 품질 측정·튜닝 |

**먼저 할 일**: Phase 0–1을 **샘플 10편**으로 끝까지(파싱→임베딩→질의 1회) 돌려 파이프라인을 검증한 뒤 794편 전체 배치.

---

## 8. 평가 (필수 — "MTEB 1위 ≠ 우리 데이터 1위")
- 손수 만든 **QA 평가셋 20~50문항**(질문 + 정답이 있는 논문 DOI).
- 지표: **Retrieval hit@k**(정답 논문이 top-k에 포함), 답변 정확도(사람/LLM 채점).
- 비교 축: 임베딩 모델(Qwen3 vs BGE-M3 vs SPECTER2), 청크 크기, 컨텍스트 프리펜드 유무, 리랭크 유무.
- 참고 논문처럼 청크지표(RC·BI)까지 보면 좋지만 v1은 hit@k 우선.

---

## 9. 리스크 & 대안

| 리스크 | 대응 |
|--------|------|
| Docling이 일부 PDF(스캔·복잡표) 실패 | Nougat/PyMuPDF 폴백, 실패목록 로깅 |
| sqlite-vec `vec0` 로딩 실패(기존 qmd 경험) | Node 경로([[project_qmd_embed_node]]) 또는 **LanceDB로 전환** |
| 한글질의-영문문서 미스매치 | 다국어 임베딩(Qwen3/BGE-M3) + 질의 번역 옵션 |
| 임베딩 품질 미달 | 도메인 파인튜닝(+10~30%) 또는 모델 교체 |
| Colab 세션 끊김(긴 배치) | 증분·체크포인트, 배치 분할 |
| 794편→수천편 확장 | LanceDB/디스크 인덱스로 전환 |

---

## 10. 즉시 시작 체크리스트
- [ ] `rag/` 디렉토리 + `config.yaml`(모델·청크파라미터) 생성
- [ ] Colab 노트북: Drive 마운트 + Docling/Qwen3 설치, **샘플 10편** 파싱
- [ ] `ingest.py` 골격(메타매칭·파싱·청킹·임베딩·적재)
- [ ] `knowledge.db` 스키마(papers/chunks/vec_chunks/chunks_fts/processed)
- [ ] 평가셋 20문항 작성(우리 논문 기반 Q&A)
- [ ] 하이브리드+리랭크 `query.py` → Claude Code에서 1회 질의 데모
- [ ] (병행) `md/` 일부를 `qmd` 컬렉션에 넣어 PoC 비교

---

### 참고 자료
- Adaptive Chunking RAG: https://discuss.pytorch.kr/t/adaptive-chunking-rag/10478
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- PDF 파서 비교(학술): https://arxiv.org/abs/2410.09871 · https://www.firecrawl.dev/blog/best-pdf-parsers
- 임베딩 모델 2026: https://milvus.io/blog/choose-embedding-model-rag-2026.md
- 벡터DB(임베디드) 비교: https://shaharia.com/blog/choosing-embeddable-vector-database-go-application/
- SPECTER2(학술 임베딩): https://huggingface.co/allenai/specter2
