# Colab 실인입 실행 가이드 (Phase 1)

`notebooks/colab_ingest.ipynb` 로 실제 PDF를 파싱→임베딩→적재까지 돌립니다.

## 0. 준비 — Colab Secrets 등록 (🔑 좌측 키 아이콘)
| 이름 | 필수 | 용도 |
|------|------|------|
| `JINA_API_KEY` | ✅ | Jina-v3 Late Chunking 임베딩 (https://jina.ai 무료키 발급) |
| `GH_TOKEN` | ⬜ 선택 | repo가 private일 때만. **public이면 불필요** |
| `GEMINI_API_KEY` | 선택 | 복잡 표/수식 VLM 폴백 (https://aistudio.google.com) |

> repo는 현재 **public** (Colab GitHub 브라우저에 노트북이 보이려면 public 필요). 따라서 `GH_TOKEN` 없이 클론됩니다. private로 되돌리면 `GH_TOKEN`(Fine-grained PAT, repo read) 등록.

## 1. 노트북 열기
- [colab.research.google.com](https://colab.research.google.com) → GitHub 탭 → `cyanluna-git/cpet.rag` → `notebooks/colab_ingest.ipynb`

- Runtime → Change runtime type → **GPU (T4)** 권장

## 2. 셀 순서대로 실행
1~6: GPU확인 → 클론 → 설치(수분) → Drive마운트 → 키설정 → 메타로드
7: **샘플 10편 실인입** (실 Docling + Jina 임베딩 + LanceDB)
8: 검증 — 벡터/FTS 검색 결과 출력
9: (선택) Obsidian Vault 생성

## 3. 기대 결과
- 7번: `결과: {'ingested': 10}` + `적재 청크: 수백`
- 8번: 쿼리 "skeletal muscle energy metabolism..."에 관련 청크가 점수순으로 나옴

## 4. 막히면
| 증상 | 해결 |
|------|------|
| `PDF_DIR 경로 확인 필요` | 5번 셀 PDF_DIR을 실제 Drive 경로로 수정 |
| Jina 401 | JINA_API_KEY 확인 |
| `SecretNotFoundError` | 선택 시크릿 미등록 — 노트북이 안전 처리하도록 수정됨(최신 pull) |
| Docling 느림 | 정상(CPU ~15s/편). 전체 794편은 GPU device='cuda' 보강 후 |
| clone 실패 | repo public 확인 (private면 GH_TOKEN 등록) |

## 5. 다음
- 샘플 OK → **전체 794편**: 7번 셀 `sample = have[:10]` → `have` 로 변경 (카드 #3122)
- LanceDB 영속화(Drive/S3), 그 후 **Phase 2**(검색·생성)

## 참고 — 실제 호출 구조 (Phase 1 구현)
- 파싱: `ingestion.parse.parse_pdf` (Docling)
- 청킹+임베딩: `ingestion.build_chunks.parsed_to_embedded_chunks` (Late Chunking)
- 임베더: `ingestion.embed.JinaEmbedder(backend='api')` → `POST api.jina.ai/v1/embeddings {late_chunking:true}`
- 적재: `ingestion.load.load_chunks` → `core.vectorstore.LanceDBStore`
- 오케스트레이션: `ingestion.ingest_corpus(papers, pdf_dir, ...)` (증분 skip)
