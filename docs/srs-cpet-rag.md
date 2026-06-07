# cpet.rag — Software Requirements Specification

> 기반: `docs/RAG_IMPLEMENTATION_PLAN.md` (v2) · `docs/decisions/0001-tech-stack.md`
> 작성 2026-06-07

## 1. Background & Motivation
운동생리·스포츠의학·CPET 분야 학술 논문 794편(증가 중)에 대해, 한국어 질의로 영문 근거를 찾아 **출처 인용과 함께** 답하는 RAG 서비스. `cpet.db`(데이터 분석 플랫폼)의 문헌지식 레이어. 이미 보유한 풍부한 메타데이터(DOI·저자·연도·저널·OpenAlex ID)를 검색·인용·그래프에 재사용한다.

## 2. Goals & Non-Goals
**Goals**: ① 학술 PDF 증분 인입·임베딩 ② 한↔영 정확 검색 ③ 인용 정확(환각 인용 차단) ④ 멀티유저(3인→확장) AWS 서버형 ⑤ Obsidian 지식그래프 연동.
**Non-Goals**: 논문 자동 수집(별도 `add_paper.py`), 일반 웹 검색, 비학술 문서, 실시간 모델 학습.

## 3. Scope
- In: PDF 파싱→청킹→임베딩→벡터스토어, 번역·하이브리드·리랭크·생성·인용검증 질의 API, RAGAS 평가, AWS 배포.
- Out: PDF 수집/저작권 확보, 결제, 외부 공개 SaaS화(추후).

## 4. Functional Requirements
- FR-1 인입: S3의 신규 PDF를 이벤트 기반 증분 처리(파싱→청킹→임베딩→적재), 중복 dedup.
- FR-2 파싱: Docling 구조보존 Markdown, 복잡 표/수식은 Gemini VLM 폴백.
- FR-3 청킹: Late Chunking(구조 경계 + 메타 프리펜드).
- FR-4 임베딩: Jina-embeddings-v3 단일(late chunking 네이티브).
- FR-5 저장: LanceDB(papers·chunks·vector·FTS·메타).
- FR-6 검색: Query Translation(KO→EN, 용어보호) → 하이브리드(BM25+벡터) → Rerank → top-k.
- FR-7 생성: Bedrock Claude, Strict Citation(미인용 시 거부) + EN→KO 역번역.
- FR-8 인용검증: 인용 태그 ↔ 검색 청크 원문 overlap 검증, 미검증 차단.
- FR-9 평가: RAGAS(Faithfulness·Context Precision/Recall) + Retrieval hit@k.
- FR-10 메타필터: 연도·저자·저널 정밀 필터(corpus_index 조인).
- FR-11 지식그래프: parsed md/ + OpenAlex 인용 wikilink → Obsidian.
- FR-12 멀티유저: Cognito 인증, API 보호.

## 5. Non-Functional Requirements
- 성능: 질의 p95 < 8s(리랭크 포함). 인입은 배치(지연 허용).
- 보안: PDF 저작권 — 비공개 S3, 자격증명 비커밋. 최소권한 IAM.
- 비용: GPU는 인입 시에만 과금, 상시 관리형 벡터DB 구독 회피.
- 확장: LanceDB→OpenSearch 승격 경로. 코퍼스 수천~수만 청크.
- 이식성: LlamaIndex 추상화로 스토어/LLM 교체 가능(Colab↔AWS).

## 6. Architecture Overview
인입(S3→Lambda→Batch GPU: Docling+VLM→Late Chunking 임베딩→LanceDB@S3) + 서빙(Fargate FastAPI+LlamaIndex: 번역→하이브리드→리랭크→Bedrock Claude→인용검증→역번역). 상세 `docs/architecture.md`.

## 7. Technology Stack
Python 3.11 · LlamaIndex · Docling(+Gemini VLM) · Jina-embeddings-v3 · LanceDB · Amazon Bedrock(Claude·Cohere Rerank) · FastAPI · RAGAS · AWS(S3·Lambda·Batch·ECS Fargate·Cognito·CloudWatch) · Terraform.

## 8. Constraints & Assumptions
- 개발 GPU는 Colab, 운영 GPU는 AWS Batch. PDF 원본은 S3(현재 Drive 보관).
- 메타 단일 진실원천 `data/corpus_index.csv`(DOI 키). 한↔영 cross-lingual.

## 9. Risk Assessment
- 인용 환각(최대 57%) → Strict Citation + overlap 검증. 복잡 표/수식 파싱 실패 → VLM 폴백. 번역 오역 → 용어 보호 사전. Colab→AWS 격차 → LlamaIndex 추상화. Bedrock 쿼터 → Anthropic API 폴백.

## 10. Success Criteria
- 샘플 10편 PoC에서 RAGAS Faithfulness·Context Precision 목표치 통과, 인용 overlap 100% 검증.
- 794편 전체 인입 완료 + 한국어 질의에 정확한 영문 근거+인용 답변.
- AWS 배포 환경에서 신규 PDF 업로드→자동 인입→질의 E2E 동작.
