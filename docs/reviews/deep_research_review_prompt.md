# Google Deep Research 리뷰 프롬프트 — 아래 전체를 복사해 붙여넣으세요

---

당신은 **프로덕션 RAG 시스템 및 학술 문헌 검색(scientific literature retrieval) 전문가**입니다.
아래에 제시하는 **RAG 챗봇 구현 계획서**를 **2026년 6월 현재 최신 베스트 프랙티스 기준으로 비판적으로 리뷰**하고, 근거(출처 링크)와 함께 개선안을 제시해 주세요. 한국어로 답하되 기술 용어는 영문 유지.

## 프로젝트 배경 (context)
- **코퍼스**: 운동생리·스포츠의학 분야 **학술 논문 PDF 794편(계속 증가)**. 영문 논문, 표·수식·2열 레이아웃 포함.
- **이미 보유한 메타데이터**: 각 논문의 DOI·제목·전체 저자·연도·저널·OpenAlex ID (CSV/JSON). 인용 그래프(OpenAlex referenced_works) 조회 가능.
- **사용 환경**: 로컬 PC는 **GPU 없음** → 무거운 작업은 **Google Colab GPU**. 질의·서빙은 로컬. 도구는 **Claude Code**(+ MCP) 사용. 산출물은 **Obsidian Vault(md) + 단일 SQLite 벡터DB(knowledge.db)**.
- **질의 특성**: **한국어 질문 → 영문 논문 검색**(다국어 필요). 출처(논문·페이지) 인용 필수.
- **운영 요구**: 새 논문이 계속 추가되므로 **증분(incremental) 인덱싱**이 필수. 1인이 아닌 **3인 공동** 사용.

## 리뷰해야 할 구현 계획서 (전문)

<<< 계획서 시작 >>>
[여기에 RAG_IMPLEMENTATION_PLAN.md 전체 내용을 붙여넣으세요]
<<< 계획서 끝 >>>

## 리뷰 요구사항 (각 항목에 대해 근거·출처 포함)

1. **기술 선택 검증 (2026년 기준 최신성)**
   - PDF 파싱: **Docling**(+Nougat) 선택이 학술 PDF에 최선인가? MinerU·Marker·올라마 기반 VLM 파서·LLM 파싱(GPT/Gemini vision) 등 더 나은/최신 대안은? 표·수식·각주 처리 품질 관점.
   - 임베딩: **Qwen3-Embedding**(본문) + **SPECTER2**(논문 그래프) 분리가 타당한가? 2026년 더 강한 다국어·과학도메인 임베딩(BGE-M3, Jina v3/v4, NV-Embed, Voyage, gte-Qwen 등)은? 한↔영 cross-lingual 성능 근거.
   - 벡터DB: **sqlite-vec** 단일 파일 선택의 한계(스케일·동시성·3인 공유)와 LanceDB/Chroma/pgvector/Qdrant 대비 트레이드오프. 수천~수만 청크로 커질 때.
   - 검색: **하이브리드(BM25+벡터)+리랭커** 구성이 충분한가? 리랭커 모델 최신 추천.

2. **청킹 전략 비판**
   - "구조기반 + 메타데이터 컨텍스트 프리펜드(저비용 Contextual Retrieval)" 가 **LLM 기반 Contextual Retrieval(Anthropic)** 대비 얼마나 효과적인가? 학술논문에서 검증된 근거는?
   - 참고한 **Adaptive Chunking**(다중전략+품질지표 자동선택)을 v2로 미룬 판단이 옳은가, 아니면 처음부터 도입해야 하나?
   - **late chunking**, **semantic chunking**, **proposition-based chunking**, **하위문서/부모-자식 청킹** 등 2025–2026 기법 중 이 코퍼스에 더 맞는 것은?

3. **빠진 요소 / 위험**
   - 평가(evaluation) 방법(hit@k + LLM 채점)이 충분한가? RAGAS·도메인 평가셋 구축 베스트 프랙티스.
   - 인용 정확도(citation faithfulness)·환각 방지·근거표시 전략이 충분한가?
   - 표/수식/그림이 많은 학술 PDF에서 흔한 실패모드와 대응.
   - 증분 인덱싱·임베딩 버전관리·3인 동시사용 시 DB 동기화(구글드라이브 위 SQLite) 위험.
   - 한국어 질의의 cross-lingual 검색 품질 저하 위험과 대안(질의 번역 vs 다국어 임베딩).

4. **복잡도/비용 트레이드오프**
   - 이 계획이 과하게 복잡하지 않은가? 동일 목표를 더 단순하게 달성할 대안(예: 기성 프레임워크 LlamaIndex/Haystack/RAGFlow, 또는 기존 보유 `qmd` 활용)과 비교.
   - Colab(무료/Pro) 제약(세션 끊김·런타임)에서 794편+증분 배치의 현실성.

5. **종합 권고**
   - **우선순위가 매겨진 개선안 목록**(High/Med/Low)으로.
   - 계획에서 **그대로 둬도 좋은 부분 / 반드시 바꿔야 할 부분**을 명확히 구분.
   - 가능하면 **최신 벤치마크·논문·블로그(2025–2026) 출처 링크** 첨부.

## 출력 형식
1. 한 문단 총평(이 계획의 강점·핵심 약점)
2. 항목별(1~4) 상세 리뷰 — 각 주장에 출처
3. 우선순위 개선안 표 (변경점 · 이유 · 출처)
4. "처음 2주에 할 일" 수정 제안

---

### 📌 사용법
1. 위 `<<< 계획서 시작 >>> … <<< 계획서 끝 >>>` 사이에 **`RAG_IMPLEMENTATION_PLAN.md` 전체를 붙여넣기**
2. [Gemini](https://gemini.google.com) → **Deep Research** 모드 선택 → 전체 붙여넣고 실행
3. (대안) ChatGPT/Claude의 Deep Research·웹검색 모드에 동일하게 사용 가능
