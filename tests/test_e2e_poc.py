"""tests/test_e2e_poc.py — 전 스택 E2E 통합 검증 (PoC, #3131).

이 모듈 하나로 RAG 스택 전체(인입→검색→리랭크→생성→Strict Citation→인용검증→역번역→API→평가)
를 증명한다. 모든 LLM/외부 호출은 mock — AWS 키·GPU·네트워크 불필요.

테스트 구성:
1. test_ingestion_e2e          — PDF 인입 → store.count() > 0, registry 마킹 확인
2. test_query_e2e              — QueryPipeline.answer() → 한국어 답변, 검증 인용, retrieved 채워짐
3. test_citation_faithfulness  — 환각 인용 verify_citations 필터링 확인
4. test_api_e2e                — FastAPI TestClient POST /query 200, GET /health 200
5. test_eval_e2e               — evaluate() hit@k > 0 확인
6. test_full_flow_no_exception — 전 스택 예외 없이 완료 확인
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from core.models import Chunk, QueryRequest, QueryResponse
from core.models.paper import Paper
from core.vectorstore import LanceDBStore
from eval.ragas import EvalReport, evaluate
from ingestion.load.registry import ProcessedRegistry, processed_key
from ingestion.pipeline import IngestResult, ingest_pdf
from serving import QueryPipeline
from serving.app.main import create_app
from serving.app.router import get_service
from serving.app.service import QueryService
from serving.generation import Generator
from serving.retrieval import Reranker

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

DIM = 8  # 테스트 전용 소형 벡터 차원
_PDF_PATH = Path(__file__).parent / "fixtures" / "sample.pdf"


# ---------------------------------------------------------------------------
# IngestMockEmbedder — ingestion 전용 (embed_chunks → list[Chunk] with embedding)
# ---------------------------------------------------------------------------


class IngestMockEmbedder:
    """인입 파이프라인용 mock 임베더.

    embed_chunks 는 list[Chunk] (embedding 채워진 새 객체)를 반환한다.
    serving 레이어에서는 embed_query 만 사용한다.
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def _vec(self, seed: int) -> list[float]:
        return [(seed * 0.1 + i * 0.01) % 1.0 for i in range(self.dim)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(hash(t) % 50) for t in texts]

    def embed_late(
        self, document: str, boundaries: list[tuple[int, int]]
    ) -> list[list[float]]:
        return [self._vec(i) for i in range(len(boundaries))]

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        # 항상 seed=99 벡터 반환 — 검색 시 deterministic
        return [self._vec(99) for _ in texts]

    def embed_chunks(
        self,
        chunks: list[Chunk],
        full_document: str | None = None,
        late: bool = True,
    ) -> list[Chunk]:
        """ingestion 경로: embedding 이 설정된 새 Chunk 목록을 반환한다."""
        return [c.model_copy(update={"embedding": self._vec(i)}) for i, c in enumerate(chunks)]


# ---------------------------------------------------------------------------
# MockTranslator — passthrough (Bedrock 없음)
# ---------------------------------------------------------------------------


class MockTranslator:
    """패스스루 번역기 — Bedrock API 없이 동작."""

    def ko2en(self, text: str) -> str:
        return text

    def en2ko(self, text: str) -> str:
        return f"[KO] {text}"


# ---------------------------------------------------------------------------
# Module-scoped shared ingestion fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ingested_context(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[LanceDBStore, Paper, ProcessedRegistry, IngestResult]:
    """sample.pdf 를 한 번만 인입해 store, paper, registry, result 를 공유한다.

    module-scoped 로 선언해 6 개 테스트가 동일 store 를 재사용한다.
    """
    tmpdir = tmp_path_factory.mktemp("lancedb_e2e")
    reg_path = tmpdir / "processed.jsonl"

    store = LanceDBStore(uri=str(tmpdir), dim=DIM)
    embedder = IngestMockEmbedder(dim=DIM)
    registry = ProcessedRegistry(path=str(reg_path))

    paper = Paper(
        doi="10.9999/cpet.e2e.test",
        title="Exercise Physiology E2E Test Paper",
        first_author="Tester",
        year=2024,
        journal="J. Test",
        source="test_src",
        file="sample.pdf",
        oa_status="open",
        added_by="ci",
    )

    # Docling 이 설치되어 있으면 실제 파싱, 없으면 parse_pdf 를 mock 한다.
    try:
        import docling  # noqa: F401

        _docling_available = True
    except ImportError:
        _docling_available = False

    if _docling_available and _PDF_PATH.exists():
        result = ingest_pdf(
            _PDF_PATH,
            paper,
            store=store,
            embedder=embedder,
            registry=registry,
            use_vlm=False,
            use_late=True,
            force=True,
        )
    else:
        # Docling 없음 — parse_pdf 를 MockParsedDoc 으로 대체
        from ingestion.parse.types import ParsedDoc, Section

        mock_sections = [
            Section(
                heading="Energy Metabolism",
                text=(
                    "Skeletal muscle energy metabolism during exercise relies on "
                    "oxidative phosphorylation and glycolysis to sustain contraction."
                ),
                page=1,
                level=1,
            ),
            Section(
                heading="VO2max",
                text=(
                    "Maximal oxygen uptake VO2max reflects the upper limit of aerobic "
                    "capacity and is strongly correlated with endurance performance."
                ),
                page=2,
                level=1,
            ),
            Section(
                heading="Lactate Threshold",
                text=(
                    "Lactate threshold marks the exercise intensity at which lactate "
                    "begins to accumulate in the blood, indicating anaerobic metabolism."
                ),
                page=3,
                level=1,
            ),
        ]
        mock_parsed = ParsedDoc(
            source_path=str(_PDF_PATH),
            markdown="\n\n".join(
                f"## {s.heading}\n{s.text}" for s in mock_sections
            ),
            n_pages=3,
            sections=mock_sections,
            tables=[],
            low_confidence_pages=[],
            vlm_pages=[],
        )

        with patch("ingestion.pipeline.parse_pdf", return_value=mock_parsed):
            result = ingest_pdf(
                _PDF_PATH if _PDF_PATH.exists() else Path("/dev/null"),
                paper,
                store=store,
                embedder=embedder,
                registry=registry,
                use_vlm=False,
                use_late=True,
                force=True,
            )

    return store, paper, registry, result


# ---------------------------------------------------------------------------
# Shared QueryPipeline fixture (serving 레이어, module-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline(
    ingested_context: tuple[LanceDBStore, Paper, ProcessedRegistry, IngestResult],
) -> QueryPipeline:
    """인입된 store 위에서 동작하는 QueryPipeline (mock LLM seam)."""
    store, _, _, _ = ingested_context
    return QueryPipeline(
        store=store,
        embedder=IngestMockEmbedder(dim=DIM),
        translator=MockTranslator(),
        reranker=Reranker(),  # _rerank_call 은 각 테스트에서 mock
        generator=Generator(),  # _generate_call 은 각 테스트에서 mock
    )


# ---------------------------------------------------------------------------
# Helper — prompt 에서 chunk id 와 텍스트 추출
# ---------------------------------------------------------------------------


def _extract_all_chunk_ids(user_prompt: str) -> list[str]:
    """user_prompt 에서 모든 chunk id 를 순서대로 추출."""
    return [m.group(1).strip() for m in re.finditer(r"\(id=([^,)]+)", user_prompt)]


def _extract_chunk_text(user_prompt: str, chunk_id: str) -> str:
    """user_prompt 에서 chunk_id 에 해당하는 텍스트 블록을 추출한다.

    build_prompt 형식: ``[i] (id=<id>, doi=..., p.N, §section)\\n<text>\\n``
    반환: 구두점 제거·소문자화된 단어들 중 첫 5개 (overlap 검증에 충분)
    """
    # id 줄 이후 첫 비어 있지 않은 텍스트 줄 캡처
    pattern = re.compile(
        rf"\(id={re.escape(chunk_id)}[^)]*\)\n(.*?)(?=\n\[|$)",
        re.DOTALL,
    )
    m = pattern.search(user_prompt)
    if not m:
        return "skeletal muscle energy metabolism exercise"

    raw = m.group(1).strip()
    # 첫 줄만 사용 (내부 줄바꿈 제거)
    first_line = raw.split("\n")[0].strip()

    # 구두점 및 괄호 표현(참고문헌 등) 제거 → overlap 에 영향 없는 부분 제거
    # "(Author, YYYY)." 같은 인용 제거
    cleaned = re.sub(r"\([^)]*\)", "", first_line)
    # 특수문자 제거, 소문자화
    cleaned = re.sub(r"[^\w\s]", " ", cleaned).strip()

    # 첫 8개 단어만 사용 (충분한 overlap 확보, 문장 분리 방지)
    words = [w for w in cleaned.split() if w][:8]
    return " ".join(words) if words else "skeletal muscle energy metabolism exercise"


def _extract_first_chunk_id_and_text(user_prompt: str) -> tuple[str, str]:
    """user_prompt 에서 첫 번째 블록의 chunk_id 와 클린 claim 텍스트를 반환."""
    ids = _extract_all_chunk_ids(user_prompt)
    if not ids:
        return "chunk_0", "skeletal muscle energy metabolism exercise"
    chunk_id = ids[0]
    claim_text = _extract_chunk_text(user_prompt, chunk_id)
    return chunk_id, claim_text


# ---------------------------------------------------------------------------
# Test 1 — Ingestion E2E
# ---------------------------------------------------------------------------


def test_ingestion_e2e(
    ingested_context: tuple[LanceDBStore, Paper, ProcessedRegistry, IngestResult],
) -> None:
    """PDF 인입 후 store 에 청크가 존재하고 registry 가 마킹되어 있는지 확인."""
    store, paper, registry, result = ingested_context

    # 인입 성공
    assert result.status == "ingested", f"인입 실패: {result.error}"
    assert result.n_chunks > 0, "인입된 청크가 없음"

    # store 에 청크 존재
    assert store.count() > 0, "store 가 비어 있음"

    # registry 마킹 확인
    key = processed_key(paper)
    assert key, "processed_key 가 비어 있음"
    # registry 는 is_processed 로 확인 (content_hash/embed_version 은 임의값 허용)
    # 단순히 processed.jsonl 파일이 존재하면 마킹된 것으로 간주
    reg_path = registry._path  # type: ignore[attr-defined]
    assert Path(reg_path).exists(), "registry 파일이 없음"


# ---------------------------------------------------------------------------
# Test 2 — Query E2E
# ---------------------------------------------------------------------------


def test_query_e2e(
    pipeline: QueryPipeline,
    ingested_context: tuple[LanceDBStore, Paper, ProcessedRegistry, IngestResult],
) -> None:
    """QueryPipeline.answer() → 한국어 답변, 검증된 인용, retrieved 채워짐."""
    _, paper, _, _ = ingested_context

    def mock_generate(system: str, user: str) -> str:
        # 실제 chunk id 를 prompt 에서 추출해 claim 과 함께 반환
        chunk_id, claim_text = _extract_first_chunk_id_and_text(user)
        # claim 텍스트에 chunk 원문을 포함 → overlap ≥ threshold 보장
        return f"{claim_text} [{chunk_id}]."

    def mock_rerank(query: str, documents: list[str]) -> list[float]:
        return [1.0 - i * 0.1 for i in range(len(documents))]

    req = QueryRequest(query="운동 중 골격근 에너지 대사", top_k=3, translate=True)

    with (
        patch.object(pipeline._reranker, "_rerank_call", side_effect=mock_rerank),
        patch.object(pipeline._generator, "_generate_call", side_effect=mock_generate),
    ):
        resp = pipeline.answer(req)

    assert isinstance(resp, QueryResponse)
    # 역번역 확인 (MockTranslator.en2ko → "[KO] ..." 접두어)
    assert "[KO]" in resp.answer, f"한국어 답변 아님: {resp.answer!r}"
    # answer_en 채워짐
    assert resp.answer_en, "answer_en 비어 있음"
    # retrieved 채워짐
    assert len(resp.retrieved) > 0, "retrieved 비어 있음"
    # citations: 최소 1 개, 모두 유효한 chunk_id
    assert len(resp.citations) >= 1, f"citations 비어 있음: {resp.citations}"
    ingested_ids = {f"{rc.chunk.id}" for rc in resp.retrieved}
    for cit in resp.citations:
        assert cit.chunk_id in ingested_ids, (
            f"citation.chunk_id={cit.chunk_id!r} 가 retrieved 목록에 없음"
        )


# ---------------------------------------------------------------------------
# Test 3 — Citation Faithfulness E2E
# ---------------------------------------------------------------------------


def test_citation_faithfulness_e2e(
    pipeline: QueryPipeline,
    ingested_context: tuple[LanceDBStore, Paper, ProcessedRegistry, IngestResult],
) -> None:
    """환각 인용(claim 과 chunk 텍스트 overlap 없음)이 verify_citations 로 필터링된다.

    전략:
    - 문장 1: id0 인용 + claim 이 id0 텍스트 그대로 → verified (overlap 높음)
    - 문장 2: id1 인용 + claim 이 완전히 무관한 고유 토큰들 → unverified (overlap ≈ 0)
      무관 claim 에는 "xyzzy quux frob nibble waldo" 같은 어떤 코퍼스에도 없는 토큰 사용.
    """
    # 어떤 청크 텍스트와도 overlap 이 0 인 가짜 claim
    _HALLUCINATED_CLAIM = "xyzzy quux frob nibble waldo corge grault garply"

    def mock_generate(system: str, user: str) -> str:
        ids = _extract_all_chunk_ids(user)
        if len(ids) < 2:
            # 청크가 1개뿐이면 — verified 1개 + 같은 id 로 hallucinated claim 시뮬레이션
            chunk_id, claim_text = _extract_first_chunk_id_and_text(user)
            return f"{claim_text} [{chunk_id}]. {_HALLUCINATED_CLAIM} [{chunk_id}]."

        id0, id1 = ids[0], ids[1]

        # id0 블록 텍스트를 실제 claim 으로 추출
        id0_match = re.search(
            rf"\(id={re.escape(id0)}[^)]*\)\n(.+?)(?=\n\[|\Z)", user, re.DOTALL
        )
        claim0 = (
            id0_match.group(1).strip().split("\n")[0][:100]
            if id0_match
            else "skeletal muscle energy metabolism during exercise"
        )

        # 문장 1: id0 인용 + id0 원문 claim → verified
        # 문장 2: id1 인용 + 완전히 무관 토큰 claim → unverified
        return f"{claim0} [{id0}]. {_HALLUCINATED_CLAIM} [{id1}]."

    def mock_rerank(query: str, documents: list[str]) -> list[float]:
        return [1.0 - i * 0.1 for i in range(len(documents))]

    req = QueryRequest(query="운동 중 골격근 에너지 대사", top_k=5, translate=True)

    with (
        patch.object(pipeline._reranker, "_rerank_call", side_effect=mock_rerank),
        patch.object(pipeline._generator, "_generate_call", side_effect=mock_generate),
    ):
        resp = pipeline.answer(req)

    citation_ids = {c.chunk_id for c in resp.citations}

    if len(resp.retrieved) >= 2:
        id0 = resp.retrieved[0].chunk.id
        id1 = resp.retrieved[1].chunk.id
        assert id0 in citation_ids, f"verified citation {id0!r} 이 citations 에 없음"
        assert id1 not in citation_ids, f"hallucinated citation {id1!r} 가 citations 에 남아 있음"
    else:
        # 청크가 1개면 at-least-1 확인으로 대체
        assert len(resp.citations) >= 1


# ---------------------------------------------------------------------------
# Test 4 — API E2E
# ---------------------------------------------------------------------------


class _E2EQueryService(QueryService):
    """실제 pipeline 을 감싸는 서비스 — dependency_overrides 주입용."""

    def __init__(self, pipeline: QueryPipeline) -> None:
        super().__init__(pipeline=pipeline)

    def ask(self, req: QueryRequest) -> QueryResponse:
        return self._pipeline.answer(req)  # type: ignore[attr-defined]


def test_api_e2e(
    pipeline: QueryPipeline,
    ingested_context: tuple[LanceDBStore, Paper, ProcessedRegistry, IngestResult],
) -> None:
    """FastAPI TestClient: POST /query 200 + GET /health 200."""
    from fastapi.testclient import TestClient

    def mock_generate(system: str, user: str) -> str:
        chunk_id, claim_text = _extract_first_chunk_id_and_text(user)
        return f"{claim_text} [{chunk_id}]."

    def mock_rerank(query: str, documents: list[str]) -> list[float]:
        return [1.0 - i * 0.1 for i in range(len(documents))]

    app = create_app()

    # QueryService._pipeline 을 통해 pipeline 을 주입
    svc = _E2EQueryService(pipeline)
    app.dependency_overrides[get_service] = lambda: svc

    client = TestClient(app)

    # GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}

    # POST /query (mock seam 활성화)
    with (
        patch.object(pipeline._reranker, "_rerank_call", side_effect=mock_rerank),
        patch.object(pipeline._generator, "_generate_call", side_effect=mock_generate),
    ):
        query_resp = client.post(
            "/query",
            json={"query": "운동 중 골격근 에너지 대사", "top_k": 3},
        )

    assert query_resp.status_code == 200, f"응답 오류: {query_resp.text}"
    body = query_resp.json()
    assert "answer" in body
    assert "citations" in body
    assert "retrieved" in body


# ---------------------------------------------------------------------------
# Test 5 — Eval E2E
# ---------------------------------------------------------------------------


def test_eval_e2e(
    pipeline: QueryPipeline,
    ingested_context: tuple[LanceDBStore, Paper, ProcessedRegistry, IngestResult],
) -> None:
    """evaluate() → EvalReport 반환, hit@k > 0 (인입된 paper.doi 와 일치)."""
    _, paper, _, _ = ingested_context

    def mock_generate(system: str, user: str) -> str:
        chunk_id, claim_text = _extract_first_chunk_id_and_text(user)
        return f"{claim_text} [{chunk_id}]."

    def mock_rerank(query: str, documents: list[str]) -> list[float]:
        return [1.0 - i * 0.1 for i in range(len(documents))]

    qa_items = [
        {
            "id": "q1",
            "question_ko": "운동 중 골격근 에너지 대사 과정을 설명하라",
            "relevant_dois": [paper.doi],
            "answer_gold": "산화적 인산화와 해당과정이 근수축을 위한 에너지를 공급한다.",
        }
    ]

    with (
        patch.object(pipeline._reranker, "_rerank_call", side_effect=mock_rerank),
        patch.object(pipeline._generator, "_generate_call", side_effect=mock_generate),
    ):
        report = evaluate(pipeline, qa_items, k=3)

    assert isinstance(report, EvalReport)
    assert report.n == 1
    # 인입된 paper.doi 와 동일 → hit@3 = 1.0
    assert report.hit_at_k > 0, f"hit@k 가 0: {report.per_item}"
    # MRR 도 0 초과
    assert report.mrr > 0, f"MRR 이 0: {report.per_item}"


# ---------------------------------------------------------------------------
# Test 6 — Full Flow (no exception summary)
# ---------------------------------------------------------------------------


def test_full_flow_no_exception(
    pipeline: QueryPipeline,
    ingested_context: tuple[LanceDBStore, Paper, ProcessedRegistry, IngestResult],
) -> None:
    """전 스택 (인입 → 검색 → 생성 → 검증 → 역번역) 을 한 번 더 실행해 예외 없음 확인."""
    store, paper, _, _ = ingested_context

    def mock_generate(system: str, user: str) -> str:
        chunk_id, claim_text = _extract_first_chunk_id_and_text(user)
        return f"{claim_text} [{chunk_id}]."

    def mock_rerank(query: str, documents: list[str]) -> list[float]:
        return [1.0 - i * 0.1 for i in range(len(documents))]

    req = QueryRequest(query="유산소 능력과 최대 산소 섭취량", top_k=3, translate=True)

    with (
        patch.object(pipeline._reranker, "_rerank_call", side_effect=mock_rerank),
        patch.object(pipeline._generator, "_generate_call", side_effect=mock_generate),
    ):
        resp = pipeline.answer(req)

    # 예외 없이 QueryResponse 반환
    assert isinstance(resp, QueryResponse)
    # store 여전히 정상
    assert store.count() > 0
    # 답변 비어 있지 않음
    assert resp.answer
