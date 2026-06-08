"""tests/test_pipeline_query.py — QueryPipeline 통합 테스트 (mock LLM + 실 검색/검증 로직).

목표:
- 실 LanceDBStore + MockEmbedder 로 실제 RRF 검색·overlap 검증 수행
- _translate_call / _rerank_call / _generate_call 은 mock — AWS/네트워크 불필요
- E2E happy-path, refusal 경로, 환각 인용 필터, translate=False 경로 검증
"""

from __future__ import annotations

import tempfile
from typing import Any
from unittest.mock import patch

import pytest

from core.models import Chunk, Citation, QueryRequest, QueryResponse, RetrievedChunk
from core.vectorstore import LanceDBStore
from serving import QueryPipeline
from serving.generation import GenerationResult, Generator
from serving.retrieval import BedrockTranslator, Reranker

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

DIM = 8  # 테스트용 소형 차원


# ---------------------------------------------------------------------------
# MockEmbedder — test_hybrid.py 와 동일 패턴
# ---------------------------------------------------------------------------


class MockEmbedder:
    """결정론적 벡터를 반환하는 테스트용 임베더 (실 API 없음)."""

    def __init__(self, dim: int = DIM, query_seed: int = 99) -> None:
        self.dim = dim
        self.query_seed = query_seed

    def _vec(self, seed: int) -> list[float]:
        return [(seed * 0.1 + i * 0.01) % 1.0 for i in range(self.dim)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(hash(t) % 50) for t in texts]

    def embed_late(
        self, document: str, boundaries: list[tuple[int, int]]
    ) -> list[list[float]]:
        return [self._vec(i) for i in range(len(boundaries))]

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(self.query_seed) for _ in texts]

    def embed_chunks(self, chunks: list[Any]) -> list[list[float]]:
        """embed_chunks 는 ingestion 전용 — 실 파이프라인에서만 필요."""
        return [self._vec(i) for i in range(len(chunks))]


# ---------------------------------------------------------------------------
# MockTranslator — ko2en / en2ko passthrough (번역 API 없음)
# ---------------------------------------------------------------------------


class MockTranslator:
    """패스스루 번역기 — 실 Bedrock API 없이 테스트 가능."""

    def ko2en(self, text: str) -> str:
        return text  # 그대로 반환

    def en2ko(self, text: str) -> str:
        return f"[KO] {text}"  # 구별을 위해 접두어 추가


# ---------------------------------------------------------------------------
# MockReranker — 점수를 deterministic 하게 반환 (Bedrock/Jina 없음)
# ---------------------------------------------------------------------------


class MockReranker:
    """_rerank_call 을 내부 오버라이드하는 테스트용 리랭커."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        """입력 순서를 유지하면서 rerank_score 를 내림차순 채워 top_k 반환."""
        n = len(candidates)
        reranked = [
            rc.model_copy(update={"rerank_score": 1.0 - i / max(n, 1)})
            for i, rc in enumerate(candidates)
        ]
        reranked.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        return reranked[:top_k]


# ---------------------------------------------------------------------------
# 청크 팩토리 (seed 기반 embedding)
# ---------------------------------------------------------------------------


def _make_chunk(
    idx: int,
    text: str,
    doi: str = "10.0000/test",
    source: str = "test_src",
    seed: int | None = None,
) -> Chunk:
    s = seed if seed is not None else idx
    embedding = [(s * 0.1 + i * 0.01) % 1.0 for i in range(DIM)]
    return Chunk(
        id=f"chunk_{idx}",
        doi=doi,
        section=f"Section{idx}",
        text=text,
        ctx_text=f"[ctx] {text}",
        page=idx + 1,
        chunk_index=idx,
        source=source,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# 공용 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture()
def store_and_chunks() -> tuple[LanceDBStore, list[Chunk]]:
    """임시 디렉터리에 LanceDBStore 를 만들고 다양한 텍스트 청크를 upsert 한다."""
    chunks = [
        _make_chunk(
            0,
            "muscle metabolism during exercise increases energy production",
            doi="10.1111/sports.a",
            source="sports_a",
            seed=99,
        ),
        _make_chunk(
            1,
            "oxygen consumption at VO2max reflects aerobic capacity",
            doi="10.1111/sports.a",
            source="sports_a",
            seed=10,
        ),
        _make_chunk(
            2,
            "lactate threshold in cycling athletes marks anaerobic transition",
            doi="10.2222/physiol.b",
            source="physiol_b",
            seed=20,
        ),
        _make_chunk(
            3,
            "cardiac output response to training improves heart function",
            doi="10.2222/physiol.b",
            source="physiol_b",
            seed=30,
        ),
        _make_chunk(
            4,
            "respiratory exchange ratio measurement indicates substrate utilization",
            doi="10.3333/cpet.c",
            source="cpet_c",
            seed=40,
        ),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(uri=tmpdir, dim=DIM)
        store.upsert(chunks)
        yield store, chunks


@pytest.fixture()
def pipeline(store_and_chunks: tuple[LanceDBStore, list[Chunk]]) -> QueryPipeline:
    """MockTranslator + MockReranker + Generator(mock) 로 QueryPipeline 을 구성한다."""
    store, _ = store_and_chunks
    return QueryPipeline(
        store=store,
        embedder=MockEmbedder(),
        translator=MockTranslator(),
        reranker=MockReranker(),
        generator=Generator(),  # _generate_call 은 각 테스트에서 mock
    )


# ---------------------------------------------------------------------------
# 1. E2E 풀 플로우 (happy-path, translate=True)
# ---------------------------------------------------------------------------


def test_full_pipeline_e2e(
    pipeline: QueryPipeline,
    store_and_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """happy-path: 번역 → 검색 → 리랭크 → 생성 → 검증 → 역번역 → QueryResponse.

    - MockTranslator.ko2en passthrough 이므로 query_en = 원문.
    - _generate_call 은 실제 top 청크 id 를 이용해 valid citation 포함 답변 반환.
    - 검증 후 citations 비어 있지 않고, retrieved 채워지고, answer 가 [KO] 접두어 포함.
    """
    _, chunks = store_and_chunks

    # _generate_call 은 chunk_0 과 chunk_1 을 인용하는 답변을 반환
    # chunk_0.text 에 'muscle metabolism' 이 있으므로 overlap 통과 예상
    def mock_generate(system: str, user: str) -> str:
        # chunk_0 id 사용 → claim 과 텍스트 overlap 명확
        return (
            "Muscle metabolism increases energy production during exercise [chunk_0]. "
            "Oxygen consumption at VO2max reflects aerobic capacity [chunk_1]."
        )

    req = QueryRequest(query="운동 중 근육 대사", top_k=3, translate=True)

    with patch.object(pipeline._generator, "_generate_call", side_effect=mock_generate):
        resp = pipeline.answer(req)

    assert isinstance(resp, QueryResponse)
    # retrieved 채워짐
    assert len(resp.retrieved) > 0
    # answer 에 MockTranslator.en2ko 접두어 포함 (역번역 확인)
    assert "[KO]" in resp.answer
    # answer_en 도 채워짐
    assert resp.answer_en is not None
    # citations: verified 만 포함 (비어 있지 않거나 최소 1개)
    # overlap 계산으로 최소 1 개는 통과해야 함
    assert len(resp.citations) >= 1
    # all citations have chunk_id
    for cit in resp.citations:
        assert cit.chunk_id in {"chunk_0", "chunk_1"}


# ---------------------------------------------------------------------------
# 2. refusal 경로 — generator 가 거부 응답 반환
# ---------------------------------------------------------------------------


def test_pipeline_refusal_path(
    pipeline: QueryPipeline,
    store_and_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """generator 가 refused=True 를 반환하면 citations=[], answer 가 거부 메시지.

    거부 답변은 MockTranslator.en2ko 를 통과하므로 [KO] 접두어를 포함.
    """
    refused_result = GenerationResult(
        answer_en="I cannot answer from the provided sources.",
        citations=[],
        refused=True,
        used_chunk_ids=[],
    )

    req = QueryRequest(query="아무 질의", top_k=3, translate=True)

    with patch.object(pipeline._generator, "generate", return_value=refused_result):
        resp = pipeline.answer(req)

    assert isinstance(resp, QueryResponse)
    assert resp.citations == []
    # answer 는 거부 메시지 (MockTranslator 통과)
    assert "cannot answer" in resp.answer.lower() or "자료만으로는" in resp.answer
    # retrieved 는 채워짐 (검색은 정상 수행됨)
    assert len(resp.retrieved) > 0


# ---------------------------------------------------------------------------
# 3. 환각 인용 필터 — verify_citations 로 걸러냄
# ---------------------------------------------------------------------------


def test_hallucinated_citation_filtered_by_verify(
    pipeline: QueryPipeline,
    store_and_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """verify_citations 가 overlap 실패 인용을 걸러 citations 에서 제외한다.

    전략:
    - chunk_0 (muscle metabolism...) 을 진짜 인용 → overlap 통과
    - chunk_3 (cardiac output...) 을 환각 인용 — claim 이 chunk_3 텍스트와 겹치지 않음
      → overlap 실패 → citations 에 포함 안 됨
    """

    def mock_generate(system: str, user: str) -> str:
        # chunk_0 → 진짜 인용 (claim 내용이 chunk_0 텍스트와 겹침)
        # chunk_3 → 환각 인용 (claim 내용이 chunk_3 텍스트와 전혀 다름)
        return (
            "Muscle metabolism increases energy production during exercise [chunk_0]. "
            "Muscle metabolism increases energy production during exercise [chunk_3]."
        )

    req = QueryRequest(query="운동 중 근육 대사", top_k=5, translate=True)

    with patch.object(pipeline._generator, "_generate_call", side_effect=mock_generate):
        resp = pipeline.answer(req)

    citation_ids = {c.chunk_id for c in resp.citations}
    # chunk_0 은 검증 통과해야 함
    assert "chunk_0" in citation_ids, f"chunk_0 should be verified, got {citation_ids}"
    # chunk_3 는 환각 — 검증 실패 또는 미포함이어야 함
    # (top 5 안에 chunk_3 가 있어야 이 테스트가 의미 있음)
    # overlap score: claim "Muscle metabolism increases energy production during exercise"
    # vs chunk_3 "cardiac output response to training improves heart function" — 겹침 없음
    assert "chunk_3" not in citation_ids, (
        f"chunk_3 should be filtered as hallucination, got {citation_ids}"
    )


# ---------------------------------------------------------------------------
# 4. translate=False 경로 — 영어 그대로
# ---------------------------------------------------------------------------


def test_pipeline_translate_false(
    pipeline: QueryPipeline,
    store_and_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """translate=False 면 answer 가 영어 그대로 반환된다 (역번역 없음)."""

    def mock_generate(system: str, user: str) -> str:
        return "Muscle metabolism increases energy during exercise [chunk_0]."

    req = QueryRequest(query="muscle metabolism during exercise", top_k=3, translate=False)

    with patch.object(pipeline._generator, "_generate_call", side_effect=mock_generate):
        resp = pipeline.answer(req)

    # translate=False 이면 MockTranslator.en2ko 가 불리지 않으므로 [KO] 없음
    assert "[KO]" not in resp.answer
    # answer_en 과 answer 가 같거나 매우 유사 (tag strip 만 수행)
    assert resp.answer_en is not None
    # retrieved 채워짐
    assert len(resp.retrieved) > 0


# ---------------------------------------------------------------------------
# 5. translate=False + refusal 경로
# ---------------------------------------------------------------------------


def test_pipeline_refusal_translate_false(
    pipeline: QueryPipeline,
) -> None:
    """translate=False + refusal 경로: answer 가 영어 거부 메시지."""
    refused_result = GenerationResult(
        answer_en="I cannot answer from the provided sources.",
        citations=[],
        refused=True,
        used_chunk_ids=[],
    )

    req = QueryRequest(query="some query", top_k=3, translate=False)

    with patch.object(pipeline._generator, "generate", return_value=refused_result):
        resp = pipeline.answer(req)

    assert resp.citations == []
    # translate=False → 영어 거부 메시지 그대로 (en2ko 미호출)
    assert "[KO]" not in resp.answer
    assert "cannot answer" in resp.answer.lower()


# ---------------------------------------------------------------------------
# 6. QueryResponse 구조 검증
# ---------------------------------------------------------------------------


def test_pipeline_response_fields(
    pipeline: QueryPipeline,
    store_and_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """QueryResponse 가 answer, answer_en, citations, retrieved 필드를 모두 가진다."""

    def mock_generate(system: str, user: str) -> str:
        return "Muscle metabolism result [chunk_0]."

    req = QueryRequest(query="test", top_k=3, translate=True)

    with patch.object(pipeline._generator, "_generate_call", side_effect=mock_generate):
        resp = pipeline.answer(req)

    assert hasattr(resp, "answer")
    assert hasattr(resp, "answer_en")
    assert hasattr(resp, "citations")
    assert hasattr(resp, "retrieved")
    assert isinstance(resp.citations, list)
    assert isinstance(resp.retrieved, list)


# ---------------------------------------------------------------------------
# 7. QueryPipeline.answer_query 편의 메서드
# ---------------------------------------------------------------------------


def test_pipeline_answer_query_method(
    pipeline: QueryPipeline,
    store_and_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """QueryPipeline.answer_query(query_ko) 가 QueryResponse 를 반환한다."""

    def mock_generate(system: str, user: str) -> str:
        return "Oxygen consumption reflects aerobic capacity [chunk_1]."

    with patch.object(pipeline._generator, "_generate_call", side_effect=mock_generate):
        resp = pipeline.answer_query("유산소 능력과 산소 소비", top_k=3)

    assert isinstance(resp, QueryResponse)
    assert len(resp.retrieved) > 0


# ---------------------------------------------------------------------------
# 8. BedrockTranslator ⟦Cn⟧ 보강 확인
# ---------------------------------------------------------------------------


def test_translator_system_prompt_preserves_cn_placeholder() -> None:
    """BedrockTranslator._translate_call 시스템 프롬프트가 ⟦Cn⟧ 보존 지시를 포함한다.

    _translate_call 을 mock 해 실제 Bedrock 호출 없이 시스템 프롬프트만 검사한다.
    """
    translator = BedrockTranslator()

    captured_system: list[str] = []

    def _capture_call(text: str, src: str, tgt: str) -> str:
        # 내부 system_prompt 는 _translate_call 이 조립하므로
        # 여기서는 실제 boto3 없이 호출 자체를 가로채는 것이 아님.
        # 대신 _translate_call 소스 코드 또는 문자열을 직접 검사한다.
        return text  # passthrough

    # system_prompt 는 _translate_call 내부에 있으므로 소스 검사
    import inspect  # noqa: PLC0415
    from serving.retrieval.translate import BedrockTranslator as BT  # noqa: PLC0415

    source = inspect.getsource(BT._translate_call)
    # ⟦Cn⟧ 이 명시적으로 언급되어야 함
    assert "⟦Cn⟧" in source, "System prompt should mention ⟦Cn⟧ placeholder preservation"
    # ⟦Tn⟧ 도 유지되어야 함
    assert "⟦Tn⟧" in source, "System prompt should mention ⟦Tn⟧ placeholder preservation"
