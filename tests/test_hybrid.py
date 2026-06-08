"""tests/test_hybrid.py — HybridRetriever 통합 테스트 (로컬 임시 LanceDB + MockEmbedder).

모든 테스트는 실제 Jina API / Bedrock 없이 로컬에서 실행 가능하다.
"""

from __future__ import annotations

import tempfile
from typing import Any

import pytest

from core.models import Chunk, Paper, RetrievedChunk
from core.vectorstore import LanceDBStore
from serving.retrieval import HybridRetriever
from serving.retrieval.hybrid import _K_RRF

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

DIM = 8  # 테스트용 작은 차원


# ---------------------------------------------------------------------------
# MockEmbedder
# ---------------------------------------------------------------------------


class MockEmbedder:
    """결정론적 벡터를 반환하는 테스트용 임베더.

    embed / embed_late / embed_query 세 메서드를 모두 구현한다.
    embed_query 는 쿼리 벡터(seed=99)를 반환 — 벡터 검색 대상 chunk 를 제어하기 쉽다.
    """

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
        """쿼리 벡터 반환 — query_seed 로 고정된 벡터."""
        return [self._vec(self.query_seed) for _ in texts]


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------


def _make_chunk(
    idx: int,
    text: str,
    doi: str = "10.0000/test",
    source: str = "test_src",
    seed: int | None = None,
) -> Chunk:
    """테스트용 Chunk 생성. embedding 은 seed 기반 결정론적 벡터로 미리 설정."""
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


@pytest.fixture()
def store_with_chunks() -> tuple[LanceDBStore, list[Chunk]]:
    """임시 디렉터리에 LanceDBStore 를 생성하고 다양한 청크를 채운다."""
    chunks = [
        # doi_a 논문 (year=2021, journal="Sports Science")
        _make_chunk(0, "muscle metabolism during exercise", doi="10.1111/sports.a", source="sports_a", seed=99),
        _make_chunk(1, "oxygen consumption at VO2max", doi="10.1111/sports.a", source="sports_a", seed=10),
        _make_chunk(2, "lactate threshold in cycling athletes", doi="10.1111/sports.a", source="sports_a", seed=20),
        # doi_b 논문 (year=2018, journal="Physiology Review")
        _make_chunk(3, "cardiac output response to training", doi="10.2222/physiol.b", source="physiol_b", seed=30),
        _make_chunk(4, "respiratory exchange ratio measurement", doi="10.2222/physiol.b", source="physiol_b", seed=40),
        # doi_c 논문 (year=2023, journal="Sports Science")
        _make_chunk(5, "CPET protocol for maximal effort", doi="10.3333/cpet.c", source="cpet_c", seed=50),
        _make_chunk(6, "VO2max prediction from submaximal test", doi="10.3333/cpet.c", source="cpet_c", seed=60),
        # doi_d 논문 (year=2015, journal="Old Journal")
        _make_chunk(7, "historical overview of exercise physiology", doi="10.4444/old.d", source="old_d", seed=70),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(uri=tmpdir, dim=DIM)
        store.upsert(chunks)
        yield store, chunks


@pytest.fixture()
def papers_by_doi() -> dict[str, Paper]:
    """테스트용 papers_by_doi 딕셔너리."""
    return {
        "10.1111/sports.a": Paper(
            doi="10.1111/sports.a",
            title="Sports A Paper",
            first_author="Smith",
            year=2021,
            journal="Sports Science",
            source="sports_a",
        ),
        "10.2222/physiol.b": Paper(
            doi="10.2222/physiol.b",
            title="Physiology B Paper",
            first_author="Kim",
            year=2018,
            journal="Physiology Review",
            source="physiol_b",
        ),
        "10.3333/cpet.c": Paper(
            doi="10.3333/cpet.c",
            title="CPET C Paper",
            first_author="Lee",
            year=2023,
            journal="Sports Science",
            source="cpet_c",
        ),
        "10.4444/old.d": Paper(
            doi="10.4444/old.d",
            title="Old D Paper",
            first_author="Jones",
            year=2015,
            journal="Old Journal",
            source="old_d",
        ),
    }


# ---------------------------------------------------------------------------
# 기본 검색 테스트
# ---------------------------------------------------------------------------


def test_retrieve_returns_list_of_retrieved_chunks(
    store_with_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """retrieve 는 RetrievedChunk 리스트를 반환한다."""
    store, _ = store_with_chunks
    embedder = MockEmbedder()
    retriever = HybridRetriever(store, embedder)

    results = retriever.retrieve("muscle metabolism", top_k=3)

    assert isinstance(results, list)
    assert all(isinstance(rc, RetrievedChunk) for rc in results)


def test_retrieve_top_k_limit(
    store_with_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """top_k 이하의 결과만 반환한다."""
    store, _ = store_with_chunks
    embedder = MockEmbedder()
    retriever = HybridRetriever(store, embedder)

    results = retriever.retrieve("muscle metabolism", top_k=3)

    assert len(results) <= 3


def test_retrieve_no_duplicate_ids(
    store_with_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """반환 결과에 중복 chunk.id 가 없다."""
    store, _ = store_with_chunks
    embedder = MockEmbedder()
    retriever = HybridRetriever(store, embedder)

    results = retriever.retrieve("muscle metabolism", top_k=5)

    ids = [rc.chunk.id for rc in results]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"


def test_retrieve_scores_descending(
    store_with_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """결과가 score 내림차순으로 정렬된다."""
    store, _ = store_with_chunks
    embedder = MockEmbedder()
    retriever = HybridRetriever(store, embedder)

    results = retriever.retrieve("oxygen consumption VO2max", top_k=5)

    scores = [rc.score for rc in results]
    assert scores == sorted(scores, reverse=True), f"Scores not sorted: {scores}"


# ---------------------------------------------------------------------------
# RRF 융합 테스트
# ---------------------------------------------------------------------------


def test_rrf_fuse_unit_single_list() -> None:
    """단일 결과 목록에 대해 RRF 점수가 1/(k+rank) 로 계산된다."""
    chunks = [_make_chunk(i, f"text {i}", seed=i) for i in range(3)]
    result_list = [RetrievedChunk(chunk=c, score=1.0) for c in chunks]

    fused = HybridRetriever._rrf_fuse([result_list])

    # rank 1 이 가장 높아야 한다
    expected_scores = [1.0 / (_K_RRF + rank) for rank in range(1, 4)]
    actual_scores = [rc.score for rc in fused]

    for expected, actual in zip(expected_scores, actual_scores):
        assert abs(expected - actual) < 1e-9, f"Expected {expected}, got {actual}"


def test_rrf_fuse_unit_two_lists_score_sum() -> None:
    """두 목록에서 공통 chunk 의 점수가 두 랭크 점수의 합이다."""
    chunk_a = _make_chunk(0, "shared chunk A", seed=0)
    chunk_b = _make_chunk(1, "unique to list1", seed=1)
    chunk_c = _make_chunk(2, "unique to list2", seed=2)

    list1 = [
        RetrievedChunk(chunk=chunk_a, score=0.9),  # rank 1 in list1
        RetrievedChunk(chunk=chunk_b, score=0.8),  # rank 2
    ]
    list2 = [
        RetrievedChunk(chunk=chunk_c, score=0.95),  # rank 1 in list2
        RetrievedChunk(chunk=chunk_a, score=0.7),   # rank 2 in list2
    ]

    fused = HybridRetriever._rrf_fuse([list1, list2])

    # chunk_a 의 기대 점수: 1/(60+1) + 1/(60+2)
    expected_a = 1.0 / (_K_RRF + 1) + 1.0 / (_K_RRF + 2)
    fused_scores = {rc.chunk.id: rc.score for rc in fused}

    assert abs(fused_scores["chunk_0"] - expected_a) < 1e-9
    # chunk_b: 1/(60+2), chunk_c: 1/(60+1)
    assert abs(fused_scores["chunk_1"] - 1.0 / (_K_RRF + 2)) < 1e-9
    assert abs(fused_scores["chunk_2"] - 1.0 / (_K_RRF + 1)) < 1e-9


def test_rrf_fuse_empty_lists() -> None:
    """빈 리스트에 대해 빈 결과를 반환한다."""
    assert HybridRetriever._rrf_fuse([]) == []
    assert HybridRetriever._rrf_fuse([[], []]) == []


def test_retrieve_rrf_reflects_both_vec_and_fts(
    store_with_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """벡터 top chunk 와 FTS top chunk 가 모두 융합 결과에 포함된다.

    전략:
    - query_seed=99 → 벡터 검색에서 seed=99 인 chunk_0("muscle metabolism") 최상위.
    - "muscle metabolism" 텍스트 쿼리 → FTS 에서도 chunk_0 가 최상위이지만
      pool=8 로 충분히 넓혀 다른 chunk 도 포함되는지 확인한다.
    - 핵심: 두 결과 집합의 합집합을 커버해야 한다 (중복 없이 dedup).
    """
    store, chunks = store_with_chunks
    # query_seed=99 → chunk_0(seed=99) 와 벡터 일치
    embedder = MockEmbedder(query_seed=99)
    retriever = HybridRetriever(store, embedder)

    results = retriever.retrieve("muscle metabolism", top_k=8, pool=8)

    # 전체 8개 chunk 중 dedup 해서 반환 — 중복 없음
    ids = [rc.chunk.id for rc in results]
    assert len(ids) == len(set(ids))

    # chunk_0 은 벡터·FTS 둘 다 top → 결과에 포함되어야 함
    assert "chunk_0" in ids, f"Expected chunk_0 in results, got: {ids}"


def test_retrieve_rrf_union_of_both_retrievers(
    store_with_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """RRF 결과가 벡터 only 또는 FTS only 결과보다 다양한 청크를 포함한다.

    단일 검색 방법 대비 RRF 가 두 결과를 합쳐 더 넓은 pool 을 커버함을 확인.
    """
    store, chunks = store_with_chunks
    # query_seed=50 → chunk_5(seed=50, "CPET protocol...") 와 벡터 가까움
    # "cardiac output" → FTS 에서 chunk_3 이 상위
    embedder = MockEmbedder(query_seed=50)
    retriever = HybridRetriever(store, embedder)

    results = retriever.retrieve("cardiac output response", top_k=6, pool=8)

    ids = set(rc.chunk.id for rc in results)
    # chunk_5 (벡터 top candidate) 또는 chunk_3 (FTS candidate) 중 적어도 하나 포함
    assert ids & {"chunk_5", "chunk_3"}, f"Neither vec-top nor fts-top found in {ids}"


# ---------------------------------------------------------------------------
# 메타 필터 테스트
# ---------------------------------------------------------------------------


def test_filter_year_gte(
    store_with_chunks: tuple[LanceDBStore, list[Chunk]],
    papers_by_doi: dict[str, Paper],
) -> None:
    """year_gte 필터로 2020년 이후 논문 청크만 반환한다."""
    store, _ = store_with_chunks
    embedder = MockEmbedder()
    retriever = HybridRetriever(store, embedder, papers_by_doi=papers_by_doi)

    results = retriever.retrieve("exercise", top_k=10, pool=8, filters={"year_gte": 2020})

    # 2020 이상: sports_a(2021), cpet_c(2023)
    allowed_dois = {"10.1111/sports.a", "10.3333/cpet.c"}
    for rc in results:
        assert rc.chunk.doi in allowed_dois, (
            f"Unexpected doi {rc.chunk.doi} in year_gte=2020 results"
        )


def test_filter_journal(
    store_with_chunks: tuple[LanceDBStore, list[Chunk]],
    papers_by_doi: dict[str, Paper],
) -> None:
    """journal 필터로 특정 저널 청크만 반환한다."""
    store, _ = store_with_chunks
    embedder = MockEmbedder()
    retriever = HybridRetriever(store, embedder, papers_by_doi=papers_by_doi)

    results = retriever.retrieve("exercise", top_k=10, pool=8, filters={"journal": "Physiology Review"})

    for rc in results:
        assert rc.chunk.doi == "10.2222/physiol.b", (
            f"Unexpected doi {rc.chunk.doi} in journal=Physiology Review results"
        )


def test_filter_source(
    store_with_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """source 필터로 특정 source 청크만 반환한다 (papers_by_doi 없어도 동작)."""
    store, _ = store_with_chunks
    embedder = MockEmbedder()
    retriever = HybridRetriever(store, embedder)  # papers_by_doi 없음

    results = retriever.retrieve("exercise", top_k=10, pool=8, filters={"source": "cpet_c"})

    for rc in results:
        assert rc.chunk.source == "cpet_c", f"Unexpected source: {rc.chunk.source}"


def test_filter_doi(
    store_with_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """doi 필터로 특정 DOI 청크만 반환한다."""
    store, _ = store_with_chunks
    embedder = MockEmbedder()
    retriever = HybridRetriever(store, embedder)

    results = retriever.retrieve("exercise", top_k=10, pool=8, filters={"doi": "10.4444/old.d"})

    for rc in results:
        assert rc.chunk.doi == "10.4444/old.d", f"Unexpected doi: {rc.chunk.doi}"


def test_filter_year_exact(
    store_with_chunks: tuple[LanceDBStore, list[Chunk]],
    papers_by_doi: dict[str, Paper],
) -> None:
    """year 정확 일치 필터."""
    store, _ = store_with_chunks
    embedder = MockEmbedder()
    retriever = HybridRetriever(store, embedder, papers_by_doi=papers_by_doi)

    results = retriever.retrieve("exercise", top_k=10, pool=8, filters={"year": 2023})

    for rc in results:
        assert rc.chunk.doi == "10.3333/cpet.c", f"Unexpected doi: {rc.chunk.doi}"


def test_filter_no_match_returns_empty(
    store_with_chunks: tuple[LanceDBStore, list[Chunk]],
    papers_by_doi: dict[str, Paper],
) -> None:
    """매칭 없는 필터는 빈 리스트를 반환한다."""
    store, _ = store_with_chunks
    embedder = MockEmbedder()
    retriever = HybridRetriever(store, embedder, papers_by_doi=papers_by_doi)

    results = retriever.retrieve("exercise", top_k=10, pool=8, filters={"year": 1900})

    assert results == []


# ---------------------------------------------------------------------------
# embed_query 호출 검증
# ---------------------------------------------------------------------------


def test_retrieve_uses_embed_query(
    store_with_chunks: tuple[LanceDBStore, list[Chunk]],
) -> None:
    """retrieve 가 embed_query 를 호출한다 (embed 는 호출하지 않음)."""
    from unittest.mock import MagicMock

    store, _ = store_with_chunks

    # embed_query 호출을 추적하는 MockEmbedder
    embedder = MockEmbedder()
    original_embed_query = embedder.embed_query
    call_count: dict[str, int] = {"embed": 0, "embed_query": 0}

    def tracking_embed(texts: list[str]) -> list[list[float]]:
        call_count["embed"] += 1
        return original_embed_query(texts)  # 같은 벡터 반환

    def tracking_embed_query(texts: list[str]) -> list[list[float]]:
        call_count["embed_query"] += 1
        return original_embed_query(texts)

    embedder.embed = tracking_embed  # type: ignore[method-assign]
    embedder.embed_query = tracking_embed_query  # type: ignore[method-assign]

    retriever = HybridRetriever(store, embedder)
    retriever.retrieve("test query", top_k=3)

    assert call_count["embed_query"] == 1, "embed_query should be called once"
    assert call_count["embed"] == 0, "embed should NOT be called by retrieve"
