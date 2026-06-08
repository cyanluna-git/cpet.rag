"""tests/test_rerank.py — Reranker 유닛 테스트 (mock _rerank_call).

모든 테스트는 실제 Bedrock / Jina API 없이 로컬에서 실행 가능하다.
`_rerank_call` 을 mock 해 결정론적 점수를 주입하고 재정렬 동작을 검증한다.
"""

from __future__ import annotations

from unittest.mock import patch


from core.models import Chunk, RetrievedChunk
from serving.retrieval import Reranker

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_chunk(idx: int, text: str) -> Chunk:
    """테스트용 Chunk 생성."""
    return Chunk(
        id=f"chunk_{idx}",
        doi=f"10.0000/test.{idx}",
        section=f"Section{idx}",
        text=text,
        ctx_text=f"[ctx] {text}",
        page=idx + 1,
        chunk_index=idx,
        source="test_src",
    )


def _make_rc(idx: int, text: str, score: float = 1.0) -> RetrievedChunk:
    """RetrievedChunk 생성 (rerank_score 는 None)."""
    return RetrievedChunk(chunk=_make_chunk(idx, text), score=score)


# ---------------------------------------------------------------------------
# mock 점수 생성기 — 쿼리 단어를 포함하는 문서에 높은 점수 부여
# ---------------------------------------------------------------------------


def _keyword_scorer(query: str, documents: list[str]) -> list[float]:
    """쿼리 단어 포함 여부로 점수를 결정하는 deterministic mock scorer.

    쿼리를 공백으로 분리해 단어 수를 세고 (0–1) 사이의 점수를 반환한다.
    """
    query_words = set(query.lower().split())
    scores: list[float] = []
    for doc in documents:
        doc_lower = doc.lower()
        matched = sum(1 for w in query_words if w in doc_lower)
        scores.append(matched / max(len(query_words), 1))
    return scores


# ---------------------------------------------------------------------------
# 기본 동작 테스트
# ---------------------------------------------------------------------------


def test_rerank_returns_top_k() -> None:
    """rerank 는 최대 top_k 개의 청크를 반환한다."""
    candidates = [_make_rc(i, f"document {i} text", score=1.0) for i in range(10)]
    reranker = Reranker(backend="bedrock")

    with patch.object(
        reranker,
        "_rerank_call",
        side_effect=lambda q, docs: _keyword_scorer(q, docs),
    ):
        results = reranker.rerank("document text", candidates, top_k=5)

    assert len(results) <= 5


def test_rerank_sets_rerank_score() -> None:
    """rerank 후 모든 반환 항목에 rerank_score 가 설정된다."""
    candidates = [_make_rc(i, f"text {i}", score=float(i)) for i in range(5)]
    reranker = Reranker(backend="jina")

    fixed_scores = [0.1, 0.9, 0.5, 0.3, 0.7]
    with patch.object(reranker, "_rerank_call", return_value=fixed_scores):
        results = reranker.rerank("query", candidates, top_k=5)

    for rc in results:
        assert rc.rerank_score is not None, "rerank_score 가 None 이어서는 안 된다"


def test_rerank_sorted_descending() -> None:
    """rerank 결과는 rerank_score 내림차순으로 정렬된다."""
    candidates = [_make_rc(i, f"text {i}", score=1.0) for i in range(5)]
    reranker = Reranker(backend="bedrock")

    fixed_scores = [0.2, 0.8, 0.5, 0.9, 0.1]
    with patch.object(reranker, "_rerank_call", return_value=fixed_scores):
        results = reranker.rerank("query", candidates, top_k=5)

    rscores = [rc.rerank_score for rc in results]
    assert rscores == sorted(rscores, reverse=True), f"정렬 안 됨: {rscores}"


# ---------------------------------------------------------------------------
# 재정렬이 초기 score 순서를 실제로 바꾸는 케이스
# ---------------------------------------------------------------------------


def test_rerank_changes_order() -> None:
    """초기 score 순서와 rerank 후 순서가 달라진다.

    candidates 는 score 내림차순(c0>c1>c2>c3>c4) 이지만
    리랭커는 c3 을 최고 점수로 평가한다 → 순서 역전.
    """
    # 초기 score: c0(0.9) > c1(0.7) > c2(0.5) > c3(0.3) > c4(0.1)
    candidates = [
        _make_rc(0, "alpha text", score=0.9),
        _make_rc(1, "beta text", score=0.7),
        _make_rc(2, "gamma text", score=0.5),
        _make_rc(3, "delta text", score=0.3),
        _make_rc(4, "epsilon text", score=0.1),
    ]
    # rerank: c3(0.99) > c1(0.80) > c4(0.60) > c0(0.40) > c2(0.10)
    rerank_scores = [0.40, 0.80, 0.10, 0.99, 0.60]
    reranker = Reranker(backend="bedrock")

    with patch.object(reranker, "_rerank_call", return_value=rerank_scores):
        results = reranker.rerank("query", candidates, top_k=5)

    # 최상위는 c3 이어야 한다 (초기 score 는 꼴찌에서 두 번째)
    assert (
        results[0].chunk.id == "chunk_3"
    ), f"최상위 청크가 chunk_3 이어야 하지만 {results[0].chunk.id} 임"

    # 초기 top(c0)은 4위로 내려가야 한다
    result_ids = [rc.chunk.id for rc in results]
    assert result_ids.index("chunk_0") > result_ids.index(
        "chunk_3"
    ), "초기 top chunk_0 이 chunk_3 보다 높은 위치에 있어서는 안 된다"


# ---------------------------------------------------------------------------
# non-mutating 테스트
# ---------------------------------------------------------------------------


def test_rerank_non_mutating() -> None:
    """rerank 는 원본 candidates 의 rerank_score 를 변경하지 않는다."""
    candidates = [_make_rc(i, f"text {i}", score=1.0) for i in range(5)]
    # 원본 rerank_score 는 모두 None
    original_rerank_scores = [rc.rerank_score for rc in candidates]

    reranker = Reranker(backend="bedrock")
    fixed_scores = [0.5, 0.9, 0.3, 0.7, 0.1]
    with patch.object(reranker, "_rerank_call", return_value=fixed_scores):
        _results = reranker.rerank("query", candidates, top_k=5)

    # 원본 candidates 의 rerank_score 는 여전히 None 이어야 한다
    for orig, rc in zip(original_rerank_scores, candidates):
        assert (
            rc.rerank_score == orig
        ), f"chunk {rc.chunk.id}: 원본 rerank_score={orig} 가 변경됨 → {rc.rerank_score}"


# ---------------------------------------------------------------------------
# 엣지 케이스
# ---------------------------------------------------------------------------


def test_rerank_empty_candidates() -> None:
    """빈 candidates → 빈 리스트 반환, _rerank_call 호출 없음."""
    reranker = Reranker(backend="bedrock")

    with patch.object(reranker, "_rerank_call") as mock_call:
        results = reranker.rerank("query", [], top_k=8)

    assert results == []
    mock_call.assert_not_called()


def test_rerank_fewer_candidates_than_top_k() -> None:
    """candidates 수가 top_k 보다 적으면 전체를 반환한다."""
    candidates = [_make_rc(i, f"text {i}", score=1.0) for i in range(3)]
    reranker = Reranker(backend="jina")

    fixed_scores = [0.4, 0.9, 0.2]
    with patch.object(reranker, "_rerank_call", return_value=fixed_scores):
        results = reranker.rerank("query", candidates, top_k=8)

    assert len(results) == 3


# ---------------------------------------------------------------------------
# 키워드 기반 결정적 점수 테스트 (rerank 순서가 의미론적으로 맞는지 확인)
# ---------------------------------------------------------------------------


def test_rerank_keyword_scorer_relevance() -> None:
    """쿼리 단어를 많이 포함한 문서가 높은 rerank_score 를 얻는다."""
    query = "VO2max oxygen consumption exercise"
    candidates = [
        _make_rc(0, "lactate threshold cycling athletes"),
        _make_rc(1, "VO2max oxygen consumption during maximal exercise test"),
        _make_rc(2, "cardiac output heart rate measurement"),
        _make_rc(3, "VO2max prediction from submaximal exercise"),
        _make_rc(4, "respiratory exchange ratio"),
    ]
    reranker = Reranker(backend="bedrock")

    with patch.object(
        reranker,
        "_rerank_call",
        side_effect=lambda q, docs: _keyword_scorer(q, docs),
    ):
        results = reranker.rerank(query, candidates, top_k=3)

    assert len(results) <= 3
    # chunk_1 은 쿼리 단어 3개 포함 → 최상위여야 함
    assert (
        results[0].chunk.id == "chunk_1"
    ), f"chunk_1 이 최상위여야 하지만 {results[0].chunk.id} 임"
    # 반환된 결과는 rerank_score 내림차순
    rscores = [rc.rerank_score for rc in results]
    assert rscores == sorted(rscores, reverse=True)
