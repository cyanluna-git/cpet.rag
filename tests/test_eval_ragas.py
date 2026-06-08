"""tests.test_eval_ragas — RAGAS 평가 루프 유닛 테스트.

ragas / LLM 호출 없음. MockPipeline 으로 결정적 QueryResponse 를 반환.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.models import QueryRequest, QueryResponse
from core.models.chunk import Chunk, RetrievedChunk
from eval.ragas import EvalReport, evaluate, hit_at_k, mrr

# ---------------------------------------------------------------------------
# 헬퍼 — 테스트 픽스처 빌더
# ---------------------------------------------------------------------------


def _make_chunk(doi: str | None, text: str = "sample text") -> Chunk:
    """Chunk 인스턴스를 생성한다."""
    return Chunk(
        id=f"{doi or 'none'}::0",
        doi=doi,
        text=text,
        ctx_text=f"[DOI:{doi}] {text}",
        chunk_index=0,
    )


def _make_rc(doi: str | None, score: float = 1.0, text: str = "sample text") -> RetrievedChunk:
    """RetrievedChunk 인스턴스를 생성한다."""
    return RetrievedChunk(chunk=_make_chunk(doi, text), score=score)


def _make_response(dois: list[str | None]) -> QueryResponse:
    """주어진 DOI 순서대로 retrieved 를 가진 QueryResponse 를 반환한다."""
    retrieved = [_make_rc(d) for d in dois]
    return QueryResponse(
        answer="테스트 답변",
        answer_en="test answer",
        citations=[],
        retrieved=retrieved,
    )


class MockPipeline:
    """고정된 QueryResponse 를 반환하는 Mock 파이프라인."""

    def __init__(self, responses: list[QueryResponse]) -> None:
        self._responses = list(responses)
        self._call_idx = 0

    def answer(self, req: QueryRequest) -> QueryResponse:
        resp = self._responses[self._call_idx]
        self._call_idx += 1
        return resp


class ErrorPipeline:
    """모든 호출에서 예외를 발생시키는 파이프라인."""

    def answer(self, req: QueryRequest) -> QueryResponse:
        raise RuntimeError("pipeline error")


# ---------------------------------------------------------------------------
# hit_at_k 단위 테스트
# ---------------------------------------------------------------------------


class TestHitAtK:
    def test_relevant_doi_present_returns_true(self) -> None:
        chunks = [_make_rc("10.1000/abc"), _make_rc("10.1000/xyz")]
        assert hit_at_k(chunks, ["10.1000/abc"], k=5) is True

    def test_relevant_doi_absent_returns_false(self) -> None:
        chunks = [_make_rc("10.1000/abc"), _make_rc("10.1000/xyz")]
        assert hit_at_k(chunks, ["10.1000/zzz"], k=5) is False

    def test_k_boundary_included(self) -> None:
        """k 번째 청크는 평가에 포함된다."""
        chunks = [_make_rc("10.1000/a"), _make_rc("10.1000/b"), _make_rc("10.1000/c")]
        assert hit_at_k(chunks, ["10.1000/c"], k=3) is True

    def test_k_boundary_excluded(self) -> None:
        """k+1 번째 청크는 평가에서 제외된다."""
        chunks = [_make_rc("10.1000/a"), _make_rc("10.1000/b"), _make_rc("10.1000/c")]
        assert hit_at_k(chunks, ["10.1000/c"], k=2) is False

    def test_doi_normalization(self) -> None:
        """URL 형식 DOI 도 정규화 후 비교한다."""
        chunks = [_make_rc("10.1000/abc")]
        assert hit_at_k(chunks, ["https://doi.org/10.1000/abc"], k=5) is True

    def test_empty_relevant_dois_returns_false(self) -> None:
        chunks = [_make_rc("10.1000/abc")]
        assert hit_at_k(chunks, [], k=5) is False

    def test_empty_retrieved_returns_false(self) -> None:
        assert hit_at_k([], ["10.1000/abc"], k=5) is False

    def test_none_doi_in_retrieved_skipped(self) -> None:
        chunks = [_make_rc(None), _make_rc("10.1000/abc")]
        assert hit_at_k(chunks, ["10.1000/abc"], k=5) is True

    def test_k_zero_returns_false(self) -> None:
        chunks = [_make_rc("10.1000/abc")]
        assert hit_at_k(chunks, ["10.1000/abc"], k=0) is False


# ---------------------------------------------------------------------------
# mrr 단위 테스트
# ---------------------------------------------------------------------------


class TestMrr:
    def test_rank_1_returns_1(self) -> None:
        chunks = [_make_rc("10.1000/abc")]
        assert mrr(chunks, ["10.1000/abc"]) == pytest.approx(1.0)

    def test_rank_2_returns_half(self) -> None:
        chunks = [_make_rc("10.1000/x"), _make_rc("10.1000/abc")]
        assert mrr(chunks, ["10.1000/abc"]) == pytest.approx(0.5)

    def test_rank_3_returns_third(self) -> None:
        chunks = [_make_rc("10.1000/x"), _make_rc("10.1000/y"), _make_rc("10.1000/abc")]
        assert mrr(chunks, ["10.1000/abc"]) == pytest.approx(1 / 3, abs=1e-3)

    def test_not_found_returns_zero(self) -> None:
        chunks = [_make_rc("10.1000/x"), _make_rc("10.1000/y")]
        assert mrr(chunks, ["10.1000/zzz"]) == pytest.approx(0.0)

    def test_empty_relevant_returns_zero(self) -> None:
        chunks = [_make_rc("10.1000/abc")]
        assert mrr(chunks, []) == pytest.approx(0.0)

    def test_empty_retrieved_returns_zero(self) -> None:
        assert mrr([], ["10.1000/abc"]) == pytest.approx(0.0)

    def test_doi_normalization(self) -> None:
        chunks = [_make_rc("10.1000/abc")]
        assert mrr(chunks, ["https://doi.org/10.1000/abc"]) == pytest.approx(1.0)

    def test_first_relevant_used(self) -> None:
        """multiple relevant_dois — 첫 번째 매칭 rank 를 사용한다."""
        chunks = [
            _make_rc("10.1000/other"),
            _make_rc("10.1000/b"),
            _make_rc("10.1000/a"),
        ]
        # "a" 는 rank 3, "b" 는 rank 2 → 먼저 나오는 "b" 기준 0.5
        result = mrr(chunks, ["10.1000/a", "10.1000/b"])
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# evaluate 통합 테스트
# ---------------------------------------------------------------------------


def _make_qa_items(
    dois_list: list[list[str | None]],
) -> list[dict[str, Any]]:
    """각 항목에 대한 QA item 목록을 생성한다."""
    items = []
    for i, dois in enumerate(dois_list):
        items.append(
            {
                "id": f"q{i}",
                "question_ko": f"질문 {i}",
                "question_en": f"question {i}",
                "answer_gold": f"gold answer {i}",
                "relevant_dois": [d for d in dois if d is not None],
                "difficulty": "med",
                "tags": [],
                "source": "test",
            }
        )
    return items


class TestEvaluate:
    def test_basic_hit_and_mrr(self) -> None:
        """hit 있는 경우 hit_at_k=1.0, mrr>0."""
        qa_items = _make_qa_items([["10.1000/abc"]])
        pipeline = MockPipeline([_make_response(["10.1000/abc"])])
        report = evaluate(pipeline, qa_items, k=5)

        assert report.n == 1
        assert report.hit_at_k == pytest.approx(1.0)
        assert report.mrr == pytest.approx(1.0)
        assert report.faithfulness is None
        assert report.context_precision is None
        assert report.context_recall is None

    def test_no_hit(self) -> None:
        """relevant doi 가 없으면 hit=False, mrr=0."""
        qa_items = _make_qa_items([["10.1000/zzz"]])
        pipeline = MockPipeline([_make_response(["10.1000/abc"])])
        report = evaluate(pipeline, qa_items, k=5)

        assert report.n == 1
        assert report.hit_at_k == pytest.approx(0.0)
        assert report.mrr == pytest.approx(0.0)

    def test_aggregate_multiple_items(self) -> None:
        """2/3 항목 hit → hit_at_k ≈ 0.667."""
        qa_items = _make_qa_items(
            [
                ["10.1000/a"],  # hit
                ["10.1000/b"],  # hit
                ["10.1000/c"],  # miss
            ]
        )
        responses = [
            _make_response(["10.1000/a"]),
            _make_response(["10.1000/b"]),
            _make_response(["10.1000/x"]),
        ]
        pipeline = MockPipeline(responses)
        report = evaluate(pipeline, qa_items, k=5)

        assert report.n == 3
        assert report.hit_at_k == pytest.approx(2 / 3, abs=1e-6)

    def test_mrr_aggregate(self) -> None:
        """rank-1 + rank-2 → MRR = (1.0 + 0.5) / 2 = 0.75."""
        qa_items = _make_qa_items(
            [
                ["10.1000/a"],
                ["10.1000/b"],
            ]
        )
        responses = [
            _make_response(["10.1000/a"]),  # rank 1 → 1.0
            _make_response(["10.1000/x", "10.1000/b"]),  # rank 2 → 0.5
        ]
        pipeline = MockPipeline(responses)
        report = evaluate(pipeline, qa_items, k=5)

        assert report.mrr == pytest.approx(0.75)

    def test_ragas_none_when_run_ragas_false(self) -> None:
        """run_ragas=False(기본) → ragas 지표 모두 None."""
        qa_items = _make_qa_items([["10.1000/abc"]])
        pipeline = MockPipeline([_make_response(["10.1000/abc"])])
        report = evaluate(pipeline, qa_items, k=5, run_ragas=False)

        assert report.faithfulness is None
        assert report.context_precision is None
        assert report.context_recall is None

    def test_ragas_scorer_injection(self) -> None:
        """ragas_scorer 주입 시 faithfulness 등이 채워진다."""
        qa_items = _make_qa_items([["10.1000/abc"]])
        pipeline = MockPipeline([_make_response(["10.1000/abc"])])

        def mock_scorer(rows: list[dict[str, Any]]) -> dict[str, float]:
            assert len(rows) == 1
            return {"faithfulness": 0.9, "context_precision": 0.8, "context_recall": 0.7}

        report = evaluate(pipeline, qa_items, k=5, ragas_scorer=mock_scorer)

        assert report.faithfulness == pytest.approx(0.9)
        assert report.context_precision == pytest.approx(0.8)
        assert report.context_recall == pytest.approx(0.7)

    def test_ragas_scorer_receives_contexts_and_answer(self) -> None:
        """ragas_scorer 에 전달되는 row 에 contexts/answer 가 올바르게 담긴다."""
        qa_items = _make_qa_items([["10.1000/abc"]])
        resp = QueryResponse(
            answer="한국어 답변",
            answer_en="english answer",
            citations=[],
            retrieved=[_make_rc("10.1000/abc", text="relevant passage")],
        )
        pipeline = MockPipeline([resp])

        captured: list[list[dict[str, Any]]] = []

        def capturing_scorer(rows: list[dict[str, Any]]) -> dict[str, float]:
            captured.append(rows)
            return {"faithfulness": 1.0, "context_precision": 1.0, "context_recall": 1.0}

        evaluate(pipeline, qa_items, k=5, ragas_scorer=capturing_scorer)

        assert len(captured) == 1
        row = captured[0][0]
        assert row["answer"] == "english answer"  # answer_en 우선
        assert row["contexts"] == ["relevant passage"]
        assert row["ground_truth"] == "gold answer 0"

    def test_pipeline_exception_records_failure_continues(self) -> None:
        """pipeline 예외 → 해당 항목 hit=False로 기록, 나머지 계속."""
        qa_items = _make_qa_items(
            [
                ["10.1000/a"],  # 예외 항목
                ["10.1000/b"],  # 정상 항목
            ]
        )

        call_count = 0

        class PartialPipeline:
            def answer(self, req: QueryRequest) -> QueryResponse:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("pipeline error")
                return _make_response(["10.1000/b"])

        report = evaluate(PartialPipeline(), qa_items, k=5)

        assert report.n == 2
        assert report.per_item[0]["hit"] is False
        assert report.per_item[0]["error"] is not None
        assert report.per_item[1]["hit"] is True
        # 집계 = (0 + 1) / 2
        assert report.hit_at_k == pytest.approx(0.5)

    def test_all_fail_returns_zero_metrics(self) -> None:
        """모든 항목 실패 → hit_at_k=0, mrr=0, n=총 항목 수."""
        qa_items = _make_qa_items([["10.1000/a"], ["10.1000/b"]])
        report = evaluate(ErrorPipeline(), qa_items, k=5)

        assert report.n == 2
        assert report.hit_at_k == pytest.approx(0.0)
        assert report.mrr == pytest.approx(0.0)

    def test_per_item_length_matches_qa_items(self) -> None:
        """per_item 의 길이는 qa_items 길이와 동일해야 한다."""
        qa_items = _make_qa_items([["10.1000/a"], ["10.1000/b"], ["10.1000/c"]])
        responses = [
            _make_response(["10.1000/a"]),
            _make_response(["10.1000/b"]),
            _make_response(["10.1000/c"]),
        ]
        pipeline = MockPipeline(responses)
        report = evaluate(pipeline, qa_items, k=5)

        assert len(report.per_item) == 3

    def test_k_respected_in_hit(self) -> None:
        """k=1 이면 첫 번째 청크만 평가한다."""
        qa_items = _make_qa_items([["10.1000/b"]])
        # "b" 는 rank 2 에 있어서 k=1 이면 miss
        pipeline = MockPipeline([_make_response(["10.1000/a", "10.1000/b"])])
        report = evaluate(pipeline, qa_items, k=1)

        assert report.hit_at_k == pytest.approx(0.0)

    def test_eval_report_is_dataclass(self) -> None:
        """EvalReport 가 올바른 타입/필드를 가진다."""
        import dataclasses

        assert dataclasses.is_dataclass(EvalReport)
        field_names = {f.name for f in dataclasses.fields(EvalReport)}
        expected = {
            "per_item",
            "hit_at_k",
            "mrr",
            "faithfulness",
            "context_precision",
            "context_recall",
            "n",
        }
        assert expected <= field_names


# ---------------------------------------------------------------------------
# EvalReport 출력 테스트
# ---------------------------------------------------------------------------


class TestEvalReportOutput:
    def test_to_markdown_contains_key_metrics(self) -> None:
        report = EvalReport(hit_at_k=0.75, mrr=0.5, n=4)
        md = report.to_markdown()
        assert "0.7500" in md
        assert "0.5000" in md
        assert "4" in md
        assert "N/A" in md  # ragas 지표 없음

    def test_to_markdown_with_ragas_scores(self) -> None:
        report = EvalReport(
            hit_at_k=1.0,
            mrr=1.0,
            faithfulness=0.9,
            context_precision=0.8,
            context_recall=0.7,
            n=1,
        )
        md = report.to_markdown()
        assert "0.9" in md
        assert "0.8" in md
        assert "0.7" in md
