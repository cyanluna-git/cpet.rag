"""eval.ragas.runner — 결정적 Retrieval 지표 + RAGAS LLM judge 평가 루프.

결정적 지표(hit@k, MRR)는 LLM 없이 항상 동작.
RAGAS LLM judge(Faithfulness, ContextPrecision/Recall)는 run_ragas=True 또는
ragas_scorer 주입 시에만 실행되며, 키 없거나 ragas 미설치 시 None 반환 + 경고.

사용 예:
    from eval.ragas import evaluate, EvalReport

    report = evaluate(pipeline, qa_items, k=8)
    print(report.to_markdown())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from core.log import get_logger
from core.metadata import normalize_doi
from core.models import QueryRequest

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """RAGAS 평가 루프 결과 컨테이너.

    Attributes:
        per_item: 항목별 결과 dict 목록.
        hit_at_k: 전체 hit@k 평균 (0.0–1.0).
        mrr: 전체 MRR 평균 (0.0–1.0).
        faithfulness: RAGAS Faithfulness 평균 (None = 평가 안 함).
        context_precision: RAGAS ContextPrecision 평균 (None = 평가 안 함).
        context_recall: RAGAS ContextRecall 평균 (None = 평가 안 함).
        n: 시도된 총 항목 수.
    """

    per_item: list[dict[str, Any]] = field(default_factory=list)
    hit_at_k: float = 0.0
    mrr: float = 0.0
    faithfulness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    n: int = 0

    def to_markdown(self) -> str:
        """사람이 읽기 편한 Markdown 요약을 반환한다."""
        lines = [
            "## RAGAS 평가 결과",
            "",
            "| 지표 | 값 |",
            "|------|----|",
            f"| n | {self.n} |",
            f"| hit@k | {self.hit_at_k:.4f} |",
            f"| MRR | {self.mrr:.4f} |",
            f"| Faithfulness | {self.faithfulness if self.faithfulness is not None else 'N/A'} |",
            f"| Context Precision | {self.context_precision if self.context_precision is not None else 'N/A'} |",
            f"| Context Recall | {self.context_recall if self.context_recall is not None else 'N/A'} |",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 결정적 지표
# ---------------------------------------------------------------------------


def hit_at_k(
    retrieved_chunks: list[Any],
    relevant_dois: list[str],
    k: int,
) -> bool:
    """retrieved 상위 k 청크의 DOI 집합에 relevant_dois 중 하나라도 포함되면 True.

    Args:
        retrieved_chunks: RetrievedChunk 목록 (순서 보존).
        relevant_dois: 정답 DOI 목록 (정규화 전).
        k: 상위 k 개만 평가.

    Returns:
        hit 여부.
    """
    if not relevant_dois:
        return False

    norm_relevant = {normalize_doi(d) for d in relevant_dois} - {None}
    if not norm_relevant:
        return False

    norm_retrieved = {normalize_doi(rc.chunk.doi) for rc in retrieved_chunks[:k]} - {None}

    return bool(norm_retrieved & norm_relevant)


def mrr(
    retrieved_chunks: list[Any],
    relevant_dois: list[str],
) -> float:
    """첫 번째 relevant 청크의 역수 순위(MRR)를 반환한다.

    Args:
        retrieved_chunks: RetrievedChunk 목록 (순서 보존).
        relevant_dois: 정답 DOI 목록 (정규화 전).

    Returns:
        MRR 값 (0.0–1.0). relevant 청크가 없으면 0.0.
    """
    if not relevant_dois:
        return 0.0

    norm_relevant = {normalize_doi(d) for d in relevant_dois} - {None}
    if not norm_relevant:
        return 0.0

    for rank, rc in enumerate(retrieved_chunks, start=1):
        norm = normalize_doi(rc.chunk.doi)
        if norm and norm in norm_relevant:
            return 1.0 / rank

    return 0.0


# ---------------------------------------------------------------------------
# 평가 루프
# ---------------------------------------------------------------------------

# ragas_scorer 계약: list[dict] → dict[str, float]
# 각 dict 는 {"question", "answer", "contexts": list[str], "ground_truth"} 형식
# 반환 dict 의 필수 키: "faithfulness", "context_precision", "context_recall"
RagasScorerFn = Callable[[list[dict[str, Any]]], dict[str, float]]


def evaluate(
    pipeline: Any,
    qa_items: list[dict[str, Any]],
    *,
    k: int = 8,
    run_ragas: bool = False,
    ragas_scorer: "RagasScorerFn | None" = None,
) -> EvalReport:
    """qa_items 으로 RAG 파이프라인을 평가한다.

    결정적 지표(hit@k, MRR)는 항상 계산한다.
    RAGAS LLM judge 는 run_ragas=True 또는 ragas_scorer 주입 시에만 실행된다.

    Args:
        pipeline: ``answer(QueryRequest) -> QueryResponse`` 메서드를 가진 파이프라인.
        qa_items: ``load_qa_set()``의 반환값 목록.
        k: hit@k 에서 사용할 k 값. pipeline.answer 의 top_k 도 이 값으로 설정.
        run_ragas: True 면 내장 lazy-import RAGAS 경로를 시도.
        ragas_scorer: 외부에서 주입하는 RAGAS 스코어 함수. 주입 시 run_ragas 보다 우선.

    Returns:
        EvalReport 집계 결과.
    """
    per_item: list[dict[str, Any]] = []
    ragas_rows: list[dict[str, Any]] = []

    for item_idx, item in enumerate(qa_items):
        item_id = item.get("id", str(item_idx))
        question = item.get("question_ko", "")
        relevant_dois: list[str] = item.get("relevant_dois", [])
        answer_gold: str = item.get("answer_gold", "")

        result: dict[str, Any] = {
            "id": item_id,
            "question": question,
            "hit": False,
            "mrr": 0.0,
            "error": None,
        }

        try:
            req = QueryRequest(query=question, top_k=k)
            resp = pipeline.answer(req)

            retrieved = resp.retrieved  # list[RetrievedChunk]

            result["hit"] = hit_at_k(retrieved, relevant_dois, k)
            result["mrr"] = mrr(retrieved, relevant_dois)

            # RAGAS row 준비 (run_ragas 또는 scorer 주입 시)
            if run_ragas or ragas_scorer is not None:
                contexts = [rc.chunk.text for rc in retrieved if rc.chunk.text]
                answer_text = resp.answer_en or resp.answer or ""
                ragas_rows.append(
                    {
                        "question": question,
                        "answer": answer_text,
                        "contexts": contexts,
                        "ground_truth": answer_gold,
                    }
                )

        except Exception as exc:
            logger.warning("항목 %s 평가 실패: %s", item_id, exc)
            result["error"] = str(exc)

        per_item.append(result)

    # ------------------------------------------------------------------
    # 집계 — n = 시도된 총 항목 수 (실패 포함)
    # ------------------------------------------------------------------
    n = len(per_item)
    agg_hit = sum(1 for r in per_item if r["hit"]) / n if n > 0 else 0.0
    agg_mrr = sum(r["mrr"] for r in per_item) / n if n > 0 else 0.0

    # ------------------------------------------------------------------
    # RAGAS LLM judge
    # ------------------------------------------------------------------
    faithfulness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    if ragas_rows:
        scores = _run_ragas_scorer(ragas_rows, ragas_scorer, run_ragas=run_ragas)
        if scores is not None:
            faithfulness = scores.get("faithfulness")
            context_precision = scores.get("context_precision")
            context_recall = scores.get("context_recall")

    return EvalReport(
        per_item=per_item,
        hit_at_k=agg_hit,
        mrr=agg_mrr,
        faithfulness=faithfulness,
        context_precision=context_precision,
        context_recall=context_recall,
        n=n,
    )


def _run_ragas_scorer(
    rows: list[dict[str, Any]],
    ragas_scorer: "RagasScorerFn | None",
    *,
    run_ragas: bool,
) -> "dict[str, float] | None":
    """RAGAS 스코어를 계산한다.

    ragas_scorer 가 주입되어 있으면 그것을 사용.
    없고 run_ragas=True 면 lazy-import 로 내장 RAGAS 를 시도.
    실패 시 None 반환 + 경고 로그.

    Args:
        rows: RAGAS 입력 row 목록.
        ragas_scorer: 주입된 스코어 함수 (있으면 우선 사용).
        run_ragas: True 면 내장 RAGAS 경로를 시도.

    Returns:
        {"faithfulness", "context_precision", "context_recall"} dict 또는 None.
    """
    if ragas_scorer is not None:
        try:
            return ragas_scorer(rows)
        except Exception as exc:
            logger.warning("ragas_scorer 실행 실패: %s", exc)
            return None

    if not run_ragas:
        return None

    # Lazy import — ragas 는 eval extra (무거운 LLM 의존성)
    try:
        from ragas import evaluate as ragas_evaluate  # noqa: PLC0415
        from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "ragas 패키지가 설치되지 않았습니다. "
            "`pip install 'cpet-rag[eval]'` 또는 `uv sync --extra eval` 로 설치하세요."
        )
        return None

    try:
        # ragas ≥ 0.2: EvaluationDataset + user_input/reference/retrieved_contexts/response
        try:
            from ragas import EvaluationDataset  # noqa: PLC0415
            from ragas.dataset_schema import SingleTurnSample  # noqa: PLC0415

            samples = [
                SingleTurnSample(
                    user_input=row["question"],
                    response=row["answer"],
                    retrieved_contexts=row["contexts"],
                    reference=row["ground_truth"],
                )
                for row in rows
            ]
            dataset = EvaluationDataset(samples=samples)
        except ImportError:
            # ragas < 0.2 폴백
            from datasets import Dataset  # noqa: PLC0415

            dataset = Dataset.from_list(rows)

        metrics = [Faithfulness(), ContextPrecision(), ContextRecall()]
        result = ragas_evaluate(dataset, metrics=metrics)

        # ragas 결과 dict 추출 (버전별 API 차이 대응)
        scores: dict[str, float] = {}
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            for col in ("faithfulness", "context_precision", "context_recall"):
                if col in df.columns:
                    scores[col] = float(df[col].mean())
        elif isinstance(result, dict):
            scores = {k: float(v) for k, v in result.items()}

        return scores if scores else None

    except Exception as exc:
        logger.warning("RAGAS 평가 실패 (LLM 키 미설정이거나 API 오류일 수 있음): %s", exc)
        return None


# ---------------------------------------------------------------------------
# 출력 헬퍼
# ---------------------------------------------------------------------------


def print_report(report: EvalReport) -> None:
    """EvalReport 를 사람이 읽기 편한 형식으로 출력한다."""
    print(report.to_markdown())
