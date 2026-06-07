"""eval.qa_set.loader — RAGAS 평가셋 로더 및 포맷 변환 유틸리티.

사용 예:
    from eval.qa_set.loader import load_qa_set, to_ragas_dataset

    items = load_qa_set()
    ragas_rows = to_ragas_dataset(items)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 선택적 Pydantic 사용 — 없어도 dict 폴백으로 동작
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, field_validator

    class EvalItem(BaseModel):
        """단일 평가 레코드를 나타내는 모델."""

        id: str
        question_ko: str
        question_en: str
        answer_gold: str
        relevant_dois: list[str]
        difficulty: str
        tags: list[str]
        source: str

        @field_validator("difficulty")
        @classmethod
        def _validate_difficulty(cls, v: str) -> str:
            allowed = {"easy", "med", "hard"}
            if v not in allowed:
                raise ValueError(f"difficulty must be one of {allowed}, got {v!r}")
            return v

    _USE_PYDANTIC = True

except ImportError:  # pydantic 미설치 환경 대비 (unlikely — 기본 의존성)
    EvalItem = None  # type: ignore[assignment,misc]
    _USE_PYDANTIC = False


# 필수 필드 집합 — pydantic 없이도 검증
_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "question_ko",
        "question_en",
        "answer_gold",
        "relevant_dois",
        "difficulty",
        "tags",
        "source",
    }
)

_DEFAULT_PATH = Path(__file__).parent / "qa_set.jsonl"


def load_qa_set(path: str | Path | None = None) -> list[dict[str, Any]]:
    """qa_set.jsonl을 읽어 평가 레코드 목록을 반환한다.

    Args:
        path: JSONL 파일 경로. None 이면 이 모듈과 같은 디렉터리의
              ``qa_set.jsonl``을 사용한다.

    Returns:
        dict 목록 (각 dict 는 EvalItem 의 필드를 포함).

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때.
        ValueError: 필수 필드가 누락된 레코드가 있을 때.
    """
    jsonl_path = Path(path) if path is not None else _DEFAULT_PATH
    if not jsonl_path.exists():
        raise FileNotFoundError(f"qa_set.jsonl 을 찾을 수 없습니다: {jsonl_path.resolve()}")

    items: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"행 {lineno}: JSON 파싱 실패 — {exc}") from exc

            missing = _REQUIRED_FIELDS - record.keys()
            if missing:
                raise ValueError(
                    f"행 {lineno} (id={record.get('id', '?')}): " f"필수 필드 누락 — {sorted(missing)}"
                )

            if _USE_PYDANTIC:
                # pydantic 검증 (difficulty enum 등)
                validated = EvalItem(**record)
                items.append(validated.model_dump())
            else:
                items.append(record)

    return items


def to_ragas_dataset(
    items: list[dict[str, Any]],
    use_korean_question: bool = True,
) -> list[dict[str, Any]]:
    """평가 레코드를 RAGAS 평가 데이터셋 형식으로 변환한다.

    반환 형식은 RAGAS < 0.2 기준 컬럼명을 사용한다:
        question, ground_truth, contexts (빈 리스트), answer (None).

    RAGAS ≥ 0.2 는 컬럼명이 user_input / reference / retrieved_contexts / response 로
    변경됐다 — #3129 구현 시 버전에 맞게 매핑을 조정할 것.

    Args:
        items: load_qa_set() 의 반환값.
        use_korean_question: True 이면 question_ko, False 이면 question_en 을 사용.
            RAG 시스템이 한국어 입력을 처리하는 경우 True(기본값).

    Returns:
        RAGAS dict 목록. contexts 와 answer 는 평가 실행 시 채워야 한다.
    """
    question_field = "question_ko" if use_korean_question else "question_en"
    rows: list[dict[str, Any]] = []
    for item in items:
        rows.append(
            {
                "question": item[question_field],
                "ground_truth": item["answer_gold"],
                "contexts": [],  # 평가 실행 시 retriever가 채운다
                "answer": None,  # 평가 실행 시 LLM이 채운다
                # 추가 메타 — RAGAS 평가 후 분석 편의용
                "_id": item["id"],
                "_relevant_dois": item["relevant_dois"],
                "_difficulty": item["difficulty"],
                "_tags": item["tags"],
            }
        )
    return rows
