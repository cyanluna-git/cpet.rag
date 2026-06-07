"""core.metadata.loader — corpus_index.csv → Paper 모델 로딩 유틸리티."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

from core.log import get_logger
from core.models.paper import Paper

logger = get_logger(__name__)

# DOI 정규화 정규식 — add_paper.py의 norm() 로직을 재사용
_DOI_RE = re.compile(r"10\.\d{4,9}/\S+")

# Optional 필드 목록 — 빈 문자열을 None 으로 치환
_OPTIONAL_STR_FIELDS = frozenset(
    {"doi", "first_author", "journal", "file", "oa_status", "added_by", "added_at"}
)


def normalize_doi(raw: Optional[str]) -> Optional[str]:
    """DOI 문자열을 정규화한다.

    - URL 접두어(https://doi.org/ 등) 제거
    - 소문자로 변환
    - 끝 구두점 제거
    - DOI 패턴이 없으면 None 반환
    """
    if not raw:
        return None
    m = _DOI_RE.search(raw.strip())
    if not m:
        return None
    return m.group(0).lower().rstrip(".")


def load_corpus_index(path: str | Path = "data/corpus_index.csv") -> list[Paper]:
    """corpus_index.csv를 읽어 Paper 목록을 반환한다.

    Args:
        path: CSV 파일 경로 (기본값: data/corpus_index.csv).

    Returns:
        Paper 인스턴스 목록.

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"corpus_index.csv 를 찾을 수 없습니다: {csv_path.resolve()}")

    papers: list[Paper] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader, start=2):  # 2 = 첫 데이터 행 (헤더 제외)
            try:
                # Optional 필드: 빈 문자열 → None
                cleaned: dict = {}
                for key, val in row.items():
                    if key in _OPTIONAL_STR_FIELDS:
                        cleaned[key] = val.strip() or None
                    else:
                        # 필수 str 필드(title, source) 는 빈 문자열 그대로 유지
                        cleaned[key] = val.strip() if val else ""

                # year 공변환 — 빈 문자열 또는 비정수 값 → None
                raw_year = cleaned.get("year")
                if raw_year:
                    try:
                        cleaned["year"] = int(raw_year)
                    except (ValueError, TypeError):
                        logger.warning("행 %d: year 변환 실패 ('%s') → None", i, raw_year)
                        cleaned["year"] = None
                else:
                    cleaned["year"] = None

                papers.append(Paper(**cleaned))
            except Exception as exc:
                logger.warning("행 %d 파싱 실패: %s — 건너뜀", i, exc)

    logger.info("corpus_index 로드 완료: %d 편", len(papers))
    return papers


def index_by_doi(papers: list[Paper]) -> dict[str, Paper]:
    """Paper 목록을 정규화된 DOI 로 색인한다.

    DOI 가 없는 Paper 는 건너뛴다.
    동일 DOI 가 중복될 경우 마지막 항목이 남는다(경고 로깅).

    Args:
        papers: Paper 인스턴스 목록.

    Returns:
        정규화된 DOI → Paper 딕셔너리.
    """
    index: dict[str, Paper] = {}
    for paper in papers:
        doi = normalize_doi(paper.doi)
        if not doi:
            continue
        if doi in index:
            logger.warning("DOI 중복: %s — 기존 항목을 덮어씁니다", doi)
        index[doi] = paper
    logger.debug("DOI 색인 완료: %d 편", len(index))
    return index
