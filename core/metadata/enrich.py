"""core.metadata.enrich — OpenAlex / Crossref 메타 보강 유틸리티.

모든 네트워크 호출은 함수 내부에서 발생한다 (모듈 임포트 시 네트워크 없음).
"""

from __future__ import annotations

from typing import Any

import httpx

from core.config.settings import Settings
from core.log import get_logger
from core.models.paper import Paper

logger = get_logger(__name__)

_TIMEOUT = 15.0
_UA = "cpet.rag/1.0 (mailto:{email})"


def _get_settings() -> Settings:
    """설정을 지연 로딩한다."""
    return Settings()


# ---------------------------------------------------------------------------
# Crossref
# ---------------------------------------------------------------------------


def crossref_meta(doi: str) -> dict[str, Any]:
    """Crossref API 에서 논문 메타데이터를 가져온다.

    Args:
        doi: 정규화된 DOI 문자열.

    Returns:
        dict with keys: title, authors (list[str]), journal, year,
        issn (list[str]), volume, issue, page. 실패 시 빈 dict.
    """
    settings = _get_settings()
    mailto = settings.crossref_mailto
    url = f"https://api.crossref.org/works/{doi}"
    params = {"mailto": mailto}
    headers = {"User-Agent": _UA.format(email=mailto)}

    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=_TIMEOUT)
        resp.raise_for_status()
        msg: dict = resp.json().get("message", {})
    except Exception as exc:
        logger.warning("Crossref 조회 실패 (doi=%s): %s", doi, exc)
        return {}

    # 저자 목록 구성
    raw_authors: list[dict] = msg.get("author") or []
    authors: list[str] = []
    for a in raw_authors:
        family = a.get("family", "")
        given = a.get("given", "")
        name = f"{family}, {given}".strip(", ") if family else a.get("name", "")
        if name:
            authors.append(name)

    # 출판 연도
    date_parts = (msg.get("issued") or {}).get("date-parts") or [[None]]
    year_raw = (date_parts[0] or [None])[0]
    year: int | None = int(year_raw) if year_raw else None

    return {
        "title": (msg.get("title") or [""])[0],
        "authors": authors,
        "journal": (msg.get("container-title") or [""])[0],
        "year": year,
        "issn": msg.get("ISSN") or [],
        "volume": msg.get("volume"),
        "issue": msg.get("issue"),
        "page": msg.get("page"),
    }


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """abstract_inverted_index (word → [positions]) 에서 원문 초록을 복원한다."""
    if not inverted_index:
        return None
    try:
        pos_word: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                pos_word.append((pos, word))
        pos_word.sort()
        return " ".join(w for _, w in pos_word)
    except Exception as exc:
        logger.warning("abstract_inverted_index 복원 실패: %s", exc)
        return None


def openalex_meta(doi: str) -> dict[str, Any]:
    """OpenAlex API 에서 논문 메타데이터를 가져온다.

    Args:
        doi: 정규화된 DOI 문자열.

    Returns:
        dict with keys: openalex_id (str), abstract (str|None),
        authors (list[str]), referenced_works (list[str]). 실패 시 빈 dict.
    """
    settings = _get_settings()
    mailto = settings.crossref_mailto  # OpenAlex 도 mailto 파라미터 지원
    url = f"https://api.openalex.org/works/doi:{doi}"
    params = {"mailto": mailto}
    headers = {"User-Agent": _UA.format(email=mailto)}

    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=_TIMEOUT)
        resp.raise_for_status()
        data: dict = resp.json()
    except Exception as exc:
        logger.warning("OpenAlex 조회 실패 (doi=%s): %s", doi, exc)
        return {}

    # 저자 목록
    raw_authors: list[dict] = data.get("authorships") or []
    authors: list[str] = []
    for a in raw_authors:
        name = (a.get("author") or {}).get("display_name", "")
        if name:
            authors.append(name)

    abstract = _reconstruct_abstract(data.get("abstract_inverted_index"))

    return {
        "openalex_id": data.get("id"),
        "abstract": abstract,
        "authors": authors,
        "referenced_works": data.get("referenced_works") or [],
    }


# ---------------------------------------------------------------------------
# enrich_paper
# ---------------------------------------------------------------------------


def enrich_paper(paper: Paper) -> Paper:
    """Crossref + OpenAlex 로 Paper 를 보강한다.

    - 이미 값이 있는 필드는 덮어쓰지 않는다 (abstract, authors, openalex_id).
    - title / journal / year 는 현재 비어 있을 때만 채운다.
    - DOI 가 없으면 No-op 으로 원본을 반환한다.
    - 네트워크 실패 시 원본을 반환한다 (예외 미전파).

    Args:
        paper: 보강할 Paper 인스턴스.

    Returns:
        보강된 NEW Paper 인스턴스 (원본 불변).
    """
    if not paper.doi:
        return paper

    updates: dict[str, Any] = {}

    # --- Crossref ---
    cr = crossref_meta(paper.doi)
    if cr:
        if not paper.title:
            title = cr.get("title", "")
            if title:
                updates["title"] = title
        if not paper.journal:
            journal = cr.get("journal", "")
            if journal:
                updates["journal"] = journal
        if paper.year is None:
            year = cr.get("year")
            if year:
                updates["year"] = year
        if not paper.authors:
            authors = cr.get("authors") or []
            if authors:
                updates["authors"] = authors

    # --- OpenAlex ---
    oa = openalex_meta(paper.doi)
    if oa:
        if not paper.openalex_id:
            oid = oa.get("openalex_id")
            if oid:
                updates["openalex_id"] = oid
        if not paper.abstract:
            abstract = oa.get("abstract")
            if abstract:
                updates["abstract"] = abstract
        if not paper.authors and not updates.get("authors"):
            authors = oa.get("authors") or []
            if authors:
                updates["authors"] = authors

    if not updates:
        return paper

    return paper.model_copy(update=updates)
