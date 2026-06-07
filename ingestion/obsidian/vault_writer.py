"""ingestion.obsidian.vault_writer — Obsidian vault 생성기.

코퍼스 Paper 목록으로부터 Obsidian 호환 마크다운 노트 vault 를 디렉터리에 기록한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from core.log import get_logger
from ingestion.obsidian.note_builder import _bare_openalex_id, note_filename, paper_to_note

if TYPE_CHECKING:
    from core.models.paper import Paper

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# build_citation_links
# ---------------------------------------------------------------------------


def build_citation_links(
    paper: "Paper",
    corpus_by_openalex: dict[str, "Paper"],
    *,
    fetch: bool = True,
) -> list["Paper"]:
    """paper 가 인용하는 논문 중 코퍼스에 존재하는 것만 반환한다.

    ``referenced_works`` 는 OpenAlex API 에서 가져온 URL 형식 ID 목록이다.
    ``corpus_by_openalex`` 는 bare ID (W…) 를 키로 하는 딕셔너리이다.

    Args:
        paper: 인용 링크를 조회할 Paper.
        corpus_by_openalex: bare OpenAlex ID → Paper 딕셔너리 (``write_vault`` 가 빌드).
        fetch: True 이면 OpenAlex API 를 호출한다. False 이면 빈 리스트 반환.

    Returns:
        코퍼스에 존재하는 인용 Paper 목록. 네트워크 실패 시 빈 리스트.
    """
    if not fetch:
        return []
    if not paper.doi:
        logger.debug("build_citation_links: DOI 없음 (%s) — 건너뜀", paper.source)
        return []

    try:
        from core.metadata import openalex_meta

        meta = openalex_meta(paper.doi)
    except Exception as exc:
        logger.warning("OpenAlex 조회 실패 (%s): %s", paper.doi, exc)
        return []

    referenced: list[str] = meta.get("referenced_works") or []
    matches: list[Paper] = []
    for ref_id in referenced:
        bare = _bare_openalex_id(ref_id)
        if bare and bare in corpus_by_openalex:
            matches.append(corpus_by_openalex[bare])

    logger.debug(
        "인용 매칭: %s → %d/%d refs found in corpus",
        paper.source,
        len(matches),
        len(referenced),
    )
    return matches


# ---------------------------------------------------------------------------
# write_vault
# ---------------------------------------------------------------------------


def write_vault(
    papers: list["Paper"],
    out_dir: str | Path,
    *,
    bodies: dict[str, str] | None = None,
    fetch_citations: bool = False,
    limit: int | None = None,
) -> int:
    """코퍼스 논문 목록으로부터 Obsidian vault 노트 파일을 기록한다.

    Args:
        papers: Paper 인스턴스 목록.
        out_dir: 노트를 기록할 출력 디렉터리 (자동 생성).
        bodies: paper_key → 마크다운 본문 딕셔너리 (선택). 키는
            ``ingestion.load.processed_key(paper)`` 와 동일 규칙
            (openalex_id → doi → file → source 우선순위).
        fetch_citations: True 이면 각 논문의 인용 링크를 OpenAlex API 로 조회한다.
            기본값 False — 794개 논문에 대한 API 호출을 방지한다.
            ⚠️ True 로 설정하면 네트워크 요청이 최대 ``len(papers)`` 회 발생한다.
        limit: None 이 아니면 처음 N 편만 처리한다 (디버그 용도).

    Returns:
        기록된 노트 수.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # corpus_by_openalex: bare OpenAlex ID → Paper (인용 매핑용)
    corpus_by_openalex: dict[str, "Paper"] = {}
    for p in papers:
        bare = _bare_openalex_id(p.openalex_id)
        if bare:
            corpus_by_openalex[bare] = p

    target_papers = papers[:limit] if limit is not None else papers
    written = 0

    for i, paper in enumerate(target_papers, start=1):
        # bodies 키: processed_key 규칙 (openalex_id > doi > file > source)
        paper_key = _processed_key(paper)
        body = (bodies or {}).get(paper_key)

        # 인용 링크
        cited: list["Paper"] = []
        if fetch_citations:
            cited = build_citation_links(paper, corpus_by_openalex, fetch=True)

        note_text = paper_to_note(paper, body=body, cited_papers=cited or None)
        fname = note_filename(paper)
        dest = out_path / fname
        dest.write_text(note_text, encoding="utf-8")
        written += 1

        if i % 50 == 0 or i == len(target_papers):
            logger.info("vault 진행: %d/%d 노트 기록 완료", i, len(target_papers))

    logger.info("write_vault 완료: %d개 노트 → %s", written, out_path)
    return written


# ---------------------------------------------------------------------------
# Internal helper (delegates to ingestion.load.processed_key)
# ---------------------------------------------------------------------------


def _processed_key(paper: "Paper") -> str:
    """Paper 의 대표 키를 반환한다 (ingestion.load.processed_key 와 동일).

    ingestion.load.processed_key 의 우선순위:
        openalex_id (URL 형식 그대로) → normalize_doi(doi) → file → source

    #3122/Colab 에서 ``bodies = {processed_key(p): markdown}`` 로 빌드한 맵과
    동일한 키를 사용하므로 반드시 이 함수를 통해 조회해야 한다.
    """
    from ingestion.load.registry import processed_key

    return processed_key(paper)
