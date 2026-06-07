"""ingestion.obsidian.note_builder — Obsidian 노트 생성 유틸리티.

YAML frontmatter + 본문 + wikilink 인용 섹션을 포함한 마크다운 문서를 빌드한다.
yaml 라이브러리에 의존하지 않는다 (json.dumps 를 이용한 안전한 YAML 값 직렬화).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models.paper import Paper

# ---------------------------------------------------------------------------
# slug helper
# ---------------------------------------------------------------------------

_SLUG_NON_WORD = re.compile(r"[^\w가-힣]+")


def slug(text: str | None, max_len: int = 40) -> str:
    """텍스트를 파일시스템 안전 slug 로 변환한다.

    - None 또는 빈 문자열이면 빈 문자열 반환
    - 영숫자·한글 이외 문자는 '-' 로 대체
    - 앞뒤 '-' 제거
    - max_len 으로 잘라낸다

    Args:
        text: 변환할 문자열 (None 허용).
        max_len: 출력 최대 길이.

    Returns:
        slug 문자열 (파일명 세그먼트로 안전한 값).
    """
    if not text:
        return ""
    return _SLUG_NON_WORD.sub("-", text).strip("-")[:max_len]


# ---------------------------------------------------------------------------
# note_filename
# ---------------------------------------------------------------------------


def note_filename(paper: "Paper") -> str:
    """Paper 에 대한 결정론적 파일시스템 안전 노트 파일명을 반환한다.

    포맷: ``{year}_{first_author_slug}_{id_slug}.md``

    - year 없으면 'nd' (no date)
    - first_author 없으면 'na'
    - ID: openalex_id → doi 우선 순위. 둘 다 없으면 source 를 fallback 으로 사용.
    - openalex_id 가 URL 형식이면 bare ID(W…) 만 추출한다.

    Args:
        paper: Paper 인스턴스.

    Returns:
        파일명 문자열 (예: ``2020_Hargreaves_W1234567890.md``).
    """
    year_part = str(paper.year) if paper.year else "nd"
    author_part = slug(paper.first_author, 20) or "na"

    # ID 우선순위: openalex_id > doi > file > source
    # file 을 source 보다 앞에 두어 batch-tag source 충돌 방지 (여러 논문이 같은 source 공유)
    raw_id: str | None = paper.openalex_id or paper.doi or paper.file or paper.source
    id_part = slug(_bare_openalex_id(raw_id) if raw_id else None, 30) or "unknown"

    return f"{year_part}_{author_part}_{id_part}.md"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _bare_openalex_id(oid: str | None) -> str | None:
    """OpenAlex URL 형식 ID 에서 bare ID (W…) 를 추출한다.

    ``https://openalex.org/W2741809807`` → ``W2741809807``
    ``W2741809807`` → ``W2741809807``
    ``None`` → ``None``
    """
    if not oid:
        return None
    # URL 형식이면 마지막 경로 요소 반환
    if "/" in oid:
        return oid.rstrip("/").rsplit("/", 1)[-1]
    return oid


def _yaml_scalar(value: object) -> str:
    """Python 값을 YAML scalar 로 안전하게 직렬화한다.

    json.dumps 를 사용하므로: str은 " 로 인용, None → null,
    int/float → 그대로, 한국어 그대로 유지 (ensure_ascii=False).
    """
    return json.dumps(value, ensure_ascii=False)


def _yaml_str_list(items: list[str]) -> str:
    """Python str 리스트를 YAML flow sequence 로 직렬화한다."""
    return json.dumps(items, ensure_ascii=False)


# ---------------------------------------------------------------------------
# paper_to_note
# ---------------------------------------------------------------------------


def paper_to_note(
    paper: "Paper",
    *,
    body: str | None = None,
    cited_papers: list["Paper"] | None = None,
) -> str:
    """Paper 에 대한 Obsidian 호환 마크다운 노트를 빌드한다.

    Args:
        paper: 노트로 변환할 Paper 인스턴스.
        body: 선택적 본문 (ParsedDoc.markdown 등). 없으면 paper.abstract 또는 placeholder.
        cited_papers: 코퍼스 내 인용 논문 목록. wikilink 섹션에 사용.

    Returns:
        Obsidian 호환 마크다운 문자열 (``---`` frontmatter 로 시작).
    """
    # --- frontmatter 빌드 ---
    bare_oa_id = _bare_openalex_id(paper.openalex_id)
    tags = ["cpet-rag", paper.source]

    fm_lines: list[str] = [
        "---",
        f"doi: {_yaml_scalar(paper.doi)}",
        f"title: {_yaml_scalar(paper.title)}",
        f"authors: {_yaml_str_list(paper.authors)}",
        f"year: {_yaml_scalar(paper.year)}",
        f"journal: {_yaml_scalar(paper.journal)}",
        f"source: {_yaml_scalar(paper.source)}",
        f"oa_status: {_yaml_scalar(paper.oa_status)}",
        f"openalex_id: {_yaml_scalar(bare_oa_id)}",
        f"tags: {_yaml_str_list(tags)}",
        "---",
    ]
    frontmatter = "\n".join(fm_lines)

    # --- 헤더 + 메타 라인 ---
    author_display = paper.first_author or (paper.authors[0] if paper.authors else "Unknown")
    year_display = str(paper.year) if paper.year else "n.d."
    journal_display = paper.journal or ""
    meta_line = (
        f"_{author_display} ({year_display}){', ' + journal_display if journal_display else ''}_"
    )

    # --- 본문 ---
    if body:
        body_text = body.strip()
    elif paper.abstract:
        body_text = paper.abstract.strip()
    else:
        body_text = "> _(본문 없음 — PDF 파싱 후 재생성 필요)_"

    # --- 인용 섹션 (cited_papers 중 실제 corpus 내 논문만) ---
    citation_section = ""
    if cited_papers:
        link_lines = []
        for cited in cited_papers:
            fname = note_filename(cited)
            # Obsidian wikilink: .md 확장자 제거 (그래프 뷰 링크 해석 표준)
            link_target = fname[:-3] if fname.endswith(".md") else fname
            # wikilink alias: 제목에서 | ] 문자 제거 (Obsidian 파서 오작동 방지)
            safe_title = cited.title.replace("|", "-").replace("]", ")")
            link_lines.append(f"- [[{link_target}|{safe_title}]]")
        if link_lines:
            citation_section = "\n## 인용 (References in corpus)\n\n" + "\n".join(link_lines)

    # --- 조합 ---
    sections = [
        frontmatter,
        f"\n# {paper.title}",
        f"\n{meta_line}",
        f"\n{body_text}",
    ]
    if citation_section:
        sections.append(citation_section)

    return "\n".join(sections) + "\n"
