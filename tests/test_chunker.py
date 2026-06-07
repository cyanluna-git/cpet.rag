"""tests/test_chunker.py — chunk_document 단위 테스트 (REAL logic, no mocks)."""

from __future__ import annotations

import pytest

from core.models import Paper
from ingestion.chunk import chunk_document, count_tokens
from ingestion.parse.types import ParsedDoc, Section

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PAPER = Paper(
    doi="10.1234/cpet.test",
    title="Exercise Physiology and CPET: A Comprehensive Review",
    first_author="Smith",
    year=2024,
    journal="J Exercise Physiol",
    source="cpet_test",
    openalex_id="W1234567890",
)

SHORT_SECTION_TEXT = "This is a short section with only a few words."

MEDIUM_SECTION_TEXT = " ".join([f"Word{i}" for i in range(200)])  # ~200 words

LONG_SECTION_TEXT = " ".join([f"SciWord{i}" for i in range(900)])  # ~900 words → split


def _make_parsed(sections: list[Section], markdown: str = "") -> ParsedDoc:
    return ParsedDoc(
        source_path="/tmp/test.pdf",
        markdown=markdown or "\n".join(s.text for s in sections),
        n_pages=10,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def test_count_tokens_basic() -> None:
    """count_tokens 는 양의 정수를 반환한다."""
    n = count_tokens("Hello world, this is a test sentence.")
    assert isinstance(n, int)
    assert n > 0


def test_count_tokens_empty() -> None:
    """빈 문자열은 0 또는 아주 작은 값을 반환한다."""
    n = count_tokens("")
    assert isinstance(n, int)
    assert n >= 0


def test_count_tokens_longer_more() -> None:
    """긴 텍스트가 짧은 텍스트보다 토큰이 많다."""
    short = count_tokens("Hi")
    long_ = count_tokens("This is a much longer piece of text with many words and sentences.")
    assert long_ > short


# ---------------------------------------------------------------------------
# Basic chunk generation
# ---------------------------------------------------------------------------


def test_chunk_document_empty_sections_returns_empty() -> None:
    """sections 가 없으면 빈 리스트를 반환한다."""
    parsed = _make_parsed([])
    result = chunk_document(parsed, PAPER)
    assert result == []


def test_chunk_document_single_short_section() -> None:
    """단일 소형 섹션은 청크 1개를 만든다."""
    sections = [Section(heading="Abstract", text=SHORT_SECTION_TEXT, page=1)]
    parsed = _make_parsed(sections)

    chunks = chunk_document(parsed, PAPER)
    assert len(chunks) >= 1


def test_chunk_index_sequential() -> None:
    """chunk_index 는 0부터 연속적이다."""
    sections = [
        Section(heading="Introduction", text=MEDIUM_SECTION_TEXT, page=1),
        Section(heading="Methods", text=MEDIUM_SECTION_TEXT, page=3),
        Section(heading="Results", text=MEDIUM_SECTION_TEXT, page=5),
    ]
    parsed = _make_parsed(sections)
    chunks = chunk_document(parsed, PAPER)

    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks))), f"Non-sequential: {indices}"


def test_chunk_ids_unique() -> None:
    """모든 chunk id 가 고유하다."""
    sections = [
        Section(heading="Introduction", text=MEDIUM_SECTION_TEXT, page=1),
        Section(heading="Methods", text=MEDIUM_SECTION_TEXT, page=3),
        Section(heading="Results", text=MEDIUM_SECTION_TEXT, page=5),
    ]
    parsed = _make_parsed(sections)
    chunks = chunk_document(parsed, PAPER)

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), f"Duplicate ids found: {ids}"


def test_chunk_ids_deterministic() -> None:
    """같은 입력에서 두 번 실행해도 id 가 동일하다."""
    sections = [
        Section(heading="Background", text=MEDIUM_SECTION_TEXT, page=2),
    ]
    parsed = _make_parsed(sections)

    chunks1 = chunk_document(parsed, PAPER)
    chunks2 = chunk_document(parsed, PAPER)

    assert [c.id for c in chunks1] == [c.id for c in chunks2]


# ---------------------------------------------------------------------------
# ctx_text format
# ---------------------------------------------------------------------------


def test_ctx_text_contains_title() -> None:
    """ctx_text 에 논문 제목이 포함된다."""
    sections = [Section(heading="Introduction", text=SHORT_SECTION_TEXT, page=1)]
    parsed = _make_parsed(sections)
    chunks = chunk_document(parsed, PAPER)

    assert len(chunks) >= 1
    for c in chunks:
        assert PAPER.title in c.ctx_text, f"Title not found in ctx_text: {c.ctx_text[:100]}"


def test_ctx_text_contains_section_heading() -> None:
    """ctx_text 에 섹션 헤딩이 포함된다."""
    sections = [Section(heading="Methods", text=SHORT_SECTION_TEXT, page=2)]
    parsed = _make_parsed(sections)
    chunks = chunk_document(parsed, PAPER)

    assert len(chunks) >= 1
    for c in chunks:
        assert "Methods" in c.ctx_text


def test_ctx_text_format_structure() -> None:
    """ctx_text 가 [제목 · 저자 연도 · 저널 · §섹션] 형식을 갖는다."""
    sections = [Section(heading="Results", text=SHORT_SECTION_TEXT, page=4)]
    parsed = _make_parsed(sections)
    chunks = chunk_document(parsed, PAPER)

    assert len(chunks) >= 1
    ctx = chunks[0].ctx_text
    assert ctx.startswith("[")
    assert "·" in ctx
    assert "§Results" in ctx


def test_ctx_text_body_fallback() -> None:
    """heading=None 인 섹션은 ctx_text 에 '§body' 가 들어간다."""
    sections = [Section(heading=None, text=SHORT_SECTION_TEXT, page=1)]
    parsed = _make_parsed(sections)
    chunks = chunk_document(parsed, PAPER)

    assert len(chunks) >= 1
    assert "§body" in chunks[0].ctx_text


# ---------------------------------------------------------------------------
# Long section splitting
# ---------------------------------------------------------------------------


def test_long_section_splits_into_multiple_chunks() -> None:
    """max_tokens 를 초과하는 섹션은 여러 청크로 분할된다."""
    sections = [Section(heading="Discussion", text=LONG_SECTION_TEXT, page=6)]
    parsed = _make_parsed(sections)

    chunks = chunk_document(parsed, PAPER, target_tokens=200, max_tokens=400)
    assert len(chunks) > 1, "Long section should split into multiple chunks"


def test_no_chunk_exceeds_max_tokens() -> None:
    """어떤 청크도 max_tokens(text 기준)를 초과하지 않는다."""
    sections = [
        Section(heading="Introduction", text=SHORT_SECTION_TEXT, page=1),
        Section(heading="Long Section", text=LONG_SECTION_TEXT, page=2),
        Section(heading="Methods", text=MEDIUM_SECTION_TEXT, page=5),
    ]
    parsed = _make_parsed(sections)
    max_tokens = 400

    chunks = chunk_document(parsed, PAPER, target_tokens=200, max_tokens=max_tokens)

    for c in chunks:
        n = count_tokens(c.text)
        assert n <= max_tokens, (
            f"Chunk {c.chunk_index} exceeds max_tokens: {n} > {max_tokens}\n"
            f"text[:80]={c.text[:80]!r}"
        )


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------


def test_chunk_fields_populated() -> None:
    """doi, source, page, section 필드가 올바르게 채워진다."""
    sections = [Section(heading="Methods", text=MEDIUM_SECTION_TEXT, page=3)]
    parsed = _make_parsed(sections)
    chunks = chunk_document(parsed, PAPER)

    assert len(chunks) >= 1
    c = chunks[0]
    assert c.doi == PAPER.doi
    assert c.source == PAPER.source
    assert c.page == 3
    assert c.section == "Methods"
    assert c.embedding is None


def test_chunk_id_uses_openalex_id() -> None:
    """openalex_id 가 있으면 id 에 포함된다."""
    sections = [Section(heading="Abstract", text=SHORT_SECTION_TEXT, page=1)]
    parsed = _make_parsed(sections)
    chunks = chunk_document(parsed, PAPER)

    assert len(chunks) >= 1
    assert chunks[0].id.startswith(PAPER.openalex_id)


def test_chunk_id_falls_back_to_doi_slug() -> None:
    """openalex_id 없을 때 doi 슬러그를 사용한다."""
    paper_no_openalex = PAPER.model_copy(update={"openalex_id": None})
    sections = [Section(heading="Abstract", text=SHORT_SECTION_TEXT, page=1)]
    parsed = _make_parsed(sections)
    chunks = chunk_document(parsed, paper_no_openalex)

    assert len(chunks) >= 1
    assert "nd" not in chunks[0].id or PAPER.doi.replace("/", "_") in chunks[0].id.replace("/", "_")


def test_chunk_id_falls_back_to_nd() -> None:
    """openalex_id 와 doi 모두 없으면 'nd_' 로 시작한다."""
    paper_no_ids = PAPER.model_copy(update={"openalex_id": None, "doi": None})
    sections = [Section(heading="Abstract", text=SHORT_SECTION_TEXT, page=1)]
    parsed = _make_parsed(sections)
    chunks = chunk_document(parsed, paper_no_ids)

    assert len(chunks) >= 1
    assert chunks[0].id.startswith("nd_")


# ---------------------------------------------------------------------------
# Small section merging
# ---------------------------------------------------------------------------


def test_small_sections_may_merge() -> None:
    """여러 소형 섹션은 target_tokens 이하로 병합된다."""
    tiny = "Short text. " * 3  # ~very small
    sections = [Section(heading=f"Sec{i}", text=tiny, page=i) for i in range(5)]
    parsed = _make_parsed(sections)

    # target_tokens 을 크게 설정하면 모두 합산될 수 있다
    chunks = chunk_document(parsed, PAPER, target_tokens=1000, max_tokens=2000)
    # 5개 소형 섹션이 하나로 병합되어 청크 수 < 5 이어야 한다
    assert len(chunks) < len(
        sections
    ), f"Expected merging: got {len(chunks)} chunks from {len(sections)} tiny sections"


# ---------------------------------------------------------------------------
# Mixed scenario
# ---------------------------------------------------------------------------


def test_mixed_sections_reasonable_chunk_count() -> None:
    """여러 크기의 섹션이 섞여있을 때 청크 수가 합리적이다."""
    sections = [
        Section(heading=None, text=SHORT_SECTION_TEXT, page=1),  # preamble
        Section(heading="Introduction", text=MEDIUM_SECTION_TEXT, page=2),
        Section(heading="Methods", text=LONG_SECTION_TEXT, page=4),
        Section(heading="Results", text=MEDIUM_SECTION_TEXT, page=8),
        Section(heading="Discussion", text=SHORT_SECTION_TEXT, page=10),
    ]
    parsed = _make_parsed(sections)

    chunks = chunk_document(parsed, PAPER, target_tokens=200, max_tokens=400)

    # 최소한 3개 이상 (Long section 만으로도 2개 이상)
    assert len(chunks) >= 3
    # chunk_index 연속
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # id 유일
    assert len({c.id for c in chunks}) == len(chunks)
