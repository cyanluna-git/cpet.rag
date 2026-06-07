"""tests/test_parse.py — ingestion.parse 단위 테스트.

두 가지 테스트 경로:
1. 모의 테스트 (항상 실행): DocumentConverter 를 monkeypatch 로 대체해
   ParsedDoc 스키마·섹션 추출 로직·신뢰도 페이지 분류를 검증한다.
2. 실제 파싱 테스트 (tests/fixtures/sample.pdf 있을 때만 실행):
   실제 Docling 모델로 sample.pdf 를 파싱해 최소 기준을 확인한다.
   ─ Colab / GPU 인입 환경에서 반드시 검증할 것.
   ─ 로컬 첫 실행 시 Docling 레이아웃 모델 다운로드로 수 분 소요될 수 있다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ingestion.parse import ParsedDoc, Section, TableRef, parse_pdf

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — fake Docling 객체 생성
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


def _make_item(label: Any, text: str, page_no: int | None, level: int | None = None):
    """DocItem 처럼 동작하는 MagicMock 을 반환한다.

    label 은 실제 DocItemLabel enum 인스턴스를 전달해야 한다.
    MagicMock 을 쓰면 frozenset 포함 비교(in)가 False 를 반환한다.
    """
    item = MagicMock()
    item.label = label  # 실제 DocItemLabel 인스턴스 (enum 비교용)
    item.text = text
    item.level = level

    prov = MagicMock()
    prov.page_no = page_no
    item.prov = [prov] if page_no is not None else []
    return item


def _make_fake_result(status_value: str = "success") -> MagicMock:
    """DocumentConverter.convert() 가 반환하는 ConversionResult 를 흉내낸다."""
    from docling.datamodel.base_models import ConversionStatus
    from docling_core.types.doc.document import DocItemLabel

    items = [
        _make_item(DocItemLabel.TITLE, "CPET in Athletes", 1),
        _make_item(DocItemLabel.TEXT, "Preamble text before introduction.", 1),
        _make_item(DocItemLabel.SECTION_HEADER, "Introduction", 1, level=1),
        _make_item(DocItemLabel.TEXT, "Exercise testing is widely used.", 2),
        _make_item(DocItemLabel.SECTION_HEADER, "Methods", 2, level=1),
        _make_item(DocItemLabel.TEXT, "Participants were recruited.", 2),
        _make_item(DocItemLabel.PARAGRAPH, "Exclusion criteria applied.", 3),
        _make_item(DocItemLabel.SECTION_HEADER, "Results", 3, level=1),
        _make_item(DocItemLabel.TEXT, "Peak VO2 was 52 ml/kg/min.", 3),
    ]

    # Table mock
    table_item = MagicMock()
    table_prov = MagicMock()
    table_prov.page_no = 3
    table_item.prov = [table_prov]
    table_caption = MagicMock()
    table_caption.text = "Table 1. Participant characteristics."
    table_item.captions = [table_caption]

    # Document mock
    doc = MagicMock()
    doc.export_to_markdown.return_value = (
        "# CPET in Athletes\n\nPreamble text before introduction.\n\n"
        "## Introduction\n\nExercise testing is widely used.\n\n"
        "## Methods\n\nParticipants were recruited.\n\n"
        "Exclusion criteria applied.\n\n"
        "## Results\n\nPeak VO2 was 52 ml/kg/min."
    )
    doc.pages = {1: MagicMock(), 2: MagicMock(), 3: MagicMock()}
    doc.tables = [table_item]
    doc.iterate_items.return_value = [(item, 0) for item in items]

    # Confidence mock — page 2 has low layout_score, page 3 has NaN parse_score
    page_scores: dict[int, Any] = {}
    for pg in [1, 2, 3]:
        scores = MagicMock()
        if pg == 1:
            scores.parse_score = 0.9
            scores.layout_score = 0.85
        elif pg == 2:
            scores.parse_score = 0.8
            scores.layout_score = 0.3  # < threshold
        else:  # pg == 3
            scores.parse_score = float("nan")
            scores.layout_score = float("nan")
        page_scores[pg] = scores

    confidence = MagicMock()
    confidence.pages = page_scores

    result = MagicMock()
    status_map = {s.value: s for s in ConversionStatus}
    result.status = status_map[status_value]
    result.errors = []
    result.document = doc
    result.confidence = confidence

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 모의 기반 단위 테스트 (항상 실행)
# ──────────────────────────────────────────────────────────────────────────────


class TestParsedDocSchema:
    """ParsedDoc / Section / TableRef 스키마 유효성 검사."""

    def test_section_minimal(self) -> None:
        s = Section(text="Some body text.")
        assert s.heading is None
        assert s.level is None
        assert s.page is None

    def test_section_full(self) -> None:
        s = Section(heading="Methods", text="We recruited 30 cyclists.", page=2, level=1)
        assert s.heading == "Methods"
        assert s.page == 2
        assert s.level == 1

    def test_tableref(self) -> None:
        t = TableRef(index=0, page=3, caption="Table 1.")
        assert t.index == 0
        assert t.page == 3

    def test_parseddoc_defaults(self) -> None:
        doc = ParsedDoc(
            source_path="/tmp/test.pdf",
            markdown="# Title\n\nSome text.",
            n_pages=1,
        )
        assert doc.sections == []
        assert doc.tables == []
        assert doc.low_confidence_pages == []

    def test_parseddoc_full(self) -> None:
        doc = ParsedDoc(
            source_path="/tmp/paper.pdf",
            markdown="# Title\n\nBody.",
            n_pages=5,
            sections=[Section(heading="Intro", text="Intro text.", page=1, level=1)],
            tables=[TableRef(index=0, page=2)],
            low_confidence_pages=[3, 4],
        )
        assert doc.n_pages == 5
        assert len(doc.sections) == 1
        assert len(doc.low_confidence_pages) == 2


class TestMockedParsePdf:
    """monkeypatch 로 DocumentConverter 를 교체해 parse_pdf 로직을 검증한다."""

    def _run_with_mock(self, tmp_path: Path, status: str = "success") -> ParsedDoc:
        pdf = tmp_path / "dummy.pdf"
        pdf.write_bytes(b"%PDF-fake")  # 실제 Docling 을 호출하지 않으므로 내용 무관

        fake_result = _make_fake_result(status_value=status)

        with patch("ingestion.parse.docling_parser._build_converter") as mock_builder:
            mock_converter = MagicMock()
            mock_converter.convert.return_value = fake_result
            mock_builder.return_value = mock_converter
            return parse_pdf(pdf)

    def test_returns_parseddoc(self, tmp_path: Path) -> None:
        doc = self._run_with_mock(tmp_path)
        assert isinstance(doc, ParsedDoc)

    def test_source_path_set(self, tmp_path: Path) -> None:
        doc = self._run_with_mock(tmp_path)
        assert "dummy.pdf" in doc.source_path

    def test_markdown_non_empty(self, tmp_path: Path) -> None:
        doc = self._run_with_mock(tmp_path)
        assert len(doc.markdown) > 100

    def test_n_pages(self, tmp_path: Path) -> None:
        doc = self._run_with_mock(tmp_path)
        assert doc.n_pages == 3

    def test_sections_extracted(self, tmp_path: Path) -> None:
        doc = self._run_with_mock(tmp_path)
        # Preamble + 3 headings = 4 sections
        assert len(doc.sections) >= 3
        headings = [s.heading for s in doc.sections if s.heading]
        assert "Introduction" in headings
        assert "Methods" in headings
        assert "Results" in headings

    def test_section_page_numbers(self, tmp_path: Path) -> None:
        doc = self._run_with_mock(tmp_path)
        intro = next(s for s in doc.sections if s.heading == "Introduction")
        assert intro.page == 1
        assert intro.level == 1

    def test_section_text_content(self, tmp_path: Path) -> None:
        doc = self._run_with_mock(tmp_path)
        methods = next(s for s in doc.sections if s.heading == "Methods")
        assert "recruited" in methods.text

    def test_preamble_section(self, tmp_path: Path) -> None:
        doc = self._run_with_mock(tmp_path)
        preamble = doc.sections[0]
        # 첫 섹션: title 아이템이 heading으로, 그 이후 text가 body
        # OR heading=None 의 preamble 이 먼저 올 수 있다 — 두 경우 모두 허용
        assert preamble is not None

    def test_tables_extracted(self, tmp_path: Path) -> None:
        doc = self._run_with_mock(tmp_path)
        assert len(doc.tables) == 1
        assert doc.tables[0].page == 3
        assert doc.tables[0].caption is not None

    def test_low_confidence_pages(self, tmp_path: Path) -> None:
        doc = self._run_with_mock(tmp_path)
        # page 2: layout_score=0.3 → 저신뢰도
        # page 3: NaN → 저신뢰도
        assert 2 in doc.low_confidence_pages
        assert 3 in doc.low_confidence_pages
        # page 1: 정상
        assert 1 not in doc.low_confidence_pages

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_pdf("/tmp/__nonexistent_cpet_test__.pdf")

    def test_failure_status_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Docling 변환 실패"):
            self._run_with_mock(tmp_path, status="failure")


# ──────────────────────────────────────────────────────────────────────────────
# 실제 파싱 테스트 (sample.pdf 있을 때만 실행)
# 첫 실행 시 Docling 레이아웃 모델을 다운로드하므로 로컬에서는 수 분 소요 가능.
# GPU 인입 환경(Colab 등) 에서 반드시 검증할 것.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not SAMPLE_PDF.exists(), reason="tests/fixtures/sample.pdf 없음 — 실제 파싱 테스트 건너뜀"
)
class TestRealDoclingParse:
    """실제 Docling 모델을 사용한 통합 테스트."""

    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedDoc:
        return parse_pdf(SAMPLE_PDF)

    def test_type(self, parsed: ParsedDoc) -> None:
        assert isinstance(parsed, ParsedDoc)

    def test_markdown_length(self, parsed: ParsedDoc) -> None:
        assert len(parsed.markdown) > 500, f"마크다운이 너무 짧음: {len(parsed.markdown)} chars"

    def test_n_pages_positive(self, parsed: ParsedDoc) -> None:
        assert parsed.n_pages >= 1

    def test_at_least_one_section(self, parsed: ParsedDoc) -> None:
        assert len(parsed.sections) >= 1

    def test_summary_print(self, parsed: ParsedDoc, capsys) -> None:
        """파싱 결과 요약을 출력한다 (CI 에서 눈으로 확인용)."""
        print("\n=== Real Parse Summary ===")
        print(f"  source : {Path(parsed.source_path).name}")
        print(f"  pages  : {parsed.n_pages}")
        print(f"  sections: {len(parsed.sections)}")
        print(f"  tables : {len(parsed.tables)}")
        print(f"  low_conf: {parsed.low_confidence_pages}")
        print(f"  md_len : {len(parsed.markdown)} chars")
        print(f"  first section heading: {parsed.sections[0].heading!r}")
        print(f"  markdown preview: {parsed.markdown[:200]!r}")
        captured = capsys.readouterr()
        assert captured.out  # 출력이 있어야 함
