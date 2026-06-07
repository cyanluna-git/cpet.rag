"""tests/test_vlm_fallback.py — ingestion.parse.vlm_fallback 단위 테스트.

네트워크 호출(Gemini REST) 과 pymupdf 렌더링은 모두 mock 처리.
GEMINI_API_KEY 없이 실행 가능 (CI 포함).

실제 Gemini 호출 검증은 Colab(GEMINI_API_KEY 설정) 환경에서 수행한다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.parse.types import ParsedDoc, Section
from ingestion.parse.vlm_fallback import (
    apply_vlm_fallback,
    render_page_image,
    vlm_extract_page,
)

# ──────────────────────────────────────────────────────────────────────────────
# 공통 픽스처
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"

CANNED_VLM_MARKDOWN = (
    "## Results\n\n"
    "| Participant | VO2max (ml/kg/min) |\n"
    "|-------------|--------------------|\n"
    "| A           | 52.3               |\n"
    "| B           | 48.7               |\n\n"
    "$$\\dot{V}O_2 = \\text{HR} \\times \\text{SV} \\times \\text{CaO}_2$$"
)


def _make_base_parsed_doc(low_confidence_pages: list[int] | None = None) -> ParsedDoc:
    """테스트용 최소 ParsedDoc 를 생성한다."""
    return ParsedDoc(
        source_path="/tmp/test.pdf",
        markdown="# Title\n\nSome body text.",
        n_pages=5,
        sections=[
            Section(heading="Introduction", text="Intro text.", page=1, level=1),
            Section(heading="Methods", text="Methods text.", page=2, level=1),
        ],
        low_confidence_pages=low_confidence_pages if low_confidence_pages is not None else [],
    )


# ──────────────────────────────────────────────────────────────────────────────
# apply_vlm_fallback 테스트
# ──────────────────────────────────────────────────────────────────────────────


class TestApplyVlmFallback:
    """apply_vlm_fallback() 의 스플라이싱·불변성·no-op 동작을 검증한다."""

    def test_splices_markdown(self, tmp_path: Path) -> None:
        """VLM 결과가 markdown 에 ## [VLM page N] 블록으로 추가되어야 한다."""
        parsed = _make_base_parsed_doc(low_confidence_pages=[3])
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-fake")

        with patch(
            "ingestion.parse.vlm_fallback.vlm_extract_page",
            return_value=CANNED_VLM_MARKDOWN,
        ):
            result = apply_vlm_fallback(parsed, pdf, api_key="fake-key")

        assert "## [VLM page 3]" in result.markdown
        assert "VO2max" in result.markdown

    def test_splices_section(self, tmp_path: Path) -> None:
        """VLM 결과가 sections 에 새 Section 으로 추가되어야 한다."""
        parsed = _make_base_parsed_doc(low_confidence_pages=[3])
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-fake")

        with patch(
            "ingestion.parse.vlm_fallback.vlm_extract_page",
            return_value=CANNED_VLM_MARKDOWN,
        ):
            result = apply_vlm_fallback(parsed, pdf, api_key="fake-key")

        vlm_sections = [s for s in result.sections if s.heading == "[VLM p.3]"]
        assert len(vlm_sections) == 1
        assert vlm_sections[0].page == 3
        assert "VO2max" in vlm_sections[0].text

    def test_vlm_pages_recorded(self, tmp_path: Path) -> None:
        """처리된 페이지 번호가 vlm_pages 에 기록되어야 한다."""
        parsed = _make_base_parsed_doc(low_confidence_pages=[2, 4])
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-fake")

        with patch(
            "ingestion.parse.vlm_fallback.vlm_extract_page",
            return_value=CANNED_VLM_MARKDOWN,
        ):
            result = apply_vlm_fallback(parsed, pdf, api_key="fake-key")

        assert result.vlm_pages == [2, 4]

    def test_non_mutating(self, tmp_path: Path) -> None:
        """원본 ParsedDoc 는 변경되지 않아야 한다."""
        parsed = _make_base_parsed_doc(low_confidence_pages=[3])
        original_md = parsed.markdown
        original_sections_len = len(parsed.sections)
        original_vlm_pages = list(parsed.vlm_pages)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-fake")

        with patch(
            "ingestion.parse.vlm_fallback.vlm_extract_page",
            return_value=CANNED_VLM_MARKDOWN,
        ):
            result = apply_vlm_fallback(parsed, pdf, api_key="fake-key")

        # 원본 불변 확인
        assert parsed.markdown == original_md
        assert len(parsed.sections) == original_sections_len
        assert parsed.vlm_pages == original_vlm_pages

        # 반환값은 달라야 함
        assert result is not parsed
        assert len(result.markdown) > len(original_md)
        assert len(result.sections) > original_sections_len

    def test_noop_when_no_low_confidence_pages(self, tmp_path: Path) -> None:
        """low_confidence_pages 가 빈 경우 원본 ParsedDoc 를 그대로 반환한다."""
        parsed = _make_base_parsed_doc(low_confidence_pages=[])
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-fake")

        with patch(
            "ingestion.parse.vlm_fallback.vlm_extract_page",
        ) as mock_extract:
            result = apply_vlm_fallback(parsed, pdf, api_key="fake-key")

        mock_extract.assert_not_called()
        assert result is parsed  # 동일 객체 반환

    def test_noop_when_explicit_pages_empty(self, tmp_path: Path) -> None:
        """pages=[] 를 명시하면 low_confidence_pages 가 있어도 no-op 이어야 한다."""
        parsed = _make_base_parsed_doc(low_confidence_pages=[2, 3])
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-fake")

        with patch(
            "ingestion.parse.vlm_fallback.vlm_extract_page",
        ) as mock_extract:
            result = apply_vlm_fallback(parsed, pdf, pages=[], api_key="fake-key")

        mock_extract.assert_not_called()
        assert result is parsed

    def test_explicit_pages_override(self, tmp_path: Path) -> None:
        """pages 인수를 명시하면 low_confidence_pages 대신 그 페이지를 처리한다."""
        parsed = _make_base_parsed_doc(low_confidence_pages=[1])  # page 1은 저신뢰도
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-fake")

        with patch(
            "ingestion.parse.vlm_fallback.vlm_extract_page",
            return_value=CANNED_VLM_MARKDOWN,
        ) as mock_extract:
            result = apply_vlm_fallback(parsed, pdf, pages=[5], api_key="fake-key")

        # page 5 만 처리되어야 함 (page 1이 아님)
        called_pages = [call.args[1] for call in mock_extract.call_args_list]
        assert called_pages == [5]
        assert result.vlm_pages == [5]

    def test_duplicate_pages_deduplicated(self, tmp_path: Path) -> None:
        """중복 페이지는 한 번만 처리되어야 한다."""
        parsed = _make_base_parsed_doc(low_confidence_pages=[])
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-fake")

        with patch(
            "ingestion.parse.vlm_fallback.vlm_extract_page",
            return_value=CANNED_VLM_MARKDOWN,
        ) as mock_extract:
            result = apply_vlm_fallback(parsed, pdf, pages=[3, 3, 3], api_key="fake-key")

        assert mock_extract.call_count == 1
        assert result.vlm_pages == [3]

    def test_multiple_pages_all_spliced(self, tmp_path: Path) -> None:
        """여러 페이지가 모두 마크다운에 추가되어야 한다."""
        parsed = _make_base_parsed_doc(low_confidence_pages=[2, 4])
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-fake")

        with patch(
            "ingestion.parse.vlm_fallback.vlm_extract_page",
            return_value=CANNED_VLM_MARKDOWN,
        ):
            result = apply_vlm_fallback(parsed, pdf, api_key="fake-key")

        assert "## [VLM page 2]" in result.markdown
        assert "## [VLM page 4]" in result.markdown


# ──────────────────────────────────────────────────────────────────────────────
# render_page_image 테스트
# ──────────────────────────────────────────────────────────────────────────────


class TestRenderPageImage:
    """render_page_image() 의 PNG 출력과 에러 처리를 검증한다."""

    @pytest.mark.skipif(
        not SAMPLE_PDF.exists(),
        reason="tests/fixtures/sample.pdf 없음",
    )
    def test_returns_png_bytes(self) -> None:
        """실제 sample.pdf 로 렌더링했을 때 PNG bytes 가 반환되어야 한다."""
        pytest.importorskip("fitz", reason="pymupdf 미설치 — render_page_image 건너뜀")

        result = render_page_image(SAMPLE_PDF, page_num=1)

        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG", "PNG 시그니처가 아님"
        assert len(result) > 1000, "PNG 가 너무 작음"

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """존재하지 않는 PDF 경로는 FileNotFoundError 를 발생시켜야 한다."""
        pytest.importorskip("fitz", reason="pymupdf 미설치 — 건너뜀")

        with pytest.raises(FileNotFoundError):
            render_page_image(tmp_path / "nonexistent.pdf", page_num=1)

    def test_import_error_without_pymupdf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """pymupdf(fitz) 가 없을 때 ImportError 가 발생해야 한다."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):
            if name == "fitz":
                raise ImportError("No module named 'fitz'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="pymupdf"):
            render_page_image("/tmp/test.pdf", page_num=1)


# ──────────────────────────────────────────────────────────────────────────────
# vlm_extract_page 테스트
# ──────────────────────────────────────────────────────────────────────────────


class TestVlmExtractPage:
    """vlm_extract_page() 의 에러 처리와 API 호출 구조를 검증한다."""

    def test_raises_when_api_key_none(self, tmp_path: Path) -> None:
        """api_key=None 이고 settings 에도 키가 없으면 RuntimeError 가 발생해야 한다."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-fake")

        fake_settings = MagicMock()
        fake_settings.gemini_api_key = None

        with patch("ingestion.parse.vlm_fallback.Settings", return_value=fake_settings):
            with pytest.raises(RuntimeError, match="Gemini API 키"):
                vlm_extract_page(pdf, page_num=1, api_key=None)

    def test_raises_when_api_key_empty_string(self, tmp_path: Path) -> None:
        """api_key="" (빈 문자열)일 때도 RuntimeError 가 발생해야 한다."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-fake")

        fake_settings = MagicMock()
        fake_settings.gemini_api_key = None

        with patch("ingestion.parse.vlm_fallback.Settings", return_value=fake_settings):
            with pytest.raises(RuntimeError, match="Gemini API 키"):
                vlm_extract_page(pdf, page_num=1, api_key="")

    def test_calls_gemini_api_with_key(self, tmp_path: Path) -> None:
        """api_key 가 있으면 _call_gemini_api 가 호출되어야 한다."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-fake")

        with (
            patch(
                "ingestion.parse.vlm_fallback.render_page_image",
                return_value=b"\x89PNG\r\n\x1a\nfake",
            ),
            patch(
                "ingestion.parse.vlm_fallback._call_gemini_api",
                return_value=CANNED_VLM_MARKDOWN,
            ) as mock_gemini,
        ):
            result = vlm_extract_page(pdf, page_num=2, api_key="test-key-123")

        mock_gemini.assert_called_once()
        call_args = mock_gemini.call_args
        assert call_args.args[1] == "test-key-123"  # api_key 가 전달됨
        assert result == CANNED_VLM_MARKDOWN

    def test_settings_key_used_when_arg_none(self, tmp_path: Path) -> None:
        """api_key=None 이고 settings.gemini_api_key 에 값이 있으면 그것을 사용해야 한다."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-fake")

        fake_settings = MagicMock()
        fake_settings.gemini_api_key = "settings-key-456"

        with (
            patch("ingestion.parse.vlm_fallback.Settings", return_value=fake_settings),
            patch(
                "ingestion.parse.vlm_fallback.render_page_image",
                return_value=b"\x89PNG\r\n\x1a\nfake",
            ),
            patch(
                "ingestion.parse.vlm_fallback._call_gemini_api",
                return_value=CANNED_VLM_MARKDOWN,
            ) as mock_gemini,
        ):
            result = vlm_extract_page(pdf, page_num=1, api_key=None)

        mock_gemini.assert_called_once()
        call_args = mock_gemini.call_args
        assert call_args.args[1] == "settings-key-456"
        assert result == CANNED_VLM_MARKDOWN
