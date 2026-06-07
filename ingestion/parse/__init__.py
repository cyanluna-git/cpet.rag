"""ingestion.parse — PDF → ParsedDoc 변환 패키지.

공개 API:
    parse_pdf           : PDF 파일을 파싱해 ParsedDoc 를 반환한다.
    ParsedDoc           : 파싱 결과 스키마 (markdown, sections, tables, low_confidence_pages).
    Section             : 섹션 스키마 (heading, text, page, level).
    TableRef            : 표 위치 참조 스키마 (index, page, caption).
    render_page_image   : PDF 페이지 → PNG bytes (pymupdf 필요).
    vlm_extract_page    : PDF 페이지 → Gemini Flash 마크다운 (GEMINI_API_KEY 필요).
    apply_vlm_fallback  : ParsedDoc + 대상 페이지 → 스플라이싱된 새 ParsedDoc.
"""

from ingestion.parse.docling_parser import parse_pdf
from ingestion.parse.types import ParsedDoc, Section, TableRef
from ingestion.parse.vlm_fallback import (
    apply_vlm_fallback,
    render_page_image,
    vlm_extract_page,
)

__all__ = [
    "parse_pdf",
    "ParsedDoc",
    "Section",
    "TableRef",
    "render_page_image",
    "vlm_extract_page",
    "apply_vlm_fallback",
]
