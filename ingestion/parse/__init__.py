"""ingestion.parse — PDF → ParsedDoc 변환 패키지.

공개 API:
    parse_pdf   : PDF 파일을 파싱해 ParsedDoc 를 반환한다.
    ParsedDoc   : 파싱 결과 스키마 (markdown, sections, tables, low_confidence_pages).
    Section     : 섹션 스키마 (heading, text, page, level).
    TableRef    : 표 위치 참조 스키마 (index, page, caption).
"""

from ingestion.parse.docling_parser import parse_pdf
from ingestion.parse.types import ParsedDoc, Section, TableRef

__all__ = [
    "parse_pdf",
    "ParsedDoc",
    "Section",
    "TableRef",
]
