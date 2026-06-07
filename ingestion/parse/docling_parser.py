"""ingestion.parse.docling_parser — Docling 기반 PDF → ParsedDoc 변환기.

기본 설정:
- OCR 비활성화 (속도 우선).  OCR 필요 시 parse_pdf(do_ocr=True) 로 호출.
- 기본 device="cpu": Mac MPS 는 float64 미지원으로 실패하므로 CPU 기본.
  GPU 서버(Colab / AWS Batch g5) 에서는 parse_pdf(..., device="cuda") 로 지정할 것.
- 첫 실행 시 Docling 레이아웃 모델을 다운로드하므로 수분 소요될 수 있다.

NOTE for #3117 (Late Chunking):
  sections 의 level 필드는 Docling DocItemLabel 에서 가져오지만,
  이 코퍼스의 PDF 다수는 모든 헤더를 level=1 로 반환한다.
  #3117 은 level 에 의존한 계층 처리보다 section 순서 기반 경계 분할을 권장한다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from core.log import get_logger
from ingestion.parse.types import ParsedDoc, Section, TableRef

if TYPE_CHECKING:
    from docling.datamodel.document import ConversionResult
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc.document import DoclingDocument

logger = get_logger(__name__)

# Docling 신뢰도 임계값 — 이 값 미만(또는 NaN) 이면 VLM 폴백 후보로 분류된다.
# TODO: 코퍼스 전체 분포 확인 후 조정 (parse_score vs layout_score 트레이드오프 고려)
LOW_CONFIDENCE_THRESHOLD: float = 0.5


def _build_converter(*, do_ocr: bool = False, device: str = "cpu") -> DocumentConverter:
    """DocumentConverter 를 빌드한다.

    Args:
        do_ocr: True 면 OCR 활성화 (스캔 PDF 처리용). 기본 False (속도 우선).
        device: 추론 디바이스. 기본 "cpu".
            - "cpu": 로컬 Mac / CPU-only 환경 (MPS 는 float64 미지원으로 실패함).
            - "cuda": GPU 서버 (AWS Batch g5 등).
            - "auto": Docling 자동 선택 (MPS 환경에서는 실패할 수 있으므로 비권장).

    Returns:
        DocumentConverter 인스턴스.
    """
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    device_map: dict[str, AcceleratorDevice] = {
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,
        "auto": AcceleratorDevice.AUTO,
    }
    accel_device = device_map.get(device, AcceleratorDevice.CPU)

    options = PdfPipelineOptions(
        do_ocr=do_ocr,
        accelerator_options=AcceleratorOptions(device=accel_device),
    )
    return DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=options),
        }
    )


def _extract_sections(document: DoclingDocument) -> list[Section]:
    """Docling DoclingDocument 를 순회해 섹션 목록을 구성한다.

    헤딩(title / section_header) 을 만나면 현재 섹션을 닫고 새 섹션을 시작한다.
    헤딩 이전에 등장하는 본문은 heading=None 인 preamble 섹션에 모인다.

    Args:
        document: Docling DoclingDocument 인스턴스.

    Returns:
        Section 목록 (순서 보존).
    """
    from docling_core.types.doc.document import DocItemLabel

    TEXT_LABELS: frozenset[DocItemLabel] = frozenset(
        [
            DocItemLabel.TEXT,
            DocItemLabel.PARAGRAPH,
            DocItemLabel.LIST_ITEM,
            DocItemLabel.FOOTNOTE,
            DocItemLabel.CAPTION,
            DocItemLabel.FORMULA,
        ]
    )
    HEADING_LABELS: frozenset[DocItemLabel] = frozenset(
        [
            DocItemLabel.TITLE,
            DocItemLabel.SECTION_HEADER,
        ]
    )

    sections: list[Section] = []
    current_heading: str | None = None
    current_page: int | None = None
    current_level: int | None = None
    current_texts: list[str] = []

    def _flush() -> None:
        """현재 버퍼를 Section 으로 닫는다."""
        text = "\n\n".join(current_texts).strip()
        if current_heading is not None or text:
            sections.append(
                Section(
                    heading=current_heading,
                    text=text,
                    page=current_page,
                    level=current_level,
                )
            )

    for item, _depth in document.iterate_items():
        label: DocItemLabel = item.label  # type: ignore[attr-defined]

        if label in HEADING_LABELS:
            _flush()
            current_texts = []
            current_heading = getattr(item, "text", None)
            # TitleItem 에는 level 이 없으므로 getattr 로 안전하게 접근
            current_level = getattr(item, "level", None)
            # prov 는 ProvenanceItem 목록
            prov_list = getattr(item, "prov", [])
            current_page = prov_list[0].page_no if prov_list else None

        elif label in TEXT_LABELS:
            text = getattr(item, "text", None)
            if text and text.strip():
                current_texts.append(text.strip())

    _flush()  # 마지막 섹션 닫기
    return sections


def _extract_tables(document: DoclingDocument) -> list[TableRef]:
    """Docling 문서에서 TableRef 목록을 추출한다."""
    tables: list[TableRef] = []
    for idx, item in enumerate(document.tables):
        prov_list = getattr(item, "prov", [])
        page = prov_list[0].page_no if prov_list else None

        # 캡션 — captions 는 RefItem 목록이므로 resolve 가 필요할 수 있다.
        caption: str | None = None
        captions_list = getattr(item, "captions", [])
        if captions_list:
            resolved = getattr(captions_list[0], "text", None)
            if resolved is None:
                # Ref 객체인 경우 문서에서 직접 resolve
                try:
                    ref_id = captions_list[0].cref  # type: ignore[attr-defined]
                    resolved_item = document.resolve_ref(ref_id)
                    resolved = getattr(resolved_item, "text", None)
                except Exception:
                    pass
            caption = resolved

        tables.append(TableRef(index=idx, page=page, caption=caption))

    return tables


def _extract_low_confidence_pages(result: ConversionResult) -> list[int]:
    """신뢰도 낮은 페이지 번호를 추출한다.

    Docling result.confidence.pages: dict[int, PageConfidenceScores]
    각 PageConfidenceScores 는 parse_score, layout_score 를 갖는다.

    NaN 점수는 빈 페이지(텍스트 없음)를 뜻하므로 명시적으로 저신뢰도로 분류한다.
    """
    low_pages: list[int] = []

    if result.confidence is None:
        logger.debug("confidence 정보 없음 — low_confidence_pages 를 [] 로 설정")
        return low_pages

    for page_no, scores in result.confidence.pages.items():
        parse_s: float = getattr(scores, "parse_score", float("nan"))
        layout_s: float = getattr(scores, "layout_score", float("nan"))

        # NaN → 저신뢰도로 간주 (빈 페이지, 스캔 이미지 등)
        is_nan = math.isnan(parse_s) or math.isnan(layout_s)
        is_low = (not math.isnan(parse_s) and parse_s < LOW_CONFIDENCE_THRESHOLD) or (
            not math.isnan(layout_s) and layout_s < LOW_CONFIDENCE_THRESHOLD
        )

        if is_nan or is_low:
            low_pages.append(page_no)
            logger.debug(
                "저신뢰도 페이지 %d — parse=%.3f layout=%.3f",
                page_no,
                parse_s if not math.isnan(parse_s) else -1.0,
                layout_s if not math.isnan(layout_s) else -1.0,
            )

    return sorted(low_pages)


def parse_pdf(
    path: str | Path,
    *,
    do_ocr: bool = False,
    device: str = "cpu",
) -> ParsedDoc:
    """PDF 파일을 파싱해 ParsedDoc 를 반환한다.

    Args:
        path: PDF 파일 경로.
        do_ocr: OCR 활성화 여부 (기본 False). 스캔 PDF 처리 시 True 로 설정.
        device: 추론 디바이스 ("cpu" | "cuda" | "auto"). 기본 "cpu".
            Mac 로컬 환경에서는 MPS 가 float64 미지원으로 실패하므로 "cpu" 권장.
            GPU 서버(Colab / AWS Batch g5) 에서는 "cuda" 를 사용하면 속도가 향상된다.

    Returns:
        ParsedDoc — markdown, sections, tables, low_confidence_pages 포함.

    Raises:
        FileNotFoundError: path 파일이 존재하지 않을 때.
        RuntimeError: Docling 변환 실패(status == failure) 시.
    """
    from docling.datamodel.base_models import ConversionStatus

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {path}")

    logger.info("PDF 파싱 시작: %s (device=%s)", path, device)

    converter = _build_converter(do_ocr=do_ocr, device=device)

    result = converter.convert(str(path))

    if result.status == ConversionStatus.FAILURE:
        errors_str = "; ".join(str(e) for e in result.errors)
        raise RuntimeError(f"Docling 변환 실패 [{path.name}]: {errors_str}")

    if result.status == ConversionStatus.PARTIAL_SUCCESS:
        errors_str = "; ".join(str(e) for e in result.errors)
        logger.warning("Docling 부분 변환 성공 [%s]: %s", path.name, errors_str)

    document = result.document

    markdown: str = document.export_to_markdown()
    n_pages: int = len(document.pages)
    sections = _extract_sections(document)
    tables = _extract_tables(document)
    low_confidence_pages = _extract_low_confidence_pages(result)

    logger.info(
        "파싱 완료: %s — %d pages, %d sections, %d tables, %d low-conf pages, md=%d chars",
        path.name,
        n_pages,
        len(sections),
        len(tables),
        len(low_confidence_pages),
        len(markdown),
    )

    return ParsedDoc(
        source_path=str(path),
        markdown=markdown,
        n_pages=n_pages,
        sections=sections,
        tables=tables,
        low_confidence_pages=low_confidence_pages,
    )
