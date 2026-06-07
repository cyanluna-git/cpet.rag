"""ingestion.parse.vlm_fallback — Gemini Flash VLM 폴백 재추출.

복잡하거나 신뢰도가 낮은 PDF 페이지(표·수식 포함)를 Gemini Flash Vision 으로
재추출해 ParsedDoc 에 스플라이싱한다.

공개 API:
    render_page_image   : PDF 페이지 → PNG bytes (pymupdf, lazy-import).
    vlm_extract_page    : PDF 페이지 → Gemini Flash 마크다운 문자열.
    apply_vlm_fallback  : ParsedDoc + 대상 페이지 → 스플라이싱된 새 ParsedDoc.

Gemini REST:
    endpoint : https://generativelanguage.googleapis.com/v1beta/models/
                 gemini-2.0-flash:generateContent?key=<API_KEY>
    method   : POST  Content-Type: application/json
    body     : { "contents": [{ "parts": [
                    { "inline_data": { "mime_type": "image/png",
                                       "data": "<base64-png>" } },
                    { "text": "<prompt>" }
                ]}]}
    response : candidates[0].content.parts[0].text

pymupdf 는 optional-ingestion 에만 있으므로 fitz import 는 render_page_image 내부에서만.
httpx 는 base 의존성이므로 최상위 import 가능.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from core.config.settings import Settings
from core.log import get_logger

if TYPE_CHECKING:
    pass  # 타입 어노테이션 전용 import 는 여기에

logger = get_logger(__name__)

# Gemini REST 엔드포인트
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_GEMINI_MODEL = "gemini-2.0-flash"
_GEMINI_ACTION = "generateContent"

# VLM 추출 프롬프트
_VLM_PROMPT = (
    "Extract this academic paper page as clean Markdown. "
    "Preserve tables as Markdown tables and equations as LaTeX. "
    "Output only the markdown."
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. 페이지 렌더링
# ──────────────────────────────────────────────────────────────────────────────


def render_page_image(pdf_path: str | Path, page_num: int, dpi: int = 200) -> bytes:
    """PDF 의 단일 페이지를 PNG bytes 로 렌더링한다.

    Args:
        pdf_path: PDF 파일 경로.
        page_num: 렌더링할 페이지 번호 (1-based).
        dpi: 출력 해상도. 기본 200 DPI.

    Returns:
        PNG 이미지 bytes.

    Raises:
        ImportError: pymupdf(fitz) 가 설치되지 않았을 때.
        FileNotFoundError: pdf_path 파일이 없을 때.
        IndexError: page_num 이 문서 범위를 벗어날 때.
    """
    try:
        import fitz  # pymupdf
    except ImportError as exc:
        raise ImportError(
            "pymupdf 가 설치되지 않았습니다. "
            "`uv sync --extra ingestion` 을 실행하세요."
        ) from exc

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    try:
        # fitz 는 0-based 인덱스 사용
        page = doc[page_num - 1]
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix)
        return pix.tobytes("png")
    finally:
        doc.close()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Gemini REST 호출 (격리된 내부 함수 — 테스트에서 이 함수만 mock 가능)
# ──────────────────────────────────────────────────────────────────────────────


def _call_gemini_api(image_bytes: bytes, api_key: str) -> str:
    """Gemini REST API 를 호출해 이미지에서 마크다운을 추출한다.

    Args:
        image_bytes: PNG 이미지 bytes.
        api_key: Gemini API 키.

    Returns:
        Gemini 가 반환한 마크다운 문자열.

    Raises:
        httpx.HTTPStatusError: API 응답이 4xx/5xx 일 때.
        ValueError: 응답에서 텍스트를 추출하지 못했을 때.
    """
    url = f"{_GEMINI_BASE_URL}/{_GEMINI_MODEL}:{_GEMINI_ACTION}"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": encoded,
                        }
                    },
                    {"text": _VLM_PROMPT},
                ]
            }
        ]
    }

    response = httpx.post(
        url,
        params={"key": api_key},
        json=payload,
        timeout=60.0,
    )
    response.raise_for_status()

    data = response.json()
    try:
        text: str = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"Gemini API 응답에서 텍스트를 추출할 수 없습니다: {data}"
        ) from exc

    return text


# ──────────────────────────────────────────────────────────────────────────────
# 3. VLM 페이지 추출
# ──────────────────────────────────────────────────────────────────────────────


def vlm_extract_page(
    pdf_path: str | Path,
    page_num: int,
    *,
    api_key: str | None = None,
) -> str:
    """PDF 페이지를 Gemini Flash 로 재추출해 마크다운을 반환한다.

    Args:
        pdf_path: PDF 파일 경로.
        page_num: 추출할 페이지 번호 (1-based).
        api_key: Gemini API 키. None 이면 settings.gemini_api_key 를 사용.

    Returns:
        Gemini 가 반환한 마크다운 문자열.

    Raises:
        RuntimeError: API 키가 없을 때.
        ImportError: pymupdf 가 없을 때.
        httpx.HTTPStatusError: Gemini API 오류 시.
    """
    # API 키 해결
    resolved_key = api_key
    if resolved_key is None:
        settings = Settings()
        resolved_key = settings.gemini_api_key

    if not resolved_key:
        raise RuntimeError(
            "Gemini API 키가 없습니다. "
            "GEMINI_API_KEY 환경변수를 설정하거나 api_key 인수를 전달하세요."
        )

    logger.debug("VLM 추출 시작: %s page %d", Path(pdf_path).name, page_num)

    image_bytes = render_page_image(pdf_path, page_num)
    markdown = _call_gemini_api(image_bytes, resolved_key)

    logger.debug(
        "VLM 추출 완료: page %d — %d chars", page_num, len(markdown)
    )
    return markdown


# ──────────────────────────────────────────────────────────────────────────────
# 4. ParsedDoc 스플라이싱
# ──────────────────────────────────────────────────────────────────────────────


def apply_vlm_fallback(
    parsed: "ParsedDoc",
    pdf_path: str | Path,
    *,
    pages: list[int] | None = None,
    api_key: str | None = None,
) -> "ParsedDoc":
    """지정한 페이지에 VLM 폴백을 적용해 새 ParsedDoc 를 반환한다.

    Args:
        parsed: 기존 ParsedDoc (변경하지 않음).
        pdf_path: PDF 파일 경로.
        pages: 재추출할 페이지 번호 목록 (1-based).
               None 이면 parsed.low_confidence_pages 를 사용.
        api_key: Gemini API 키. None 이면 settings 에서 로드.

    Returns:
        VLM 결과가 스플라이싱된 새 ParsedDoc.
        대상 페이지가 없으면 parsed 를 그대로 반환.

    스플라이싱 전략:
        - markdown: 문서 끝에 ``## [VLM page N]\\n\\n<vlm_markdown>`` 블록 추가.
        - sections: 같은 페이지의 기존 섹션 뒤에 새 Section(heading="[VLM p.N]") 추가.
        - vlm_pages: 처리된 페이지 번호를 기록 (중복 제거, 정렬).
    """
    from ingestion.parse.types import ParsedDoc, Section

    target_pages = pages if pages is not None else parsed.low_confidence_pages

    if not target_pages:
        logger.debug("VLM 폴백 대상 페이지 없음 — ParsedDoc 변경 없음")
        return parsed

    new_markdown = parsed.markdown
    new_sections = list(parsed.sections)
    new_vlm_pages: list[int] = list(parsed.vlm_pages)

    for page_num in sorted(set(target_pages)):
        logger.info("VLM 폴백 처리: page %d", page_num)

        vlm_md = vlm_extract_page(pdf_path, page_num, api_key=api_key)

        # markdown 스플라이싱 — 끝에 구분 블록 추가
        block_heading = f"## [VLM page {page_num}]"
        new_markdown = f"{new_markdown}\n\n{block_heading}\n\n{vlm_md}"

        # sections 스플라이싱 — 새 섹션 추가
        vlm_section = Section(
            heading=f"[VLM p.{page_num}]",
            text=vlm_md,
            page=page_num,
            level=2,
        )
        new_sections.append(vlm_section)

        # vlm_pages 기록
        if page_num not in new_vlm_pages:
            new_vlm_pages.append(page_num)

    new_vlm_pages = sorted(new_vlm_pages)

    new_parsed = parsed.model_copy(
        update={
            "markdown": new_markdown,
            "sections": new_sections,
            "vlm_pages": new_vlm_pages,
        }
    )

    logger.info(
        "VLM 폴백 완료: %d pages 처리 — vlm_pages=%s",
        len(new_vlm_pages),
        new_vlm_pages,
    )
    return new_parsed
