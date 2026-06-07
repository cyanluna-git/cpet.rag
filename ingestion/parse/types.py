"""ingestion.parse.types — PDF 파싱 출력 스키마.

ParsedDoc 과 Section 은 다운스트림 #3116(VLM 폴백) · #3117(Late Chunking) 에서
직접 참조하는 공개 계약이다.  필드 구조를 바꿀 때는 하위 태스크 영향을 먼저 확인할 것.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Section(BaseModel):
    """섹션 헤더에서 다음 헤더 직전까지의 텍스트 블록.

    #3117(Late Chunking) 이 섹션 경계(heading + page)를 사용한다.
    """

    heading: str | None = None
    """섹션 제목. 첫 번째 헤더 이전 서문(preamble)은 None."""

    text: str
    """이 섹션에 속하는 본문 텍스트를 연결한 문자열."""

    page: int | None = None
    """헤딩이 등장한 PDF 페이지 번호 (1-based). Docling prov 에서 추출."""

    level: int | None = None
    """섹션 계층. Docling SectionHeaderItem.level 값(1=최상위).
    TitleItem 등 level 을 갖지 않는 헤더는 None."""


class TableRef(BaseModel):
    """파싱된 표의 위치 참조.

    #3116(VLM 폴백) 이 실패한 표를 재처리할 때 page 를 활용한다.
    """

    index: int
    """문서 내 표 순서 (0-based)."""

    page: int | None = None
    """표가 위치한 PDF 페이지 번호 (1-based)."""

    caption: str | None = None
    """Docling 이 추출한 표 캡션 텍스트 (없으면 None)."""


class ParsedDoc(BaseModel):
    """Docling 으로 파싱된 단일 학술 PDF 의 구조화 표현.

    downstream 계약:
    - #3116 VLM 폴백: low_confidence_pages, tables 를 참조해 재파싱 대상을 결정한다.
    - #3117 Late Chunking: sections 를 순회해 섹션 경계(heading, page, level)를 사용한다.
    """

    source_path: str
    """파싱에 사용된 PDF 파일의 경로 (절대 경로 권장)."""

    markdown: str
    """Docling export_to_markdown() 로 생성한 구조 보존 마크다운 전문."""

    n_pages: int
    """PDF 전체 페이지 수 (Docling result.document.pages dict 길이)."""

    sections: list[Section] = Field(default_factory=list)
    """제목 계층 구조로 분할된 섹션 목록. 서문(heading=None)이 첫 번째로 올 수 있다."""

    tables: list[TableRef] = Field(default_factory=list)
    """문서 내 모든 표의 참조 목록 (위치·캡션 포함)."""

    low_confidence_pages: list[int] = Field(default_factory=list)
    """Docling 신뢰도 점수가 낮거나 텍스트가 없는 페이지 번호 목록 (1-based).
    VLM 폴백 후보 페이지. NaN 점수(빈 페이지)도 포함된다.

    TODO: LOW_CONFIDENCE_THRESHOLD(현재 0.5)는 코퍼스 특성에 따라 조정 필요.
    신뢰도 지표는 parse_score, layout_score 중 낮은 값을 기준으로 사용한다.
    """

    vlm_pages: list[int] = Field(default_factory=list)
    """VLM 폴백으로 재추출된 페이지 번호 목록 (1-based).
    apply_vlm_fallback() 이 처리한 페이지가 여기 기록된다.
    """
