"""core.models.paper — 논문 서지 레코드 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """학술 논문 서지 정보. corpus_index.csv 컬럼 + OpenAlex 보강 필드."""

    # corpus_index.csv 컬럼
    doi: str | None = None
    title: str
    first_author: str | None = None
    year: int | None = None
    journal: str | None = None
    source: str  # corpus_index.csv 파일명 기반 식별자 (중복 키 보조)
    file: str | None = None  # 원본 PDF 경로 (S3 또는 로컬 상대 경로)
    oa_status: str | None = None
    added_by: str | None = None
    added_at: str | None = None

    # OpenAlex 보강 필드
    openalex_id: str | None = None
    authors: list[str] = Field(default_factory=list)  # 전체 저자 목록
    abstract: str | None = None
