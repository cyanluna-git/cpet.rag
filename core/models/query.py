"""core.models.query — 질의 요청/응답 및 인용 스키마."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.models.chunk import RetrievedChunk


class Citation(BaseModel):
    """엄격 인용 검증용 인용 레코드.

    answer 내 인용 태그는 반드시 retrieved chunk 원문과 overlap 검증을 통과해야 한다.
    """

    doi: str | None = None
    title: str | None = None
    page: int | None = None
    chunk_id: str  # 인용 근거 Chunk.id
    quote: str  # 검색 청크에서 추출한 정확한 지지 문장


class QueryRequest(BaseModel):
    """RAG 파이프라인 질의 요청."""

    query: str  # 한국어 질문
    top_k: int = 8
    filters: dict[str, Any] | None = None  # 연도·저자·저널 필터
    translate: bool = True  # Query Translation 샌드위치 활성화 여부


class QueryResponse(BaseModel):
    """RAG 파이프라인 질의 응답."""

    answer: str  # 한국어 답변
    answer_en: str | None = None  # 영문 중간 답변 (디버깅·평가용)
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
