"""core.models.chunk — 검색 단위(청크) 및 검색 결과 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """벡터스토어에 저장되는 검색 단위."""

    id: str  # 전역 유일 식별자 (예: "{doi}::{chunk_index}")
    doi: str | None = None  # 부모 논문 DOI
    section: str | None = None  # 섹션명 (예: "Introduction", "Methods")
    text: str  # 원문 텍스트
    ctx_text: str  # 메타 접두어 포함 임베딩용 텍스트
    page: int | None = None  # 원본 PDF 페이지 번호
    chunk_index: int  # 논문 내 청크 순서 (0-based)
    embedding: list[float] | None = None  # Jina-v3 임베딩 벡터
    source: str | None = None  # 논문 source 식별자 (corpus_index.csv 참조)


class RetrievedChunk(BaseModel):
    """하이브리드/벡터 검색 결과 — 청크 + 점수."""

    chunk: Chunk
    score: float  # 하이브리드·벡터 유사도 점수
    rerank_score: float | None = None  # Bedrock Reranker 재순위 점수
