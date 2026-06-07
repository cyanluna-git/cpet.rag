"""ingestion.build_chunks — 청킹 + 임베딩 파이프라인 연결 모듈.

#3118(load) · #3119(pipeline) 에서 호출하는 단일 진입점.
"""

from __future__ import annotations

from core.interfaces import Embedder
from core.models import Chunk, Paper
from ingestion.chunk import chunk_document
from ingestion.parse.types import ParsedDoc


def parsed_to_embedded_chunks(
    parsed: ParsedDoc,
    paper: Paper,
    embedder: Embedder,
    *,
    target_tokens: int = 512,
    max_tokens: int = 1000,
    overlap_tokens: int = 64,
    use_late_chunking: bool = True,
) -> list[Chunk]:
    """ParsedDoc → 임베딩 완성 Chunk 목록.

    흐름:
        1. ``chunk_document`` 로 구조 인식 청킹 (text, ctx_text 생성).
        2. ``embedder.embed_chunks`` 로 임베딩 설정.
           - ``use_late_chunking=True`` (기본): parsed.markdown 을 full_document 로
             전달해 Late Chunking 활성화.
           - ``use_late_chunking=False``: ctx_text 기반 표준 임베딩.

    Args:
        parsed: Docling 파싱 결과 (sections, markdown 포함).
        paper: 논문 서지 정보.
        embedder: core.interfaces.Embedder 구현체.
        target_tokens: 청크 목표 토큰 수.
        max_tokens: 청크 최대 토큰 수 (hard cap).
        overlap_tokens: 분할 시 overlap 토큰 수.
        use_late_chunking: True 이면 embed_late 경로, False 이면 표준 embed.

    Returns:
        embedding 이 채워진 Chunk 목록.
    """
    chunks = chunk_document(
        parsed,
        paper,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    if not chunks:
        return []

    full_document: str | None = parsed.markdown if use_late_chunking else None
    return embedder.embed_chunks(chunks, full_document=full_document, late=use_late_chunking)
