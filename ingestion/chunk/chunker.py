"""ingestion.chunk.chunker — 구조 인식 청킹 (Late Chunking 전처리).

## 설계 결정

### 토큰 카운팅
tiktoken(cl100k_base)을 lazy import로 시도한다. unavailable 시 휴리스틱
``ceil(len(text.split()) * 1.3)`` 로 폴백 (영어 학술 논문 기준 BPE 오버헤드 근사).
cap 검사는 `text` 기준으로 수행한다 (ctx_text 는 메타 접두어가 포함되어 더 길지만,
임베딩 입력 품질 문제이며 청킹 경계 결정은 본문 기준이 직관적).

### 섹션 경계 정책
- sections 를 flat 순서로 순회한다 (level 은 unreliable).
- 단일 섹션이 max_tokens 를 초과하면 overlap_tokens 슬라이딩 윈도우로 분할.
- 인접 소형 섹션 병합: 현재 청크 누적이 target_tokens 미만이고 다음 섹션도 작을 때
  동일 청크로 합산한다. 두 섹션의 합계가 target_tokens 를 초과하면 병합하지 않는다.
  (병합 단위에 걸친 overlap 은 적용하지 않는다 — 섹션 경계 보존 우선)

### id 포맷
``{openalex_id or slug(doi) or 'nd'}_{chunk_index}``
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from core.log import get_logger
from core.models import Chunk, Paper
from ingestion.parse.types import ParsedDoc, Section

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_TIKTOKEN_ENC: object | None = None  # cached encoder or False
_TIKTOKEN_TRIED = False


def count_tokens(text: str) -> int:
    """text 의 토큰 수를 추정한다.

    tiktoken(cl100k_base) 사용 가능 시 정확값, 불가 시 단어 수 기반 근사값.
    근사식: ceil(word_count * 1.3)  — GPT-4/Jina tokeniser BPE 오버헤드 보정.
    """
    global _TIKTOKEN_ENC, _TIKTOKEN_TRIED
    if not _TIKTOKEN_TRIED:
        _TIKTOKEN_TRIED = True
        try:
            import tiktoken  # type: ignore[import]

            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
            logger.debug("count_tokens: tiktoken cl100k_base loaded")
        except ImportError:
            _TIKTOKEN_ENC = False  # type: ignore[assignment]
            logger.debug("count_tokens: tiktoken unavailable — using heuristic")

    if _TIKTOKEN_ENC:
        return len(_TIKTOKEN_ENC.encode(text))  # type: ignore[union-attr]
    # Heuristic fallback
    return math.ceil(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# ID helper
# ---------------------------------------------------------------------------


def _slug(doi: str | None) -> str | None:
    """DOI 에서 파일 시스템 안전 슬러그를 생성한다."""
    if not doi:
        return None
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", doi)
    return slug[:64]  # 과도하게 길어지는 것 방지


def _make_id(paper: Paper, chunk_index: int) -> str:
    base = paper.openalex_id or _slug(paper.doi) or "nd"
    return f"{base}_{chunk_index}"


# ---------------------------------------------------------------------------
# Context-prefixed text builder
# ---------------------------------------------------------------------------


def _ctx_text(paper: Paper, section_heading: str | None, text: str) -> str:
    """메타데이터 접두어를 포함한 임베딩 입력 텍스트를 반환한다."""
    return (
        f"[{paper.title} · {paper.first_author or ''} {paper.year or ''}"
        f" · {paper.journal or ''} · §{section_heading or 'body'}]\n{text}"
    )


# ---------------------------------------------------------------------------
# Split single long section into overlap-windowed chunks
# ---------------------------------------------------------------------------


def _split_section(
    section: Section,
    paper: Paper,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    start_index: int,
) -> list[Chunk]:
    """max_tokens 를 초과하는 단일 섹션을 슬라이딩 윈도우로 분할한다.

    단어 단위 슬라이드 (토큰 근사 기반).
    overlap 은 이전 청크의 마지막 overlap_tokens 분량 단어를 재사용한다.
    """
    words = section.text.split()
    if not words:
        return []

    # words-per-token 역수 추정 (tiktoken 사용 불가일 경우 근사값 사용)
    sample = " ".join(words[:200]) if len(words) >= 200 else section.text
    sample_tokens = count_tokens(sample)
    sample_words = len(sample.split())
    words_per_token: float = (sample_words / sample_tokens) if sample_tokens > 0 else 0.77

    target_words = max(1, int(target_tokens * words_per_token))
    overlap_words = max(0, int(overlap_tokens * words_per_token))

    chunks: list[Chunk] = []
    i = 0
    chunk_index = start_index

    while i < len(words):
        window = words[i : i + target_words]
        text = " ".join(window)

        # 재측정 후 hard cap 보정 (근사 오류 보상)
        while count_tokens(text) > max_tokens and len(window) > 1:
            window = window[:-1]
            text = " ".join(window)

        chunk = Chunk(
            id=_make_id(paper, chunk_index),
            doi=paper.doi,
            section=section.heading,
            text=text,
            ctx_text=_ctx_text(paper, section.heading, text),
            page=section.page,
            chunk_index=chunk_index,
            source=paper.source,
            embedding=None,
        )
        chunks.append(chunk)
        chunk_index += 1

        advance = max(1, len(window) - overlap_words)
        i += advance

    return chunks


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def chunk_document(
    parsed: ParsedDoc,
    paper: Paper,
    *,
    target_tokens: int = 512,
    max_tokens: int = 1000,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """ParsedDoc 을 Chunk 목록으로 변환한다.

    Args:
        parsed: 파싱된 문서 (Section 목록 포함).
        paper: 부모 논문 서지 정보.
        target_tokens: 청크 목표 토큰 수 (split 기준).
        max_tokens: 청크 최대 토큰 수 (hard cap, text 기준).
        overlap_tokens: 분할 시 이전 청크 재사용 토큰 수.

    Returns:
        Chunk 목록 (chunk_index 0-based 연속).
    """
    sections = parsed.sections
    if not sections:
        logger.warning("chunk_document: ParsedDoc.sections 가 비어있습니다 (doi=%s)", paper.doi)
        return []

    chunks: list[Chunk] = []
    chunk_index = 0

    # 소형 섹션 병합 버퍼
    merge_buffer: list[Section] = []
    merge_token_count = 0

    def _flush_merge_buffer() -> None:
        nonlocal merge_buffer, merge_token_count, chunk_index
        if not merge_buffer:
            return

        combined_text = "\n\n".join(s.text for s in merge_buffer if s.text.strip())
        if not combined_text.strip():
            merge_buffer = []
            merge_token_count = 0
            return

        # 병합된 헤딩: 첫 번째 섹션의 heading 사용
        heading = merge_buffer[0].heading
        page = merge_buffer[0].page

        chunk = Chunk(
            id=_make_id(paper, chunk_index),
            doi=paper.doi,
            section=heading,
            text=combined_text,
            ctx_text=_ctx_text(paper, heading, combined_text),
            page=page,
            chunk_index=chunk_index,
            source=paper.source,
            embedding=None,
        )
        chunks.append(chunk)
        chunk_index += 1
        merge_buffer = []
        merge_token_count = 0

    for section in sections:
        text = section.text.strip() if section.text else ""
        if not text:
            continue

        token_count = count_tokens(text)

        if token_count > max_tokens:
            # 먼저 버퍼를 플러시하고, 긴 섹션은 분할
            _flush_merge_buffer()
            new_chunks = _split_section(
                section,
                paper,
                target_tokens=target_tokens,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                start_index=chunk_index,
            )
            chunks.extend(new_chunks)
            chunk_index += len(new_chunks)

        elif token_count <= target_tokens:
            # 소형 섹션: 버퍼에 누적 (다음 섹션과 병합 가능)
            if merge_token_count + token_count > target_tokens:
                # 병합하면 target 초과 → 먼저 플러시
                _flush_merge_buffer()

            merge_buffer.append(section)
            merge_token_count += token_count

        else:
            # target_tokens < token_count <= max_tokens: 단일 청크
            _flush_merge_buffer()

            chunk = Chunk(
                id=_make_id(paper, chunk_index),
                doi=paper.doi,
                section=section.heading,
                text=text,
                ctx_text=_ctx_text(paper, section.heading, text),
                page=section.page,
                chunk_index=chunk_index,
                source=paper.source,
                embedding=None,
            )
            chunks.append(chunk)
            chunk_index += 1

    # 남은 버퍼 플러시
    _flush_merge_buffer()

    logger.info(
        "chunk_document: %d 청크 생성 (doi=%s, sections=%d)",
        len(chunks),
        paper.doi,
        len(sections),
    )
    return chunks
