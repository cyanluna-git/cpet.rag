"""tests/test_embedder.py — JinaEmbedder 단위 테스트 (API call mocked).

실제 Jina API / GPU 없이 테스트 가능하다.
_embed_call / _embed_late_call 을 mock 해 결정론적 벡터를 반환한다.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core import interfaces
from core.models import Chunk, Paper
from ingestion.embed import JinaEmbedder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_DIM = 1024
PAPER = Paper(
    doi="10.5678/embed.test",
    title="CPET Embedder Test Paper",
    first_author="Park",
    year=2024,
    journal="Test Journal",
    source="embed_test",
    openalex_id="W9999",
)


def _make_vector(seed: int, dim: int = EMBED_DIM) -> list[float]:
    """결정론적 테스트 벡터를 생성한다."""
    return [(seed * 0.001 + i * 0.0001) % 1.0 for i in range(dim)]


def _make_chunk(idx: int, text: str = "Sample text for embedding.") -> Chunk:
    ctx = (
        f"[{PAPER.title} · {PAPER.first_author} {PAPER.year}"
        f" · {PAPER.journal} · §Section{idx}]\n{text}"
    )
    return Chunk(
        id=f"W9999_{idx}",
        doi=PAPER.doi,
        section=f"Section{idx}",
        text=text,
        ctx_text=ctx,
        page=idx + 1,
        chunk_index=idx,
        source=PAPER.source,
        embedding=None,
    )


# ---------------------------------------------------------------------------
# isinstance check
# ---------------------------------------------------------------------------


def test_isinstance_embedder_protocol() -> None:
    """JinaEmbedder() 는 core.interfaces.Embedder 를 만족한다."""
    embedder = JinaEmbedder()
    assert isinstance(embedder, interfaces.Embedder)


def test_zero_arg_construction() -> None:
    """JinaEmbedder() 는 API 키 없이도 생성 가능하다."""
    embedder = JinaEmbedder()
    assert embedder.dim == EMBED_DIM
    assert embedder.model == "jinaai/jina-embeddings-v3"


def test_custom_dim() -> None:
    """dim 파라미터가 적용된다."""
    embedder = JinaEmbedder(dim=256)
    assert embedder.dim == 256


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------


def test_embed_returns_correct_shape() -> None:
    """embed 는 입력 수와 동일한 벡터 수, 각 dim 길이를 반환한다."""
    texts = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
    fake_vecs = [_make_vector(i) for i in range(len(texts))]

    embedder = JinaEmbedder()
    with patch.object(embedder, "_embed_call", return_value=fake_vecs) as mock_call:
        result = embedder.embed(texts)

    mock_call.assert_called_once_with(texts)
    assert len(result) == len(texts)
    for vec in result:
        assert len(vec) == EMBED_DIM


def test_embed_single_text() -> None:
    """단일 텍스트도 정상 처리된다."""
    texts = ["single sentence"]
    fake_vecs = [_make_vector(0)]

    embedder = JinaEmbedder()
    with patch.object(embedder, "_embed_call", return_value=fake_vecs):
        result = embedder.embed(texts)

    assert len(result) == 1
    assert len(result[0]) == EMBED_DIM


def test_embed_empty_list() -> None:
    """빈 리스트를 전달하면 빈 리스트를 반환한다."""
    embedder = JinaEmbedder()
    with patch.object(embedder, "_embed_call", return_value=[]) as mock_call:
        result = embedder.embed([])

    assert result == []


# ---------------------------------------------------------------------------
# embed_late()
# ---------------------------------------------------------------------------


def test_embed_late_returns_one_vector_per_boundary() -> None:
    """embed_late 는 각 boundary 마다 벡터 하나를 반환한다."""
    # document = "alpha\nbeta\ngamma"
    parts = ["alpha", "beta", "gamma"]
    document = "\n".join(parts)
    boundaries: list[tuple[int, int]] = []
    pos = 0
    for t in parts:
        boundaries.append((pos, pos + len(t)))
        pos += len(t) + 1  # "\n"

    fake_vecs = [_make_vector(i) for i in range(len(boundaries))]

    embedder = JinaEmbedder()
    with patch.object(embedder, "_embed_late_call", return_value=fake_vecs) as mock_call:
        result = embedder.embed_late(document, boundaries)

    # _embed_late_call 은 document[s:e] 로 복원된 texts 를 받는다
    expected_texts = [document[s:e] for s, e in boundaries]
    mock_call.assert_called_once_with(expected_texts)

    assert len(result) == len(boundaries)
    for vec in result:
        assert len(vec) == EMBED_DIM


def test_embed_late_char_boundaries_correct() -> None:
    """document[s:e] 가 각 chunk text 를 정확히 복원한다."""
    texts = ["first chunk text", "second chunk text", "third chunk text"]
    sep = "\n"
    document = sep.join(texts)

    boundaries: list[tuple[int, int]] = []
    pos = 0
    for i, t in enumerate(texts):
        boundaries.append((pos, pos + len(t)))
        pos += len(t) + (1 if i < len(texts) - 1 else 0)

    for i, (s, e) in enumerate(boundaries):
        assert (
            document[s:e] == texts[i]
        ), f"Boundary mismatch at {i}: {document[s:e]!r} != {texts[i]!r}"


# ---------------------------------------------------------------------------
# embed_chunks()
# ---------------------------------------------------------------------------


def test_embed_chunks_standard_path_sets_embedding() -> None:
    """full_document=None 이면 standard path; 각 Chunk.embedding 이 설정된다."""
    chunks = [_make_chunk(i) for i in range(3)]
    fake_vecs = [_make_vector(i) for i in range(len(chunks))]

    embedder = JinaEmbedder()
    with patch.object(embedder, "_embed_call", return_value=fake_vecs):
        result = embedder.embed_chunks(chunks, full_document=None, late=False)

    assert len(result) == len(chunks)
    for i, c in enumerate(result):
        assert c.embedding is not None
        assert len(c.embedding) == EMBED_DIM


def test_embed_chunks_non_mutating() -> None:
    """원본 Chunk 는 변경되지 않는다 (non-mutating)."""
    chunks = [_make_chunk(0)]
    fake_vecs = [_make_vector(0)]

    embedder = JinaEmbedder()
    with patch.object(embedder, "_embed_call", return_value=fake_vecs):
        result = embedder.embed_chunks(chunks, full_document=None, late=False)

    assert chunks[0].embedding is None, "Original chunk should not be mutated"
    assert result[0].embedding is not None


def test_embed_chunks_late_path_calls_embed_late() -> None:
    """full_document 제공 시 embed_late 경로를 사용한다."""
    chunks = [_make_chunk(i) for i in range(4)]
    fake_vecs = [_make_vector(i) for i in range(len(chunks))]

    embedder = JinaEmbedder()
    with (
        patch.object(embedder, "_embed_late_call", return_value=fake_vecs) as mock_late,
        patch.object(embedder, "_embed_call") as mock_std,
    ):
        result = embedder.embed_chunks(chunks, full_document="full doc text", late=True)

    mock_late.assert_called_once()
    mock_std.assert_not_called()

    assert len(result) == len(chunks)
    for c in result:
        assert c.embedding is not None
        assert len(c.embedding) == EMBED_DIM


def test_embed_chunks_late_false_uses_standard() -> None:
    """full_document 가 있어도 late=False 이면 standard path를 사용한다."""
    chunks = [_make_chunk(0), _make_chunk(1)]
    fake_vecs = [_make_vector(i) for i in range(len(chunks))]

    embedder = JinaEmbedder()
    with (
        patch.object(embedder, "_embed_call", return_value=fake_vecs) as mock_std,
        patch.object(embedder, "_embed_late_call") as mock_late,
    ):
        result = embedder.embed_chunks(chunks, full_document="doc", late=False)

    mock_std.assert_called_once()
    mock_late.assert_not_called()


def test_embed_chunks_empty_returns_empty() -> None:
    """빈 chunks 리스트는 빈 리스트를 반환한다."""
    embedder = JinaEmbedder()
    result = embedder.embed_chunks([], full_document=None)
    assert result == []


def test_embed_chunks_standard_uses_ctx_text() -> None:
    """standard path 에서 ctx_text 를 embed_call 에 전달한다."""
    chunks = [_make_chunk(i, text=f"text {i}") for i in range(2)]
    fake_vecs = [_make_vector(i) for i in range(len(chunks))]

    embedder = JinaEmbedder()
    with patch.object(embedder, "_embed_call", return_value=fake_vecs) as mock_std:
        embedder.embed_chunks(chunks, full_document=None, late=False)

    called_texts = mock_std.call_args[0][0]
    for chunk, called_text in zip(chunks, called_texts):
        assert called_text == chunk.ctx_text


# ---------------------------------------------------------------------------
# MRL dim truncation
# ---------------------------------------------------------------------------


def test_embed_chunks_mrl_dim_truncation() -> None:
    """embed_dim < API 반환 dim 이면 벡터가 절단된다."""
    chunks = [_make_chunk(0)]
    # API 가 2048 dim 을 반환한다고 가정
    full_vec = [0.5] * 2048
    embedder = JinaEmbedder(dim=256)

    with patch.object(embedder, "_embed_call", return_value=[full_vec]):
        result = embedder.embed_chunks(chunks, full_document=None, late=False)

    assert len(result[0].embedding) == 256  # type: ignore[arg-type]
