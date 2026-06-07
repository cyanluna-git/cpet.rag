"""tests/test_load.py — load_chunks 및 ProcessedRegistry 단위 테스트.

실제 LanceDB (임시 디렉터리) 사용 — 외부 API 의존 없음.
``uv run --extra vectorstore pytest tests/test_load.py -q`` 로 실행.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.models import Chunk, Paper
from ingestion.load import ProcessedRegistry, load_chunks, processed_key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 4  # 테스트용 소형 차원


def _vec(seed: int, dim: int = DIM) -> list[float]:
    """결정론적 단위 벡터 (임베딩 대용)."""
    v = [(seed + i + 1) * 0.1 for i in range(dim)]
    norm = sum(x**2 for x in v) ** 0.5
    return [x / norm for x in v]


def _chunk(idx: int, *, doi: str = "10.9999/test", with_embedding: bool = True) -> Chunk:
    return Chunk(
        id=f"{doi}::{idx}",
        doi=doi,
        section="Methods",
        text=f"chunk text {idx}",
        ctx_text=f"[ctx] chunk text {idx}",
        page=idx + 1,
        chunk_index=idx,
        source="test_source",
        embedding=_vec(idx) if with_embedding else None,
    )


def _store(tmp_path: Path, dim: int = DIM):
    """임시 LanceDBStore 를 생성한다."""
    from core.vectorstore import LanceDBStore

    return LanceDBStore(uri=str(tmp_path / "lancedb"), dim=dim)


# ---------------------------------------------------------------------------
# load_chunks
# ---------------------------------------------------------------------------


class TestLoadChunks:
    def test_basic_load_returns_count(self, tmp_path: Path) -> None:
        """정상 임베딩 청크를 적재하면 적재 수를 반환한다."""
        store = _store(tmp_path)
        chunks = [_chunk(i) for i in range(5)]

        count = load_chunks(chunks, store=store)

        assert count == 5
        assert store.count() == 5

    def test_load_empty_returns_zero(self, tmp_path: Path) -> None:
        """빈 리스트를 전달하면 0 을 반환한다."""
        store = _store(tmp_path)
        assert load_chunks([], store=store) == 0
        assert store.count() == 0

    def test_upsert_idempotent(self, tmp_path: Path) -> None:
        """같은 id 청크를 두 번 적재하면 중복 없이 upsert 된다."""
        store = _store(tmp_path)
        chunks = [_chunk(i) for i in range(3)]

        load_chunks(chunks, store=store)
        load_chunks(chunks, store=store)  # 동일 id 재적재

        assert store.count() == 3  # 중복 없음

    def test_missing_embedding_raises(self, tmp_path: Path) -> None:
        """embedding=None 인 청크가 포함되면 ValueError 를 발생시킨다."""
        store = _store(tmp_path)
        chunks = [_chunk(0), _chunk(1, with_embedding=False), _chunk(2)]

        with pytest.raises(ValueError, match="embedding=None"):
            load_chunks(chunks, store=store)

    def test_missing_embedding_error_includes_id(self, tmp_path: Path) -> None:
        """ValueError 메시지에 문제 청크 id 가 포함된다."""
        store = _store(tmp_path)
        bad_chunk = _chunk(99, with_embedding=False)

        with pytest.raises(ValueError) as exc_info:
            load_chunks([bad_chunk], store=store)

        assert bad_chunk.id in str(exc_info.value)

    def test_all_missing_embeddings_raise_before_upsert(self, tmp_path: Path) -> None:
        """embedding 검사는 upsert 이전에 수행된다 — store 에 아무것도 없어야 한다."""
        store = _store(tmp_path)
        chunks = [_chunk(i, with_embedding=False) for i in range(3)]

        with pytest.raises(ValueError):
            load_chunks(chunks, store=store)

        assert store.count() == 0  # upsert 미실행

    def test_partial_update_extends_store(self, tmp_path: Path) -> None:
        """추가 청크 적재 시 기존 청크와 합산된다."""
        store = _store(tmp_path)
        load_chunks([_chunk(0), _chunk(1)], store=store)
        load_chunks([_chunk(2), _chunk(3)], store=store)

        assert store.count() == 4


# ---------------------------------------------------------------------------
# ProcessedRegistry
# ---------------------------------------------------------------------------


class TestProcessedRegistry:
    def test_mark_and_is_processed(self, tmp_path: Path) -> None:
        """mark_processed 후 is_processed 가 True 를 반환한다."""
        reg = ProcessedRegistry(path=tmp_path / "processed.jsonl")
        reg.mark_processed("W1234", embed_version="jina-v3@1024", n_chunks=10)

        assert reg.is_processed("W1234") is True
        assert reg.is_processed("W1234", embed_version="jina-v3@1024") is True

    def test_unknown_key_returns_false(self, tmp_path: Path) -> None:
        """기록되지 않은 키는 False 를 반환한다."""
        reg = ProcessedRegistry(path=tmp_path / "processed.jsonl")
        assert reg.is_processed("NOTEXIST") is False

    def test_embed_version_mismatch_returns_false(self, tmp_path: Path) -> None:
        """embed_version 이 다르면 False 를 반환한다 (재적재 강제)."""
        reg = ProcessedRegistry(path=tmp_path / "processed.jsonl")
        reg.mark_processed("W5678", embed_version="jina-v3@1024")

        assert reg.is_processed("W5678", embed_version="jina-v3@256") is False

    def test_content_hash_mismatch_returns_false(self, tmp_path: Path) -> None:
        """content_hash 가 다르면 False 를 반환한다."""
        reg = ProcessedRegistry(path=tmp_path / "processed.jsonl")
        reg.mark_processed("W9000", content_hash="abc123", embed_version="jina-v3@1024")

        assert reg.is_processed("W9000", content_hash="def456") is False

    def test_persistence_across_reload(self, tmp_path: Path) -> None:
        """save → 새 인스턴스 load 후에도 레코드가 유지된다."""
        path = tmp_path / "processed.jsonl"

        reg1 = ProcessedRegistry(path=path)
        reg1.mark_processed("W_PERSIST", embed_version="v1", n_chunks=7)
        reg1.save()

        reg2 = ProcessedRegistry(path=path)
        assert reg2.is_processed("W_PERSIST") is True
        assert reg2.is_processed("W_PERSIST", embed_version="v1") is True

    def test_append_persists_without_explicit_save(self, tmp_path: Path) -> None:
        """mark_processed 는 파일에 즉시 append 하므로 save() 없이도 지속된다."""
        path = tmp_path / "processed.jsonl"

        reg1 = ProcessedRegistry(path=path)
        reg1.mark_processed("W_APPEND", embed_version="v1")
        # save() 미호출

        reg2 = ProcessedRegistry(path=path)
        reg2.load()
        assert reg2.is_processed("W_APPEND", embed_version="v1") is True

    def test_all_processed_returns_all_keys(self, tmp_path: Path) -> None:
        """all_processed() 가 모든 키를 반환한다."""
        reg = ProcessedRegistry(path=tmp_path / "processed.jsonl")
        reg.mark_processed("K1")
        reg.mark_processed("K2")
        reg.mark_processed("K3")

        keys = reg.all_processed()
        assert keys == {"K1", "K2", "K3"}

    def test_reload_last_write_wins(self, tmp_path: Path) -> None:
        """동일 key 를 두 번 mark 하면 마지막 값이 우선한다."""
        path = tmp_path / "processed.jsonl"

        reg1 = ProcessedRegistry(path=path)
        reg1.mark_processed("DUP", embed_version="v1")
        reg1.mark_processed("DUP", embed_version="v2")

        reg2 = ProcessedRegistry(path=path)
        assert reg2.is_processed("DUP", embed_version="v2") is True
        assert reg2.is_processed("DUP", embed_version="v1") is False

    def test_save_deduplicates_file(self, tmp_path: Path) -> None:
        """save() 는 동일 key 중복 없이 단일 행만 기록한다."""
        path = tmp_path / "processed.jsonl"

        reg = ProcessedRegistry(path=path)
        reg.mark_processed("DEDUP", embed_version="v1")
        reg.mark_processed("DEDUP", embed_version="v2")
        reg.save()

        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["embed_version"] == "v2"

    def test_empty_registry_no_file_needed(self, tmp_path: Path) -> None:
        """파일이 없어도 인스턴스를 생성할 수 있다."""
        path = tmp_path / "nonexistent" / "processed.jsonl"
        reg = ProcessedRegistry(path=path)
        assert reg.is_processed("X") is False


# ---------------------------------------------------------------------------
# processed_key
# ---------------------------------------------------------------------------


class TestProcessedKey:
    def _paper(
        self,
        openalex_id: str | None = None,
        doi: str | None = None,
        file: str | None = None,
        source: str = "fallback_source",
    ) -> Paper:
        return Paper(
            title="Test",
            source=source,
            openalex_id=openalex_id,
            doi=doi,
            file=file,
        )

    def test_prefers_openalex_id(self) -> None:
        paper = self._paper(openalex_id="W999", doi="10.1/abc", file="file.pdf")
        assert processed_key(paper) == "W999"

    def test_falls_back_to_doi(self) -> None:
        paper = self._paper(doi="10.1234/abc", file="file.pdf")
        assert processed_key(paper) == "10.1234/abc"

    def test_normalizes_doi(self) -> None:
        paper = self._paper(doi="https://doi.org/10.1234/ABC")
        # normalize_doi 는 소문자 반환
        assert processed_key(paper) == "10.1234/abc"

    def test_falls_back_to_file(self) -> None:
        paper = self._paper(file="papers/foo.pdf")
        assert processed_key(paper) == "papers/foo.pdf"

    def test_falls_back_to_source(self) -> None:
        paper = self._paper(source="my_source")
        assert processed_key(paper) == "my_source"
