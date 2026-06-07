"""core.vectorstore.lancedb_store — LanceDB 기반 VectorStore 구현.

local 디렉터리(./data/lancedb) 또는 S3 URI(s3://...) 모두 지원.
core.interfaces.VectorStore Protocol 을 만족한다.
"""

from __future__ import annotations

import logging
from typing import Any

import pyarrow as pa

from core.config.settings import Settings
from core.log import get_logger
from core.models import Chunk, RetrievedChunk

_settings = Settings()

# 벡터 컬럼 거리 지표 — cosine 유사도 사용 (Jina-v3 최적화)
_METRIC = "cosine"


def _score_from_distance(distance: float) -> float:
    """cosine distance → similarity (0~1). distance = 1 - cosine_similarity."""
    return max(0.0, 1.0 - distance)


class LanceDBStore:
    """LanceDB-backed VectorStore.

    Parameters
    ----------
    uri:
        DB 경로 또는 S3 URI. None 이면 settings.lancedb_uri → ./data/lancedb 순서로 폴백.
    table:
        테이블 이름. 기본 "chunks".
    dim:
        임베딩 차원 수. None 이면 settings.embed_dim(1024) 사용.
    """

    def __init__(
        self,
        uri: str | None = None,
        table: str = "chunks",
        dim: int | None = None,
    ) -> None:
        import lancedb  # lazy import — vectorstore extra 없으면 ImportError 명시

        self._logger: logging.Logger = get_logger(__name__)

        resolved_uri: str = uri or _settings.lancedb_uri or "./data/lancedb"
        self._dim: int = dim if dim is not None else _settings.embed_dim
        self._table_name: str = table

        self._logger.debug("Connecting to LanceDB at %s (dim=%d)", resolved_uri, self._dim)
        self._db: Any = lancedb.connect(resolved_uri)

        # PyArrow 스키마 — 추론 대신 명시적 정의 (None 컬럼도 타입 보장)
        self._schema: pa.Schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("doi", pa.string()),
                pa.field("section", pa.string()),
                pa.field("text", pa.string()),
                pa.field("ctx_text", pa.string()),
                pa.field("page", pa.int32()),
                pa.field("chunk_index", pa.int32()),
                pa.field("source", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self._dim)),
            ]
        )

        self._tbl: Any = self._open_or_create_table()
        self._logger.info(
            "LanceDBStore ready — table=%s, rows=%d", self._table_name, self.count()
        )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _open_or_create_table(self) -> Any:
        """테이블이 없으면 생성, 있으면 열어서 반환한다."""
        existing = set(self._db.table_names())
        if self._table_name in existing:
            self._logger.debug("Opening existing table '%s'", self._table_name)
            return self._db.open_table(self._table_name)
        self._logger.info("Creating table '%s'", self._table_name)
        return self._db.create_table(self._table_name, schema=self._schema)

    def _chunks_to_arrow(self, chunks: list[Chunk]) -> pa.Table:
        """Chunk 목록 → PyArrow Table (명시 스키마로 캐스팅)."""
        rows: list[dict[str, Any]] = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk id='{chunk.id}' has embedding=None — embed before upsert."
                )
            rows.append(
                {
                    "id": chunk.id,
                    "doi": chunk.doi or "",
                    "section": chunk.section or "",
                    "text": chunk.text,
                    "ctx_text": chunk.ctx_text,
                    "page": chunk.page if chunk.page is not None else 0,
                    "chunk_index": chunk.chunk_index,
                    "source": chunk.source or "",
                    "vector": chunk.embedding,  # list[float] → float32 list
                }
            )
        return pa.Table.from_pylist(rows, schema=self._schema)

    def _row_to_chunk(self, row: dict[str, Any]) -> Chunk:
        """LanceDB 검색 결과 row → Chunk 복원."""
        return Chunk(
            id=row["id"],
            doi=row.get("doi") or None,
            section=row.get("section") or None,
            text=row["text"],
            ctx_text=row["ctx_text"],
            page=row.get("page") or None,
            chunk_index=row["chunk_index"],
            source=row.get("source") or None,
            embedding=list(row["vector"]) if row.get("vector") is not None else None,
        )

    # ------------------------------------------------------------------
    # VectorStore Protocol 메서드
    # ------------------------------------------------------------------

    def upsert(self, chunks: list[Chunk]) -> None:
        """청크를 upsert(merge-insert on id). FTS 인덱스를 재생성한다.

        Notes
        -----
        - embedding=None 인 청크가 포함되면 ValueError.
        - FTS 인덱스는 upsert 마다 replace=True 로 재구성.
          대규모 인입(#3118)에서는 bulk insert 후 한 번만 호출 권장.
        """
        if not chunks:
            return

        data = self._chunks_to_arrow(chunks)
        self._logger.debug("Upserting %d chunk(s) into table '%s'", len(chunks), self._table_name)

        builder = self._tbl.merge_insert("id")
        builder.when_matched_update_all()
        builder.when_not_matched_insert_all()
        builder.execute(data)

        # FTS 인덱스 재생성 (merge 후 새 row 검색 가능하도록)
        self._tbl.create_fts_index("text", replace=True)
        self._logger.debug("FTS index refreshed on 'text'")

    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """벡터 ANN 검색 → top_k RetrievedChunk 반환."""
        rows = (
            self._tbl.search(vector, vector_column_name="vector")
            .metric(_METRIC)
            .limit(top_k)
            .to_list()
        )
        results: list[RetrievedChunk] = []
        for row in rows:
            distance: float = float(row.get("_distance", 0.0))
            score = _score_from_distance(distance)
            results.append(RetrievedChunk(chunk=self._row_to_chunk(row), score=score))
        return results

    def fts(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Full-text (BM25) 검색 → top_k RetrievedChunk 반환."""
        rows = (
            self._tbl.search(query, query_type="fts")
            .limit(top_k)
            .to_list()
        )
        results: list[RetrievedChunk] = []
        for row in rows:
            bm25_score: float = float(row.get("_score", 0.0))
            results.append(RetrievedChunk(chunk=self._row_to_chunk(row), score=bm25_score))
        return results

    # ------------------------------------------------------------------
    # 보조 메서드
    # ------------------------------------------------------------------

    def count(self) -> int:
        """저장된 청크 수를 반환한다."""
        return self._tbl.count_rows()
