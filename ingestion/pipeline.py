"""ingestion.pipeline — 단일 오케스트레이션 진입점 (parse → VLM → chunk → embed → load).

## 설계

각 단계는 개별 모듈(ingestion.parse / build_chunks / embed / load)에서 구현됐다.
이 모듈은 그것들을 조합하는 얇은 오케스트레이터로, 비즈니스 로직은 없다.

## LlamaIndex 메모

현재 파이프라인은 custom orchestrator로 충분하다:
  - Docling 파싱 결과(ParsedDoc) → custom chunker → Jina Late Chunking → LanceDB upsert

LlamaIndex IngestionPipeline (from llama_index.core.ingestion import IngestionPipeline)으로
래핑할 경우 다음 위치가 적합하다:
  - parse + VLM → ``TransformComponent`` (``__call__(nodes, **kw)``)
  - chunk_document → ``TransformComponent``
  - embed_chunks → ``BaseEmbedding`` (embed_batch_size 설정)
  LlamaIndex의 캐싱(docstore) 및 AsyncPipeline 이점이 생기나, 현재 커스텀 흐름 대비
  추가 이점이 작아 도입은 코퍼스 규모 확장 시 재검토한다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from core.config import settings
from core.log import get_logger
from core.models.paper import Paper
from ingestion.build_chunks import parsed_to_embedded_chunks
from ingestion.load import ProcessedRegistry, load_chunks, processed_key

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """단일 논문 인입 결과."""

    paper_key: str
    doi: str | None
    status: Literal["ingested", "skipped", "failed", "no_pdf"]
    n_chunks: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Core: ingest_pdf
# ---------------------------------------------------------------------------


def ingest_pdf(
    pdf_path: str | Path,
    paper: Paper,
    *,
    store: object,
    embedder: object,
    registry: ProcessedRegistry | None = None,
    use_vlm: bool = True,
    use_late: bool = True,
    force: bool = False,
) -> IngestResult:
    """단일 PDF 를 파싱 → 청킹 → 임베딩 → 적재한다 (증분 skip 지원).

    Parameters
    ----------
    pdf_path:
        처리할 PDF 파일 경로.
    paper:
        논문 서지 메타데이터 (Paper 모델).
    store:
        VectorStore 구현체 (``core.interfaces.VectorStore`` 호환).
    embedder:
        Embedder 구현체 (``core.interfaces.Embedder`` + ``embed_chunks`` 호환).
    registry:
        ProcessedRegistry. None 이면 skip 체크를 건너뛴다.
    use_vlm:
        True 이면 low_confidence_pages 가 있을 때 VLM 폴백을 시도한다.
        API 키 없음 또는 VLM 실패 시 로그 후 표준 파싱 결과로 계속 진행한다.
    use_late:
        True 이면 Late Chunking 임베딩을 사용한다.
    force:
        True 이면 registry skip 체크를 무시하고 재인입한다.

    Returns
    -------
    IngestResult
        paper_key, doi, status, n_chunks, error 를 담은 결과 객체.
    """
    pdf_path = Path(pdf_path)
    key = processed_key(paper)
    embed_version = f"{settings.embed_model}@{settings.embed_dim}"

    # content_hash: 파일이 존재하면 SHA-256, 없으면 key 사용
    if pdf_path.exists():
        raw_bytes = pdf_path.read_bytes()
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
    else:
        content_hash = key

    # incremental skip check
    if (
        registry is not None
        and not force
        and registry.is_processed(key, content_hash=content_hash, embed_version=embed_version)
    ):
        logger.info("ingest_pdf: skip (already processed) key=%s", key)
        return IngestResult(
            paper_key=key,
            doi=paper.doi,
            status="skipped",
        )

    try:
        # --- 1. Parse ---
        # lazy import: docling 은 ingestion extra 에서만 사용 가능
        from ingestion.parse.docling_parser import parse_pdf
        from ingestion.parse.vlm_fallback import apply_vlm_fallback

        parsed = parse_pdf(str(pdf_path))

        # --- 2. VLM fallback (optional, non-fatal) ---
        if use_vlm and parsed.low_confidence_pages:
            try:
                parsed = apply_vlm_fallback(parsed, str(pdf_path))
                logger.info(
                    "ingest_pdf: VLM fallback applied for %d page(s) — key=%s",
                    len(parsed.vlm_pages),
                    key,
                )
            except Exception as vlm_exc:
                logger.warning(
                    "ingest_pdf: VLM fallback failed (key=%s) — continuing without VLM. error=%s",
                    key,
                    vlm_exc,
                )

        # --- 3. Chunk + Embed ---
        chunks = parsed_to_embedded_chunks(parsed, paper, embedder, use_late_chunking=use_late)

        # --- 4. Load ---
        n_loaded = load_chunks(chunks, store=store)

        # --- 5. Mark processed ---
        if registry is not None:
            registry.mark_processed(
                key,
                content_hash=content_hash,
                embed_version=embed_version,
                n_chunks=n_loaded,
            )

        logger.info("ingest_pdf: ingested key=%s n_chunks=%d", key, n_loaded)
        return IngestResult(
            paper_key=key,
            doi=paper.doi,
            status="ingested",
            n_chunks=n_loaded,
        )

    except Exception as exc:
        logger.error("ingest_pdf: FAILED key=%s error=%s", key, exc, exc_info=True)
        return IngestResult(
            paper_key=key,
            doi=paper.doi,
            status="failed",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Corpus: ingest_corpus
# ---------------------------------------------------------------------------


def ingest_corpus(
    papers: list[Paper],
    pdf_dir: str | Path,
    *,
    store: object | None = None,
    embedder: object | None = None,
    registry: ProcessedRegistry | None = None,
    limit: int | None = None,
    **kw: object,
) -> list[IngestResult]:
    """코퍼스 전체(또는 일부)를 증분 인입한다.

    Parameters
    ----------
    papers:
        처리할 Paper 목록 (``core.metadata.load_corpus_index()`` 결과 등).
    pdf_dir:
        PDF 파일이 위치한 로컬 디렉터리. ``paper.file`` 을 이 경로에 붙여 경로를 계산한다.
    store:
        VectorStore 구현체. None 이면 ``LanceDBStore()`` 를 생성한다.
    embedder:
        Embedder 구현체. None 이면 ``JinaEmbedder()`` 를 생성한다.
    registry:
        ProcessedRegistry. None 이면 ``ProcessedRegistry()`` 를 생성한다.
    limit:
        처리할 최대 논문 수. None 이면 모두 처리한다.
    **kw:
        ``ingest_pdf`` 에 그대로 전달되는 키워드 인수 (use_vlm, use_late, force 등).

    Returns
    -------
    list[IngestResult]
        처리 결과 목록. 순서는 papers 와 동일하다 (no_pdf 포함).
    """
    pdf_dir = Path(pdf_dir)

    # lazy 기본값 생성 — import 는 extras 필요 시에만
    if store is None:
        from core.vectorstore import LanceDBStore

        store = LanceDBStore()

    if embedder is None:
        from ingestion.embed import JinaEmbedder

        embedder = JinaEmbedder()

    if registry is None:
        registry = ProcessedRegistry()

    total = len(papers) if limit is None else min(limit, len(papers))
    target = papers[:total]

    results: list[IngestResult] = []

    for i, paper in enumerate(target, start=1):
        # PDF 경로 해석
        if not paper.file:
            logger.info("[%d/%d] no_pdf (file=None) — key=%s", i, total, processed_key(paper))
            results.append(
                IngestResult(
                    paper_key=processed_key(paper),
                    doi=paper.doi,
                    status="no_pdf",
                )
            )
            continue

        pdf_path = pdf_dir / paper.file
        if not pdf_path.exists():
            logger.info(
                "[%d/%d] no_pdf (missing) — key=%s path=%s",
                i,
                total,
                processed_key(paper),
                pdf_path,
            )
            results.append(
                IngestResult(
                    paper_key=processed_key(paper),
                    doi=paper.doi,
                    status="no_pdf",
                )
            )
            continue

        logger.info("[%d/%d] ingesting key=%s", i, total, processed_key(paper))
        result = ingest_pdf(
            pdf_path,
            paper,
            store=store,
            embedder=embedder,
            registry=registry,
            **kw,
        )
        results.append(result)
        logger.info(
            "[%d/%d] done — status=%s n_chunks=%d", i, total, result.status, result.n_chunks
        )

    # summary tally
    tally: dict[str, int] = {"ingested": 0, "skipped": 0, "no_pdf": 0, "failed": 0}
    for r in results:
        tally[r.status] = tally.get(r.status, 0) + 1

    print(
        f"\n[ingest_corpus] summary — "
        f"total={total} "
        f"ingested={tally['ingested']} "
        f"skipped={tally['skipped']} "
        f"no_pdf={tally['no_pdf']} "
        f"failed={tally['failed']}"
    )

    return results
