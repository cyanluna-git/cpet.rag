#!/usr/bin/env python3
"""배치 인입 CLI — 코퍼스 전체(또는 일부) PDF를 LanceDB에 증분 인입한다.

Usage examples:
    # 전체 794편 인입 (Colab/운영)
    uv run python scripts/ingest_all.py --pdf-dir /path/to/pdf

    # 50편만 인입
    uv run python scripts/ingest_all.py --pdf-dir /path/to/pdf --limit 50

    # dry-run: PDF 존재 여부만 점검, 실제 인입 없음
    uv run python scripts/ingest_all.py --pdf-dir /path/to/pdf --dry-run

    # 증분 무시하고 재인입
    uv run python scripts/ingest_all.py --pdf-dir /path/to/pdf --force

    # 커스텀 LanceDB URI / corpus CSV
    uv run python scripts/ingest_all.py --pdf-dir /path/to/pdf \\
        --lancedb-uri s3://my-bucket/lancedb \\
        --corpus data/corpus_index.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from core.config import settings
from core.log import get_logger
from core.metadata import load_corpus_index

logger = get_logger(__name__)

# 실패 목록 기본 저장 경로 — tests 에서 monkeypatch 가능하도록 module-level 상수
FAILURES_CSV: Path = Path("data/ingest_failures.csv")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ingest_all",
        description="코퍼스 PDF를 LanceDB에 증분 인입한다.",
    )
    parser.add_argument(
        "--pdf-dir",
        required=True,
        help="PDF 파일이 위치한 디렉터리 경로",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="처리할 최대 논문 수 (기본값: 전체)",
    )
    parser.add_argument(
        "--lancedb-uri",
        default=None,
        help=f"LanceDB URI (기본값: settings.lancedb_uri={settings.lancedb_uri!r})",
    )
    parser.add_argument(
        "--corpus",
        default="data/corpus_index.csv",
        help="corpus_index CSV 경로 (기본값: data/corpus_index.csv)",
    )
    parser.add_argument(
        "--backend",
        default="api",
        choices=["api", "local"],
        help="Jina 임베딩 백엔드 (기본값: api)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="증분 skip 무시 — 이미 처리된 논문도 재인입",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="PDF 존재 여부만 점검하고 실제 인입은 수행하지 않음",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점. 성공 0, 실패(failed 논문 있음) 1을 반환한다."""
    args = _parse_args(argv)

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_dir():
        logger.error("--pdf-dir 가 존재하지 않습니다: %s", pdf_dir)
        print(f"[ERROR] --pdf-dir 가 존재하지 않습니다: {pdf_dir}", file=sys.stderr)
        return 1

    # --- 1. corpus 로드 ---
    logger.info("corpus 로드: %s", args.corpus)
    papers = load_corpus_index(args.corpus)
    logger.info("총 %d 편 로드", len(papers))

    # --- 2. PDF 존재하는 paper 만 필터 ---
    have = [p for p in papers if p.file and (pdf_dir / p.file).exists()]
    missing_count = len(papers) - len(have)

    logger.info(
        "PDF 존재: %d / %d  (누락: %d)",
        len(have),
        len(papers),
        missing_count,
    )
    if missing_count > 0:
        logger.info("%d 편의 PDF 파일이 --pdf-dir 에 없습니다 (no_pdf 처리됨)", missing_count)

    # --- dry-run: 존재 여부 점검만 ---
    if args.dry_run:
        print(
            f"\n[dry-run] corpus={args.corpus} pdf_dir={pdf_dir}\n"
            f"  총 논문     : {len(papers)}\n"
            f"  PDF 존재    : {len(have)}\n"
            f"  PDF 누락    : {missing_count}\n"
            f"  (limit={args.limit})\n"
            f"  ※ 실제 인입 없음 (--dry-run 모드)"
        )
        return 0

    # --- 3. 컴포넌트 구성 ---
    lancedb_uri: str | None = args.lancedb_uri or settings.lancedb_uri

    from core.vectorstore import LanceDBStore
    from ingestion import ingest_corpus
    from ingestion.embed import JinaEmbedder
    from ingestion.load import ProcessedRegistry

    store = LanceDBStore(uri=lancedb_uri)
    embedder = JinaEmbedder(backend=args.backend)
    registry = ProcessedRegistry()

    logger.info(
        "인입 시작 — uri=%s backend=%s force=%s limit=%s",
        lancedb_uri,
        args.backend,
        args.force,
        args.limit,
    )

    # --- 4. 인입 실행 ---
    # have 를 전달 → ingest_corpus 내부에서 limit 슬라이싱 수행
    results = ingest_corpus(
        have,
        pdf_dir,
        store=store,
        embedder=embedder,
        registry=registry,
        limit=args.limit,
        force=args.force,
    )

    # --- 5. 결과 집계 ---
    tally: dict[str, int] = {"ingested": 0, "skipped": 0, "no_pdf": 0, "failed": 0}
    total_chunks = 0
    failed_rows: list[dict[str, str | None]] = []

    for r in results:
        tally[r.status] = tally.get(r.status, 0) + 1
        total_chunks += r.n_chunks
        if r.status == "failed":
            failed_rows.append({"doi": r.doi, "error": r.error})

    # no_pdf from pre-filter (papers not in have)
    tally["no_pdf"] = tally.get("no_pdf", 0) + missing_count

    try:
        store_count = store.count()
    except Exception:
        store_count = -1

    print(
        f"\n[ingest_all] 완료 ─────────────────────────────\n"
        f"  corpus      : {args.corpus}\n"
        f"  pdf_dir     : {pdf_dir}\n"
        f"  lancedb_uri : {lancedb_uri}\n"
        f"  limit       : {args.limit}\n"
        f"  force       : {args.force}\n"
        f"  ───────────────────────────────────────────\n"
        f"  ingested    : {tally['ingested']}\n"
        f"  skipped     : {tally['skipped']}\n"
        f"  no_pdf      : {tally['no_pdf']}\n"
        f"  failed      : {tally['failed']}\n"
        f"  총 청크     : {total_chunks}\n"
        f"  store.count : {store_count}\n"
        f"  ───────────────────────────────────────────"
    )

    # --- 6. 실패 CSV 저장 ---
    if failed_rows:
        _write_failures(failed_rows, FAILURES_CSV)

    return 1 if tally["failed"] > 0 else 0


def _write_failures(rows: list[dict[str, str | None]], path: Path) -> None:
    """실패 목록을 CSV 로 저장한다 (doi, error 컬럼)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["doi", "error"])
            writer.writeheader()
            writer.writerows(rows)
        logger.info("실패 목록 저장: %s (%d 건)", path, len(rows))
        print(f"  실패 목록   : {path} ({len(rows)} 건)")
    except OSError as exc:
        logger.error("실패 CSV 저장 실패: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
