"""ingestion — 인입 파이프라인 패키지 (배치·GPU 환경).

공개 API:
    ingest_pdf      : 단일 PDF 인입 (parse → VLM → chunk → embed → load).
    ingest_corpus   : 코퍼스 전체 증분 인입.
    IngestResult    : 인입 결과 스키마.
"""

from ingestion.pipeline import IngestResult, ingest_corpus, ingest_pdf

__all__ = ["ingest_pdf", "ingest_corpus", "IngestResult"]
