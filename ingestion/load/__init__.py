"""ingestion.load — LanceDB 벡터스토어 적재 + 증분 인입 레지스트리."""

from ingestion.load.loader import load_chunks
from ingestion.load.registry import ProcessedRegistry, processed_key

__all__ = ["load_chunks", "ProcessedRegistry", "processed_key"]
