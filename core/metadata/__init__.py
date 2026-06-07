"""core.metadata — corpus_index 로더 + OpenAlex/Crossref 보강 유틸리티."""

from core.metadata.enrich import crossref_meta, enrich_paper, openalex_meta
from core.metadata.loader import index_by_doi, load_corpus_index, normalize_doi

__all__ = [
    "load_corpus_index",
    "index_by_doi",
    "normalize_doi",
    "crossref_meta",
    "openalex_meta",
    "enrich_paper",
]
