"""core.interfaces — 공통 인터페이스 Protocol 스텁.

구체 구현 없음 — 시그니처·타입힌트만 정의한다.
"""

from typing import Any, Protocol, runtime_checkable

from core.models import Chunk, RetrievedChunk


@runtime_checkable
class VectorStore(Protocol):
    """벡터스토어 추상 인터페이스 (LanceDB·OpenSearch·pgvector 교체 대비)."""

    def upsert(self, chunks: list[Chunk]) -> None:
        """청크 목록을 저장/갱신한다."""
        ...

    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """벡터 유사도 검색 후 top_k 청크를 반환한다."""
        ...

    def fts(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Full-text search 후 top_k 청크를 반환한다."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """텍스트 임베딩 인터페이스 (Jina-v3 · self-host 공용)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """texts 각 항목을 임베딩 벡터로 변환한다."""
        ...

    def embed_late(
        self, document: str, boundaries: list[tuple[int, int]]
    ) -> list[list[float]]:
        """Late Chunking: document 전체 컨텍스트 기반으로 각 boundary 구간의 벡터를 반환한다."""
        ...


@runtime_checkable
class LLMClient(Protocol):
    """LLM 생성 인터페이스 (Bedrock Claude 기본)."""

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """prompt를 받아 텍스트 응답을 반환한다."""
        ...


@runtime_checkable
class Translator(Protocol):
    """한-영 번역 인터페이스 (Query Translation 샌드위치 사용)."""

    def ko2en(self, text: str) -> str:
        """한국어를 영어로 번역한다."""
        ...

    def en2ko(self, text: str) -> str:
        """영어를 한국어로 번역한다."""
        ...
