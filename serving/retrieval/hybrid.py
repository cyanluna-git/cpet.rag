"""serving.retrieval.hybrid — BM25(FTS) + 벡터 검색 RRF 융합 HybridRetriever.

## 설계
- 쿼리 임베딩: JinaEmbedder.embed_query (task="retrieval.query") — 품질 이슈 해결.
- 벡터 ANN + FTS 두 결과 목록을 Reciprocal Rank Fusion (RRF) 으로 병합.
- 메타 필터(year, year_gte, journal, author, source, doi)를 융합 후 적용.
- 의존성: core.* 만 사용. lancedb·httpx 는 생성자에서 주입받는다.

## #3123 리랭커 연동
HybridRetriever.retrieve(query, top_k=pool) 로 넓은 후보 집합을 얻은 뒤
Reranker 가 RetrievedChunk.rerank_score 를 채우고 재정렬 → 최종 k 개 반환.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.log import get_logger
from core.metadata.loader import normalize_doi
from core.models import RetrievedChunk

if TYPE_CHECKING:
    from core.interfaces import Embedder, VectorStore
    from core.models.paper import Paper

logger = get_logger(__name__)

# RRF 기본 상수 (논문에서 권장하는 60)
_K_RRF: int = 60


class HybridRetriever:
    """BM25(FTS) + 벡터 검색을 RRF 로 융합하는 하이브리드 리트리버.

    Parameters
    ----------
    store:
        core.interfaces.VectorStore 구현체 (LanceDBStore 등).
    embedder:
        embed_query 메서드를 가진 임베더 (JinaEmbedder 등).
    papers_by_doi:
        메타 필터용 {정규화 DOI → Paper} 딕셔너리.
        None 이면 chunk 자체 필드(doi/source)만으로 필터링.
    """

    def __init__(
        self,
        store: "VectorStore",
        embedder: "Embedder",
        *,
        papers_by_doi: "dict[str, Paper] | None" = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._papers_by_doi = papers_by_doi or {}

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        pool: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """하이브리드 검색을 실행하고 최종 top_k 청크를 반환한다.

        Parameters
        ----------
        query:
            사용자 쿼리 문자열.
        top_k:
            최종 반환 청크 수 (필터 적용 후).
        pool:
            벡터/FTS 각각의 후보 수 (융합 전 pool 크기).
        filters:
            메타 필터 딕셔너리. 지원 키:
            - ``year``: 정확히 일치하는 출판 연도 (int).
            - ``year_gte``: 출판 연도 >= 값 (int).
            - ``journal``: 저널명 정확 일치 (str, case-insensitive).
            - ``author``: first_author 포함 여부 (str, case-insensitive).
            - ``source``: chunk.source 정확 일치 (str).
            - ``doi``: 정규화 DOI 정확 일치 (str).

        Returns
        -------
        list[RetrievedChunk]
            RRF 융합 점수 내림차순 정렬, 중복 없음, 메타 필터 통과 청크.
        """
        # 1. 쿼리 임베딩 (retrieval.query task)
        qvec: list[float] = self._embedder.embed_query([query])[0]
        logger.debug("HybridRetriever.retrieve: query=%r pool=%d top_k=%d", query, pool, top_k)

        # 2. 벡터 + FTS 검색
        vec_results = self._store.search(qvec, pool)
        fts_results = self._store.fts(query, pool)
        logger.debug("vec=%d fts=%d chunks retrieved", len(vec_results), len(fts_results))

        # 3. RRF 융합 (전체 dedup + 정렬)
        fused = self._rrf_fuse([vec_results, fts_results])

        # 4. 메타 필터 적용
        if filters:
            fused = [rc for rc in fused if self._passes_filter(rc, filters)]
            logger.debug("after filter: %d chunks", len(fused))

        # 5. top_k 반환
        return fused[:top_k]

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fuse(
        result_lists: list[list[RetrievedChunk]],
        k: int = _K_RRF,
    ) -> list[RetrievedChunk]:
        """여러 검색 결과 목록을 Reciprocal Rank Fusion 으로 병합한다.

        Parameters
        ----------
        result_lists:
            각 검색 방법의 결과 목록 (순서 = 랭크).
        k:
            RRF 상수 (기본 60). 랭크 민감도 조절: 작을수록 상위 랭크에 집중.

        Returns
        -------
        list[RetrievedChunk]
            chunk.id 로 dedup, RRF 점수 내림차순 정렬. score 에 융합 점수 설정.
        """
        # chunk.id → (RetrievedChunk, 누적 RRF 점수)
        scores: dict[str, tuple[RetrievedChunk, float]] = {}

        for result_list in result_lists:
            for rank, rc in enumerate(result_list, start=1):  # rank 는 1-based
                rrf_score = 1.0 / (k + rank)
                cid = rc.chunk.id
                if cid in scores:
                    prev_rc, prev_score = scores[cid]
                    scores[cid] = (prev_rc, prev_score + rrf_score)
                else:
                    scores[cid] = (rc, rrf_score)

        # 점수를 RetrievedChunk.score 에 반영하고 내림차순 정렬
        fused: list[RetrievedChunk] = []
        for rc, total_score in scores.values():
            fused.append(RetrievedChunk(chunk=rc.chunk, score=total_score))

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused

    def _passes_filter(self, rc: RetrievedChunk, filters: dict[str, Any]) -> bool:
        """청크가 메타 필터 조건을 모두 통과하는지 확인한다.

        papers_by_doi 가 있으면 Paper 메타를 조회해 필터링한다.
        없으면 chunk 자체 필드(doi/source)만 사용한다.
        Paper 를 찾을 수 없거나 필드가 None 인 경우 해당 조건은 불일치로 처리한다.
        """
        chunk = rc.chunk

        # Paper 조회 (DOI 정규화 후 lookup)
        paper: "Paper | None" = None
        if self._papers_by_doi:
            norm = normalize_doi(chunk.doi)
            if norm:
                paper = self._papers_by_doi.get(norm)

        for key, value in filters.items():
            if key == "doi":
                norm_filter = normalize_doi(str(value))
                norm_chunk = normalize_doi(chunk.doi)
                if norm_chunk != norm_filter:
                    return False

            elif key == "source":
                if chunk.source != value:
                    return False

            elif key == "year":
                year = paper.year if paper else None
                if year is None or year != int(value):
                    return False

            elif key == "year_gte":
                year = paper.year if paper else None
                if year is None or year < int(value):
                    return False

            elif key == "journal":
                journal = paper.journal if paper else None
                if journal is None or journal.lower() != str(value).lower():
                    return False

            elif key == "author":
                if paper is None:
                    return False
                target = str(value).lower()
                first = (paper.first_author or "").lower()
                all_authors = [a.lower() for a in (paper.authors or [])]
                if target not in first and not any(target in a for a in all_authors):
                    return False

        return True
