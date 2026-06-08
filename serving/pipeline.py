"""serving.pipeline — 질의 오케스트레이션 파이프라인 (Phase-3 L3).

한국어 질의 → 번역 → 하이브리드 검색 → 리랭크 → 생성(Strict Citation)
           → 인용검증 → 미검증 태그 제거 → 역번역 → QueryResponse 반환

모든 LLM/외부 호출 seam (BedrockTranslator._translate_call,
Reranker._rerank_call, Generator._generate_call) 은 테스트에서 mock 가능.

vectorstore/ingestion 의존성은 answer_query 내에서만 lazy import 하므로
base 의존성만으로 `from serving import QueryPipeline, answer_query` 가 가능하다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.citation import strip_unverified, verify_citations
from core.log import get_logger
from core.models import QueryRequest, QueryResponse
from serving.generation import Generator, back_translate_answer
from serving.retrieval import BedrockTranslator, HybridRetriever, Reranker

if TYPE_CHECKING:
    from core.interfaces import Embedder, Translator, VectorStore
    from core.models.paper import Paper

logger = get_logger(__name__)

# 번역 거부 시 반환 한국어 메시지
_KO_REFUSAL = "제공된 자료만으로는 답변할 수 없습니다."


class QueryPipeline:
    """질의 오케스트레이션 DI 컨테이너.

    Parameters
    ----------
    store:
        core.interfaces.VectorStore 구현체.
    embedder:
        embed_query 를 가진 임베더.
    translator:
        core.interfaces.Translator 구현체. None 이면 BedrockTranslator() 생성.
    reranker:
        Reranker 인스턴스. None 이면 Reranker() 생성.
    generator:
        Generator 인스턴스. None 이면 Generator() 생성.
    papers_by_doi:
        메타 필터용 {정규화 DOI → Paper} 딕셔너리. None 이면 빈 dict.
    """

    def __init__(
        self,
        *,
        store: "VectorStore",
        embedder: "Embedder",
        translator: "Translator | None" = None,
        reranker: "Reranker | None" = None,
        generator: "Generator | None" = None,
        papers_by_doi: "dict[str, Paper] | None" = None,
    ) -> None:
        self._translator: "Translator" = translator or BedrockTranslator()
        self._reranker: Reranker = reranker or Reranker()
        self._generator: Generator = generator or Generator()
        self._retriever = HybridRetriever(store, embedder, papers_by_doi=papers_by_doi)

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def answer(self, req: QueryRequest) -> QueryResponse:
        """QueryRequest → QueryResponse 를 오케스트레이션한다.

        단계:
        1. 번역 (translate=True 면 ko2en)
        2. 하이브리드 검색 (pool=50)
        3. 리랭크
        4. 생성 (Strict Citation)
        5. 거부 확인
        6. 인용 검증 (overlap)
        7. 미검증 태그 제거
        8. 역번역 (translate=True 면 en2ko)

        Parameters
        ----------
        req:
            질의 요청. query, top_k, filters, translate 필드를 사용한다.

        Returns
        -------
        QueryResponse
            answer(한국어 또는 영어), answer_en, citations(검증 통과), retrieved.
        """
        # 1. 쿼리 번역
        try:
            query_en: str = self._translator.ko2en(req.query) if req.translate else req.query
            logger.info(
                "QueryPipeline.answer: query=%r → query_en=%r translate=%s",
                req.query,
                query_en,
                req.translate,
            )
        except Exception as exc:
            logger.error("번역 실패 — 원문 쿼리를 그대로 사용: %s", exc)
            query_en = req.query

        # 2. 하이브리드 검색
        try:
            candidates = self._retriever.retrieve(
                query_en,
                top_k=req.top_k,
                pool=50,
                filters=req.filters,
            )
            logger.info("검색 완료: candidates=%d", len(candidates))
        except Exception as exc:
            logger.error("검색 실패 — 빈 결과 반환: %s", exc)
            refusal_answer = _KO_REFUSAL if req.translate else "I cannot answer from the provided sources."
            return QueryResponse(
                answer=refusal_answer,
                answer_en="I cannot answer from the provided sources.",
                citations=[],
                retrieved=[],
            )

        # 3. 리랭크
        try:
            top = self._reranker.rerank(query_en, candidates, top_k=req.top_k)
            logger.info("리랭크 완료: top=%d", len(top))
        except Exception as exc:
            logger.error("리랭크 실패 — 검색 결과 그대로 사용: %s", exc)
            top = candidates[: req.top_k]

        # 4. 생성
        try:
            gen = self._generator.generate(query_en, top)
            logger.info(
                "생성 완료: refused=%s citations=%d answer_len=%d",
                gen.refused,
                len(gen.citations),
                len(gen.answer_en),
            )
        except Exception as exc:
            logger.error("생성 실패 — 안전 응답 반환: %s", exc)
            refusal_answer = _KO_REFUSAL if req.translate else "I cannot answer from the provided sources."
            return QueryResponse(
                answer=refusal_answer,
                answer_en="I cannot answer from the provided sources.",
                citations=[],
                retrieved=top,
            )

        # 5. 거부 확인
        if gen.refused:
            logger.info("생성 거부됨 → 거부 응답 반환")
            refusal_answer: str
            if req.translate:
                try:
                    refusal_answer = self._translator.en2ko(gen.answer_en)
                except Exception:
                    refusal_answer = _KO_REFUSAL
            else:
                refusal_answer = gen.answer_en
            return QueryResponse(
                answer=refusal_answer,
                answer_en=gen.answer_en,
                citations=[],
                retrieved=top,
            )

        # 6. 인용 검증 (overlap)
        try:
            vr = verify_citations(gen.answer_en, gen.citations, top)
            logger.info(
                "인용 검증: verified=%d unverified=%d faithfulness=%.3f",
                len(vr.verified),
                len(vr.unverified),
                vr.faithfulness,
            )
        except Exception as exc:
            logger.error("인용 검증 실패 — citations 전체 사용: %s", exc)
            # 안전 폴백: 검증 건너뜀
            clean_en = gen.answer_en
            verified_citations = gen.citations
        else:
            # 7. 미검증 태그 제거
            clean_en = strip_unverified(gen.answer_en, vr.unverified)
            verified_citations = vr.verified

        # 8. 역번역
        try:
            if req.translate:
                answer_ko = back_translate_answer(clean_en, self._translator)
            else:
                answer_ko = clean_en
        except Exception as exc:
            logger.error("역번역 실패 — 영문 답변 그대로 반환: %s", exc)
            answer_ko = clean_en

        logger.info("QueryPipeline.answer 완료: answer_len=%d", len(answer_ko))
        return QueryResponse(
            answer=answer_ko,
            answer_en=clean_en,
            citations=verified_citations,
            retrieved=top,
        )

    def answer_query(
        self,
        query_ko: str,
        *,
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
    ) -> QueryResponse:
        """편의 메서드 — translate=True 로 QueryRequest 를 구성해 answer() 를 호출한다.

        Parameters
        ----------
        query_ko:
            한국어 질의 문자열.
        top_k:
            반환 청크 수.
        filters:
            메타 필터 딕셔너리.

        Returns
        -------
        QueryResponse
        """
        req = QueryRequest(query=query_ko, top_k=top_k, filters=filters)
        return self.answer(req)


def answer_query(
    query_ko: str,
    *,
    top_k: int = 8,
    filters: dict[str, Any] | None = None,
) -> QueryResponse:
    """모듈 수준 편의 함수 — 기본 QueryPipeline 으로 질의를 처리한다.

    store/embedder 를 lazy import 로 생성하므로 base 의존성으로도 import 가능하다.
    실제 호출 시에는 vectorstore/ingestion 패키지가 필요하다.

    Parameters
    ----------
    query_ko:
        한국어 질의 문자열.
    top_k:
        반환 청크 수.
    filters:
        메타 필터 딕셔너리.

    Returns
    -------
    QueryResponse
    """
    # lazy import — vectorstore/ingestion extra 필요 (호출 시점에만)
    from core.config.settings import Settings  # noqa: PLC0415
    from core.vectorstore import LanceDBStore  # noqa: PLC0415
    from ingestion.embed.jina_embedder import JinaEmbedder  # noqa: PLC0415

    _settings = Settings()
    uri: str = _settings.lancedb_uri or "./data/lancedb"
    store = LanceDBStore(uri=uri)
    embedder = JinaEmbedder()

    pipeline = QueryPipeline(store=store, embedder=embedder)
    return pipeline.answer_query(query_ko, top_k=top_k, filters=filters)
