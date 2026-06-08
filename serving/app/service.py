"""serving.app.service — QueryService: 비즈니스 로직 레이어.

QueryPipeline 을 감싸 HTTP 레이어와 파이프라인 사이의 비즈니스 로직을 담당한다.
import-time 에 무거운 의존성(LanceDBStore, JinaEmbedder)을 로드하지 않는다 — 첫 요청
또는 명시적 build_default_pipeline() 호출 시에만 초기화된다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.log import get_logger
from core.models import QueryRequest, QueryResponse
from serving.pipeline import QueryPipeline

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def build_default_pipeline() -> QueryPipeline:
    """settings 기반으로 기본 QueryPipeline 을 생성한다.

    LanceDBStore / JinaEmbedder lazy import — vectorstore/ingestion extra 필요.
    import-time 에 호출 금지; 첫 요청 시점 또는 명시적 startup 에서만 호출할 것.
    """
    # lazy import — vectorstore/ingestion 패키지가 있을 때만 동작
    from core.config.settings import Settings  # noqa: PLC0415
    from core.vectorstore import LanceDBStore  # noqa: PLC0415
    from ingestion.embed.jina_embedder import JinaEmbedder  # noqa: PLC0415

    _settings = Settings()
    uri: str = _settings.lancedb_uri or "./data/lancedb"
    store = LanceDBStore(uri=uri)
    embedder = JinaEmbedder()

    logger.info("build_default_pipeline: uri=%s", uri)
    return QueryPipeline(store=store, embedder=embedder)


class QueryService:
    """QueryPipeline 을 감싸는 서비스 레이어.

    pipeline 이 None 으로 주입되면, 첫 ask() 호출 시 build_default_pipeline() 으로
    lazy 초기화한다 (import-time 에 무거운 것 로드 금지).

    Parameters
    ----------
    pipeline:
        QueryPipeline 구현체. None 이면 첫 요청 시 lazy 구성.
    """

    def __init__(self, pipeline: QueryPipeline | None = None) -> None:
        self._pipeline: QueryPipeline | None = pipeline

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def ask(self, req: QueryRequest) -> QueryResponse:
        """QueryRequest → QueryResponse 를 수행한다.

        파이프라인이 아직 초기화되지 않았으면 build_default_pipeline() 을 호출해
        lazy 구성한다.

        Parameters
        ----------
        req:
            질의 요청.

        Returns
        -------
        QueryResponse
        """
        if self._pipeline is None:
            logger.info("QueryService: lazy-initializing default pipeline")
            self._pipeline = build_default_pipeline()

        return self._pipeline.answer(req)
