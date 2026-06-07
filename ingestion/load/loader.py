"""ingestion.load.loader — LanceDB 벡터스토어 일괄 적재 모듈.

단일 `upsert` 호출로 배치 적재를 수행한다 — 청크별 루프는 FTS 인덱스를
매번 재구성하므로 금지.
"""

from __future__ import annotations

from core.interfaces import VectorStore
from core.log import get_logger
from core.models import Chunk

logger = get_logger(__name__)


def load_chunks(
    chunks: list[Chunk],
    store: VectorStore | None = None,
) -> int:
    """embedded Chunk 목록을 벡터스토어에 배치 upsert한다.

    Parameters
    ----------
    chunks:
        embedding 이 채워진 Chunk 목록. embedding=None 인 청크가
        하나라도 포함되면 ValueError 를 발생시킨다.
    store:
        대상 VectorStore. None 이면 기본 LanceDBStore() 를 사용한다.

    Returns
    -------
    int
        실제 upsert 된 청크 수.

    Raises
    ------
    ValueError
        embedding=None 인 청크가 포함된 경우.
    """
    if not chunks:
        logger.debug("load_chunks: 빈 청크 목록 — 건너뜀")
        return 0

    # embedding 유효성 검사 — 조기 실패로 FTS 재구성 비용 방지
    missing = [c.id for c in chunks if c.embedding is None]
    if missing:
        raise ValueError(
            f"load_chunks: embedding=None 인 청크 {len(missing)}개 — "
            f"embed 후 재시도하세요. ids={missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    if store is None:
        # lazy import — vectorstore extra 없으면 ImportError
        from core.vectorstore import LanceDBStore

        store = LanceDBStore()

    logger.info("load_chunks: %d 청크 upsert 시작", len(chunks))
    store.upsert(chunks)  # 단일 호출 — FTS 인덱스 1회 재구성
    logger.info("load_chunks: %d 청크 upsert 완료", len(chunks))

    return len(chunks)
