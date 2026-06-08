"""serving.retrieval.rerank — 크로스인코더 리랭커 (Cohere Rerank via Bedrock / Jina Reranker).

## 설계
- `Reranker.rerank(query, candidates, top_k)`:
    1. 후보(pool ~50개) → `_rerank_call` → input-aligned 점수 리스트
    2. 각 RetrievedChunk 에 `rerank_score` 설정(non-mutating: model_copy)
    3. rerank_score 내림차순 정렬 → 상위 top_k 반환

- `_rerank_call` 은 테스트에서 mock 할 수 있는 seam.
  실제 구현은 backend 에 따라 Bedrock 또는 Jina 를 lazy import 로 호출.

## Bedrock Cohere Rerank 호출 포맷 (bedrock-agent-runtime)
```
client = boto3.client("bedrock-agent-runtime", region_name=region)
response = client.rerank(
    rerankingConfiguration={
        "type": "BEDROCK_RERANKING_MODEL",
        "bedrockRerankingConfiguration": {
            "modelConfiguration": {
                "modelArn": "arn:aws:bedrock:{region}::foundation-model/{model_id}",
            },
            "numberOfResults": len(documents),
        },
    },
    sources=[
        {
            "type": "INLINE",
            "inlineDocumentSource": {
                "type": "TEXT",
                "textDocument": {"text": doc},
            },
        }
        for doc in documents
    ],
    queries=[{"type": "TEXT", "textQuery": {"text": query}}],
)
# response["results"]: [{"index": int, "relevanceScore": float, "document": ...}, ...]
```

## Jina Reranker 호출 포맷
```
POST https://api.jina.ai/v1/rerank
Authorization: Bearer {api_key}
{
  "model": "jina-reranker-v2-base-multilingual",
  "query": query,
  "documents": documents,
  "top_n": len(documents)   # 전체 점수 반환 — top_k 슬라이싱은 rerank() 에서
}
# results[i]: {"index": int, "relevance_score": float}
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.config.settings import Settings
from core.log import get_logger
from core.models import RetrievedChunk

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Jina Reranker 기본 모델
_JINA_DEFAULT_MODEL: str = "jina-reranker-v2-base-multilingual"
_JINA_RERANK_URL: str = "https://api.jina.ai/v1/rerank"


class Reranker:
    """크로스인코더 리랭커 — 하이브리드 검색 후보(pool)를 재정렬해 top_k 반환.

    Parameters
    ----------
    model:
        모델 ID. None 이면 settings.bedrock_rerank_model 사용.
        Jina backend 는 ``jina-reranker-v2-base-multilingual`` 등을 지정.
    backend:
        ``"bedrock"`` — Cohere Rerank via Amazon Bedrock Agent Runtime (boto3).
        ``"jina"`` — Jina Reranker API (httpx).
    region:
        AWS 리전 (bedrock backend 전용). None 이면 settings.aws_region 사용.
    api_key:
        Jina API 키 (jina backend 전용). None 이면 settings.jina_api_key 사용.
    """

    def __init__(
        self,
        model: str | None = None,
        backend: str = "bedrock",
        region: str | None = None,
        api_key: str | None = None,
    ) -> None:
        _settings = Settings()
        self._backend: str = backend
        self._region: str = region or _settings.aws_region
        self._api_key: str | None = api_key or _settings.jina_api_key
        # bedrock: bedrock_rerank_model (예: "cohere.rerank-v3-5:0")
        # jina: 기본 모델명 또는 명시적 지정
        if model is not None:
            self._model: str | None = model
        elif backend == "bedrock":
            self._model = _settings.bedrock_rerank_model
        else:
            self._model = _JINA_DEFAULT_MODEL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        """후보 청크를 크로스인코더로 재정렬하고 상위 top_k 를 반환한다.

        Parameters
        ----------
        query:
            사용자 질의 문자열.
        candidates:
            HybridRetriever.retrieve(pool=50) 등으로 얻은 후보 청크 목록.
        top_k:
            최종 반환 청크 수.

        Returns
        -------
        list[RetrievedChunk]
            rerank_score 내림차순 정렬, 상위 top_k 청크.
            각 항목은 원본 candidates 의 model_copy(non-mutating).
        """
        if not candidates:
            return []

        documents = [c.chunk.text for c in candidates]
        logger.debug(
            "Reranker.rerank: backend=%s query=%r candidates=%d top_k=%d",
            self._backend,
            query,
            len(candidates),
            top_k,
        )

        # 점수 리스트는 documents 순서와 1:1 대응
        scores: list[float] = self._rerank_call(query, documents)

        # non-mutating: model_copy 로 rerank_score 설정
        reranked: list[RetrievedChunk] = [
            rc.model_copy(update={"rerank_score": s}) for rc, s in zip(candidates, scores)
        ]

        # rerank_score 내림차순 정렬
        reranked.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)

        result = reranked[:top_k]
        logger.debug(
            "Reranker.rerank 완료: top_k=%d best_score=%.4f",
            len(result),
            result[0].rerank_score if result else 0.0,
        )
        return result

    # ------------------------------------------------------------------
    # Internal seam — 테스트에서 subclass 또는 patch.object 로 mock
    # ------------------------------------------------------------------

    def _rerank_call(self, query: str, documents: list[str]) -> list[float]:
        """리랭커 API 를 호출해 documents 순서에 대응하는 relevance 점수를 반환한다.

        Parameters
        ----------
        query:
            질의 텍스트.
        documents:
            문서 텍스트 목록 (candidates 와 동일 순서).

        Returns
        -------
        list[float]
            ``documents[i]`` 에 대한 relevance score (0.0–1.0).
            길이는 반드시 ``len(documents)`` 와 같아야 한다.

        Raises
        ------
        RuntimeError:
            boto3/httpx 미설치, API 키 미설정, 모델 ID 미설정 등 설정 오류.
        """
        if self._backend == "bedrock":
            return self._rerank_bedrock(query, documents)
        elif self._backend == "jina":
            return self._rerank_jina(query, documents)
        else:
            raise RuntimeError(
                f"알 수 없는 backend: {self._backend!r}. 'bedrock' 또는 'jina' 를 사용하세요."
            )

    # ------------------------------------------------------------------
    # Backend implementations (lazy import)
    # ------------------------------------------------------------------

    def _rerank_bedrock(self, query: str, documents: list[str]) -> list[float]:
        """Bedrock Agent Runtime Rerank API 로 Cohere Rerank 를 호출한다.

        Bedrock bedrock-agent-runtime ``rerank`` API 를 사용한다.
        모델 ARN 포맷: ``arn:aws:bedrock:{region}::foundation-model/{model_id}``

        질의는 ``queries=[{"type": "TEXT", "textQuery": {"text": query}}]`` 로 전달한다.
        응답 ``results`` 는 relevance 내림차순이며 각 항목에 ``index`` 필드가 있다.
        input-aligned 점수 벡터로 재배열해 반환한다.

        Raises
        ------
        RuntimeError:
            boto3 미설치 또는 bedrock_rerank_model 미설정.
        """
        try:
            import boto3  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "boto3 가 설치되어 있지 않습니다. `uv pip install cpet-rag[aws]` 를 실행하세요."
            ) from exc

        if not self._model:
            raise RuntimeError(
                "bedrock_rerank_model 이 설정되지 않았습니다. "
                ".env 에 BEDROCK_RERANK_MODEL 을 추가하세요. "
                "예: cohere.rerank-v3-5:0"
            )

        model_arn = f"arn:aws:bedrock:{self._region}::foundation-model/{self._model}"
        client = boto3.client("bedrock-agent-runtime", region_name=self._region)

        response = client.rerank(
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {
                        "modelArn": model_arn,
                    },
                    "numberOfResults": len(documents),
                },
            },
            sources=[
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {
                        "type": "TEXT",
                        "textDocument": {"text": doc},
                    },
                }
                for doc in documents
            ],
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
        )

        # response["results"] 는 relevance 내림차순, 각 항목: {"index": int, "relevanceScore": float}
        results = response.get("results", [])
        scores: list[float] = [0.0] * len(documents)
        for item in results:
            idx: int = item["index"]
            score: float = float(item["relevanceScore"])
            if 0 <= idx < len(documents):
                scores[idx] = score

        logger.info(
            "Bedrock Rerank 완료: model=%s docs=%d",
            self._model,
            len(documents),
        )
        return scores

    def _rerank_jina(self, query: str, documents: list[str]) -> list[float]:
        """Jina Reranker API 를 호출한다.

        엔드포인트: ``POST https://api.jina.ai/v1/rerank``
        요청 바디::

            {
              "model": "jina-reranker-v2-base-multilingual",
              "query": query,
              "documents": documents,
              "top_n": len(documents)
            }

        응답 ``results[i]``: ``{"index": int, "relevance_score": float}``
        index 기반으로 input-aligned 점수 벡터를 재구성해 반환한다.

        Raises
        ------
        RuntimeError:
            jina_api_key 미설정 또는 HTTP 에러.
        """
        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "httpx 가 설치되어 있지 않습니다. `uv pip install httpx` 를 실행하세요."
            ) from exc

        if not self._api_key:
            raise RuntimeError(
                "Jina API 키가 설정되지 않았습니다. " ".env 에 JINA_API_KEY 를 추가하세요."
            )

        model = self._model or _JINA_DEFAULT_MODEL
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),  # 전체 점수 반환, top_k 슬라이싱은 rerank() 에서
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                _JINA_RERANK_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        # data["results"] 는 relevance 내림차순, 각 항목: {"index": int, "relevance_score": float}
        results = data.get("results", [])
        scores: list[float] = [0.0] * len(documents)
        for item in results:
            idx: int = item["index"]
            score: float = float(item["relevance_score"])
            if 0 <= idx < len(documents):
                scores[idx] = score

        logger.info(
            "Jina Rerank 완료: model=%s docs=%d",
            model,
            len(documents),
        )
        return scores
