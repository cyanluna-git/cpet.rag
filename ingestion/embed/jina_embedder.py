"""ingestion.embed.jina_embedder — Jina-embeddings-v3 Late Chunking 임베더.

## 설계 결정

### 경계 (boundaries) 규약 — char offset 기반
embed_late(document, boundaries) 의 boundaries 는 **문자(char) 오프셋** 쌍이다.
- API path: boundaries 는 chunk 텍스트 목록 재구성에만 사용
  (`document[s:e]` 로 각 chunk text 추출 → Jina late_chunking=true 로 전달).
  Jina API 는 chunk 문자열 배열 + late_chunking=true 를 받아 full-context 임베딩을 반환.
- Local path: tokenizer offset_mapping 을 사용해 char range → token span 변환 후 mean-pool.

### embed_chunks 의 Late 경로 document 구성
full_document 가 None 이면 standard path(ctx_text 임베딩).
full_document 가 주어지면 Late 경로:
  - document = "\n".join(c.text for c in chunks)  # chunk text 연결
  - boundaries = 각 chunk.text 의 연결 문서 내 char offset
  - ctx_text 대신 text 를 경계 기반으로 임베딩 (document-context 부여가 목적)
  NOTE: 연결 document 는 parsed.markdown 과 정확히 일치하지 않으므로
        Jina API 에 넘기는 문자열은 chunk text 목록으로 직접 전달한다.
        full_document 파라미터는 "late 모드 활성화" 플래그로 사용되며
        실제 API call 에는 chunk.text 배열이 전달된다.

### 실제 API 엔드포인트 (Colab/prod 실행 시)
  POST https://api.jina.ai/v1/embeddings
  Headers: Authorization: Bearer <JINA_API_KEY>
  Body: {
    "model": "jinaai/jina-embeddings-v3",
    "task": "retrieval.passage",
    "late_chunking": true,       # Late Chunking 활성화
    "input": ["chunk1", "chunk2", ...]
  }
  → 각 입력 문자열이 전체 문서 컨텍스트를 조건으로 임베딩됨.
  dim 파라미터로 MRL(Matryoshka Representation Learning) 차원 절단 가능.

### isinstance(JinaEmbedder(), Embedder) == True
runtime_checkable Protocol — embed / embed_late 메서드 존재 여부만 확인.
"""

from __future__ import annotations

import copy
from typing import Any

import httpx

from core.config.settings import Settings
from core.log import get_logger
from core.models import Chunk

logger = get_logger(__name__)

# Jina REST API 엔드포인트
JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"


class JinaEmbedder:
    """Jina-embeddings-v3 임베더.

    두 가지 백엔드를 지원한다:
    - ``"api"`` (기본): Jina REST API via httpx — API 키 필요. 외부 환경에서 사용.
    - ``"local"``: transformers/sentence-transformers 로컬 실행 — GPU/Colab 환경.

    API 키는 생성 시 불필요 (None 허용); 실제 embed 호출 시점에 필요하다.
    """

    def __init__(
        self,
        model: str | None = None,
        dim: int | None = None,
        api_key: str | None = None,
        backend: str = "api",
    ) -> None:
        settings = Settings()
        self.model: str = model or settings.embed_model
        self.dim: int = dim or settings.embed_dim
        self.api_key: str | None = api_key or settings.jina_api_key
        self.backend: str = backend

        # local 백엔드용 모델 캐시 (lazy init)
        self._local_model: Any = None

    # ------------------------------------------------------------------
    # Internal: API calls (mockable in tests)
    # ------------------------------------------------------------------

    # Jina v3 컨텍스트 = 8192 토큰. late chunking 은 입력을 이어붙여 한 컨텍스트에
    # 넣으므로 윈도우 합이 한도를 넘으면 400. 보수적으로 char 예산으로 분할한다.
    _LATE_WINDOW_CHARS = 20000   # ~5-6k 토큰 (8192 한도 아래)
    _STD_BATCH_CHARS = 60000     # 표준 요청 토큰 상한 회피

    def _post_jina(self, payload: dict[str, Any], timeout: float = 180.0) -> list[list[float]]:
        """Jina embeddings POST. 4xx 시 응답 본문을 포함해 명확히 raise."""
        response = httpx.post(
            JINA_EMBEDDINGS_URL,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Jina API {response.status_code} for {JINA_EMBEDDINGS_URL}: "
                f"{response.text[:600]}"
            )
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    @staticmethod
    def _windows_by_chars(texts: list[str], budget: int) -> list[list[str]]:
        """연속 texts 를 누적 char 예산 단위로 묶는다 (순서 보존)."""
        groups: list[list[str]] = []
        cur: list[str] = []
        clen = 0
        for t in texts:
            if cur and clen + len(t) > budget:
                groups.append(cur)
                cur, clen = [], 0
            cur.append(t)
            clen += len(t)
        if cur:
            groups.append(cur)
        return groups

    def _embed_call(self, texts: list[str]) -> list[list[float]]:
        """표준 임베딩 API 호출 (배치 분할). 테스트에서 이 메서드를 mock 한다."""
        if self.backend == "local":
            return self._local_encode(texts)
        if not self.api_key:
            raise ValueError("JinaEmbedder: jina_api_key 가 설정되지 않았습니다.")
        out: list[list[float]] = []
        for batch in self._windows_by_chars(texts, self._STD_BATCH_CHARS):
            out.extend(
                self._post_jina(
                    {
                        "model": self.model,
                        "task": "retrieval.passage",
                        "input": batch,
                        "dimensions": self.dim,
                    },
                    timeout=120.0,
                )
            )
        return out

    def _embed_late_call(self, texts: list[str]) -> list[list[float]]:
        """Late Chunking API 호출. 컨텍스트(8192토큰)를 넘지 않도록 윈도우 분할 후
        각 윈도우를 late_chunking=true 로 호출(윈도우 내 청크끼리 full-context 공유).
        테스트에서 이 메서드를 mock 한다.
        """
        if self.backend == "local":
            return self._local_late_encode(texts)
        if not self.api_key:
            raise ValueError("JinaEmbedder: jina_api_key 가 설정되지 않았습니다.")
        windows = self._windows_by_chars(texts, self._LATE_WINDOW_CHARS)
        logger.info("late chunking: %d chunks → %d windows", len(texts), len(windows))
        out: list[list[float]] = []
        for window in windows:
            out.extend(
                self._post_jina(
                    {
                        "model": self.model,
                        "task": "retrieval.passage",
                        "late_chunking": True,
                        "input": window,
                        "dimensions": self.dim,
                    }
                )
            )
        return out

    # ------------------------------------------------------------------
    # Local backend helpers (lazy import)
    # ------------------------------------------------------------------

    def _init_local_model(self) -> None:
        """transformers / sentence-transformers 모델을 lazy 로드한다."""
        if self._local_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            self._local_model = SentenceTransformer(self.model, trust_remote_code=True)
            logger.info("JinaEmbedder: local SentenceTransformer model loaded (%s)", self.model)
        except ImportError as exc:
            raise ImportError(
                "local 백엔드는 sentence-transformers 가 필요합니다. "
                "uv pip install sentence-transformers 후 재시도하세요."
            ) from exc

    def _local_encode(self, texts: list[str]) -> list[list[float]]:
        """sentence-transformers 표준 encode."""
        self._init_local_model()
        vecs = self._local_model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [v[: self.dim].tolist() for v in vecs]

    def _local_late_encode(self, texts: list[str]) -> list[list[float]]:
        """Local late chunking: tokenize → forward → mean-pool per chunk.

        texts 를 연결한 full document 를 토크나이즈하고, 각 chunk 의
        char span → token span 으로 변환 후 해당 토큰 임베딩을 mean-pool 한다.
        """
        self._init_local_model()
        try:
            import torch  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("local late chunking 은 torch 가 필요합니다.") from exc

        tokenizer = self._local_model.tokenizer  # type: ignore[union-attr]

        # 연결 문서 구성 (char boundaries 추적)
        sep = "\n"
        document = sep.join(texts)
        char_boundaries: list[tuple[int, int]] = []
        pos = 0
        for i, t in enumerate(texts):
            char_boundaries.append((pos, pos + len(t)))
            pos += len(t) + (len(sep) if i < len(texts) - 1 else 0)

        encoding = tokenizer(
            document,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        )
        offset_mapping = encoding["offset_mapping"][0]  # (seq_len, 2)

        with torch.no_grad():
            model_obj = self._local_model[0] if hasattr(self._local_model, "__getitem__") else self._local_model  # type: ignore[union-attr]
            outputs = model_obj.auto_model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )
            token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden)

        results: list[list[float]] = []
        for char_s, char_e in char_boundaries:
            mask = [
                (char_s <= int(t_s) and int(t_e) <= char_e) for t_s, t_e in offset_mapping.tolist()
            ]
            selected = token_embeddings[[i for i, m in enumerate(mask) if m]]
            if selected.shape[0] == 0:
                # fallback: zero vector
                results.append([0.0] * self.dim)
            else:
                pooled = selected.mean(dim=0)
                # L2 normalize
                norm = pooled.norm()
                if norm > 0:
                    pooled = pooled / norm
                results.append(pooled[: self.dim].tolist())

        return results

    # ------------------------------------------------------------------
    # Public interface (satisfies core.interfaces.Embedder)
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """texts 각 항목을 임베딩 벡터로 변환한다 (표준 경로).

        ctx_text (메타 접두어 포함) 를 입력으로 사용하는 것을 권장한다.
        """
        logger.debug("JinaEmbedder.embed: %d texts", len(texts))
        return self._embed_call(texts)

    def embed_late(
        self,
        document: str,
        boundaries: list[tuple[int, int]],
    ) -> list[list[float]]:
        """Late Chunking: document 컨텍스트 기반으로 각 boundary 구간 벡터를 반환한다.

        Args:
            document: 전체 연결 문서 (chunk.text 들을 "\\n" 으로 join 한 문자열).
            boundaries: 각 chunk 의 char offset 쌍 (start, end) 리스트.
                        document[s:e] 가 해당 chunk text 를 정확히 복원해야 한다.

        Returns:
            경계 수 == len(boundaries) 의 임베딩 벡터 리스트.

        구현 전략:
        - API backend: document[s:e] 로 chunk text 를 복원한 후
          late_chunking=true 로 Jina API 에 전달 (Jina 가 내부에서 전체 컨텍스트 조건화).
        - Local backend: tokenize(document) → token embeddings → mean-pool per char span.
        """
        logger.debug("JinaEmbedder.embed_late: %d boundaries", len(boundaries))
        texts = [document[s:e] for s, e in boundaries]
        return self._embed_late_call(texts)

    def embed_chunks(
        self,
        chunks: list[Chunk],
        full_document: str | None = None,
        late: bool = True,
    ) -> list[Chunk]:
        """chunks 에 embedding 을 설정한 NEW Chunk 목록을 반환한다 (non-mutating).

        Args:
            chunks: embedding=None 인 Chunk 목록.
            full_document: 제공 시 Late Chunking 경로 활성화.
                           parsed.markdown 또는 "\n".join(c.text) 를 전달한다.
                           None 이면 표준 경로 (ctx_text 임베딩).
            late: full_document 와 함께 Late Chunking 사용 여부 (기본 True).

        Returns:
            embedding 이 채워진 새 Chunk 목록.

        Late 경로 document 구성:
            - chunk.text 연결로 alignment-guaranteed document 를 내부 구성한다.
            - boundaries = 각 chunk.text 의 연결 문자열 내 char offset.
            - full_document 파라미터는 late 모드 활성화 트리거이며,
              실제 경계 문서는 chunk.text 연결로 재구성한다.
              (parsed.markdown 과 chunk.text 간 공백/헤더 불일치로 인한 alignment 오류 방지)
        """
        if not chunks:
            return []

        if full_document is not None and late:
            # Late Chunking 경로: chunk.text 연결 document + char boundaries
            sep = "\n"
            doc_parts: list[str] = [c.text for c in chunks]
            concat_doc = sep.join(doc_parts)

            boundaries: list[tuple[int, int]] = []
            pos = 0
            for i, text in enumerate(doc_parts):
                boundaries.append((pos, pos + len(text)))
                pos += len(text) + (len(sep) if i < len(doc_parts) - 1 else 0)

            logger.info(
                "embed_chunks: late chunking (%d chunks, doc_len=%d chars)",
                len(chunks),
                len(concat_doc),
            )
            vectors = self.embed_late(concat_doc, boundaries)
        else:
            # 표준 경로: ctx_text 임베딩
            logger.info("embed_chunks: standard embedding (%d chunks)", len(chunks))
            vectors = self.embed([c.ctx_text for c in chunks])

        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"embed_chunks: 반환 벡터 수({len(vectors)}) != 청크 수({len(chunks)})"
            )

        result: list[Chunk] = []
        for chunk, vec in zip(chunks, vectors):
            new_chunk = copy.copy(chunk)
            # dim 절단 (MRL 지원)
            new_chunk.embedding = vec[: self.dim]
            result.append(new_chunk)

        return result
