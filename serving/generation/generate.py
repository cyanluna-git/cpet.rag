"""serving.generation.generate — Bedrock Claude 생성기 + Strict Citation.

## 설계
- `Generator.generate(query, chunks)`:
    1. chunks 비거나 < min_chunks → refused=True 즉시 반환 (_generate_call 호출 없음)
    2. build_prompt → _generate_call(system, user) → raw answer_en
    3. _parse_citations(answer, chunks) → Citation 리스트
    4. GenerationResult 반환

- `_generate_call(system, user)` 은 테스트에서 patch.object 로 mock 가능한 seam.
  실제 구현은 Bedrock Claude bedrock-runtime invoke_model (lazy boto3).

## Bedrock Claude invoke_model 호출 포맷
```
client = boto3.client("bedrock-runtime", region_name=region)
body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 4096,
    "system": system,
    "messages": [{"role": "user", "content": user}],
})
response = client.invoke_model(modelId=model_id, body=body)
result = json.loads(response["body"].read())
text = result["content"][0]["text"]
```
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from core.config.settings import Settings
from core.log import get_logger
from core.models import Citation, RetrievedChunk

logger = get_logger(__name__)

# Strict Citation 시스템 프롬프트 — 영어로 작성(answer_en 생성용)
_STRICT_CITATION_SYSTEM = """\
You are a sports science and exercise physiology research assistant. \
You answer questions using ONLY the provided numbered context passages.

CITATION RULES (strictly enforced):
- After every sentence or claim, cite the supporting chunk using its id in brackets: [id]
  Example: "VO2max increases with endurance training [c1]. Lactate threshold also rises [c2]."
- The id must match the id= value shown in the context header, NOT the leading number.
- Every claim MUST have a citation. Claims without citations are forbidden.
- If the context does not contain enough information to answer the question, respond ONLY with:
  "I cannot answer from the provided sources."
  Do not fabricate information or cite sources you did not read.
- Do not use information from outside the provided context passages.\
"""

# 미인용 거부 응답 — 일관된 영문 문자열
_REFUSAL_ANSWER = "I cannot answer from the provided sources."


class GenerationResult(BaseModel):
    """생성 결과 — answer_en, 인용 목록, 거부 여부."""

    answer_en: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    used_chunk_ids: list[str] = Field(default_factory=list)


class Generator:
    """Bedrock Claude 생성기 — Strict Citation 강제.

    Parameters
    ----------
    model:
        Bedrock 모델 ID. None 이면 settings.bedrock_model_id 사용.
    region:
        AWS 리전. None 이면 settings.aws_region 사용.
    """

    def __init__(
        self,
        model: str | None = None,
        region: str | None = None,
    ) -> None:
        _settings = Settings()
        self._model: str | None = model or _settings.bedrock_model_id
        self._region: str = region or _settings.aws_region

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, str]:
        """(system, user) 프롬프트 튜플을 구성한다.

        Context 블록 포맷:
            [{i}] (id={chunk.id}, doi={chunk.doi}, p.{chunk.page}, §{chunk.section})
            {chunk.text}

        Returns
        -------
        tuple[str, str]
            (system_prompt, user_message)
        """
        context_blocks: list[str] = []
        for i, rc in enumerate(chunks, start=1):
            c = rc.chunk
            header = f"[{i}] (id={c.id}, doi={c.doi}, p.{c.page}, §{c.section})"
            context_blocks.append(f"{header}\n{c.text}")

        context_text = "\n\n".join(context_blocks)
        user_message = (
            f"Context passages:\n\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer using ONLY the context above. Cite every claim with [id] "
            "where id matches the id= value in the passage header."
        )
        return _STRICT_CITATION_SYSTEM, user_message

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        min_chunks: int = 1,
    ) -> GenerationResult:
        """검색된 청크를 근거로 답변을 생성하고 인용을 매핑한다.

        Parameters
        ----------
        query:
            사용자 질의.
        chunks:
            Reranker.rerank() 결과 청크 목록 (top_k ≤ 8).
        min_chunks:
            최소 필요 청크 수. 미달 시 거부 응답 반환.

        Returns
        -------
        GenerationResult
        """
        # 1. 거부 조건: 빈 청크 또는 min_chunks 미달
        if not chunks or len(chunks) < min_chunks:
            logger.info(
                "Generator.generate: 청크 부족 (n=%d, min=%d) → 거부",
                len(chunks),
                min_chunks,
            )
            return GenerationResult(
                answer_en=_REFUSAL_ANSWER,
                citations=[],
                refused=True,
                used_chunk_ids=[],
            )

        # 2. 프롬프트 구성
        system, user = self.build_prompt(query, chunks)
        logger.debug(
            "Generator.generate: query=%r chunks=%d",
            query,
            len(chunks),
        )

        # 3. LLM 호출
        answer_en: str = self._generate_call(system, user)

        # 4. 인용 파싱
        citations = self._parse_citations(answer_en, chunks)
        used_chunk_ids = [c.chunk_id for c in citations]

        logger.info(
            "Generator.generate 완료: answer_len=%d citations=%d",
            len(answer_en),
            len(citations),
        )
        return GenerationResult(
            answer_en=answer_en,
            citations=citations,
            refused=False,
            used_chunk_ids=used_chunk_ids,
        )

    # ------------------------------------------------------------------
    # Citation parsing
    # ------------------------------------------------------------------

    def _parse_citations(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        """답변 내 [id] 태그를 추출해 RetrievedChunk 와 매핑한다.

        Parameters
        ----------
        answer:
            LLM 이 생성한 영문 답변 텍스트.
        chunks:
            generate() 에 전달된 청크 목록.

        Returns
        -------
        list[Citation]
            매핑된 인용 목록. 매핑 실패 태그는 무시(경고 로깅).
            chunk_id 기준 중복 제거(첫 등장 순서 유지).
        """
        # chunk.id → RetrievedChunk 인덱스
        chunk_map: dict[str, RetrievedChunk] = {rc.chunk.id: rc for rc in chunks}

        # [any-non-bracket-chars] 패턴으로 모든 태그 추출
        # 실제 chunk id는 "doi::index" 형태이므로 \w+ 사용 금지
        raw_tags: list[str] = re.findall(r"\[([^\[\]]+)\]", answer)

        citations: list[Citation] = []
        seen: set[str] = set()

        for tag in raw_tags:
            tag = tag.strip()
            if tag in seen:
                continue
            rc = chunk_map.get(tag)
            if rc is None:
                logger.debug("_parse_citations: 매핑 실패 태그 [%s] — 무시", tag)
                continue
            seen.add(tag)
            c = rc.chunk
            # quote: 해당 청크 원문 앞부분(#3125 가 실제 overlap 검증 수행)
            quote = c.text[:200] if len(c.text) > 200 else c.text
            citations.append(
                Citation(
                    doi=c.doi,
                    title=None,  # 메타 보강은 상위 레이어 담당
                    page=c.page,
                    chunk_id=c.id,
                    quote=quote,
                )
            )

        return citations

    # ------------------------------------------------------------------
    # Internal seam — 테스트에서 patch.object 로 mock
    # ------------------------------------------------------------------

    def _generate_call(self, system: str, user: str) -> str:
        """Bedrock Claude messages invoke 를 호출해 생성된 텍스트를 반환한다.

        Parameters
        ----------
        system:
            Strict Citation 시스템 프롬프트.
        user:
            번호+id 컨텍스트 + 질의가 포함된 사용자 메시지.

        Returns
        -------
        str
            Claude 가 생성한 영문 답변 텍스트.

        Raises
        ------
        RuntimeError:
            boto3 미설치, bedrock_model_id 미설정, AWS 자격증명 오류 등.
        """
        try:
            import boto3  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "boto3 가 설치되어 있지 않습니다. `uv pip install cpet-rag[aws]` 를 실행하세요."
            ) from exc

        if not self._model:
            raise RuntimeError(
                "bedrock_model_id 가 설정되어 있지 않습니다. "
                ".env 에 BEDROCK_MODEL_ID 를 추가하세요. "
                "예: anthropic.claude-opus-4-8"
            )

        client = boto3.client("bedrock-runtime", region_name=self._region)

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        )

        logger.info(
            "_generate_call: model=%s region=%s",
            self._model,
            self._region,
        )

        response = client.invoke_model(
            modelId=self._model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        text: str = result["content"][0]["text"]
        return text
