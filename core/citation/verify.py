"""core.citation.verify — 인용 overlap 검증 (환각 인용 차단).

## 설계

### overlap_score
claim 과 chunk_text 간 **unigram containment** 을 측정한다.

    score = |claim_tokens ∩ chunk_tokens| / |claim_tokens|

- 비대칭(asymmetric): 분모는 claim 토큰 수 — 긴 청크 대비 짧은 claim 에 robust.
- 토큰화: 소문자, 구두점 제거, 내장 불용어 제거.
- claim_tokens 가 공집합이면 0.0 반환.

### verify_citations
각 인용에 대해 그 인용이 붙은 claim 과의 max overlap 을 계산한다.
threshold 이상이면 verified, 미만이면 unverified (환각 인용 의심).

### strip_unverified
미검증 [id] 태그만 제거하고 claim 문장 본문은 유지한다.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

from core.models import Citation, RetrievedChunk

# ---------------------------------------------------------------------------
# 내장 불용어 집합 (외부 의존 없이 순수 파이썬)
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "as",
        "also",
        "both",
        "each",
        "all",
        "which",
        "than",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "between",
        "about",
        "against",
        "while",
        "so",
        "not",
        "no",
    }
)

# [id] 태그 패턴 (chunk id는 \w, /, ., : 등 포함)
_TAG_PATTERN = re.compile(r"\[([^\[\]]+)\]")

# 문장 분리: 마침표/느낌표/물음표 뒤에 공백이 오는 경우
_SENTENCE_SPLITTER = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """인용 overlap 검증 결과."""

    verified: list[Citation] = field(default_factory=list)
    unverified: list[Citation] = field(default_factory=list)
    faithfulness: float = 0.0  # verified / 전체 인용 수 (0~1)
    all_grounded: bool = False


# ---------------------------------------------------------------------------
# 토큰화 헬퍼
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> frozenset[str]:
    """텍스트를 소문자로 만들고 구두점을 제거한 후 불용어를 제거한 토큰 집합을 반환한다."""
    # 구두점 제거
    translator = str.maketrans("", "", string.punctuation)
    cleaned = text.lower().translate(translator)
    tokens = cleaned.split()
    return frozenset(t for t in tokens if t and t not in _STOPWORDS)


# ---------------------------------------------------------------------------
# 1. extract_claims
# ---------------------------------------------------------------------------


def extract_claims(answer: str) -> list[tuple[str, list[str]]]:
    """답변 텍스트를 문장 단위로 분할하고, 각 문장의 인용 태그를 추출한다.

    Parameters
    ----------
    answer:
        LLM 이 생성한 영문 답변. 인용 태그 [chunk_id] 를 포함.

    Returns
    -------
    list[tuple[str, list[str]]]
        [(claim_sentence, [cited_ids]), ...].
        claim_sentence 는 인용 태그가 제거된 깨끗한 텍스트.
        태그가 없는 문장도 빈 리스트와 함께 포함된다.
    """
    sentences = _SENTENCE_SPLITTER.split(answer.strip())
    result: list[tuple[str, list[str]]] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 태그 추출
        cited_ids = [m.group(1).strip() for m in _TAG_PATTERN.finditer(sentence)]

        # 태그 제거 후 공백 정리
        clean = _TAG_PATTERN.sub("", sentence).strip()
        # 연속 공백 → 단일 공백
        clean = re.sub(r"\s{2,}", " ", clean)

        result.append((clean, cited_ids))

    return result


# ---------------------------------------------------------------------------
# 2. overlap_score
# ---------------------------------------------------------------------------


def overlap_score(claim: str, chunk_text: str) -> float:
    """claim 과 chunk_text 간 unigram containment 를 계산한다.

    containment = |claim_tokens ∩ chunk_tokens| / |claim_tokens|

    Parameters
    ----------
    claim:
        인용 태그가 제거된 주장 문장 (또는 raw 문장).
    chunk_text:
        인용 근거 청크 원문.

    Returns
    -------
    float
        0.0~1.0. claim_tokens 가 공집합이면 0.0.
    """
    claim_tokens = _tokenize(claim)
    if not claim_tokens:
        return 0.0

    chunk_tokens = _tokenize(chunk_text)
    intersection = claim_tokens & chunk_tokens
    return len(intersection) / len(claim_tokens)


# ---------------------------------------------------------------------------
# 3. verify_citations
# ---------------------------------------------------------------------------


def verify_citations(
    answer: str,
    citations: list[Citation],
    chunks: list[RetrievedChunk],
    *,
    threshold: float = 0.3,
) -> VerificationResult:
    """생성 답변의 각 인용이 실제 청크 원문에 근거하는지 검증한다.

    Parameters
    ----------
    answer:
        LLM 이 생성한 영문 답변 (answer_en). 인용 태그 [chunk_id] 포함.
    citations:
        Generator._parse_citations 가 반환한 Citation 목록.
    chunks:
        검색 단계의 RetrievedChunk 목록 (chunk_id → chunk.text 인덱스 구성용).
    threshold:
        overlap_score 최솟값. 이상이면 verified, 미만이면 unverified.
        기본값 0.3: 짧은 claim (~5 토큰)에서 최소 1~2 핵심 토큰 일치를 요구하는
        균형점. 너무 낮으면 환각 인용 통과, 너무 높으면 동의어 표현 허용 안 됨.

    Returns
    -------
    VerificationResult
    """
    if not citations:
        return VerificationResult(
            verified=[],
            unverified=[],
            faithfulness=1.0,  # 인용이 없으면 vacuously 검증됨
            all_grounded=True,
        )

    # chunk_id → chunk.text 인덱스
    chunk_text_map: dict[str, str] = {rc.chunk.id: rc.chunk.text for rc in chunks}

    # claim → cited_ids 매핑 구성
    claims = extract_claims(answer)
    # chunk_id → list of claim sentences
    id_to_claims: dict[str, list[str]] = {}
    for claim_sentence, cited_ids in claims:
        for cid in cited_ids:
            id_to_claims.setdefault(cid, []).append(claim_sentence)

    verified: list[Citation] = []
    unverified: list[Citation] = []

    for citation in citations:
        cid = citation.chunk_id
        chunk_text = chunk_text_map.get(cid)

        # 청크가 인덱스에 없으면 unverified
        if chunk_text is None:
            unverified.append(citation)
            continue

        # 이 citation 을 참조하는 claim 이 없으면 unverified
        claim_sentences = id_to_claims.get(cid, [])
        if not claim_sentences:
            unverified.append(citation)
            continue

        # claim 들 중 max overlap 으로 판정
        max_score = max(overlap_score(cl, chunk_text) for cl in claim_sentences)

        if max_score >= threshold:
            # quote 채우기: claim 과 chunk_text 간 공유 토큰이 등장하는 구간
            updated_citation = citation.model_copy(
                update={"quote": _extract_quote(claim_sentences[0], chunk_text)}
            )
            verified.append(updated_citation)
        else:
            unverified.append(citation)

    total = len(citations)
    faithfulness = len(verified) / total if total > 0 else 1.0
    all_grounded = len(unverified) == 0

    return VerificationResult(
        verified=verified,
        unverified=unverified,
        faithfulness=faithfulness,
        all_grounded=all_grounded,
    )


def _extract_quote(claim: str, chunk_text: str, max_len: int = 200) -> str:
    """claim 과 공유 토큰이 가장 많이 등장하는 chunk_text 구간을 반환한다.

    간단한 슬라이딩 윈도우 대신, claim 의 첫 번째 공유 토큰이 등장하는
    문장을 chunk_text 에서 찾는다. 없으면 앞부분을 반환한다.
    """
    claim_tokens = _tokenize(claim)
    # chunk 를 문장 단위로 분리해 가장 많이 겹치는 문장을 찾는다
    chunk_sentences = _SENTENCE_SPLITTER.split(chunk_text)
    best_sentence = ""
    best_count = 0
    for sent in chunk_sentences:
        common = len(_tokenize(sent) & claim_tokens)
        if common > best_count:
            best_count = common
            best_sentence = sent

    if best_sentence:
        return best_sentence[:max_len] if len(best_sentence) > max_len else best_sentence

    # fallback: chunk 앞부분
    return chunk_text[:max_len] if len(chunk_text) > max_len else chunk_text


# ---------------------------------------------------------------------------
# 4. strip_unverified
# ---------------------------------------------------------------------------


def strip_unverified(answer: str, unverified: list[Citation]) -> str:
    """답변에서 미검증 인용 태그만 제거한다. 주장 문장 본문은 유지된다.

    Parameters
    ----------
    answer:
        원본 LLM 답변 텍스트.
    unverified:
        verify_citations 가 반환한 미검증 Citation 목록.

    Returns
    -------
    str
        미검증 [chunk_id] 태그가 제거된 답변.
        verified 태그와 문장 본문은 그대로 유지된다.
    """
    if not unverified:
        return answer

    result = answer
    for citation in unverified:
        # chunk_id 에 포함된 regex 특수 문자(./:) 이스케이프
        escaped_id = re.escape(citation.chunk_id)
        # 태그 앞의 공백도 함께 제거 (orphan space 방지)
        pattern = re.compile(r"\s*\[" + escaped_id + r"\]")
        result = pattern.sub("", result)

    return result
