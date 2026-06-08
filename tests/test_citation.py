"""tests/test_citation.py — core.citation.verify 유닛 테스트 (순수 로직).

모든 테스트는 LLM/네트워크 없이 로컬에서 실행 가능하다.
"""

from __future__ import annotations

import pytest

from core.citation import (
    VerificationResult,
    extract_claims,
    overlap_score,
    strip_unverified,
    verify_citations,
)
from core.models import Citation, Chunk, RetrievedChunk

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, text: str, doi: str = "10.0000/test") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=chunk_id,
            doi=doi,
            section="Methods",
            text=text,
            ctx_text=f"[ctx] {chunk_id}",
            page=1,
            chunk_index=0,
            source="test",
        ),
        score=0.9,
    )


def _make_citation(chunk_id: str, doi: str = "10.0000/test") -> Citation:
    return Citation(
        doi=doi,
        title=None,
        page=1,
        chunk_id=chunk_id,
        quote="placeholder",
    )


# ---------------------------------------------------------------------------
# overlap_score 테스트
# ---------------------------------------------------------------------------


class TestOverlapScore:
    def test_identical_text_returns_near_one(self) -> None:
        """동일 텍스트 → ~1.0."""
        text = "mitochondrial density increases with endurance training"
        score = overlap_score(text, text)
        assert score == pytest.approx(1.0)

    def test_unrelated_text_returns_near_zero(self) -> None:
        """무관한 텍스트 → ~0.0."""
        claim = "mitochondrial density increases"
        chunk = "stock market volatility affects investor behavior"
        score = overlap_score(claim, chunk)
        # 공유 토큰 없어야 함
        assert score < 0.1

    def test_ac_example_above_half(self) -> None:
        """AC 수락 조건: mitochondrial density increases vs chunk → > 0.5."""
        claim = "mitochondrial density increases"
        chunk = "Endurance training increases mitochondrial density in skeletal muscle"
        score = overlap_score(claim, chunk)
        assert score > 0.5

    def test_partial_overlap_midrange(self) -> None:
        """부분 겹침 → 0 초과 1 미만의 중간값."""
        claim = "lactate threshold and VO2max improve"
        chunk = "VO2max is a key predictor of endurance performance in athletes"
        score = overlap_score(claim, chunk)
        assert 0.0 < score < 1.0

    def test_empty_claim_returns_zero(self) -> None:
        """빈 claim → 0.0 (div-by-zero 없음)."""
        assert overlap_score("", "some text here") == 0.0

    def test_stopword_only_claim_returns_zero(self) -> None:
        """불용어만 있는 claim → 0.0."""
        score = overlap_score("the and of", "mitochondrial density")
        assert score == 0.0

    def test_case_insensitive(self) -> None:
        """대소문자 무관."""
        score1 = overlap_score("Mitochondrial Density", "mitochondrial density increases")
        score2 = overlap_score("mitochondrial density", "Mitochondrial Density increases")
        assert score1 == pytest.approx(score2)

    def test_punctuation_ignored(self) -> None:
        """구두점 무시."""
        score1 = overlap_score("VO2max, increases.", "VO2max increases with training")
        score2 = overlap_score("VO2max increases", "VO2max increases with training")
        assert score1 == pytest.approx(score2)


# ---------------------------------------------------------------------------
# extract_claims 테스트
# ---------------------------------------------------------------------------


class TestExtractClaims:
    def test_single_sentence_single_tag(self) -> None:
        """단일 문장, 단일 태그."""
        answer = "Mitochondrial density increases [c1]."
        claims = extract_claims(answer)
        assert len(claims) == 1
        claim_text, cited_ids = claims[0]
        assert "c1" in cited_ids
        assert "[c1]" not in claim_text
        assert "Mitochondrial density increases" in claim_text

    def test_multi_tag_per_sentence(self) -> None:
        """인접 태그 [c1][c2] — 두 개 모두 추출."""
        answer = "VO2max and lactate both improve [c1][c2]."
        claims = extract_claims(answer)
        assert len(claims) == 1
        _, cited_ids = claims[0]
        assert "c1" in cited_ids
        assert "c2" in cited_ids

    def test_multi_sentence_split(self) -> None:
        """여러 문장 분리."""
        answer = (
            "Mitochondrial density increases [c1]. "
            "Lactate threshold also rises [c2]. "
            "Cardiac output improves [c3]."
        )
        claims = extract_claims(answer)
        assert len(claims) == 3
        assert "c1" in claims[0][1]
        assert "c2" in claims[1][1]
        assert "c3" in claims[2][1]

    def test_tag_removed_from_claim_text(self) -> None:
        """태그가 claim 텍스트에서 완전히 제거된다."""
        answer = "Training effect [10.1234/exercise.2023::5]."
        claims = extract_claims(answer)
        claim_text, cited_ids = claims[0]
        assert "[" not in claim_text
        assert "10.1234/exercise.2023::5" in cited_ids

    def test_sentence_without_tag_included(self) -> None:
        """태그 없는 문장도 빈 cited_ids 로 포함된다."""
        answer = "General statement. Cited statement [c1]."
        claims = extract_claims(answer)
        assert len(claims) == 2
        # 첫 문장: 태그 없음
        assert claims[0][1] == []
        # 두 번째: c1 태그
        assert "c1" in claims[1][1]

    def test_doi_period_not_split_sentence(self) -> None:
        """DOI 내부 마침표는 문장 분리 기준이 아니다."""
        answer = "Result from doi::5 chunk [10.1234/ex::5]. Next sentence [c2]."
        claims = extract_claims(answer)
        # "10.1234/ex." 는 문장 구분자가 아님 → 두 문장
        assert len(claims) == 2

    def test_clean_text_has_no_double_spaces(self) -> None:
        """태그 제거 후 연속 공백이 없다."""
        answer = "Effect [c1] observed [c2] clearly."
        claims = extract_claims(answer)
        claim_text = claims[0][0]
        assert "  " not in claim_text


# ---------------------------------------------------------------------------
# verify_citations 테스트
# ---------------------------------------------------------------------------


class TestVerifyCitations:
    def test_strong_overlap_verified(self) -> None:
        """claim 이 chunk 와 강하게 겹치면 verified, faithfulness 높음."""
        chunk_text = (
            "Endurance training increases mitochondrial density in skeletal muscle. "
            "This adaptation improves oxidative capacity significantly."
        )
        answer = "Mitochondrial density increases with endurance training [c1]."
        citations = [_make_citation("c1")]
        chunks = [_make_chunk("c1", chunk_text)]

        result = verify_citations(answer, citations, chunks)

        assert isinstance(result, VerificationResult)
        assert len(result.verified) == 1
        assert len(result.unverified) == 0
        assert result.faithfulness == pytest.approx(1.0)
        assert result.all_grounded is True

    def test_hallucinated_citation_unverified(self) -> None:
        """claim 이 chunk 와 무관(다른 주제) → unverified, faithfulness 낮음."""
        chunk_text = "Endurance training increases mitochondrial density in skeletal muscle."
        # 전혀 관계없는 claim 에 c1 을 인용
        answer = "Stock market volatility is driven by investor sentiment [c1]."
        citations = [_make_citation("c1")]
        chunks = [_make_chunk("c1", chunk_text)]

        result = verify_citations(answer, citations, chunks)

        assert len(result.verified) == 0
        assert len(result.unverified) == 1
        assert result.faithfulness == pytest.approx(0.0)
        assert result.all_grounded is False

    def test_mixed_verified_and_unverified(self) -> None:
        """일부 verified, 일부 unverified 혼합 케이스."""
        chunk1_text = "Mitochondrial density increases with endurance training."
        chunk2_text = "Lactate threshold rises with aerobic conditioning."

        answer = (
            "Mitochondrial density increases with training [c1]. "
            "Bond yields affect inflation [c2]."  # 환각: c2 와 무관
        )
        citations = [_make_citation("c1"), _make_citation("c2")]
        chunks = [
            _make_chunk("c1", chunk1_text),
            _make_chunk("c2", chunk2_text),
        ]

        result = verify_citations(answer, citations, chunks)

        verified_ids = {c.chunk_id for c in result.verified}
        unverified_ids = {c.chunk_id for c in result.unverified}
        assert "c1" in verified_ids
        assert "c2" in unverified_ids
        assert result.faithfulness == pytest.approx(0.5)
        assert result.all_grounded is False

    def test_chunk_not_in_index_unverified(self) -> None:
        """chunks 인덱스에 없는 chunk_id 를 가진 citation → unverified."""
        answer = "Some claim [unknown_id]."
        citations = [_make_citation("unknown_id")]
        chunks = [_make_chunk("c1", "some text")]

        result = verify_citations(answer, citations, chunks)

        assert len(result.unverified) == 1
        assert result.unverified[0].chunk_id == "unknown_id"

    def test_empty_citations_returns_faithful(self) -> None:
        """인용이 없으면 faithfulness=1.0, all_grounded=True (vacuous)."""
        result = verify_citations("No citations here.", [], [])
        assert result.faithfulness == pytest.approx(1.0)
        assert result.all_grounded is True
        assert result.verified == []
        assert result.unverified == []

    def test_custom_threshold_boundary(self) -> None:
        """threshold 경계: 점수 == threshold 이면 verified."""
        # c1 에 대해 overlap_score 가 정확히 threshold 이상이 되는 케이스 구성
        # claim 토큰: {mitochondrial, density} (불용어 제거 후)
        # chunk: 두 토큰 모두 포함 → 2/2 = 1.0 → threshold=0.9 이하이면 verified
        claim_part = "mitochondrial density"
        chunk_text = "mitochondrial density increases in muscle"
        answer = f"{claim_part} [c1]."
        citations = [_make_citation("c1")]
        chunks = [_make_chunk("c1", chunk_text)]

        # threshold=1.0 이면 경계 이상 (1.0 >= 1.0) → verified
        result_pass = verify_citations(answer, citations, chunks, threshold=1.0)
        assert len(result_pass.verified) == 1

    def test_below_threshold_unverified(self) -> None:
        """threshold 를 매우 높게 설정하면 unverified 처리."""
        chunk_text = "Mitochondrial density increases with endurance training."
        answer = "Mitochondrial density increases [c1]."
        citations = [_make_citation("c1")]
        chunks = [_make_chunk("c1", chunk_text)]

        # overlap=1.0, threshold=1.01 → 불가능하므로 threshold=2.0 사용 (float)
        # 실제로 max score 는 1.0 이므로 threshold > 1.0 이면 unverified
        result = verify_citations(answer, citations, chunks, threshold=1.01)
        assert len(result.unverified) == 1

    def test_verified_citation_quote_updated(self) -> None:
        """verified citation 의 quote 가 chunk 본문 기반으로 채워진다."""
        chunk_text = "Endurance training increases mitochondrial density in skeletal muscle."
        answer = "Mitochondrial density increases with endurance training [c1]."
        citations = [_make_citation("c1")]
        chunks = [_make_chunk("c1", chunk_text)]

        result = verify_citations(answer, citations, chunks)

        assert len(result.verified) == 1
        # quote 가 placeholder 에서 실제 텍스트로 교체됨
        assert result.verified[0].quote != "placeholder"
        assert len(result.verified[0].quote) > 0

    def test_citation_with_no_matching_claim_unverified(self) -> None:
        """citation 이 answer 의 어떤 claim 에서도 참조되지 않으면 unverified."""
        chunk_text = "VO2max is a key aerobic capacity indicator."
        # answer 에는 c1 태그가 없지만 citation 에는 있음
        answer = "General statement without any citation tags."
        citations = [_make_citation("c1")]
        chunks = [_make_chunk("c1", chunk_text)]

        result = verify_citations(answer, citations, chunks)

        assert len(result.unverified) == 1

    def test_doi_style_chunk_id(self) -> None:
        """실제 doi::index 형태 chunk id 도 올바르게 처리된다."""
        doi_id = "10.1234/exercise.2023::5"
        chunk_text = "VO2max improved significantly after 8 weeks of endurance training."
        answer = f"VO2max improved with endurance training [{doi_id}]."
        citations = [_make_citation(doi_id, doi="10.1234/exercise.2023")]
        chunks = [_make_chunk(doi_id, chunk_text, doi="10.1234/exercise.2023")]

        result = verify_citations(answer, citations, chunks)

        assert len(result.verified) == 1
        assert result.verified[0].chunk_id == doi_id


# ---------------------------------------------------------------------------
# strip_unverified 테스트
# ---------------------------------------------------------------------------


class TestStripUnverified:
    def test_removes_unverified_tag(self) -> None:
        """미검증 태그가 제거된다."""
        answer = "Some claim [c1]. Another claim [c2]."
        unverified = [_make_citation("c2")]
        result = strip_unverified(answer, unverified)
        assert "[c2]" not in result
        assert "Another claim" in result

    def test_keeps_verified_tag(self) -> None:
        """verified 태그는 그대로 유지된다."""
        answer = "Some claim [c1]. Another claim [c2]."
        unverified = [_make_citation("c2")]
        result = strip_unverified(answer, unverified)
        assert "[c1]" in result

    def test_keeps_sentence_body(self) -> None:
        """미검증 태그 제거 시 주장 문장 본문은 유지된다."""
        answer = "Mitochondrial density increases [c1]."
        unverified = [_make_citation("c1")]
        result = strip_unverified(answer, unverified)
        assert "Mitochondrial density increases" in result
        assert "[c1]" not in result

    def test_empty_unverified_returns_unchanged(self) -> None:
        """미검증 목록이 비어 있으면 답변이 그대로 반환된다."""
        answer = "Some claim [c1]. Another [c2]."
        result = strip_unverified(answer, [])
        assert result == answer

    def test_doi_chunk_id_stripped_safely(self) -> None:
        """DOI 형태 chunk_id 도 올바르게 이스케이프해서 제거된다."""
        doi_id = "10.1234/exercise.2023::5"
        answer = f"VO2max improved [{doi_id}]. Next sentence."
        unverified = [_make_citation(doi_id)]
        result = strip_unverified(answer, unverified)
        assert f"[{doi_id}]" not in result
        assert "VO2max improved" in result

    def test_no_orphan_double_space(self) -> None:
        """태그 제거 후 leading/orphan 공백이 남지 않는다."""
        answer = "Claim [c1] is important."
        unverified = [_make_citation("c1")]
        result = strip_unverified(answer, unverified)
        assert "  " not in result

    def test_multiple_unverified_tags_all_removed(self) -> None:
        """여러 미검증 태그가 모두 제거된다."""
        answer = "A [c1]. B [c2]. C [c3]."
        unverified = [_make_citation("c1"), _make_citation("c3")]
        result = strip_unverified(answer, unverified)
        assert "[c1]" not in result
        assert "[c3]" not in result
        assert "[c2]" in result


# ---------------------------------------------------------------------------
# 통합 시나리오: verify → strip 파이프라인
# ---------------------------------------------------------------------------


class TestVerifyAndStripPipeline:
    def test_pipeline_removes_hallucinated_keeps_verified(self) -> None:
        """verify → strip 파이프라인: 환각 태그 제거, 검증된 태그 유지."""
        chunk1_text = "Mitochondrial density increases with endurance training."
        chunk2_text = "Lactate threshold rises with aerobic conditioning."

        answer = (
            "Mitochondrial density increases [c1]. "
            "Bond market yields correlate with inflation [c2]."  # 환각
        )
        citations = [_make_citation("c1"), _make_citation("c2")]
        chunks = [
            _make_chunk("c1", chunk1_text),
            _make_chunk("c2", chunk2_text),
        ]

        vr = verify_citations(answer, citations, chunks)
        cleaned = strip_unverified(answer, vr.unverified)

        # c1 태그 유지
        assert "[c1]" in cleaned
        # c2 태그 제거
        assert "[c2]" not in cleaned
        # 양쪽 문장 본문 유지
        assert "Mitochondrial density increases" in cleaned
        assert "Bond market yields" in cleaned
        # faithfulness
        assert vr.faithfulness == pytest.approx(0.5)
