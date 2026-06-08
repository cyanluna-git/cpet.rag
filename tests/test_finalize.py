"""tests/test_finalize.py — back_translate_answer / finalize_answer 유닛 테스트.

실제 LLM 호출 없음 — mock translator 사용.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from serving.generation import GenerationResult, back_translate_answer, finalize_answer

# ---------------------------------------------------------------------------
# Mock translator 헬퍼
# ---------------------------------------------------------------------------


def _make_translator(prefix: str = "[KO] ") -> MagicMock:
    """en2ko 가 prefix 를 붙여 반환하는 mock translator.

    placeholder ⟦C{n}⟧ 는 그대로 통과시키므로 태그 보존 로직을 검증할 수 있다.
    """
    translator = MagicMock()
    translator.en2ko.side_effect = lambda text: prefix + text
    return translator


def _passthrough_translator() -> MagicMock:
    """en2ko 가 입력을 그대로 반환하는 translator (순수 복원 검증용)."""
    translator = MagicMock()
    translator.en2ko.side_effect = lambda text: text
    return translator


# ---------------------------------------------------------------------------
# 인용 태그 보존 핵심 테스트
# ---------------------------------------------------------------------------


class TestBackTranslateAnswerCitationPreservation:
    def test_simple_citation_preserved(self) -> None:
        """단일 [c1] 태그가 번역 후에도 동일하게 보존된다."""
        translator = _make_translator()
        answer = "Endurance training increases VO2max [c1]."
        result = back_translate_answer(answer, translator)

        assert "[c1]" in result

    def test_doi_style_citation_preserved(self) -> None:
        """DOI 형태 [10.1016/j.x_5] 태그가 보존된다."""
        translator = _make_translator()
        answer = "Lactate threshold rises [10.1016/j.x_5]."
        result = back_translate_answer(answer, translator)

        assert "[10.1016/j.x_5]" in result

    def test_complex_chunk_id_preserved(self) -> None:
        """복합 chunk id (::, ., /) 태그가 보존된다."""
        translator = _make_translator()
        chunk_id = "10.1234/exercise.2023::5"
        answer = f"VO2max improved [{chunk_id}]."
        result = back_translate_answer(answer, translator)

        assert f"[{chunk_id}]" in result

    def test_tag_replaced_with_placeholder_before_translation(self) -> None:
        """번역기에 전달되는 텍스트에 [c1] 이 없고 ⟦C0⟧ 가 있다.

        Spy: en2ko 호출 인수를 캡처해 보호 상태를 직접 검증한다.
        """
        captured: list[str] = []

        translator = MagicMock()
        translator.en2ko.side_effect = lambda t: (captured.append(t), t)[1]

        answer = "VO2max increases [c1]."
        back_translate_answer(answer, translator)

        assert len(captured) == 1
        translated_input = captured[0]

        # 보호됨: [c1] 대신 ⟦C0⟧
        assert "[c1]" not in translated_input
        assert "⟦C0⟧" in translated_input

    def test_no_placeholder_remaining_in_result(self) -> None:
        """번역 결과에 ⟦C…⟧ placeholder 가 남지 않는다."""
        translator = _make_translator()
        answer = "Claim one [c1]. Claim two [c2]."
        result = back_translate_answer(answer, translator)

        assert not re.search(r"⟦C\d+⟧", result), "Placeholder ⟦C…⟧ leaked into result"

    def test_multiple_tags_all_preserved(self) -> None:
        """여러 태그 [c1], [c2], [10.1016/j.x_5] 가 모두 보존된다."""
        translator = _make_translator()
        answer = (
            "Mitochondrial density [c1]. Lactate threshold [c2]. " "Stroke volume [10.1016/j.x_5]."
        )
        result = back_translate_answer(answer, translator)

        assert "[c1]" in result
        assert "[c2]" in result
        assert "[10.1016/j.x_5]" in result

    def test_duplicate_tags_all_preserved(self) -> None:
        """같은 [c1] 이 여러 번 등장해도 모두 복원된다."""
        translator = _make_translator()
        answer = "First [c1]. Second [c1]. Third [c1]."
        result = back_translate_answer(answer, translator)

        # [c1] 이 3번 모두 존재해야 함
        occurrences = len(re.findall(r"\[c1\]", result))
        assert occurrences == 3

    def test_translation_actually_applied(self) -> None:
        """태그 보존과 동시에 실제 번역이 적용된다 (prefix 마커 확인)."""
        translator = _make_translator(prefix="[KO] ")
        answer = "VO2max increases [c1]."
        result = back_translate_answer(answer, translator)

        # 번역기가 호출됨 (prefix 존재)
        assert "[KO]" in result
        # 인용 태그도 보존됨
        assert "[c1]" in result


# ---------------------------------------------------------------------------
# protect_citations=False 경로
# ---------------------------------------------------------------------------


class TestBackTranslateAnswerNoProtect:
    def test_no_protect_calls_translator_directly(self) -> None:
        """protect_citations=False 이면 번역기에 원문을 그대로 전달한다."""
        translator = _make_translator()
        answer = "VO2max increases [c1]."
        result = back_translate_answer(answer, translator, protect_citations=False)

        # 번역기는 1회 호출됨
        translator.en2ko.assert_called_once_with(answer)
        assert result is not None

    def test_no_protect_returns_translator_output(self) -> None:
        """protect_citations=False 이면 translator.en2ko 결과가 그대로 반환된다."""
        translator = _make_translator(prefix="[KO] ")
        answer = "Some answer."
        result = back_translate_answer(answer, translator, protect_citations=False)

        assert result == "[KO] Some answer."


# ---------------------------------------------------------------------------
# 태그 없는 답변
# ---------------------------------------------------------------------------


class TestBackTranslateAnswerNoTags:
    def test_no_tags_translated_normally(self) -> None:
        """인용 태그가 없는 답변도 정상 번역된다."""
        translator = _make_translator(prefix="[KO] ")
        answer = "VO2max is a key aerobic capacity indicator."
        result = back_translate_answer(answer, translator)

        assert "[KO]" in result
        # VO2max 텍스트가 번역 결과에 포함
        assert "VO2max" in result

    def test_no_tags_translator_called_once(self) -> None:
        """태그 없을 때도 translator.en2ko 는 정확히 1회 호출된다."""
        translator = _passthrough_translator()
        back_translate_answer("No citations here.", translator)
        translator.en2ko.assert_called_once()


# ---------------------------------------------------------------------------
# finalize_answer 테스트
# ---------------------------------------------------------------------------


class TestFinalizeAnswer:
    def test_refused_returns_ko_refusal_no_translator_call(self) -> None:
        """refused=True 이면 KO 거부 메시지를 반환하고 translator 를 호출하지 않는다."""
        translator = _make_translator()
        gen_result = GenerationResult(
            answer_en="I cannot answer from the provided sources.",
            citations=[],
            refused=True,
            used_chunk_ids=[],
        )

        result = finalize_answer(gen_result, translator)

        # 한국어 거부 메시지
        assert isinstance(result, str)
        assert len(result) > 0
        # translator 미호출
        translator.en2ko.assert_not_called()

    def test_refused_message_is_korean(self) -> None:
        """refused 응답이 한국어 문자를 포함한다."""
        translator = _passthrough_translator()
        gen_result = GenerationResult(
            answer_en="I cannot answer from the provided sources.",
            refused=True,
        )
        result = finalize_answer(gen_result, translator)

        # 한글 유니코드 범위(AC00-D7A3) 포함 확인
        assert any("가" <= ch <= "힣" for ch in result)

    def test_normal_answer_back_translated(self) -> None:
        """refused=False 이면 answer_en 을 back_translate_answer 로 번역한다."""
        translator = _make_translator(prefix="[KO] ")
        gen_result = GenerationResult(
            answer_en="VO2max increases with training [c1].",
            refused=False,
        )

        result = finalize_answer(gen_result, translator)

        assert "[KO]" in result
        assert "[c1]" in result

    def test_normal_answer_citations_preserved(self) -> None:
        """정상 답변의 인용 태그가 최종 출력에 보존된다."""
        translator = _make_translator()
        doi_id = "10.1016/j.resp.2023::3"
        gen_result = GenerationResult(
            answer_en=f"Stroke volume increases [{doi_id}].",
            refused=False,
        )

        result = finalize_answer(gen_result, translator)

        assert f"[{doi_id}]" in result

    def test_finalize_answer_returns_str(self) -> None:
        """finalize_answer 는 항상 str 을 반환한다 (refused/normal 양쪽)."""
        translator = _passthrough_translator()

        for refused in [True, False]:
            gen_result = GenerationResult(answer_en="Test answer [c1].", refused=refused)
            result = finalize_answer(gen_result, translator)
            assert isinstance(result, str)
