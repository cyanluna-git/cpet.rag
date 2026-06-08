"""tests.test_translate — BedrockTranslator 단위 테스트 (mock LLM).

실제 Bedrock 호출 없음. `_translate_call` 을 monkeypatch 로 대체.
"""

from __future__ import annotations

import pytest

from core import interfaces
from serving.retrieval import BedrockTranslator, EN2KO_GLOSSARY, KO2EN_GLOSSARY
from serving.retrieval.translate import _protect, _restore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def translator_passthrough(monkeypatch: pytest.MonkeyPatch) -> BedrockTranslator:
    """_translate_call 이 입력 텍스트를 그대로 반환하는 translator (passthrough mock).

    placeholder 는 건드리지 않으므로 보호/복원 로직을 투명하게 테스트할 수 있다.
    """
    t = BedrockTranslator()

    def mock_translate(text: str, src: str, tgt: str) -> str:  # noqa: ARG001
        # 실제 번역 없이 입력 그대로 반환 — placeholder 보존 확인용
        return text

    monkeypatch.setattr(t, "_translate_call", mock_translate)
    return t


@pytest.fixture()
def translator_ko2en_mock(monkeypatch: pytest.MonkeyPatch) -> BedrockTranslator:
    """ko2en 용 mock: 한국어 텍스트를 "[TRANSLATED] {text}" 형태로 변환."""
    t = BedrockTranslator()

    def mock_translate(text: str, src: str, tgt: str) -> str:  # noqa: ARG001
        return f"[TRANSLATED] {text}"

    monkeypatch.setattr(t, "_translate_call", mock_translate)
    return t


# ---------------------------------------------------------------------------
# 1. 인터페이스 충족 확인
# ---------------------------------------------------------------------------


class TestTranslatorProtocol:
    def test_isinstance_translator(self) -> None:
        """BedrockTranslator 가 core.interfaces.Translator 를 충족하는지 확인."""
        t = BedrockTranslator()
        assert isinstance(t, interfaces.Translator)

    def test_ko2en_returns_str(self, translator_passthrough: BedrockTranslator) -> None:
        t = translator_passthrough
        result = t.ko2en("일반 문장")
        assert isinstance(result, str)

    def test_en2ko_returns_str(self, translator_passthrough: BedrockTranslator) -> None:
        t = translator_passthrough
        result = t.en2ko("A general sentence without terms.")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 2. 용어 보호 라운드트립 — 핵심 테스트
# ---------------------------------------------------------------------------


class TestTermProtectionRoundtrip:
    def test_anaerobic_threshold_restored(self, translator_passthrough: BedrockTranslator) -> None:
        """무산소성 역치 → placeholder → 복원 후 anaerobic threshold."""
        result = translator_passthrough.ko2en("무산소성 역치는 운동 강도의 지표다.")
        assert "anaerobic threshold" in result
        assert "무산소성 역치" not in result

    def test_vo2max_protected_in_korean_text(
        self, translator_passthrough: BedrockTranslator
    ) -> None:
        """VO2max 는 한국어 질의 안에 있어도 보호된 채 영어로 복원된다 (card 예시)."""
        result = translator_passthrough.ko2en("무산소성 역치는 VO2max와 관련이 있다.")
        assert "anaerobic threshold" in result
        assert "VO2max" in result

    def test_peak_vo2_restored(self, translator_passthrough: BedrockTranslator) -> None:
        """최고산소섭취량 → peak VO2 복원 확인."""
        result = translator_passthrough.ko2en("심부전 환자의 최고산소섭취량을 측정한다.")
        assert "peak VO2" in result

    def test_cardiac_output_restored(self, translator_passthrough: BedrockTranslator) -> None:
        """EN 답변의 cardiac output → en2ko 시 심박출량 복원."""
        result = translator_passthrough.en2ko("Cardiac output increases during exercise.")
        assert "심박출량" in result

    def test_lactate_threshold_restored(self, translator_passthrough: BedrockTranslator) -> None:
        """lactate threshold → en2ko 시 젖산 역치 복원."""
        result = translator_passthrough.en2ko(
            "The lactate threshold is used to set training zones."
        )
        assert "젖산 역치" in result

    def test_multiple_terms_all_restored(self, translator_passthrough: BedrockTranslator) -> None:
        """여러 보호 용어가 모두 복원되는지 확인."""
        result = translator_passthrough.ko2en(
            "최대산소섭취량, 무산소성 역치, 심박출량은 CPET 핵심 지표다."
        )
        assert "VO2max" in result
        assert "anaerobic threshold" in result
        assert "cardiac output" in result
        assert "CPET" in result

    def test_cpet_acronym_preserved(self, translator_passthrough: BedrockTranslator) -> None:
        """CPET 약어는 ko2en 에서 변형 없이 보존된다."""
        result = translator_passthrough.ko2en("CPET 에서 운동부하검사를 실시한다.")
        assert "CPET" in result


# ---------------------------------------------------------------------------
# 3. Placeholder 누락 없음 — 전부 복원
# ---------------------------------------------------------------------------


class TestPlaceholderCleanup:
    def test_no_placeholder_leak_ko2en(self, translator_passthrough: BedrockTranslator) -> None:
        """번역 후 ⟦Tn⟧ 형태의 placeholder 가 남지 않아야 한다."""
        import re

        result = translator_passthrough.ko2en("최대산소섭취량과 무산소성 역치 측정이 중요하다.")
        assert not re.search(r"⟦T\d+⟧", result), f"placeholder 누락: {result!r}"

    def test_no_placeholder_leak_en2ko(self, translator_passthrough: BedrockTranslator) -> None:
        result = translator_passthrough.en2ko(
            "VO2max and anaerobic threshold are key CPET metrics."
        )
        import re

        assert not re.search(r"⟦T\d+⟧", result), f"placeholder 누락: {result!r}"

    def test_no_placeholder_when_mock_wraps(self, translator_ko2en_mock: BedrockTranslator) -> None:
        """mock 이 텍스트를 변형해도 placeholder 는 복원되어야 한다."""
        import re

        result = translator_ko2en_mock.ko2en("무산소성 역치는 VO2max와 관련이 있다.")
        assert not re.search(r"⟦T\d+⟧", result), f"placeholder 누락: {result!r}"
        # 용어도 복원되어야 함
        assert "anaerobic threshold" in result
        assert "VO2max" in result


# ---------------------------------------------------------------------------
# 4. 사전에 없는 일반 문장 통과
# ---------------------------------------------------------------------------


class TestNoTermPassthrough:
    def test_generic_ko_sentence(self, translator_passthrough: BedrockTranslator) -> None:
        """사전에 없는 한국어 문장 — 번역 호출은 되지만 오류 없이 통과."""
        text = "오늘 날씨가 맑고 기온이 높다."
        result = translator_passthrough.ko2en(text)
        assert isinstance(result, str)
        # passthrough mock 이라 원문이 그대로 나옴
        assert result == text

    def test_generic_en_sentence(self, translator_passthrough: BedrockTranslator) -> None:
        """사전에 없는 영어 문장 — 오류 없이 통과."""
        text = "The weather is sunny today."
        result = translator_passthrough.en2ko(text)
        assert isinstance(result, str)
        assert result == text


# ---------------------------------------------------------------------------
# 5. passthrough 모드
# ---------------------------------------------------------------------------


class TestPassthroughMode:
    def test_ko2en_passthrough(self) -> None:
        """passthrough=True 면 번역 없이 원문 반환."""
        t = BedrockTranslator(passthrough=True)
        text = "무산소성 역치"
        assert t.ko2en(text) == text

    def test_en2ko_passthrough(self) -> None:
        t = BedrockTranslator(passthrough=True)
        text = "anaerobic threshold"
        assert t.en2ko(text) == text


# ---------------------------------------------------------------------------
# 6. 내부 함수 단위 테스트
# ---------------------------------------------------------------------------


class TestInternalFunctions:
    def test_protect_longest_first(self) -> None:
        """최대산소섭취량(7자) 이 산소섭취량(5자) 보다 먼저 치환되어야 한다."""
        from serving.retrieval.translate import _KO_KEYS_SORTED

        keys_that_matter = [k for k in _KO_KEYS_SORTED if "산소섭취량" in k]
        # 최대산소섭취량 이 산소섭취량 보다 먼저 (인덱스가 더 낮음)
        idx_maxvo2 = keys_that_matter.index("최대산소섭취량")
        idx_o2 = keys_that_matter.index("산소섭취량")
        assert idx_maxvo2 < idx_o2, "최대산소섭취량 이 산소섭취량 보다 먼저 정렬되어야 함"

    def test_protect_and_restore_symmetric(self) -> None:
        """_protect + _restore 가 대칭 복원을 보장한다."""
        from serving.retrieval.translate import _KO_KEYS_SORTED

        text = "최대산소섭취량, 무산소성 역치"
        protected, mapping = _protect(text, _KO_KEYS_SORTED, KO2EN_GLOSSARY)
        # placeholder 가 삽입됨
        assert "⟦T" in protected
        assert "최대산소섭취량" not in protected
        # 복원하면 EN 용어로 대체
        restored = _restore(protected, mapping)
        assert "VO2max" in restored
        assert "anaerobic threshold" in restored

    def test_glossary_coverage(self) -> None:
        """사전에 20개 이상의 KO 항목이 있는지 확인."""
        assert len(KO2EN_GLOSSARY) >= 20, f"사전이 너무 작음: {len(KO2EN_GLOSSARY)}개"
