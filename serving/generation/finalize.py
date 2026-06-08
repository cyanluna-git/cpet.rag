"""serving.generation.finalize — EN→KO 역번역 + 인용 태그 보존.

## 설계

`back_translate_answer`:
    1. 답변의 모든 [id] 인용 태그를 ⟦C{n}⟧ placeholder 로 치환·기억
    2. translator.en2ko(text) 로 영한 번역 수행
    3. placeholder 를 원래 [id] 태그로 복원

인용 태그 placeholder 는 ⟦C{n}⟧ 을 사용한다.
- BedrockTranslator 내부 용어 보호 placeholder ⟦T{n}⟧ 과 프리픽스가 달라 충돌 없음.
- _check_no_placeholders 의 ⟦T\\d+⟧ 패턴이 ⟦C…⟧ 에 반응하지 않는다.

`finalize_answer`:
    GenerationResult 를 받아 refused 이면 KO 거부 메시지를,
    아니면 back_translate_answer 결과를 반환한다.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from core.log import get_logger

if TYPE_CHECKING:
    from core.interfaces import Translator
    from serving.generation.generate import GenerationResult

logger = get_logger(__name__)

# [id] 인용 태그 패턴 — verify.py / generate.py 와 동일한 패턴 재사용.
# id 는 `.:/_::` 등 특수문자 포함 가능.
_CITATION_TAG_PATTERN = re.compile(r"\[([^\[\]]+)\]")

# 인용 보호용 placeholder 포맷 — C 프리픽스로 ⟦T{n}⟧ 용어 placeholder 와 구분.
_CITE_PLACEHOLDER_FMT = "⟦C{n}⟧"

# 번역 거부 시 반환할 한국어 메시지 (refused=True 경우)
_KO_REFUSAL = "제공된 자료만으로는 답변할 수 없습니다."


def back_translate_answer(
    answer_en: str,
    translator: "Translator",
    *,
    protect_citations: bool = True,
) -> str:
    """영어 답변을 한국어로 역번역한다 (인용 태그 보존).

    Parameters
    ----------
    answer_en:
        LLM 이 생성한 영문 답변 텍스트. [id] 형식의 인용 태그를 포함할 수 있다.
    translator:
        core.interfaces.Translator Protocol 을 충족하는 번역기.
        en2ko() 가 이미 도메인 용어 glossary 보호를 처리한다.
    protect_citations:
        True(기본값) 면 [id] 태그를 ⟦C{n}⟧ placeholder 로 보호한 뒤 번역,
        번역 후 원래 태그로 복원한다.
        False 면 보호 없이 translator.en2ko(answer_en) 를 그대로 반환한다.

    Returns
    -------
    str
        한국어 번역 텍스트. 인용 태그 [id] 는 번역 전과 동일하게 보존된다.
    """
    if not protect_citations:
        return translator.en2ko(answer_en)

    # 1. 인용 태그를 placeholder 로 치환하면서 복원 맵 구성
    counter = 0
    placeholder_map: dict[str, str] = {}  # placeholder → 원래 [id] 전체 문자열

    def _replace_tag(m: re.Match[str]) -> str:
        nonlocal counter
        placeholder = _CITE_PLACEHOLDER_FMT.format(n=counter)
        placeholder_map[placeholder] = m.group(0)  # "[id]" 전체 보존
        counter += 1
        return placeholder

    protected = _CITATION_TAG_PATTERN.sub(_replace_tag, answer_en)

    logger.debug(
        "back_translate_answer: %d 인용 태그 보호됨",
        len(placeholder_map),
    )

    # 2. EN→KO 번역 (translator 내부에서 도메인 용어 보호도 수행)
    translated = translator.en2ko(protected)

    # 3. placeholder 를 원래 [id] 태그로 복원
    restored = translated
    for placeholder, original_tag in placeholder_map.items():
        restored = restored.replace(placeholder, original_tag)

    # 미복원 placeholder 감지 (복원 누락 경고)
    remaining = re.findall(r"⟦C\d+⟧", restored)
    if remaining:
        logger.warning(
            "back_translate_answer: 복원되지 않은 인용 placeholder 감지: %s",
            remaining,
        )

    logger.debug(
        "back_translate_answer 완료: answer_len=%d → translated_len=%d",
        len(answer_en),
        len(restored),
    )
    return restored


def finalize_answer(
    gen_result: "GenerationResult",
    translator: "Translator",
) -> str:
    """GenerationResult 를 최종 한국어 답변 문자열로 변환한다.

    Parameters
    ----------
    gen_result:
        Generator.generate() 의 반환값.
        verify_citations → strip_unverified 처리가 완료된 answer_en 을 담고 있어야 한다.
    translator:
        core.interfaces.Translator Protocol 을 충족하는 번역기.

    Returns
    -------
    str
        한국어 최종 답변.
        refused=True 면 한국어 거부 메시지를 반환한다 (translator 호출 없음).
    """
    if gen_result.refused:
        logger.info("finalize_answer: refused → 한국어 거부 메시지 반환")
        return _KO_REFUSAL

    return back_translate_answer(gen_result.answer_en, translator)
