"""serving.retrieval.translate — Query Translation 샌드위치 구현.

번역 흐름:
  KO 질의 → (1) 용어 보호(placeholder 치환) → (2) KO→EN 번역 API
           → (3) placeholder 복원(EN 용어) → EN 질의

  EN 답변 → (1) 용어 보호(EN 측 placeholder 치환) → (2) EN→KO 번역 API
           → (3) placeholder 복원(KO 용어) → KO 답변

boto3 는 `aws` optional group 에만 포함. lazy import 로 base 의존성 깨지지 않게.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from core.config.settings import Settings
from core.log import get_logger
from serving.retrieval.glossary import EN2KO_GLOSSARY, KO2EN_GLOSSARY

if TYPE_CHECKING:
    pass  # 타입 힌트 전용 임포트

logger = get_logger(__name__)

# placeholder 포맷: ⟦T{n}⟧  — ASCII 기반 대괄호가 아닌 유니코드 레버를 사용해
# 번역 모델이 글자를 분해하기 어렵게 한다.
_PLACEHOLDER_PATTERN = re.compile(r"⟦T\d+⟧")


def _build_sorted_keys(glossary: dict[str, str]) -> list[str]:
    """내림차순 길이 정렬 — 서브스트링 오치환 방지."""
    return sorted(glossary.keys(), key=len, reverse=True)


_KO_KEYS_SORTED: list[str] = _build_sorted_keys(KO2EN_GLOSSARY)
_EN_KEYS_SORTED: list[str] = _build_sorted_keys(EN2KO_GLOSSARY)


def _is_ascii_term(term: str) -> bool:
    """알파벳(ASCII) 으로만 구성된 용어인지 확인. 영어 단어 경계 처리 대상 판단."""
    return all(c.isascii() for c in term)


def _protect(
    text: str, glossary_keys: list[str], glossary: dict[str, str]
) -> tuple[str, dict[str, str]]:
    """텍스트에서 전문 용어를 ⟦Tn⟧ placeholder 로 치환한다.

    - ASCII 용어: `\\b` 단어 경계 + case-insensitive 매칭 (문장 첫 글자 대문자, "Cardiac" 등 대응)
    - 한국어 용어: 정확 문자열 매칭 (단어 경계 불필요)

    Args:
        text: 원본 텍스트.
        glossary_keys: 내림차순 정렬된 사전 키 목록.
        glossary: 원문→대응어 매핑.

    Returns:
        (보호된 텍스트, {placeholder → 대응어} 매핑)
    """
    protected = text
    placeholder_map: dict[str, str] = {}
    counter = 0
    for term in glossary_keys:
        escaped = re.escape(term)
        if _is_ascii_term(term):
            # 단어 경계 적용: "AT" 이 "weather" 안에서 매칭되지 않도록
            pattern = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
        else:
            # 한국어: 단어 경계 없이 정확 매칭
            pattern = re.compile(escaped)
        if pattern.search(protected):
            placeholder = f"⟦T{counter}⟧"
            protected = pattern.sub(placeholder, protected)
            placeholder_map[placeholder] = glossary[term]
            counter += 1
    return protected, placeholder_map


def _restore(text: str, placeholder_map: dict[str, str]) -> str:
    """번역된 텍스트의 ⟦Tn⟧ placeholder 를 대응어로 복원한다."""
    restored = text
    for placeholder, target_term in placeholder_map.items():
        restored = restored.replace(placeholder, target_term)
    return restored


def _check_no_placeholders(text: str) -> None:
    """남은 placeholder 가 있으면 경고를 로깅한다 (복원 누락 감지)."""
    remaining = _PLACEHOLDER_PATTERN.findall(text)
    if remaining:
        logger.warning("번역 후 복원되지 않은 placeholder 감지: %s", remaining)


class BedrockTranslator:
    """한-영 번역기 (용어 보호 사전 + Bedrock Claude 백엔드).

    core.interfaces.Translator Protocol 을 충족한다.

    Args:
        model: Bedrock 모델 ID. None 이면 settings.bedrock_model_id 사용.
        region: AWS 리전. None 이면 settings.aws_region 사용.
        passthrough: True 면 번역 없이 원문 그대로 반환 (영어 질의 입력 대비).
    """

    def __init__(
        self,
        model: str | None = None,
        region: str | None = None,
        passthrough: bool = False,
    ) -> None:
        _settings = Settings()
        self._model: str | None = model or _settings.bedrock_model_id
        self._region: str = region or _settings.aws_region
        self._passthrough = passthrough
        # boto3 클라이언트는 실제 호출 시 lazy 초기화 (_translate_call 내부)

    # ------------------------------------------------------------------
    # Public API (Translator Protocol)
    # ------------------------------------------------------------------

    def ko2en(self, text: str) -> str:
        """한국어 질의를 영어로 번역한다 (용어 보호 포함).

        Args:
            text: 한국어 텍스트 (질의).

        Returns:
            영어 번역 텍스트.
        """
        if self._passthrough:
            return text

        protected, placeholder_map = _protect(text, _KO_KEYS_SORTED, KO2EN_GLOSSARY)
        logger.debug("ko2en protect: %d 용어 보호됨", len(placeholder_map))

        translated = self._translate_call(protected, src="ko", tgt="en")

        result = _restore(translated, placeholder_map)
        _check_no_placeholders(result)
        logger.debug("ko2en 완료: %r → %r", text[:60], result[:60])
        return result

    def en2ko(self, text: str) -> str:
        """영어 답변을 한국어로 역번역한다 (용어 보호 포함).

        Args:
            text: 영어 텍스트 (LLM 생성 답변).

        Returns:
            한국어 번역 텍스트.
        """
        if self._passthrough:
            return text

        protected, placeholder_map = _protect(text, _EN_KEYS_SORTED, EN2KO_GLOSSARY)
        logger.debug("en2ko protect: %d 용어 보호됨", len(placeholder_map))

        translated = self._translate_call(protected, src="en", tgt="ko")

        result = _restore(translated, placeholder_map)
        _check_no_placeholders(result)
        logger.debug("en2ko 완료: %r → %r", text[:60], result[:60])
        return result

    # ------------------------------------------------------------------
    # Internal seam — 테스트에서 mock 가능
    # ------------------------------------------------------------------

    def _translate_call(self, text: str, src: str, tgt: str) -> str:
        """Bedrock Claude 를 통해 실제 번역을 수행한다.

        이 메서드는 테스트에서 monkeypatch/mock 대상.
        boto3 미설치 또는 모델 ID 미설정 시 RuntimeError 를 발생시킨다.

        Args:
            text: 보호 처리된 번역 대상 텍스트 (placeholder 포함 가능).
            src: 소스 언어 코드 ('ko' | 'en').
            tgt: 타겟 언어 코드 ('en' | 'ko').

        Returns:
            번역된 텍스트 (placeholder 는 그대로 유지되어야 함).

        Raises:
            RuntimeError: boto3 미설치 또는 bedrock_model_id 미설정.
        """
        try:
            import boto3  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "boto3 가 설치되어 있지 않습니다. `uv pip install cpet-rag[aws]` 를 실행하세요."
            ) from exc

        if not self._model:
            raise RuntimeError(
                "bedrock_model_id 가 설정되지 않았습니다. .env 에 BEDROCK_MODEL_ID 를 추가하세요."
            )

        lang_labels = {"ko": "Korean", "en": "English"}
        src_label = lang_labels.get(src, src)
        tgt_label = lang_labels.get(tgt, tgt)

        system_prompt = (
            "You are a precise academic translator specializing in exercise physiology, "
            "cardiopulmonary exercise testing (CPET), and sports medicine.\n"
            "Rules:\n"
            "1. Translate the user text from {src} to {tgt} accurately.\n"
            "2. Preserve ALL ⟦Tn⟧ and ⟦Cn⟧ tokens EXACTLY — do not translate, rephrase, "
            "or split them. They are protected placeholders (⟦Tn⟧ = terminology, "
            "⟦Cn⟧ = citation tag). Output them verbatim without any modification.\n"
            "3. Use formal academic register appropriate for medical/scientific literature.\n"
            "4. Output ONLY the translated text, no commentary."
        ).format(src=src_label, tgt=tgt_label)

        client = boto3.client("bedrock-runtime", region_name=self._region)

        body: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": text,
                }
            ],
        }

        import json  # noqa: PLC0415

        response = client.invoke_model(
            modelId=self._model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        response_body = json.loads(response["body"].read())
        # Anthropic Messages API 응답: content[0].text
        translated: str = response_body["content"][0]["text"].strip()
        logger.info("Bedrock 번역 완료 (%s→%s): %d 자", src, tgt, len(translated))
        return translated
