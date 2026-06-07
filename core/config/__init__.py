"""core.config — 설정 패키지. 모듈 레벨 settings 인스턴스를 노출한다."""

from core.config.settings import Settings

settings = Settings()

__all__ = ["Settings", "settings"]
