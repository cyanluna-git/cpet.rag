"""core.log — 공통 로거 팩토리.

``log.py``로 명명 (logging.py 사용 시 표준 라이브러리 shadowing 발생).
LOG_LEVEL 환경변수(또는 settings.log_level)로 레벨을 제어한다.
"""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """name 기반 Logger를 반환한다. 핸들러 중복 추가를 방지한다."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        level_name: str = os.environ.get("LOG_LEVEL", "INFO").upper()
        level: int = getattr(logging, level_name, logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger
