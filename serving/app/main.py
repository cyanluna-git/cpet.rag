"""serving.app.main — FastAPI 앱 엔트리포인트.

uvicorn serving.app.main:app --host 0.0.0.0 --port 8110

create_app() 은 import-time 에 파이프라인을 로드하지 않는다 (lazy).
테스트/임포트가 가볍게 유지된다.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from serving.app.router import router


def create_app() -> FastAPI:
    """FastAPI 앱을 생성해 반환한다.

    - router 등록
    - CORS 미들웨어 (로컬 dev 용, 필요 시 settings 로 제한)
    - startup 에서 무거운 파이프라인 로드 안 함 (lazy)
    """
    _app = FastAPI(
        title="cpet.rag API",
        description="운동생리·CPET 학술 문헌 RAG 서빙 API",
        version="0.1.0",
    )

    # CORS — 로컬 개발 + 프론트엔드 3110 허용
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3110", "http://127.0.0.1:3110"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _app.include_router(router)

    return _app


app = create_app()
