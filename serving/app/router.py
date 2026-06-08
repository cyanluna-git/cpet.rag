"""serving.app.router — FastAPI APIRouter: 라우팅 레이어.

비즈니스 로직 없음. 요청 수신 → service.ask() 위임 → 응답 반환만 담당.
에러는 통일 포맷 {error:{code, message}} 으로 반환한다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from core.models import QueryRequest, QueryResponse
from serving.app.service import QueryService

router = APIRouter()

# ---------------------------------------------------------------------------
# Dependency — 테스트에서 app.dependency_overrides[get_service] 로 주입 가능
# ---------------------------------------------------------------------------

_default_service: QueryService | None = None


def get_service() -> QueryService:
    """FastAPI Depends 로 QueryService 를 제공한다.

    싱글턴 패턴: 최초 호출 시 QueryService() 를 생성해 재사용한다.
    테스트에서는 app.dependency_overrides[get_service] 로 mock 주입.
    """
    global _default_service
    if _default_service is None:
        _default_service = QueryService()  # pipeline=None → lazy 초기화
    return _default_service


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/health")
def health() -> dict[str, str]:
    """헬스 체크 엔드포인트."""
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    service: QueryService = Depends(get_service),
) -> QueryResponse | JSONResponse:
    """QueryRequest → QueryResponse 를 반환한다.

    service.ask() 에서 예외 발생 시 통일 에러 포맷으로 500 응답한다.
    """
    try:
        return service.ask(req)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "INTERNAL_ERROR", "message": str(exc)}},
        )
