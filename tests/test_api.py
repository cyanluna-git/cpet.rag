"""tests/test_api.py — FastAPI 서빙 레이어 테스트 (TestClient + mock service).

실제 LanceDB / LLM 로드 없이 동작한다. get_service dependency 를 override 해서
mock QueryService 를 주입한다.

테스트 시나리오:
1. POST /query → 200, mock 반환값과 일치
2. GET /health → 200 {status: ok}
3. 잘못된 body(query 누락) → 422
4. service 가 예외 던질 때 → 500 + {error:{code,message}}
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from core.models import QueryRequest, QueryResponse
from serving.app.main import create_app
from serving.app.router import get_service
from serving.app.service import QueryService

# ---------------------------------------------------------------------------
# Mock QueryService
# ---------------------------------------------------------------------------

_CANNED_RESPONSE = QueryResponse(
    answer="운동 중 근육 대사는 에너지 생산을 증가시킵니다.",
    answer_en="Muscle metabolism during exercise increases energy production.",
    citations=[],
    retrieved=[],
)


class _MockQueryService(QueryService):
    """ask() 가 canned QueryResponse 를 반환하는 테스트용 서비스."""

    def __init__(self) -> None:
        # 부모 __init__ 에 pipeline=None 이지만 ask 를 override 하므로
        # 실제로 파이프라인이 초기화되지 않는다.
        super().__init__(pipeline=None)

    def ask(self, req: QueryRequest) -> QueryResponse:  # noqa: ARG002
        return _CANNED_RESPONSE


class _ErrorQueryService(QueryService):
    """ask() 가 항상 RuntimeError 를 던지는 테스트용 서비스."""

    def __init__(self) -> None:
        super().__init__(pipeline=None)

    def ask(self, req: QueryRequest) -> QueryResponse:  # noqa: ARG002
        raise RuntimeError("pipeline failure")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """정상 mock service 로 TestClient 를 생성한다."""
    _app = create_app()
    _app.dependency_overrides[get_service] = lambda: _MockQueryService()
    return TestClient(_app)


@pytest.fixture()
def error_client() -> TestClient:
    """예외 던지는 mock service 로 TestClient 를 생성한다.

    raise_server_exceptions=False: 500 응답을 예외 없이 받기 위함.
    """
    _app = create_app()
    _app.dependency_overrides[get_service] = lambda: _ErrorQueryService()
    return TestClient(_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 1. POST /query — 정상 응답
# ---------------------------------------------------------------------------


def test_query_success(client: TestClient) -> None:
    """POST /query 로 QueryRequest 전송 시 200 + QueryResponse 반환."""
    payload = {"query": "운동 중 근육 대사", "top_k": 3}
    resp = client.post("/query", json=payload)

    assert resp.status_code == 200
    body = resp.json()

    # 필수 필드
    assert "answer" in body
    assert "citations" in body
    assert "retrieved" in body

    # mock 반환값과 일치
    assert body["answer"] == _CANNED_RESPONSE.answer
    assert body["answer_en"] == _CANNED_RESPONSE.answer_en
    assert body["citations"] == []
    assert body["retrieved"] == []


# ---------------------------------------------------------------------------
# 2. GET /health
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    """GET /health → 200 {status: ok}."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 3. 잘못된 body (query 누락) → 422
# ---------------------------------------------------------------------------


def test_query_missing_body(client: TestClient) -> None:
    """query 필드 없는 body → 422 Unprocessable Entity."""
    resp = client.post("/query", json={"top_k": 3})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 4. service 예외 → 500 + {error:{code, message}}
# ---------------------------------------------------------------------------


def test_query_service_error(error_client: TestClient) -> None:
    """service.ask() 가 예외 던질 때 500 + 통일 에러 포맷 반환."""
    payload = {"query": "운동 중 근육 대사", "top_k": 3}
    resp = error_client.post("/query", json=payload)

    assert resp.status_code == 500
    body = resp.json()
    assert "error" in body
    assert "code" in body["error"]
    assert "message" in body["error"]
    # 에러 메시지가 예외 내용 포함
    assert "pipeline failure" in body["error"]["message"]
