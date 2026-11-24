"""`cayu.server.app_errors`: an application's routes answer its exceptions as API errors."""

from __future__ import annotations

import pytest
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from cayu._resource_access_errors import ResourceAccessDenied
from cayu.artifacts import ArtifactStoreUnavailableError
from cayu.server.app_errors import (
    CAYU_CONFLICTS,
    DEFAULT_APP_ERRORS,
    AppErrorRoute,
    AppErrors,
    Conflict,
    ErrorStatus,
    Forbidden,
    Invalid,
    NotFound,
)
from cayu.storage import KnowledgeRevisionConflict
from cayu.tasks import TaskCreationConflict


class InvoiceLocked(Exception):
    pass


def _raise(error: Exception):
    raise error


def _client(errors: AppErrors | None = None) -> TestClient:
    route_class = AppErrorRoute if errors is None else errors.route_class()
    router = APIRouter(prefix="/app", route_class=route_class)

    @router.get("/fail/{kind}")
    async def fail(kind: str) -> dict[str, str]:
        _raise(
            {
                "not-found": NotFound("Invoice not found."),
                "forbidden": Forbidden("Only finance may approve."),
                "conflict": Conflict("Invoice is being paid."),
                "invalid": Invalid("Amount must be positive."),
                "lookup": LookupError("row 7 of /srv/books/march.db"),
                "key": KeyError("invoice-7"),
                "file": FileNotFoundError(2, "No such file", "/srv/books/march.xlsx"),
                "permission": PermissionError("/srv/books is read-only"),
                "denied": ResourceAccessDenied(),
                "revision": KnowledgeRevisionConflict(
                    "entry", expected_revision=1, actual_revision=2
                ),
                "task": TaskCreationConflict("Task id reused with other content."),
                "value": ValueError("input_value='sk-secret'"),
                "storage": ArtifactStoreUnavailableError("bucket timed out"),
                "http": HTTPException(418, "teapot"),
                "starlette": StarletteHTTPException(401, "Sign in."),
                "locked": InvoiceLocked("Invoice is being paid."),
                "other": RuntimeError("boom"),
            }[kind]
        )
        return {}

    def person() -> str:
        raise Forbidden("Sign in first.")

    @router.get("/me")
    async def me(who: str = Depends(person)) -> dict[str, str]:
        return {"who": who}

    @router.get("/typed/{count}")
    async def typed(count: int) -> dict[str, int]:
        return {"count": count}

    server = FastAPI()
    server.include_router(router)
    return TestClient(server, raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("kind", "status", "detail"),
    [
        ("not-found", 404, "Invoice not found."),
        ("forbidden", 403, "Only finance may approve."),
        ("conflict", 409, "Invoice is being paid."),
        ("invalid", 422, "Amount must be positive."),
        # Built-in exceptions from anywhere answer with fixed texts, not their messages.
        ("lookup", 404, "Not found."),
        ("key", 404, "Not found."),
        ("file", 404, "Not found."),
        ("permission", 403, "Not allowed."),
        ("denied", 404, "Resource unavailable."),
        ("task", 409, "The request conflicts with the stored state."),
        ("value", 422, "Invalid request."),
        ("storage", 503, "File storage is unavailable."),
        ("http", 418, "teapot"),
        ("starlette", 401, "Sign in."),
    ],
)
def test_default_errors_answer_with_status_and_detail(kind: str, status: int, detail: str):
    response = _client().get(f"/app/fail/{kind}")

    assert response.status_code == status
    assert response.json() == {"detail": detail}


def test_cayu_conflicts_answer_409_before_value_error():
    assert issubclass(TaskCreationConflict, ValueError)
    assert KnowledgeRevisionConflict in CAYU_CONFLICTS

    response = _client().get("/app/fail/revision")

    assert response.status_code == 409


def test_unmapped_exception_stays_a_server_error():
    assert _client().get("/app/fail/other").status_code == 500
    assert _client().get("/app/fail/locked").status_code == 500


def test_dependency_errors_are_mapped_too():
    response = _client().get("/app/me")

    assert response.status_code == 403
    assert response.json() == {"detail": "Sign in first."}


def test_request_validation_is_unchanged():
    response = _client().get("/app/typed/many")

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["path", "count"]


def test_extended_rules_are_checked_first():
    errors = DEFAULT_APP_ERRORS.extend(
        ErrorStatus((InvoiceLocked,), 409),
        ErrorStatus((ValueError,), 400, detail=lambda error: {"message": str(error)}),
    )
    client = _client(errors)

    assert client.get("/app/fail/locked").json() == {"detail": "Invoice is being paid."}
    value = client.get("/app/fail/invalid")
    assert (value.status_code, value.json()) == (
        400,
        {"detail": {"message": "Amount must be positive."}},
    )
    # The defaults still answer what the new rules don't name.
    assert client.get("/app/fail/permission").status_code == 403


def test_http_error_on_its_own():
    assert DEFAULT_APP_ERRORS.http_error(RuntimeError("boom")) is None
    mapped = DEFAULT_APP_ERRORS.http_error(NotFound("gone"))
    assert mapped is not None and (mapped.status_code, mapped.detail) == (404, "gone")
    starlette = StarletteHTTPException(401)
    assert DEFAULT_APP_ERRORS.http_error(starlette) is starlette


def test_extend_catch_all_keeps_http_exceptions():
    errors = DEFAULT_APP_ERRORS.extend(ErrorStatus((Exception,), 500, detail="Internal error"))
    response = _client(errors).get("/app/fail/starlette")
    assert (response.status_code, response.json()) == (401, {"detail": "Sign in."})


def test_error_status_validates_its_rule():
    with pytest.raises(TypeError):
        ErrorStatus((), 404)
    with pytest.raises(TypeError):
        ErrorStatus((BaseException,), 404)
    with pytest.raises(ValueError):
        ErrorStatus((LookupError,), 200)
    with pytest.raises(TypeError):
        AppErrors((LookupError, 404))  # type: ignore[arg-type]
