from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient
from pydantic import ValidationError

from cayu import CayuApp, PostgresSessionStore, SQLiteSessionStore
from cayu.runtime import InMemorySessionStore, SessionStore
from cayu.server import ServerConfig, create_server
from cayu.server.routes import (
    InterruptSessionBody,
    ResumeBody,
    RunBody,
    ToolApprovalBody,
    ToolApprovalRecoveryBody,
    ToolRoundRecoveryBody,
    UserInputRecoveryBody,
    UserInputResolveBody,
)
from cayu.storage.migrations import SchemaMode

CONTROL_PLANE_PROMPT_MAX_BYTES = 64 * 1024
CONTROL_PLANE_METADATA_MAX_BYTES = 64 * 1024
CONTROL_PLANE_METADATA_MAX_MEMBERS = 1024
CONTROL_PLANE_REQUEST_MAX_BYTES = 1024 * 1024


@pytest.mark.parametrize("body_type", [RunBody, ResumeBody])
def test_control_plane_prompt_enforces_utf8_boundary(body_type) -> None:
    base = {"session_id": "session-1"} if body_type is ResumeBody else {}

    accepted = body_type(**base, prompt="x" * CONTROL_PLANE_PROMPT_MAX_BYTES)
    assert len(accepted.prompt.encode("utf-8")) == CONTROL_PLANE_PROMPT_MAX_BYTES

    with pytest.raises(ValidationError):
        body_type(**base, prompt="x" * (CONTROL_PLANE_PROMPT_MAX_BYTES + 1))

    multibyte = "\u00e9" * (CONTROL_PLANE_PROMPT_MAX_BYTES // 2)
    accepted = body_type(**base, prompt=multibyte)
    assert len(accepted.prompt.encode("utf-8")) == CONTROL_PLANE_PROMPT_MAX_BYTES

    with pytest.raises(ValidationError):
        body_type(**base, prompt=multibyte + "x")


METADATA_BODY_CASES = [
    (InterruptSessionBody, {}),
    (
        ToolApprovalBody,
        {
            "session_id": "session-1",
            "approval_id": "approval-1",
            "tool_round_id": "round-1",
            "tool_call_id": "call-1",
            "decision": "approve",
        },
    ),
    (
        ToolApprovalRecoveryBody,
        {
            "session_id": "session-1",
            "approval_id": "approval-1",
            "tool_round_id": "round-1",
            "tool_call_id": "call-1",
            "outcome": "completed",
            "message": "verified",
        },
    ),
    (
        ToolRoundRecoveryBody,
        {
            "session_id": "session-1",
            "round_id": "round-1",
            "tool_call_id": "call-1",
            "outcome": "completed",
            "message": "verified",
        },
    ),
    (
        UserInputResolveBody,
        {
            "session_id": "session-1",
            "input_id": "input-1",
            "answer": "approved",
        },
    ),
    (
        UserInputRecoveryBody,
        {
            "session_id": "session-1",
            "input_id": "input-1",
            "answer": "approved",
            "tool_call_id": "call-1",
            "outcome": "completed",
            "message": "verified",
        },
    ),
]


@pytest.mark.parametrize(("body_type", "base"), METADATA_BODY_CASES)
def test_control_plane_metadata_enforces_durable_json_and_encoded_boundary(
    body_type,
    base,
) -> None:
    exact_ascii = {"v": "x" * (CONTROL_PLANE_METADATA_MAX_BYTES - len(b'{"v":""}'))}
    accepted = body_type(**base, metadata=exact_ascii)
    assert accepted.metadata == exact_ascii

    with pytest.raises(ValidationError):
        body_type(**base, metadata={"v": exact_ascii["v"] + "x"})

    exact_multibyte = {"v": "\u00e9" * ((CONTROL_PLANE_METADATA_MAX_BYTES - len(b'{"v":""}')) // 2)}
    accepted = body_type(**base, metadata=exact_multibyte)
    assert accepted.metadata == exact_multibyte

    with pytest.raises(ValidationError):
        body_type(**base, metadata={"v": exact_multibyte["v"] + "x"})

    invalid_values = [
        {"v": "before\x00after"},
        {"v": "\ud800"},
        {"v": float("nan")},
        {"v": float("inf")},
        {"v": 2**63},
        {1: "non-string key"},
    ]
    for invalid in invalid_values:
        with pytest.raises(ValidationError):
            body_type(**base, metadata=invalid)


@pytest.mark.parametrize(("body_type", "base"), METADATA_BODY_CASES)
def test_control_plane_metadata_bounds_nesting_and_cardinality(body_type, base) -> None:
    nested: dict[str, object] = {"leaf": True}
    for _ in range(31):
        nested = {"child": nested}
    assert body_type(**base, metadata=nested).metadata == nested

    nested = {"child": nested}
    with pytest.raises(ValidationError):
        body_type(**base, metadata=nested)

    accepted = {f"k{index:04d}": None for index in range(CONTROL_PLANE_METADATA_MAX_MEMBERS)}
    assert body_type(**base, metadata=accepted).metadata == accepted

    rejected = dict(accepted)
    rejected["overflow"] = None
    with pytest.raises(ValidationError):
        body_type(**base, metadata=rejected)


def test_control_plane_request_body_limit_rejects_declared_and_chunked_bodies_before_run() -> None:
    secret = "control-plane-request-secret-canary"
    store = InMemorySessionStore()
    client = TestClient(
        create_server(
            CayuApp(session_store=store, enable_logging=False),
            config=ServerConfig.local_development(),
        )
    )

    declared_body = json.dumps(
        {
            "session_id": "declared-body-session",
            "prompt": "hello",
            "padding": secret + ("x" * CONTROL_PLANE_REQUEST_MAX_BYTES),
        }
    )
    declared = client.post(
        "/api/run",
        content=declared_body,
        headers={"Content-Type": "application/json"},
    )

    assert declared.status_code == 413
    assert declared.json() == {"detail": "Control-plane request exceeds the server byte limit."}
    assert declared.headers["cache-control"] == "private, no-store"
    assert secret not in declared.text
    assert asyncio.run(store.load("declared-body-session")) is None

    def oversized_chunks():
        yield b'{"session_id":"chunked-body-session","prompt":"hello","padding":"'
        yield secret.encode()
        yield b"x" * CONTROL_PLANE_REQUEST_MAX_BYTES
        yield b'"}'

    chunked = client.post(
        "/api/run",
        content=oversized_chunks(),
        headers={"Content-Type": "application/json"},
    )

    assert chunked.status_code == 413
    assert chunked.json() == {"detail": "Control-plane request exceeds the server byte limit."}
    assert chunked.headers["cache-control"] == "private, no-store"
    assert secret not in chunked.text
    assert asyncio.run(store.load("chunked-body-session")) is None


def test_control_plane_validation_response_does_not_reflect_rejected_prompt() -> None:
    secret = "control-plane-prompt-secret-canary"
    store = InMemorySessionStore()
    client = TestClient(
        create_server(
            CayuApp(session_store=store, enable_logging=False),
            config=ServerConfig.local_development(),
        )
    )

    response = client.post(
        "/api/run",
        json={
            "session_id": "invalid-prompt-session",
            "prompt": secret + ("x" * CONTROL_PLANE_PROMPT_MAX_BYTES),
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid control-plane request."}
    assert response.headers["cache-control"] == "private, no-store"
    assert secret not in response.text
    assert asyncio.run(store.load("invalid-prompt-session")) is None


def test_control_plane_rejects_duplicate_json_keys_before_route_validation() -> None:
    store = InMemorySessionStore()
    client = TestClient(
        create_server(
            CayuApp(session_store=store, enable_logging=False),
            config=ServerConfig.local_development(),
        )
    )

    response = client.post(
        "/api/run",
        content=(
            b'{"session_id":"duplicate-key-session","prompt":"hello","metadata":{"key":1,"key":2}}'
        ),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid control-plane request."}
    assert response.headers["cache-control"] == "private, no-store"
    assert asyncio.run(store.load("duplicate-key-session")) is None


def _assert_rejected_control_plane_request_does_not_mutate_store(
    store: SessionStore,
    *,
    session_id: str,
) -> None:
    client = TestClient(
        create_server(
            CayuApp(session_store=store, enable_logging=False),
            config=ServerConfig.local_development(),
        )
    )
    response = client.post(
        "/api/run",
        json={
            "session_id": session_id,
            "prompt": "x" * (CONTROL_PLANE_PROMPT_MAX_BYTES + 1),
        },
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid control-plane request."}

    async def verify() -> None:
        try:
            assert await store.load(session_id) is None
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    asyncio.run(verify())


def test_rejected_control_plane_request_does_not_mutate_sqlite(tmp_path) -> None:
    _assert_rejected_control_plane_request_does_not_mutate_store(
        SQLiteSessionStore(tmp_path / "control-plane-admission.sqlite"),
        session_id="sqlite-rejected-control-plane",
    )


async def _drop_postgres_cayu_tables(dsn: str) -> None:
    import psycopg
    from psycopg import sql

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = current_schema() AND tablename LIKE 'cayu\\_%' ESCAPE '\\'"
            )
            for (table_name,) in await cur.fetchall():
                await cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table_name))
                )
        await conn.commit()


def test_rejected_control_plane_request_does_not_mutate_postgres(postgres_dsn) -> None:
    asyncio.run(_drop_postgres_cayu_tables(postgres_dsn))
    try:
        _assert_rejected_control_plane_request_does_not_mutate_store(
            PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE),
            session_id="postgres-rejected-control-plane",
        )
    finally:
        asyncio.run(_drop_postgres_cayu_tables(postgres_dsn))


def test_control_plane_request_ceiling_is_documented_in_openapi() -> None:
    client = TestClient(
        create_server(
            CayuApp(enable_logging=False),
            config=ServerConfig.local_development(),
        )
    )

    schema = client.get("/openapi.json").json()
    for path in (
        "/api/run",
        "/api/resume",
        "/api/sessions/{session_id}/interrupt",
        "/api/tool-approvals/resolve",
        "/api/tool-approvals/recover",
        "/api/tool-rounds/recover",
        "/api/user-input/resolve",
        "/api/user-input/recover",
    ):
        response_schema = schema["paths"][path]["post"]["responses"]["413"]["content"][
            "application/json"
        ]["schema"]
        assert response_schema == {"$ref": "#/components/schemas/ApiErrorResponse"}
