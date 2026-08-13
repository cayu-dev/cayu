from __future__ import annotations

import asyncio
import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from cayu import (
    InMemorySessionStore,
    InvocationOriginClaim,
    InvocationOriginTrust,
    RunRequest,
    SessionExecutionSource,
    SessionIdentity,
    SessionInvocation,
    SQLiteSessionStore,
)
from cayu.runtime.sessions import (
    fork_session_invocation,
    run_request_with_runtime_invocation,
)


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def test_root_sdk_invocation_is_unattributed_without_a_host_claim() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(agent_name="assistant", messages=[]),
            identity=_identity(),
        )

        assert session.invocation.root_session_id == session.id
        assert type(session.invocation.root_invocation_id) is str
        assert UUID(session.invocation.root_invocation_id).version == 4
        assert str(UUID(session.invocation.root_invocation_id)) == (
            session.invocation.root_invocation_id
        )
        assert json.loads(session.invocation.model_dump_json())["root_invocation_id"] == (
            session.invocation.root_invocation_id
        )
        assert session.invocation.source is SessionExecutionSource.SDK_RUN
        assert session.invocation.origin.trust is InvocationOriginTrust.UNATTRIBUTED
        assert session.invocation.origin.subject is None
        assert session.invocation.origin.tenant is None

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "invalid_id",
    [
        "not-a-uuid",
        "00000000-0000-1000-8000-000000000000",
        "00000000-0000-4000-0000-000000000000",
        "F055BEDC-62CF-4FA4-979A-D0378CA93131",
        "f055bedc62cf4fa4979ad0378ca93131",
        UUID("f055bedc-62cf-4fa4-979a-d0378ca93131"),
    ],
)
def test_root_invocation_identity_requires_a_canonical_uuid4_string(invalid_id: object) -> None:
    with pytest.raises(ValidationError):
        SessionInvocation.model_validate(
            {
                "origin": {"trust": "unattributed"},
                "root_invocation_id": invalid_id,
                "root_session_id": "root",
                "source": "sdk_run",
            }
        )


def test_root_sdk_invocation_persists_a_bounded_host_assertion(tmp_path) -> None:
    database = tmp_path / "provenance.sqlite"

    async def create() -> None:
        store = SQLiteSessionStore(database)
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="root",
                messages=[],
                invocation_origin=InvocationOriginClaim(
                    subject="application-user",
                    tenant="customer-a",
                ),
            ),
            identity=_identity(),
        )
        assert session.invocation.origin.trust is InvocationOriginTrust.HOST_ASSERTED
        await store.close()

    async def reopen() -> None:
        store = SQLiteSessionStore(database)
        loaded = await store.load("root")
        assert loaded is not None
        assert loaded.invocation.origin.subject == "application-user"
        assert loaded.invocation.origin.tenant == "customer-a"
        assert loaded.invocation.root_session_id == "root"
        assert loaded.invocation.source is SessionExecutionSource.SDK_RUN
        await store.close()

    asyncio.run(create())
    asyncio.run(reopen())


def test_sqlite_derives_child_invocation_inside_the_create_transaction(tmp_path) -> None:
    database = tmp_path / "atomic-provenance.sqlite"

    async def exercise() -> None:
        store = SQLiteSessionStore(database)
        await store.create(
            RunRequest(agent_name="assistant", session_id="root", messages=[]),
            identity=_identity(),
        )
        statements: list[str] = []
        store._connection.set_trace_callback(statements.append)
        try:
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="child",
                    parent_session_id="root",
                    messages=[],
                ),
                identity=_identity(),
            )
        finally:
            store._connection.set_trace_callback(None)
            await store.close()

        normalized = [" ".join(statement.upper().split()) for statement in statements]
        transaction_start = normalized.index("BEGIN IMMEDIATE")
        parent_read = next(
            index
            for index, statement in enumerate(normalized)
            if "FROM CAYU_SESSIONS" in statement and "WHERE ID = 'ROOT'" in statement
        )
        assert transaction_start < parent_read

    asyncio.run(exercise())


def test_derived_session_inherits_exact_root_origin() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        root = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="root",
                messages=[],
                invocation_origin=InvocationOriginClaim(subject="application-user"),
            ),
            identity=_identity(),
        )
        child_request = run_request_with_runtime_invocation(
            RunRequest(
                agent_name="assistant",
                session_id="child",
                parent_session_id=root.id,
                messages=[],
            ),
            source=SessionExecutionSource.SUBAGENT,
        )
        child = await store.create(child_request, identity=_identity())

        assert child.invocation.origin == root.invocation.origin
        assert child.invocation.root_invocation_id == root.invocation.root_invocation_id
        assert child.invocation.root_session_id == root.id
        assert child.invocation.source is SessionExecutionSource.SUBAGENT

    asyncio.run(exercise())


def test_derived_session_rejects_an_origin_override() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        root = await store.create(
            RunRequest(agent_name="assistant", session_id="root", messages=[]),
            identity=_identity(),
        )

        with pytest.raises(ValueError, match="must inherit"):
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="child",
                    parent_session_id=root.id,
                    invocation_origin=InvocationOriginClaim(subject="forged-user"),
                    messages=[],
                ),
                identity=_identity(),
            )

    asyncio.run(exercise())


def test_fork_invocation_preserves_root_and_identifies_the_fork_boundary() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        root = await store.create(
            RunRequest(agent_name="assistant", session_id="root", messages=[]),
            identity=_identity(),
        )

        invocation = fork_session_invocation(root)

        assert invocation.origin == root.invocation.origin
        assert invocation.root_invocation_id == root.invocation.root_invocation_id
        assert invocation.root_session_id == root.id
        assert invocation.source is SessionExecutionSource.FORK

    asyncio.run(exercise())
