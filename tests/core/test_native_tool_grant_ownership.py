"""Native grant owners compose with transaction capabilities and preserve rollback."""

from __future__ import annotations

import asyncio
import base64
import inspect
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

import cayu
from cayu.events import Event, EventType
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions.base import SessionRunFenced
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tools.grants import (
    PreparedTargetedToolGrant,
    TargetedToolGrant,
    TargetedToolUseRequest,
    build_targeted_tool_grant_record,
    persisted_targeted_tool_grant_batch_fingerprint,
    targeted_tool_grant_event,
)

OPERATIONS = (
    "issue_targeted_tool_grants",
    "list_targeted_tool_grants",
    "load_targeted_tool_grant_state",
    "bind_targeted_tool_grant_use",
    "revoke_targeted_tool_grant",
    "reconstruct_targeted_tool_grants",
)
TABLES = (
    "cayu_targeted_tool_grants",
    "cayu_targeted_tool_grant_uses",
    "cayu_public_authority_aliases",
    "cayu_events",
    "cayu_persisted_event_side_effects",
)


def assert_owner_imports(backend):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys
class RejectAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"cayu.storage.sqlite", "cayu.storage.postgres", "psycopg", "psycopg_pool"}:
            raise AssertionError(f"Unexpected adapter/driver dependency: {fullname}")
sys.meta_path.insert(0, RejectAdapters())
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_tool_grants")
assert callable(owner.issue_targeted_tool_grants)
assert callable(owner.reconstruct_targeted_tool_grants)
""",
            backend,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sqlite_tool_grant_owner_imports_without_adapters_or_driver():
    assert_owner_imports("sqlite")


def codec():
    key = base64.urlsafe_b64encode(bytes([53]) * 32).decode("ascii").rstrip("=")
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr(key)})
    )


async def seed(store, alias_codec):
    session = await store.create(
        RunRequest(session_id="grant-owner", agent_name="assistant", messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake"),
    )
    session = await store.transition_status(
        session.id,
        from_statuses={SessionStatus.PENDING},
        to_status=SessionStatus.RUNNING,
    )
    record = build_targeted_tool_grant_record(
        PreparedTargetedToolGrant(
            request=TargetedToolGrant(request_id="request", tool_id="cayu:remember", max_calls=2),
            tool_name="remember",
            catalogue_revision="sha256:" + "1" * 64,
            descriptor_version="sha256:" + "2" * 64,
            schema_fingerprint="sha256:" + "3" * 64,
        ),
        session_id=session.id,
        interaction_id="interaction",
        generation_id="sha256:" + "4" * 64,
        agent_name=session.agent_name,
        task_id=None,
        environment_name=session.environment_name,
        principal=session.invocation.origin.subject,
        tenant=session.invocation.origin.tenant,
        issued_at=datetime(2026, 1, 1, tzinfo=UTC),
        codec=alias_codec,
    )
    await store.append_event(
        session.id,
        Event(
            type=EventType.INTERACTION_STARTED,
            session_id=session.id,
            interaction_id=record.interaction_id,
            payload={
                "targeted_tool_grant_count": 1,
                "targeted_tool_grant_batch_fingerprint": persisted_targeted_tool_grant_batch_fingerprint(
                    (record,)
                ),
            },
        ),
    )
    issued = targeted_tool_grant_event(
        record,
        event_type=EventType.TARGETED_TOOL_GRANT_ISSUED,
        timestamp=record.issued_at,
        outcome="issued",
        event_id_suffix="issued",
    )
    return session, record, issued


def compose(owner, execute, capabilities):
    return SimpleNamespace(
        **{
            name: partial(
                getattr(owner, name),
                execute,
                **{
                    key: value
                    for key, value in capabilities.items()
                    if key in inspect.signature(getattr(owner, name)).parameters
                },
            )
            for name in OPERATIONS
        }
    )


async def exercise(ops, session, record, issued, snapshot, fault):
    async def read(operation, *args, **kwargs):
        fault.reading = True
        try:
            return await operation(*args, **kwargs)
        finally:
            fault.reading = False

    async def rollback_then_retry(operation):
        before = await snapshot()
        fault.enabled = True
        try:
            with pytest.raises(RuntimeError, match="audit write failed"):
                await operation()
        finally:
            fault.enabled = False
        assert await snapshot() == before
        return await operation()

    sid, epoch = session.id, session.run_epoch
    assert await read(ops.list_targeted_tool_grants, sid) == ()
    assert (await read(ops.load_targeted_tool_grant_state, sid)).records == ()
    issue = partial(
        ops.issue_targeted_tool_grants,
        sid,
        expected_run_epoch=epoch,
        records=(record,),
        events=(issued,),
    )
    assert (await rollback_then_retry(issue)).outcomes == ("issued",)
    assert (await issue()).outcomes == ("reused",)
    assert await read(ops.list_targeted_tool_grants, sid, interaction_id=record.interaction_id) == (
        record,
    )
    reconstruct = partial(
        ops.reconstruct_targeted_tool_grants,
        sid,
        expected_run_epoch=epoch,
        interaction_id=record.interaction_id,
        generation_id=record.generation_id,
        agent_name=record.agent_name,
        task_id=record.task_id,
        environment_name=record.environment_name,
        principal=record.principal,
        tenant=record.tenant,
        catalogue_revision=record.catalogue_revision,
        descriptors_by_id={
            record.tool_id: (record.tool_name, record.descriptor_version, record.schema_fingerprint)
        },
        capability_ceiling_names=frozenset({record.tool_name}),
        observed_at=record.issued_at,
    )
    assert (await rollback_then_retry(reconstruct)).valid == (record,)
    request = TargetedToolUseRequest(
        **{
            key: getattr(record, key)
            for key in (
                "tool_ref",
                "session_id",
                "interaction_id",
                "generation_id",
                "agent_name",
                "task_id",
                "environment_name",
                "principal",
                "tenant",
                "catalogue_revision",
                "descriptor_version",
                "schema_fingerprint",
                "tool_id",
                "tool_name",
            )
        },
        model_step_id="step",
        outer_tool_call_id="call",
        arguments_sha256="sha256:" + "5" * 64,
        invocation_id="invocation",
        expected_run_epoch=epoch,
    )
    before = await snapshot()
    with pytest.raises(SessionRunFenced):
        await ops.bind_targeted_tool_grant_use(
            request.model_copy(update={"expected_run_epoch": epoch + 1}),
            observed_at=record.issued_at,
        )
    assert await snapshot() == before
    bind = partial(ops.bind_targeted_tool_grant_use, request, observed_at=record.issued_at)
    bound = await rollback_then_retry(bind)
    assert bound.disposition == "bound"
    assert (await bind()).disposition == "rejoined"
    state = await read(ops.load_targeted_tool_grant_state, sid)
    assert state.records[0].used_calls == 1
    assert state.uses == (bound.binding,)
    revoke = partial(
        ops.revoke_targeted_tool_grant,
        record.tool_ref,
        session_id=sid,
        expected_run_epoch=epoch,
        reason="operator-revoked",
        revoked_at=record.issued_at + timedelta(seconds=1),
    )
    revoked = await rollback_then_retry(revoke)
    assert revoked.revocation_reason == "operator-revoked"
    assert await revoke() == revoked
    assert (await read(ops.load_targeted_tool_grant_state, sid)).records == (revoked,)


def test_sqlite_tool_grant_owner_composes_with_direct_connection(tmp_path):
    from cayu.storage import _sqlite_connection
    from cayu.storage import _sqlite_tool_grants as owner
    from cayu.storage import sqlite as adapter

    async def run():
        alias_codec = codec()
        path = tmp_path / "grants.sqlite"
        store = adapter.SQLiteSessionStore(path, public_authority_alias_codec=alias_codec)
        try:
            session, record, issued = await seed(store, alias_codec)
        finally:
            await store.close()
        connection = _sqlite_connection.connect(path)
        # The interaction authority was registered during seeding. These native
        # SQL functions use the same codec without retaining the store instance.
        adapter.SQLiteSessionStore._register_public_authority_alias_sql_function(
            SimpleNamespace(_public_authority_alias_codec=alias_codec), connection
        )
        active = False
        fault = SimpleNamespace(enabled=False, reading=False)

        async def execute(operation):
            nonlocal active
            assert not active
            active = True
            try:
                return operation(connection)
            finally:
                active = False
                assert not connection.in_transaction

        def append_events(conn, session_id, events, *, activity_at):
            assert active and conn is connection and conn.in_transaction
            adapter._append_events_in_transaction(conn, session_id, events, activity_at=activity_at)
            if fault.enabled:
                raise RuntimeError("audit write failed")

        def append_once(conn, event, *, activity_at):
            result = adapter._append_event_once_in_transaction(conn, event, activity_at=activity_at)
            if fault.enabled:
                raise RuntimeError("audit write failed")
            return result

        def closure_owners(targets, *, connection):
            assert active and connection.in_transaction and tuple(targets) == (session.id,)
            return ()

        def get_codec():
            assert active is fault.reading
            return alias_codec

        ops = compose(
            owner,
            execute,
            dict(
                store_now=lambda: record.issued_at,
                get_codec=get_codec,
                decode_grant=adapter._targeted_tool_grant_from_row,
                decode_use=adapter._targeted_tool_use_from_row,
                validate_use_counts=adapter._validate_targeted_tool_use_counts,
                append_events=append_events,
                append_event_once=append_once,
                closure_owners=closure_owners,
            ),
        )

        async def snapshot():
            assert not active
            return (
                *tuple(
                    tuple(
                        tuple(row)
                        for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1")
                    )
                    for table in TABLES
                ),
                tuple(connection.execute("SELECT last_activity_at FROM cayu_sessions").fetchone()),
            )

        try:
            await exercise(ops, session, record, issued, snapshot, fault)
        finally:
            connection.close()

    asyncio.run(run())
