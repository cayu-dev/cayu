"""Native event publication composes with explicit transaction capabilities."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

import cayu
from cayu.events import Event, EventType
from cayu.sessions.base import BudgetReservationIdentityConflict
from cayu.sessions.mcp_manifest_history import (
    McpManifestBaseline,
    _mcp_authoritative_manifest_hash,
    _mcp_manifest_session_ref,
)
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest


def _assert_import_without_adapters_or_postgres_driver(backend):
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
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_event_publication")
assert callable(owner.append_events)
assert callable(owner.compare_and_publish_mcp_manifest_checks)
""",
            backend,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sqlite_event_publication_imports_without_adapters_or_postgres_driver():
    _assert_import_without_adapters_or_postgres_driver("sqlite")


async def _seed(store):
    session = await store.create(
        RunRequest(session_id="event-owner", agent_name="assistant", messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake"),
    )
    await store.append_event(
        session.id,
        Event(
            id="attempt-marker",
            type="custom.cayu.workflow.attempt",
            session_id=session.id,
            workflow_name="maintenance",
            payload={"attempt_id": "attempt"},
        ),
    )
    return session.id


def _publication(session_id):
    history_key = "sha256:" + "1" * 64
    event = Event(
        id="manifest-event",
        type=EventType.MCP_MANIFEST_CHECKED,
        session_id=session_id,
        payload={
            "history_key": history_key,
            "manifest_identity": "sha256:" + "2" * 64,
            "source_manifest_hash": "sha256:" + "3" * 64,
            "server_hash": "sha256:" + "4" * 64,
            "manifest_hash": _mcp_authoritative_manifest_hash(
                source_manifest_hash="sha256:" + "3" * 64,
                server_hash="sha256:" + "4" * 64,
                tools=(),
                exposed_tools=(),
            ),
            "status": "first_seen",
            "outcome": "accepted",
        },
    )
    baseline = McpManifestBaseline(
        history_key=history_key,
        generation=1,
        manifest_identity=event.payload["manifest_identity"],
        manifest_hash=event.payload["manifest_hash"],
        source_manifest_hash=event.payload["source_manifest_hash"],
        server_hash=event.payload["server_hash"],
        tools=(),
        exposed_tools=(),
        accepted_session_ref=_mcp_manifest_session_ref(session_id),
        accepted_event_id=event.id,
        accepted_at=event.timestamp,
    )
    return history_key, dict(
        expected_generations={history_key: None},
        baseline_updates={history_key: baseline},
        events=[event],
    )


async def _exercise_admission(ops, session_id):
    claim = dict(
        reservation_id="reservation", publication_session_id=session_id, publication_id="budget"
    )
    await ops.claim_budget_reservation_identity(**claim)
    await ops.claim_budget_reservation_identity(**claim)
    with pytest.raises(BudgetReservationIdentityConflict):
        await ops.claim_budget_reservation_identity(**{**claim, "publication_id": "other"})
    event = Event(
        id="workflow-step",
        type=EventType.WORKFLOW_STEP_STARTED,
        session_id=session_id,
        workflow_name="maintenance",
        payload={"attempt_id": "attempt", "step_id": "step"},
    )
    publish = partial(
        ops.append_workflow_step_started,
        session_id,
        workflow_name="maintenance",
        attempt_id="attempt",
    )
    assert await publish(event)
    assert not await publish(event)
    stale = event.model_copy(
        update={"id": "stale-step", "payload": {"attempt_id": "stale", "step_id": "step"}}
    )
    assert not await ops.append_workflow_step_started(
        session_id, stale, workflow_name="maintenance", attempt_id="stale"
    )


def test_sqlite_event_publication_composes_with_direct_connection(tmp_path):
    import sqlite3

    from cayu.storage import _sqlite_connection
    from cayu.storage import _sqlite_event_publication as owner
    from cayu.storage import sqlite as adapter

    async def run():
        path = tmp_path / "events.sqlite"
        store = adapter.SQLiteSessionStore(path)
        try:
            sid = await _seed(store)
        finally:
            await store.close()
        connection = _sqlite_connection.connect(path)
        # This independently composed fixture has no public-authority codec.
        for name, arity, value in (
            ("cayu_public_authority_alias", 3, None),
            ("cayu_public_authority_aliases", 3, "[]"),
            ("cayu_public_authority_active_key_id", 0, None),
            ("cayu_public_authority_keyring_fingerprint", 0, None),
        ):
            connection.create_function(
                name, arity, lambda *_, value=value: value, deterministic=True
            )
        active = False

        async def execute(operation):
            nonlocal active
            assert not active
            active = True
            try:
                return operation(connection)
            finally:
                active = False
                assert not connection.in_transaction

        def closure_owners(targets, *, connection=connection):
            assert active and connection.in_transaction and tuple(targets) == (sid,)
            return ()

        common = dict(
            store_now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            closure_owners=closure_owners,
            first_existing_event_id=adapter._first_existing_event_id,
        )
        ops = SimpleNamespace(
            claim_budget_reservation_identity=partial(
                owner.claim_budget_reservation_identity,
                execute,
                closure_owners=closure_owners,
                raise_write_conflict=adapter._raise_session_write_conflict,
                claim_identity=adapter._claim_budget_reservation_identity,
            ),
            append_workflow_step_started=partial(
                owner.append_workflow_step_started,
                execute,
                **common,
                touch_activity=adapter._touch_session_activity,
            ),
        )
        read = partial(owner.load_mcp_manifest_baselines, execute)
        publish = partial(
            owner.compare_and_publish_mcp_manifest_checks,
            execute,
            sid,
            **common,
            touch_activity=adapter._touch_session_activity,
        )

        def snapshot():
            return tuple(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "cayu_events",
                    "cayu_persisted_event_side_effects",
                    "cayu_mcp_manifest_baselines",
                )
            ) + tuple(connection.execute("SELECT last_activity_at FROM cayu_sessions").fetchone())

        try:
            await _exercise_admission(ops, sid)
            event = Event(id="batch", type="custom.batch", session_id=sid)
            await owner.append_events(
                execute, sid, [event], **common, append_events=adapter._append_events_in_transaction
            )
            with pytest.raises(ValueError, match="Event already exists"):
                await owner.append_events(
                    execute,
                    sid,
                    [event],
                    **common,
                    append_events=adapter._append_events_in_transaction,
                )
            key, publication = _publication(sid)
            before = snapshot()
            connection.execute("""CREATE TEMP TRIGGER fail_manifest_baseline
                BEFORE INSERT ON cayu_mcp_manifest_baselines
                BEGIN SELECT RAISE(ABORT, 'baseline write failed'); END""")
            with pytest.raises(sqlite3.IntegrityError, match="baseline write failed"):
                await publish(**publication)
            assert snapshot() == before
            assert (await read((key,))).baselines == {}
            connection.execute("DROP TRIGGER fail_manifest_baseline")
            assert (await publish(**publication)).published
            assert not (await publish(**publication)).published
            assert (await read((key,))).baselines == publication["baseline_updates"]
            assert snapshot()[:3] == tuple(count + 1 for count in before[:3])
        finally:
            connection.close()

    asyncio.run(run())
