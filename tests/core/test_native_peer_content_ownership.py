"""Peer persistence composes with native capabilities without retaining a store."""

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
from tests.core.test_peer_content import _delivery_request

import cayu
from cayu.collaboration.peer_content import (
    PeerContentConflict,
    PeerContentExposureRequest,
    PeerContentUnavailable,
)
from cayu.sessions.base import RunRequest
from cayu.sessions.records import SessionIdentity


def _assert_import_without_adapters(backend):
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
        if fullname in {"cayu.storage.sqlite", "cayu.storage.postgres"}:
            raise AssertionError(f"Peer owner imported an adapter: {fullname}")

sys.meta_path.insert(0, RejectAdapters())
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_peer_content")
assert callable(owner.append_peer_content)
assert callable(owner.retry_pending_peer_content)
""",
            backend,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


async def _seed_sessions(store):
    sessions = []
    for sid in ("source", "target"):
        sessions.append(
            await store.create(
                RunRequest(session_id=sid, agent_name="assistant", messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
        )
    return sessions


def _binding_values(session, participant):
    # Only identity columns are read by peer persistence; no binding decoder is used.
    return (
        session.id,
        "request",
        session.id,
        session.instance_id,
        "scope",
        "owner",
        "owner-v1",
        participant,
        participant + "-v1",
        1,
        1,
        1,
        "creator",
        "authorization",
        "input",
        "profile",
        "{}",
        "{}",
    )


def _request(sessions, suffix="direct"):
    return _delivery_request(
        source=sessions[0],
        target=sessions[1],
        sender=SimpleNamespace(participant_id="sender", incarnation="sender-v1"),
        consumer=SimpleNamespace(participant_id="consumer", incarnation="consumer-v1"),
        suffix=suffix,
    )


async def _exercise(ops, request, assert_idle):
    request = request.model_copy(
        update={
            "attempt_key": request.attempt_key.model_copy(update={"target_transcript_cursor": 1})
        }
    )
    assert await ops.read_peer_content(request.append_key) is None
    assert await ops.read_peer_content_attempt(request) is None
    pending = await ops.append_peer_content(request)
    assert pending.status == "pending" and pending.reason == "target_cursor_changed"
    assert await ops.list_pending_peer_content() == (request,)
    assert await ops.list_pending_peer_content(after_operation_key=request.operation_key) == ()

    def denied(session):
        assert session.id == request.append_key.target_session_id
        raise RuntimeError("Qualification denied")

    with pytest.raises(RuntimeError, match="Qualification denied"):
        await ops.append_peer_content(request, qualify_target=denied, pending_transcript_cursor=0)
    # Advancing a pending attempt deletes its old row before qualification.
    assert await ops.read_peer_content(request.append_key) == pending
    retry = dict(
        expected_session_instance_id=request.append_key.target_session_instance_id,
        expected_run_epoch=request.attempt_key.target_run_epoch,
        expected_transcript_cursor=0,
    )
    with pytest.raises(PeerContentUnavailable, match="fresh export authorization"):
        await ops.retry_pending_peer_content(request.append_key.target_session_id, **retry)

    async def admit(value, **kwargs):
        assert_idle()
        assert value == request and kwargs == {"pending_transcript_cursor": 0}
        return await ops.append_peer_content(value, qualify_target=lambda session: None, **kwargs)

    results = await ops.retry_pending_peer_content(
        request.append_key.target_session_id, admit=admit, **retry
    )
    (receipt,) = results
    assert receipt.status == "appended"
    assert await ops.list_pending_peer_content() == ()
    assert (await ops.append_peer_content(request)).replayed
    assert await ops.read_peer_content_attempt(request) == receipt
    assert await ops.read_peer_content(request.append_key) == receipt
    exposure = PeerContentExposureRequest(
        operation_key="exposure-op",
        append_key=request.append_key,
        exposure_id="exposure",
        model_attempt_id="model-attempt",
        provider_name="fake",
        capability_version=1,
        outcome="exposed",
    )
    begun = await ops.begin_peer_content_exposure(exposure)
    assert begun.outcome == "pending"
    assert (await ops.begin_peer_content_exposure(exposure)).replayed
    settled = await ops.record_peer_content_exposure(exposure)
    assert settled.outcome == "exposed"
    assert await ops.read_peer_content_exposure(request.append_key, exposure.exposure_id) == settled
    assert (await ops.record_peer_content_exposure(exposure)).replayed
    with pytest.raises(PeerContentConflict):
        await ops.record_peer_content_exposure(
            exposure.model_copy(update={"outcome": "not_exposed"})
        )
    assert (await ops.exclude_peer_content(request, reason="withdrawn")).status == "appended"
    assert_idle()


def test_sqlite_peer_imports_without_store_adapters():
    _assert_import_without_adapters("sqlite")


def test_sqlite_peer_operations_compose_with_direct_connection(tmp_path):
    from cayu.storage import _sqlite_connection, _sqlite_peer_content, _sqlite_records
    from cayu.storage._participant_bindings_schema import PARTICIPANT_BINDING_COLUMNS
    from cayu.storage.sqlite import SQLiteSessionStore

    async def exercise():
        path = tmp_path / "peer.sqlite"
        store = SQLiteSessionStore(path)
        try:
            sessions = await _seed_sessions(store)
        finally:
            await store.close()
        connection = _sqlite_connection.connect(path)
        # Match a connection opened without an authority codec, including event triggers.
        connection.create_function(
            "cayu_public_authority_alias", 3, lambda *_: None, deterministic=True
        )
        connection.create_function(
            "cayu_public_authority_aliases", 3, lambda *_: "[]", deterministic=True
        )
        connection.create_function(
            "cayu_public_authority_active_key_id", 0, lambda: None, deterministic=True
        )
        connection.create_function(
            "cayu_public_authority_keyring_fingerprint", 0, lambda: None, deterministic=True
        )
        try:
            connection.execute("UPDATE cayu_sessions SET status = 'running'")
            for session, participant in zip(sessions, ("sender", "consumer"), strict=True):
                connection.execute(
                    f"INSERT INTO cayu_participant_session_bindings ({','.join(PARTICIPANT_BINDING_COLUMNS)}) "
                    f"VALUES ({','.join('?' for _ in PARTICIPANT_BINDING_COLUMNS)})",
                    _binding_values(session, participant),
                )
            connection.commit()
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

            def assert_idle():
                assert not active and not connection.in_transaction

            async def read_creation(target):
                raise AssertionError("Direct targets must not read creation decisions")

            def load_session(sid):
                assert active and connection.in_transaction
                return _sqlite_records.load_session(connection, sid)

            ops = SimpleNamespace(
                **{
                    name: partial(getattr(_sqlite_peer_content, name), execute)
                    for name in (
                        "read_peer_content",
                        "read_peer_content_attempt",
                        "list_pending_peer_content",
                        "begin_peer_content_exposure",
                        "record_peer_content_exposure",
                        "read_peer_content_exposure",
                        "exclude_peer_content",
                    )
                }
            )
            ops.append_peer_content = partial(
                _sqlite_peer_content.append_peer_content,
                execute,
                load_unlocked=load_session,
                ownership_clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )
            ops.retry_pending_peer_content = partial(
                _sqlite_peer_content.retry_pending_peer_content,
                execute,
                read_creation_decision=read_creation,
            )
            await _exercise(ops, _request(sessions), assert_idle)
            failed = _request(sessions, "rollback")
            connection.execute("""CREATE TEMP TRIGGER fail_peer_receipt
                BEFORE INSERT ON cayu_peer_content_receipts
                BEGIN SELECT RAISE(ABORT, 'receipt write failed'); END""")
            import sqlite3

            with pytest.raises(sqlite3.IntegrityError, match="receipt write failed"):
                await ops.append_peer_content(failed, qualify_target=lambda session: None)
            assert_idle()
            assert await ops.read_peer_content(failed.append_key) is None
            assert (
                connection.execute("SELECT COUNT(*) FROM cayu_session_message_queue").fetchone()[0]
                == 1
            )
            assert connection.execute("SELECT COUNT(*) FROM cayu_events").fetchone()[0] == 1
            connection.execute("DROP TRIGGER fail_peer_receipt")
            excluded = await ops.exclude_peer_content(failed, reason="withdrawn")
            assert excluded.status == "excluded"
            assert (await ops.exclude_peer_content(failed, reason="withdrawn")).replayed
        finally:
            connection.close()

    asyncio.run(exercise())
