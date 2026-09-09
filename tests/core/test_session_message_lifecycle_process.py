"""Actual process loss after committed SQLite queue actions, before owner cleanup."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sqlite3
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Literal

import pytest
from pydantic import SecretStr

from cayu import (
    CayuApp,
    EnqueueSessionMessageRequest,
    EventQuery,
    ResolutionActor,
    ResolutionActorSource,
    RunRequest,
    SessionIdentity,
    SessionMessageAccessContext,
    SessionMessageAccessDenied,
    SessionMessageAccessPolicy,
    SessionMessageActionRequest,
    SessionMessageConflict,
    SessionMessageDeliveryMode,
    SessionMessageQuery,
    SessionStatus,
    SQLiteSessionStore,
)
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.vaults import SecretRedactor

_SESSION = "process-queue-target"
_CONTEXT = SessionMessageAccessContext(subject="queue-operator", tenant="tenant-a")
_CONTENT = "private-process-steering-canary"
_SECRET = "workload-process-secret-canary"
_MALFORMED = '{"unregistered-malformed-process-canary":'
_EXIT_AFTER_COMMIT = 73
_ACTION_PERMISSIONS = ("enqueue", "inspect", "withdraw", "quarantine")


class _DurableOwnershipPolicy(SessionMessageAccessPolicy):
    """An application permission table, not a grant inferred from session metadata."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def authorize(
        self,
        context: SessionMessageAccessContext,
        *,
        session_id: str,
        session_instance_id: str,
        action: Literal["inspect", "enqueue", "source", "withdraw", "quarantine"],
    ) -> bool:
        connection = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)
        try:
            return (
                connection.execute(
                    "SELECT 1 FROM application_permissions "
                    "WHERE subject = ? AND tenant = ? AND session_id = ? "
                    "AND session_instance_id = ? AND action = ?",
                    (context.subject, context.tenant, session_id, session_instance_id, action),
                ).fetchone()
                is not None
            )
        finally:
            connection.close()


def _persist_permissions(path: Path, instance_id: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE application_permissions (subject TEXT NOT NULL, tenant TEXT NOT NULL, "
            "session_id TEXT NOT NULL, session_instance_id TEXT NOT NULL, action TEXT NOT NULL, "
            "PRIMARY KEY(subject, tenant, session_id, session_instance_id, action))"
        )
        connection.executemany(
            "INSERT INTO application_permissions VALUES (?, ?, ?, ?, ?)",
            [
                (_CONTEXT.subject, _CONTEXT.tenant, _SESSION, instance_id, action)
                for action in _ACTION_PERMISSIONS
            ],
        )
        connection.commit()
    finally:
        connection.close()


def _keyring() -> PublicAuthorityAliasKeyring:
    # Stable, test-only configuration shared by independent processes.
    return PublicAuthorityAliasKeyring(
        active_key_id="process-test",
        keys={
            "process-test": SecretStr(
                base64.urlsafe_b64encode(bytes(range(32))).decode().rstrip("=")
            )
        },
    )


def _store(directory: Path) -> SQLiteSessionStore:
    return SQLiteSessionStore(
        directory / "queue.sqlite",
        public_authority_alias_codec=PublicAuthorityAliasCodec(_keyring()),
    )


def _app(directory: Path, store: SQLiteSessionStore) -> CayuApp:
    return CayuApp(
        session_store=store,
        session_message_access_policy=_DurableOwnershipPolicy(directory / "ownership.sqlite"),
        secret_redactor=SecretRedactor(_SECRET),
        public_authority_alias_keyring=_keyring(),
        enable_logging=False,
    )


def _assert_content_free(value: str) -> None:
    assert _CONTENT not in value
    assert _SECRET not in value
    assert "unregistered-malformed-process-canary" not in value


def _assert_safe_warnings(caught: list[warnings.WarningMessage]) -> None:
    for warning in caught:
        _assert_content_free(str(warning.message))


async def _commit_and_die(
    directory: Path,
    action: Literal["withdraw", "quarantine"],
    caught: list[warnings.WarningMessage],
) -> None:
    database = directory / "queue.sqlite"
    store = _store(directory)
    session = await store.create(
        RunRequest(agent_name="unused", session_id=_SESSION, messages=[]),
        identity=SessionIdentity(provider_name="never-called", model="never-called"),
    )
    _persist_permissions(directory / "ownership.sqlite", session.instance_id)
    app = _app(directory, store)
    accepted = await app.enqueue_session_message(
        EnqueueSessionMessageRequest(
            session_id=_SESSION,
            idempotency_key="enqueue-before-loss",
            content=f"{_CONTENT}: {_SECRET}",
            delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
        ),
        context=_CONTEXT,
    )
    assert _SECRET not in accepted.message.content
    assert _CONTENT in accepted.message.content
    _assert_content_free(accepted.event.model_dump_json())

    if action == "quarantine":
        # Real external mutation, committed through a separate connection. A
        # corrupt row must remain quarantinable without parsing its content.
        connection = sqlite3.connect(database)
        try:
            changed = connection.execute(
                "UPDATE cayu_session_message_queue SET conditions_json = ? WHERE queue_id = ?",
                (_MALFORMED, accepted.message.queue_id),
            )
            assert changed.rowcount == 1
            connection.commit()
        finally:
            connection.close()

    page = await app.inspect_session_messages(
        SessionMessageQuery(session_id=_SESSION), context=_CONTEXT
    )
    assert len(page.records) == 1
    record = page.records[0]
    if action == "quarantine":
        assert record.validity == "unreadable" and record.message is None
        _assert_content_free(page.model_dump_json())
    request = SessionMessageActionRequest(
        session_id=_SESSION,
        session_instance_id=page.session_instance_id,
        queue_id=record.queue_id,
        expected_revision=record.revision,
        idempotency_key="terminal-before-loss",
        action=action,
    )
    result = await app.apply_session_message_action(request, context=_CONTEXT)
    status = "withdrawn" if action == "withdraw" else "quarantined"
    assert not result.replayed and result.record.status == status
    assert result.event.payload["actor"] == {
        "subject": _CONTEXT.subject,
        "tenant": _CONTEXT.tenant,
        "source": "request",
    }
    _assert_content_free(result.event.model_dump_json())
    assert await store.load_transcript(_SESSION) == []

    # Positive commit barrier: a different SQLite connection sees the terminal
    # receipt, not merely an in-process result or uncommitted writer state.
    observer = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    try:
        durable = observer.execute(
            "SELECT status, terminal_json FROM cayu_session_message_queue WHERE queue_id = ?",
            (request.queue_id,),
        ).fetchone()
        assert durable is not None and durable[0] == status
        receipt = json.loads(durable[1])
        assert receipt["action"]["expected_revision"] == request.expected_revision
        assert receipt["action"]["idempotency_key"] == request.idempotency_key
        assert receipt["event"]["type"] == f"session.message.{status}"
        _assert_content_free(durable[1])
    finally:
        observer.close()
    _assert_safe_warnings(caught)
    barrier = json.dumps(
        {
            "phase": "committed",
            "pid": os.getpid(),
            "request": request.model_dump(mode="json"),
            "event": result.event.model_dump(mode="json"),
            "revision": result.record.revision,
        }
    )
    _assert_content_free(barrier)
    print(barrier, flush=True)
    # Deliberately no store.close(), asyncio.run() shutdown, or app cleanup.
    os._exit(_EXIT_AFTER_COMMIT)


async def _fresh_process_replay(
    directory: Path, barrier: dict, caught: list[warnings.WarningMessage]
) -> None:
    store = _store(directory)
    try:
        app = _app(directory, store)
        request = SessionMessageActionRequest.model_validate(barrier["request"])
        expected_status = "withdrawn" if request.action == "withdraw" else "quarantined"
        query = SessionMessageQuery(session_id=_SESSION)
        before = await store.query_events(EventQuery(session_id=_SESSION))

        # Reconstructing the same exact queue/action identity is not authority.
        for denied_context in (
            SessionMessageAccessContext(subject=_CONTEXT.subject, tenant="tenant-b"),
            SessionMessageAccessContext(subject="other-operator", tenant=_CONTEXT.tenant),
        ):
            with pytest.raises(SessionMessageAccessDenied):
                await app.inspect_session_messages(query, context=denied_context)
            with pytest.raises(SessionMessageAccessDenied):
                await app.apply_session_message_action(request, context=denied_context)
        with pytest.raises(SessionMessageAccessDenied):
            await app.apply_session_message_action(
                request.model_copy(
                    update={
                        "requested_by": ResolutionActor(
                            subject=_CONTEXT.subject,
                            tenant=_CONTEXT.tenant,
                            source=ResolutionActorSource.HTTP_AUTH,
                            claims={"role": "admin"},
                        )
                    }
                ),
                context=_CONTEXT,
            )
        assert await store.query_events(EventQuery(session_id=_SESSION)) == before

        for _ in range(2):
            replay = await app.apply_session_message_action(request, context=_CONTEXT)
            assert replay.replayed and replay.record.status == expected_status
            assert replay.event.model_dump(mode="json") == barrier["event"]
            assert replay.record.revision == barrier["revision"]
            _assert_content_free(replay.event.model_dump_json())
        with pytest.raises(SessionMessageConflict):
            await app.apply_session_message_action(
                request.model_copy(update={"expected_revision": "f" * 64}),
                context=_CONTEXT,
            )
        page = await app.inspect_session_messages(query, context=_CONTEXT)
        assert page.session_instance_id == request.session_instance_id
        assert len(page.records) == 1 and page.records[0].status == expected_status
        assert page.records[0].queue_id == request.queue_id
        if request.action == "quarantine":
            assert page.records[0].message is None and page.records[0].validity == "unreadable"
            _assert_content_free(page.model_dump_json())
            observer = sqlite3.connect(directory / "queue.sqlite")
            try:
                assert (
                    observer.execute(
                        "SELECT conditions_json FROM cayu_session_message_queue WHERE queue_id = ?",
                        (request.queue_id,),
                    ).fetchone()[0]
                    == _MALFORMED
                )
            finally:
                observer.close()
        else:
            assert page.records[0].message is not None
            assert _CONTENT in page.records[0].message.content
            assert _SECRET not in page.records[0].message.content
        assert await store.load_transcript(_SESSION) == []
        session = await store.load(_SESSION)
        assert session is not None and session.status is SessionStatus.PENDING

        # Even a later live delivery boundary cannot turn the terminal record
        # back into input. No provider, network, or model is needed for this check.
        await store.transition_status(
            _SESSION,
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )
        batch = await store.deliver_queued_session_messages(
            _SESSION,
            include_on_idle=True,
            delivery_id="fresh-process-empty-drain",
        )
        assert batch.messages == () and batch.events == () and not batch.has_more
        assert await store.load_transcript(_SESSION) == []
        assert await store.load_transcript_cursor(_SESSION) == 0
        events = await store.query_events(EventQuery(session_id=_SESSION))
        terminal = [
            item for item in events if str(item.event.type) == f"session.message.{expected_status}"
        ]
        assert len(terminal) == 1
        assert terminal[0].event.payload["queue_id"] == request.queue_id
        assert terminal[0].event.payload["actor"] == {
            "subject": _CONTEXT.subject,
            "tenant": _CONTEXT.tenant,
            "source": "request",
        }
        assert [str(item.event.type) for item in events] == [
            "session.message.queued",
            f"session.message.{expected_status}",
        ]
        for item in events:
            _assert_content_free(item.event.model_dump_json())
        _assert_safe_warnings(caught)
        print(
            json.dumps(
                {
                    "phase": "replayed",
                    "pid": os.getpid(),
                    "status": expected_status,
                    "terminal_events": len(terminal),
                    "transcript_cursor": 0,
                }
            ),
            flush=True,
        )
    finally:
        await store.close()


def _process(
    directory: Path, mode: str, *, action: str = "", input_text: str = ""
) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode, str(directory), action],
        cwd=root,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root)))},
        input=input_text,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


@pytest.mark.parametrize("action", ["withdraw", "quarantine"])
def test_sqlite_committed_action_survives_process_termination_and_scoped_sdk_replay(
    tmp_path, action
):
    crashed = _process(tmp_path, "commit", action=action)
    _assert_content_free(crashed.stdout + crashed.stderr)
    assert crashed.returncode == _EXIT_AFTER_COMMIT, crashed.stderr
    barrier = json.loads(crashed.stdout)
    assert barrier["phase"] == "committed" and barrier["pid"] != os.getpid()
    replayed = _process(tmp_path, "replay", input_text=crashed.stdout)
    _assert_content_free(replayed.stdout + replayed.stderr)
    assert replayed.returncode == 0, replayed.stderr
    result = json.loads(replayed.stdout)
    assert result["phase"] == "replayed"
    assert result["pid"] not in {os.getpid(), barrier["pid"]}
    assert result["terminal_events"] == 1 and result["transcript_cursor"] == 0
    assert result["status"] == ("withdrawn" if action == "withdraw" else "quarantined")


if __name__ == "__main__":
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        directory = Path(sys.argv[2])
        if sys.argv[1] == "commit":
            assert sys.argv[3] in {"withdraw", "quarantine"}
            asyncio.run(
                _commit_and_die(
                    directory,
                    "withdraw" if sys.argv[3] == "withdraw" else "quarantine",
                    recorded,
                )
            )
        else:
            assert sys.argv[1] == "replay"
            asyncio.run(_fresh_process_replay(directory, json.loads(sys.stdin.read()), recorded))
            _assert_safe_warnings(recorded)
