"""A recovery item dismisses where its session's environment isn't registered.

A Cayu Cloud Agent registers a per-run environment in the process that runs the
session, and a later Release doesn't have it either. The web process that settles
the inbox must still abandon such a session, by the listed item id or by
``--session``, without the environment and without a 404.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from pydantic import SecretStr
from tests.runtime.runtime_environment_support import RecordingEnvironmentFactory
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.server.test_server_inbox import _as, _authenticate, _items, _portal, _settle

import cayu.cli.inbox as inbox_cli
from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    RunRequest,
    SessionStatus,
    Tool,
    ToolEffect,
    ToolSpec,
)
from cayu.cli import main
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.server import ServerConfig, create_server
from cayu.sessions import IncompleteSessionRecoveryAction
from cayu.sessions.records import SessionStatusConflict
from cayu.storage import SQLiteSessionStore
from cayu.tasks import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tools import UserInputTool
from cayu.vaults.redaction import SecretRedactor

_ENVIRONMENT = "run-1-slot-1"
_SECRET = "scan-run-secret-canary"


class _LostAcknowledgement(Tool):
    """An external call whose acknowledgement never arrived: its outcome is unknown."""

    spec = ToolSpec(name="record", effect=ToolEffect.EXTERNAL)

    async def run(self, ctx, args):
        raise ConnectionError("External acknowledgement lost.")


class _Blocking(Tool):
    spec = ToolSpec(name="record", effect=ToolEffect.EXTERNAL)

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, ctx, args):
        self.started.set()
        await self.release.wait()
        return "recorded"


def _codec() -> PublicAuthorityAliasCodec:
    key = base64.urlsafe_b64encode(bytes(range(32))).decode().rstrip("=")
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr(key)})
    )


def _store(path, *, projected: bool) -> SQLiteSessionStore:
    return SQLiteSessionStore(path, public_authority_alias_codec=_codec() if projected else None)


def _application(
    store,
    *,
    environment: RecordingEnvironmentFactory | None = None,
    tool: Tool | None = None,
    call: str = "record",
    arguments: dict | None = None,
    redactor: SecretRedactor | None = None,
    task_store=None,
) -> CayuApp:
    app = CayuApp(
        session_store=store, secret_redactor=redactor, task_store=task_store, enable_logging=False
    )
    app.register_provider(
        VersionedFakeProvider(
            [
                ModelStreamEvent.tool_call(id="call-1", name=call, arguments=arguments or {}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="scanner", model="fake-model"),
        tools=[tool or _LostAcknowledgement(), UserInputTool()],
    )
    if environment is not None:
        app.register_environment_factory(environment.environment.spec, environment)
    return app


def _slot() -> RecordingEnvironmentFactory:
    return RecordingEnvironmentFactory(Environment(EnvironmentSpec(name=_ENVIRONMENT)))


async def _run_in_slot(path, session_id: str, **options) -> None:
    """Run the session as the scan's process did: in a per-run environment it registered."""

    store = _store(path, projected=False)
    writer = _application(store, environment=_slot(), **options)
    async for _ in writer.run(
        RunRequest(
            session_id=session_id,
            agent_name="scanner",
            environment_name=_ENVIRONMENT,
            messages=[Message.text("user", "go")],
        )
    ):
        pass
    await writer.aclose()
    await store.close()


async def _assert_abandoned(store, session_id: str, reason: str) -> None:
    session = await store.load(session_id)
    assert session.status is SessionStatus.FAILED
    [failed] = [
        event
        for event in await store.load_events(session_id)
        if event.type is EventType.SESSION_FAILED
    ]
    assert failed.payload["error_type"] == "SessionAbandoned"
    assert failed.payload["metadata"]["reason"] == reason
    [unknown] = failed.payload["tool_effect_outcome_unknown"]
    assert unknown["tool_call_id"] == "call-1"
    # Nothing here could release the binding, so it stays recorded for where it can be.
    checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint(session_id)
    assert _ENVIRONMENT in checkpoint["environment_factory_reconnect"]


@pytest.mark.parametrize("projected", [False, True], ids=["plain-ids", "cloud-projected-ids"])
@pytest.mark.parametrize("by", ["item", "session"])
def test_a_recovery_item_dismisses_where_its_environment_isnt_registered(
    tmp_path, projected, by
) -> None:
    path = tmp_path / "sessions.sqlite"
    session_id = f"scan-{_SECRET}" if projected else "scan"
    asyncio.run(_run_in_slot(path, session_id))
    store = _store(path, projected=projected)
    # A later Release's web process: the run's environment was never registered here,
    # and on Cloud session ids are exposed as opaque references.
    reader = _application(store, redactor=SecretRedactor([_SECRET]) if projected else None)
    with TestClient(create_server(reader, config=ServerConfig.protected(_authenticate))) as client:
        [item] = _items(client, "priya")
        assert item["kind"] == "recovery"
        if projected:
            assert item["session_id"].startswith("cayu_authority_v1.")
            assert _SECRET not in json.dumps(item)
        else:
            assert item["session_id"] == session_id
        if by == "session":
            # `cayu inbox dismiss --session` reads the session's item first.
            listed = client.get(
                "/api/inbox", params={"session_id": item["session_id"]}, headers=_as("priya")
            )
            assert [found["id"] for found in listed.json()["items"]] == [item["id"]]
        assert client.get(f"/api/inbox/{item['id']}", headers=_as("priya")).status_code == 200

        dismissed = _settle(client, item["id"], "priya", decision="dismiss", reason="Run ended.")

        assert dismissed.status_code == 200, dismissed.text
        settlement = dismissed.json()["item"]["settlement"]
        assert (settlement["decision"], settlement["reason"]) == ("dismiss", "Run ended.")
        assert _items(client, "priya") == []
        again = _settle(client, item["id"], "priya", decision="dismiss", reason="Run ended.")
        assert again.status_code == 200, again.text
        assert again.json()["replayed"] is True
        _portal(client).call(_assert_abandoned, store, session_id, "Run ended.")


def test_a_recovery_item_with_a_running_task_stays_pending_without_its_environment(
    tmp_path,
) -> None:
    path = tmp_path / "sessions.sqlite"
    asyncio.run(_run_in_slot(path, "scan"))
    store = _store(path, projected=False)
    tasks = InMemoryTaskStore()
    reader = _application(store, task_store=tasks)
    with TestClient(create_server(reader, config=ServerConfig.protected(_authenticate))) as client:
        _portal(client).call(
            tasks.create_task, TaskCreate(task_id="scan-job", type="scan", session_id="scan")
        )
        [item] = _items(client, "priya")
        refused = _settle(client, item["id"], "priya", decision="dismiss", reason="Run ended.")
        assert refused.status_code == 409, refused.text
        detail = refused.json()["detail"]
        assert detail["code"] == "not_settleable"
        assert "tasks are all terminal" in detail["message"]
        assert detail["item"]["status"] == "pending"
        assert [listed["id"] for listed in _items(client, "priya")] == [item["id"]]
        session = _portal(client).call(store.load, "scan")
        assert session.status is SessionStatus.INTERRUPTED


def test_abandonment_without_the_environment_refuses_a_pause_that_isnt_an_unknown_outcome(
    tmp_path,
) -> None:
    path = tmp_path / "sessions.sqlite"
    asyncio.run(
        _run_in_slot(path, "asked", call="ask_user", arguments={"question": "Which branch?"})
    )

    async def scenario() -> None:
        store = _store(path, projected=False)
        reader = _application(store)
        before = await store.load("asked")
        with pytest.raises(SessionStatusConflict, match="isn.t registered in this application"):
            await reader.abandon_session("asked", reason="work_cancelled")
        assert await store.load("asked") == before
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())


def test_abandonment_without_the_environment_leaves_a_live_owner_alone(tmp_path) -> None:
    async def scenario() -> None:
        store = _store(tmp_path / "sessions.sqlite", projected=False)
        blocking = _Blocking()
        writer = _application(store, environment=_slot(), tool=blocking)

        async def run() -> None:
            async for _ in writer.run(
                RunRequest(
                    session_id="live",
                    agent_name="scanner",
                    environment_name=_ENVIRONMENT,
                    messages=[Message.text("user", "go")],
                )
            ):
                pass

        running = asyncio.create_task(run())
        try:
            await asyncio.wait_for(blocking.started.wait(), 10)
            reader = _application(store)
            result = await reader.abandon_session("live", reason="work_cancelled")
            assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT not in result.actions
            assert (await store.load("live")).status is SessionStatus.RUNNING
        finally:
            blocking.release.set()
            await asyncio.wait_for(running, 10)
        assert (await store.load("live")).status is not SessionStatus.FAILED
        await writer.aclose()
        await store.close()

    asyncio.run(scenario())


def test_the_cli_dismisses_a_recovery_item_by_session_without_its_environment(
    monkeypatch, tmp_path, capsys
) -> None:
    path = tmp_path / "sessions.sqlite"
    asyncio.run(_run_in_slot(path, "scan"))

    def build(target, **_kwargs):
        return _application(_store(path, projected=False))

    monkeypatch.setattr(
        inbox_cli,
        "resolve_project",
        lambda target, **_kwargs: SimpleNamespace(
            root=tmp_path, target=target or "app:build", service_target=None
        ),
    )
    monkeypatch.setattr(inbox_cli, "project_context", lambda _root: nullcontext())
    monkeypatch.setattr(inbox_cli, "build_project_app", build)

    code = main(
        [
            "inbox",
            "dismiss",
            "--session",
            "scan",
            "--reason",
            "Run ended.",
            "--reviewer",
            "priya",
            "--json",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert code == 0, output
    asyncio.run(_assert_abandoned(_store(path, projected=False), "scan", "Run ended."))


def test_abandonment_without_the_environment_measures_inactivity_from_last_activity(
    tmp_path,
) -> None:
    path = tmp_path / "sessions.sqlite"
    asyncio.run(_run_in_slot(path, "scan"))
    now = datetime.now(UTC)
    # Its status changed an hour ago, but it was active a few seconds ago.
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE cayu_sessions SET updated_at = ?, last_activity_at = ? WHERE id = ?",
            (
                (now - timedelta(hours=1)).isoformat(),
                (now - timedelta(seconds=5)).isoformat(),
                "scan",
            ),
        )

    async def scenario() -> None:
        store = _store(path, projected=False)
        # With a task store, a refusal must not retire the abandonment's task-set claim.
        reader = _application(store, task_store=InMemoryTaskStore())
        with pytest.raises(SessionStatusConflict, match="activity within the last 60 seconds"):
            await reader.abandon_session("scan", reason="work_cancelled", inactive_for_seconds=60)
        assert (await store.load("scan")).status is SessionStatus.INTERRUPTED
        result = await reader.abandon_session("scan", reason="work_cancelled")
        assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT in result.actions
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())


def test_abandonment_without_the_environment_leaves_a_session_another_recovery_claimed(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "sessions.sqlite"
    asyncio.run(_run_in_slot(path, "scan"))

    async def scenario() -> None:
        store = _store(path, projected=False)
        reader = _application(store)
        abandonment = reader._session_abandonment
        unregistered = abandonment._environment_unregistered

        async def claimed_in_between(session_id: str) -> bool:
            # A process that has the environment claims the session for recovery
            # after this one's recovery gave up on it, before the fallback commits.
            def claim(_session, checkpoint):
                now = datetime.now(UTC)
                claimed = dict(checkpoint or {})
                claimed["incomplete_session_recovery_claim"] = {
                    "version": 1,
                    "claim_id": "other-process",
                    "claimed_at": now.isoformat(),
                    "claim_expires_at": (now + timedelta(minutes=5)).isoformat(),
                }
                return claimed

            await store.transform_checkpoint(session_id, claim)
            return await unregistered(session_id)

        monkeypatch.setattr(abandonment, "_environment_unregistered", claimed_in_between)
        with pytest.raises(SessionStatusConflict, match="Another recovery owns the session"):
            await reader.abandon_session("scan", reason="work_cancelled")
        session = await store.load("scan")
        assert session.status is SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint("scan")
        assert checkpoint["incomplete_session_recovery_claim"]["claim_id"] == "other-process"
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())


def test_abandonment_without_the_environment_rechecks_activity_in_its_transition(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "sessions.sqlite"
    asyncio.run(_run_in_slot(path, "scan"))
    with sqlite3.connect(path) as connection:
        idle_since = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        connection.execute(
            "UPDATE cayu_sessions SET updated_at = ?, last_activity_at = ? WHERE id = ?",
            (idle_since, idle_since, "scan"),
        )

    async def scenario() -> None:
        store = _store(path, projected=False)
        reader = _application(store)
        recovery = reader._incomplete_recovery
        unknown_tool_calls = recovery._unknown_tool_calls

        async def written_in_between(*args, **kwargs):
            # Another process writes the session's checkpoint after the store-time check.
            await store.transform_checkpoint("scan", lambda _session, current: dict(current))
            return await unknown_tool_calls(*args, **kwargs)

        monkeypatch.setattr(recovery, "_unknown_tool_calls", written_in_between)
        with pytest.raises(SessionStatusConflict, match="activity within the last 60 seconds"):
            await reader.abandon_session("scan", reason="work_cancelled", inactive_for_seconds=60)
        assert (await store.load("scan")).status is SessionStatus.INTERRUPTED
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())
