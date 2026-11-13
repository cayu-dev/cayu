"""A recovery item whose session already failed can be dismissed.

A run can fail after a tool call started and before anything recorded how it ended,
as a run refused by its sandbox's admission does. Its session is ``failed`` with its
own terminal evidence, but its checkpoint keeps the round, so the call stays a
recovery item. Dismissing it never needs the call's arguments or the session's
environment: it retires the round, and the session stays failed.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from tests.core.pending_action_conformance import _pending_call
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.server.test_server_inbox import _as, _authenticate

from cayu import AgentSpec, CayuApp
from cayu.approvals import ToolArgumentReview
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime._session_abandonment import abandoned_session_from_checkpoint
from cayu.server import ServerConfig, create_server
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest

__all__ = ["conformance_postgres_dsn", "session_store_case"]

SESSION = "scout-20261026-135200-verify-1"
CALL = "call-exec"
ARGUMENTS = {"shell": "bash .scout/target-test tests/test_records.py"}
ROUND = {
    "model_step_id": "mstep_" + "a" * 32,
    "model_attempt_id": "matt_" + "a" * 32,
    "tool_round_id": "tround_" + "a" * 32,
}
_PROVIDER = VersionedFakeProvider([])


async def _fail_while_the_call_ran(
    store,
    *,
    environment: str | None,
    calls: tuple[str, ...] = (CALL,),
    results: tuple[str, ...] = (),
    session_id: str = SESSION,
) -> None:
    """The run started ``exec_command`` and failed before it recorded how the call ended."""

    await store.create(
        RunRequest(
            agent_name="verifier",
            session_id=session_id,
            messages=[Message.text("user", "verify")],
            environment_name=environment,
        ),
        identity=SessionIdentity(provider_name=_PROVIDER.name, model="fake-model"),
    )
    await store.append_events(
        session_id,
        [
            *(
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="exec_command",
                    payload={**ROUND, "tool_call_id": call, "arguments": ARGUMENTS},
                )
                for call in calls
            ),
            # Results the run published for the first call, if any.
            *(
                Event(
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id=session_id,
                    tool_name="exec_command",
                    payload={
                        **ROUND,
                        "tool_call_id": calls[0],
                        "result": {"content": content, "is_error": False},
                    },
                )
                for content in results
            ),
            Event(
                type=EventType.SESSION_FAILED,
                session_id=session_id,
                payload={
                    "error": "Execution candidate was refused: runner_poisoned.",
                    "error_type": "ExecutionAdmissionError",
                },
            ),
        ],
    )
    await store.checkpoint(
        session_id,
        {
            "pending_tool_round": {
                **ROUND,
                "agent_name": "verifier",
                "tool_calls": [
                    _pending_call(call, "exec_command", arguments=ARGUMENTS) for call in calls
                ],
            }
        },
    )
    await store.update_status(session_id, SessionStatus.FAILED)


def _app(store, **options) -> CayuApp:
    # A review policy is configured, so the item's arguments are quarantined: shown only
    # in a person's review, never in the item.
    app = CayuApp(
        session_store=store,
        human_review_policy=ToolArgumentReview({"exec_command": ("shell",)}, binding_key=b"k" * 32),
        enable_logging=False,
        **options,
    )
    app.register_provider(_PROVIDER, default=True)
    app.register_agent(AgentSpec(name="verifier", model="fake-model"))
    return app


async def _with_client(app: CayuApp, body):
    transport = httpx.ASGITransport(
        app=create_server(app, config=ServerConfig.protected(_authenticate))
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await body(client)


async def _items(client: httpx.AsyncClient) -> list[dict]:
    response = await client.get("/api/inbox", headers=_as("priya"))
    assert response.status_code == 200, response.text
    return response.json()["items"]


async def _settle(client: httpx.AsyncClient, item_id: str, person: str, **body):
    return await client.post(f"/api/inbox/{item_id}/settle", json=body, headers=_as(person))


@pytest.mark.parametrize("environment", ["20261026-135200-slot-3", None])
def test_every_store_dismisses_a_recovery_item_whose_session_already_failed(
    session_store_case, environment
) -> None:
    async def run():
        store = await _open_store(session_store_case)
        try:
            await _fail_while_the_call_ran(store, environment=environment)
            app = _app(store)

            async def decide(client: httpx.AsyncClient):
                [item] = await _items(client)
                unexplained = await _settle(client, item["id"], "priya", decision="dismiss")
                dismissed = await _settle(
                    client, item["id"], "priya", decision="dismiss", reason="Run ended."
                )
                left = await _items(client)
                state = await client.get(f"/api/inbox/{item['id']}", headers=_as("tom"))
                again = await _settle(
                    client, item["id"], "priya", decision="dismiss", reason="Run ended."
                )
                other = await _settle(
                    client, item["id"], "tom", decision="dismiss", reason="Run ended."
                )
                return item, unexplained, dismissed, left, state, again, other

            outcome = await _with_client(app, decide)
            session = await store.load(SESSION)
            events = await store.load_events(SESSION)
            checkpoint = await store.load_checkpoint(SESSION)
            return outcome, session, events, checkpoint
        finally:
            await _close_store(store)

    (item, unexplained, dismissed, left, state, again, other), session, events, checkpoint = (
        asyncio.run(run())
    )

    # An attention-backed item, its arguments quarantined, its session failed with a
    # started call that recorded no end. It offers dismiss all the same.
    assert item["kind"] == "recovery"
    assert item["attention_id"] is not None
    assert item["tool_name"] == "exec_command"
    assert item["arguments_unavailable_reason"] == "quarantined"
    assert item["detail"] == (
        "Tool started but no terminal result was recorded before the session failed."
    )
    assert item["decisions"] == ["dismiss"]

    assert unexplained.status_code == 422, unexplained.text
    assert dismissed.status_code == 200, dismissed.text
    assert dismissed.json()["replayed"] is False
    settled = dismissed.json()["item"]
    assert settled["status"] == "settled"
    settlement = settled["settlement"]
    assert (settlement["decision"], settlement["reason"]) == ("dismiss", "Run ended.")
    assert settlement["decided_by"] == {"subject": "priya", "tenant": None, "source": "http_auth"}
    assert settlement["decided_at"] is not None
    assert left == []
    assert state.status_code == 200, state.text
    assert state.json()["settlement"] == settlement
    assert again.status_code == 200, again.text
    assert again.json()["replayed"] is True
    assert again.json()["item"]["settlement"] == settlement
    assert other.status_code == 409, other.text
    assert other.json()["detail"]["code"] == "already_settled"

    # The session stays failed with its own failure; nothing ran and nothing failed again.
    assert session.status is SessionStatus.FAILED
    failures = [event for event in events if event.type is EventType.SESSION_FAILED]
    assert [failure.payload["error_type"] for failure in failures] == ["ExecutionAdmissionError"]
    assert not any(
        event.type
        in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED, EventType.SESSION_RESUMED}
        for event in events
    )
    assert "pending_tool_round" not in checkpoint
    record = abandoned_session_from_checkpoint(checkpoint)
    assert record is not None
    assert record["reason"] == "inbox_dismissed"
    assert record["metadata"]["reason"] == "Run ended."
    assert [call["tool_call_id"] for call in record["unknown_tool_calls"]] == [CALL]
    assert "abandoned_at" in record["after_failure"]


def test_a_failed_session_whose_owner_is_live_is_refused_and_stays_pending(
    tmp_path, monkeypatch
) -> None:
    from cayu.storage import SQLiteSessionStore

    async def run():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        try:
            await _fail_while_the_call_ran(store, environment="20261026-135200-slot-3")
            app = _app(store)

            async def live(_session):
                return "live"

            monkeypatch.setattr(
                app._session_abandonment._incomplete_recovery, "execution_owner", live
            )

            async def decide(client: httpx.AsyncClient):
                [item] = await _items(client)
                refused = await _settle(
                    client, item["id"], "priya", decision="dismiss", reason="Run ended."
                )
                return item, refused, await _items(client)

            outcome = await _with_client(app, decide)
            return outcome, await store.load_checkpoint(SESSION)
        finally:
            await store.close()

    (item, refused, left), checkpoint = asyncio.run(run())
    assert refused.status_code == 409, refused.text
    detail = refused.json()["detail"]
    assert detail["code"] == "not_settleable"
    assert "owner is live" in detail["message"]
    assert detail["item"]["status"] == "pending"
    assert detail["item"]["settlement"] is None
    assert [listed["id"] for listed in left] == [item["id"]]
    assert "pending_tool_round" in checkpoint
    assert abandoned_session_from_checkpoint(checkpoint) is None


def _sqlite_case(tmp_path, scenario):
    from cayu.storage import SQLiteSessionStore

    async def run():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        try:
            return await scenario(store)
        finally:
            await store.close()

    return asyncio.run(run())


def test_every_started_call_of_the_round_is_recorded_as_unknown(tmp_path) -> None:
    async def scenario(store):
        await _fail_while_the_call_ran(
            store, environment="20261026-135200-slot-3", calls=("first", "second")
        )
        app = _app(store)

        async def decide(client: httpx.AsyncClient):
            [item] = await _items(client)
            dismissed = await _settle(
                client, item["id"], "priya", decision="dismiss", reason="Run ended."
            )
            return dismissed, await _items(client)

        dismissed, left = await _with_client(app, decide)
        return dismissed, left, await store.load_checkpoint(SESSION)

    dismissed, left, checkpoint = _sqlite_case(tmp_path, scenario)
    assert dismissed.status_code == 200, dismissed.text
    assert dismissed.json()["item"]["settlement"]["decision"] == "dismiss"
    assert left == []
    record = abandoned_session_from_checkpoint(checkpoint)
    assert record is not None
    assert [call["tool_call_id"] for call in record["unknown_tool_calls"]] == ["first", "second"]


def test_a_call_with_contradictory_results_is_dismissed_as_unknown(tmp_path) -> None:
    async def scenario(store):
        # Two different results for one call: its outcome is still unknown.
        await _fail_while_the_call_ran(
            store, environment="20261026-135200-slot-3", results=("first", "second")
        )
        app = _app(store)

        async def decide(client: httpx.AsyncClient):
            [item] = await _items(client)
            dismissed = await _settle(
                client, item["id"], "priya", decision="dismiss", reason="Run ended."
            )
            return item, dismissed, await _items(client)

        outcome = await _with_client(app, decide)
        return *outcome, await store.load_checkpoint(SESSION)

    item, dismissed, left, checkpoint = _sqlite_case(tmp_path, scenario)
    assert item["decisions"] == ["dismiss"]
    assert "contradictory" in item["detail"]
    assert dismissed.status_code == 200, dismissed.text
    assert dismissed.json()["item"]["settlement"]["decision"] == "dismiss"
    assert left == []
    record = abandoned_session_from_checkpoint(checkpoint)
    assert record is not None
    assert [call["tool_call_id"] for call in record["unknown_tool_calls"]] == [CALL]


def test_a_recent_activity_refusal_leaves_the_failed_round_abandonable(tmp_path) -> None:
    from cayu.sessions.records import SessionStatusConflict
    from cayu.sessions.recovery import IncompleteSessionRecoveryAction
    from cayu.tasks.memory import InMemoryTaskStore

    async def scenario(store):
        await _fail_while_the_call_ran(store, environment="20261026-135200-slot-3")
        # With a task store, a refusal must not retire the abandonment's task-set claim.
        app = _app(store, task_store=InMemoryTaskStore())
        with pytest.raises(SessionStatusConflict, match="activity within the last 3600 seconds"):
            await app.abandon_session(SESSION, reason="work_cancelled", inactive_for_seconds=3600)
        assert "pending_tool_round" in await store.load_checkpoint(SESSION)
        result = await app.abandon_session(SESSION, reason="work_cancelled")
        return result, await store.load_checkpoint(SESSION)

    result, checkpoint = _sqlite_case(tmp_path, scenario)
    assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT in result.actions
    assert "pending_tool_round" not in checkpoint
    assert abandoned_session_from_checkpoint(checkpoint)["reason"] == "work_cancelled"


def test_a_failed_session_with_an_unfinished_child_is_refused_and_stays_pending(
    tmp_path,
) -> None:
    async def scenario(store):
        await _fail_while_the_call_ran(store, environment="20261026-135200-slot-3")
        await store.create(
            RunRequest(
                agent_name="verifier",
                session_id="child",
                parent_session_id=SESSION,
                messages=[Message.text("user", "check one file")],
            ),
            identity=SessionIdentity(provider_name=_PROVIDER.name, model="fake-model"),
        )
        app = _app(store)

        async def decide(client: httpx.AsyncClient):
            [item] = [item for item in await _items(client) if item["session_id"] == SESSION]
            refused = await _settle(
                client, item["id"], "priya", decision="dismiss", reason="Run ended."
            )
            return item, refused

        item, refused = await _with_client(app, decide)
        return item, refused, await store.load_checkpoint(SESSION)

    _item, refused, checkpoint = _sqlite_case(tmp_path, scenario)
    assert refused.status_code == 409, refused.text
    detail = refused.json()["detail"]
    assert detail["code"] == "not_settleable"
    assert "child sessions that haven't ended" in detail["message"]
    assert detail["item"]["status"] == "pending"
    assert "pending_tool_round" in checkpoint
    assert abandoned_session_from_checkpoint(checkpoint) is None


def test_round_evidence_keeps_only_the_rounds_events_without_their_results() -> None:
    from cayu.sessions import _pending_tool_round as pending_rounds
    from cayu.sessions.pending_actions import round_evidence_record
    from cayu.sessions.records import EventRecord

    pending_round = pending_rounds.PendingToolRound.model_validate(
        {
            **ROUND,
            "agent_name": "verifier",
            "tool_calls": [_pending_call(CALL, "exec_command", arguments=ARGUMENTS)],
        }
    )
    large = "x" * 100_000

    def completed(identity: dict[str, str]) -> EventRecord:
        return EventRecord(
            sequence=1,
            event=Event(
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=SESSION,
                tool_name="exec_command",
                payload={
                    **identity,
                    "tool_call_id": CALL,
                    "result": {"content": large, "is_error": False},
                },
            ),
        )

    other = {
        "model_step_id": "mstep_" + "b" * 32,
        "model_attempt_id": "matt_" + "b" * 32,
        "tool_round_id": "tround_" + "b" * 32,
    }
    assert round_evidence_record(completed(other), pending_round) is None
    kept = round_evidence_record(completed(ROUND), pending_round)
    assert kept is not None
    assert kept.event.payload["tool_call_id"] == CALL
    assert large not in json.dumps(kept.event.payload)
