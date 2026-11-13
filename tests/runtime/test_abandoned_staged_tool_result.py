"""Abandoning a run publishes the tool result its lost worker staged and didn't publish.

A tool call that finished stages its terminal event in the pending tool round, then
publishes it. A worker lost between the two (a deploy that replaced it, say) leaves
the result staged. Where the run's environment isn't registered, the session is
abandoned instead of recovered, and its round is retired into the abandonment
record. The staged result is published first, under the abandonment's claim: as
recorded when its after-tool hooks had run, with the result withheld when they
hadn't. Nothing is dispatched again. The main cases run on SQLite and PostgreSQL.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

pytest.importorskip("fastapi")

from tests.core.test_targeted_tool_grants import _codec, _GatewayProvider, _GatewayRememberTool
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.runtime.test_orphaned_session_reconciliation import (
    _request,
    _slot,
    _stores,
)
from tests.runtime.test_orphaned_session_reconciliation import (
    database_url as database_url,
)

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    RunRequest,
    SessionStatus,
    Tool,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import IncompleteSessionRecoveryAction
from cayu.storage import SQLiteSessionStore
from cayu.tools.exposure import StaticToolExposurePolicy
from cayu.tools.grants import TargetedToolGrant

_TERMINALS = frozenset({EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED})


def _spec(name: str) -> ToolSpec:
    # One call at a time, so a round's first call stages before its second starts.
    return ToolSpec(
        name=name,
        effect=ToolEffect.EXTERNAL,
        parallel_safe=False,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name=f"tests:abandoned-staged:{name}", behavior_version="1", implementation_version="1"
        ),
    )


class _Record(Tool):
    """An external call that records something and returns a receipt."""

    spec = _spec("record")

    def __init__(self) -> None:
        self.runs = 0

    async def run(self, ctx, args):
        self.runs += 1
        return ToolResult(content="recorded", structured={"receipt": "r-1"})


class _DieInside(Tool):
    """An external call the worker dies inside: its outcome is unknown."""

    spec = _spec("deliver")

    async def run(self, ctx, args):
        if os.environ.get("CAYU_STAGED_TEST_CRASH"):
            os._exit(137)
        raise AssertionError("An abandoned call must not run again.")


def _application(store, *, environment=None, tool: Tool | None = None, sibling: bool = False):
    """The scan's application; with ``sibling``, the round's second call kills its worker."""

    app = CayuApp(session_store=store, enable_logging=False)
    calls = [ModelStreamEvent.tool_call(id="call-1", name="record", arguments={})]
    if sibling:
        calls.append(ModelStreamEvent.tool_call(id="call-2", name="deliver", arguments={}))
    provider = VersionedFakeProvider(
        [*calls, ModelStreamEvent.completed({"finish_reason": "tool_calls"})]
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="scanner", model="fake-model"), tools=[tool or _Record(), _DieInside()]
    )
    if environment is not None:
        app.register_environment_factory(environment.environment.spec, environment)
    return app, provider


class _ExternalGatewayRemember(_GatewayRememberTool):
    # An external effect stages its result before publishing it.
    spec = _GatewayRememberTool.spec.model_copy(update={"effect": ToolEffect.EXTERNAL})


def _gateway_application(store, *, environment=None):
    """An agent that reaches its tool only through ``call_tool`` and a targeted grant."""

    app = CayuApp(session_store=store, enable_logging=False)
    provider = _GatewayProvider()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        targeted_tool_mode="call_tool",
        tools=(_ExternalGatewayRemember(),),
        tool_exposure_policy=StaticToolExposurePolicy(profile_id="targeted-only", tools=()),
    )
    if environment is not None:
        app.register_environment_factory(environment.environment.spec, environment)
    return app, provider


def _gateway_request() -> RunRequest:
    return RunRequest(
        agent_name="assistant",
        session_id="scan",
        environment_name=_request("scan").environment_name,
        messages=[Message.text("user", "Review and remember one fact.")],
        tool_grants=(
            TargetedToolGrant(
                request_id="remember-reviewed-fact",
                tool_id="cayu:remember",
                max_calls=1,
                lifetime_seconds=60,
            ),
        ),
    )


def _die_around_terminal_append(store, *, after: bool) -> None:
    """Exit the worker as it appends the call's terminal event, before or after it."""

    for name in ("append_event", "append_event_and_claim_side_effect", "append_events"):
        original = getattr(store, name)

        def append(*args, _original=original, **kwargs):
            values = [*args, *kwargs.values()]
            events = any(
                getattr(item, "type", None) in _TERMINALS
                for value in values
                for item in (value if isinstance(value, list) else [value])
            )

            async def run():
                if events and not after:
                    os._exit(137)
                result = await _original(*args, **kwargs)
                if events and after:
                    os._exit(137)
                return result

            return run()

        setattr(store, name, append)


def _run_worker() -> None:
    """The run's worker: the tool returns, and the worker dies at ``CAYU_STAGED_TEST_CRASH``."""

    async def scenario() -> None:
        crash = os.environ["CAYU_STAGED_TEST_CRASH"]
        if crash == "gateway":
            store = SQLiteSessionStore(
                os.environ["CAYU_STAGED_TEST_STORE"], public_authority_alias_codec=_codec()
            )
        else:
            store = _stores(
                os.environ["CAYU_STAGED_TEST_DATABASE_URL"], os.environ["CAYU_STAGED_TEST_STORE"]
            ).session_store
        if crash == "before_hooks":
            # Its result is staged; publishing it, hooks first, never starts.
            from cayu.runtime._tool_round_staging import _ToolRoundPublicationCoordinator

            async def die(*args, **kwargs):
                os._exit(137)

            _ToolRoundPublicationCoordinator.start_publication = die
        elif crash != "sibling":
            # Its hooks ran and its stage holds the final event; the append is lost,
            # or (``appended``) it commits and the round's commit is lost. With
            # ``sibling`` the first call's result is staged and the worker dies inside
            # the second call.
            _die_around_terminal_append(store, after=crash == "appended")
        if crash == "gateway":
            app, _provider = _gateway_application(store, environment=_slot())
            request = _gateway_request()
        else:
            app, _provider = _application(store, environment=_slot(), sibling=crash == "sibling")
            request = _request("scan")
        async for _ in app.run(request):
            pass
        raise AssertionError("The worker did not die as it published the tool result.")

    asyncio.run(scenario())


def _orphan(tmp_path, database_url: str, crash: str) -> None:
    child = subprocess.run(
        [sys.executable, "-c", f"from {__name__} import _run_worker; _run_worker()"],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(path for path in sys.path if path),
            "CAYU_STAGED_TEST_STORE": str(tmp_path / "cayu.db"),
            "CAYU_STAGED_TEST_DATABASE_URL": database_url,
            "CAYU_STAGED_TEST_CRASH": crash,
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 137, (child.stdout, child.stderr)


async def _staged(store) -> list[dict]:
    checkpoint = await store.load_checkpoint("scan")
    return checkpoint["pending_tool_round"]["staged_terminals"]


def _terminals(events) -> list:
    return [event for event in events if event.type in _TERMINALS]


@pytest.mark.parametrize("crash", ["before_append", "before_hooks"])
def test_abandoning_a_run_publishes_the_result_its_lost_worker_staged(
    tmp_path, database_url, crash
) -> None:
    _orphan(tmp_path, database_url, crash)

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        try:
            [stage] = await _staged(store)
            assert stage["hooks_state"] == ("completed" if crash == "before_append" else "pending")
            assert _terminals(await store.load_events("scan")) == []
            # A later Release's web process: the run's environment isn't registered here.
            tool = _Record()
            reader, provider = _application(store, tool=tool)

            [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)

            assert result.status is SessionStatus.FAILED
            # The call finished, so its outcome is known.
            assert result.actions == (IncompleteSessionRecoveryAction.ABANDONED_ORPHANED_RUN,)
            events = await store.load_events("scan")
            [terminal] = _terminals(events)
            assert terminal.id == stage["event"]["id"]
            assert terminal.payload["tool_call_id"] == "call-1"
            if crash == "before_append":
                # Its hooks had run: the result is published as the worker recorded it.
                assert terminal.type is EventType.TOOL_CALL_COMPLETED
                assert terminal.payload["result"]["content"] == "recorded"
                assert terminal.payload["result"]["structured"] == {"receipt": "r-1"}
            else:
                # Its hooks hadn't run and could have changed it, so it is withheld.
                assert terminal.type is EventType.TOOL_CALL_FAILED
                withheld = terminal.payload["result"]["structured"]
                assert withheld["reason"] == "abandoned_before_tool_result_hooks"
                assert "recorded" not in terminal.payload["result"]["content"]
            # Published under the abandonment, before the session failed.
            [failed] = [event for event in events if event.type is EventType.SESSION_FAILED]
            assert events.index(terminal) < events.index(failed)
            assert failed.payload["tool_effect_outcome_unknown"] == []
            record = (await store.load_checkpoint("scan"))["abandoned_session"]
            assert record["unknown_tool_calls"] == []
            assert record["published_tool_calls"] == [
                {
                    "tool_round_id": terminal.payload["tool_round_id"],
                    "tool_call_id": "call-1",
                    "tool_name": "record",
                    "event_id": terminal.id,
                    "result": "recorded" if crash == "before_append" else "withheld",
                }
            ]
            # Nothing ran again: no tool, no model.
            assert tool.runs == 0
            assert provider.requests == []
            # A second pass and a close change nothing.
            assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
            closed = await reader.close_session("scan", reason="Worker lost.")
            assert closed.status is SessionStatus.FAILED
            assert await store.load_events("scan") == events
            await reader.aclose()
        finally:
            await stores.close()

    asyncio.run(scenario())


def test_a_result_the_lost_worker_published_is_not_published_again(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'cayu.db'}"
    _orphan(tmp_path, database_url, "appended")

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        try:
            # The worker appended the call's event and died before its round committed.
            [stage] = await _staged(store)
            [published] = _terminals(await store.load_events("scan"))
            assert published.id == stage["event"]["id"]
            reader, provider = _application(store)

            [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)

            assert result.status is SessionStatus.FAILED
            assert _terminals(await store.load_events("scan")) == [published]
            record = (await store.load_checkpoint("scan"))["abandoned_session"]
            # Its stage is listed with the event the worker appended.
            [listed] = record["published_tool_calls"]
            assert (listed["event_id"], listed["result"]) == (published.id, "recorded")
            assert provider.requests == []
            await reader.aclose()
        finally:
            await stores.close()

    asyncio.run(scenario())


def _fail_once(patch, owner, name: str, error: BaseException) -> list[str]:
    """Make ``owner.name`` raise ``error`` on its first call only."""

    original = getattr(owner, name)
    calls: list[str] = []

    async def once(*args, **kwargs):
        calls.append(name)
        if len(calls) == 1:
            raise error
        return await original(*args, **kwargs)

    patch.setattr(owner, name, once)
    return calls


@pytest.mark.parametrize("crash", ["before_hooks", "sibling"])
def test_an_abandonment_interrupted_while_publishing_finishes_on_the_next_pass(
    tmp_path, monkeypatch, crash
) -> None:
    """A pass that stops partway publishes nothing twice and labels what it published.

    ``before_hooks`` stops before its publication starts. ``sibling`` stops after
    publishing, before the session fails: its round's second call never sealed its
    secret scope, so the first call's stage reads back differently from the event
    already published for it, which must not count as conflicting evidence.
    """

    database_url = f"sqlite:///{tmp_path / 'cayu.db'}"
    _orphan(tmp_path, database_url, crash)

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        try:
            [stage] = await _staged(store)
            assert stage["tool_call_id"] == "call-1"
            reader, provider = _application(store, sibling=crash == "sibling")
            if crash == "before_hooks":
                from cayu.runtime._tool_round_staging import _ToolRoundPublicationCoordinator

                failed = _fail_once(
                    monkeypatch,
                    _ToolRoundPublicationCoordinator,
                    "start_publication",
                    ConnectionError("store unavailable"),
                )
            else:
                failed = _fail_once(
                    monkeypatch,
                    reader._incomplete_recovery,
                    "_commit_abandonment",
                    ConnectionError("store unavailable"),
                )

            [first] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)
            assert first.actions == (IncompleteSessionRecoveryAction.FAILED,)
            assert (await store.load("scan")).status is SessionStatus.RUNNING
            [second] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)

            assert len(failed) == 2
            assert second.status is SessionStatus.FAILED
            [terminal] = _terminals(await store.load_events("scan"))
            assert terminal.id == stage["event"]["id"]
            assert terminal.type is EventType.TOOL_CALL_FAILED
            record = (await store.load_checkpoint("scan"))["abandoned_session"]
            [listed] = record["published_tool_calls"]
            assert (listed["tool_call_id"], listed["event_id"], listed["result"]) == (
                "call-1",
                terminal.id,
                "withheld",
            )
            # Only the call with no staged result has an unknown outcome.
            assert [call["tool_call_id"] for call in record["unknown_tool_calls"]] == (
                ["call-2"] if crash == "sibling" else []
            )
            assert provider.requests == []
            await reader.aclose()
        finally:
            await stores.close()

    asyncio.run(scenario())


def test_a_targeted_call_keeps_its_gateway_attribution(tmp_path) -> None:
    """A result staged for a ``call_tool`` call is published with its grant and use."""

    _orphan(tmp_path, f"sqlite:///{tmp_path / 'cayu.db'}", "gateway")

    async def scenario() -> None:
        store = SQLiteSessionStore(tmp_path / "cayu.db", public_authority_alias_codec=_codec())
        try:
            [stage] = await _staged(store)
            reader, provider = _gateway_application(store)

            [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)

            assert result.status is SessionStatus.FAILED
            events = await store.load_events("scan")
            [started] = [event for event in events if event.type is EventType.TOOL_CALL_STARTED]
            [completed] = _terminals(events)
            assert completed.id == stage["event"]["id"]
            assert completed.type is EventType.TOOL_CALL_COMPLETED
            assert completed.payload["result"]["content"] == (
                "remembered: Keep the gateway identity stable."
            )
            for field_name in ("dispatch_kind", "model_tool_name", "grant_id", "use_id"):
                assert completed.payload[field_name] == started.payload[field_name]
            assert completed.payload["dispatch_kind"] == "gateway"
            assert provider.requests == []
            await reader.aclose()
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("payload", "withheld"),
    [
        # A tool's own result may carry any JSON under "reason".
        ({"result": {"structured": {"reason": {"code": "not_found"}}}}, False),
        ({"result": {"structured": {"reason": ["a"]}}, "recovered": True}, False),
        # Only recovery's own replacement, marked recovered, counts as withheld.
        ({"result": {"structured": {"reason": "abandoned_before_tool_result_hooks"}}}, False),
        (
            {
                "result": {"structured": {"reason": "abandoned_before_tool_result_hooks"}},
                "recovered": True,
            },
            True,
        ),
        (
            {"result": {"structured": None}, "recovered": True, "secret_scope_incomplete": True},
            True,
        ),
    ],
)
def test_a_result_is_labelled_withheld_only_by_recovery(payload, withheld) -> None:
    from cayu.events import Event
    from cayu.runtime._abandoned_tool_result_publication import _withheld

    event = Event(
        type=EventType.TOOL_CALL_COMPLETED,
        session_id="scan",
        tool_name="record",
        payload={"tool_call_id": "call-1", **payload},
    )
    assert _withheld(event) is withheld
