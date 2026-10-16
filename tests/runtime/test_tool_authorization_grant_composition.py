"""Grant policy composition with taint, mandatory controls, reconciliation and use identity."""

from __future__ import annotations

import asyncio

import pytest
from tests.runtime.authorization_grant_support import Harness, _command_turn
from tests.runtime.runtime_provider_support import FakeProvider

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals import (
    InMemoryToolAuthorizationGrantStore,
    ToolAuthorizationGrantControl,
    ToolAuthorizationGrantPolicy,
    ToolAuthorizationGrantRequest,
    ToolAuthorizationGrantScopeKind,
    ToolAuthorizationGrantService,
    ToolAuthorizationGrantStatus,
)
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.policy import (
    AllowAllToolPolicy,
    GuardedToolPolicy,
    TaintAwareToolPolicy,
    ToolPolicyDecision,
)

SESSION = ToolAuthorizationGrantRequest(scope=ToolAuthorizationGrantScopeKind.SESSION)


def _types(events) -> list[str]:
    return [str(event.type) for event in events]


# 1. Taint-aware policies keep protecting through the grant wrapper


class _Recording(Tool):
    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(
            name=name,
            input_schema={"type": "object"},
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name=f"tests:grants:{name}", behavior_version="1", implementation_version="1"
            ),
        )
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content=f"{self.spec.name} ok")


def _calls(*names: str, call_prefix: str = "c") -> list[ModelStreamEvent]:
    return [
        *(
            ModelStreamEvent.tool_call(id=f"{call_prefix}{index}", name=name, arguments={})
            for index, name in enumerate(names)
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


_DONE = [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({"finish_reason": "stop"})]


class _PortableProvider(FakeProvider):
    @property
    def execution_profile_identity(self):
        from tests.core._execution_profile_fixtures import versioned_test_provider_identity

        return versioned_test_provider_identity(self)


def _taint_app(store, turns, decision: ToolPolicyDecision, *, guarded: bool):
    inbox, send = _Recording("read_inbox"), _Recording("send")
    taint = TaintAwareToolPolicy(
        taint_sources={"read_inbox": ("untrusted",)},
        protected_tools={"send": ("untrusted",)},
        decision=decision,
    )
    inner = GuardedToolPolicy(guards=(), then=taint) if guarded else taint
    policy = ToolAuthorizationGrantPolicy(
        policy_id="mixed",
        store=InMemoryToolAuthorizationGrantStore(),
        grantable={"exec_command": "process_command"},
        then=inner,
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(_PortableProvider(turns), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[inbox, send],
        tool_policy=GuardedToolPolicy(guards=(), then=policy) if guarded else policy,
    )
    return app, send


def _protected(events, decision: ToolPolicyDecision) -> bool:
    kinds = _types(events)
    if decision is ToolPolicyDecision.DENY:
        return str(EventType.TOOL_CALL_BLOCKED) in kinds
    return str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in kinds


@pytest.mark.parametrize("decision", [ToolPolicyDecision.DENY, ToolPolicyDecision.REQUIRE_APPROVAL])
@pytest.mark.parametrize("guarded", [False, True])
def test_wrapped_taint_policy_protects_a_later_round(tmp_path, decision, guarded) -> None:
    turns = [_calls("read_inbox"), _calls("send", call_prefix="d"), _DONE]
    app, send = _taint_app(InMemorySessionStore(), turns, decision, guarded=guarded)

    async def scenario():
        return [
            event
            async for event in app.run(
                RunRequest(agent_name="assistant", messages=[Message.text("user", "go")])
            )
        ]

    events = asyncio.run(scenario())
    assert _protected(events, decision)
    assert send.calls == []


@pytest.mark.parametrize("decision", [ToolPolicyDecision.DENY, ToolPolicyDecision.REQUIRE_APPROVAL])
def test_wrapped_taint_policy_protects_the_same_round(decision) -> None:
    app, send = _taint_app(
        InMemorySessionStore(), [_calls("read_inbox", "send"), _DONE], decision, guarded=False
    )

    async def scenario():
        return [
            event
            async for event in app.run(
                RunRequest(agent_name="assistant", messages=[Message.text("user", "go")])
            )
        ]

    events = asyncio.run(scenario())
    assert _protected(events, decision)
    assert send.calls == []


def test_wrapped_taint_policy_protects_after_a_restart(tmp_path) -> None:
    path = tmp_path / "sessions.sqlite"
    decision = ToolPolicyDecision.DENY

    async def scenario():
        first, _ = _taint_app(
            SQLiteSessionStore(path), [_calls("read_inbox"), _DONE], decision, guarded=False
        )
        async for _event in first.run(
            RunRequest(
                agent_name="assistant", session_id="t1", messages=[Message.text("user", "a")]
            )
        ):
            pass
        await first.session_store.close()
        second, send = _taint_app(
            SQLiteSessionStore(path),
            [_calls("send", call_prefix="d"), _DONE],
            decision,
            guarded=False,
        )
        events = [
            event
            async for event in second.resume(
                ResumeRequest(session_id="t1", messages=[Message.text("user", "b")])
            )
        ]
        await second.session_store.close()
        return events, send

    events, send = asyncio.run(scenario())
    assert _protected(events, decision)
    assert send.calls == []


# 2. Reconciliation never abandons a grant whose approval succeeds concurrently


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_reconcile_racing_the_approval_never_abandons_the_grant(tmp_path, backend) -> None:
    sessions = SQLiteSessionStore(tmp_path / "s.sqlite") if backend == "sqlite" else None
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["make", "test"]), sessions=sessions)

    async def scenario() -> None:
        await harness.run("s1")
        prepared = await harness.service.prepare(await harness.approval("s1"), SESSION)
        store = harness.sessions
        original = store.query_events
        raced = {"done": False}

        async def query_then_resolve(*args, **kwargs):
            page = await original(*args, **kwargs)
            if not raced["done"]:
                raced["done"] = True
                store.query_events = original
                async for _event in harness.app.resolve_tool_approval(prepared.approval_request):
                    pass
            return page

        store.query_events = query_then_resolve
        reconciled = await harness.service.reconcile(prepared.grant.grant_id)
        assert raced["done"]
        assert reconciled.status is not ToolAuthorizationGrantStatus.ABANDONED
        settled = await harness.service.reconcile(prepared.grant.grant_id)
        assert settled.status is ToolAuthorizationGrantStatus.ACTIVE

    asyncio.run(scenario())


# 3. Mandatory controls are enforced at authorization, whatever the downstream decision


def test_service_has_no_separate_mandatory_controls(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["ls"]))
    with pytest.raises(TypeError):
        ToolAuthorizationGrantService(
            harness.app,
            store=harness.grants,
            authority=harness.service._authority,
            mandatory_controls=(ToolAuthorizationGrantControl(tool_name="exec_command"),),
        )


@pytest.mark.parametrize("tool_name", ["exec_command", "deploy"])
def test_mandatory_controls_require_approval_even_when_downstream_allows(tool_name) -> None:
    tool = _Recording(tool_name)
    policy = ToolAuthorizationGrantPolicy(
        policy_id="mandatory",
        store=InMemoryToolAuthorizationGrantStore(),
        grantable={"exec_command": "process_command"},
        then=AllowAllToolPolicy(),
        mandatory_controls=(ToolAuthorizationGrantControl(tool_name=tool_name),),
    )
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(FakeProvider([_calls(tool_name), _DONE]), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"), tools=[tool], tool_policy=policy
    )

    async def scenario():
        return [
            event
            async for event in app.run(
                RunRequest(agent_name="assistant", messages=[Message.text("user", "go")])
            )
        ]

    events = asyncio.run(scenario())
    assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in _types(events)
    assert tool.calls == []


def test_a_control_added_later_overrides_an_existing_grant(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["make", "test"]))
    workspace = ToolAuthorizationGrantRequest(scope=ToolAuthorizationGrantScopeKind.WORKSPACE)

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", workspace)
        controlled = Harness(
            tmp_path,
            turns=_command_turn("call_2", ["make", "test"]),
            sessions=harness.sessions,
            grants=harness.grants,
            mandatory_controls=(ToolAuthorizationGrantControl(tool_name="exec_command"),),
        )
        later = await controlled.run("s2")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in _types(later)
        assert controlled.tool.calls == []

    asyncio.run(scenario())


# 4. Each execution under a grant leaves its own use record


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_repeated_provider_call_ids_in_later_rounds_record_each_use(tmp_path, backend) -> None:
    from cayu.storage import SQLiteToolAuthorizationGrantStore

    grants = (
        SQLiteToolAuthorizationGrantStore(tmp_path / "g.sqlite") if backend == "sqlite" else None
    )
    turns = (
        _command_turn("call_1", ["make", "test"])
        + _command_turn("call_1", ["make", "test"])
        + _command_turn("call_1", ["make", "test"])
    )
    harness = Harness(tmp_path, turns=turns, grants=grants)

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", SESSION)
        await harness.run("s1")
        await harness.run("s1")
        (grant,) = await harness.service.list_grants()
        uses = await harness.service.list_uses(grant.grant_id)
        assert len(harness.tool.calls) == 3
        assert len(uses) == 2
        assert len({use.tool_round_id for use in uses}) == 2

    asyncio.run(scenario())


# 5. Closing the grant stream finishes the runtime's own cleanup first


def _probe_resolution(harness: Harness) -> dict:
    state = {"inner_closed": False}
    original = harness.app.resolve_tool_approval

    def probe(request, *, context=None):
        async def stream():
            inner = original(request, context=context)
            try:
                async for event in inner:
                    yield event
            finally:
                await inner.aclose()
                state["inner_closed"] = True

        return stream()

    harness.app.resolve_tool_approval = probe
    return state


def test_closing_the_grant_stream_awaits_runtime_cleanup(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["make", "test"]))

    async def scenario() -> None:
        await harness.run("s1")
        state = _probe_resolution(harness)
        stream = harness.service.approve_and_grant(await harness.approval("s1"), SESSION)
        async for event in stream:
            if event.type is EventType.TOOL_CALL_APPROVED:
                break
        await stream.aclose()
        assert state["inner_closed"] is True
        assert (await harness.sessions.load("s1")).status.value != "running"
        (grant,) = await harness.service.list_grants()
        assert (await harness.service.reconcile(grant.grant_id)).status is (
            ToolAuthorizationGrantStatus.ACTIVE
        )

    asyncio.run(scenario())


def test_a_reconciliation_failure_still_awaits_runtime_cleanup(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["make", "test"]))

    async def scenario() -> None:
        await harness.run("s1")
        state = _probe_resolution(harness)
        original_activate = harness.grants.activate

        async def failing_activate(*args, **kwargs):
            raise RuntimeError("grant store unavailable")

        harness.grants.activate = failing_activate
        with pytest.raises(RuntimeError, match="grant store unavailable"):
            async for _event in harness.service.approve_and_grant(
                await harness.approval("s1"), SESSION
            ):
                pass
        assert state["inner_closed"] is True
        assert (await harness.sessions.load("s1")).status.value != "running"
        harness.grants.activate = original_activate
        (grant,) = await harness.service.list_grants()
        assert (await harness.service.reconcile(grant.grant_id)).status is (
            ToolAuthorizationGrantStatus.ACTIVE
        )

    asyncio.run(scenario())


# 6. Reconciliation never abandons a grant whose request digests changed meanwhile


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_reconcile_uses_the_current_request_digests(tmp_path, backend) -> None:
    from cayu.storage import SQLiteToolAuthorizationGrantStore

    grants = (
        SQLiteToolAuthorizationGrantStore(tmp_path / "g.sqlite") if backend == "sqlite" else None
    )
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["make", "test"]), grants=grants)

    async def scenario() -> None:
        await harness.run("s1")
        first = await harness.service.prepare(await harness.approval("s1"), SESSION)
        service = harness.service
        original_outcome = service._approval_outcome
        paused = {"done": False}

        async def outcome_after_a_second_prepare(grant):
            if not paused["done"]:
                paused["done"] = True
                # The same resolver retries with a new reason: identical grant terms,
                # a second request digest, then the approval succeeds with it.
                second = await service.prepare(
                    await harness.approval("s1", reason="updated reason"), SESSION
                )
                assert len(second.grant.resolution_request_digests) == 2
                async for _event in harness.app.resolve_tool_approval(second.approval_request):
                    pass
            return await original_outcome(grant)

        service._approval_outcome = outcome_after_a_second_prepare
        settled = await service.reconcile(first.grant.grant_id)
        assert paused["done"]
        assert settled.status is ToolAuthorizationGrantStatus.ACTIVE
        assert len(harness.tool.calls) == 1

    asyncio.run(scenario())
