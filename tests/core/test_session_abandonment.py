from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from tests.core.test_abandoned_session_resume import (
    _child,
    _crash_child,
    _gate_policy,
    _Proposal,
    _proposal_app,
)
from tests.core.test_session_execution_presence import _app, _stores
from tests.runtime.runtime_provider_support import VersionedFakeProvider

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, SessionStatus, ToolEffect
from cayu.approvals import ToolApprovalRequest, UserInputResponse
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.sessions import (
    ForkSessionRequest,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    SessionExecutionConfig,
)
from cayu.sessions.records import SessionStatusConflict
from cayu.tasks import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tools import UserInputTool


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("recovered_first", [False, True])
def test_abandon_session_fails_unknown_tool_outcome_without_redispatch(
    backend, recovered_first, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "abandon-receipt"
            sid = "abandon-" + uuid4().hex
            await _crash_child(
                _child(store, backend, request, receipt, sid, ToolEffect.EXTERNAL, "crash")
            )
            provider = VersionedFakeProvider([])
            observer = reopen()
            app = _proposal_app(
                observer,
                _Proposal(ToolEffect.EXTERNAL, receipt),
                provider,
                lease_seconds=1.5,
            )
            if recovered_first:
                [swept] = await app.recover_abandoned_sessions(inactive_for_seconds=0)
                assert swept.status is SessionStatus.INTERRUPTED
                assert IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT in swept.actions

            result = await app.abandon_session(
                sid, reason="work_cancelled", metadata={"ticket": "T-1"}
            )

            assert result.status is SessionStatus.FAILED, result
            assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT in result.actions
            session = await observer.load(sid)
            assert session.status is SessionStatus.FAILED
            assert receipt.with_suffix(".calls").read_text().count("\n") == 1
            assert provider.requests == []
            events = await observer.load_events(sid)
            [failed] = [event for event in events if event.type is EventType.SESSION_FAILED]
            [started] = [event for event in events if event.type is EventType.TOOL_CALL_STARTED]
            round_id = started.payload["tool_round_id"]
            assert failed.payload["error_type"] == "SessionAbandoned"
            assert failed.payload["reason"] == "work_cancelled"
            assert failed.payload["metadata"] == {"ticket": "T-1"}
            assert failed.payload["tool_effect_outcome_unknown"] == [
                {"tool_round_id": round_id, "tool_call_id": "proposal", "tool_name": "propose"}
            ]
            assert not any(
                event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
                for event in events
            )
            checkpoint = await runtime_checkpoint_session_store(observer).load_checkpoint(sid)
            assert "pending_tool_round" not in checkpoint
            assert (
                checkpoint["abandoned_session"]["unknown_tool_calls"]
                == (failed.payload["tool_effect_outcome_unknown"])
            )
            retained_abandonment = checkpoint["abandoned_session"]
            for replacement in (
                {"application": "ordinary state"},
                {"abandoned_session": {"reason": "forged"}},
            ):
                await observer.checkpoint(sid, replacement)
                assert (await observer.load_checkpoint(sid))[
                    "abandoned_session"
                ] == retained_abandonment
                with pytest.raises(SessionStatusConflict, match="abandoned session"):
                    _ = [
                        event
                        async for event in app.resume(
                            ResumeRequest(
                                session_id=sid, messages=[Message.text("user", "Continue")]
                            )
                        )
                    ]
            assert not provider.requests
            effect = await ToolEffectStateOwner(observer).resolve_call(
                session, tool_round_id=round_id, tool_call_id="proposal"
            )
            assert effect.state == "outcome_unknown"

            repeated = await app.abandon_session(sid)
            assert repeated.status is SessionStatus.FAILED
            assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT not in repeated.actions
            assert [event.type for event in await observer.load_events(sid)].count(
                EventType.SESSION_FAILED
            ) == 1
            assert receipt.with_suffix(".calls").read_text().count("\n") == 1
            assert provider.requests == []

            # Abandonment ends this session only; a fork does not inherit it.
            fork_id = sid + "-fork"
            async for _ in app.fork_session(
                ForkSessionRequest(source_session_id=sid, session_id=fork_id)
            ):
                pass
            fork_checkpoint = await runtime_checkpoint_session_store(observer).load_checkpoint(
                fork_id
            )
            assert "abandoned_session" not in (fork_checkpoint or {})

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite"])
def test_abandonment_requires_the_task_store_of_an_attached_round(
    backend, request, sqlite_resources, tmp_path, monkeypatch
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "abandon-attached-receipt"
            sid = "abandon-attached-" + uuid4().hex
            await _crash_child(
                _child(store, backend, request, receipt, sid, ToolEffect.EXTERNAL, "crash")
            )
            observer = reopen()
            app = _proposal_app(
                observer,
                _Proposal(ToolEffect.EXTERNAL, receipt),
                VersionedFakeProvider([]),
                lease_seconds=1.5,
            )
            # The crashed round names its task, but this app has no task store.
            import cayu.runtime._session_abandonment as abandonment

            real = abandonment.pending_tool_round_from_checkpoint

            def attached(checkpoint):
                pending = real(checkpoint)
                return None if pending is None else pending.model_copy(update={"task_id": "job"})

            monkeypatch.setattr(abandonment, "pending_tool_round_from_checkpoint", attached)
            before = await observer.load(sid)
            with pytest.raises(SessionStatusConflict, match="requires its task store"):
                await app.abandon_session(sid)
            assert await observer.load(sid) == before

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite"])
def test_abandonment_waits_for_the_owning_task_to_be_terminal(
    backend, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "abandon-task-receipt"
            sid = "abandon-task-" + uuid4().hex
            await _crash_child(
                _child(store, backend, request, receipt, sid, ToolEffect.EXTERNAL, "crash")
            )
            provider = VersionedFakeProvider([])
            observer = reopen()
            tasks = InMemoryTaskStore()
            app = _app(
                observer,
                provider,
                tools=[_Proposal(ToolEffect.EXTERNAL, receipt)],
                task_store=tasks,
            )
            await tasks.create_task(TaskCreate(task_id="job", type="job", session_id=sid))
            before = await observer.load(sid)

            with pytest.raises(SessionStatusConflict, match="tasks are all terminal"):
                await app.abandon_session(sid)
            assert await observer.load(sid) == before

            # While the task is active the sweep only interrupts the session.
            [swept] = await app.recover_abandoned_sessions(
                inactive_for_seconds=0, abandon_sessions_of_terminal_tasks=True
            )
            assert swept.status is SessionStatus.INTERRUPTED
            assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT not in swept.actions

            # A later sweep revisits it once the task is terminal.
            await tasks.cancel_task("job")
            [result] = await app.recover_abandoned_sessions(
                inactive_for_seconds=0, abandon_sessions_of_terminal_tasks=True
            )
            assert result.status is SessionStatus.FAILED, result
            assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT in result.actions
            assert (await observer.load(sid)).status is SessionStatus.FAILED
            assert (
                await app.recover_abandoned_sessions(
                    inactive_for_seconds=0, abandon_sessions_of_terminal_tasks=True
                )
                == ()
            )
            assert receipt.with_suffix(".calls").read_text().count("\n") == 1
            assert provider.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite"])
def test_abandoned_session_sweep_keeps_sessions_without_tasks(
    backend, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "keep-receipt"
            sid = "keep-" + uuid4().hex
            await _crash_child(
                _child(store, backend, request, receipt, sid, ToolEffect.EXTERNAL, "crash")
            )
            app = _app(
                reopen(),
                VersionedFakeProvider([]),
                tools=[_Proposal(ToolEffect.EXTERNAL, receipt)],
                task_store=InMemoryTaskStore(),
            )
            [result] = await app.recover_abandoned_sessions(
                inactive_for_seconds=0, abandon_sessions_of_terminal_tasks=True
            )
            assert result.status is SessionStatus.INTERRUPTED
            assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT not in result.actions

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite"])
def test_abandon_session_leaves_a_live_executor_alone(backend, request, sqlite_resources, tmp_path):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "abandon-live-receipt"
            sid = "abandon-live-" + uuid4().hex
            child = _child(store, backend, request, receipt, sid, ToolEffect.EXTERNAL, "block")
            try:
                assert (
                    await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), 30)
                    == "committed\n"
                )
                app = _proposal_app(
                    reopen(),
                    _Proposal(ToolEffect.EXTERNAL, receipt),
                    VersionedFakeProvider([]),
                    lease_seconds=1.5,
                )
                result = await app.abandon_session(sid)
                assert result.actions == (IncompleteSessionRecoveryAction.SKIPPED_EXECUTION_OWNER,)
                assert (await store.load(sid)).status is SessionStatus.RUNNING
            finally:
                if child.poll() is None:
                    child.kill()
                await asyncio.to_thread(child.wait)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite"])
def test_abandoned_session_refuses_resume_before_cleanup(
    backend, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "abandon-fence-receipt"
            sid = "abandon-fence-" + uuid4().hex
            await _crash_child(
                _child(store, backend, request, receipt, sid, ToolEffect.EXTERNAL, "crash")
            )
            provider = VersionedFakeProvider([])
            observer = reopen()
            tasks = InMemoryTaskStore()
            app = _app(
                observer,
                provider,
                tools=[_Proposal(ToolEffect.EXTERNAL, receipt)],
                task_store=tasks,
            )
            await tasks.create_task(TaskCreate(task_id="job", type="job", session_id=sid))
            await tasks.cancel_task("job")
            # Stop after the first pass, as a crash before the cleanup pass would.
            first = await app._session_recovery.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=sid, inactive_for_seconds=0, reason="work_cancelled"
                ),
                abandon_unknown_tool_effects=True,
            )
            assert first.status is SessionStatus.FAILED
            assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT in first.actions

            with pytest.raises(SessionStatusConflict, match="abandoned session"):
                async for _ in app.resume(
                    ResumeRequest(session_id=sid, messages=[Message.text("user", "Continue")])
                ):
                    pass
            assert provider.requests == []
            assert (await observer.load(sid)).status is SessionStatus.FAILED
            # Without an environment the first pass already records the failure.
            [failed] = [
                event
                for event in await observer.load_events(sid)
                if event.type is EventType.SESSION_FAILED
            ]
            assert failed.payload["error_type"] == "SessionAbandoned"
            assert failed.payload["reason"] == "work_cancelled"
            assert (
                await app.recover_abandoned_sessions(
                    inactive_for_seconds=0, abandon_sessions_of_terminal_tasks=True
                )
                == ()
            )
            repeated = await app.abandon_session(sid)
            assert repeated.status is SessionStatus.FAILED
            assert provider.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite"])
def test_one_abandonment_failure_does_not_strand_the_rest_of_the_sweep(
    backend, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            sids = []
            for index in range(2):
                receipt = tmp_path / f"isolated-receipt-{index}"
                sid = f"isolated-{index}-" + uuid4().hex
                await _crash_child(
                    _child(store, backend, request, receipt, sid, ToolEffect.EXTERNAL, "crash")
                )
                sids.append(sid)
            observer = reopen()
            tasks = InMemoryTaskStore()
            app = _app(
                observer,
                VersionedFakeProvider([]),
                tools=[_Proposal(ToolEffect.EXTERNAL, tmp_path / "isolated-receipt-0")],
                task_store=tasks,
            )
            for index, sid in enumerate(sids):
                await tasks.create_task(
                    TaskCreate(task_id=f"job-{index}", type="job", session_id=sid)
                )
                await tasks.cancel_task(f"job-{index}")
            abandonment = app._session_abandonment
            real = abandonment.abandon
            failing = {sids[0]}

            async def abandon(session_id, **kwargs):
                if session_id in failing:
                    failing.discard(session_id)
                    raise RuntimeError("factory reconnect unavailable")
                return await real(session_id, **kwargs)

            abandonment.abandon = abandon
            results = await app.recover_abandoned_sessions(
                inactive_for_seconds=0, abandon_sessions_of_terminal_tasks=True
            )
            first = [result for result in results if result.session_id == sids[0]]
            second = [result for result in results if result.session_id == sids[1]]
            # The failure is reported and the sweep goes on to the next session;
            # its revisit of unfinished sessions then retries the first one.
            assert IncompleteSessionRecoveryAction.FAILED in first[0].actions
            assert first[0].status is SessionStatus.INTERRUPTED
            assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT in first[-1].actions
            assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT in second[0].actions
            for sid in sids:
                assert (await observer.load(sid)).status is SessionStatus.FAILED
            assert (
                await app.recover_abandoned_sessions(
                    inactive_for_seconds=0, abandon_sessions_of_terminal_tasks=True
                )
                == ()
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite"])
@pytest.mark.parametrize("gate", ["approval", "input"])
@pytest.mark.parametrize("task_finished_later", [False, True])
def test_sweep_abandons_unknown_outcomes_held_by_a_resolved_gate(
    backend, gate, task_finished_later, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / f"gated-{gate}-receipt"
            sid = f"gated-{gate}-" + uuid4().hex
            # The gate is resolved and the approved or answered external call
            # dies mid-call, leaving its outcome unknown under the gate.
            await _crash_child(
                _child(
                    store, backend, request, receipt, sid, ToolEffect.EXTERNAL, "crash", gate=gate
                )
            )
            resolution = receipt.with_suffix(".resolution.json").read_text()
            observer = reopen()
            tasks = InMemoryTaskStore()
            tool = _Proposal(ToolEffect.EXTERNAL, receipt)
            provider = VersionedFakeProvider([])
            app = CayuApp(
                session_store=observer,
                task_store=tasks,
                enable_logging=False,
                session_execution=SessionExecutionConfig(
                    heartbeat_interval_seconds=0.05, lease_seconds=1.5
                ),
            )
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[tool, UserInputTool()] if gate == "input" else [tool],
                tool_policy=_gate_policy(gate, receipt),
            )
            await tasks.create_task(TaskCreate(task_id="job", type="job", session_id=sid))
            if task_finished_later:
                [swept] = await app.recover_abandoned_sessions(
                    inactive_for_seconds=0, abandon_sessions_of_terminal_tasks=True
                )
                assert swept.status is SessionStatus.INTERRUPTED
                assert IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT in swept.actions
            await tasks.cancel_task("job")

            [result] = await app.recover_abandoned_sessions(
                inactive_for_seconds=0, abandon_sessions_of_terminal_tasks=True
            )
            assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT in result.actions, result
            assert result.status is SessionStatus.FAILED
            assert (await observer.load(sid)).status is SessionStatus.FAILED
            checkpoint = await runtime_checkpoint_session_store(observer).load_checkpoint(sid)
            for retired in (
                "pending_tool_round",
                "pending_tool_approval",
                "approval_resolution_intent",
                "pending_user_input",
                "user_input_resolution_intent",
            ):
                assert retired not in checkpoint
            [unknown] = checkpoint["abandoned_session"]["unknown_tool_calls"]
            assert unknown["tool_call_id"] == "proposal"
            [failed] = [
                event
                for event in await observer.load_events(sid)
                if event.type is EventType.SESSION_FAILED
            ]
            assert failed.payload["error_type"] == "SessionAbandoned"
            assert failed.payload["tool_effect_outcome_unknown"] == [unknown]
            # The gate left with the round, so it cannot be resolved again.
            pending = "tool approval" if gate == "approval" else "user input"
            with pytest.raises(RuntimeError, match=f"no pending {pending}"):
                stream = (
                    app.resolve_tool_approval(ToolApprovalRequest.model_validate_json(resolution))
                    if gate == "approval"
                    else app.resolve_user_input(UserInputResponse.model_validate_json(resolution))
                )
                _ = [event async for event in stream]
            assert (await observer.load(sid)).status is SessionStatus.FAILED
            assert receipt.with_suffix(".calls").read_text().count("\n") == 1
            assert provider.requests == []
            assert (
                await app.recover_abandoned_sessions(
                    inactive_for_seconds=0, abandon_sessions_of_terminal_tasks=True
                )
                == ()
            )

    asyncio.run(scenario())


async def _resolve_gate(app, store, session_id):
    checkpoint = await store.load_checkpoint(session_id)
    if "pending_tool_approval" in checkpoint:
        from cayu.approvals import ToolApprovalDecision

        gate = checkpoint["pending_tool_approval"]
        stream = app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=gate["approval_id"],
                tool_round_id=gate["tool_round_id"],
                tool_call_id=gate["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            )
        )
    else:
        stream = app.resolve_user_input(
            UserInputResponse(
                session_id=session_id,
                input_id=checkpoint["pending_user_input"]["input_id"],
                answer="yes",
            )
        )
    _ = [event async for event in stream]


async def _gated_parent_with_paused_child(
    parent_gate, child_gate, *, child_resolved, child_tool=None
):
    """Resolve a parent's gate so its foreground child pauses on its own gate.

    With ``child_resolved`` the child's gate is resolved too, and the child's
    external call then loses its acknowledgement under that resolved gate.
    """

    from tests.core.test_foreground_child_restart import _RestartRecordingTool
    from tests.core.test_foreground_subagent_recovery import _identity, _Provider
    from tests.core.test_session_abandonment_sweep import (
        _RetryableReconciler,
        _spec,
        _UnknownEffect,
    )

    from cayu.providers import ModelStreamEvent
    from cayu.runtime.tool_effects import ToolEffectReconciliationRegistration
    from cayu.sessions.memory import InMemorySessionStore
    from cayu.sessions.queries import SessionQuery
    from cayu.sessions.requests import RunRequest
    from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
    from cayu.tools.subagents import SubagentSpec, SubagentTool

    store = InMemorySessionStore()
    parent_calls = [
        ModelStreamEvent.tool_call(
            id="delegate", name="subagent", arguments={"agent": "child", "task": "work"}
        )
    ]
    if parent_gate == "input":
        parent_calls.insert(
            0,
            ModelStreamEvent.tool_call(
                id="answer", name="ask_user", arguments={"question": "Start?"}
            ),
        )
    child_calls = [ModelStreamEvent.tool_call(id="leaf", name="record", arguments={"value": 7})]
    if child_gate == "input":
        child_calls.insert(
            0,
            ModelStreamEvent.tool_call(
                id="question", name="ask_user", arguments={"question": "Continue?"}
            ),
        )
    provider = _Provider(
        [
            [*parent_calls, ModelStreamEvent.completed()],
            [*child_calls, ModelStreamEvent.completed()],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="test"),
        tools=[
            SubagentTool(
                app,
                agents={"child": SubagentSpec(agent_name="child")},
                execution_profile_identity=_identity("abandon-gated-delegation"),
            ),
            UserInputTool(),
        ],
        tool_policy=AlwaysRequireApprovalToolPolicy(
            tools=["subagent"] if parent_gate == "approval" else []
        ),
    )
    app.register_agent(
        AgentSpec(name="child", model="test"),
        tools=[
            child_tool
            if child_tool is not None
            else _UnknownEffect()
            if child_resolved
            else _RestartRecordingTool(),
            UserInputTool(),
        ],
        tool_policy=AlwaysRequireApprovalToolPolicy(
            tools=["record"] if child_gate == "approval" else []
        ),
        tool_effect_reconcilers={
            "record": ToolEffectReconciliationRegistration(
                reconciler=_RetryableReconciler(), spec=_spec(supports_lookup=True)
            )
        }
        if child_resolved
        else None,
    )
    _ = [
        event
        async for event in app.run(
            RunRequest(
                session_id="parent", agent_name="parent", messages=[Message.text("user", "go")]
            )
        )
    ]
    await _resolve_gate(app, store, "parent")
    [child] = (await store.list_sessions(SessionQuery(parent_session_id="parent"))).sessions
    assert child.status is SessionStatus.INTERRUPTED
    if child_resolved:
        await _resolve_gate(app, store, child.id)
    assert (await store.load(child.id)).status is SessionStatus.INTERRUPTED
    assert (await store.load("parent")).status is SessionStatus.INTERRUPTED
    return app, store


@pytest.mark.parametrize("parent_gate", ["approval", "input"])
@pytest.mark.parametrize("child_gate", ["approval", "input"])
def test_abandonment_keeps_a_gated_parent_whose_child_awaits_a_human(parent_gate, child_gate):
    from cayu.sessions.pending_action_contracts import PendingActionQuery

    async def scenario():
        app, store = await _gated_parent_with_paused_child(
            parent_gate, child_gate, child_resolved=False
        )
        before = await store.load_checkpoint("parent")

        result = await app.abandon_session("parent")

        # The parent's open outcome is its child's wait for a human decision.
        assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT not in result.actions
        assert (await store.load("parent")).status is SessionStatus.INTERRUPTED
        after = await store.load_checkpoint("parent")
        assert "abandoned_session" not in after
        gate_key = "pending_tool_approval" if parent_gate == "approval" else "pending_user_input"
        assert after[gate_key] == before[gate_key]
        assert await store.query_pending_actions(PendingActionQuery(session_id="parent"))

    asyncio.run(scenario())


@pytest.mark.parametrize("parent_gate", ["approval", "input"])
@pytest.mark.parametrize("child_gate", ["approval", "input"])
def test_abandonment_ends_a_gated_parent_whose_child_lost_a_resolved_outcome(
    parent_gate, child_gate
):
    async def scenario():
        app, store = await _gated_parent_with_paused_child(
            parent_gate, child_gate, child_resolved=True
        )
        from cayu.sessions.queries import SessionQuery

        [child] = (await store.list_sessions(SessionQuery(parent_session_id="parent"))).sessions
        child_checkpoint = await store.load_checkpoint(child.id)
        # The child's gate stays in place, resolved, while its call is unknown.
        assert {"pending_tool_approval", "pending_user_input"} & set(child_checkpoint)

        result = await app.abandon_session("parent")

        # Nobody is left to decide anything, so the parent can be abandoned.
        assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT in result.actions, result
        assert (await store.load("parent")).status is SessionStatus.FAILED

    asyncio.run(scenario())


@pytest.mark.parametrize("child_gate", ["approval", "input"])
def test_abandonment_keeps_a_parent_whose_child_resumes_during_the_check(child_gate, monkeypatch):
    from tests.core.test_session_abandonment_sweep import _UnknownEffect

    from cayu.runtime._incomplete_session_recovery import IncompleteSessionRecovery
    from cayu.sessions.queries import SessionQuery

    entered, release = asyncio.Event(), asyncio.Event()

    class _Blocking(_UnknownEffect):
        async def run(self, ctx, args):
            entered.set()
            await release.wait()
            raise ConnectionError("External acknowledgement lost.")

    async def scenario():
        app, store = await _gated_parent_with_paused_child(
            "approval", child_gate, child_resolved=False, child_tool=_Blocking()
        )
        [child] = (await store.list_sessions(SessionQuery(parent_session_id="parent"))).sessions
        real = IncompleteSessionRecovery._foreground_child_owns_pending_action
        resumed: list[asyncio.Task] = []

        async def resume_child_first(self, snapshot):
            # The child's gate is resolved after the caller read its status.
            if not resumed:
                resumed.append(asyncio.create_task(_resolve_gate(app, store, child.id)))
                await asyncio.wait_for(entered.wait(), 10)
            return await real(self, snapshot)

        monkeypatch.setattr(
            IncompleteSessionRecovery, "_foreground_child_owns_pending_action", resume_child_first
        )
        try:
            result = await app.abandon_session("parent")
            assert resumed
            assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT not in result.actions
            assert (await store.load("parent")).status is SessionStatus.INTERRUPTED
            assert "abandoned_session" not in await store.load_checkpoint("parent")
        finally:
            release.set()
            await asyncio.gather(*resumed, return_exceptions=True)

    asyncio.run(scenario())
