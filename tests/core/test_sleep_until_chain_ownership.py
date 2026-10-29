"""Who may continue a chain of subagent calls that waits on a sleeping subagent."""

from __future__ import annotations

import asyncio
import contextvars
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_sleep_until import _Clock
from tests.core.test_sleep_until_children import (
    _JOB,
    _child,
    _clocked_stores,
    _due,
    _owned_family,
    _parent_of,
    _parked_parent,
    _record_completions,
    _results,
    _until_idle,
    _waits_on_sleeping_child,
)

from cayu import Message, RunRequest
from cayu.events import EventType
from cayu.sessions._foreground_child_checkpoint import foreground_child_state_from_checkpoint
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import InterruptSessionRequest
from cayu.tasks import TaskCreate, TaskStatus, session_wakes
from cayu.tasks.continuation_claims import TaskContinuationExpectation
from cayu.tasks.handoff import new_interrupted_task_continuation_handoff_id
from cayu.tasks.session_wakes import SessionWakeOutcome


async def _model_calls(sessions, session_id: str) -> int:
    return len(
        await sessions.query_events(
            EventQuery(session_id=session_id, event_type=EventType.MODEL_STARTED)
        )
    )


def test_a_peer_sweep_during_the_attempt_leaves_the_parent_to_the_attempts_claim(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock)
        peer = _owned_family(sessions, tasks, clock)
        completions = _record_completions(tasks, monkeypatch)
        deliver = session_wakes._deliver
        active = app._session_control.has_active_tasks
        settling: set[str] = set()

        def still_settling(session_id):
            return session_id in settling or active(session_id)

        async def then_peer_sweeps(owned_app, **kwargs):
            # The child's ending is delivered while its run still settles, so that
            # delivery is put off; another process sweeps it before the attempt goes on.
            settling.add(kwargs["session_id"])
            try:
                outcome = await deliver(owned_app, **kwargs)
            finally:
                settling.clear()
            # A fresh context, as in another process: it holds none of this attempt's claims.
            await asyncio.create_task(
                peer.recover_persisted_event_side_effects(), context=contextvars.Context()
            )
            return outcome

        try:
            job, _, wake = await _parked_parent(app, "job-peer")
            _due(clock)
            monkeypatch.setattr(session_wakes, "_deliver", then_peer_sweeps)
            monkeypatch.setattr(app._session_control, "has_active_tasks", still_settling)
            assert (await _until_idle(app)).handled == 1
            assert (await sessions.load(job.session_id)).status is SessionStatus.COMPLETED
            assert completions[job.id] == [f"idle-worker:continue:{wake['input_id']}"]
        finally:
            await app.aclose()
            await peer.aclose()

    asyncio.run(scenario())


def test_an_attempt_that_died_holding_the_claim_leaves_the_parent_to_the_next_attempt(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock)
        completions = _record_completions(tasks, monkeypatch)
        try:
            job, child, wake = await _parked_parent(app, "job-dead-claim")
            _due(clock)
            # An attempt claims the parked task, continues the child, and dies.
            receipt = await tasks.load_interrupted_task_handoff_receipt(
                job.id, job.interrupted_handoff_id
            )
            claimed = await tasks.claim_interrupted_task_continuation(
                f"dead-worker:continue:{wake['input_id']}",
                handoff_id=new_interrupted_task_continuation_handoff_id(),
                task_id=job.id,
                expected=TaskContinuationExpectation(
                    task_id=job.id,
                    session_id=job.session_id,
                    session_instance_id=job.session_instance_id,
                    prior_handoff_id=job.interrupted_handoff_id,
                    operation_sha256="0" * 64,
                    parked_wait=receipt.request.parked,
                ),
                lease_seconds=1,
            )
            assert claimed.task is not None
            await session_wakes._deliver(app, session_id=child.id, wake_id=wake["input_id"])
            await session_wakes._cancel_session_wake_attempts(app, wake["input_id"])
            assert (await sessions.load(child.id)).status is SessionStatus.COMPLETED
            # A sweep holding no claim leaves the parent alone.
            await app.recover_persisted_event_side_effects()
            assert (await sessions.load(job.session_id)).status is SessionStatus.INTERRUPTED
            assert job.id not in completions

            # Once the dead claim expires, the task is parked again and a new attempt
            # continues the parent under its own claim.
            await asyncio.sleep(1.1)  # Task leases follow the store's own clock.
            await _until_idle(app)
            parked = await tasks.load_task(job.id)
            assert parked.worker_id is None
            clock.advance(2 * session_wakes._RETRY_BACKOFF.total_seconds())
            await _until_idle(app)
            assert (await sessions.load(job.session_id)).status is SessionStatus.COMPLETED
            [(worker)] = completions[job.id]
            assert worker is not None and worker.endswith(f":continue:{wake['input_id']}")
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_interrupting_a_task_owned_parent_hands_its_task_off_unparked() -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock)
        try:
            job, _, _ = await _parked_parent(app, "job-stop-parent")
            async for _ in app.interrupt_session(
                InterruptSessionRequest(session_id=job.session_id, reason="stop")
            ):
                pass
            assert await session_wakes.reconcile_parked_handoffs(app) == 1
            released = await tasks.load_task(job.id)
            receipt = await tasks.load_interrupted_task_handoff_receipt(
                job.id, released.interrupted_handoff_id
            )
            assert receipt.request.parked is None
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_cancelling_a_task_parked_on_its_childs_sleep_stops_the_chain() -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock)
        try:
            job, child, wake = await _parked_parent(app, "job-cancel")
            assert (await tasks.cancel_task(job.id)).status is TaskStatus.CANCELLED
            _due(clock)
            await _until_idle(app)
            [attempt] = await session_wakes.session_wake_attempts(app, wake["input_id"])
            assert attempt.result == {"outcome": "closed"}
            # Neither session ran another model turn after the cancel.
            assert await _model_calls(sessions, job.session_id) == 1
            assert await _model_calls(sessions, child.id) == 1
            assert (await sessions.load(child.id)).status is SessionStatus.INTERRUPTED
            assert (await sessions.load_checkpoint(child.id)).get("pending_wake") is None
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_child_of_a_task_backed_run_no_worker_owns_does_not_sleep() -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock)
        try:
            await tasks.create_task(TaskCreate(task_id="job-direct", type=_JOB))
            async for _ in app.run(
                RunRequest(
                    agent_name="parent",
                    session_id=_parent_of("job-direct"),
                    task_id="job-direct",
                    messages=[Message.text("user", "delegate")],
                )
            ):
                pass
            child = await _child(sessions, _parent_of("job-direct"))
            [refused] = _results(await sessions.load_transcript(child.id), "call_sleep")
            assert refused.is_error is True and "did not pause" in refused.content
            assert [task.id for task in await tasks.list_tasks()] == ["job-direct"]
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_child_ending_while_its_task_is_parked_leaves_nothing_claimable() -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock)
        try:
            job, child, wake = await _parked_parent(app, "job-idle-sweep")
            _due(clock)
            await session_wakes._deliver(app, session_id=child.id, wake_id=wake["input_id"])
            assert (await sessions.load(job.session_id)).status is SessionStatus.INTERRUPTED
            # The parent is the wake's to continue: no delivery stays claimable, so a
            # recovery loop has nothing to spin on.
            assert (
                await sessions.list_persisted_event_side_effect_deliveries(
                    claimable_only=True, limit=10
                )
                == []
            )
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_an_attempt_dying_between_two_hops_keeps_the_task_parked_for_the_next(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock, nested=True)
        completions = _record_completions(tasks, monkeypatch)
        try:
            await tasks.create_task(TaskCreate(task_id="job-two-hops", type=_JOB))
            await _until_idle(app, "first-worker")
            middle = await _child(sessions, _parent_of("job-two-hops"))
            leaf = await _child(sessions, middle.id)
            wake = await _waits_on_sleeping_child(sessions, middle.id, leaf.id)
            _due(clock)
            # As after a crash between the hops: the leaf and the middle ended, the
            # task's own session still waits.
            leaves = session_wakes._leaves_parent_to_its_wake

            async def only_the_top(owner, child_id):
                return child_id != leaf.id and await leaves(owner, child_id)

            monkeypatch.setattr(session_wakes, "_leaves_parent_to_its_wake", only_the_top)
            await session_wakes._deliver(app, session_id=leaf.id, wake_id=wake["input_id"])
            monkeypatch.setattr(session_wakes, "_leaves_parent_to_its_wake", leaves)
            assert (await sessions.load(middle.id)).status is SessionStatus.COMPLETED
            assert (
                await sessions.load(_parent_of("job-two-hops"))
            ).status is SessionStatus.INTERRUPTED

            assert await session_wakes.reconcile_parked_handoffs(app) == 0
            await _until_idle(app)
            assert (
                await sessions.load(_parent_of("job-two-hops"))
            ).status is SessionStatus.COMPLETED
            assert completions["job-two-hops"] == [f"idle-worker:continue:{wake['input_id']}"]
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_child_inside_an_approved_round_of_a_task_owned_parent_does_not_sleep() -> None:
    from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
    from cayu.tools.policy import AlwaysRequireApprovalToolPolicy

    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(
            sessions,
            tasks,
            clock,
            parent_policy=AlwaysRequireApprovalToolPolicy(tools=["subagent"]),
        )
        try:
            await tasks.create_task(TaskCreate(task_id="job-gate", type=_JOB))
            await _until_idle(app, "first-worker")
            requested = next(
                event
                for event in await sessions.load_events(_parent_of("job-gate"))
                if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            # The approval is settled under a claim on the handed-off task, as a
            # worker's continuation does.
            claimed = (
                await tasks.claim_interrupted_task_continuation(
                    "approving-worker",
                    handoff_id=new_interrupted_task_continuation_handoff_id(),
                    lease_seconds=300,
                )
            ).task
            assert claimed is not None and claimed.id == "job-gate"
            async for _ in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=_parent_of("job-gate"),
                    approval_id=requested.payload["approval_id"],
                    tool_call_id=requested.payload["tool_call_id"],
                    tool_round_id=requested.payload["tool_round_id"],
                    decision=ToolApprovalDecision.APPROVE,
                    task_worker_id=claimed.worker_id,
                    task_handoff_id=claimed.interrupted_handoff_id,
                )
            ):
                pass
            child = await _child(sessions, _parent_of("job-gate"))
            [refused] = _results(await sessions.load_transcript(child.id), "call_sleep")
            assert refused.is_error is True and "did not pause" in refused.content
            assert await session_wakes.session_wake_attempts(app, "none") == ()
            assert [task.id for task in await tasks.list_tasks() if task.type != _JOB] == []
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_deleting_a_sleeping_child_is_refused_while_its_parents_task_runs() -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock)
        try:
            job, child, _ = await _parked_parent(app, "job-delete-child")
            with pytest.raises(ValueError, match=f"inside task {job.id} through session"):
                await app.delete_session(
                    app.project_session_id_for_exposure(child.id), finished_only=False
                )
            assert await sessions.load(child.id) is not None
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_background_child_whose_wake_ran_out_of_retries_needs_recovery() -> None:
    from tests.core.test_sleep_until_children import _spawned_background_sleeper

    from cayu.tools.subagents import project_terminal_subagent_result

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        app, child, wake = await _spawned_background_sleeper(clock)
        sessions, tasks = app.session_store, app.task_store
        try:
            await session_wakes._cancel_session_wake_attempts(app, wake["input_id"])
            deadline = datetime.fromisoformat(wake["wake"]["deadline"])
            last = session_wakes._attempt_task(
                child.id, wake["input_id"], deadline, 2, clock() + timedelta(hours=1), deadline
            )
            await app.create_task(last)
            await session_wakes._cancel_session_wake_attempts(app, wake["input_id"])
            summary = await project_terminal_subagent_result(sessions, child.id, task_store=tasks)
            assert summary["status"] == "needs_recovery"
            assert summary["retrieval_status"] == "ready" and summary["is_error"] is True
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_cancelling_a_task_whose_child_already_woke_closes_the_attempt_cleanly() -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock)
        try:
            job, child, wake = await _parked_parent(app, "job-cancel-woken")
            _due(clock)
            # As after a crash between the hops: the child woke and ended without the
            # task's claim, and its parent still waits.
            await session_wakes._deliver(app, session_id=child.id, wake_id=wake["input_id"])
            assert (await sessions.load(child.id)).status is SessionStatus.COMPLETED
            assert (await tasks.cancel_task(job.id)).status is TaskStatus.CANCELLED
            await _until_idle(app)
            attempts = await session_wakes.session_wake_attempts(app, wake["input_id"])
            assert [(task.status, task.result) for task in attempts] == [
                (TaskStatus.COMPLETED, {"outcome": "closed"})
            ]
            assert await _model_calls(sessions, job.session_id) == 1
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_subagent_deeper_than_the_followed_chain_does_not_sleep() -> None:
    from tests.core.test_foreground_subagent_recovery import _identity
    from tests.core.test_sleep_until import _StableScriptedProvider
    from tests.core.test_sleep_until_children import _sleep, _text

    from cayu import AgentSpec, CayuApp
    from cayu.providers.base import ModelStreamEvent
    from cayu.sessions._foreground_child_checkpoint import MAX_FOREGROUND_CHAIN_DEPTH
    from cayu.tools import SleepUntilTool
    from cayu.tools.subagents import SubagentSpec, SubagentTool

    depth = MAX_FOREGROUND_CHAIN_DEPTH + 1

    def respond(request):
        level = int(request.messages[0].content[0].text.split()[-1])
        answered = any(
            hasattr(part, "tool_call_id")
            for message in request.messages
            for part in message.content
        )
        if answered:
            return _text(f"level {level} done")
        if level == depth:
            return _sleep(60)
        return [
            ModelStreamEvent.tool_call(
                id=f"spawn_{level}",
                name="subagent",
                arguments={"agent": "next", "task": f"level {level + 1}"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]

    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
        app.register_provider(_StableScriptedProvider(response_factory=respond), default=True)
        for level in range(depth):
            app.register_agent(
                AgentSpec(name=f"level-{level}", model="test"),
                tools=[
                    SubagentTool(
                        app,
                        agents={"next": SubagentSpec(agent_name=f"level-{level + 1}")},
                        execution_profile_identity=_identity(f"deep-{level}"),
                    )
                ],
            )
        app.register_agent(AgentSpec(name=f"level-{depth}", model="test"), tools=[SleepUntilTool()])
        try:
            async for _ in app.run(
                RunRequest(
                    agent_name="level-0",
                    session_id="deep-0",
                    messages=[Message.text("user", "level 0")],
                )
            ):
                pass
            session_id = "deep-0"
            for _ in range(depth):
                session_id = (await _child(sessions, session_id)).id
            [refused] = _results(await sessions.load_transcript(session_id), "call_sleep")
            assert refused.is_error is True and "did not pause" in refused.content
            assert (await sessions.load("deep-0")).status is SessionStatus.COMPLETED
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_child_inside_an_answered_round_of_a_task_owned_parent_does_not_sleep() -> None:
    from tests.core.test_foreground_subagent_recovery import _identity
    from tests.core.test_sleep_until import _StableScriptedProvider
    from tests.core.test_sleep_until_children import _by_role, _spawn

    from cayu import AgentSpec, CayuApp
    from cayu.approvals.user_input import UserInputResponse
    from cayu.providers.base import ModelStreamEvent
    from cayu.tools import SleepUntilTool, UserInputTool
    from cayu.tools.subagents import SubagentSpec, SubagentTool

    def respond(request):
        user = request.messages[0].content[0].text
        answered = any(
            hasattr(part, "tool_call_id")
            for message in request.messages
            for part in message.content
        )
        if user == "delegate" and not answered:
            # The question pauses the round first; the subagent runs once it is answered.
            return [
                ModelStreamEvent.tool_call(
                    id="call_ask", name="ask_user", arguments={"question": "Go?"}
                ),
                *_spawn()[:1],
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        return _by_role(request, seconds=60)

    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
        app.register_provider(_StableScriptedProvider(response_factory=respond), default=True)
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                UserInputTool(),
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child")},
                    execution_profile_identity=_identity("answered-round-parent"),
                ),
            ],
        )
        app.register_agent(AgentSpec(name="child", model="test"), tools=[SleepUntilTool()])
        try:
            await tasks.create_task(TaskCreate(task_id="job-answer", type=_JOB))
            await _until_idle(app, "first-worker")
            asked = next(
                event
                for event in await sessions.load_events(_parent_of("job-answer"))
                if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            claimed = (
                await tasks.claim_interrupted_task_continuation(
                    "answering-worker",
                    handoff_id=new_interrupted_task_continuation_handoff_id(),
                    lease_seconds=300,
                )
            ).task
            assert claimed is not None and claimed.id == "job-answer"
            async for _ in app.resolve_user_input(
                UserInputResponse(
                    session_id=_parent_of("job-answer"),
                    input_id=asked.payload["input_id"],
                    answer="go",
                    task_worker_id=claimed.worker_id,
                    task_handoff_id=claimed.interrupted_handoff_id,
                )
            ):
                pass
            child = await _child(sessions, _parent_of("job-answer"))
            [refused] = _results(await sessions.load_transcript(child.id), "call_sleep")
            assert refused.is_error is True and "did not pause" in refused.content
            assert [task.id for task in await tasks.list_tasks() if task.type != _JOB] == []
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_durable_child_running_after_its_wake_is_reported_running_not_a_conflict() -> None:
    from tests.core.test_foreground_subagent_recovery import _identity, _Provider
    from tests.core.test_sleep_until_children import _by_role, _delegate

    from cayu import AgentSpec, CayuApp
    from cayu.messages import ToolResultPart
    from cayu.tasks import TaskStoreDispatcher
    from cayu.tools import SleepUntilTool
    from cayu.tools.subagents import (
        SubagentSpec,
        SubagentTool,
        _durable_task_stops_result_wait,
        _wait_for_subagent_terminal,
        project_terminal_subagent_result,
    )

    release, started = asyncio.Event(), asyncio.Event()

    class _HeldAfterWake(_Provider):
        async def stream(self, request):
            woke = request.messages[0].content[0].text != "delegate" and any(
                isinstance(part, ToolResultPart)
                for message in request.messages
                for part in message.content
            )
            if woke:
                started.set()
                await release.wait()
            for event in _by_role(request):
                yield event

    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
            clock=clock,
        )
        app.register_provider(_HeldAfterWake([]), default=True)
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child", mode="durable")},
                    execution_profile_identity=_identity("durable-running-after-wake"),
                )
            ],
        )
        app.register_agent(AgentSpec(name="child", model="test"), tools=[SleepUntilTool()])
        try:
            await _delegate(app, "p_durable_running")
            assert await app.dispatcher.process_next(app, worker_id="queue") is not None
            child = await _child(sessions, "p_durable_running")
            wake = (await sessions.load_checkpoint(child.id))["pending_wake"]["continuation"]
            clock.advance(2)
            delivery = asyncio.create_task(
                session_wakes.deliver_session_wake(
                    app, session_id=child.id, wake_id=wake["input_id"]
                )
            )
            await asyncio.wait_for(started.wait(), 10)
            running = await sessions.load(child.id)
            assert running.status is SessionStatus.RUNNING

            summary = await project_terminal_subagent_result(sessions, child.id, task_store=tasks)
            assert summary["status"] == "running"
            assert summary["retrieval_status"] == "not_ready"
            assert summary["is_error"] is False
            assert not await _durable_task_stops_result_wait(sessions, tasks, running)
            waited = await _wait_for_subagent_terminal(
                sessions, running, wait=True, timeout_s=0.2, task_store=tasks
            )
            assert waited.status is SessionStatus.RUNNING

            release.set()
            await delivery
            summary = await project_terminal_subagent_result(sessions, child.id, task_store=tasks)
            assert summary["status"] == "completed" and summary["is_error"] is False
        finally:
            release.set()
            await app.aclose()

    asyncio.run(scenario())


def test_a_child_sleeping_again_parks_the_task_on_its_new_wake_despite_a_peer_sweep(
    monkeypatch,
) -> None:
    from tests.core.test_sleep_until_children import _OWNED_SLEEP_SECONDS, _by_role, _sleep

    from cayu.messages import ToolResultPart

    def twice(request):
        if request.messages[0].content[0].text == "delegate":
            return _by_role(request)
        slept = [
            part
            for message in request.messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        ]
        if len(slept) < 2:
            return _sleep(_OWNED_SLEEP_SECONDS, call_id=f"call_sleep_{len(slept)}")
        return _by_role(request)

    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock, script=twice)
        peer = _owned_family(sessions, tasks, clock, script=twice)
        completions = _record_completions(tasks, monkeypatch)
        deliver = session_wakes._deliver
        active = app._session_control.has_active_tasks
        settling: set[str] = set()

        async def then_peer_sweeps(owned_app, **kwargs):
            # The child's second pause is delivered while its run still settles, and
            # another process's sweep takes it before the attempt goes on.
            settling.add(kwargs["session_id"])
            try:
                outcome = await deliver(owned_app, **kwargs)
            finally:
                settling.clear()
            await asyncio.create_task(
                peer.recover_persisted_event_side_effects(), context=contextvars.Context()
            )
            return outcome

        try:
            job, child, first = await _parked_parent(app, "job-sleeps-twice")
            _due(clock)
            monkeypatch.setattr(session_wakes, "_deliver", then_peer_sweeps)
            monkeypatch.setattr(
                app._session_control,
                "has_active_tasks",
                lambda session_id: session_id in settling or active(session_id),
            )
            await _until_idle(app)
            monkeypatch.setattr(session_wakes, "_deliver", deliver)
            second = (await sessions.load_checkpoint(child.id))["pending_wake"]["continuation"]
            assert second["input_id"] != first["input_id"]
            parked = await tasks.load_task(job.id)
            receipt = await tasks.load_interrupted_task_handoff_receipt(
                job.id, parked.interrupted_handoff_id
            )
            assert receipt.request.parked.wait_id == second["input_id"]
            # The attempt refreshed the parent's reference before handing the task back.
            wait = (await sessions.load_checkpoint(job.session_id))["foreground_child_wait"]
            assert wait["child_action_id"] == second["input_id"]

            _due(clock)
            await _until_idle(app)
            assert (await sessions.load(job.session_id)).status is SessionStatus.COMPLETED
            assert completions[job.id] == [f"idle-worker:continue:{second['input_id']}"]
        finally:
            await app.aclose()
            await peer.aclose()

    asyncio.run(scenario())


def test_cancelling_a_nested_chain_stops_every_waiting_call() -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock, nested=True)
        try:
            await tasks.create_task(TaskCreate(task_id="job-nested-cancel", type=_JOB))
            await _until_idle(app, "first-worker")
            top = _parent_of("job-nested-cancel")
            middle = await _child(sessions, top)
            leaf = await _child(sessions, middle.id)
            await _waits_on_sleeping_child(sessions, middle.id, leaf.id)
            assert (await tasks.cancel_task("job-nested-cancel")).status is TaskStatus.CANCELLED
            _due(clock)
            await _until_idle(app)
            await app.recover_persisted_event_side_effects()
            for session_id in (top, middle.id, leaf.id):
                assert (await sessions.load(session_id)).status is SessionStatus.INTERRUPTED
                assert await _model_calls(sessions, session_id) == 1
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_cancelling_after_the_middle_ended_still_stops_the_tasks_own_call(monkeypatch) -> None:
    from cayu.sessions.pending_action_contracts import PendingActionQuery
    from cayu.sessions.records import PendingActionKind

    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock, nested=True)
        try:
            await tasks.create_task(TaskCreate(task_id="job-middle-ended", type=_JOB))
            await _until_idle(app, "first-worker")
            top = _parent_of("job-middle-ended")
            middle = await _child(sessions, top)
            leaf = await _child(sessions, middle.id)
            wake = await _waits_on_sleeping_child(sessions, middle.id, leaf.id)
            _due(clock)
            # As after an attempt died between the hops: the leaf and the middle ended,
            # and the task's own call still waits, its middle's ending acknowledged.
            leaves = session_wakes._leaves_parent_to_its_wake

            async def only_the_top(owner, child_id):
                return child_id != leaf.id and await leaves(owner, child_id)

            monkeypatch.setattr(session_wakes, "_leaves_parent_to_its_wake", only_the_top)
            await session_wakes._deliver(app, session_id=leaf.id, wake_id=wake["input_id"])
            monkeypatch.setattr(session_wakes, "_leaves_parent_to_its_wake", leaves)
            assert (await sessions.load(middle.id)).status is SessionStatus.COMPLETED

            assert (await tasks.cancel_task("job-middle-ended")).status is TaskStatus.CANCELLED
            await _until_idle(app)
            [attempt] = await session_wakes.session_wake_attempts(app, wake["input_id"])
            assert attempt.result == {"outcome": "closed"}
            assert (await sessions.load(top)).status is SessionStatus.INTERRUPTED
            assert await _model_calls(sessions, top) == 1
            # Nothing is left waiting on the chain: the task's own call was stopped. As
            # whenever a call waiting on a subagent is stopped, the subagent call's
            # unsettled outcome is left for a person to settle.
            pending = await sessions.query_pending_actions(PendingActionQuery(session_id=top))
            assert [action.kind for action in pending.actions] == [
                PendingActionKind.MANUAL_RECOVERY
            ]
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_chain_whose_task_ended_after_its_attempt_died_is_stopped_by_reconciliation(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock, nested=True)
        try:
            await tasks.create_task(TaskCreate(task_id="job-no-attempt", type=_JOB))
            await _until_idle(app, "first-worker")
            top = _parent_of("job-no-attempt")
            middle = await _child(sessions, top)
            leaf = await _child(sessions, middle.id)
            wake = await _waits_on_sleeping_child(sessions, middle.id, leaf.id)
            _due(clock)
            leaves = session_wakes._leaves_parent_to_its_wake

            async def only_the_top(owner, child_id):
                return child_id != leaf.id and await leaves(owner, child_id)

            # The attempt died after the leaf and the middle ended: none is left.
            monkeypatch.setattr(session_wakes, "_leaves_parent_to_its_wake", only_the_top)
            await session_wakes._deliver(app, session_id=leaf.id, wake_id=wake["input_id"])
            monkeypatch.setattr(session_wakes, "_leaves_parent_to_its_wake", leaves)
            await session_wakes._cancel_session_wake_attempts(app, wake["input_id"])
            assert (await tasks.cancel_task("job-no-attempt")).status is TaskStatus.CANCELLED
            assert (await sessions.load_checkpoint(top)).get("foreground_child_wait") is not None

            await session_wakes.reconcile_session_wakes(app)
            wait, _ = foreground_child_state_from_checkpoint(await sessions.load_checkpoint(top))
            assert wait is not None and await session_wakes._call_closed(app, wait)
            assert await _model_calls(sessions, top) == 1
        finally:
            await app.aclose()

    asyncio.run(scenario())


def _deep_family(clock, sessions, tasks, *, sleeper_registered: bool, leaf: str):
    """A chain of foreground subagents one deeper than the runtime follows a sleep."""

    from tests.core.test_foreground_subagent_recovery import _identity
    from tests.core.test_sleep_until import _StableScriptedProvider
    from tests.core.test_sleep_until_children import _text
    from tests.core.test_tool_round_execution_identities import _RecordingTool

    from cayu import AgentSpec, CayuApp
    from cayu.messages import ToolResultPart
    from cayu.providers.base import ModelStreamEvent
    from cayu.sessions._foreground_child_checkpoint import MAX_FOREGROUND_CHAIN_DEPTH
    from cayu.tools import SleepUntilTool, UserInputTool
    from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
    from cayu.tools.subagents import SubagentSpec, SubagentTool

    depth = MAX_FOREGROUND_CHAIN_DEPTH + 1

    def respond(request):
        text = request.messages[0].content[0].text
        level = 0 if text == "delegate" else int(text.split()[-1])
        if any(
            isinstance(part, ToolResultPart)
            for message in request.messages
            for part in message.content
        ):
            return _text(f"level {level} done")
        if level < depth:
            call = ModelStreamEvent.tool_call(
                id=f"spawn_{level}",
                name="subagent",
                arguments={"agent": "next", "task": f"level {level + 1}"},
            )
        elif leaf == "ask":
            call = ModelStreamEvent.tool_call(
                id="call_ask", name="ask_user", arguments={"question": "Which?"}
            )
        else:
            call = ModelStreamEvent.tool_call(
                id="call_record", name="record", arguments={"value": 1}
            )
        return [call, ModelStreamEvent.completed({"finish_reason": "tool_calls"})]

    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
    app.register_provider(_StableScriptedProvider(response_factory=respond), default=True)
    names = ["parent", *(f"level-{level}" for level in range(1, depth + 1))]
    for level, name in enumerate(names[:-1]):
        app.register_agent(
            AgentSpec(name=name, model="test"),
            tools=[
                *([SleepUntilTool()] if sleeper_registered and level == 0 else []),
                SubagentTool(
                    app,
                    agents={"next": SubagentSpec(agent_name=names[level + 1])},
                    execution_profile_identity=_identity(f"deep-person-{level}"),
                ),
            ],
        )
    if leaf == "ask":
        app.register_agent(AgentSpec(name=names[-1], model="test"), tools=[UserInputTool()])
    else:
        app.register_agent(
            AgentSpec(name=names[-1], model="test"),
            tools=[_RecordingTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(tools=["record"]),
        )
    return app, depth


async def _deepest(sessions, top: str, depth: int):
    chain = [top]
    for _ in range(depth):
        chain.append((await _child(sessions, chain[-1])).id)
    return chain


async def _settle_leaf(app, sessions, leaf_id: str, leaf: str) -> None:
    from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
    from cayu.approvals.user_input import UserInputResponse

    events = await sessions.load_events(leaf_id)
    if leaf == "ask":
        asked = next(e for e in events if e.type is EventType.SESSION_AWAITING_USER_INPUT)
        stream = app.resolve_user_input(
            UserInputResponse(
                session_id=app.project_session_id_for_exposure(leaf_id),
                input_id=asked.payload["input_id"],
                answer="this",
            )
        )
    else:
        requested = next(e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
        stream = app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=app.project_session_id_for_exposure(leaf_id),
                approval_id=requested.payload["approval_id"],
                tool_call_id=requested.payload["tool_call_id"],
                tool_round_id=requested.payload["tool_round_id"],
                decision=ToolApprovalDecision.APPROVE,
            )
        )
    async for _ in stream:
        pass


@pytest.mark.parametrize(
    ("leaf", "sleeper_registered"), [("ask", False), ("ask", True), ("approve", True)]
)
def test_a_person_settling_the_leaf_of_a_chain_deeper_than_sleeps_go_continues_it(
    leaf, sleeper_registered
) -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app, depth = _deep_family(
            clock, sessions, tasks, sleeper_registered=sleeper_registered, leaf=leaf
        )
        try:
            async for _ in app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="deep-person",
                    messages=[Message.text("user", "delegate")],
                )
            ):
                pass
            chain = await _deepest(sessions, "deep-person", depth)
            await _settle_leaf(app, sessions, chain[-1], leaf)
            for _ in range(100):
                if (await sessions.load("deep-person")).status is SessionStatus.COMPLETED:
                    break
                await asyncio.sleep(0.05)
            for session_id in chain:
                assert (await sessions.load(session_id)).status is SessionStatus.COMPLETED
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("sleeper_registered", [False, True])
def test_a_worker_hands_off_a_deep_chain_waiting_on_a_person_unparked(
    sleeper_registered, monkeypatch
) -> None:
    from cayu.tasks import TaskHandlerOutcome

    if not sleeper_registered:
        # Without a sleep tool, handing off never looks for a sleep at all.
        async def no_sleep_lookup(*_args):
            raise AssertionError("A wake-less application looked for a sleeping chain.")

        monkeypatch.setattr(session_wakes, "_sleeping_wait", no_sleep_lookup)

    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app, depth = _deep_family(
            clock, sessions, tasks, sleeper_registered=sleeper_registered, leaf="ask"
        )

        async def run_top(owned_app, task, worker_id):
            async for _ in owned_app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="deep-task",
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                    messages=[Message.text("user", "delegate")],
                )
            ):
                pass
            if (
                await owned_app.session_store.load("deep-task")
            ).status is SessionStatus.INTERRUPTED:
                return TaskHandlerOutcome.SESSION_INTERRUPTED
            return None

        try:
            await tasks.create_task(TaskCreate(task_id="job-deep", type=_JOB))
            from cayu.tasks import run_task_worker

            assert (
                await run_task_worker(
                    app, tasks, run_top, worker_id="deep-worker", max_tasks=1, poll_interval_s=0.01
                )
                == 1
            )
            job = await tasks.load_task("job-deep")
            receipt = await tasks.load_interrupted_task_handoff_receipt(
                job.id, job.interrupted_handoff_id
            )
            assert job.worker_id is None and receipt.request.parked is None
            chain = await _deepest(sessions, "deep-task", depth)
            await _settle_leaf(app, sessions, chain[-1], "ask")
            for _ in range(100):
                if (await sessions.load("deep-task")).status is SessionStatus.COMPLETED:
                    break
                await asyncio.sleep(0.05)
            assert (await sessions.load("deep-task")).status is SessionStatus.COMPLETED
            assert (await tasks.load_task("job-deep")).status is TaskStatus.COMPLETED
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_woken_child_whose_subagent_asks_a_person_is_not_left_parked_on_its_wake() -> None:
    from tests.core.test_foreground_subagent_recovery import _identity
    from tests.core.test_sleep_until import _StableScriptedProvider
    from tests.core.test_sleep_until_children import (
        _OWNED_SLEEP_SECONDS,
        _results,
        _sleep,
        _spawn,
        _text,
    )

    from cayu import AgentSpec, CayuApp
    from cayu.approvals.user_input import UserInputResponse
    from cayu.messages import ToolResultPart
    from cayu.providers.base import ModelStreamEvent
    from cayu.tools import SleepUntilTool, UserInputTool
    from cayu.tools.subagents import SubagentSpec, SubagentTool

    def respond(request):
        user = request.messages[0].content[0].text
        if any(
            isinstance(part, ToolResultPart)
            for message in request.messages
            for part in message.content
        ):
            return _text(f"{user} done")
        if user == "delegate":
            return _spawn()
        if user == "wait":
            # The sleep pauses the round; its subagent runs once the child wakes.
            return [
                *_sleep(_OWNED_SLEEP_SECONDS)[:1],
                ModelStreamEvent.tool_call(
                    id="spawn_asker", name="subagent", arguments={"agent": "asker", "task": "ask"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        return [
            ModelStreamEvent.tool_call(
                id="call_ask", name="ask_user", arguments={"question": "Which?"}
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]

    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
        app.register_provider(_StableScriptedProvider(response_factory=respond), default=True)
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child")},
                    execution_profile_identity=_identity("woken-asks-parent"),
                )
            ],
        )
        app.register_agent(
            AgentSpec(name="child", model="test"),
            tools=[
                SleepUntilTool(),
                SubagentTool(
                    app,
                    agents={"asker": SubagentSpec(agent_name="asker")},
                    execution_profile_identity=_identity("woken-asks-child"),
                ),
            ],
        )
        app.register_agent(AgentSpec(name="asker", model="test"), tools=[UserInputTool()])
        try:
            job, child, wake = await _parked_parent(app, "job-woken-asks")
            _due(clock)
            assert (
                await session_wakes.deliver_session_wake(
                    app, session_id=child.id, wake_id=wake["input_id"]
                )
                is SessionWakeOutcome.CONTINUED
            )
            asker = await _child(sessions, child.id)
            # The child's wake was delivered; it now waits on a person through its
            # subagent, so the task is not parked on that wake.
            handed = await tasks.load_task(job.id)
            receipt = await tasks.load_interrupted_task_handoff_receipt(
                job.id, handed.interrupted_handoff_id
            )
            assert receipt.request.parked is None
            # A person is waited on, so the parent is listed for one, whatever its
            # reference last recorded.
            from cayu.server.routes import _delegation_ends_in_wake
            from cayu.sessions.pending_action_contracts import PendingActionQuery

            [waiting] = (
                await sessions.query_pending_actions(PendingActionQuery(session_id=job.session_id))
            ).actions
            assert not await _delegation_ends_in_wake(sessions, waiting)

            asked = next(
                event
                for event in await sessions.load_events(asker.id)
                if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            async for _ in app.resolve_user_input(
                UserInputResponse(
                    session_id=app.project_session_id_for_exposure(asker.id),
                    input_id=asked.payload["input_id"],
                    answer="this",
                )
            ):
                pass
            for _ in range(100):
                if (await sessions.load(job.session_id)).status is SessionStatus.COMPLETED:
                    break
                await asyncio.sleep(0.05)
            for session_id in (asker.id, child.id, job.session_id):
                assert (await sessions.load(session_id)).status is SessionStatus.COMPLETED
            assert len(_results(await sessions.load_transcript(job.session_id), "spawn")) == 1
            assert (await tasks.load_task(job.id)).status is TaskStatus.COMPLETED
        finally:
            await app.aclose()

    asyncio.run(scenario())
