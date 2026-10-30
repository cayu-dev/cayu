"""Recovering a task-owned turn that died before it parked on its external wait.

Whichever process recovers it, with or without the task store, the turn is never run
again without its task: it is parked when its answer was complete, and otherwise its
wait is stopped and the task handed back to its recovered-task handler, which can
resume the session.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.core.test_external_wait_hosting import WORKER, RegistrarPolicy, _reserve
from tests.core.test_external_waits_in_tasks import _JOB, _job, _session_id, _WorkerDied
from tests.core.test_external_waits_in_tasks_edges import (
    BACKENDS,
    _die,
    _expire,
    _start_and_die,
    _storeless,
    _stores,
)
from tests.core.test_sleep_until import _Clock

from cayu import AgentSpec, CayuApp, Message
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.external_waits import ExternalWaitScope
from cayu.sessions.records import SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.sessions.requests import ResumeRequest
from cayu.tasks import TaskCreate, TaskStatus, complete_managed_task, run_task_worker_until_idle
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec


class _Lookup(Tool):
    """An ordinary tool the turn calls before its final answer."""

    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="lookup",
            description="Look the render up.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            effect=ToolEffect.NONE,
        )
        super().__init__()
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls += 1
        return ToolResult(content="queued")


def _answer():
    return [
        ModelStreamEvent.text_delta("ok"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def _lookup_then_answers():
    return ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(name="lookup", arguments={}, id="call-1"),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            *(_answer() for _ in range(4)),
        ]
    )


@asynccontextmanager
async def hosted(backend, tmp_path, request, *, provider=None, tools=()):
    clock = _Clock(datetime.now(UTC))
    sessions, tasks = _stores(backend, clock, tmp_path, request)
    provider = provider or ScriptedModelProvider([_answer() for _ in range(4)])
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="root", model="model"), tools=list(tools))
    scope = ExternalWaitScope(application_scope="video-" + uuid4().hex, generation=1)
    hosting = app.host_external_waits(
        access_policy=RegistrarPolicy(), context=WORKER, scopes={scope: ("renderer",)}
    )
    try:
        yield clock, app, hosting, scope, provider
    finally:
        await app.aclose()
        await hosting.aclose()
        if backend != "memory":
            await tasks.close()
            await sessions.close()


async def _resume_for_a_person(app, **options):
    """Run the recovered-task handler, which resumes the session under the task."""

    resumed: list[str] = []

    async def resume(app, task, worker_id):
        async for _ in app.resume(
            ResumeRequest(
                session_id=task.session_id,
                messages=[Message.text("user", "Carry on without the render.")],
                task_worker_id=worker_id,
                task_handoff_id=task.interrupted_handoff_id,
                **options,
            )
        ):
            pass
        resumed.append(task.id)
        current = await app.task_store.load_task(task.id)
        if current.status is TaskStatus.RUNNING:
            await complete_managed_task(app.task_store, current, worker_id, {"resumed": True})

    async def refuse(app, task, worker_id):
        raise AssertionError(f"The application handler got {task.type}.")

    await run_task_worker_until_idle(
        app,
        app.task_store,
        refuse,
        worker_id="hosting-worker",
        recovered_interrupted_task_handler=resume,
    )
    return resumed


async def _assert_stopped_and_resumed(
    app, hosting, correlation, job, provider, *, turns, **resume_options
):
    assert await _resume_for_a_person(app, **resume_options) == [job]
    wait = await hosting.waits.inspect(correlation, context=WORKER)
    assert wait.outcome.kind == "cancelled" and not wait.pending_handoff
    session = await app.session_store.load(_session_id(job))
    assert session.status is SessionStatus.COMPLETED
    assert len(provider.requests) == turns
    assert (await app.task_store.load_task(job)).status is TaskStatus.COMPLETED


@pytest.mark.parametrize("entrance", ["abandoned", "incomplete"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_peer_without_the_task_store_leaves_a_turn_that_died_after_its_task_started(
    backend, entrance, tmp_path, request, monkeypatch
):
    from cayu.runtime._session_task_lifecycle import SessionTaskLifecycle

    async def scenario():
        async with hosted(backend, tmp_path, request) as (clock, app, hosting, scope, provider):
            job, correlation, _ = await _start_and_die(
                app, hosting, scope, monkeypatch, SessionTaskLifecycle, "start_task", after=True
            )
            await _expire(backend, clock)
            storeless, _ = _storeless(app, provider)
            try:
                if entrance == "abandoned":
                    results = await storeless.recover_abandoned_sessions(inactive_for_seconds=0)
                else:
                    results = (
                        await storeless.recover_incomplete_session(
                            IncompleteSessionRecoveryRequest(
                                session_id=_session_id(job), inactive_for_seconds=0
                            )
                        ),
                    )
            finally:
                await storeless.aclose()
            # No model stage yet; the wait itself names the task, so the turn is left alone.
            [result] = [r for r in results if r.session_id == _session_id(job)]
            assert result.status is SessionStatus.RUNNING
            await _assert_stopped_and_resumed(app, hosting, correlation, job, provider, turns=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_hosting_worker_stops_the_wait_of_a_turn_another_process_interrupted(
    backend, tmp_path, request, monkeypatch
):
    """A process that recovered the turn plainly anyway, such as an older one."""

    from cayu import external_wait_tasks
    from cayu.runtime._session_task_lifecycle import SessionTaskLifecycle

    async def scenario():
        async with hosted(backend, tmp_path, request) as (clock, app, hosting, scope, provider):
            job, correlation, _ = await _start_and_die(
                app, hosting, scope, monkeypatch, SessionTaskLifecycle, "start_task", after=True
            )
            await _expire(backend, clock)
            storeless, _ = _storeless(app, provider)
            with monkeypatch.context() as patch:

                async def unguarded(*args, **kwargs):
                    return None

                patch.setattr(external_wait_tasks, "turn_held_by_task_wait", unguarded)
                try:
                    [result] = (
                        await storeless.recover_incomplete_session(
                            IncompleteSessionRecoveryRequest(
                                session_id=_session_id(job), inactive_for_seconds=0
                            )
                        ),
                    )
                finally:
                    await storeless.aclose()
            assert result.status is SessionStatus.INTERRUPTED
            await _assert_stopped_and_resumed(app, hosting, correlation, job, provider, turns=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_recovery_never_runs_another_step_of_a_task_turn_that_died_after_a_tool_round(
    backend, tmp_path, request, monkeypatch
):
    async def scenario():
        lookup = _Lookup()
        provider = _lookup_then_answers()
        async with hosted(backend, tmp_path, request, provider=provider, tools=[lookup]) as (
            clock,
            app,
            hosting,
            scope,
            _,
        ):
            job = f"job-{uuid4().hex[:8]}"
            correlation, registration = await _reserve(hosting, scope, key=job)
            await hosting.register(registration, context=WORKER)
            await app.task_store.create_task(TaskCreate(task_id=job, type=_JOB))
            target = provider.preflight_model_target

            def dies_before_the_answer(*, model):
                # The tool round is complete; the worker dies before the next model step.
                if lookup.calls:
                    raise _WorkerDied
                return target(model=model)

            with monkeypatch.context() as patch:
                patch.setattr(provider, "preflight_model_target", dies_before_the_answer)
                await _die(app, _job(hosting, registration))
            assert lookup.calls == 1 and len(provider.requests) == 1
            await _expire(backend, clock)
            # The answer wasn't complete: no step runs without the task, the wait is
            # stopped and the person's handler resumes the session under the task.
            await _assert_stopped_and_resumed(app, hosting, correlation, job, provider, turns=2)
            assert lookup.calls == 1

    asyncio.run(scenario())


def test_a_root_turn_that_died_before_parking_is_still_recovered_to_its_wait(
    tmp_path, request, monkeypatch
):
    from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait

    async def scenario():
        async with hosted("memory", tmp_path, request) as (clock, app, hosting, scope, provider):
            _, registration = await _reserve(hosting, scope, key="root-turn")
            await hosting.register(registration, context=WORKER)
            from cayu import RunRequest

            with monkeypatch.context() as patch:

                async def dies(*args, **kwargs):
                    raise _WorkerDied

                patch.setattr(_ExternalExecutionToWait, "park", dies)
                with pytest.raises(_WorkerDied):
                    await hosting.run_to_wait(
                        RunRequest(
                            agent_name="root",
                            session_id="root-turn",
                            messages=[Message.text("user", "Submit the render.")],
                        ),
                        registration,
                        context=WORKER,
                    )
            clock.advance(3600)
            receipt = await hosting.adapter.recover_to_wait(
                registration, context=WORKER, inactive_for_seconds=0
            )
            assert receipt.wait.handoff == "pending" and receipt.wait.outcome is None
            session = await app.session_store.load("root-turn")
            assert session.status is SessionStatus.INTERRUPTED
            assert len(provider.requests) == 1

    asyncio.run(scenario())


def _job_with(hosting, registration, **options):
    """The task handler, running its session to the wait with extra request options."""

    from cayu import RunRequest
    from cayu.tasks import TaskHandlerOutcome

    async def handle(app, task, worker_id):
        await hosting.run_to_wait(
            RunRequest(
                agent_name="root",
                session_id=_session_id(task.id),
                task_id=task.id,
                task_worker_id=worker_id,
                task_lease_expires_at=task.lease_expires_at,
                messages=[Message.text("user", "Complete this turn, then wait.")],
                max_steps=3,
                **options,
            ),
            registration,
            context=WORKER,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    return handle


def _completing(kind):
    """The provider, tools and request options of a turn a tool round completes."""

    from tests.core.test_tool_completion import FinalTool

    from cayu.context import StructuredOutputSpec
    from cayu.evals.testing import scripted_structured_output
    from cayu.runtime import ToolCompletionPolicy

    if kind == "final_tool":
        first = [
            ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="question"),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
        options = {"tool_completion": ToolCompletionPolicy(tool_names=["ask_customer"])}
        tools = [FinalTool()]
    else:
        first = list(scripted_structured_output({"answer": "OK"}, id="answer"))
        options = {
            "structured_output": StructuredOutputSpec(
                json_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                strategy="tool",
            )
        }
        tools = []
    provider = ScriptedModelProvider([first, *(list(first) for _ in range(3))])
    return provider, tools, options


@pytest.mark.parametrize("kind", ["final_tool", "structured_output_tool"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_task_turn_a_tool_round_completed_is_recovered_to_its_wait(
    backend, kind, tmp_path, request, monkeypatch
):
    from tests.core.test_external_wait_hosting import _delivery
    from tests.core.test_external_waits_in_tasks import _drain
    from tests.core.test_external_waits_in_tasks_edges import _generic, _parked

    from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait

    async def scenario():
        provider, tools, options = _completing(kind)
        async with hosted(backend, tmp_path, request, provider=provider, tools=tools) as (
            clock,
            app,
            hosting,
            scope,
            _,
        ):
            job = f"job-{uuid4().hex[:8]}"
            correlation, registration = await _reserve(hosting, scope, key=job)
            await hosting.register(registration, context=WORKER)
            await app.task_store.create_task(TaskCreate(task_id=job, type=_JOB))
            handler = _job_with(hosting, registration, **options)

            async def dies(*args, **kwargs):
                raise _WorkerDied

            with monkeypatch.context() as patch:
                patch.setattr(_ExternalExecutionToWait, "park", dies)
                await _die(app, handler)
            assert len(provider.requests) == 1
            await _expire(backend, clock)
            # The tool round completed the turn: recovered to its wait, without a step.
            assert await _generic(app) == []
            assert (await _parked(app, job)).kind == "external_wait"
            assert len(provider.requests) == 1
            await hosting.deliver(_delivery(correlation), context=WORKER)
            clock.advance(60)
            await _drain(app, handler)
            assert (await app.task_store.load_task(job)).status is TaskStatus.COMPLETED
            assert len(provider.requests) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_stopping_the_wait_of_a_final_tool_turn_hands_its_task_back(
    backend, tmp_path, request, monkeypatch
):
    """Even a turn a final tool completed is handed back when its wait is stopped."""

    from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
    from cayu.runtime._incomplete_session_recovery import IncompleteSessionRecovery

    async def scenario():
        provider, tools, options = _completing("final_tool")
        async with hosted(backend, tmp_path, request, provider=provider, tools=tools) as (
            clock,
            app,
            hosting,
            scope,
            _,
        ):
            job = f"job-{uuid4().hex[:8]}"
            correlation, registration = await _reserve(hosting, scope, key=job)
            await hosting.register(registration, context=WORKER)
            await app.task_store.create_task(TaskCreate(task_id=job, type=_JOB))

            async def dies(*args, **kwargs):
                raise _WorkerDied

            with monkeypatch.context() as patch:
                patch.setattr(_ExternalExecutionToWait, "park", dies)
                await _die(app, _job_with(hosting, registration, **options))
            await _expire(backend, clock)

            async def not_completing(*args, **kwargs):
                return False

            # Its wait is stopped instead of the turn being recovered to it, while its
            # final tool's completion is still recorded.
            monkeypatch.setattr(
                IncompleteSessionRecovery, "_closed_round_completes_turn", not_completing
            )
            await _assert_stopped_and_resumed(
                app, hosting, correlation, job, provider, turns=2, max_steps=3, **options
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_recovery_never_reruns_a_tool_a_task_turn_died_in(backend, tmp_path, request):
    class _DiesInTool(_Lookup):
        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            self.calls += 1
            if self.calls == 1:
                raise _WorkerDied
            return ToolResult(content="queued")

    async def scenario():
        lookup = _DiesInTool()
        provider = _lookup_then_answers()
        async with hosted(backend, tmp_path, request, provider=provider, tools=[lookup]) as (
            clock,
            app,
            hosting,
            scope,
            _,
        ):
            job = f"job-{uuid4().hex[:8]}"
            correlation, registration = await _reserve(hosting, scope, key=job)
            await hosting.register(registration, context=WORKER)
            await app.task_store.create_task(TaskCreate(task_id=job, type=_JOB))
            await _die(app, _job(hosting, registration))
            await _expire(backend, clock)
            # Its unfinished round never runs again without the task: the wait is
            # stopped, and the person's handler resumes the session.
            await _assert_stopped_and_resumed(app, hosting, correlation, job, provider, turns=2)
            assert lookup.calls == 1

    asyncio.run(scenario())
