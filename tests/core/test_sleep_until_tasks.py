"""``sleep_until`` in a session a task owns: the task parks until its wake continues it."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from tests.core.test_sleep_until import _sleep_results, _sleep_then_answer, _StableScriptedProvider

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.events import EventType
from cayu.providers.base import ModelRequest
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.sessions.records import PendingActionKind, SessionStatus
from cayu.sessions.requests import InterruptSessionRequest
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks import (
    SESSION_WAKE_TASK_TYPE,
    InMemoryTaskStore,
    SessionWakeOutcome,
    TaskCreate,
    TaskHandlerOutcome,
    TaskStatus,
    complete_managed_task,
    deliver_session_wake,
    run_task_worker,
    run_task_worker_until_idle,
    session_wake_attempts,
)
from cayu.tasks.handoff import TaskParkedWait
from cayu.tools import SleepUntilTool

_SLEEP_SECONDS = 1
_JOB = "job"


def _stores(backend: str, tmp_path, request):
    if backend == "memory":
        return InMemorySessionStore(), InMemoryTaskStore()
    if backend == "sqlite":
        return SQLiteSessionStore(tmp_path / "sessions.db"), SQLiteTaskStore(tmp_path / "tasks.db")
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    dsn = request.getfixturevalue("postgres_dsn")
    return (
        PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE),
        PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE),
    )


def _app(sessions, tasks, calls: list[ModelRequest]) -> CayuApp:
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
    app.register_provider(
        _StableScriptedProvider(
            response_factory=_sleep_then_answer({"duration_seconds": _SLEEP_SECONDS}, calls=calls)
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="sleeper", model="scripted-model"), tools=[SleepUntilTool()])
    return app


def _session_id(task_id: str) -> str:
    return f"s-{task_id}"


async def _run_job(app: CayuApp, task, worker_id: str):
    """The documented worker handler: run the session under the task."""

    if task.type != _JOB:
        raise AssertionError(f"Unexpected task {task.type}.")
    async for _ in app.run(
        RunRequest(
            agent_name="sleeper",
            session_id=_session_id(task.id),
            task_id=task.id,
            task_worker_id=worker_id,
            task_lease_expires_at=task.lease_expires_at,
            messages=[Message.text("user", "wait, then report")],
        )
    ):
        pass
    session = await app.session_store.load(_session_id(task.id))
    if session is not None and session.status is SessionStatus.INTERRUPTED:
        return TaskHandlerOutcome.SESSION_INTERRUPTED
    return None


async def _never_continued(app, task, worker_id):
    raise AssertionError(f"A generic continuation claimed parked task {task.id}.")


async def _work(app: CayuApp, *, worker_id: str = "worker", **options) -> int:
    return await run_task_worker(
        app,
        app.task_store,
        _run_job,
        worker_id=worker_id,
        poll_interval_s=0.01,
        recovered_interrupted_task_handler=_never_continued,
        **options,
    )


async def _parked_job(app: CayuApp, task_id: str):
    tasks = app.task_store
    await tasks.create_task(TaskCreate(task_id=task_id, type=_JOB))
    assert await _work(app, max_tasks=1) == 1
    job = await tasks.load_task(task_id)
    checkpoint = await app.session_store.load_checkpoint(_session_id(task_id))
    wake_id = checkpoint["pending_wake"]["continuation"]["input_id"]
    return job, wake_id


async def _wake_due(app: CayuApp, wake_id: str) -> None:
    [wake] = [
        task
        for task in await app.task_store.list_tasks()
        if task.type == SESSION_WAKE_TASK_TYPE and task.input["wake_id"] == wake_id
    ]
    delay = (wake.available_at - datetime.now(UTC)).total_seconds()
    await asyncio.sleep(max(delay, 0) + 0.05)


async def _close(app: CayuApp) -> None:
    await app.aclose()


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_task_owned_session_parks_its_task_and_the_wake_continues_it(
    backend, tmp_path, request
) -> None:
    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = _stores(backend, tmp_path, request)
        app = _app(sessions, tasks, calls)
        try:
            job, wake_id = await _parked_job(app, f"job-park-{backend}")
            # The worker is released and the task parked on its wake, still running.
            assert job.status is TaskStatus.RUNNING
            assert job.worker_id is None and job.lease_expires_at is None
            receipt = await tasks.load_interrupted_task_handoff_receipt(
                job.id, job.interrupted_handoff_id
            )
            assert receipt.request.parked == TaskParkedWait(kind="session_wake", wait_id=wake_id)
            handoffs = await sessions.query_events(
                EventQuery(
                    session_id=_session_id(job.id), event_type=EventType.TASK_INTERRUPTED_HANDOFF
                )
            )
            assert handoffs[-1].event.payload["parked_wait_id"] == wake_id

            # Before the deadline nothing is ready: an until-idle run handles nothing,
            # never offers the parked task to a generic continuation, and reports the
            # wake's time through its ordinary delayed task.
            idle = await run_task_worker_until_idle(
                app,
                tasks,
                _run_job,
                worker_id="idle-worker",
                recovered_interrupted_task_handler=_never_continued,
            )
            assert idle.handled == 0 and idle.idle
            assert idle.continuations_pending is False
            assert idle.next_available_at is not None
            assert len(calls) == 1

            await _wake_due(app, wake_id)
            idle = await run_task_worker_until_idle(
                app,
                tasks,
                _run_job,
                worker_id="idle-worker",
                recovered_interrupted_task_handler=_never_continued,
            )
            assert idle.handled == 1 and idle.idle
            # The wake continued the session itself; nothing waits on a continuation.
            assert idle.continuations_pending is False

            session = await sessions.load(_session_id(job.id))
            assert session.status is SessionStatus.COMPLETED
            [result] = _sleep_results(await sessions.load_transcript(session.id))
            assert result.is_error is False and '"status":"woke"' in result.content.replace(" ", "")
            assert len(calls) == 2
            assert (await tasks.load_task(job.id)).status is TaskStatus.COMPLETED
        finally:
            await _close(app)

    asyncio.run(scenario())


def test_a_parked_task_is_continued_by_a_worker_in_a_new_process_after_a_restart(
    tmp_path,
) -> None:
    async def scenario() -> None:
        calls: list[ModelRequest] = []
        first = _app(
            SQLiteSessionStore(tmp_path / "s.db"), SQLiteTaskStore(tmp_path / "t.db"), calls
        )
        try:
            job, _ = await _parked_job(first, "job-restart")
        finally:
            await _close(first)
        await asyncio.sleep(_SLEEP_SECONDS + 0.1)

        sessions, tasks = SQLiteSessionStore(tmp_path / "s.db"), SQLiteTaskStore(tmp_path / "t.db")
        second = _app(sessions, tasks, calls)
        try:
            idle = await run_task_worker_until_idle(
                second,
                tasks,
                _run_job,
                worker_id="after-restart",
                recovered_interrupted_task_handler=_never_continued,
            )
            assert idle.handled == 1
            assert (await sessions.load(_session_id(job.id))).status is SessionStatus.COMPLETED
            assert len(_sleep_results(await sessions.load_transcript(_session_id(job.id)))) == 1
            assert (await tasks.load_task(job.id)).status is TaskStatus.COMPLETED
            assert len(calls) == 2
        finally:
            await _close(second)

    asyncio.run(scenario())


def test_a_worker_lost_before_the_handoff_is_recovered_parked() -> None:
    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = _app(sessions, tasks, calls)
        await tasks.create_task(TaskCreate(task_id="job-lost", type=_JOB))

        async def run_then_die(app, task, worker_id):
            await _run_job(app, task, worker_id)
            raise asyncio.CancelledError

        claimed = await tasks.claim_task("doomed", lease_seconds=1)
        with pytest.raises(asyncio.CancelledError):
            await run_then_die(app, claimed, "doomed")
        checkpoint = await sessions.load_checkpoint(_session_id("job-lost"))
        wake_id = checkpoint["pending_wake"]["continuation"]["input_id"]
        # The worker never handed the task off; its lease expires with the session asleep.
        await asyncio.sleep(1.1)
        await run_task_worker_until_idle(
            app,
            tasks,
            _run_job,
            worker_id="recovering",
            recovered_interrupted_task_handler=_never_continued,
        )
        # Recovery hands the task off parked, never to the generic continuation, and
        # the wake that is already due continues it.
        handoffs = await sessions.query_events(
            EventQuery(
                session_id=_session_id("job-lost"), event_type=EventType.TASK_INTERRUPTED_HANDOFF
            )
        )
        recovered = [record.event.payload for record in handoffs]
        assert recovered and recovered[0]["handoff_status"] == "recovered"
        assert recovered[0]["parked_wait_id"] == wake_id
        assert (await tasks.load_task("job-lost")).status is TaskStatus.COMPLETED
        assert len(calls) == 2

    asyncio.run(asyncio.wait_for(scenario(), 30))


def test_a_cancelled_parked_task_stops_the_sleep_instead_of_continuing_it() -> None:
    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = _app(sessions, tasks, calls)
        job, wake_id = await _parked_job(app, "job-cancel")
        cancelled = await tasks.cancel_task(job.id)
        assert cancelled.status is TaskStatus.CANCELLED

        await _wake_due(app, wake_id)
        assert await _work(app, max_tasks=1) == 1
        session = await sessions.load(_session_id(job.id))
        assert session.status is SessionStatus.INTERRUPTED
        checkpoint = await sessions.load_checkpoint(session.id)
        assert "pending_wake" not in checkpoint
        assert _sleep_results(await sessions.load_transcript(session.id)) == []
        assert len(calls) == 1
        [wake] = [t for t in await tasks.list_tasks() if t.type == SESSION_WAKE_TASK_TYPE]
        assert wake.status is TaskStatus.COMPLETED
        assert wake.result == {"outcome": SessionWakeOutcome.CLOSED.value}

    asyncio.run(scenario())


def test_interrupting_a_parked_session_hands_its_task_to_the_generic_continuation() -> None:
    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = _app(sessions, tasks, calls)
        job, wake_id = await _parked_job(app, "job-interrupt")
        async for _ in app.interrupt_session(
            InterruptSessionRequest(session_id=_session_id(job.id), reason="stop waiting")
        ):
            pass
        # The interruption cancels the wake; the next worker's reconciliation
        # releases the task from the closed sleep to the generic continuation.
        [attempt] = await session_wake_attempts(app, wake_id)
        assert attempt.status is TaskStatus.CANCELLED
        continued: list[str] = []
        await _continue_generically(app, tasks, continued)
        assert continued == [job.id]
        assert len(calls) == 1
        assert _sleep_results(await sessions.load_transcript(_session_id(job.id))) == []

    asyncio.run(asyncio.wait_for(scenario(), 30))


def test_a_task_owned_sleep_is_a_wake_for_inspection_like_a_root_one() -> None:
    from cayu.cloud_reporting import read_cloud_activity

    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = _app(sessions, tasks, calls)
        job, wake_id = await _parked_job(app, "job-inspect")
        page = await sessions.query_pending_actions(
            PendingActionQuery(session_id=_session_id(job.id))
        )
        [record] = page.actions
        assert record.kind is PendingActionKind.WAKE and record.wake_id == wake_id
        activity = await read_cloud_activity(app)
        [run] = [run for run in activity.runs if run.session_id == _session_id(job.id)]
        [waiting] = run.waiting_for
        assert waiting.kind == "wake" and waiting.wake_at is not None

    asyncio.run(scenario())


def test_two_postgres_workers_racing_at_the_deadline_continue_the_session_once(request) -> None:
    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = _stores("postgres", None, request)
        app = _app(sessions, tasks, calls)
        peer_sessions, peer_tasks = _stores("postgres", None, request)
        peer = _app(peer_sessions, peer_tasks, calls)
        try:
            job, wake_id = await _parked_job(app, "job-race")
            await _wake_due(app, wake_id)
            stop = asyncio.Event()

            async def worker(owner: CayuApp, name: str) -> int:
                return await run_task_worker(
                    owner,
                    owner.task_store,
                    _run_job,
                    worker_id=name,
                    poll_interval_s=0.01,
                    stop=stop,
                    recovered_interrupted_task_handler=_never_continued,
                )

            async def duplicate(owner: CayuApp):
                try:
                    return await deliver_session_wake(
                        owner, session_id=_session_id(job.id), wake_id=wake_id
                    )
                except Exception as error:
                    return type(error).__name__

            racers = [
                asyncio.create_task(worker(app, "racer-a")),
                asyncio.create_task(worker(peer, "racer-b")),
            ]
            duplicates = await asyncio.gather(duplicate(app), duplicate(peer))
            for _ in range(200):
                if (await tasks.load_task(job.id)).status is TaskStatus.COMPLETED:
                    break
                await asyncio.sleep(0.05)
            stop.set()
            await asyncio.gather(*racers)

            assert (await tasks.load_task(job.id)).status is TaskStatus.COMPLETED
            transcript = await sessions.load_transcript(_session_id(job.id))
            assert len(_sleep_results(transcript)) == 1
            assert len(calls) == 2
            assert duplicates.count(SessionWakeOutcome.CONTINUED) <= 1
        finally:
            await _close(app)
            await _close(peer)

    asyncio.run(asyncio.wait_for(scenario(), 60))


def test_the_wake_waits_for_a_worker_that_has_not_handed_the_task_off_yet() -> None:
    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = _app(sessions, tasks, calls)
        await tasks.create_task(TaskCreate(task_id="job-slow", type=_JOB))
        claimed = await tasks.claim_task("slow-worker")
        await _run_job(app, claimed, "slow-worker")
        checkpoint = await sessions.load_checkpoint(_session_id("job-slow"))
        wake_id = checkpoint["pending_wake"]["continuation"]["input_id"]
        await _wake_due(app, wake_id)
        # The wake is due while the task still has its worker: it is retried, never
        # continued without the task.
        with pytest.raises(Exception, match="still holds the sleeping session"):
            await deliver_session_wake(app, session_id=_session_id("job-slow"), wake_id=wake_id)
        assert len(calls) == 1
        assert (await tasks.load_task("job-slow")).worker_id == "slow-worker"

    asyncio.run(asyncio.wait_for(scenario(), 30))


def test_an_until_idle_run_that_parks_a_task_does_not_ask_to_run_again_at_once() -> None:
    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = _app(sessions, tasks, calls)
        await tasks.create_task(TaskCreate(task_id="job-idle", type=_JOB))
        idle = await run_task_worker_until_idle(
            app,
            tasks,
            _run_job,
            worker_id="idle-worker",
            recovered_interrupted_task_handler=_never_continued,
        )
        assert idle.handled == 1 and idle.idle
        # The handled task parked: nothing is ready until its wake.
        assert idle.continuations_pending is False
        [wake] = [t for t in await tasks.list_tasks() if t.type == SESSION_WAKE_TASK_TYPE]
        assert idle.next_available_at == wake.available_at

    asyncio.run(scenario())


def test_a_wake_due_during_the_handoff_continues_once_the_worker_lets_go() -> None:
    from cayu.tasks.handoff import interrupted_task_handoff_request

    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = _app(sessions, tasks, calls)
        await tasks.create_task(TaskCreate(task_id="job-late-handoff", type=_JOB))
        claimed = await tasks.claim_task("slow-worker")
        await _run_job(app, claimed, "slow-worker")
        session_id = _session_id("job-late-handoff")
        checkpoint = await sessions.load_checkpoint(session_id)
        wake_id = checkpoint["pending_wake"]["continuation"]["input_id"]
        await _wake_due(app, wake_id)

        async def hand_off_late() -> None:
            await asyncio.sleep(0.3)
            held = await tasks.load_task("job-late-handoff")
            session = await sessions.load(session_id)
            await tasks.release_interrupted_task_worker(
                interrupted_task_handoff_request(
                    held,
                    session_run_epoch=session.run_epoch,
                    parked=TaskParkedWait(kind="session_wake", wait_id=wake_id),
                )
            )

        outcome, _ = await asyncio.gather(
            deliver_session_wake(app, session_id=session_id, wake_id=wake_id), hand_off_late()
        )
        assert outcome is SessionWakeOutcome.CONTINUED
        assert (await tasks.load_task("job-late-handoff")).status is TaskStatus.COMPLETED
        assert len(calls) == 2

    asyncio.run(asyncio.wait_for(scenario(), 30))


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_deleting_a_sleeping_task_owned_session_waits_for_its_task_to_be_cancelled(
    backend, tmp_path, request
) -> None:
    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = _stores(backend, tmp_path, request)
        app = _app(sessions, tasks, calls)
        try:
            job, wake_id = await _parked_job(app, f"job-delete-{backend}")
            session_id = _session_id(job.id)
            # Like any session whose task still runs, it is not deleted, so its
            # parked task is never left behind without a session.
            with pytest.raises(ValueError, match="Cancel the task first"):
                await app.delete_session(session_id, finished_only=False)
            assert (await sessions.load(session_id)).status is SessionStatus.INTERRUPTED
            assert (await tasks.load_task(job.id)).status is TaskStatus.RUNNING

            assert (await tasks.cancel_task(job.id)).status is TaskStatus.CANCELLED
            report = await app.delete_session(session_id, finished_only=False)
            assert report is not None and report.complete
            assert await sessions.load(session_id) is None
            [wake] = [
                task
                for task in await tasks.list_tasks()
                if task.type == SESSION_WAKE_TASK_TYPE and task.input["wake_id"] == wake_id
            ]
            assert wake.status is TaskStatus.CANCELLED
            assert len(calls) == 1
        finally:
            await _close(app)

    asyncio.run(scenario())


def test_registration_refuses_a_task_store_without_parked_handoffs() -> None:
    class OlderStore(InMemoryTaskStore):
        """Inherits the built-in store but has not declared its own handoff claims."""

    class DeclaredStore(InMemoryTaskStore):
        supports_bound_interrupted_task_handoffs = True
        supports_parked_interrupted_task_handoffs = True

    class BoundOnlyStore(InMemoryTaskStore):
        """Claims bound handoffs, but its scan predates parked ones."""

        supports_bound_interrupted_task_handoffs = True

    for store in (OlderStore(), BoundOnlyStore()):
        with pytest.raises(ValueError, match="parked interrupted-task handoffs"):
            _app(InMemorySessionStore(), store, [])
    assert _app(InMemorySessionStore(), DeclaredStore(), [])._uses_session_wakes


def test_a_woken_round_that_needs_manual_recovery_is_not_parked_on_its_wake() -> None:
    from tests.core._execution_profile_fixtures import rebind_test_invocation
    from tests.core.test_sleep_until import _Clock, _sleeping, _tool_results, _work

    from cayu.providers.base import ModelStreamEvent
    from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
    from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
    from cayu.tasks.worker import _session_parked_wait
    from cayu.tools.base import Tool, ToolEffect, ToolSpec

    class Charge(Tool):
        spec = ToolSpec(
            name="charge",
            description="Charge the card once.",
            effect=ToolEffect.IDEMPOTENT,
            input_schema={"type": "object"},
            max_terminal_payload_bytes=4096,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:sleep-until-tasks:charge",
                behavior_version="1",
                implementation_version="1",
            ),
        )

        async def run(self, arguments, ctx):
            raise SystemExit("worker died mid-charge")

    def respond(request: ModelRequest):
        if not _tool_results(request):
            return [
                ModelStreamEvent.tool_call(
                    id="call_sleep", name="sleep_until", arguments={"duration_seconds": 1}
                ),
                ModelStreamEvent.tool_call(id="call_charge", name="charge", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
        app.register_provider(_StableScriptedProvider(response_factory=respond), default=True)
        app.register_agent(
            AgentSpec(name="sleeper", model="scripted-model"), tools=[SleepUntilTool(), Charge()]
        )
        session_id = "s_manual"
        pending = await _sleeping(app, sessions, session_id)
        wake_id = pending["input_id"]
        # Asleep: a task owning this session would be parked on the wake.
        assert await _session_parked_wait(app, session_id) == TaskParkedWait(
            kind="session_wake", wait_id=wake_id
        )
        clock.advance(1.1)
        with pytest.raises(SystemExit):
            await deliver_session_wake(app, session_id=session_id, wake_id=wake_id)
        await rebind_test_invocation(sessions, session_id)
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert await _work(app) == 1
        [action] = (
            await sessions.query_pending_actions(PendingActionQuery(session_id=session_id))
        ).actions
        assert action.kind is PendingActionKind.MANUAL_RECOVERY
        # The round now waits on a person, so its task is handed off unparked.
        assert await _session_parked_wait(app, session_id) is None

    asyncio.run(scenario())


def test_a_delivery_racing_one_in_flight_settles_as_already_delivered() -> None:
    from tests.core.test_sleep_until import _tool_results

    from cayu.providers.base import ModelStreamEvent
    from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
    from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec

    entered, release = asyncio.Event(), asyncio.Event()

    class Hold(Tool):
        spec = ToolSpec(
            name="hold",
            description="Holds the woken round open.",
            effect=ToolEffect.IDEMPOTENT,
            input_schema={"type": "object"},
            max_terminal_payload_bytes=4096,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:sleep-until-tasks:hold",
                behavior_version="1",
                implementation_version="1",
            ),
        )

        async def run(self, ctx, args):
            entered.set()
            await release.wait()
            return ToolResult(content="held")

    def respond(request: ModelRequest):
        calls.append(request)
        if not _tool_results(request):
            return [
                ModelStreamEvent.tool_call(
                    id="call_sleep", name="sleep_until", arguments={"duration_seconds": 1}
                ),
                ModelStreamEvent.tool_call(id="call_hold", name="hold", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]

    calls: list[ModelRequest] = []

    async def scenario() -> None:
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        app.register_provider(_StableScriptedProvider(response_factory=respond), default=True)
        app.register_agent(
            AgentSpec(name="sleeper", model="scripted-model"), tools=[SleepUntilTool(), Hold()]
        )
        job, wake_id = await _parked_job(app, "job-in-flight")
        session_id = _session_id(job.id)
        await _wake_due(app, wake_id)
        winner = asyncio.create_task(
            deliver_session_wake(app, session_id=session_id, wake_id=wake_id)
        )
        await entered.wait()
        # The winner holds the task inside the woken round; this delivery races it.
        loser = asyncio.create_task(
            deliver_session_wake(app, session_id=session_id, wake_id=wake_id)
        )
        await asyncio.sleep(0.3)
        release.set()
        assert await winner is SessionWakeOutcome.CONTINUED
        assert await loser is SessionWakeOutcome.ALREADY_DELIVERED
        assert (await tasks.load_task(job.id)).status is TaskStatus.COMPLETED
        assert len(calls) == 2

    asyncio.run(asyncio.wait_for(scenario(), 30))


def test_until_idle_still_asks_to_continue_a_task_its_wake_handed_off_unparked() -> None:
    from cayu.tasks.session_wakes import reconcile_parked_handoffs
    from cayu.tasks.worker import _may_await_continuation

    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = _app(sessions, tasks, calls)
        job, wake_id = await _parked_job(app, "job-unparked-by-wake")
        [wake] = await session_wake_attempts(app, wake_id)
        # While its owner stays parked on the sleep, a handled wake leaves nothing.
        assert await _may_await_continuation(tasks, wake) is False
        async for _ in app.interrupt_session(
            InterruptSessionRequest(session_id=_session_id(job.id), reason="stop waiting")
        ):
            pass
        assert await reconcile_parked_handoffs(app) == 1
        # Its owner now waits, unparked, on the generic continuation.
        assert await _may_await_continuation(tasks, wake) is True

    asyncio.run(asyncio.wait_for(scenario(), 30))


def test_a_direct_task_backed_run_without_a_worker_refuses_to_sleep() -> None:
    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = _app(sessions, tasks, calls)
        task = await tasks.create_task(TaskCreate(task_id="job-direct", type=_JOB))
        # No worker owns this run, so nothing could hand its task off parked.
        async for _ in app.run(
            RunRequest(
                agent_name="sleeper",
                session_id=_session_id(task.id),
                task_id=task.id,
                messages=[Message.text("user", "wait, then report")],
            )
        ):
            pass
        session = await sessions.load(_session_id(task.id))
        assert session.status is SessionStatus.COMPLETED
        [result] = _sleep_results(await sessions.load_transcript(session.id))
        assert result.is_error is True and "did not pause" in result.content
        assert [t.type for t in await tasks.list_tasks()] == [_JOB]
        assert (await tasks.load_task(task.id)).status is TaskStatus.COMPLETED

    asyncio.run(scenario())


def test_a_woken_task_owned_session_can_sleep_again_under_its_task() -> None:
    from tests.core.test_sleep_until import _tool_results

    from cayu.messages import ToolResultPart
    from cayu.providers.base import ModelStreamEvent

    calls: list[ModelRequest] = []

    def respond(request: ModelRequest):
        calls.append(request)
        slept = [r for r in _tool_results(request) if r.tool_call_id.startswith("call_sleep")]
        if len(slept) < 2:
            return [
                ModelStreamEvent.tool_call(
                    id=f"call_sleep_{len(slept) + 1}",
                    name="sleep_until",
                    arguments={"duration_seconds": _SLEEP_SECONDS},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]

    async def scenario() -> None:
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        app.register_provider(_StableScriptedProvider(response_factory=respond), default=True)
        app.register_agent(
            AgentSpec(name="sleeper", model="scripted-model"), tools=[SleepUntilTool()]
        )
        job, first_wake = await _parked_job(app, "job-twice")
        await _wake_due(app, first_wake)
        await run_task_worker_until_idle(
            app, tasks, _run_job, worker_id="w", recovered_interrupted_task_handler=_never_continued
        )
        parked = await tasks.load_task(job.id)
        receipt = await tasks.load_interrupted_task_handoff_receipt(
            parked.id, parked.interrupted_handoff_id
        )
        second_wake = receipt.request.parked.wait_id
        assert second_wake != first_wake and parked.status is TaskStatus.RUNNING
        await _wake_due(app, second_wake)
        await run_task_worker_until_idle(
            app, tasks, _run_job, worker_id="w", recovered_interrupted_task_handler=_never_continued
        )
        assert (await tasks.load_task(job.id)).status is TaskStatus.COMPLETED
        results = [
            r.is_error
            for m in await sessions.load_transcript(_session_id(job.id))
            for r in m.content
            if isinstance(r, ToolResultPart) and r.tool_call_id.startswith("call_sleep")
        ]
        assert results == [False, False] and len(calls) == 3

    asyncio.run(asyncio.wait_for(scenario(), 30))


async def _continue_generically(app, tasks, claimed: list[str], *, max_tasks: int = 1) -> None:
    async def generic(app, task, worker_id):
        claimed.append(task.id)
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    await run_task_worker(
        app,
        tasks,
        _run_job,
        worker_id="after-stop",
        poll_interval_s=0.01,
        max_tasks=max_tasks,
        recovered_interrupted_task_handler=generic,
    )


def test_interrupting_a_sleep_with_no_wake_attempt_left_still_releases_its_task() -> None:
    from cayu.tasks.session_wakes import _cancel_session_wake_attempts

    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = _app(sessions, tasks, calls)
        job, wake_id = await _parked_job(app, "job-no-attempt")
        await _cancel_session_wake_attempts(app, wake_id)
        async for _ in app.interrupt_session(
            InterruptSessionRequest(session_id=_session_id(job.id), reason="stop waiting")
        ):
            pass
        claimed: list[str] = []
        # The next worker's wake reconciliation releases it to the generic continuation.
        await _continue_generically(app, tasks, claimed)
        assert claimed == [job.id]
        assert len(calls) == 1

    asyncio.run(asyncio.wait_for(scenario(), 30))


def test_a_parked_handoff_committed_after_its_sleep_closed_is_released(monkeypatch) -> None:
    from cayu.tasks import worker as task_worker

    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = _app(sessions, tasks, calls)
        await tasks.create_task(TaskCreate(task_id="job-late-park", type=_JOB))
        session_id = _session_id("job-late-park")
        read_before_stop = task_worker._session_parked_wait

        async def stopped_after_the_handoff_read_its_wake(app, sid):
            # The handoff reads the open wake, then an operator stops the sleep
            # before the parked handoff commits.
            parked = await read_before_stop(app, sid)
            async for _ in app.interrupt_session(
                InterruptSessionRequest(session_id=sid, reason="stop waiting")
            ):
                pass
            return parked

        monkeypatch.setattr(
            task_worker, "_session_parked_wait", stopped_after_the_handoff_read_its_wake
        )
        assert await _work(app, max_tasks=1) == 1
        monkeypatch.setattr(task_worker, "_session_parked_wait", read_before_stop)
        job = await tasks.load_task("job-late-park")
        receipt = await tasks.load_interrupted_task_handoff_receipt(
            job.id, job.interrupted_handoff_id
        )
        assert receipt.request.parked is not None
        assert "pending_wake" not in await sessions.load_checkpoint(session_id)

        claimed: list[str] = []
        await _continue_generically(app, tasks, claimed)
        assert claimed == [job.id]
        assert len(calls) == 1

    asyncio.run(asyncio.wait_for(scenario(), 30))


def test_a_parked_handoff_still_sleeping_on_its_wake_is_left_alone() -> None:
    from cayu.tasks.session_wakes import reconcile_parked_handoffs

    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = _app(sessions, tasks, calls)
        job, _ = await _parked_job(app, "job-still-asleep")
        before = await tasks.load_task(job.id)
        assert await reconcile_parked_handoffs(app) == 0
        assert await tasks.load_task(job.id) == before

    asyncio.run(scenario())


def test_an_until_idle_run_continues_a_task_its_own_reconciliation_released(monkeypatch) -> None:
    from cayu.tasks import session_wakes

    async def scenario() -> None:
        calls: list[ModelRequest] = []
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = _app(sessions, tasks, calls)
        job, _ = await _parked_job(app, "job-released-mid-run")
        async for _ in app.interrupt_session(
            InterruptSessionRequest(session_id=_session_id(job.id), reason="stop waiting")
        ):
            pass
        release = session_wakes.reconcile_parked_handoffs
        passes: list[int] = []

        async def interrupted_after_the_startup_pass(app, **options):
            # The startup pass still saw the sleep; the stop lands before the
            # worker's first reclaim pass, which releases the task.
            passes.append(1)
            return 0 if len(passes) == 1 else await release(app, **options)

        monkeypatch.setattr(
            session_wakes, "reconcile_parked_handoffs", interrupted_after_the_startup_pass
        )
        claimed: list[str] = []

        async def generic(app, task, worker_id):
            claimed.append(task.id)
            await complete_managed_task(tasks, task, worker_id, {"settled": True})

        idle = await run_task_worker_until_idle(
            app,
            tasks,
            _run_job,
            worker_id="idle",
            reclaim_every_s=0.01,
            recovered_interrupted_task_handler=generic,
        )
        assert len(passes) >= 2
        # The released task was continued in this run, not left behind as idle.
        assert claimed == [job.id]
        assert idle.idle and (await tasks.load_task(job.id)).status is TaskStatus.COMPLETED

    asyncio.run(asyncio.wait_for(scenario(), 30))
