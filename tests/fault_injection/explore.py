"""Scenarios and the exhaustive exploration over injection points."""

from __future__ import annotations

import asyncio
import faulthandler
import os
import sys
import threading
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from typing import Any, TypeVar

from tests.fault_injection.harness import (
    APPROVAL_PROMPT,
    TOOLS_PROMPT,
    Boundary,
    BoundaryTool,
    FaultKind,
    Injector,
    ManualClock,
    ScriptedProvider,
    SimulatedCrash,
    runtime_frames,
    session_store,
    task_store,
)
from tests.fault_injection.invariants import (
    TERMINAL_TASK_STATUSES,
    Invariant,
    Stage,
    store_violations,
    task_group_violations,
    task_violations,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions._checkpoint_publication import _initial_transcript_pending_interaction_id
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.tasks import (
    TaskCreate,
    TaskGraphCreate,
    TaskGraphNode,
    TaskGroupCreate,
    TaskGroupPolicy,
    TaskGroupQuiescencePolicy,
)
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import Task
from cayu.tasks.worker import DurableWorkerMetrics, run_task_worker
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult

SESSION_ID = "fault-injection-session"
RECOVERY_SESSION_ID = "fault-injection-recovery-session"
TASK_ID = "fault-injection-task"
GROUP_ID = "fault-injection-group"
FINALIZER_TASK_ID = "fault-injection-finalizer"
TASK_TYPE = "respond"
AGENT_NAME = "fault-injection-agent"
WORKER_ID = "fault-injection-worker-a"
RECOVERY_WORKER_ID = "fault-injection-worker-b"
CONTINUE_PROMPT = "fault-injection: continue"
# Bound a hang without slowing healthy cases, which settle in milliseconds.
_PHASE_TIMEOUT_S = 10.0
_SHUTDOWN_TIMEOUT_S = 10.0
_CHECK_TIMEOUT_S = 10.0
_LIVENESS_TIMEOUT_S = 20.0
# Every lease the dead or cancelled owner held has expired by then.
_RECOVERY_CLOCK_ADVANCE = timedelta(days=1)
_LIVENESS_ATTEMPTS = 4
_WORKER_POLL_S = 0.005
# The recovery worker has finished once it polls this many times in a row
# without claiming, running or changing the task. Counting polls rather than
# seconds keeps the check independent of the host's speed.
_RECOVERY_IDLE_POLLS = 40
# Only a worker that keeps changing the task without settling it reaches this.
_TASK_RECOVERY_LIMIT_S = 60.0
# Every await of a case is bounded well below this; only a synchronous hang
# reaches it, and the watchdog then names the point before ending the process.
_POINT_WATCHDOG_S = 600.0


class _ReviewPolicy(ToolPolicy):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:fault-injection:review-policy",
            behavior_version="1",
            implementation_version="1",
        )

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name == "reviewed":
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason="Fault-injection review.",
            )
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


@dataclass
class World:
    """The durable stores and the apps that share them across a simulated restart."""

    injector: Injector
    with_tasks: bool
    clock: ManualClock = field(default_factory=ManualClock)

    def __post_init__(self) -> None:
        self.store = session_store(self.injector, self.clock)
        self.tasks = task_store(self.injector, self.clock) if self.with_tasks else None

    def app(self) -> CayuApp:
        app = CayuApp(session_store=self.store, task_store=self.tasks, enable_logging=False)
        app.register_provider(ScriptedProvider(self.injector), default=True)
        app.register_agent(
            AgentSpec(name=AGENT_NAME, model="fault-injection-model"),
            tools=[BoundaryTool("echo", self.injector), BoundaryTool("reviewed", self.injector)],
            tool_policy=_ReviewPolicy(),
        )
        return app


def run_turn(app: CayuApp, prompt: str) -> AsyncIterator[Event]:
    return app.run(
        RunRequest(
            agent_name=AGENT_NAME,
            session_id=SESSION_ID,
            messages=[Message.text("user", prompt)],
        )
    )


def run_task_turn(app: CayuApp, task: Task, worker_id: str, session_id: str):
    return app.run(
        RunRequest(
            agent_name=AGENT_NAME,
            session_id=session_id,
            task_id=task.id,
            task_worker_id=worker_id,
            task_lease_expires_at=task.lease_expires_at,
            messages=[Message.text("user", TOOLS_PROMPT)],
        )
    )


def resume_turn(app: CayuApp, prompt: str) -> AsyncIterator[Event]:
    return app.resume(ResumeRequest(session_id=SESSION_ID, messages=[Message.text("user", prompt)]))


def approve(app: CayuApp, approval: dict[str, Any]) -> AsyncIterator[Event]:
    return app.resolve_tool_approval(
        ToolApprovalRequest(
            session_id=SESSION_ID,
            approval_id=approval["approval_id"],
            tool_round_id=approval["tool_round_id"],
            tool_call_id=approval["tool_call_id"],
            decision=ToolApprovalDecision.APPROVE,
        )
    )


def settle_unknown_approval(app: CayuApp, approval: dict[str, Any]) -> AsyncIterator[Event]:
    return app.recover_tool_approval(
        ToolApprovalRecoveryRequest(
            session_id=SESSION_ID,
            approval_id=approval["approval_id"],
            tool_round_id=approval["tool_round_id"],
            tool_call_id=approval["tool_call_id"],
            outcome=ToolApprovalRecoveryOutcome.FAILED,
            message="fault-injection: the call's outcome is unknown",
        )
    )


def pending_approval(events: list[Event]) -> dict[str, Any]:
    approvals = [
        event.payload["approval"]
        for event in events
        if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    ]
    assert approvals, "The approval scenario's setup did not request approval."
    return approvals[-1]


Consume = Callable[[AsyncIterator[Event]], Awaitable[None]]
Phase = Callable[["World", CayuApp, list[Event], Consume], Awaitable[object]]


@dataclass(frozen=True)
class Scenario:
    """A setup that runs unarmed, then the phase whose boundaries are explored.

    ``phase`` hands every Runtime event stream it opens to ``consume``, which is
    where a ``stream-close`` fault closes it.
    """

    name: str
    phase: Phase
    setup: Callable[[World, CayuApp], Awaitable[list[Event]]] | None = None
    with_tasks: bool = False


def _streamed(open_stream: Callable[[CayuApp, list[Event]], AsyncIterator[Event]]) -> Phase:
    async def phase(world: World, app: CayuApp, setup: list[Event], consume: Consume) -> None:
        await consume(open_stream(app, setup))

    return phase


def _turn(prompt: str) -> Callable[[World, CayuApp], Awaitable[list[Event]]]:
    async def setup(world: World, app: CayuApp) -> list[Event]:
        return [event async for event in run_turn(app, prompt)]

    return setup


def _member() -> TaskCreate:
    return TaskCreate(task_id=TASK_ID, type=TASK_TYPE, assigned_agent_name=AGENT_NAME)


async def _create_task(world: World, app: CayuApp) -> list[Event]:
    assert world.tasks is not None
    await world.tasks.create_task(_member())
    return []


async def _create_task_group(world: World, app: CayuApp) -> list[Event]:
    """One member under a quiescent group, so its execution leaves an obligation."""

    await app.create_task_group(
        TaskGroupCreate(
            group_id=GROUP_ID,
            graph=TaskGraphCreate(
                graph_id=f"{GROUP_ID}-graph",
                nodes=(
                    TaskGraphNode(task=_member()),
                    TaskGraphNode(task=TaskCreate(task_id=FINALIZER_TASK_ID, type="finalize")),
                ),
            ),
            member_task_ids=(TASK_ID,),
            policy=TaskGroupPolicy(kind="all"),
            quiescence=TaskGroupQuiescencePolicy(timeout_seconds=60),
            finalizer_task_id=FINALIZER_TASK_ID,
        )
    )
    return []


async def _work_task(world: World, app: CayuApp, setup: list[Event], consume: Consume) -> None:
    """One worker claims the task and runs its session through a parallel tool round."""

    assert world.tasks is not None

    async def handler(app: CayuApp, task: Task, worker_id: str) -> None:
        await consume(run_task_turn(app, task, worker_id, SESSION_ID))

    await run_task_worker(
        app,
        world.tasks,
        handler,
        worker_id=WORKER_ID,
        query=TaskQuery(type=TASK_TYPE),
        max_tasks=1,
        poll_interval_s=_WORKER_POLL_S,
        reclaim=False,
        recover_interrupted_handoffs=False,
    )


# Runtime paths these scenarios do not reach yet. Each needs a further scenario
# or setup: a completed but unpromoted model stage before resume, a queued
# on_idle follow-up, model failover, a task-backed resume, manual recovery
# streams, runner cleanup artifacts, environment factories and differing
# execution profiles. Generation fencing needs two live owners racing for one
# lease, and durable-index shifts a model switch; both are out of reach of one
# fault in one owner.
SCENARIOS: dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in (
        Scenario("run", _streamed(lambda app, _: run_turn(app, "fault-injection: hello"))),
        Scenario("tool-round", _streamed(lambda app, _: run_turn(app, TOOLS_PROMPT))),
        Scenario(
            "resume",
            _streamed(lambda app, _: resume_turn(app, TOOLS_PROMPT)),
            setup=_turn("fault-injection: hello"),
        ),
        Scenario(
            "approval",
            _streamed(lambda app, setup: approve(app, pending_approval(setup))),
            setup=_turn(APPROVAL_PROMPT),
        ),
        Scenario("task", _work_task, setup=_create_task, with_tasks=True),
        Scenario("task-group", _work_task, setup=_create_task_group, with_tasks=True),
    )
}


@dataclass(frozen=True)
class Violation:
    scenario: str
    fault: FaultKind | None
    point: str
    invariant: str
    stage: str
    detail: str
    outcome: str

    def describe(self) -> str:
        fault = self.fault.value if self.fault else "no fault"
        return (
            f"{self.scenario} {fault} at {self.point} [{self.stage}] "
            f"{self.invariant}: {self.detail} (phase outcome: {self.outcome})"
        )


@dataclass
class CaseResult:
    trace: list[Boundary]
    events: list[str]
    reached: bool
    violations: list[Violation]


async def _drive(
    phase: Callable[[Consume], Awaitable[object]],
    injector: Injector,
) -> tuple[list[Event], BaseException | None, bool]:
    """Run one phase in its own task; return its events, error and whether it hung."""

    events: list[Event] = []
    closing = injector.kind is FaultKind.STREAM_CLOSE

    async def consume(stream: AsyncIterator[Event]) -> None:
        async for event in stream:
            events.append(event)
            if closing and injector.fired is None:
                label = injector.next_label(f"event:{event.type}")
                if label == injector.target:
                    injector.fired = label
                    await stream.aclose()  # type: ignore[attr-defined]
                    return

    task = asyncio.create_task(phase(consume))
    injector.driver = task
    injector.armed = True
    try:
        done, _ = await asyncio.wait({task}, timeout=_PHASE_TIMEOUT_S)
    finally:
        injector.armed = False
    if not done:
        task.cancel()
        await asyncio.wait({task}, timeout=_SHUTDOWN_TIMEOUT_S)
        return events, None, True
    if task.cancelled():
        return events, asyncio.CancelledError(), False
    return events, task.exception(), False


def _describe(error: BaseException | None) -> str:
    if error is None:
        return "returned"
    if isinstance(error, BaseExceptionGroup):
        return f"{type(error).__name__}({', '.join(type(e).__name__ for e in error.exceptions)})"
    return f"{type(error).__name__}: {str(error)[:160]}"


def _origin(error: BaseException) -> str:
    frames = runtime_frames(traceback.extract_tb(error.__traceback__))
    return f" at {frames[-1]}" if frames else ""


_T = TypeVar("_T")


async def _bounded(awaitable: Awaitable[_T], timeout: float) -> tuple[bool, _T | None]:
    """Await in a task of its own; ``(False, None)`` when it does not finish in time.

    ``asyncio.wait_for`` waits for the cancelled awaitable to finish, so one that
    swallows cancellation would block it forever. This cancels and abandons it.
    """

    task = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if not done:
        task.cancel()
        return False, None
    return True, task.result()


async def _cancel_leftover_tasks(before: set[asyncio.Task[Any]]) -> int:
    """Cancel the tasks started since ``before``; return how many refuse to stop."""

    current = asyncio.current_task()
    leftovers = [task for task in asyncio.all_tasks() - before if task is not current]
    for task in leftovers:
        task.cancel()
    if not leftovers:
        return 0
    _, pending = await asyncio.wait(leftovers, timeout=_SHUTDOWN_TIMEOUT_S)
    return len(pending)


async def _close(app: CayuApp, *, injected_group: bool = False) -> tuple[bool, str | None]:
    """Shut one app down; return whether shutdown finished, and why it failed.

    ``aclose()`` reports a failed or unfinished shutdown in its outcome rather
    than by raising, so an outcome that is not settled is a failure too. The one
    exception is the injected cleanup failure itself: when a grouped cancellation
    fired inside background work, shutdown truthfully reports that work's
    ``BaseExceptionGroup``, and only steps that failed that way are excused.
    """

    try:
        finished, outcome = await _bounded(app.aclose(), _SHUTDOWN_TIMEOUT_S)
    except SimulatedCrash:
        return True, None
    if not finished or outcome is None:
        return False, "app.aclose() did not finish"
    unexpected = [
        step
        for step in outcome.unsettled
        if not (
            injected_group
            and step.status == "failed"
            and step.failure_type == BaseExceptionGroup.__name__
        )
    ]
    if unexpected or (not outcome.settled and not outcome.unsettled):
        return True, f"app.aclose() did not settle: {outcome.summary()}"
    return True, None


async def _continue_to_completion(app: CayuApp, world: World) -> str | None:
    """Drive the recovered session to a completed turn; return why it could not.

    A pending approval is approved, or, once the Runtime has asked for manual
    recovery of a call that started without a result, settled as failed. A
    session without one is resumed with a new user message.
    """

    manual_recovery = False
    for _ in range(_LIVENESS_ATTEMPTS):
        session = await world.store.load(SESSION_ID)
        checkpoint = await world.store.load_checkpoint(SESSION_ID) or {}
        approval = checkpoint.get("pending_tool_approval")
        try:
            if approval is None:
                stream = resume_turn(app, CONTINUE_PROMPT)
            elif manual_recovery:
                stream = settle_unknown_approval(app, approval)
            else:
                stream = approve(app, approval)
            events = [event async for event in stream]
        except Exception as error:  # Reported as the violation detail.
            return f"status={session.status.value}: {_describe(error)}{_origin(error)}"
        if events and events[-1].type == EventType.SESSION_COMPLETED:
            return None
        manual_recovery = any(
            event.payload.get("error_type") == "ToolApprovalManualRecoveryRequired"
            for event in events
        )
    session = await world.store.load(SESSION_ID)
    return f"status={session.status.value} after {_LIVENESS_ATTEMPTS} attempts"


async def _settle_task_with_another_worker(app: CayuApp, world: World) -> str | None:
    """Let a second worker reclaim and settle the task; return why it could not.

    It runs a reclaimed task in a new session, and continues a task whose
    session was interrupted by resuming that session under the task.
    """

    assert world.tasks is not None
    stop = asyncio.Event()

    async def handler(app: CayuApp, task: Task, worker_id: str) -> None:
        async for _ in run_task_turn(app, task, worker_id, RECOVERY_SESSION_ID):
            pass

    async def continue_interrupted(app: CayuApp, task: Task, worker_id: str) -> None:
        assert task.session_id is not None
        async for _ in app.resume(
            ResumeRequest(
                session_id=task.session_id,
                messages=[Message.text("user", CONTINUE_PROMPT)],
                task_worker_id=worker_id,
                task_handoff_id=task.interrupted_handoff_id,
            )
        ):
            pass

    metrics = DurableWorkerMetrics()
    worker = asyncio.create_task(
        run_task_worker(
            app,
            world.tasks,
            handler,
            worker_id=RECOVERY_WORKER_ID,
            query=TaskQuery(type=TASK_TYPE),
            poll_interval_s=_WORKER_POLL_S,
            minimum_idle_delay_s=_WORKER_POLL_S,
            maximum_idle_delay_s=_WORKER_POLL_S,
            metrics=metrics,
            reclaim_every_s=_WORKER_POLL_S,
            reconcile_abandoned_after_s=0,
            recovered_interrupted_task_handler=continue_interrupted,
            stop=stop,
        )
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TASK_RECOVERY_LIMIT_S
    progress: object = None
    idle_from = 0
    while not worker.done():
        task = await world.tasks.load_task(TASK_ID)
        if task is not None and task.status in TERMINAL_TASK_STATUSES:
            break
        snapshot = metrics.snapshot()
        signature = (snapshot.successful_claims, snapshot.active_handlers, task)
        if signature != progress:
            progress, idle_from = signature, snapshot.empty_claims
        elif snapshot.empty_claims - idle_from >= _RECOVERY_IDLE_POLLS:
            break
        if loop.time() > deadline:
            stop.set()
            return f"the recovery worker kept working for {_TASK_RECOVERY_LIMIT_S:.0f}s"
        await asyncio.sleep(_WORKER_POLL_S)
    stop.set()
    done, _ = await asyncio.wait({worker}, timeout=_SHUTDOWN_TIMEOUT_S)
    if not done:
        worker.cancel()
        return "the recovery worker does not stop"
    if not worker.cancelled() and (error := worker.exception()) is not None:
        return f"the recovery worker raised {_describe(error)}{_origin(error)}"
    task = await world.tasks.load_task(TASK_ID)
    if task is None or task.status not in TERMINAL_TASK_STATUSES:
        return f"task status={task.status.value if task else None} after recovery"
    return None


async def run_case(scenario: Scenario, fault: FaultKind | None, point: str | None) -> CaseResult:
    """Run one scenario with at most one fault and check every invariant.

    Every stage is bounded. A stage that does not finish is recorded as a
    ``shutdown-or-recovery-hangs`` violation of this point, and the case stops.
    """

    injector = Injector(kind=fault, target=point)
    world = World(injector, scenario.with_tasks)
    events: list[Event] = []
    outcome = "not started"
    violations: list[Violation] = []

    def record(stage: str, found: list[tuple[str, str]]) -> None:
        violations.extend(
            Violation(
                scenario.name,
                fault,
                point or "-",
                invariant,
                stage,
                detail,
                outcome,
            )
            for invariant, detail in found
        )

    class _StageHung(Exception):
        pass

    async def bounded(stage: str, awaitable: Awaitable[_T], timeout: float) -> _T | None:
        finished, result = await _bounded(awaitable, timeout)
        if not finished:
            record(stage, [(Invariant.SHUTDOWN_HANG, f"{stage} did not finish in {timeout:.0f}s")])
            raise _StageHung
        return result

    async def check(stage: str, recovery: CayuApp | None = None) -> None:
        record(stage, await bounded(stage, store_violations(world.store), _CHECK_TIMEOUT_S) or [])
        if world.tasks is not None:
            found = await bounded(
                stage, task_violations(world.tasks, world.store), _CHECK_TIMEOUT_S
            )
            record(stage, found or [])
            if recovery is not None:
                found = await bounded(
                    stage, task_group_violations(recovery, GROUP_ID), _CHECK_TIMEOUT_S
                )
                record(stage, found or [])

    app = world.app()
    recovery: CayuApp | None = None
    try:
        setup_events = (
            []
            if scenario.setup is None
            else await bounded(Stage.SETUP, scenario.setup(world, app), _PHASE_TIMEOUT_S) or []
        )
        before = set(asyncio.all_tasks())
        events, error, hung = await _drive(
            lambda consume: scenario.phase(world, app, setup_events, consume),
            injector,
        )
        outcome = "hung" if hung else _describe(error)
        if hung:
            record(Stage.PHASE, [(Invariant.SHUTDOWN_HANG, "the phase did not finish")])
        if fault is FaultKind.CRASH:
            if refusing := await _cancel_leftover_tasks(before):
                detail = f"{refusing} of the dead app's tasks ignore cancellation"
                record(Stage.CRASH, [(Invariant.SHUTDOWN_HANG, detail)])
            injector.dead = False
        else:
            # The process is still alive: let its background settlement run first.
            for _ in range(3):
                await asyncio.sleep(0)
            await check(Stage.LIVE)
            finished, failure = await _close(
                app,
                injected_group=fault is FaultKind.GROUPED_CANCEL and injector.fired is not None,
            )
            if failure is not None:
                record(Stage.SHUTDOWN, [(Invariant.SHUTDOWN_HANG, failure)])
            if finished:
                await check(Stage.AFTER_SHUTDOWN)
        world.clock.advance(_RECOVERY_CLOCK_ADVANCE)
        recovery = world.app()
        try:
            finished, _ = await _bounded(
                recovery.recover_abandoned_sessions(inactive_for_seconds=0), _SHUTDOWN_TIMEOUT_S
            )
        except Exception as recovery_error:
            detail = f"recovery raised {_describe(recovery_error)}{_origin(recovery_error)}"
            record(Stage.RECOVERY, [(Invariant.NOT_RESUMABLE, detail)])
        else:
            if not finished:
                record(
                    Stage.RECOVERY, [(Invariant.SHUTDOWN_HANG, "abandoned-session recovery hangs")]
                )
        if world.tasks is not None:
            reason = await bounded(
                Stage.TASK_RECOVERY,
                _settle_task_with_another_worker(recovery, world),
                _TASK_RECOVERY_LIMIT_S + 2 * _SHUTDOWN_TIMEOUT_S,
            )
            if reason is not None:
                record(Stage.TASK_RECOVERY, [(Invariant.TASK_UNSETTLED, reason)])
            await check(Stage.RECOVERED, recovery)
        else:
            await check(Stage.RECOVERED)
            # A session whose setup never published its initial transcript fails
            # closed by design ("start a new session"); every other one continues.
            if await world.store.load(SESSION_ID) is not None and (
                _initial_transcript_pending_interaction_id(
                    await world.store.load_checkpoint(SESSION_ID)
                )
                is None
            ):
                reason = await bounded(
                    Stage.LIVENESS, _continue_to_completion(recovery, world), _LIVENESS_TIMEOUT_S
                )
                if reason is not None:
                    record(Stage.LIVENESS, [(Invariant.NOT_RESUMABLE, reason)])
                else:
                    await check(Stage.LIVENESS)
    except _StageHung:
        pass
    if recovery is not None:
        _, failure = await _close(recovery)
        if failure is not None:
            record(Stage.RECOVERY_SHUTDOWN, [(Invariant.SHUTDOWN_HANG, failure)])
    if refusing := await _cancel_leftover_tasks(set()):
        detail = f"{refusing} tasks ignore cancellation at the end of the case"
        record(Stage.TEARDOWN, [(Invariant.SHUTDOWN_HANG, detail)])
    return CaseResult(
        trace=list(injector.trace),
        events=[str(event.type) for event in events],
        reached=injector.fired is not None,
        violations=violations,
    )


def _run_isolated(scenario: Scenario, fault: FaultKind | None, point: str | None) -> CaseResult:
    """Run one case on an event loop of its own that never waits on abandoned tasks.

    ``asyncio.run`` waits without a bound for every task left at the end; a task
    that swallows cancellation, which :func:`run_case` has already recorded,
    would block it forever. A watchdog names the point if the case itself hangs.
    """

    watchdog = threading.Timer(_POINT_WATCHDOG_S, _report_hang, (scenario.name, fault, point))
    watchdog.daemon = True
    watchdog.start()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(run_case(scenario, fault, point))
        for shutdown in (loop.shutdown_asyncgens(), loop.shutdown_default_executor()):
            loop.run_until_complete(_bounded(shutdown, _SHUTDOWN_TIMEOUT_S))
        return result
    finally:
        watchdog.cancel()
        asyncio.set_event_loop(None)
        loop.close()


def _report_hang(scenario: str, fault: FaultKind | None, point: str | None) -> None:
    fault_name = fault.value if fault else "no fault"
    print(
        f"fault injection: {scenario} {fault_name} at {point or '-'} did not finish "
        f"in {_POINT_WATCHDOG_S:.0f}s",
        file=sys.__stderr__,
        flush=True,
    )
    faulthandler.dump_traceback(file=sys.__stderr__, all_threads=True)
    os._exit(1)


class Coverage(StrEnum):
    """Which of a scenario's injection points an exploration covers.

    ``sample`` takes the first point of every distinct call site: each boundary
    paired with the innermost Runtime function that crossed it, and every
    ``stream-close`` point. ``rest`` takes every other point, so the two together
    are exhaustive.
    """

    SAMPLE = "sample"
    REST = "rest"


@dataclass
class Exploration:
    scenario: str
    fault: FaultKind
    points: list[str]
    unreached: list[str]
    violations: list[Violation]


def injection_points(
    baseline: CaseResult, fault: FaultKind, coverage: Coverage | None = None
) -> list[str]:
    """Labels of the points a fault of this kind is injected at, in trace order.

    A label is the boundary (or, for ``stream-close``, the event type) and its
    occurrence among equal ones, such as ``store.load:enter#3``, so it survives
    unrelated changes elsewhere in the scenario. ``coverage`` narrows them to the
    sample or the rest; without it, every point is returned.
    """

    labeller = Injector()
    if fault is FaultKind.STREAM_CLOSE:
        points = [labeller.next_label(f"event:{event}") for event in baseline.events]
        return [] if coverage is Coverage.REST else points
    boundaries = [
        boundary for boundary in baseline.trace if fault is not FaultKind.CRASH or boundary.write
    ]
    labelled = [(labeller.next_label(boundary.site), boundary) for boundary in boundaries]
    if coverage is None:
        return [label for label, _ in labelled]
    # Without callers every boundary would collapse into one call site.
    assert any(boundary.caller for boundary in boundaries), "no Runtime caller under src/cayu/"
    first_by_call_site: dict[tuple[str, str], str] = {}
    for label, boundary in labelled:
        first_by_call_site.setdefault((boundary.site, boundary.caller), label)
    sample = set(first_by_call_site.values())
    if coverage is Coverage.SAMPLE:
        return [label for label, _ in labelled if label in sample]
    return [label for label, _ in labelled if label not in sample]


def run_point(scenario: Scenario, fault: FaultKind, point: str) -> CaseResult:
    """Inject one fault at one point, in a fresh event loop."""

    return _run_isolated(scenario, fault, point)


def baseline_points(
    scenario: Scenario, fault: FaultKind, coverage: Coverage | None = None
) -> list[str]:
    baseline = _run_isolated(scenario, None, None)
    assert not baseline.violations, [v.describe() for v in baseline.violations]
    return injection_points(baseline, fault, coverage)


def explore(scenario: Scenario, fault: FaultKind, coverage: Coverage) -> Exploration:
    """Inject one fault at each covered point of the scenario, each in a fresh event loop."""

    points = baseline_points(scenario, fault, coverage)
    unreached: list[str] = []
    violations: list[Violation] = []
    for point in points:
        result = run_point(scenario, fault, point)
        if not result.reached:
            unreached.append(point)
        violations.extend(result.violations)
    return Exploration(scenario.name, fault, points, unreached, violations)
