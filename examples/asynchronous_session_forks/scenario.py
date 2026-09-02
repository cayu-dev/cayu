"""Application-owned orchestration of independently dispatched session forks."""

from __future__ import annotations

import asyncio
import contextlib
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from cayu import (
    AgentSpec,
    CayuApp,
    DispatchHandle,
    DispatchRequest,
    DispatchStatus,
    Event,
    EventType,
    ExecutionProfileBehaviorIdentity,
    ForkSessionRequest,
    ForkSourceSnapshot,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ResumeRequest,
    RetryPolicy,
    RunRequest,
    Session,
    SessionStatus,
    Task,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    TaskStoreDispatcher,
    TaskTerminalizationRequest,
    ToolCapabilityCeiling,
)
from cayu.core import MessageRole, TextPart, ToolResultPart
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import ModelCompletionStageResult, RuntimePublicationRequest

TRUNK_SESSION_ID = "async-forks-trunk"
CHILD_SESSION_IDS = {
    "A": "async-forks-child-a",
    "B": "async-forks-child-b",
    "C": "async-forks-child-c",
    "D": "async-forks-child-d",
}


@dataclass(frozen=True, slots=True)
class AsynchronousForkTrace:
    """Bounded evidence produced by the deterministic example."""

    source: ForkSourceSnapshot
    trace: tuple[str, ...]
    queue_task_ids: dict[str, str]
    first_result: str
    selected_child: str
    child_statuses: dict[str, str]
    provider_calls: dict[str, int]
    provider_completions: dict[str, int]
    provider_cancellations: dict[str, int]
    tool_invocations: int
    tool_mutations: int


class _InjectedWorkerLoss(BaseException):
    """Simulate process loss without letting ordinary exception recovery run."""


class _CommitThenLoseAcknowledgementSessionStore(InMemorySessionStore):
    """Inject worker loss after provider and terminal session writes commit."""

    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.lose_provider_completion_for: str | None = None
        self.lose_terminal_publication_for: str | None = None

    async def complete_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        publication: RuntimePublicationRequest,
    ) -> ModelCompletionStageResult:
        result = await super().complete_model_completion_stage(
            session_id,
            stage_id=stage_id,
            publication=publication,
        )
        if session_id == self.lose_provider_completion_for:
            self.lose_provider_completion_for = None
            raise _InjectedWorkerLoss("worker lost after durable provider completion")
        return result

    async def append_event(self, session_id: str, event: Event) -> None:
        await super().append_event(session_id, event)
        if (
            session_id == self.lose_terminal_publication_for
            and event.type is EventType.SESSION_COMPLETED
        ):
            self.lose_terminal_publication_for = None
            raise _InjectedWorkerLoss("worker lost after durable terminal publication")


class _CommitThenLoseAcknowledgementTaskStore(InMemoryTaskStore):
    """Inject bounded queue failures after their durable writes commit."""

    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self.lose_create_acknowledgement = True
        self.lose_terminal_acknowledgement = True
        self.pause_next_claim = False
        self.claim_committed = asyncio.Event()
        self.claimed_task: Task | None = None

    async def create_task(self, request: TaskCreate) -> Task:
        task = await super().create_task(request)
        if self.lose_create_acknowledgement:
            self.lose_create_acknowledgement = False
            raise ConnectionError("injected queue-publication acknowledgement loss")
        return task

    async def claim_task(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        lease_seconds: int = 300,
    ) -> Task | None:
        task = await super().claim_task(
            worker_id,
            query,
            lease_seconds=lease_seconds,
        )
        if task is not None and self.pause_next_claim:
            self.pause_next_claim = False
            self.claimed_task = task
            self.claim_committed.set()
            await asyncio.Event().wait()
        return task

    async def terminalize_task(
        self,
        request: TaskTerminalizationRequest,
    ) -> Task:
        task = await super().terminalize_task(request)
        if self.lose_terminal_acknowledgement:
            self.lose_terminal_acknowledgement = False
            raise ConnectionError("injected terminal-publication acknowledgement loss")
        return task


class _ExactlyOnceEffect:
    def __init__(self) -> None:
        self.invocations = 0
        self.mutations = 0
        self._receipts: dict[str, ToolResult] = {}

    def apply(self, idempotency_key: str) -> ToolResult:
        self.invocations += 1
        existing = self._receipts.get(idempotency_key)
        if existing is not None:
            return existing
        self.mutations += 1
        result = ToolResult(
            content="recorded child A effect",
            structured={"mutation": "recorded", "ordinal": self.mutations},
        )
        self._receipts[idempotency_key] = result
        return result


class _RecordEffectTool(Tool):
    spec = ToolSpec(
        name="record_effect",
        description="Record one deterministic idempotent child effect.",
        input_schema={
            "type": "object",
            "properties": {"child": {"const": "A"}},
            "required": ["child"],
            "additionalProperties": False,
        },
        effect=ToolEffect.IDEMPOTENT,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="examples:asynchronous-session-forks:record-effect",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(self, effect: _ExactlyOnceEffect) -> None:
        self._effect = effect

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if args != {"child": "A"}:
            raise ValueError("record_effect received unexpected arguments")
        if ctx.idempotency_key is None:
            raise RuntimeError("record_effect requires a Runtime idempotency key")
        return self._effect.apply(ctx.idempotency_key)


def _message_text(request: ModelRequest) -> tuple[str, ...]:
    return tuple(
        part.text
        for message in request.messages
        for part in message.content
        if isinstance(part, TextPart)
    )


def _has_tool_result(request: ModelRequest, tool_name: str) -> bool:
    return any(
        isinstance(part, ToolResultPart) and part.tool_name == tool_name
        for message in request.messages
        for part in message.content
    )


class _BarrierProvider(ModelProvider):
    """Provider fixture controlled only by events, never timing sleeps."""

    name = "deterministic-asynchronous-forks"

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.completions: Counter[str] = Counter()
        self.cancellations: Counter[str] = Counter()
        self.started = {name: asyncio.Event() for name in CHILD_SESSION_IDS}
        self.release = {name: asyncio.Event() for name in ("A", "C", "D")}

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="examples:asynchronous-session-forks:provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        texts = _message_text(request)
        child = next(
            (name for name in CHILD_SESSION_IDS if f"worker:{name}" in texts),
            None,
        )
        operation = (
            f"child:{child}"
            if child is not None
            else "checkpoint:15"
            if "trunk:checkpoint-15" in texts
            else "checkpoint:14"
            if "trunk:checkpoint-14" in texts
            else "consume:B"
            if any(text.startswith("consume:B:") for text in texts)
            else "evaluate"
            if any(text.startswith("evaluate:") for text in texts)
            else "unknown"
        )
        self.calls[operation] += 1

        if child is not None:
            self.started[child].set()
            if child in self.release and not (
                child == "A" and _has_tool_result(request, "record_effect")
            ):
                try:
                    await self.release[child].wait()
                except asyncio.CancelledError:
                    self.cancellations[operation] += 1
                    raise
            if child == "D":
                raise RuntimeError("deterministic child D failure")
            if child == "A" and not _has_tool_result(request, "record_effect"):
                yield ModelStreamEvent.tool_call(
                    id="record-child-a",
                    name="record_effect",
                    arguments={"child": "A"},
                )
                self.completions[operation] += 1
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            output = f"result:{child}"
        elif operation == "consume:B":
            output = "used:result:B"
        elif operation == "evaluate":
            output = "selected:B"
        else:
            output = operation

        yield ModelStreamEvent.text_delta(output)
        self.completions[operation] += 1
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _build_app(
    sessions: _CommitThenLoseAcknowledgementSessionStore,
    tasks: _CommitThenLoseAcknowledgementTaskStore,
    provider: _BarrierProvider,
    effect: _ExactlyOnceEffect,
) -> tuple[CayuApp, TaskStoreDispatcher]:
    dispatcher = TaskStoreDispatcher(
        tasks,
        recover_stalled_sessions_after_seconds=0,
    )
    app = CayuApp(
        session_store=sessions,
        task_store=tasks,
        dispatcher=dispatcher,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="worker", model="deterministic-model"),
        tools=[_RecordEffectTool(effect)],
    )
    return app, dispatcher


def _invocation(child: str) -> ResumeRequest:
    tool_names = ("record_effect",) if child == "A" else ()
    return ResumeRequest(
        session_id=CHILD_SESSION_IDS[child],
        messages=[Message.text("user", f"worker:{child}")],
        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=tool_names),
        metadata={
            "application_worker": child,
            "declared_artifacts": [{"artifact_id": f"candidate-{child.lower()}"}],
        },
        max_steps=3,
        retry_policy=RetryPolicy(
            max_attempts=1,
            max_unknown_attempts=1,
            initial_delay_s=0.0,
            max_delay_s=0.0,
            jitter_s=0.0,
        ),
    )


def _dispatch_id(child: str) -> str:
    return f"dispatch-{child.lower()}"


def _dispatch_request(child: str, invocation: ResumeRequest) -> DispatchRequest:
    return DispatchRequest(
        session_id=invocation.session_id,
        dispatch_id=_dispatch_id(child),
        messages=invocation.messages,
        tool_capability_ceiling=invocation.tool_capability_ceiling,
        tool_grants=invocation.tool_grants,
        profile_adoption=invocation.profile_adoption,
        metadata=invocation.metadata,
        max_steps=invocation.max_steps,
        limits=invocation.limits,
        budget_limits=invocation.budget_limits,
        retry_policy=invocation.retry_policy,
        structured_output=invocation.structured_output,
        thinking=invocation.thinking,
    )


async def _assistant_text(store: InMemorySessionStore, session_id: str) -> str:
    transcript = await store.load_transcript_snapshot(session_id)
    for record in reversed(transcript.records):
        if record.message.role is not MessageRole.ASSISTANT:
            continue
        texts = [part.text for part in record.message.content if isinstance(part, TextPart)]
        if texts:
            return "".join(texts)
    raise RuntimeError(f"Session {session_id} has no assistant result.")


async def _require_session(store: InMemorySessionStore, session_id: str) -> Session:
    session = await store.load(session_id)
    if session is None:
        raise AssertionError(f"Session {session_id} disappeared.")
    return session


async def _collect(stream) -> list:
    return [event async for event in stream]


async def run_asynchronous_fork_trace() -> AsynchronousForkTrace:
    """Run the exact-source, early-dispatch, independent-result tracer."""

    sessions = _CommitThenLoseAcknowledgementSessionStore()
    tasks = _CommitThenLoseAcknowledgementTaskStore()
    provider = _BarrierProvider()
    effect = _ExactlyOnceEffect()
    trace: list[str] = []

    producer, _ = _build_app(sessions, tasks, provider, effect)
    await _collect(
        producer.run(
            RunRequest(
                agent_name="worker",
                session_id=TRUNK_SESSION_ID,
                causal_budget_id="async-forks-budget",
                messages=[Message.text("user", "trunk:checkpoint-14")],
                metadata={"application_checkpoint": 14},
            )
        )
    )
    source = await producer.snapshot_fork_source(TRUNK_SESSION_ID)
    trace.append("trunk_checkpoint_14_complete")

    invocations = {child: _invocation(child) for child in CHILD_SESSION_IDS}
    fork_requests = {
        child: ForkSessionRequest(
            source_session_id=TRUNK_SESSION_ID,
            session_id=CHILD_SESSION_IDS[child],
            tool_capability_ceiling=invocation.tool_capability_ceiling,
            expected_source=source,
            initial_invocation=invocation,
            initial_dispatch_id=_dispatch_id(child),
            metadata={"application_worker": child},
        )
        for child, invocation in invocations.items()
    }
    for child in CHILD_SESSION_IDS:
        await _collect(producer.fork_session(fork_requests[child]))
        producer, _ = _build_app(sessions, tasks, provider, effect)
        await _collect(producer.fork_session(fork_requests[child]))
    trace.append("fork_admission_replayed_after_producer_reconstruction")

    dispatch_requests = {
        child: _dispatch_request(child, invocations[child]) for child in CHILD_SESSION_IDS
    }
    handles: dict[str, DispatchHandle] = {}
    for child in ("B", "A", "C", "D"):
        producer, _ = _build_app(sessions, tasks, provider, effect)
        first = await producer.dispatch(dispatch_requests[child])
        replay = await producer.dispatch(dispatch_requests[child])
        if first.metadata["queue_task_id"] != replay.metadata["queue_task_id"]:
            raise AssertionError("Dispatch retry changed queue identity.")
        handles[child] = first
    if provider.calls != Counter({"checkpoint:14": 1}):
        raise AssertionError("Durable admission executed child model work.")
    trace.append("all_children_durably_dispatched_before_model_execution")

    await _collect(
        producer.resume(
            ResumeRequest(
                session_id=TRUNK_SESSION_ID,
                messages=[Message.text("user", "trunk:checkpoint-15")],
                metadata={"application_checkpoint": 15},
            )
        )
    )
    trace.append("trunk_checkpoint_15_complete_while_children_pending")

    tasks.pause_next_claim = True
    pre_crash_worker, pre_crash_dispatcher = _build_app(sessions, tasks, provider, effect)
    lost_worker = asyncio.create_task(
        pre_crash_dispatcher.process_next(
            pre_crash_worker,
            worker_id="worker-before-reconstruction",
        )
    )
    await tasks.claim_committed.wait()
    lost_worker.cancel("injected worker loss after durable claim")
    with contextlib.suppress(asyncio.CancelledError):
        await lost_worker
    claimed = tasks.claimed_task
    if claimed is None:
        raise AssertionError("Claim barrier did not retain the claimed task.")
    if claimed.lease_expires_at is None:
        raise AssertionError("Claim barrier retained a task without a lease.")
    await tasks.release_task(
        claimed.id,
        "worker-before-reconstruction",
        lease_expires_at=claimed.lease_expires_at,
    )
    trace.append("worker_reconstructed_after_claim_boundary")

    worker, dispatcher = _build_app(sessions, tasks, provider, effect)
    result_b = await dispatcher.process_next(
        worker,
        worker_id="worker-b",
    )
    if result_b is None or result_b.status is not DispatchStatus.COMPLETED:
        raise AssertionError(f"Child B did not settle first: {result_b!r}.")

    processing: dict[str, asyncio.Task[DispatchHandle | None]] = {}
    for child in ("A", "C", "D"):
        processing[child] = asyncio.create_task(
            dispatcher.process_next(worker, worker_id=f"worker-{child.lower()}")
        )
        await provider.started[child].wait()
    first_result = await _assistant_text(sessions, CHILD_SESSION_IDS["B"])
    live_statuses = {
        child: (await _require_session(sessions, CHILD_SESSION_IDS[child])).status
        for child in ("A", "C", "D")
    }
    if any(status is not SessionStatus.RUNNING for status in live_statuses.values()):
        raise AssertionError("A sibling settled before child B was consumed.")
    trace.append("child_b_observed_while_siblings_running")

    consumer, _ = _build_app(sessions, tasks, provider, effect)
    await _collect(
        consumer.run(
            RunRequest(
                agent_name="worker",
                session_id="async-forks-first-result-consumer",
                causal_budget_id=source.causal_budget_id,
                messages=[Message.text("user", f"consume:B:{first_result}")],
            )
        )
    )
    if await _assistant_text(sessions, "async-forks-first-result-consumer") != "used:result:B":
        raise AssertionError("Child B result was not immediately usable.")
    trace.append("child_b_result_used_before_siblings_settled")

    provider.release["D"].set()
    result_d = await processing["D"]
    if result_d is None or result_d.status is not DispatchStatus.FAILED:
        raise AssertionError("Child D did not fail independently.")
    if (
        await _require_session(sessions, CHILD_SESSION_IDS["B"])
    ).status is not SessionStatus.COMPLETED:
        raise AssertionError("Child D failure changed completed child B.")
    surviving_statuses = {
        child: (await _require_session(sessions, CHILD_SESSION_IDS[child])).status
        for child in ("A", "C")
    }
    if any(status is not SessionStatus.RUNNING for status in surviving_statuses.values()):
        raise AssertionError("Child D failure changed a running sibling.")
    trace.append("child_d_failed_without_altering_siblings")

    sessions.lose_terminal_publication_for = CHILD_SESSION_IDS["A"]
    provider.release["A"].set()
    try:
        await processing["A"]
    except _InjectedWorkerLoss:
        pass
    else:
        raise AssertionError("Terminal-publication worker loss was not injected.")
    trace.append("worker_lost_after_durable_terminal_publication")
    claimed_a = await tasks.load_task(handles["A"].metadata["queue_task_id"])
    if claimed_a is None or claimed_a.lease_expires_at is None:
        raise AssertionError("Child A lost its claimed dispatch lease.")
    await tasks.release_task(
        claimed_a.id,
        "worker-a",
        lease_expires_at=claimed_a.lease_expires_at,
    )
    terminal_recovery_worker, terminal_recovery_dispatcher = _build_app(
        sessions,
        tasks,
        provider,
        effect,
    )
    result_a = await terminal_recovery_dispatcher.process_next(
        terminal_recovery_worker,
        worker_id="worker-a-after-terminal-publication",
    )
    if result_a is None or result_a.status is not DispatchStatus.COMPLETED:
        raise AssertionError("Child A did not complete after child B.")
    trace.append("terminal_publication_recovered_without_redispatch")
    trace.append("child_a_completed_later")

    sessions.lose_provider_completion_for = CHILD_SESSION_IDS["C"]
    provider.release["C"].set()
    try:
        await processing["C"]
    except _InjectedWorkerLoss:
        pass
    else:
        raise AssertionError("Provider-completion worker loss was not injected.")
    trace.append("worker_lost_after_durable_provider_completion")
    claimed_c = await tasks.load_task(handles["C"].metadata["queue_task_id"])
    if claimed_c is None or claimed_c.lease_expires_at is None:
        raise AssertionError("Child C lost its claimed dispatch lease.")
    await tasks.release_task(
        claimed_c.id,
        "worker-c",
        lease_expires_at=claimed_c.lease_expires_at,
    )
    calls_before_c_recovery = Counter(provider.calls)
    provider_recovery_worker, provider_recovery_dispatcher = _build_app(
        sessions,
        tasks,
        provider,
        effect,
    )
    recovery_c = await provider_recovery_worker.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=CHILD_SESSION_IDS["C"],
            inactive_for_seconds=0,
            reason="Application no longer needs child C after worker reconstruction.",
        )
    )
    if recovery_c.status is not SessionStatus.INTERRUPTED or not recovery_c.events:
        raise AssertionError(
            f"Provider-completion recovery did not interrupt child C: {recovery_c!r}."
        )
    result_c = await provider_recovery_dispatcher.process_next(
        provider_recovery_worker,
        worker_id="worker-c-after-provider-completion",
    )
    if result_c is None or result_c.status is not DispatchStatus.INTERRUPTED:
        raise AssertionError("Child C did not interrupt independently.")
    if Counter(provider.calls) != calls_before_c_recovery:
        raise AssertionError("Child C provider completion was dispatched more than once.")
    trace.append("provider_completion_recovered_without_redispatch")
    trace.append("child_c_interrupted_independently")

    candidate_a = await _assistant_text(sessions, CHILD_SESSION_IDS["A"])
    evaluator, _ = _build_app(sessions, tasks, provider, effect)
    await _collect(
        evaluator.run(
            RunRequest(
                agent_name="worker",
                session_id="async-forks-application-evaluator",
                causal_budget_id=source.causal_budget_id,
                messages=[
                    Message.text(
                        "user",
                        f"evaluate:A={candidate_a};B={first_result};choose one session",
                    )
                ],
            )
        )
    )
    evaluator_output = await _assistant_text(sessions, "async-forks-application-evaluator")
    selected_child = evaluator_output.removeprefix("selected:")
    if selected_child not in {"A", "B"}:
        raise AssertionError("Application evaluator selected an unavailable child.")
    trace.append("application_owned_evaluator_selected_settled_child")

    calls_before_recovery = Counter(provider.calls)
    mutations_before_recovery = effect.mutations
    recovered, recovered_dispatcher = _build_app(sessions, tasks, provider, effect)
    for child in CHILD_SESSION_IDS:
        await _collect(recovered.fork_session(fork_requests[child]))
        replay = await recovered.dispatch(dispatch_requests[child])
        if replay.metadata["queue_task_id"] != handles[child].metadata["queue_task_id"]:
            raise AssertionError("Recovered dispatch changed queue identity.")
    if (
        await recovered_dispatcher.process_next(
            recovered,
            worker_id="worker-after-terminal-reconstruction",
        )
        is not None
    ):
        raise AssertionError("Terminal reconstruction left duplicate work claimable.")
    if (
        Counter(provider.calls) != calls_before_recovery
        or effect.mutations != mutations_before_recovery
    ):
        raise AssertionError("Recovery duplicated provider or tool work.")
    trace.append("terminal_reconstruction_created_no_duplicate_work")

    child_statuses: dict[str, str] = {}
    for child, session_id in CHILD_SESSION_IDS.items():
        session = await _require_session(sessions, session_id)
        child_statuses[child] = session.status.value
        if session.parent_session_id != TRUNK_SESSION_ID:
            raise AssertionError(f"Child {child} lost parent lineage.")
        if session.causal_budget_id != source.causal_budget_id:
            raise AssertionError(f"Child {child} lost causal-budget identity.")
        if session.metadata.get("cayu:fork_source_snapshot") != source.model_dump(mode="json"):
            raise AssertionError(f"Child {child} lost exact source authority.")

    terminal_tasks = {
        child: await tasks.load_task(handle.metadata["queue_task_id"])
        for child, handle in handles.items()
    }
    if any(
        task is None or task.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED}
        for task in terminal_tasks.values()
    ):
        raise AssertionError("A child queue task is not terminal.")
    if effect.invocations != 1 or effect.mutations != 1:
        raise AssertionError("Child A tool effect was not exactly once.")

    return AsynchronousForkTrace(
        source=source,
        trace=tuple(trace),
        queue_task_ids={
            child: str(handle.metadata["queue_task_id"]) for child, handle in handles.items()
        },
        first_result=first_result,
        selected_child=selected_child,
        child_statuses=child_statuses,
        provider_calls=dict(sorted(provider.calls.items())),
        provider_completions=dict(sorted(provider.completions.items())),
        provider_cancellations=dict(sorted(provider.cancellations.items())),
        tool_invocations=effect.invocations,
        tool_mutations=effect.mutations,
    )
