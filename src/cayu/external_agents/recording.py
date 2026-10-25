"""Session recording and after-step governance usable with any adapter."""

from __future__ import annotations

import asyncio
import codecs
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, aclosing, suppress
from hashlib import sha256
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator

from cayu._resource_access_errors import ResourceAccessDenied
from cayu.applications import CayuApp
from cayu.budgets.base import (
    budget_check_from_events,
    budget_check_payload,
    budget_limits_for_session,
    events_for_budget_window,
    request_budget_limits_for_session,
)
from cayu.budgets.usage import normalize_usage_metrics, usage_metrics_payload
from cayu.deadlines import (
    ExecutionDeadlineExceeded,
    current_execution_deadline,
    effective_deadline,
    execution_deadline_scope,
)
from cayu.events import Event, EventType
from cayu.external_agents._authority import cancel_run_once, register_run, release_run
from cayu.external_agents._cleanup import ExternalCleanupIncomplete, settle_external_cleanup
from cayu.external_agents.base import (
    ExternalAgentAdapter,
    ExternalAgentBudgetExceeded,
    ExternalAgentContext,
    ExternalAgentError,
    ExternalAgentEvent,
    ExternalAgentResult,
)
from cayu.external_agents.environment import join_cleanup
from cayu.messages import Message
from cayu.resource_access import execution_access, require_dispatch
from cayu.runtime._task_lease_authority import managed_task_lease_mutation
from cayu.runtime.application_lifecycle import _admitted_entrance
from cayu.sessions._invocation_scope import (
    _activate_owned_session_run_fence,
    _deactivate_session_run_fence,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import Session, SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest, copy_run_request, run_request_with_task_invocation
from cayu.tasks.creation import TaskCreate
from cayu.tasks.records import Task
from cayu.tasks.worker import complete_managed_task, fail_managed_task


class ExternalAgentRecorder:
    """Persist bounded, redacted observations and check normal budget scopes."""

    def __init__(self, app: CayuApp, session: Session, request: RunRequest) -> None:
        self.app = app
        self.session = session
        self.seen: dict[str, str] = {}
        self.usage_observed = False
        self.pending_usage: dict[str, Event] = {}
        self.usage_lock = asyncio.Lock()
        self.gateway_usage_authoritative = False
        self.budget_exhausted = False
        self.observed_bytes = 0
        self.redactor = app._secret_redactor
        self.text_stream = self.redactor.stream_bytes(max_retained_bytes=1024 * 1024)
        self.text_decoder = codecs.getincrementaldecoder("utf-8")()
        self.limits = (
            *budget_limits_for_session(
                policy=app.budget_policy,
                agent_name=session.agent_name,
                causal_budget_id=session.causal_budget_id,
            ),
            *request_budget_limits_for_session(
                limits=request.budget_limits,
                agent_name=session.agent_name,
                causal_budget_id=session.causal_budget_id,
            ),
        )
        if any(limit.reservation is not None for limit in self.limits):
            raise ValueError("External agents cannot reserve a model call they do not dispatch.")

    def protect_secret(self, value: str) -> None:
        if self.seen:
            raise ValueError("External secrets must be declared before the first SDK observation.")
        self.redactor = self.redactor.with_secret(value)
        self.text_stream = self.redactor.stream_bytes(max_retained_bytes=1024 * 1024)

    def claim_model_usage(self) -> None:
        if self.usage_observed and not self.gateway_usage_authoritative:
            raise ValueError("Gateway accounting must be selected before SDK usage is recorded.")
        self.gateway_usage_authoritative = True

    async def emit(self, event_type: EventType | str, payload: dict[str, Any]) -> Event:
        return await self.app.emit_event(
            Event(
                type=event_type,
                session_id=self.session.id,
                agent_name=self.session.agent_name,
                payload=self.redactor.redact_json(
                    {
                        "provider_name": self.session.provider_name,
                        "model": self.session.model,
                        **payload,
                    }
                ),
            )
        )

    async def require_access(self) -> None:
        async with execution_access(
            self.session.invocation.resource_access,
            self.app.resource_access_policy,
            self.session.labels,
            store=self.app.session_store,
            session_id=self.session.id,
        ):
            await require_dispatch()

    async def check_budget(self) -> None:
        async with self.usage_lock:
            await self.require_access()
            if self.pending_usage:
                raise ExternalAgentError("External run accounting requires reconciliation.")
            if len(self.seen) >= 100_000 or self.observed_bytes >= 64 * 1024 * 1024:
                raise ExternalAgentError("External observation limit reached.")
            current = await self.app.session_store.load(self.session.id)
            if (
                current is None
                or current.instance_id != self.session.instance_id
                or current.run_epoch != self.session.run_epoch
                or current.status is not SessionStatus.RUNNING
            ):
                raise ExternalAgentError("External session no longer has execution authority.")
            for limit in self.limits:
                budget_now = self.app._clock()
                if limit.scope in {"session", "run"}:
                    events = events_for_budget_window(
                        await self.app.session_store.load_events(self.session.id),
                        limit.window,
                        now=budget_now,
                    )
                else:
                    events = await self.app.budget_store.load_events_for_budget(
                        scope=limit.scope, key=limit.key, window=limit.window
                    )
                check = budget_check_from_events(
                    limit=limit,
                    events=events,
                    provider_name=self.session.provider_name,
                    model=self.session.model,
                    effective_at=budget_now,
                )
                await self.emit(EventType.BUDGET_CHECKED, budget_check_payload(check))
                if check.limit_reached and check.action == "notify":
                    since, until = limit.window.bounds(now=budget_now)
                    if not await self.app.session_store.event_exists(
                        EventQuery(
                            session_id=self.session.id
                            if limit.scope in {"session", "run"}
                            else None,
                            agent_name=limit.key if limit.scope == "agent" else None,
                            causal_budget_id=limit.key if limit.scope == "causal" else None,
                            event_type=EventType.BUDGET_LIMIT_REACHED,
                            budget_limit_id=check.budget_limit_id,
                            since=since,
                            until=until,
                            limit=1,
                        )
                    ):
                        await self.emit(EventType.BUDGET_LIMIT_REACHED, budget_check_payload(check))
                if check.limit_reached and check.action == "interrupt":
                    self.budget_exhausted = True
                    raise ExternalAgentBudgetExceeded(
                        "External agent budget cannot admit more work."
                    )

    async def record_model_usage(self, observation: ExternalAgentEvent) -> None:
        """Settle an already admitted gateway call even after SDK quota exhaustion."""
        if observation.kind != "usage":
            raise ValueError("Model accounting requires a usage observation.")
        await self.record(observation, check_after=False, admitted_model_usage=True)

    async def record(
        self,
        observation: ExternalAgentEvent,
        *,
        check_after: bool = True,
        admitted_model_usage: bool = False,
    ) -> None:
        from cayu._validation import canonical_durable_json_bytes

        material = canonical_durable_json_bytes(
            {
                "kind": observation.kind,
                "model": observation.model,
                "payload": observation.payload,
                "usage_dialect": observation.usage_dialect,
            },
            "external observation",
        )
        if len(material) > 8 * 1024 * 1024:
            raise ExternalAgentError("External observation exceeds its size limit.")
        digest = sha256(material).hexdigest()
        previous = self.seen.get(observation.id)
        if previous is not None:
            if previous != digest:
                raise ExternalAgentError("Conflicting external observation identity.")
            return
        if not admitted_model_usage and len(self.seen) >= 100_000:
            raise ExternalAgentError("External observation limit reached.")
        self.observed_bytes += len(material)
        if not admitted_model_usage and self.observed_bytes > 64 * 1024 * 1024:
            raise ExternalAgentError("External run exceeds its observation size limit.")
        if observation.kind == "usage":
            async with self.usage_lock:
                event = self.pending_usage.get(observation.id) or Event(
                    id=f"external-usage-{sha256((self.session.id + observation.id).encode()).hexdigest()}",
                    type=EventType.MODEL_COMPLETED,
                    session_id=self.session.id,
                    agent_name=self.session.agent_name,
                    payload={
                        "usage": self.redactor.redact_json(observation.payload),
                        "external_agent": True,
                        "provider_name": self.session.provider_name,
                        "model": observation.model or self.session.model,
                        **(
                            {
                                "usage_metrics": usage_metrics_payload(
                                    normalize_usage_metrics(
                                        provider_name=self.session.provider_name,
                                        model=observation.model or self.session.model,
                                        raw_usage=observation.payload,
                                        usage_dialect=observation.usage_dialect,
                                    )
                                )
                            }
                            if observation.usage_dialect is not None
                            else {}
                        ),
                    },
                )
                self.pending_usage[observation.id] = event
                await self.app.emit_event(event)
                self.pending_usage.pop(observation.id)
                self.seen[observation.id] = digest
                self.usage_observed = True
            if check_after:
                await self.check_budget()
        elif observation.kind == "text":
            safe = self.text_decoder.decode(
                self.text_stream.feed(observation.payload["text"].encode())
            )
            if safe:
                await self.emit("custom.external_agent.text", {"text": safe})
        elif observation.kind in {"tool_call", "tool_result"}:
            # SDK observations have no native publication contract. Exact
            # arguments and results belong to the governed bridge's policy and
            # tool events. SDK completion items can repeat the entire input.
            summary = {}
            for key, aliases in {
                "name": ("name", "tool"),
                "call_id": ("call_id", "id", "tool_use_id"),
                "server": ("server",),
                "type": ("type",),
                "status": ("status",),
            }.items():
                value = next(
                    (
                        observation.payload[alias]
                        for alias in aliases
                        if alias in observation.payload
                    ),
                    None,
                )
                if isinstance(value, str):
                    summary[key] = self.redactor.redact_text(value)[:256]
            if type(observation.payload.get("is_error")) is bool:
                summary["is_error"] = observation.payload["is_error"]
            await self.emit(f"custom.external_agent.{observation.kind}", summary)
        else:
            await self.emit(f"custom.external_agent.{observation.kind}", observation.payload)

        self.seen[observation.id] = digest

    async def settle_usage(self) -> None:
        async with self.usage_lock:
            for observation_id, event in tuple(self.pending_usage.items()):
                await self.app.emit_event(event)
                self.pending_usage.pop(observation_id)
                self.usage_observed = True

    async def finish_text(self) -> None:
        safe = self.text_decoder.decode(self.text_stream.finish_complete(), final=True)
        if safe:
            await self.emit("custom.external_agent.text", {"text": safe})


async def _settle_task(
    app: CayuApp, task: Task, request: RunRequest, *, error: dict[str, Any] | None
) -> None:
    assert app.task_store is not None
    worker_id = request.task_worker_id
    if worker_id is not None:
        # The attachment may acknowledge an intermediate heartbeat generation.
        # The original handler lease authenticates every subsequent mutation
        # against the managed worker's current, serialized lease authority.
        task = task.model_copy(update={"lease_expires_at": request.task_lease_expires_at})
        if error is None:
            await complete_managed_task(
                app.task_store, task, worker_id, {"session_id": task.session_id}
            )
        else:
            await fail_managed_task(app.task_store, task, worker_id, error)
    elif error is None:
        await app.task_store.complete_task(task.id, {"session_id": task.session_id})
    else:
        await app.task_store.fail_task(task.id, error)


@_admitted_entrance
async def run_external_agent(
    app: CayuApp,
    request: RunRequest,
    adapter: ExternalAgentAdapter,
    *,
    prompt: str,
    output_schema: dict[str, Any] | None = None,
    authorize: Callable[[str, dict[str, Any], str], Awaitable[bool]] | None = None,
) -> ExternalAgentResult:
    """Run one fresh vendor harness, recording a Cayu session and ordinary task.

    Use an existing worker-owned ``request.task_id`` for durable scheduling.
    Retry/restart always allocates a fresh session and fresh vendor thread.
    The adapter owns sandbox teardown and must join it before its stream closes.
    See ``cayu guide external-agents``.
    """
    supported = {
        "agent_name",
        "messages",
        "metadata",
        "labels",
        "execution_deadline",
        "budget_limits",
        "task_id",
        "task_worker_id",
        "task_lease_expires_at",
    }
    unsupported = set(request.model_dump(exclude_defaults=True, exclude=supported))
    if unsupported:
        raise ValueError(
            "External runs do not support RunRequest fields: " + ", ".join(sorted(unsupported))
        )
    request = copy_run_request(request)
    request = request.model_copy(
        update={
            "execution_deadline": effective_deadline(
                current_execution_deadline(), request.execution_deadline
            )
        }
    )
    if app.task_store is None:
        raise ValueError("External agent runs require a TaskStore.")
    if request.execution_deadline.expires_at is None:
        raise ValueError("External agent runs require a finite execution_deadline.")
    applicable_limits = (
        *budget_limits_for_session(
            policy=app.budget_policy,
            agent_name=request.agent_name,
            causal_budget_id=request.task_id or "external-preflight",
        ),
        *request_budget_limits_for_session(
            limits=request.budget_limits,
            agent_name=request.agent_name,
            causal_budget_id=request.task_id or "external-preflight",
        ),
    )
    if any(limit.reservation is not None for limit in applicable_limits):
        raise ValueError("External agents do not support model spend reservations.")
    if output_schema is not None:
        Draft202012Validator.check_schema(output_schema)
    try:
        async with execution_deadline_scope(request.execution_deadline):
            return await _run_external_agent(
                app,
                request,
                adapter,
                prompt=prompt,
                output_schema=output_schema,
                authorize=authorize,
            )
    except TimeoutError as exc:
        if "execution_deadline" in exc.__dict__:
            raise ExecutionDeadlineExceeded(request.execution_deadline, "external_agent") from exc
        raise


async def _run_external_agent(
    app: CayuApp,
    request: RunRequest,
    adapter: ExternalAgentAdapter,
    *,
    prompt: str,
    output_schema: dict[str, Any] | None,
    authorize: Callable[[str, dict[str, Any], str], Awaitable[bool]] | None,
) -> ExternalAgentResult:
    assert app.task_store is not None
    implicit_task = request.task_id is None
    task_id = request.task_id or str(uuid4())
    session_id = str(uuid4())
    request = request.model_copy(update={"task_id": task_id, "session_id": session_id})
    session: Session | None = None
    recorder: ExternalAgentRecorder | None = None
    attached: Task | None = None
    monitor: asyncio.Task[None] | None = None
    launched = False
    authority = None
    access = AsyncExitStack()
    owner = asyncio.current_task()
    assert owner is not None

    async def observe_interrupt() -> None:
        assert session is not None
        while True:
            await asyncio.sleep(0.1)
            try:
                if recorder is not None:
                    await recorder.require_access()
            except ResourceAccessDenied:
                cancel_run_once(session, (owner,), message="External session access was revoked.")
                return
            current = await app.session_store.load(session_id)
            if current is not None and current.status is SessionStatus.COMPLETED:
                return
            if (
                current is None
                or current.instance_id != session.instance_id
                or current.run_epoch != session.run_epoch
                or current.status is not SessionStatus.RUNNING
            ):
                cancel_run_once(
                    session, (owner,), message="External session execution authority ended."
                )
                return

    async def stop_monitor() -> None:
        nonlocal monitor
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            monitor = None

    try:
        request.execution_deadline.require_admission("external session setup")
        if implicit_task:
            await app.create_task(
                TaskCreate(
                    task_id=task_id,
                    session_id=session_id,
                    type="external_agent",
                    input={"agent": adapter.name},
                )
            )
        task = await app.task_store.load_task(task_id)
        if task is not None and (task.retry_series is not None or task.work_contract is not None):
            raise ValueError(
                "External agents require an ordinary task without a retry series or work contract."
            )
        snapshot = await app.task_store.load_invocation_snapshot(task_id)
        if snapshot is None:
            raise ValueError("External task does not exist.")
        request = run_request_with_task_invocation(request, snapshot)
        request = request.model_copy(
            update={"labels": request.labels or dict(snapshot.invocation.access_labels)}
        )
        if snapshot.invocation.resource_access is not None:
            await access.enter_async_context(
                execution_access(
                    snapshot.invocation.resource_access,
                    app.resource_access_policy,
                    request.labels,
                    store=app.session_store,
                    session_id=session_id,
                )
            )
        # Persist only protected input; the original prompt stays in the owned SDK call.
        persisted = request.model_copy(
            update={
                "messages": [
                    Message.model_validate(
                        app._secret_redactor.redact_json(message.model_dump(mode="json"))
                    )
                    for message in (request.messages or [Message.text("user", prompt)])
                ],
                "metadata": app._secret_redactor.redact_json(request.metadata),
            }
        )
        session = await app.session_store.create(
            persisted,
            identity=SessionIdentity(
                provider_name=adapter.provider_name,
                model=adapter.model,
                runtime_name="external_agent",
            ),
        )
        session = await app.session_store.transition_status(
            session_id, from_statuses={SessionStatus.PENDING}, to_status=SessionStatus.RUNNING
        )
        _activate_owned_session_run_fence(session)
        app._session_control.register_active_control_task(session_id, owner)
        monitor = asyncio.create_task(observe_interrupt())
        recorder = ExternalAgentRecorder(app, session, request)
        attached = await app._session_tasks.start_task(
            task_id=task_id,
            session=session,
            worker_id=request.task_worker_id,
            lease_expires_at=request.task_lease_expires_at,
        )
        await app._session_control.execution_presence.ensure(session, task_id=attached.id)
        await recorder.emit(EventType.SESSION_STARTED, {"external_agent": adapter.name})
        await recorder.emit(EventType.TASK_STARTED, {"task_id": attached.id})
        output: Any = None
        completed = False
        context = ExternalAgentContext(
            session.id,
            prompt,
            request.execution_deadline,
            output_schema,
            authorize,
            recorder.check_budget,
            recorder.protect_secret,
            lambda value: recorder.redactor.redact_json(value) != value,
            recorder.record_model_usage,
            recorder.claim_model_usage,
        )
        authority = register_run(app, session, context, request.budget_limits)
        async with execution_deadline_scope(request.execution_deadline):
            request.execution_deadline.require_admission("external agent launch")
            await recorder.check_budget()
            launched = True
            async with aclosing(adapter.stream(context)) as stream:
                async for observation in stream:
                    if recorder.gateway_usage_authoritative:
                        await recorder.check_budget()
                        if (
                            observation.kind == "usage"
                            and observation.payload.get("cost_status") != "reported"
                        ):
                            continue
                    already_recorded = observation.id in recorder.seen
                    await recorder.record(observation)
                    if observation.kind == "result" and not already_recorded:
                        if completed:
                            raise ExternalAgentError("External adapter returned multiple results.")
                        completed = True
                        output = recorder.redactor.redact_json(observation.payload.get("output"))
            if not completed:
                raise ExternalAgentError("External adapter exited without a result.")
        await recorder.settle_usage()
        if context.pending_model_usage:
            raise ExternalAgentError("External run accounting requires reconciliation.")
        if not recorder.usage_observed:
            await recorder.record(ExternalAgentEvent("usage", {}, "runtime:missing-usage"))
        await recorder.finish_text()
        if output_schema is not None:
            Draft202012Validator(output_schema).validate(output)
        await join_cleanup(stop_monitor())
        await app.session_store.append_transcript_messages(
            session.id, [Message.text("assistant", str(app._secret_redactor.redact_json(output)))]
        )
        await app.session_store.transition_status(
            session.id, from_statuses={SessionStatus.RUNNING}, to_status=SessionStatus.COMPLETED
        )
        await recorder.emit(EventType.SESSION_COMPLETED, {"external_agent": adapter.name})
        await _settle_task(app, attached, request, error=None)
        return ExternalAgentResult(session.id, attached.id, output)
    except BaseException as failure:
        if isinstance(failure, asyncio.CancelledError) and failure.args == (
            "External session access was revoked.",
        ):
            failure = ResourceAccessDenied()
        if isinstance(failure, ExternalCleanupIncomplete) and authority is not None:

            async def retain_cleanup(cleanup_failure=failure) -> None:
                if recorder is not None:
                    with suppress(Exception):
                        await recorder.emit(
                            "custom.external_agent.cleanup_pending", {"external_agent": True}
                        )
                await settle_external_cleanup(cleanup_failure, authority.quiesced)

            try:
                await join_cleanup(retain_cleanup())
            except asyncio.CancelledError as cancellation:
                # Cleanup has joined before join_cleanup restores cancellation.
                # Retain it through terminal settlement as for adapter cancellation.
                failure = cancellation
                if cancellation.args == ("External session access was revoked.",):
                    failure = ResourceAccessDenied()
        operator_interrupted = False
        if (
            recorder is not None
            and recorder.budget_exhausted
            and not isinstance(failure, asyncio.CancelledError)
        ):
            failure = ExternalAgentBudgetExceeded("External agent budget cannot admit more work.")

        async def settle(exc: BaseException = failure) -> None:
            nonlocal operator_interrupted
            # The adapter stream has joined its cleanup. Retire the observer
            # before our own terminal transition can look like lost authority.
            await stop_monitor()
            if recorder is not None:
                await recorder.settle_usage()
            assert app.task_store is not None
            expired = request.execution_deadline.expired
            error = {
                "error": "ExecutionDeadlineExceeded" if expired else type(exc).__name__,
                "external_agent": adapter.name,
            }
            current = await app.session_store.load(session_id)
            operator_interrupted = (
                not expired
                and session is not None
                and current is not None
                and current.instance_id == session.instance_id
                and current.run_epoch == session.run_epoch
                and current.status is SessionStatus.INTERRUPTING
            )
            if current is not None and current.status is SessionStatus.COMPLETED:
                if attached is not None:
                    await _settle_task(app, attached, request, error=None)
                return
            cancelled = operator_interrupted or (
                isinstance(exc, asyncio.CancelledError) and not expired
            )
            if current is not None:
                await app.session_store.transition_status(
                    session_id,
                    from_statuses={
                        SessionStatus.PENDING,
                        SessionStatus.RUNNING,
                        SessionStatus.INTERRUPTING,
                    },
                    to_status=SessionStatus.INTERRUPTED if cancelled else SessionStatus.FAILED,
                )
            if recorder is not None:
                recorder.text_stream.abort()
                if launched and not recorder.gateway_usage_authoritative:
                    # An interrupted loop may have incurred an unreported call,
                    # even if earlier steps supplied complete usage.
                    await recorder.emit(
                        EventType.MODEL_COMPLETED,
                        {
                            "usage": {},
                            "external_agent": True,
                            "usage_outcome": "unknown_after_interruption",
                        },
                    )
                await recorder.emit(
                    EventType.SESSION_INTERRUPTED if cancelled else EventType.SESSION_FAILED, error
                )
            # A storage acknowledgement can fail after attaching the task.
            # Recover only the exact session and owner this entrance created.
            settling_task = attached
            if settling_task is None:
                candidate = await app.task_store.load_task(task_id)
                if (
                    candidate is not None
                    and candidate.session_id == session_id
                    and candidate.worker_id == request.task_worker_id
                ):
                    settling_task = candidate
            # Worker cancellation settlement belongs to the worker (including
            # restartable shutdown and operator cancellation).
            if settling_task is not None:
                if operator_interrupted and request.task_worker_id is not None:
                    async with managed_task_lease_mutation(
                        task_id=settling_task.id,
                        worker_id=request.task_worker_id,
                        handoff_id=settling_task.interrupted_handoff_id,
                        presented_lease_expires_at=request.task_lease_expires_at,
                    ) as lease:
                        assert lease is not None
                        await app.task_store.request_claimed_task_cancellation(
                            settling_task.id,
                            request.task_worker_id,
                            lease,
                            {"error": "ExternalAgentInterrupted"},
                        )
                elif not cancelled:
                    await _settle_task(app, settling_task, request, error=error)
                elif request.task_worker_id is None:
                    await app.task_store.cancel_task(settling_task.id)
            elif implicit_task:
                assert app.task_store is not None
                owned_task = await app.task_store.load_task(task_id)
                if owned_task is not None and owned_task.worker_id is None:
                    if cancelled:
                        await app.task_store.cancel_task(task_id)
                    else:
                        await app.task_store.fail_task(task_id, error)

        try:
            await join_cleanup(settle())
        except BaseException as settlement_error:
            if isinstance(failure, asyncio.CancelledError):
                raise failure from None
            raise ExternalAgentError(
                f"External cleanup failed ({type(settlement_error).__name__})."
            ) from None
        if operator_interrupted and (
            request.task_worker_id is not None or not isinstance(failure, asyncio.CancelledError)
        ):
            raise ExternalAgentError("External session was interrupted by its operator.") from None
        if isinstance(failure, (asyncio.CancelledError, ExternalAgentError)):
            raise failure from None
        raise ExternalAgentError(f"External harness failed ({type(failure).__name__}).") from None
    finally:

        async def release() -> None:
            await stop_monitor()
            if session is not None:
                await app._session_control.execution_presence.stop_and_wait(
                    session_id, run_epoch=session.run_epoch
                )

        try:
            await join_cleanup(release())
        finally:
            app._session_control.unregister_active_control_task(session_id, owner)
            _deactivate_session_run_fence(session_id)
            release_run(app, session_id)
            await access.aclose()
