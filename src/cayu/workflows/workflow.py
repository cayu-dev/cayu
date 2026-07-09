"""Workflow helpers for resumable fan-out, typed edges, and gated item loops.

Each ``step`` wraps one ``app.run`` and journals progress under the workflow run
id. Every context has an ``attempt_id``; the attempt fence stops older contexts
before they can continue journaling or running side effects.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._validation import copy_json_value, copy_label_map, require_clean_nonblank
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, MessageRole, TextPart, ToolCallPart
from cayu.core.thinking import ThinkingConfig
from cayu.core.workflows import Workflow, WorkflowSpec
from cayu.runtime import (
    BudgetLimit,
    CayuApp,
    IncompleteSessionRecoveryRequest,
    RetryPolicy,
    RunLimits,
    RunRequest,
    SessionStatus,
    StructuredOutputSpec,
    StructuredOutputStrategy,
)
from cayu.runtime.budgets import copy_request_budget_limits
from cayu.runtime.retry_policy import copy_retry_policy
from cayu.runtime.stop_policy import copy_run_limits
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    validate_structured_output_text,
    validate_structured_output_tool_arguments,
)
from cayu.workflows.journal import (
    WORKFLOW_ATTEMPT_EVENT_TYPE,
    WORKFLOW_JOURNAL_PROVIDER,
    EventStoreJournal,
    WorkflowJournal,
    WorkflowJournalContext,
    validate_workflow_journal_event_type,
)
from cayu.workflows.models import (
    GateOutcome,
    ParallelResult,
    StepError,
    StepFailure,
    StepResult,
    normalize_gate_outcome,
)

# gated_loop namespaces its per-item completion under this prefix so an item's
# resume key can never collide with an inner ``do`` step's own ``step_id``.
_GATED_ITEM_PREFIX = "gated-loop:"
_LIVE_CHILD_STATUSES = {
    SessionStatus.PENDING,
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
}
_MISSING = object()

JournalFactory = Callable[[WorkflowJournalContext], WorkflowJournal]


class DuplicateStepIdError(ValueError):
    """Raised when one workflow execution reuses a durable step id."""


class WorkflowSupersededError(RuntimeError):
    """Raised when a newer attempt has taken over the same workflow run id."""


class StepRunOptions(BaseModel):
    """Runtime controls forwarded from :func:`step` to the child ``RunRequest``.

    Workflow-owned lineage fields are derived from the journal anchor rather
    than exposed here, so parent/child budget roll-ups stay coherent.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    provider_name: str | None = None
    environment_name: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: StrictInt = Field(default=16, ge=1, le=256)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    thinking: ThinkingConfig | None = None
    task_id: str | None = None
    task_worker_id: str | None = None

    @field_validator("provider_name", "environment_name", "task_id", "task_worker_id")
    @classmethod
    def validate_optional_nonblank(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels", allow_reserved=False)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("limits")
    @classmethod
    def copy_limits(cls, value: RunLimits) -> RunLimits:
        return copy_run_limits(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("retry_policy")
    @classmethod
    def copy_retry_policy(cls, value: RetryPolicy | None) -> RetryPolicy | None:
        if value is None:
            return None
        return copy_retry_policy(value)

    @field_validator("thinking")
    @classmethod
    def validate_thinking(cls, value: ThinkingConfig | None) -> ThinkingConfig | None:
        if value is not None and type(value) is not ThinkingConfig:
            raise TypeError("thinking must be a ThinkingConfig instance.")
        return value

    @model_validator(mode="after")
    def validate_task_worker_handoff(self) -> StepRunOptions:
        if self.task_worker_id is not None and self.task_id is None:
            raise ValueError("StepRunOptions.task_worker_id requires task_id.")
        return self


def _default_journal_factory(
    context: WorkflowJournalContext,
) -> WorkflowJournal:
    return EventStoreJournal(
        context.session_store,
        context.session_id,
        context.workflow_name,
        event_emitter=context.emit_events,
    )


def _json_safe(value: Any) -> Any:
    """Coerce arbitrary gate detail into something an ``Event.payload`` accepts."""
    try:
        return copy_json_value(value, "detail")
    except ValueError:
        return str(value)


def _current_task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


class _StepRunEventState:
    def __init__(self) -> None:
        self.text_buffer: list[str] = []
        self.final_text = ""
        self.output: Any = None
        self.output_validated = False
        self.failure: str | None = None
        self.interrupted: str | None = None

    def record(self, event: Event) -> None:
        event_type = event.type
        if event_type == EventType.MODEL_STARTED:
            self.text_buffer = []
        elif event_type == EventType.MODEL_TEXT_DELTA:
            self.text_buffer.append(str(event.payload.get("delta", "")))
        elif event_type == EventType.MODEL_COMPLETED:
            self.final_text = "".join(self.text_buffer)
        elif event_type == EventType.STRUCTURED_OUTPUT_VALIDATED:
            self.output = event.payload.get("output")
            self.output_validated = True
        elif event_type == EventType.SESSION_FAILED:
            self.failure = str(event.payload.get("error", "session failed"))
        elif event_type == EventType.SESSION_INTERRUPTED:
            self.interrupted = str(event.payload.get("interruption_type") or "interrupted")


class WorkflowContext:
    """Per-run binding for helper state, journal access, and attempt fencing."""

    def __init__(
        self,
        *,
        app: CayuApp,
        spec: WorkflowSpec,
        session_id: str,
        journal: WorkflowJournal,
    ) -> None:
        self.app = app
        self.spec = spec
        self.session_id = require_clean_nonblank(session_id, "session_id")
        self.journal = journal
        self.attempt_id = uuid4().hex
        self._auto_loop_name_used = False
        self._gated_loop_names: set[str] = set()
        self._claimed_step_ids: set[str] = set()
        self._claimed_step_ids_lock = asyncio.Lock()
        self._fence_established = False
        self._fence_lock = asyncio.Lock()
        self._anchor_loaded = False
        self._anchor = None

    @property
    def workflow_name(self) -> str:
        return self.spec.name

    def next_gated_loop_name(self) -> str:
        """Return the one safe automatic loop namespace.

        Additional loops need explicit names because resume replays ``run`` from
        the top, and call-order-derived names can shift when control flow shifts.
        """
        if self._auto_loop_name_used:
            raise ValueError(
                "Only one gated_loop per workflow run may use the automatic "
                "name; pass an explicit name= to every additional gated_loop "
                "so resume namespaces do not depend on execution order."
            )
        self._auto_loop_name_used = True
        return "loop0"

    def _register_gated_loop_name(self, loop_name: str) -> None:
        if loop_name in self._gated_loop_names:
            raise ValueError(
                f"Duplicate gated_loop name {loop_name!r} in workflow "
                f"{self.workflow_name!r}. Each loop needs a distinct namespace "
                "or completed items cross-skip between loops."
            )
        self._gated_loop_names.add(loop_name)

    async def _check_fence(self, *, establish: bool = True) -> None:
        """Verify this context is still the newest attempt.

        Terminal and custom events use ``establish=False`` so an older context
        cannot become newest merely by trying to publish a late event.
        """
        if not self._fence_established:
            async with self._fence_lock:
                if not self._fence_established:
                    if establish:
                        # ``event`` stamps attempt_id into every payload; the
                        # marker needs nothing more.
                        await self.journal.append(self.event(WORKFLOW_ATTEMPT_EVENT_TYPE))
                        self._fence_established = True
                    else:
                        latest = await self.journal.latest_attempt_id()
                        if latest is not None and latest != self.attempt_id:
                            self._raise_superseded(latest)
                        if latest is None:
                            await self.journal.append(self.event(WORKFLOW_ATTEMPT_EVENT_TYPE))
                            self._fence_established = True
        latest = await self.journal.latest_attempt_id()
        if latest is not None and latest != self.attempt_id:
            self._raise_superseded(latest)

    def _raise_superseded(self, latest_attempt_id: str) -> None:
        raise WorkflowSupersededError(
            f"Workflow run {self.session_id!r} was taken over by a newer "
            f"attempt ({latest_attempt_id!r}); this attempt ({self.attempt_id!r}) "
            "stops to avoid double-executing steps."
        )

    async def _append_current_attempt_event(self, event: Event) -> None:
        """Append a public workflow event only while this attempt is current."""
        if await self.journal.append_current_attempt(event, attempt_id=self.attempt_id):
            return
        latest = await self.journal.latest_attempt_id()
        if latest is not None:
            self._raise_superseded(latest)
        raise WorkflowSupersededError(
            f"Workflow run {self.session_id!r} no longer owns attempt "
            f"{self.attempt_id!r}; event {str(event.type)!r} was not journaled."
        )

    async def _workflow_anchor(self):
        """Load the run's anchor session once; it never changes during a run."""
        if not self._anchor_loaded:
            self._anchor = await self.app.session_store.load(self.session_id)
            self._anchor_loaded = True
        return self._anchor

    def event(
        self,
        event_type: EventType | str,
        *,
        payload: dict[str, Any] | None = None,
        agent_name: str | None = None,
    ) -> Event:
        """Build a workflow/custom event stamped with this context's attempt id."""
        validate_workflow_journal_event_type(event_type)
        return Event(
            type=event_type,
            session_id=self.session_id,
            workflow_name=self.workflow_name,
            agent_name=agent_name,
            payload={**(payload or {}), "attempt_id": self.attempt_id},
        )

    async def _claim_step_id(self, step_id: str) -> None:
        """Reserve a durable step id for this in-process workflow execution."""
        async with self._claimed_step_ids_lock:
            if step_id in self._claimed_step_ids:
                raise DuplicateStepIdError(
                    _workflow_step_duplicate_message(
                        workflow_name=self.workflow_name,
                        step_id=step_id,
                    )
                )
            self._claimed_step_ids.add(step_id)

    async def _release_step_id(self, step_id: str) -> None:
        async with self._claimed_step_ids_lock:
            self._claimed_step_ids.discard(step_id)

    async def emit_custom_event(
        self,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
        agent_name: str | None = None,
    ) -> Event:
        """Journal + return one custom workflow event.

        ``custom.cayu.`` is reserved so user events cannot forge or mask attempt
        fence markers.
        """
        if not event_type.startswith("custom."):
            raise ValueError("Workflow custom events must use the custom. namespace.")
        if event_type.startswith("custom.cayu."):
            raise ValueError("The custom.cayu. namespace is reserved for cayu internals.")
        await self._check_fence(establish=False)
        event = self.event(event_type, payload=payload, agent_name=agent_name)
        await self._append_current_attempt_event(event)
        return event

    async def start(self, payload: dict[str, Any] | None = None) -> Event:
        """Journal + return the ``workflow.started`` event. Use ``yield await``."""
        await self._check_fence()
        event = self.event(
            EventType.WORKFLOW_STARTED,
            payload={"workflow": self.workflow_name, **(payload or {})},
        )
        await self._append_current_attempt_event(event)
        return event

    async def completed(self, payload: dict[str, Any] | None = None) -> Event:
        """Journal + return the ``workflow.completed`` event. Use ``yield await``."""
        await self._check_fence(establish=False)
        event = self.event(
            EventType.WORKFLOW_COMPLETED,
            payload={"workflow": self.workflow_name, **(payload or {})},
        )
        await self._append_current_attempt_event(event)
        return event


class WorkflowBase(Workflow):
    """Base class for workflows authored as async generators."""

    def __init__(
        self,
        app: CayuApp,
        *,
        spec: WorkflowSpec | None = None,
        journal_factory: JournalFactory | None = None,
    ) -> None:
        if not isinstance(app, CayuApp):
            raise TypeError("WorkflowBase requires a CayuApp.")
        resolved_spec = spec if spec is not None else getattr(type(self), "spec", None)
        if not isinstance(resolved_spec, WorkflowSpec):
            raise ValueError(
                "A WorkflowBase needs a WorkflowSpec — set a class-level `spec` "
                "or pass `spec=` to the constructor."
            )
        self.app = app
        self.spec = resolved_spec
        self._journal_factory = journal_factory or _default_journal_factory

    def context(self, session_id: str) -> WorkflowContext:
        """Build the per-run context (and its journal) for one workflow run."""
        session_id = require_clean_nonblank(session_id, "session_id")
        journal_context = WorkflowJournalContext(
            session_store=self.app.session_store,
            session_id=session_id,
            workflow_name=self.spec.name,
            emit_events=lambda events: self.app._emit_many(session_id, events),
        )
        journal = self._journal_factory(journal_context)
        return WorkflowContext(
            app=self.app,
            spec=self.spec,
            session_id=session_id,
            journal=journal,
        )


def _resolve_structured_output(
    schema: dict[str, Any] | None,
    structured_output: StructuredOutputSpec | None,
    step_id: str,
) -> StructuredOutputSpec | None:
    if schema is not None and structured_output is not None:
        raise ValueError("Pass either `schema=` or `structured_output=`, not both.")
    if structured_output is not None:
        return structured_output
    if schema is None:
        return None
    return StructuredOutputSpec(
        name=step_id,
        json_schema=schema,
        strategy=StructuredOutputStrategy.TOOL,
        max_retries=2,
        repair_prompt=(
            "Call the structured-output tool again with an `output` value that "
            "matches the schema exactly."
        ),
    )


def _workflow_step_duplicate_message(
    *,
    workflow_name: str,
    step_id: str,
) -> str:
    return (
        f"Duplicate step_id {step_id!r} in workflow {workflow_name!r}. "
        "Each logical step in one workflow execution needs a distinct step_id."
    )


def _workflow_lineage_from_anchor(
    ctx: WorkflowContext,
    anchor,
) -> tuple[str | None, str]:
    if (
        anchor is not None
        and anchor.provider_name == WORKFLOW_JOURNAL_PROVIDER
        and anchor.metadata.get("cayu.workflow") == ctx.workflow_name
    ):
        return ctx.session_id, anchor.causal_budget_id or ctx.session_id
    return None, ctx.session_id


class _StepInvocation(Coroutine[Any, Any, StepResult]):
    def __init__(
        self,
        ctx: WorkflowContext,
        *,
        agent: str,
        step_id: str,
        prompt: str | None,
        messages: list[Message] | None,
        schema: dict[str, Any] | None,
        structured_output: StructuredOutputSpec | None,
        session_id: str | None,
        run_options: StepRunOptions | None,
    ) -> None:
        self.ctx = ctx
        self.step_id = require_clean_nonblank(step_id, "step_id")
        if self.step_id.startswith(_GATED_ITEM_PREFIX):
            raise ValueError(
                f"step_id {self.step_id!r} uses the reserved {_GATED_ITEM_PREFIX!r} "
                "namespace; gated_loop journals its items there, and a matching "
                "step_id would make the loop silently skip that item on resume."
            )
        self.workflow_name = ctx.workflow_name
        self._agent = agent
        self._prompt = prompt
        self._messages = messages
        self._schema = schema
        self._structured_output = structured_output
        self._session_id = session_id
        self._run_options = run_options
        self._parallel_claimed = False
        self._coro: Coroutine[Any, Any, StepResult] | None = None

    async def claim_for_parallel(self) -> None:
        if self._coro is not None:
            raise TypeError("parallel() received a step() awaitable that was already started.")
        await self.ctx._claim_step_id(self.step_id)
        self._parallel_claimed = True

    async def release_parallel_claim(self) -> None:
        if self._parallel_claimed:
            await self.ctx._release_step_id(self.step_id)
            self._parallel_claimed = False

    def _ensure_coro(self) -> Coroutine[Any, Any, StepResult]:
        if self._coro is None:
            self._coro = _run_step(
                self.ctx,
                agent=self._agent,
                step_id=self.step_id,
                prompt=self._prompt,
                messages=self._messages,
                schema=self._schema,
                structured_output=self._structured_output,
                session_id=self._session_id,
                run_options=self._run_options,
                preclaimed_step_id=self._parallel_claimed,
            )
        return self._coro

    def send(self, value: Any) -> Any:
        return self._ensure_coro().send(value)

    def throw(self, *args: Any) -> Any:
        return self._ensure_coro().throw(*args)

    def close(self) -> None:
        if self._coro is not None:
            self._coro.close()

    def __await__(self):
        return self._ensure_coro().__await__()


def step(
    ctx: WorkflowContext,
    *,
    agent: str,
    step_id: str,
    prompt: str | None = None,
    messages: list[Message] | None = None,
    schema: dict[str, Any] | None = None,
    structured_output: StructuredOutputSpec | None = None,
    session_id: str | None = None,
    run_options: StepRunOptions | None = None,
) -> Coroutine[Any, Any, StepResult]:
    """Run one child agent session and return a typed :class:`StepResult`.

    Progress is journaled under the workflow run id. Child failures and
    interruptions raise :class:`StepError`; takeover raises
    :class:`WorkflowSupersededError`.
    """
    if not isinstance(ctx, WorkflowContext):
        raise TypeError("step() requires a WorkflowContext.")
    return _StepInvocation(
        ctx,
        agent=agent,
        step_id=step_id,
        prompt=prompt,
        messages=messages,
        schema=schema,
        structured_output=structured_output,
        session_id=session_id,
        run_options=run_options,
    )


async def _run_step(
    ctx: WorkflowContext,
    *,
    agent: str,
    step_id: str,
    prompt: str | None,
    messages: list[Message] | None,
    schema: dict[str, Any] | None,
    structured_output: StructuredOutputSpec | None,
    session_id: str | None,
    run_options: StepRunOptions | None,
    preclaimed_step_id: bool,
) -> StepResult:
    agent = require_clean_nonblank(agent, "agent")
    step_id = require_clean_nonblank(step_id, "step_id")
    if (prompt is None) == (messages is None):
        raise ValueError("Pass exactly one of `prompt=` or `messages=`.")
    if run_options is not None and type(run_options) is not StepRunOptions:
        raise TypeError("run_options must be a StepRunOptions instance.")
    opts = run_options or StepRunOptions()
    spec = _resolve_structured_output(schema, structured_output, step_id)
    run_messages = messages if messages is not None else [Message.text("user", prompt or "")]
    if not preclaimed_step_id:
        await ctx._claim_step_id(step_id)
    await ctx._check_fence()

    completed_child_session_id, started_child_session_id = await ctx.journal.step_replay_ids(
        step_id=step_id,
        attempt_id=ctx.attempt_id,
    )
    if completed_child_session_id is not None:
        completed_child = await ctx.app.session_store.load(completed_child_session_id)
        if completed_child is not None and completed_child.status == SessionStatus.COMPLETED:
            return await _step_result_from_completed_child(
                ctx,
                step_id=step_id,
                child_session_id=completed_child_session_id,
                structured_output=spec,
            )

    child_session_id = session_id
    if child_session_id is None and started_child_session_id is not None:
        started_child = await ctx.app.session_store.load(started_child_session_id)
        if started_child is None or started_child.status != SessionStatus.FAILED:
            if started_child is not None and started_child.status in _LIVE_CHILD_STATUSES:
                await ctx.app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id=started_child_session_id,
                        reason="workflow_step_replay_recovered_started_child",
                        metadata={"workflow": ctx.workflow_name, "step_id": step_id},
                    )
                )
            child_session_id = started_child_session_id
    if child_session_id is None:
        child_session_id = f"{ctx.session_id}:{step_id}:{uuid4().hex[:8]}"

    # Resume onto an already-started child reuses its journaled STARTED record;
    # appending a second one would leave unpaired started/completed events.
    if child_session_id != started_child_session_id:
        started_event = ctx.event(
            EventType.WORKFLOW_STEP_STARTED,
            agent_name=agent,
            payload={
                "step_id": step_id,
                "agent": agent,
                "child_session_id": child_session_id,
            },
        )
        if not await ctx.journal.append_step_started(
            started_event,
            attempt_id=ctx.attempt_id,
        ):
            (
                completed_child_session_id,
                started_child_session_id,
            ) = await ctx.journal.step_replay_ids(
                step_id=step_id,
                attempt_id=ctx.attempt_id,
            )
            if completed_child_session_id is not None:
                completed_child = await ctx.app.session_store.load(completed_child_session_id)
                if (
                    completed_child is not None
                    and completed_child.status == SessionStatus.COMPLETED
                ):
                    return await _step_result_from_completed_child(
                        ctx,
                        step_id=step_id,
                        child_session_id=completed_child_session_id,
                        structured_output=spec,
                    )
            raise WorkflowSupersededError(
                f"Workflow step {step_id!r} in run {ctx.session_id!r} was already "
                "reserved by another attempt; this attempt stops to avoid "
                "double-executing the child session."
            )
    # The shipped journal anchors a session under the workflow run id, ensured
    # by the attempt-fence append above. When present, record lineage so budget
    # roll-ups and parent/child views span the whole workflow; custom journals
    # that do not anchor a session still keep children budget-linked to the
    # workflow id.
    anchor = await ctx._workflow_anchor()
    parent_session_id, causal_budget_id = _workflow_lineage_from_anchor(ctx, anchor)
    request = RunRequest(
        agent_name=agent,
        session_id=child_session_id,
        messages=run_messages,
        max_steps=opts.max_steps,
        structured_output=spec,
        provider_name=opts.provider_name,
        environment_name=opts.environment_name,
        labels=opts.labels,
        metadata=opts.metadata,
        limits=opts.limits,
        budget_limits=opts.budget_limits,
        retry_policy=opts.retry_policy,
        thinking=opts.thinking,
        task_id=opts.task_id,
        task_worker_id=opts.task_worker_id,
        parent_session_id=parent_session_id,
        causal_budget_id=causal_budget_id,
    )

    existing_child = await ctx.app.session_store.load(child_session_id)
    if existing_child is not None:
        if existing_child.status == SessionStatus.COMPLETED:
            result = await _step_result_from_completed_child(
                ctx,
                step_id=step_id,
                child_session_id=child_session_id,
                structured_output=spec,
            )
            await _append_step_completed(ctx, agent, result)
            return result
        if existing_child.status == SessionStatus.INTERRUPTED:
            interrupted = await _latest_child_interruption_type(ctx, child_session_id)
            raise StepError(
                f"child session interrupted ({interrupted}); "
                f"resolve the pause on session {child_session_id!r} before re-running",
                step_id=step_id,
                session_id=child_session_id,
            )
        if existing_child.status == SessionStatus.FAILED:
            raise StepError(
                f"child session already failed: {child_session_id!r}",
                step_id=step_id,
                session_id=child_session_id,
            )
        raise StepError(
            f"child session {child_session_id!r} already exists with status "
            f"{existing_child.status.value}; wait for it to finish or use a distinct session id",
            step_id=step_id,
            session_id=child_session_id,
        )

    state = _StepRunEventState()
    run_stream = ctx.app.run(request)
    try:
        async for event in run_stream:
            state.record(event)
    except asyncio.CancelledError:
        aclose = getattr(run_stream, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.shield(aclose())
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(
                ctx.app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id=child_session_id,
                        reason="workflow_step_cancelled",
                        metadata={"workflow": ctx.workflow_name, "step_id": step_id},
                    )
                )
            )
        raise
    except StepError:
        raise
    except Exception as exc:
        raise StepError(
            str(exc),
            step_id=step_id,
            session_id=child_session_id,
        ) from exc

    if state.failure is not None:
        raise StepError(state.failure, step_id=step_id, session_id=child_session_id)
    if state.interrupted is not None:
        if _current_task_is_cancelling():
            raise asyncio.CancelledError()
        raise StepError(
            f"child session interrupted ({state.interrupted}); "
            f"resolve the pause on session {child_session_id!r} before re-running",
            step_id=step_id,
            session_id=child_session_id,
        )

    output = (
        await _resolve_structured_step_output(ctx, child_session_id, spec, state.output)
        if state.output_validated
        else state.output
    )
    result = StepResult(
        step_id=step_id,
        session_id=child_session_id,
        text=state.final_text,
        output=output,
    )
    await _append_step_completed(ctx, agent, result)
    return result


async def _append_step_completed(
    ctx: WorkflowContext,
    agent: str,
    result: StepResult,
) -> None:
    if not await ctx.journal.append_current_attempt(
        ctx.event(
            EventType.WORKFLOW_STEP_COMPLETED,
            agent_name=agent,
            payload={
                "step_id": result.step_id,
                "child_session_id": result.session_id,
                "has_output": result.has_output,
            },
        ),
        attempt_id=ctx.attempt_id,
    ):
        latest = await ctx.journal.latest_attempt_id()
        if latest is not None:
            ctx._raise_superseded(latest)
        raise WorkflowSupersededError(
            f"Workflow run {ctx.session_id!r} no longer owns attempt "
            f"{ctx.attempt_id!r}; step completion was not journaled."
        )


async def _step_result_from_completed_child(
    ctx: WorkflowContext,
    *,
    step_id: str,
    child_session_id: str,
    structured_output: StructuredOutputSpec | None,
) -> StepResult:
    state = _StepRunEventState()
    for event in await ctx.app.session_store.load_events(child_session_id):
        state.record(event)
    output = (
        await _resolve_structured_step_output(
            ctx,
            child_session_id,
            structured_output,
            state.output,
        )
        if state.output_validated
        else state.output
    )
    return StepResult(
        step_id=step_id,
        session_id=child_session_id,
        text=state.final_text,
        output=output,
    )


async def _resolve_structured_step_output(
    ctx: WorkflowContext,
    child_session_id: str,
    spec: StructuredOutputSpec | None,
    fallback: Any,
) -> Any:
    if spec is None:
        return fallback
    recovered = await _raw_structured_output_from_transcript(ctx, child_session_id, spec)
    if recovered is not _MISSING:
        return recovered
    return fallback


async def _raw_structured_output_from_transcript(
    ctx: WorkflowContext,
    child_session_id: str,
    spec: StructuredOutputSpec,
) -> Any:
    transcript = await ctx.app.session_store.load_transcript(child_session_id)
    latest: Any = _MISSING
    if spec.strategy == StructuredOutputStrategy.TOOL:
        for message in transcript:
            if message.role != MessageRole.ASSISTANT:
                continue
            for part in message.content:
                if isinstance(part, ToolCallPart) and part.tool_name == STRUCTURED_OUTPUT_TOOL_NAME:
                    validation = validate_structured_output_tool_arguments(part.arguments, spec)
                    if validation.valid:
                        latest = validation.output
        return latest

    for message in transcript:
        if message.role != MessageRole.ASSISTANT:
            continue
        text = "".join(part.text for part in message.content if isinstance(part, TextPart))
        if not text:
            continue
        validation = validate_structured_output_text(text, spec)
        if validation.valid:
            latest = validation.output
    return latest


async def _latest_child_interruption_type(ctx: WorkflowContext, child_session_id: str) -> str:
    events = await ctx.app.session_store.load_events(child_session_id)
    for event in reversed(events):
        if event.type == EventType.SESSION_INTERRUPTED:
            return str(event.payload.get("interruption_type") or "interrupted")
    return "interrupted"


async def parallel(steps: Iterable[Awaitable[StepResult]]) -> ParallelResult:
    """Run steps concurrently and collect outcomes in submission order.

    Branch failures become :class:`StepFailure`, but duplicate-step and takeover
    signals propagate because the whole workflow must stop.
    """
    step_list = list(steps)
    await _preclaim_parallel_step_ids(step_list)
    tasks = [asyncio.ensure_future(step) for step in step_list]
    if not tasks:
        return ParallelResult()

    task_indexes = {task: index for index, task in enumerate(tasks)}
    pending = set(tasks)
    gathered: list[StepResult | BaseException | None] = [None] * len(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                index = task_indexes[task]
                if task.cancelled():
                    # cancelling() counts cancel() requests. Zero means the
                    # step raised CancelledError itself (e.g. a leaked cancel
                    # scope) — that fails this one branch; nobody asked the
                    # fan-out to cancel, so healthy siblings keep going. A
                    # requested cancellation propagates to the whole fan-out.
                    if task.cancelling() == 0:
                        invocation = step_list[index]
                        converted = StepError(
                            "step raised CancelledError without being cancelled",
                            step_id=invocation.step_id
                            if isinstance(invocation, _StepInvocation)
                            else None,
                        )
                        converted.__cause__ = asyncio.CancelledError()
                        gathered[index] = converted
                        continue
                    raise asyncio.CancelledError()
                try:
                    gathered[index] = task.result()
                except (DuplicateStepIdError, WorkflowSupersededError):
                    # Stop-the-workflow signals, not branch failures: a
                    # superseded run downgraded to a StepFailure would let a
                    # filter-and-continue caller keep double-running steps.
                    raise
                except BaseException as exc:
                    gathered[index] = exc
    finally:
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    results: list[StepResult | StepFailure] = []
    for outcome in gathered:
        if isinstance(outcome, StepResult):
            results.append(outcome)
        elif isinstance(outcome, StepError):
            results.append(
                StepFailure(
                    error=outcome.message,
                    error_type=type(outcome.__cause__ or outcome).__name__,
                    step_id=outcome.step_id,
                    session_id=outcome.session_id,
                )
            )
        elif isinstance(outcome, BaseException):
            results.append(StepFailure(error=str(outcome), error_type=type(outcome).__name__))
        else:  # pragma: no cover - gather only yields results or exceptions
            raise TypeError("parallel() steps must resolve to StepResult values.")
    return ParallelResult(results=tuple(results))


async def _preclaim_parallel_step_ids(steps: list[Awaitable[StepResult]]) -> None:
    acquired: list[_StepInvocation] = []
    seen: set[tuple[str, str, str]] = set()
    try:
        for candidate in steps:
            if not isinstance(candidate, _StepInvocation):
                continue
            key = (candidate.ctx.session_id, candidate.workflow_name, candidate.step_id)
            if key in seen:
                raise DuplicateStepIdError(
                    _workflow_step_duplicate_message(
                        workflow_name=candidate.workflow_name,
                        step_id=candidate.step_id,
                    )
                )
            seen.add(key)
            await candidate.claim_for_parallel()
            acquired.append(candidate)
    except BaseException:
        for invocation in reversed(acquired):
            await invocation.release_parallel_claim()
        _close_unscheduled_awaitables(steps)
        raise


def _close_unscheduled_awaitables(steps: list[Awaitable[StepResult]]) -> None:
    for candidate in steps:
        close = getattr(candidate, "close", None)
        if callable(close):
            close()


async def pipeline(
    initial: Any,
    stages: Iterable[Callable[[Any], Awaitable[StepResult]]],
) -> StepResult:
    """Chain stages, feeding each :class:`StepResult` into the next stage."""
    stage_list = list(stages)
    if not stage_list:
        raise ValueError("pipeline() requires at least one stage.")
    value: Any = initial
    result: StepResult | None = None
    for stage in stage_list:
        result = await stage(value)
        if not isinstance(result, StepResult):
            raise TypeError("Each pipeline stage must return a StepResult.")
        value = result
    assert result is not None
    return result


async def gated_loop(
    ctx: WorkflowContext,
    items: Iterable[Any],
    *,
    do: Callable[[Any], Awaitable[StepResult]],
    gate: Callable[[Any, StepResult], Awaitable[GateOutcome | bool]],
    on_pass: Callable[[Any, StepResult, GateOutcome], Awaitable[None]] | None = None,
    on_fail: Callable[[Any, StepResult, GateOutcome], Awaitable[None]] | None = None,
    key: Callable[[Any], str] | None = None,
    name: str | None = None,
):
    """Process items sequentially with resumable pass/fail journaling.

    Use stable item keys when the item list can change across resumes. Hooks run
    before the item completion is journaled, so ``on_pass`` and ``on_fail`` must
    tolerate at-least-once execution after a crash.
    """
    loop_name = (
        require_clean_nonblank(name, "gated_loop name")
        if name is not None
        else ctx.next_gated_loop_name()
    )
    ctx._register_gated_loop_name(loop_name)
    await ctx._check_fence()
    completed = await ctx.journal.completed_step_ids(attempt_id=ctx.attempt_id)
    seen_item_keys: set[str] = set()
    for index, item in enumerate(items):
        raw_key = key(item) if key is not None else str(index)
        item_key = require_clean_nonblank(raw_key, "gated_loop key")
        if item_key in seen_item_keys:
            raise ValueError(f"Duplicate gated_loop key {item_key!r}.")
        seen_item_keys.add(item_key)
        journal_id = f"{_GATED_ITEM_PREFIX}{loop_name}:{item_key}"
        if journal_id in completed:
            continue

        await ctx._check_fence()
        started_event = ctx.event(
            EventType.WORKFLOW_STEP_STARTED,
            payload={"step_id": journal_id, "item_key": item_key, "kind": "gated_loop"},
        )
        if not await ctx.journal.append_step_started(
            started_event,
            attempt_id=ctx.attempt_id,
        ):
            latest = await ctx.journal.latest_attempt_id()
            if latest is not None:
                ctx._raise_superseded(latest)
            raise WorkflowSupersededError(
                f"Workflow loop item {journal_id!r} in run {ctx.session_id!r} was already reserved."
            )
        yield started_event

        result = await do(item)
        if not isinstance(result, StepResult):
            raise TypeError("gated_loop `do` must return a StepResult.")
        outcome = normalize_gate_outcome(await gate(item, result))

        if outcome.passed:
            if on_pass is not None:
                await on_pass(item, result, outcome)
            verdict = "pass"
        else:
            if on_fail is not None:
                await on_fail(item, result, outcome)
            verdict = "fail"

        completed_event = ctx.event(
            EventType.WORKFLOW_STEP_COMPLETED,
            payload={
                "step_id": journal_id,
                "item_key": item_key,
                "kind": "gated_loop",
                "passed": outcome.passed,
                "outcome": verdict,
                "detail": _json_safe(outcome.detail),
                "child_session_id": result.session_id,
            },
        )
        if not await ctx.journal.append_current_attempt(
            completed_event,
            attempt_id=ctx.attempt_id,
        ):
            latest = await ctx.journal.latest_attempt_id()
            if latest is not None:
                ctx._raise_superseded(latest)
            raise WorkflowSupersededError(
                f"Workflow loop item {journal_id!r} in run {ctx.session_id!r} "
                "no longer owns the current attempt."
            )
        yield completed_event
