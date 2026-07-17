"""API routes for the cayu server."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal
from unicodedata import category as unicode_category
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from sse_starlette.sse import EventSourceResponse

if TYPE_CHECKING:
    from starlette.types import Receive, Scope, Send

from cayu._validation import (
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
    require_durable_json_text,
    require_unicode_scalar_text,
)
from cayu.artifacts import (
    ArtifactListResult,
    ArtifactScope,
    ArtifactStore,
    ArtifactStoreUnavailableError,
    InvalidArtifactIdError,
    copy_artifact_read_result,
)
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, MessageRole
from cayu.core.thinking import ThinkingConfig
from cayu.runtime.approvals import (
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.runtime.budgets import BudgetLimit, copy_request_budget_limits
from cayu.runtime.costs import (
    CausalBudgetCostSummary,
    PriceBook,
    SessionCostSummary,
)
from cayu.runtime.costs import (
    estimate_causal_budget_cost as build_causal_budget_cost_summary,
)
from cayu.runtime.costs import (
    estimate_session_cost as build_session_cost_summary,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.sessions import (
    SESSION_MESSAGE_CONTENT_MAX_BYTES,
    CompactSessionRequest,
    EnqueueSessionMessageRequest,
    EventOrder,
    EventQuery,
    EventRecord,
    InterruptSessionRequest,
    LabelSelectorOperator,
    LabelSelectorRequirement,
    PendingActionKind,
    PendingActionQuery,
    PendingActionResultTooLarge,
    PendingActionSession,
    ResumeRequest,
    RunRequest,
    Session,
    SessionDebugState,
    SessionMessageDeliveryMode,
    SessionOrder,
    SessionOutcome,
    SessionQuery,
    SessionStatus,
    TranscriptQuery,
    decode_session_cursor,
    event_summary_from_records,
    session_outcome_from_records,
)
from cayu.runtime.stop_policy import RunLimits
from cayu.runtime.structured_output import StructuredOutputSpec
from cayu.runtime.tasks import Task, TaskCreate, TaskOrder, TaskQuery, TaskStatus
from cayu.runtime.tool_rounds import ToolRoundRecoveryRequest
from cayu.runtime.usage import (
    CacheUsageMetrics,
    CausalBudgetUsageSummary,
    SessionUsageSummary,
    UsageMetrics,
    causal_budget_usage_summary,
    usage_metrics_from_event_payload,
)
from cayu.runtime.user_input import UserInputRecoveryRequest, UserInputResponse
from cayu.server.auth import AuthContext, AuthDependency, server_auth_dependency
from cayu.server.contracts import (
    ARTIFACT_CONTENT_ENDPOINT_RESPONSES,
    ARTIFACT_ENDPOINT_ERROR_RESPONSES,
    PENDING_ACTION_ENDPOINT_RESPONSES,
    SERVER_API_PREFIX,
    STREAMING_ENDPOINT_RESPONSES,
    AgentsResponse,
    ApiReviewedKnowledgeEntry,
    ApiSession,
    ApiTaskDetail,
    ApiTaskListItem,
    ArtifactReadResponse,
    ArtifactsResponse,
    CausalBudgetSummaryResponse,
    ClientGenerationContract,
    EnvironmentsResponse,
    HealthResponse,
    ListSessionEventsResponse,
    ListSessionsResponse,
    PendingActionsResponse,
    PendingKnowledgeDetailResponse,
    PendingKnowledgeListResponse,
    ServerContractResponse,
    SessionsSummaryResponse,
    SessionStateResponse,
    SessionSummaryResponse,
    SessionTranscriptResponse,
    UsageBreakdownItem,
)
from cayu.server.sse import (
    SSE_ERROR_TEXT_MAX_BYTES,
    SSE_OBSERVER_MAX_BYTES,
    SSE_OBSERVER_MAX_FRAMES,
    SSE_REPLAY_PAGE_EVENTS,
    SSE_REPLAY_START_MARKER_FORMAT,
    SSE_SEND_TIMEOUT_SECONDS,
    SseErrorCode,
    SseErrorKind,
    SseEventFrameTooLargeError,
    SseObserverLaggedError,
    error_to_sse_message,
    event_to_sse_message,
    parse_last_event_id,
    sse_message_data_bytes,
)
from cayu.storage import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeListItem,
    KnowledgeReviewWorkflow,
    KnowledgeVisibility,
)

logger = logging.getLogger(__name__)


class _ObserverLifecycleEventSourceResponse(EventSourceResponse):
    """Release a waiting detached pump even if response startup fails."""

    def __init__(
        self,
        *args: Any,
        observer_started: asyncio.Event,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._observer_started = observer_started

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._observer_started.set()


_MutationAcceptanceStage = Literal[
    "empty",
    "before_first_event",
    "after_first_event",
    "accepted",
]

NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PersistableNonBlankString = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[^\x00]+$",
    ),
]
ReplaySafeSessionId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._~-]*$",
    ),
]
MutationIdHeader = Annotated[
    str | None,
    Header(
        alias="Cayu-Mutation-ID",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._~-]*$",
        description=(
            "Client-generated mutation identity used to correlate an ambiguous "
            "SSE reconnect with its durable server acceptance event. Send the same "
            "value on the initial mutation and every Last-Event-ID replay request; "
            "it is a replay correlation key, not permission to repeat the POST."
        ),
    ),
]
ArtifactIdPath = Annotated[str, StringConstraints(min_length=1)]
# Server-entrypoint step budget. The default preserves the historical value while the
# ceiling matches the runtime's own ``max_steps`` bound (RunRequest/ResumeRequest and the
# tool-approval bodies all cap at 256) so a request cannot ask for an unbounded run.
_DEFAULT_RUN_MAX_STEPS = 20
_MAX_RUN_STEPS = 256
_EVENT_PAGE_LIMIT_MAX = 1000
_TRANSCRIPT_PAGE_LIMIT_MAX = 1000
_ARTIFACT_PAGE_LIMIT_MAX = 500
_ARTIFACT_PAGE_OFFSET_MAX = 10_000
_ARTIFACT_CONTENT_BYTES_MAX = 64 * 1024 * 1024
_ARTIFACT_FILENAME_HEADER_UTF8_MAX_BYTES = 512
_ARTIFACT_FILENAME_HEADER_ASCII_MAX_CHARS = 255
_ARTIFACT_ID_HEADER_MAX_CHARS = 512
_ARTIFACT_UNSAFE_FILENAME_UNICODE_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
_KNOWLEDGE_REVIEW_PREVIEW_CHARS = 1200
_KNOWLEDGE_PENDING_DETAIL_MAX_CHUNKS = 50
_KNOWLEDGE_PENDING_DETAIL_MAX_BYTES = 128_000
_SERVER_INTERRUPTIBLE_SESSION_STATUSES = {
    SessionStatus.PENDING,
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
    SessionStatus.INTERRUPTED,
}
_REPLAY_ACTIVE_SESSION_STATUSES = {
    SessionStatus.PENDING,
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
}
_REPLAY_TERMINAL_EVENT_BY_STATUS = {
    SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
    SessionStatus.FAILED: EventType.SESSION_FAILED,
    SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}
_REPLAY_TERMINAL_EVENT_TYPES = frozenset(_REPLAY_TERMINAL_EVENT_BY_STATUS.values())
_REPLAY_TERMINAL_LINEAGE_EVENT_TYPES = {
    EventType.HOOK_STARTED,
    EventType.HOOK_COMPLETED,
    EventType.HOOK_FAILED,
}
_REPLAY_POST_TERMINAL_EVENT_TYPES = {
    EventType.SERVER_MUTATION_ACCEPTED,
    EventType.SESSION_INTERRUPTION_CASCADE_RETRY_REQUESTED,
    EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED,
    EventType.SESSION_INTERRUPTION_CASCADE_FAILED,
}
_REPLAY_OPERATION_START_EVENT_TYPES = {
    EventType.SESSION_STARTED,
    EventType.SESSION_RESUMED,
    EventType.ENVIRONMENT_FACTORY_STARTED,
    EventType.ENVIRONMENT_BINDING_STARTED,
}
# Replays check quickly after reconnect, then back off while a live session is
# quiet so idle streams do not continuously hammer the durable stores.
_REPLAY_POLL_INTERVAL_MIN_S = 0.05
_REPLAY_POLL_INTERVAL_MAX_S = 1.0


def _next_replay_poll_interval(current: float, *, received_events: bool) -> float:
    if received_events:
        return _REPLAY_POLL_INTERVAL_MIN_S
    return min(current * 2, _REPLAY_POLL_INTERVAL_MAX_S)


def _replay_terminal_boundary_id(event: Event) -> str | None:
    """Return the terminal session event represented by an observed record."""
    if event.type in _REPLAY_TERMINAL_EVENT_TYPES:
        return event.id
    if event.type not in _REPLAY_TERMINAL_LINEAGE_EVENT_TYPES:
        return None
    terminal_event_type = event.payload.get("terminal_event_type")
    if terminal_event_type not in _REPLAY_TERMINAL_EVENT_TYPES:
        return None
    terminal_event_id = event.payload.get("terminal_event_id")
    if type(terminal_event_id) is str and terminal_event_id.strip():
        return terminal_event_id
    return None


# Detached event pumps must outlive their SSE consumer (a client disconnect must not
# cancel agent work), so hold strong references until each pump finishes — the event
# loop only keeps weak references to tasks.
_detached_event_pumps: set[asyncio.Task[None]] = set()
_ARTIFACT_SAFE_INLINE_CONTENT_TYPES = frozenset(
    {
        "application/json",
        "application/pdf",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
        "text/csv",
        "text/markdown",
        "text/plain",
    }
)


def _redacted_stream_error_text(cayu_app: Any, error: BaseException) -> str:
    """Return a secret-redacted error string, or a non-sensitive fallback."""
    fallback = f"{type(error).__name__}: stream failed."
    try:
        redacted = cayu_app.redact_json(str(error))
    except Exception:
        return fallback
    if type(redacted) is not str:
        return fallback
    return redacted


def _stream_error_sse_message(
    cayu_app: Any,
    error: BaseException,
    *,
    kind: SseErrorKind,
    code: SseErrorCode,
    retryable: bool,
    session_id: str,
) -> dict[str, str]:
    return error_to_sse_message(
        error,
        kind=kind,
        code=code,
        retryable=retryable,
        session_id=session_id,
        error_text=_redacted_stream_error_text(cayu_app, error),
    )


async def _close_event_stream(event_stream: AsyncIterator[Event]) -> None:
    close = getattr(event_stream, "aclose", None)
    if close is not None:
        # Cleanup is best-effort and must not replace the primary stream outcome.
        # CancelledError is a BaseException, so suppress it explicitly rather than
        # letting a secondary close failure bypass the observer's terminal signal.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await close()


def _preaccept_error_detail(cayu_app: Any, error: BaseException) -> str:
    text = _redacted_stream_error_text(cayu_app, error)
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= SSE_ERROR_TEXT_MAX_BYTES:
        return encoded.decode("utf-8")
    suffix = b"..."
    prefix = encoded[: SSE_ERROR_TEXT_MAX_BYTES - len(suffix)].decode("utf-8", errors="ignore")
    return prefix + suffix.decode()


def _log_mutation_acceptance_failure(
    cayu_app: Any,
    error: BaseException,
    *,
    session_id: str,
    stage: str,
) -> None:
    """Record a bounded, redacted diagnostic for a sanitized HTTP 500."""
    with contextlib.suppress(Exception):
        logger.error(
            "SSE mutation acceptance failed: stage=%s session_id=%r error_type=%s error=%r",
            stage,
            session_id,
            type(error).__name__,
            _preaccept_error_detail(cayu_app, error),
        )


async def _accepted_event_stream_response(
    event_stream: AsyncIterator[Event],
    *,
    cayu_app: Any,
    session_id: str,
    after_accept: Callable[[Event], Awaitable[None]] | None = None,
    conflict_error_types: tuple[type[Exception], ...] = (ValueError,),
) -> EventSourceResponse:
    """Establish one durable event before accepting an SSE mutation response.

    Reconnects are allowed only after the mutation has a durable replay boundary.
    Advancing the runtime here also makes a new session's store insert the atomic
    identity claim before route-owned task creation or an HTTP 200 response.
    """
    loop = asyncio.get_running_loop()
    acceptance: asyncio.Future[tuple[BaseException | None, _MutationAcceptanceStage]] = (
        loop.create_future()
    )
    observer_started = asyncio.Event()

    response, pump_task, abandon_observer = _start_detached_event_stream_response(
        event_stream,
        cayu_app=cayu_app,
        session_id=session_id,
        acceptance=acceptance,
        after_first_event=after_accept,
        observer_started=observer_started,
    )
    try:
        acceptance_error, stage = await asyncio.shield(acceptance)
    except asyncio.CancelledError:
        # Starting the detached driver transfers runtime ownership away from the
        # request task. A client timeout/disconnect must abandon only its observer;
        # cancelling the driver is reserved for Cayu's interruption protocol.
        abandon_observer()
        raise
    except BaseException:
        pump_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await pump_task
        raise

    if acceptance_error is None:
        return response

    with contextlib.suppress(asyncio.CancelledError, Exception):
        await pump_task
    if not isinstance(acceptance_error, Exception):
        raise acceptance_error
    if stage == "empty":
        _log_mutation_acceptance_failure(
            cayu_app,
            acceptance_error,
            session_id=session_id,
            stage="before_first_event",
        )
        raise HTTPException(
            status_code=500,
            detail="Mutation completed without producing a durable event.",
        ) from acceptance_error
    if stage == "after_first_event":
        _log_mutation_acceptance_failure(
            cayu_app,
            acceptance_error,
            session_id=session_id,
            stage=stage,
        )
        raise HTTPException(
            status_code=500,
            detail="Mutation setup failed after its durable acceptance event.",
        ) from acceptance_error
    if isinstance(acceptance_error, KeyError):
        raise HTTPException(
            status_code=404,
            detail=_preaccept_error_detail(cayu_app, acceptance_error),
        ) from acceptance_error
    if isinstance(acceptance_error, conflict_error_types):
        raise HTTPException(
            status_code=409,
            detail=_preaccept_error_detail(cayu_app, acceptance_error),
        ) from acceptance_error
    _log_mutation_acceptance_failure(
        cayu_app,
        acceptance_error,
        session_id=session_id,
        stage=stage,
    )
    raise HTTPException(
        status_code=500,
        detail="Mutation failed before streaming began.",
    ) from acceptance_error


def _detached_event_stream_response(
    event_stream: AsyncIterator[Event],
    *,
    cayu_app: Any,
    session_id: str,
) -> EventSourceResponse:
    response, _pump_task, _abandon_observer = _start_detached_event_stream_response(
        event_stream,
        cayu_app=cayu_app,
        session_id=session_id,
    )
    return response


def _start_detached_event_stream_response(
    event_stream: AsyncIterator[Event],
    *,
    cayu_app: Any,
    session_id: str,
    acceptance: asyncio.Future[tuple[BaseException | None, _MutationAcceptanceStage]] | None = None,
    after_first_event: Callable[[Event], Awaitable[None]] | None = None,
    observer_started: asyncio.Event | None = None,
) -> tuple[EventSourceResponse, asyncio.Task[None], Callable[[], None]]:
    """Run ``event_stream`` to completion in a detached task; stream it as an observer.

    The run is driven by the pump task, not by the SSE consumer: a client disconnect
    stops the observer while the session still runs to a terminal state (finalized,
    not stranded RUNNING). Each frame carries a resumable ``id:`` field, and a runtime
    failure surfaces as a terminal structured ``error`` frame instead of an aborted
    connection.
    """
    # Reserve one slot for the terminal done/error signal in addition to the
    # advertised data-frame window. A producer that emits exactly the maximum
    # number of data frames must still be able to close cleanly.
    queue: asyncio.Queue[tuple[str, Any, int]] = asyncio.Queue(maxsize=SSE_OBSERVER_MAX_FRAMES + 1)
    queued_bytes = 0
    queued_event_frames = 0
    observer_accepting = True
    observer_terminal = False

    def discard_queued_items() -> None:
        nonlocal queued_bytes, queued_event_frames
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        queued_bytes = 0
        queued_event_frames = 0

    def force_observer_error(item: BaseException) -> None:
        nonlocal observer_terminal
        if not observer_accepting or observer_terminal:
            return
        observer_terminal = True
        discard_queued_items()
        queue.put_nowait(("observer_error", item, 0))

    def abandon_observer() -> None:
        nonlocal observer_accepting
        observer_accepting = False
        discard_queued_items()
        if observer_started is not None:
            observer_started.set()

    def observer_capacity_exceeded(kind: str, data_bytes: int) -> bool:
        return (
            (kind == "event" and queued_event_frames >= SSE_OBSERVER_MAX_FRAMES)
            or queue.full()
            or queued_bytes + data_bytes > SSE_OBSERVER_MAX_BYTES
        )

    async def enqueue(
        kind: str,
        item: Any,
        *,
        data_bytes: int = 0,
        terminal: bool = False,
    ) -> None:
        nonlocal observer_terminal, queued_bytes, queued_event_frames
        if not observer_accepting or observer_terminal:
            return
        if observer_capacity_exceeded(kind, data_bytes):
            # A synchronous event source can produce a full window without an
            # await. Give an already-running observer one scheduling turn to
            # consume queued work before deciding that it has actually lagged.
            # This does not wait for a slow network client or backpressure the
            # detached runtime beyond that single cooperative handoff.
            await asyncio.sleep(0)
            if not observer_accepting or observer_terminal:
                return
        if observer_capacity_exceeded(kind, data_bytes):
            force_observer_error(
                SseObserverLaggedError(
                    "The live observer fell behind its bounded buffer; "
                    "reconnect from the last received event or use durable history."
                )
            )
            return
        queue.put_nowait((kind, item, data_bytes))
        queued_bytes += data_bytes
        if kind == "event":
            queued_event_frames += 1
        if terminal:
            observer_terminal = True

    def resolve_acceptance(
        error: BaseException | None,
        stage: _MutationAcceptanceStage,
    ) -> None:
        if acceptance is not None and not acceptance.done():
            acceptance.set_result((error, stage))

    async def publish_event(event: Event) -> None:
        if not observer_accepting or observer_terminal:
            return
        try:
            message = event_to_sse_message(event)
        except SseEventFrameTooLargeError as exc:
            await enqueue("observer_error", exc, terminal=True)
            return
        await enqueue(
            "event",
            message,
            data_bytes=sse_message_data_bytes(message),
        )

    async def pump() -> None:
        saw_first_event = False
        deferred_cancellation: asyncio.CancelledError | None = None

        async def publish_without_losing_cancellation(event: Event) -> None:
            nonlocal deferred_cancellation
            while True:
                try:
                    await publish_event(event)
                    return
                except asyncio.CancelledError as exc:
                    # The runtime iterator is paused at its yield boundary here.
                    # Preserve the cancellation and retry the bounded observer write;
                    # the main loop reissues it only after resuming the iterator.
                    deferred_cancellation = exc

        async def finish_after_first_event_callback(event: Event) -> None:
            """Finish acceptance bookkeeping without swallowing an interrupt."""
            nonlocal deferred_cancellation
            if after_first_event is None:
                return
            callback_task = asyncio.ensure_future(after_first_event(event))
            while True:
                try:
                    await asyncio.shield(callback_task)
                except asyncio.CancelledError as exc:
                    if callback_task.cancelled():
                        raise RuntimeError(
                            "Mutation acceptance bookkeeping was cancelled."
                        ) from exc
                    # The runtime owns this task, so an operator interruption can
                    # arrive while route-owned task bookkeeping is in progress.
                    # Let that one-shot write finish, then redeliver cancellation
                    # after the runtime iterator has resumed past its yield.
                    deferred_cancellation = exc
                    continue
                return

        try:
            while True:
                if deferred_cancellation is not None:
                    current_task = asyncio.current_task()
                    if current_task is None:
                        raise deferred_cancellation
                    # Scheduling (rather than immediately throwing) lets anext()
                    # resume through wrapper yield boundaries first. Cancellation is
                    # then delivered at the runtime's next real await (factory,
                    # binding, provider stream, or interruption status check).
                    asyncio.get_running_loop().call_soon(current_task.cancel)
                    deferred_cancellation = None
                try:
                    event = await anext(event_stream)
                except StopAsyncIteration:
                    if not saw_first_event:
                        resolve_acceptance(
                            RuntimeError("Mutation completed without producing a durable event."),
                            "empty",
                        )
                    break

                is_first_event = not saw_first_event
                saw_first_event = True
                if is_first_event and acceptance is not None:
                    try:
                        await finish_after_first_event_callback(event)
                    except BaseException as exc:
                        resolve_acceptance(exc, "after_first_event")
                        await _close_event_stream(event_stream)
                        raise
                    else:
                        resolve_acceptance(None, "accepted")

                await publish_without_losing_cancellation(event)

                if (
                    is_first_event
                    and observer_started is not None
                    and deferred_cancellation is None
                ):
                    try:
                        await observer_started.wait()
                    except asyncio.CancelledError as exc:
                        deferred_cancellation = exc
        except asyncio.CancelledError as exc:
            if not saw_first_event:
                resolve_acceptance(exc, "before_first_event")
            await enqueue("done", None, terminal=True)
            raise
        except Exception as exc:
            if not saw_first_event:
                resolve_acceptance(exc, "before_first_event")
            await _close_event_stream(event_stream)
            await enqueue("runtime_error", exc, terminal=True)
        except BaseException as exc:
            if not saw_first_event:
                resolve_acceptance(exc, "before_first_event")
            await _close_event_stream(event_stream)
            await enqueue("done", None, terminal=True)
            raise
        else:
            await enqueue("done", None, terminal=True)

    pump_task = asyncio.create_task(pump())
    _detached_event_pumps.add(pump_task)
    pump_task.add_done_callback(_detached_event_pumps.discard)

    async def observe() -> AsyncIterator[dict[str, str]]:
        nonlocal observer_accepting, queued_bytes, queued_event_frames
        if observer_started is not None:
            observer_started.set()
        try:
            while True:
                kind, item, data_bytes = await queue.get()
                queued_bytes -= data_bytes
                if kind == "event":
                    queued_event_frames -= 1
                    yield item
                    continue
                if kind == "runtime_error":
                    yield _stream_error_sse_message(
                        cayu_app,
                        item,
                        kind="runtime",
                        code="runtime_failed",
                        retryable=False,
                        session_id=session_id,
                    )
                    return
                if kind == "observer_error":
                    frame_too_large = isinstance(item, SseEventFrameTooLargeError)
                    yield _stream_error_sse_message(
                        cayu_app,
                        item,
                        kind="observer",
                        code=("event_frame_too_large" if frame_too_large else "observer_lagged"),
                        retryable=not frame_too_large,
                        session_id=session_id,
                    )
                    return
                return
        finally:
            abandon_observer()

    if observer_started is None:
        response = EventSourceResponse(observe(), send_timeout=SSE_SEND_TIMEOUT_SECONDS)
    else:
        response = _ObserverLifecycleEventSourceResponse(
            observe(),
            send_timeout=SSE_SEND_TIMEOUT_SECONDS,
            observer_started=observer_started,
        )
    return response, pump_task, abandon_observer


class RunBody(BaseModel):
    prompt: NonBlankString
    session_id: ReplaySafeSessionId | None = Field(
        default=None,
        description=(
            "Optional caller-selected identity for replay-safe run observation. "
            f"Before the first event, reconnect with `{SSE_REPLAY_START_MARKER_FORMAT}` "
            "using this session ID."
        ),
    )
    agent: NonBlankString = "assistant"
    model: NonBlankString | None = None
    causal_budget_id: NonBlankString | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    max_steps: StrictInt = Field(default=_DEFAULT_RUN_MAX_STEPS, ge=1, le=_MAX_RUN_STEPS)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels", allow_reserved=False)


class ResumeBody(BaseModel):
    session_id: NonBlankString
    prompt: NonBlankString
    max_steps: StrictInt = Field(default=_DEFAULT_RUN_MAX_STEPS, ge=1, le=_MAX_RUN_STEPS)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)


class InterruptSessionBody(BaseModel):
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    requested_by: ResolutionActor | None = None


class CompactSessionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: PersistableNonBlankString = Field(max_length=256)
    expected_run_epoch: StrictInt = Field(ge=0)
    expected_transcript_cursor: StrictInt = Field(ge=0)
    instructions: NonBlankString | None = Field(default=None, max_length=4096)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    requested_by: ResolutionActor | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @model_validator(mode="after")
    def validate_durable_text(self) -> CompactSessionBody:
        require_durable_json_text(
            self.model_dump(mode="json", exclude={"requested_by"}),
            "CompactSessionBody",
        )
        if self.requested_by is not None:
            require_durable_json_text(
                self.requested_by.model_dump(mode="json", exclude={"claims"}),
                "CompactSessionBody.requested_by",
            )
        return self


class EnqueueSessionMessageBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: NonBlankString = Field(max_length=256)
    content: NonBlankString = Field(max_length=SESSION_MESSAGE_CONTENT_MAX_BYTES)
    delivery_mode: SessionMessageDeliveryMode
    requested_by: ResolutionActor | None = None

    @field_validator("content")
    @classmethod
    def validate_content_size(cls, value: str) -> str:
        require_durable_json_text(value, "content")
        if len(value.encode("utf-8")) > SESSION_MESSAGE_CONTENT_MAX_BYTES:
            raise ValueError(
                "content exceeds the maximum encoded size of "
                f"{SESSION_MESSAGE_CONTENT_MAX_BYTES} bytes."
            )
        return value

    @model_validator(mode="after")
    def validate_durable_text(self) -> EnqueueSessionMessageBody:
        require_durable_json_text(
            self.model_dump(mode="json", exclude={"requested_by"}),
            "EnqueueSessionMessageBody",
        )
        if self.requested_by is not None:
            require_durable_json_text(
                self.requested_by.model_dump(mode="json", exclude={"claims"}),
                "EnqueueSessionMessageBody.requested_by",
            )
        return self


class UpdateSessionLabelsBody(BaseModel):
    # Required + extra="forbid": a missing/typo'd key must 422, never silently replace
    # all labels with {} (these are full-replacement mutations).
    model_config = ConfigDict(extra="forbid")

    labels: dict[str, str]


class UpdateSessionMetadataBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any]


class SessionCostBody(BaseModel):
    pricing: PriceBook
    currency: NonBlankString = "USD"


class SessionsSummaryBody(BaseModel):
    pricing: PriceBook | None = None
    currency: NonBlankString = "USD"


class TaskHoldBody(BaseModel):
    reason: NonBlankString | None = None
    payload: dict[str, Any] | None = None


class ToolApprovalBody(BaseModel):
    """Body for resolving a pending tool approval.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default
    to ``None``: the resumed run inherits the original run's configuration
    persisted on the pending approval. Explicit values override it.
    """

    session_id: NonBlankString
    approval_id: NonBlankString
    decision: ToolApprovalDecision
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)


class ToolApprovalRecoveryBody(BaseModel):
    """Body for recovering an approved tool call with an unknown result.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default
    to ``None``: the resumed run inherits the original run's configuration
    persisted on the pending approval. Explicit values override it.
    """

    session_id: NonBlankString
    approval_id: NonBlankString
    tool_call_id: NonBlankString
    outcome: ToolApprovalRecoveryOutcome
    message: NonBlankString
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)


class ToolRoundRecoveryBody(BaseModel):
    """Body for recovering a crashed ordinary tool call with an operator outcome.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default
    to ``None``: the resumed run applies the runtime defaults (a pending tool
    round persists no run configuration). Explicit values override them.
    """

    session_id: NonBlankString
    round_id: NonBlankString
    tool_call_id: NonBlankString
    outcome: ToolApprovalRecoveryOutcome
    message: NonBlankString
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)


class UserInputResolveBody(BaseModel):
    """Body for answering a session paused by ``ask_user``.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default to ``None``: the
    resumed run inherits the original run's configuration persisted on the pending user input.
    """

    session_id: NonBlankString
    input_id: NonBlankString
    answer: NonBlankString
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)


class UserInputRecoveryBody(BaseModel):
    """Body for recovering a user-input round stuck on ``manual_recovery_required``.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default to ``None``: the
    resumed run inherits the original run's configuration persisted on the pending user input.
    """

    session_id: NonBlankString
    input_id: NonBlankString
    answer: NonBlankString
    tool_call_id: NonBlankString
    outcome: ToolApprovalRecoveryOutcome
    message: NonBlankString
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)


def _request_actor(
    auth_context: AuthContext | None,
    body_actor: ResolutionActor | None,
    *,
    field_name: Literal["requested_by", "resolved_by"],
) -> ResolutionActor | None:
    """Derive a typed operator actor for an authenticated control-plane route.

    With auth configured, provenance comes from the verified caller and a
    body-supplied actor is rejected loudly (mirroring the reserved ``cayu:``
    label rejection) — a silent override would let clients believe they
    recorded an actor the audit trail replaced. Dev-mode bodies are accepted
    but re-stamped ``source="request"``, so a request can never claim
    server-verified (``http_auth``) or system provenance.
    """
    if auth_context is not None:
        if body_actor is not None:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} is derived from the authenticated caller and "
                "cannot be supplied in the request body.",
            )
        try:
            return ResolutionActor(
                subject=auth_context.subject,
                tenant=auth_context.tenant,
                source=ResolutionActorSource.HTTP_AUTH,
                claims=auth_context.claims,
            )
        except ValueError as exc:
            # AuthContext.subject is unconstrained, so an auth backend can hand
            # back a reserved ``cayu:``-prefixed subject; surface that as a
            # clean 400 instead of an unhandled 500.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if body_actor is None:
        return None
    try:
        return ResolutionActor(
            subject=body_actor.subject,
            tenant=body_actor.tenant,
            source=ResolutionActorSource.REQUEST,
            claims=body_actor.claims,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _request_resolution_actor(
    auth_context: AuthContext | None,
    body_resolved_by: ResolutionActor | None,
) -> ResolutionActor | None:
    return _request_actor(auth_context, body_resolved_by, field_name="resolved_by")


def _request_interruption_actor(
    auth_context: AuthContext | None,
    body_requested_by: ResolutionActor | None,
) -> ResolutionActor | None:
    return _request_actor(auth_context, body_requested_by, field_name="requested_by")


def _serialize_event_record(record: EventRecord) -> dict[str, Any]:
    event = record.event
    return {
        "sequence": record.sequence,
        "id": event.id,
        "type": str(event.type),
        "session_id": event.session_id,
        "agent_name": event.agent_name,
        "environment_name": event.environment_name,
        "workflow_name": event.workflow_name,
        "tool_name": event.tool_name,
        "payload": event.payload,
        "timestamp": event.timestamp.isoformat(),
    }


def _serialize_session_outcome(outcome: SessionOutcome) -> dict[str, Any]:
    return {
        "session_id": outcome.session_id,
        "status": outcome.status.value,
        "reason": outcome.reason,
        "details": outcome.details,
        "retry": outcome.retry,
        "terminal_event": (
            None
            if outcome.terminal_event is None
            else _serialize_event_record(outcome.terminal_event)
        ),
        "latest_retry_event": (
            None
            if outcome.latest_retry_event is None
            else _serialize_event_record(outcome.latest_retry_event)
        ),
    }


def _serialize_session_base(session: Session | PendingActionSession) -> dict[str, Any]:
    # Shared list-view fields. The list endpoint omits the (potentially large,
    # unbounded) per-session metadata; callers fetch a single session to get it.
    return {
        "id": session.id,
        "status": session.status.value,
        "agent_name": session.agent_name,
        "provider_name": session.provider_name,
        "model": session.model,
        "parent_session_id": session.parent_session_id,
        "causal_budget_id": session.causal_budget_id,
        "runtime_name": session.runtime_name,
        "runtime_version": session.runtime_version,
        "environment_name": session.environment_name,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "labels": session.labels,
    }


def _serialize_session(session: Session) -> dict[str, Any]:
    return {**_serialize_session_base(session), "metadata": session.metadata}


def _redact_control_plane_json(cayu_app: Any, value: Any, field_name: str) -> Any:
    copied = copy_json_value(value, field_name)
    redactor = getattr(cayu_app, "redact_json", None)
    if callable(redactor):
        return redactor(copied)
    return copied


def _serialize_tool(cayu_app: Any, tool: Any) -> dict[str, Any]:
    effect = getattr(tool.effect, "value", str(tool.effect))
    return {
        "name": tool.name,
        "description": _redact_control_plane_json(cayu_app, tool.description, "description"),
        "input_schema": _redact_control_plane_json(cayu_app, tool.schema, "input_schema"),
        "parallel_safe": tool.parallel_safe,
        "effect": effect,
    }


def _serialize_agent(cayu_app: Any, agent: Any) -> dict[str, Any]:
    spec = agent.spec
    thinking = (
        None
        if spec.thinking is None
        else _redact_control_plane_json(cayu_app, spec.thinking.model_dump(mode="json"), "thinking")
    )
    tools = [_serialize_tool(cayu_app, tool) for tool in agent.tools.values()]
    return {
        "name": spec.name,
        "provider_name": spec.provider_name,
        "model": spec.model,
        "tool_count": len(tools),
        "tools": sorted(tools, key=lambda item: item["name"]),
        "metadata": _redact_control_plane_json(cayu_app, spec.metadata, "metadata"),
        "provider_options": _redact_control_plane_json(
            cayu_app,
            spec.provider_options,
            "provider_options",
        ),
        "thinking": thinking,
        "has_system_prompt": spec.system_prompt is not None and bool(spec.system_prompt.strip()),
    }


def _object_type_name(value: Any) -> str | None:
    if value is None:
        return None
    return type(value).__name__


def _object_id(value: Any) -> str | None:
    if value is None:
        return None
    object_id = getattr(value, "id", None)
    return object_id if isinstance(object_id, str) and object_id.strip() else None


def _workspace_instruction_summary(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return "inline"
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return "inline"
    mode = getattr(value, "mode", None)
    if isinstance(mode, str):
        return mode
    return type(value).__name__


def _serialize_environment(cayu_app: Any, record: Any) -> dict[str, Any]:
    environment = record.environment
    workspace = environment.workspace
    artifact_store = environment.artifact_store
    bound_workspace = record.bound_workspace
    bound_payload = None
    if bound_workspace is not None:
        bound_payload = {
            "source_workspace_id": _object_id(bound_workspace.source_workspace),
            "bound_workspace_id": _object_id(bound_workspace.workspace),
            "runner_type": _object_type_name(bound_workspace.runner),
            "path": bound_workspace.path,
            "metadata": _redact_control_plane_json(
                cayu_app,
                bound_workspace.metadata,
                "metadata",
            ),
        }
    return {
        "name": record.spec.name,
        "metadata": _redact_control_plane_json(cayu_app, record.spec.metadata, "metadata"),
        "is_factory": record.factory is not None,
        "workspace_id": _object_id(workspace),
        "artifact_store_id": _object_id(artifact_store),
        "runner_type": _object_type_name(environment.runner),
        "binding_type": _object_type_name(environment.binding),
        "vault_type": _object_type_name(environment.vault),
        "proxy_type": _object_type_name(environment.proxy),
        "knowledge_store_type": _object_type_name(environment.knowledge_store),
        "mcp_server_count": len(environment.mcp_servers),
        "workspace_instructions": _workspace_instruction_summary(
            environment.workspace_instructions
        ),
        "bound_workspace": bound_payload,
    }


def _artifact_stores_by_id(cayu_app: Any) -> dict[str, ArtifactStore]:
    stores: dict[str, ArtifactStore] = {}
    for record in cayu_app.list_environment_registrations():
        store = record.environment.artifact_store
        if isinstance(store, ArtifactStore):
            try:
                store_id = require_clean_nonblank(store.id, "artifact_store.id")
                store_id = require_unicode_scalar_text(store_id, "artifact_store.id")
            except ValueError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="An artifact store has an invalid id configuration.",
                ) from exc
            existing = stores.get(store_id)
            if existing is not None and existing is not store:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Multiple registered environments use the same artifact_store_id: "
                        f"{store_id}. Configure unique artifact store ids."
                    ),
                )
            stores[store_id] = store
    return stores


def _serialize_artifact(cayu_app: Any, metadata: Any, *, artifact_store_id: str) -> dict[str, Any]:
    return {
        "id": metadata.id,
        "artifact_store_id": artifact_store_id,
        "filename": metadata.filename,
        "content_type": metadata.content_type,
        "size_bytes": metadata.size_bytes,
        "scope": metadata.scope.value,
        "session_id": metadata.session_id,
        "agent_name": metadata.agent_name,
        "environment_name": metadata.environment_name,
        "created_at": metadata.created_at.isoformat(),
        "metadata": _redact_control_plane_json(cayu_app, metadata.metadata, "metadata"),
    }


def _artifact_sort_key(artifact: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(artifact["created_at"]),
        str(artifact["artifact_store_id"]),
        str(artifact["id"]),
    )


def _decode_artifact_text(content: bytes, content_type: str) -> str | None:
    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_content_type.startswith("text/") or normalized_content_type in {
        "application/json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
    }:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("utf-8", errors="replace")
    return None


def _artifact_read_preview(cayu_app: Any, read: Any) -> tuple[str, str | None]:
    text_preview = _decode_artifact_text(read.content, read.metadata.content_type)
    if text_preview is None:
        return base64.b64encode(read.content).decode("ascii"), None
    redacted_preview = _redact_control_plane_json(cayu_app, text_preview, "artifact.content")
    if not isinstance(redacted_preview, str):
        raise TypeError("Artifact text preview redaction must return a string.")
    return base64.b64encode(redacted_preview.encode("utf-8")).decode("ascii"), redacted_preview


def _artifact_content_disposition(filename: str, disposition: str) -> str:
    safe_filename = "".join(
        "_"
        if char in {"/", "\\"}
        or unicode_category(char) in _ARTIFACT_UNSAFE_FILENAME_UNICODE_CATEGORIES
        else char
        for char in filename
    ).strip()
    if not safe_filename:
        safe_filename = "artifact"
    safe_filename = _truncate_utf8_filename(
        safe_filename,
        max_bytes=_ARTIFACT_FILENAME_HEADER_UTF8_MAX_BYTES,
    )
    ascii_filename = "".join(
        char if 0x20 <= ord(char) < 0x7F and char not in {'"', "/", "\\"} else "_"
        for char in safe_filename
    ).strip()
    if not ascii_filename:
        ascii_filename = "artifact"
    ascii_filename = ascii_filename[:_ARTIFACT_FILENAME_HEADER_ASCII_MAX_CHARS]
    encoded_filename = quote(safe_filename, safe="", encoding="utf-8", errors="replace")
    return f"{disposition}; filename=\"{ascii_filename}\"; filename*=UTF-8''{encoded_filename}"


def _truncate_utf8_filename(value: str, *, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    stem, separator, extension = value.rpartition(".")
    if not separator:
        stem = value
    suffix = f".{extension}" if separator and 0 < len(extension) <= 32 else ""
    suffix_bytes = suffix.encode("utf-8")
    if len(suffix_bytes) >= max_bytes:
        suffix = ""
        suffix_bytes = b""
        stem = value
    prefix_bytes = stem.encode("utf-8")[: max_bytes - len(suffix_bytes)]
    prefix = prefix_bytes.decode("utf-8", errors="ignore")
    return f"{prefix}{suffix}" or "artifact"


def _artifact_content_disposition_kind(content_type: str, requested: str) -> str:
    if requested != "inline":
        return "attachment"
    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_content_type in _ARTIFACT_SAFE_INLINE_CONTENT_TYPES:
        return "inline"
    return "attachment"


def _artifact_header_value(value: str, fallback: str) -> str:
    for candidate in (value, fallback, "unknown"):
        stripped = candidate.strip()
        if stripped and all(0x20 <= ord(char) < 0x7F for char in stripped):
            return stripped[:_ARTIFACT_ID_HEADER_MAX_CHARS]
    return "unknown"


def _usage_breakdown(
    events: list[Event],
    *,
    key_fn: Callable[[UsageMetrics], tuple[str | None, str | None]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for event in events:
        if event.type != EventType.MODEL_COMPLETED:
            continue
        metrics = usage_metrics_from_event_payload(event.payload)
        if metrics is None:
            continue
        provider_name, model = key_fn(metrics)
        key = (provider_name, model)
        bucket = buckets.setdefault(
            key,
            {
                "provider_name": provider_name,
                "model": model,
                "session_ids": set(),
                "model_steps": 0,
                "usage": UsageMetrics(provider_name=provider_name, model=model),
            },
        )
        bucket["session_ids"].add(event.session_id)
        bucket["model_steps"] += 1
        bucket["usage"] = _add_usage_metrics(bucket["usage"], metrics)

    items = [
        UsageBreakdownItem(
            provider_name=provider_name,
            model=model,
            session_count=len(bucket["session_ids"]),
            model_steps=bucket["model_steps"],
            usage=bucket["usage"],
        ).model_dump()
        for (provider_name, model), bucket in buckets.items()
    ]
    return sorted(
        items,
        key=lambda item: (
            -item["usage"]["total_tokens"],
            item["provider_name"] or "",
            item["model"] or "",
        ),
    )


def _add_usage_metrics(left: UsageMetrics, right: UsageMetrics) -> UsageMetrics:
    return UsageMetrics(
        provider_name=left.provider_name,
        requested_model=left.requested_model or right.requested_model,
        model=left.model,
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        reasoning_output_tokens=left.reasoning_output_tokens + right.reasoning_output_tokens,
        cache=CacheUsageMetrics(
            read_tokens=left.cache.read_tokens + right.cache.read_tokens,
            write_tokens=left.cache.write_tokens + right.cache.write_tokens,
            write_5m_tokens=left.cache.write_5m_tokens + right.cache.write_5m_tokens,
            write_1h_tokens=left.cache.write_1h_tokens + right.cache.write_1h_tokens,
            write_unknown_ttl_tokens=(
                left.cache.write_unknown_ttl_tokens + right.cache.write_unknown_ttl_tokens
            ),
            cached_input_tokens=left.cache.cached_input_tokens + right.cache.cached_input_tokens,
            uncached_input_tokens=left.cache.uncached_input_tokens
            + right.cache.uncached_input_tokens,
        ),
    )


def _parse_session_label_filters(values: list[str] | None) -> dict[str, str]:
    if values is None:
        return {}
    labels: dict[str, str] = {}
    for raw in values:
        if type(raw) is not str or "=" not in raw:
            raise HTTPException(
                status_code=422,
                detail="Session label filters must use `key=value`.",
            )
        key, value = raw.split("=", 1)
        try:
            parsed = copy_label_map({key: value}, "label")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        parsed_key, parsed_value = next(iter(parsed.items()))
        if parsed_key in labels:
            raise HTTPException(
                status_code=422,
                detail=f"Duplicate session label filter: {parsed_key}",
            )
        labels[parsed_key] = parsed_value
    return labels


def _parse_session_label_selectors(
    values: list[str] | None,
) -> tuple[LabelSelectorRequirement, ...]:
    if values is None:
        return ()
    selectors: list[LabelSelectorRequirement] = []
    for raw in values:
        if type(raw) is not str:
            raise HTTPException(status_code=422, detail="Label selector must be a string.")
        for expression in _split_label_selector(raw):
            selectors.append(_parse_label_selector_expression(expression))
    return tuple(selectors)


def _split_label_selector(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise HTTPException(status_code=422, detail="Invalid label selector.")
        elif char == "," and depth == 0:
            part = value[start:index].strip()
            if not part:
                raise HTTPException(status_code=422, detail="Invalid label selector.")
            parts.append(part)
            start = index + 1
    if depth != 0:
        raise HTTPException(status_code=422, detail="Invalid label selector.")
    part = value[start:].strip()
    if not part:
        raise HTTPException(status_code=422, detail="Invalid label selector.")
    parts.append(part)
    return parts


def _parse_label_selector_expression(expression: str) -> LabelSelectorRequirement:
    try:
        if expression.startswith("!"):
            key = expression[1:].strip()
            return LabelSelectorRequirement(
                key=key,
                operator=LabelSelectorOperator.NOT_EXISTS,
            )
        if " notin " in expression:
            key, raw_values = expression.split(" notin ", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.NOT_IN,
                values=_parse_label_selector_values(raw_values),
            )
        if " in " in expression:
            key, raw_values = expression.split(" in ", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.IN,
                values=_parse_label_selector_values(raw_values),
            )
        if "!=" in expression:
            key, value = expression.split("!=", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.NOT_IN,
                values=(value.strip(),),
            )
        if "==" in expression:
            key, value = expression.split("==", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.IN,
                values=(value.strip(),),
            )
        if "=" in expression:
            key, value = expression.split("=", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.IN,
                values=(value.strip(),),
            )
        return LabelSelectorRequirement(
            key=expression.strip(),
            operator=LabelSelectorOperator.EXISTS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _parse_label_selector_values(raw_values: str) -> tuple[str, ...]:
    raw_values = raw_values.strip()
    if not raw_values.startswith("(") or not raw_values.endswith(")"):
        raise HTTPException(status_code=422, detail="Label selector values must use `(a,b)`.")
    values = tuple(value.strip() for value in raw_values[1:-1].split(","))
    if any(not value for value in values):
        raise HTTPException(status_code=422, detail="Label selector values cannot be blank.")
    return values


def _clean_optional_query_value(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    try:
        return require_clean_nonblank(value, field_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _trace_context_metadata(http_request: Request) -> dict[str, Any]:
    # Carry an inbound W3C trace context into the session metadata so an
    # OpenTelemetryEventSink can root the session span under the caller's trace.
    # Used as a shared dependency by every route that starts a traced session.
    metadata: dict[str, Any] = {}
    traceparent = http_request.headers.get("traceparent")
    if traceparent:
        metadata["traceparent"] = traceparent
        tracestate = http_request.headers.get("tracestate")
        if tracestate:
            metadata["tracestate"] = tracestate
    return metadata


TraceContextMetadata = Annotated[dict[str, Any], Depends(_trace_context_metadata)]


def _serialize_message_part(part: Any) -> dict[str, Any]:
    if part.type == "thinking":
        # The opaque round-trip state (Anthropic signatures / redacted blobs) is
        # provider-internal and must not be exposed to transcript API consumers.
        return part.model_dump(mode="json", exclude={"provider_state"})
    return part.model_dump(mode="json")


def _serialize_transcript_message(index: int, message: Message) -> dict[str, Any]:
    return {
        "index": index,
        "role": str(message.role),
        "content": [_serialize_message_part(part) for part in message.content],
    }


def _serialize_task_list_item(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "type": task.type,
        "title": task.title,
        "description": task.description,
        "status": task.status.value,
        "status_reason": task.status_reason,
        "status_payload": task.status_payload,
        "session_id": task.session_id,
        "parent_task_id": task.parent_task_id,
        "assigned_agent_name": task.assigned_agent_name,
        "worker_id": task.worker_id,
        "lease_expires_at": (task.lease_expires_at.isoformat() if task.lease_expires_at else None),
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }


def _serialize_task_detail(task: Task) -> dict[str, Any]:
    return {
        **_serialize_task_list_item(task),
        "input": task.input,
        "result": task.result,
        "error": task.error,
        "metadata": task.metadata,
        "started_at": task.started_at.isoformat() if task.started_at else None,
    }


def _serialize_knowledge_entry_base(entry: KnowledgeEntry) -> dict[str, Any]:
    return {
        "entry_id": entry.id,
        "namespace": entry.namespace,
        "kind": entry.kind,
        "visibility": entry.visibility.value,
        "status": entry.status.value,
        "title": entry.title,
        "labels": dict(entry.labels),
        "aspects": list(entry.aspects),
        "impact_targets": list(entry.impact_targets),
        "source_type": entry.source_type,
        "source_uri": entry.source_uri,
        "source_id": entry.source_id,
        "created_by_type": entry.created_by_type.value,
        "created_by": entry.created_by,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "importance": entry.importance,
        "importance_source": entry.importance_source,
        "confidence": entry.confidence,
    }


def _serialize_knowledge_list_item(item: KnowledgeListItem) -> dict[str, Any]:
    return {
        **_serialize_knowledge_entry_base(item.entry),
        "chunk_count": item.chunk_count,
        "text_preview": item.text_preview,
    }


def _serialize_reviewed_knowledge_entry(entry: KnowledgeEntry) -> dict[str, Any]:
    return {
        **_serialize_knowledge_entry_base(entry),
        "text_preview": _knowledge_text_preview(entry.text),
    }


def _serialize_knowledge_detail(entry: KnowledgeEntry) -> dict[str, Any]:
    return {
        **_serialize_knowledge_entry_base(entry),
        "text": entry.text,
        "metadata": entry.metadata,
        "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
    }


def _serialize_knowledge_chunk(chunk: KnowledgeChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "entry_id": chunk.entry_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "content_hash": chunk.content_hash,
        "source_uri": chunk.source_uri,
        "metadata": dict(chunk.metadata),
    }


def _knowledge_text_preview(text: str) -> str:
    if len(text) <= _KNOWLEDGE_REVIEW_PREVIEW_CHARS:
        return text
    return f"{text[:_KNOWLEDGE_REVIEW_PREVIEW_CHARS]}..."


def _parse_knowledge_label_filters(values: list[str] | None) -> dict[str, str]:
    if values is None:
        return {}
    labels: dict[str, str] = {}
    for raw in values:
        if type(raw) is not str or "=" not in raw:
            raise HTTPException(
                status_code=422,
                detail="Knowledge label filters must use `key=value`.",
            )
        key, value = raw.split("=", 1)
        try:
            parsed = copy_label_map({key: value}, "label")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        parsed_key, parsed_value = next(iter(parsed.items()))
        if parsed_key in labels:
            raise HTTPException(
                status_code=422,
                detail=f"Duplicate knowledge label filter: {parsed_key}",
            )
        labels[parsed_key] = parsed_value
    return labels


def _parse_knowledge_string_filters(values: list[str] | None, field_name: str) -> list[str]:
    if values is None:
        return []
    result: list[str] = []
    for value in values:
        try:
            result.append(require_clean_nonblank(value, field_name))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return list(dict.fromkeys(result))


def create_router(
    *,
    cayu_app,
    session_store,
    task_store,
    knowledge_store=None,
    knowledge_review_namespace: str | None = None,
    knowledge_review_labels: dict[str, str] | None = None,
    auth: AuthDependency | None = None,
    api_path: str = SERVER_API_PREFIX,
    openapi_url: str | None = "/openapi.json",
    replay_idle_timeout_s: float = 300.0,
) -> APIRouter:
    """Create an APIRouter with standard cayu endpoints.

    Args:
        auth: FastAPI-compatible dependency guarding the CAYU control plane.
            Production callers should pass this through ``create_server``; only
            explicit dev-mode callers should leave it unset. It protects every
            control-plane route that can start, change, inspect, or reveal runtime
            state; only the health route stays open for load balancers. It must
            return ``AuthContext`` or a compatible mapping and raise
            ``HTTPException`` (401/403) to deny a request. Authentication does not
            add tenant-level authorization or storage isolation:
            ``AuthContext.tenant`` is operator provenance only.
        api_path: URL path prefix for the CAYU control plane. Defaults to
            ``/api``.
        openapi_url: Public OpenAPI schema URL advertised by ``/contract`` for
            client generation. Pass ``None`` when generated OpenAPI is disabled.
        replay_idle_timeout_s: Maximum time an active replay stream may wait
            without seeing a new persisted event before emitting an error and closing.
    """

    if (
        isinstance(replay_idle_timeout_s, bool)
        or not isinstance(replay_idle_timeout_s, (int, float))
        or replay_idle_timeout_s <= 0
    ):
        raise ValueError("replay_idle_timeout_s must be a positive number.")
    replay_idle_timeout_s = float(replay_idle_timeout_s)

    api_prefix = _normalize_api_path(api_path)
    router = APIRouter(prefix=api_prefix)
    auth_context_openapi_schema = AuthContext.model_json_schema()

    # Shared dependency list for control-plane routes. FastAPI treats an empty
    # sequence like no dependencies, so `auth=None` keeps current dev behavior.
    auth_dependency = server_auth_dependency(auth) if auth is not None else None
    protected: list[Any] = [Depends(auth_dependency)] if auth_dependency is not None else []

    async def _optional_auth_context(request: Request) -> AuthContext | None:
        # The interruption and approval/user-input resolution routes take this as a
        # handler parameter INSTEAD of `dependencies=protected`: one callable
        # both guards the route (its 401/403 raises before the handler body)
        # and yields the verified caller identity for typed operator
        # provenance. Splitting guard and extraction into two differently
        # wrapped callables would invoke the user's auth dependency twice per
        # request (FastAPI caches per-callable, not per-underlying-auth).
        # Handlers must take this via the `= Depends(...)` default form: with
        # `from __future__ import annotations`, a function-local Annotated
        # alias is an unresolvable string annotation that FastAPI silently
        # degrades to a required query parameter.
        if auth_dependency is None:
            return None
        return await auth_dependency(request)

    optional_auth_context = Depends(_optional_auth_context)

    def _mutation_acceptance_callback(
        *,
        mutation_id: str | None,
        mutation_kind: str,
        session_id: str,
        after_accept: Callable[[Event], Awaitable[None]] | None = None,
    ) -> Callable[[Event], Awaitable[None]] | None:
        """Compose route bookkeeping with an exact durable mutation boundary."""
        if mutation_id is None:
            return after_accept

        async def record_acceptance(first_event: Event) -> None:
            # Route-owned setup remains part of acceptance. The mutation marker
            # is deliberately written last so it never claims that a request was
            # accepted when prerequisite setup failed.
            if after_accept is not None:
                await after_accept(first_event)
            if first_event.session_id != session_id:
                raise RuntimeError(
                    "Mutation acceptance event belongs to a different session: "
                    f"{first_event.session_id}"
                )
            await cayu_app.emit_event(
                Event(
                    type=EventType.SERVER_MUTATION_ACCEPTED,
                    session_id=session_id,
                    agent_name=first_event.agent_name,
                    environment_name=first_event.environment_name,
                    workflow_name=first_event.workflow_name,
                    tool_name=first_event.tool_name,
                    payload={
                        "mutation_id": mutation_id,
                        "mutation_kind": mutation_kind,
                        "accepted_event_id": first_event.id,
                        "accepted_event_type": str(first_event.type),
                    },
                )
            )

        return record_acceptance

    @router.get(
        "/contract",
        response_model=ServerContractResponse,
        dependencies=protected,
        description=(
            "Return the versioned Cayu server contract. Authentication controls access "
            "to protected routes, but AuthContext.tenant is actor provenance only and "
            "does not filter or isolate Cayu data."
        ),
        openapi_extra={"x-cayu-auth-context": auth_context_openapi_schema},
    )
    async def get_contract():
        return ServerContractResponse(
            api_prefix=api_prefix,
            client_generation=ClientGenerationContract(openapi_url=openapi_url),
        )

    async def _marker_record(session_id: str, event_id: str) -> EventRecord:
        """Persisted event named by a ``Last-Event-ID`` marker.

        Unknown event markers are rejected rather than silently widening the replay
        to full history. The explicit ``session_id:`` marker owns replay-from-start.
        """
        records = await session_store.query_events(
            EventQuery(session_id=session_id, event_id=event_id, limit=1)
        )
        if not records:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Last-Event-ID event was not found in the requested session. "
                    f"Use `{SSE_REPLAY_START_MARKER_FORMAT}` only for explicit "
                    "replay from the beginning."
                ),
            )
        return records[0]

    async def _marker_terminal_boundary_id(marker_record: EventRecord) -> str | None:
        """Resolve terminal lineage represented by a durable replay marker.

        Terminal events carry their own lineage. Terminal-hook telemetry names its
        terminal event explicitly, but that reference still has to resolve to a
        matching durable record in the same operation epoch. Interruption-cascade
        telemetry is also framework-owned and post-terminal, but does not carry an
        explicit reference, so resolve its latest prior terminal event. Reject both
        indirect associations if a later operation started before the marker.
        """
        marker_event = marker_record.event
        if marker_event.type in _REPLAY_TERMINAL_EVENT_TYPES:
            return marker_event.id

        terminal_record: EventRecord | None = None
        if marker_event.type in _REPLAY_TERMINAL_LINEAGE_EVENT_TYPES:
            referenced_terminal_id = _replay_terminal_boundary_id(marker_event)
            if referenced_terminal_id is None:
                return None
            terminal_records = await session_store.query_events(
                EventQuery(
                    session_id=marker_event.session_id,
                    event_id=referenced_terminal_id,
                    limit=1,
                )
            )
            if not terminal_records:
                return None
            candidate = terminal_records[0]
            if (
                candidate.sequence >= marker_record.sequence
                or candidate.event.type != marker_event.payload.get("terminal_event_type")
                or candidate.event.type not in _REPLAY_TERMINAL_EVENT_TYPES
            ):
                return None
            terminal_record = candidate
        elif marker_event.type in _REPLAY_POST_TERMINAL_EVENT_TYPES:
            terminal_records = await session_store.query_events(
                EventQuery(
                    session_id=marker_event.session_id,
                    event_types=tuple(sorted(_REPLAY_TERMINAL_EVENT_TYPES, key=str)),
                    before_sequence=marker_record.sequence,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            if terminal_records:
                terminal_record = terminal_records[0]
        else:
            return None

        if terminal_record is None:
            return None
        later_operation_starts = await session_store.query_events(
            EventQuery(
                session_id=marker_event.session_id,
                event_types=tuple(sorted(_REPLAY_OPERATION_START_EVENT_TYPES, key=str)),
                after_sequence=terminal_record.sequence,
                before_sequence=marker_record.sequence,
                limit=1,
            )
        )
        if later_operation_starts:
            return None
        return terminal_record.event.id

    async def _replay_events_response(
        http_request: Request,
        *,
        expected_session_id: str | None = None,
    ) -> EventSourceResponse | None:
        """SSE resume for reconnecting clients (``Last-Event-ID`` header).

        Instead of starting new work, replay the session's persisted events after the
        last one the client saw and keep following until the session reaches a
        terminal status (the detached pump finishes the run even after a disconnect).
        """
        last_event_id = http_request.headers.get("last-event-id")
        if last_event_id is None:
            return None
        marker = parse_last_event_id(
            last_event_id,
            expected_session_id=expected_session_id,
        )
        if marker is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Last-Event-ID must use `session_id:event_id` or the explicit "
                    f"`{SSE_REPLAY_START_MARKER_FORMAT}` start marker."
                ),
            )
        session_id, last_seen_event_id = marker
        if expected_session_id is not None and session_id != expected_session_id:
            raise HTTPException(
                status_code=422,
                detail="Last-Event-ID session does not match the request session_id.",
            )
        state = await session_store.load_state(session_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}",
            )
        marker_record = (
            None
            if last_seen_event_id is None
            else await _marker_record(session_id, last_seen_event_id)
        )
        after_sequence = None if marker_record is None else marker_record.sequence

        async def replay() -> AsyncIterator[dict[str, str]]:
            replay_after_sequence = after_sequence
            operation_in_progress = state.status in _REPLAY_ACTIVE_SESSION_STATUSES
            observed_terminal_event_id = (
                None
                if marker_record is None or operation_in_progress
                else await _marker_terminal_boundary_id(marker_record)
            )
            loop = asyncio.get_running_loop()
            idle_deadline = loop.time() + replay_idle_timeout_s
            poll_interval = _REPLAY_POLL_INTERVAL_MIN_S

            while True:
                page = await session_store.query_events(
                    EventQuery(
                        session_id=session_id,
                        after_sequence=replay_after_sequence,
                        limit=SSE_REPLAY_PAGE_EVENTS,
                    )
                )
                for record in page:
                    try:
                        message = event_to_sse_message(record.event)
                    except SseEventFrameTooLargeError as exc:
                        yield _stream_error_sse_message(
                            cayu_app,
                            exc,
                            kind="observer",
                            code="event_frame_too_large",
                            retryable=False,
                            session_id=session_id,
                        )
                        return
                    replay_after_sequence = record.sequence
                    terminal_boundary_id = _replay_terminal_boundary_id(record.event)
                    if record.event.type in _REPLAY_TERMINAL_EVENT_TYPES:
                        observed_terminal_event_id = terminal_boundary_id
                        operation_in_progress = False
                    elif record.event.type in _REPLAY_OPERATION_START_EVENT_TYPES:
                        observed_terminal_event_id = None
                        operation_in_progress = True
                    elif (
                        terminal_boundary_id is not None
                        and not operation_in_progress
                        and terminal_boundary_id != observed_terminal_event_id
                    ):
                        # A hook may establish a boundary when replay began after
                        # the terminal event. Validate its durable reference and
                        # operation epoch before trusting payload-carried lineage.
                        validated_boundary_id = await _marker_terminal_boundary_id(record)
                        if validated_boundary_id is not None:
                            observed_terminal_event_id = validated_boundary_id
                    elif operation_in_progress:
                        observed_terminal_event_id = None
                    yield message
                if page:
                    idle_deadline = loop.time() + replay_idle_timeout_s
                    poll_interval = _REPLAY_POLL_INTERVAL_MIN_S
                if len(page) == SSE_REPLAY_PAGE_EVENTS:
                    continue
                current = await session_store.load_state(session_id)
                if current is None:
                    return
                terminal_event_type = _REPLAY_TERMINAL_EVENT_BY_STATUS.get(current.status)
                if terminal_event_type is not None:
                    terminal_records = await session_store.query_events(
                        EventQuery(
                            session_id=session_id,
                            event_type=terminal_event_type,
                            order_by=EventOrder.SEQUENCE_DESC,
                            limit=1,
                        )
                    )
                    if (
                        terminal_records
                        and observed_terminal_event_id == terminal_records[0].event.id
                    ):
                        return
                elif current.status in _REPLAY_ACTIVE_SESSION_STATUSES:
                    operation_in_progress = True
                    observed_terminal_event_id = None
                else:
                    return
                remaining = idle_deadline - loop.time()
                if remaining <= 0:
                    timeout_error = TimeoutError(
                        f"Replay for session {session_id} received no events for "
                        f"{replay_idle_timeout_s:g} seconds."
                    )
                    yield _stream_error_sse_message(
                        cayu_app,
                        timeout_error,
                        kind="observer",
                        code="replay_idle_timeout",
                        retryable=True,
                        session_id=session_id,
                    )
                    return
                await asyncio.sleep(min(poll_interval, remaining))
                poll_interval = _next_replay_poll_interval(
                    poll_interval,
                    received_events=bool(page),
                )

        return EventSourceResponse(replay(), send_timeout=SSE_SEND_TIMEOUT_SECONDS)

    @router.post(
        "/run",
        dependencies=protected,
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def run_agent(
        body: RunBody,
        http_request: Request,
        trace_metadata: TraceContextMetadata,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session_id = body.session_id or f"session-{uuid4().hex}"

        if task_store is not None:
            task_id = str(uuid4())

            async def create_run_task(first_event: Event) -> None:
                if first_event.session_id != session_id:
                    raise RuntimeError(
                        "Run acceptance event belongs to a different session: "
                        f"{first_event.session_id}"
                    )
                state = await session_store.load_state(session_id)
                if state is None:
                    raise RuntimeError(f"Run session disappeared during acceptance: {session_id}")
                # The runtime is paused at ``first_event`` while this callback
                # executes. COMPLETED/FAILED therefore describe a terminal-prefix
                # stream that needs no task. INTERRUPTING/INTERRUPTED can instead
                # be a concurrent operator request; the task must still be linked
                # so the interrupted session can resume it later.
                if state.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
                    return
                await task_store.create_running_task(
                    TaskCreate(
                        task_id=task_id,
                        type="run",
                        title=body.prompt[:80],
                        session_id=session_id,
                        assigned_agent_name=body.agent,
                        input={"prompt": body.prompt},
                    )
                )

            after_accept = create_run_task
        else:
            task_id = None
            after_accept = None

        request = RunRequest(
            agent_name=body.agent,
            session_id=session_id,
            model=body.model,
            causal_budget_id=body.causal_budget_id,
            task_id=task_id,
            labels=body.labels,
            messages=[Message.text("user", body.prompt)],
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            metadata=trace_metadata,
            thinking=body.thinking,
        )

        return await _accepted_event_stream_response(
            cayu_app.run(request),
            cayu_app=cayu_app,
            session_id=session_id,
            after_accept=_mutation_acceptance_callback(
                mutation_id=mutation_id,
                mutation_kind="run",
                session_id=session_id,
                after_accept=after_accept,
            ),
        )

    @router.post(
        "/resume",
        dependencies=protected,
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def resume_agent(
        body: ResumeBody,
        http_request: Request,
        trace_metadata: TraceContextMetadata,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session = await session_store.load(body.session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = ResumeRequest(
            session_id=body.session_id,
            messages=[Message.text("user", body.prompt)],
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            metadata=trace_metadata,
            thinking=body.thinking,
        )

        return await _accepted_event_stream_response(
            cayu_app.resume(request),
            cayu_app=cayu_app,
            session_id=body.session_id,
            after_accept=_mutation_acceptance_callback(
                mutation_id=mutation_id,
                mutation_kind="resume",
                session_id=body.session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @router.post(
        "/sessions/{session_id}/compact",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def compact_session(
        session_id: PersistableNonBlankString,
        body: CompactSessionBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=session_id,
        )
        if replay is not None:
            return replay
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}",
            )

        request = CompactSessionRequest(
            session_id=session_id,
            idempotency_key=body.idempotency_key,
            expected_run_epoch=body.expected_run_epoch,
            expected_transcript_cursor=body.expected_transcript_cursor,
            instructions=body.instructions,
            limits=body.limits,
            budget_limits=body.budget_limits,
            requested_by=_request_interruption_actor(auth_context, body.requested_by),
        )
        return await _accepted_event_stream_response(
            cayu_app.compact_session(request),
            cayu_app=cayu_app,
            session_id=session_id,
            after_accept=_mutation_acceptance_callback(
                mutation_id=mutation_id,
                mutation_kind="session.compact",
                session_id=session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @router.post(
        "/sessions/{session_id}/messages",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def enqueue_session_message(
        session_id: NonBlankString,
        body: EnqueueSessionMessageBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=session_id,
        )
        if replay is not None:
            return replay
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}",
            )

        async def operation() -> AsyncIterator[Event]:
            result = await cayu_app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key=body.idempotency_key,
                    content=body.content,
                    delivery_mode=body.delivery_mode,
                    requested_by=_request_interruption_actor(
                        auth_context,
                        body.requested_by,
                    ),
                )
            )
            yield result.event

        return await _accepted_event_stream_response(
            operation(),
            cayu_app=cayu_app,
            session_id=session_id,
            after_accept=_mutation_acceptance_callback(
                mutation_id=mutation_id,
                mutation_kind="session.message.enqueue",
                session_id=session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @router.post(
        "/sessions/{session_id}/interrupt",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def interrupt_session(
        session_id: NonBlankString,
        http_request: Request,
        body: InterruptSessionBody | None = None,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=session_id,
        )
        if replay is not None:
            return replay
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}",
            )
        if session.status not in _SERVER_INTERRUPTIBLE_SESSION_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=f"Session cannot be interrupted from status: {session.status.value}",
            )

        request = InterruptSessionRequest(
            session_id=session_id,
            reason=body.reason if body is not None else None,
            metadata=body.metadata if body is not None else {},
            requested_by=_request_interruption_actor(
                auth_context,
                body.requested_by if body is not None else None,
            ),
        )
        return await _accepted_event_stream_response(
            cayu_app.interrupt_session(request),
            cayu_app=cayu_app,
            session_id=session_id,
            after_accept=_mutation_acceptance_callback(
                mutation_id=mutation_id,
                mutation_kind="interrupt",
                session_id=session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @router.post(
        "/tool-approvals/resolve",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def resolve_tool_approval(
        body: ToolApprovalBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session = await session_store.load(body.session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = ToolApprovalRequest(
            session_id=body.session_id,
            approval_id=body.approval_id,
            decision=body.decision,
            reason=body.reason,
            metadata=body.metadata,
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
        )

        return await _accepted_event_stream_response(
            cayu_app.resolve_tool_approval(request),
            cayu_app=cayu_app,
            session_id=body.session_id,
            after_accept=_mutation_acceptance_callback(
                mutation_id=mutation_id,
                mutation_kind="tool_approval.resolve",
                session_id=body.session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @router.post(
        "/tool-approvals/recover",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def recover_tool_approval(
        body: ToolApprovalRecoveryBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session = await session_store.load(body.session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = ToolApprovalRecoveryRequest(
            session_id=body.session_id,
            approval_id=body.approval_id,
            tool_call_id=body.tool_call_id,
            outcome=body.outcome,
            message=body.message,
            structured=body.structured,
            artifacts=body.artifacts,
            reason=body.reason,
            metadata=body.metadata,
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
        )

        return await _accepted_event_stream_response(
            cayu_app.recover_tool_approval(request),
            cayu_app=cayu_app,
            session_id=body.session_id,
            after_accept=_mutation_acceptance_callback(
                mutation_id=mutation_id,
                mutation_kind="tool_approval.recover",
                session_id=body.session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @router.post(
        "/tool-rounds/recover",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def recover_tool_round(
        body: ToolRoundRecoveryBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session = await session_store.load(body.session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = ToolRoundRecoveryRequest(
            session_id=body.session_id,
            round_id=body.round_id,
            tool_call_id=body.tool_call_id,
            outcome=body.outcome,
            message=body.message,
            structured=body.structured,
            artifacts=body.artifacts,
            reason=body.reason,
            metadata=body.metadata,
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
        )

        return await _accepted_event_stream_response(
            cayu_app.recover_tool_round(request),
            cayu_app=cayu_app,
            session_id=body.session_id,
            after_accept=_mutation_acceptance_callback(
                mutation_id=mutation_id,
                mutation_kind="tool_round.recover",
                session_id=body.session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @router.post(
        "/user-input/resolve",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def resolve_user_input(
        body: UserInputResolveBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session = await session_store.load(body.session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        response = UserInputResponse(
            session_id=body.session_id,
            input_id=body.input_id,
            answer=body.answer,
            structured=body.structured,
            artifacts=body.artifacts,
            metadata=body.metadata,
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
        )

        return await _accepted_event_stream_response(
            cayu_app.resolve_user_input(response),
            cayu_app=cayu_app,
            session_id=body.session_id,
            after_accept=_mutation_acceptance_callback(
                mutation_id=mutation_id,
                mutation_kind="user_input.resolve",
                session_id=body.session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @router.post(
        "/user-input/recover",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def recover_user_input(
        body: UserInputRecoveryBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session = await session_store.load(body.session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = UserInputRecoveryRequest(
            session_id=body.session_id,
            input_id=body.input_id,
            answer=body.answer,
            tool_call_id=body.tool_call_id,
            outcome=body.outcome,
            message=body.message,
            structured=body.structured,
            artifacts=body.artifacts,
            reason=body.reason,
            metadata=body.metadata,
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
        )

        return await _accepted_event_stream_response(
            cayu_app.recover_user_input(request),
            cayu_app=cayu_app,
            session_id=body.session_id,
            after_accept=_mutation_acceptance_callback(
                mutation_id=mutation_id,
                mutation_kind="user_input.recover",
                session_id=body.session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @router.get("/agents", response_model=AgentsResponse, dependencies=protected)
    async def list_agents():
        agents = [
            _serialize_agent(cayu_app, cayu_app.get_agent(name)) for name in cayu_app.list_agents()
        ]
        return {"agents": agents, "total_count": len(agents)}

    @router.get("/agents/{agent_name}", response_model=AgentsResponse, dependencies=protected)
    async def get_agent(agent_name: NonBlankString):
        try:
            agent = cayu_app.get_agent(agent_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Agent not found") from exc
        serialized = _serialize_agent(cayu_app, agent)
        return {"agents": [serialized], "total_count": 1}

    @router.get(
        "/environments",
        response_model=EnvironmentsResponse,
        dependencies=protected,
    )
    async def list_environments():
        records = cayu_app.list_environment_registrations()
        environments = [_serialize_environment(cayu_app, record) for record in records]
        return {"environments": environments, "total_count": len(environments)}

    @router.get(
        "/environments/{environment_name}",
        response_model=EnvironmentsResponse,
        dependencies=protected,
    )
    async def get_environment(environment_name: NonBlankString):
        record = next(
            (
                item
                for item in cayu_app.list_environment_registrations()
                if item.spec.name == environment_name
            ),
            None,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="Environment not found")
        return {"environments": [_serialize_environment(cayu_app, record)], "total_count": 1}

    @router.get(
        "/artifacts",
        response_model=ArtifactsResponse,
        responses=ARTIFACT_ENDPOINT_ERROR_RESPONSES,
        dependencies=protected,
    )
    async def list_artifacts(
        limit: Annotated[int, Query(ge=1, le=_ARTIFACT_PAGE_LIMIT_MAX)] = 100,
        offset: Annotated[int, Query(ge=0, le=_ARTIFACT_PAGE_OFFSET_MAX)] = 0,
        artifact_store_id: Annotated[str | None, Query()] = None,
        scope: ArtifactScope | None = None,
        session_id: Annotated[str | None, Query()] = None,
        agent_name: Annotated[str | None, Query()] = None,
        environment_name: Annotated[str | None, Query()] = None,
    ):
        requested_store_id = _clean_optional_query_value(
            artifact_store_id,
            "artifact_store_id",
        )
        requested_session_id = _clean_optional_query_value(session_id, "session_id")
        requested_agent_name = _clean_optional_query_value(agent_name, "agent_name")
        requested_environment_name = _clean_optional_query_value(
            environment_name,
            "environment_name",
        )
        stores = _artifact_stores_by_id(cayu_app)
        if requested_store_id is not None:
            store = stores.get(requested_store_id)
            if store is None:
                raise HTTPException(status_code=404, detail="Artifact store not found")
            selected_stores = {requested_store_id: store}
        else:
            selected_stores = stores

        artifacts: list[dict[str, Any]] = []
        total_count: int | None = 0
        truncated = False
        per_store_limit = offset + limit
        for store_id, store in selected_stores.items():
            try:
                result = await store.list(
                    scope=scope,
                    session_id=requested_session_id,
                    agent_name=requested_agent_name,
                    environment_name=requested_environment_name,
                    limit=per_store_limit,
                )
                if type(result) is not ArtifactListResult:
                    raise TypeError("Artifact stores must return ArtifactListResult from list().")
                page = ArtifactListResult(
                    artifacts=result.artifacts,
                    total_count=result.total_count,
                    truncated=result.truncated,
                )
            except (ArtifactStoreUnavailableError, OSError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Artifact store is unavailable.",
                ) from exc
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Artifact store returned invalid artifact data.",
                ) from exc
            artifacts.extend(
                _serialize_artifact(cayu_app, artifact, artifact_store_id=store_id)
                for artifact in page.artifacts
            )
            if page.total_count is None:
                total_count = None
                truncated = True
            elif total_count is not None:
                total_count += page.total_count
            truncated = truncated or page.truncated

        artifacts.sort(key=_artifact_sort_key, reverse=True)
        page_artifacts = artifacts[offset : offset + limit]
        next_offset = None
        has_more = False
        if total_count is None:
            if len(artifacts) > offset + limit or truncated:
                has_more = True
                candidate_next_offset = offset + limit
                if candidate_next_offset <= _ARTIFACT_PAGE_OFFSET_MAX:
                    next_offset = candidate_next_offset
        elif offset + limit < total_count:
            has_more = True
            candidate_next_offset = offset + limit
            if candidate_next_offset <= _ARTIFACT_PAGE_OFFSET_MAX:
                next_offset = candidate_next_offset
        truncated = has_more
        return {
            "artifacts": page_artifacts,
            "total_count": total_count,
            "truncated": truncated,
            "limit": limit,
            "offset": offset,
            "next_offset": next_offset,
        }

    async def _read_artifact_from_request(
        artifact_id: str,
        artifact_store_id: str | None,
        *,
        max_bytes: int | None = None,
    ):
        try:
            artifact_id = require_clean_nonblank(artifact_id, "artifact_id")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Artifact not found") from exc
        stores = _artifact_stores_by_id(cayu_app)
        requested_store_id = _clean_optional_query_value(
            artifact_store_id,
            "artifact_store_id",
        )
        if requested_store_id is not None:
            store = stores.get(requested_store_id)
            if store is None:
                raise HTTPException(status_code=404, detail="Artifact store not found")
            try:
                read = await store.read_bytes(artifact_id, max_bytes=max_bytes)
                return requested_store_id, copy_artifact_read_result(
                    read,
                    expected_artifact_id=artifact_id,
                    max_content_bytes=max_bytes,
                )
            except InvalidArtifactIdError as exc:
                raise HTTPException(status_code=404, detail="Artifact not found") from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Artifact not found") from exc
            except (ArtifactStoreUnavailableError, OSError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Artifact store is unavailable.",
                ) from exc
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Artifact store returned invalid artifact data.",
                ) from exc

        matches = []
        for store_id, store in stores.items():
            try:
                read = await store.read_bytes(artifact_id, max_bytes=max_bytes)
                matches.append(
                    (
                        store_id,
                        copy_artifact_read_result(
                            read,
                            expected_artifact_id=artifact_id,
                            max_content_bytes=max_bytes,
                        ),
                    )
                )
            except (FileNotFoundError, InvalidArtifactIdError):
                continue
            except (ArtifactStoreUnavailableError, OSError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Artifact store is unavailable.",
                ) from exc
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Artifact store returned invalid artifact data.",
                ) from exc
        if not matches:
            raise HTTPException(status_code=404, detail="Artifact not found")
        if len(matches) > 1:
            raise HTTPException(
                status_code=409,
                detail="Artifact id exists in multiple stores; pass artifact_store_id.",
            )
        return matches[0]

    @router.get(
        "/artifacts/{artifact_id}",
        response_model=ArtifactReadResponse,
        responses=ARTIFACT_ENDPOINT_ERROR_RESPONSES,
        dependencies=protected,
    )
    async def get_artifact(
        artifact_id: ArtifactIdPath,
        artifact_store_id: Annotated[str | None, Query()] = None,
        max_bytes: Annotated[int, Query(ge=1, le=262_144)] = 64_000,
    ):
        store_id, read = await _read_artifact_from_request(
            artifact_id,
            artifact_store_id,
            max_bytes=max_bytes,
        )
        preview_base64, text_preview = _artifact_read_preview(cayu_app, read)
        return {
            "artifact": _serialize_artifact(cayu_app, read.metadata, artifact_store_id=store_id),
            "preview_base64": preview_base64,
            "text_preview": text_preview,
            "total_bytes": read.total_bytes,
            "truncated": read.truncated,
        }

    @router.get(
        "/artifacts/{artifact_id}/content",
        response_class=Response,
        responses=ARTIFACT_CONTENT_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def get_artifact_content(
        artifact_id: ArtifactIdPath,
        artifact_store_id: Annotated[str, Query(min_length=1)],
        disposition: Annotated[Literal["attachment", "inline"], Query()] = "attachment",
        max_bytes: Annotated[int, Query(ge=1, le=_ARTIFACT_CONTENT_BYTES_MAX)] = (
            _ARTIFACT_CONTENT_BYTES_MAX
        ),
    ):
        store_id, read = await _read_artifact_from_request(
            artifact_id,
            artifact_store_id,
            max_bytes=max_bytes,
        )
        if read.truncated or read.total_bytes > max_bytes or len(read.content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Artifact exceeds the requested max_bytes for direct content "
                    "response. Use the bounded artifact preview, increase max_bytes "
                    "up to the server maximum, or use a store-native streaming/range "
                    "reader for artifacts above that maximum."
                ),
            )
        response_disposition = _artifact_content_disposition_kind(
            read.metadata.content_type,
            disposition,
        )
        return Response(
            content=read.content,
            media_type=read.metadata.content_type,
            headers={
                "Content-Disposition": _artifact_content_disposition(
                    read.metadata.filename,
                    response_disposition,
                ),
                "X-Content-Type-Options": "nosniff",
                "X-Cayu-Artifact-Id": _artifact_header_value(read.metadata.id, artifact_id),
                "X-Cayu-Artifact-Store-Id": _artifact_header_value(store_id, "artifact-store"),
                "Cache-Control": "private, no-store",
            },
        )

    @router.get(
        "/pending-actions",
        response_model=PendingActionsResponse,
        responses=PENDING_ACTION_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def list_pending_actions(
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        session_id: Annotated[str | None, Query()] = None,
        q: Annotated[str | None, Query()] = None,
        kind: PendingActionKind | None = None,
        agent_name: Annotated[str | None, Query()] = None,
        environment_name: Annotated[str | None, Query()] = None,
        cursor: Annotated[str | None, Query()] = None,
    ):
        requested_session_id = _clean_optional_query_value(session_id, "session_id")
        search = _clean_optional_query_value(q, "q")
        requested_agent_name = _clean_optional_query_value(agent_name, "agent_name")
        requested_environment_name = _clean_optional_query_value(
            environment_name, "environment_name"
        )
        requested_cursor = _clean_optional_query_value(cursor, "cursor")
        if requested_cursor is not None:
            try:
                decode_session_cursor(requested_cursor)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            result = await session_store.query_pending_actions(
                PendingActionQuery(
                    session_id=requested_session_id,
                    kind=kind,
                    agent_name=requested_agent_name,
                    environment_name=requested_environment_name,
                    q=search,
                    cursor=requested_cursor,
                    limit=limit,
                )
            )
        except PendingActionResultTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        return {
            "actions": [
                {
                    **action.model_dump(
                        mode="json",
                        exclude={"session", "event"},
                    ),
                    "session": _serialize_session_base(action.session),
                    "event": _serialize_event_record(action.event),
                }
                for action in result.actions
            ],
            "issues": [issue.model_dump(mode="json") for issue in result.issues],
            "next_cursor": result.next_cursor,
            "has_more": result.has_more,
            "total_count": result.total_count,
            "inspected_candidate_count": result.inspected_candidate_count,
        }

    @router.get("/sessions", response_model=ListSessionsResponse, dependencies=protected)
    async def list_sessions(
        limit: Annotated[int, Query(ge=1, le=1000)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
        cursor: Annotated[str | None, Query()] = None,
        q: Annotated[str | None, Query()] = None,
        status: SessionStatus | None = None,
        debug_state: SessionDebugState | None = None,
        agent_name: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        environment_name: str | None = None,
        parent_session_id: str | None = None,
        causal_budget_id: str | None = None,
        order_by: SessionOrder = SessionOrder.UPDATED_AT_DESC,
        label: Annotated[list[str] | None, Query()] = None,
        label_selector: Annotated[list[str] | None, Query()] = None,
    ):
        labels = _parse_session_label_filters(label)
        label_selectors = _parse_session_label_selectors(label_selector)
        try:
            result = await session_store.list_sessions(
                SessionQuery(
                    q=_clean_optional_query_value(q, "q"),
                    status=status,
                    debug_state=debug_state,
                    agent_name=_clean_optional_query_value(agent_name, "agent_name"),
                    provider_name=_clean_optional_query_value(provider_name, "provider_name"),
                    model=_clean_optional_query_value(model, "model"),
                    environment_name=_clean_optional_query_value(
                        environment_name,
                        "environment_name",
                    ),
                    parent_session_id=_clean_optional_query_value(
                        parent_session_id,
                        "parent_session_id",
                    ),
                    causal_budget_id=_clean_optional_query_value(
                        causal_budget_id,
                        "causal_budget_id",
                    ),
                    labels=labels,
                    label_selectors=label_selectors,
                    limit=limit,
                    offset=offset,
                    cursor=cursor,
                    include_total_count=True,
                    order_by=order_by,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "sessions": [_serialize_session_base(session) for session in result.sessions],
            "next_cursor": result.next_cursor,
            "total_count": result.total_count,
        }

    @router.post(
        "/sessions/summary",
        response_model=SessionsSummaryResponse,
        dependencies=protected,
    )
    async def get_sessions_summary(
        body: SessionsSummaryBody | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 1000,
        offset: Annotated[int, Query(ge=0)] = 0,
        cursor: Annotated[str | None, Query()] = None,
        q: Annotated[str | None, Query()] = None,
        status: SessionStatus | None = None,
        debug_state: SessionDebugState | None = None,
        agent_name: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        environment_name: str | None = None,
        parent_session_id: str | None = None,
        causal_budget_id: str | None = None,
        order_by: SessionOrder = SessionOrder.UPDATED_AT_DESC,
        label: Annotated[list[str] | None, Query()] = None,
        label_selector: Annotated[list[str] | None, Query()] = None,
    ):
        body = body or SessionsSummaryBody()
        labels = _parse_session_label_filters(label)
        label_selectors = _parse_session_label_selectors(label_selector)
        try:
            result = await session_store.list_sessions(
                SessionQuery(
                    q=_clean_optional_query_value(q, "q"),
                    status=status,
                    debug_state=debug_state,
                    agent_name=_clean_optional_query_value(agent_name, "agent_name"),
                    provider_name=_clean_optional_query_value(provider_name, "provider_name"),
                    model=_clean_optional_query_value(model, "model"),
                    environment_name=_clean_optional_query_value(
                        environment_name,
                        "environment_name",
                    ),
                    parent_session_id=_clean_optional_query_value(
                        parent_session_id,
                        "parent_session_id",
                    ),
                    causal_budget_id=_clean_optional_query_value(
                        causal_budget_id,
                        "causal_budget_id",
                    ),
                    labels=labels,
                    label_selectors=label_selectors,
                    limit=limit,
                    offset=offset,
                    cursor=cursor,
                    include_total_count=True,
                    order_by=order_by,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        sessions = result.sessions
        session_event_records_by_id: dict[str, list[EventRecord]] = {}
        session_ids = [session.id for session in sessions]
        all_event_records = await _query_all_session_event_records(session_ids)
        for record in all_event_records:
            session_event_records_by_id.setdefault(record.event.session_id, []).append(record)
        for session in sessions:
            session_event_records_by_id.setdefault(session.id, [])

        usage_event_records = [
            record
            for record in all_event_records
            if record.event.type in {EventType.MODEL_COMPLETED, EventType.TOOL_CALL_STARTED}
        ]
        usage_events = [
            record.event
            for record in sorted(usage_event_records, key=lambda record: record.sequence)
        ]
        usage_summary = causal_budget_usage_summary(
            causal_budget_id="session-query",
            session_ids=session_ids,
            events=usage_events,
        ).model_dump()
        usage_summary.pop("causal_budget_id", None)
        model_events = [
            record.event
            for record in sorted(all_event_records, key=lambda record: record.sequence)
            if record.event.type == EventType.MODEL_COMPLETED
        ]
        provider_breakdown = _usage_breakdown(
            model_events,
            key_fn=lambda metrics: (metrics.provider_name, None),
        )
        model_breakdown = _usage_breakdown(
            model_events,
            key_fn=lambda metrics: (metrics.provider_name, metrics.model),
        )

        cost_summary = None
        if body.pricing is not None:
            aggregate_cost = build_session_cost_summary(
                session_id="session-query",
                events=model_events,
                pricing=body.pricing,
                currency=body.currency,
            ).model_dump(mode="json")
            aggregate_cost.pop("session_id", None)
            aggregate_cost["session_ids"] = session_ids
            aggregate_cost["session_count"] = len(session_ids)
            aggregate_cost["session_costs"] = [
                build_session_cost_summary(
                    session_id=session.id,
                    events=[
                        record.event
                        for record in session_event_records_by_id[session.id]
                        if record.event.type == EventType.MODEL_COMPLETED
                    ],
                    pricing=body.pricing,
                    currency=body.currency,
                ).model_dump(mode="json")
                for session in sessions
            ]
            cost_summary = aggregate_cost

        session_items = []
        for session in sessions:
            records = session_event_records_by_id[session.id]
            outcome = session_outcome_from_records(session, records)
            event_summary = event_summary_from_records(session.id, records)
            session_items.append(
                {
                    "session": _serialize_session(session),
                    "outcome": _serialize_session_outcome(outcome),
                    "events": {
                        "total_events": event_summary.total_events,
                        "counts_by_type": event_summary.counts_by_type,
                        "latest_event": (
                            None
                            if event_summary.latest_event is None
                            else _serialize_event_record(event_summary.latest_event)
                        ),
                    },
                }
            )

        return {
            "session_count": len(sessions),
            "sessions": session_items,
            "next_cursor": result.next_cursor,
            "total_count": result.total_count,
            "usage": usage_summary,
            "provider_breakdown": provider_breakdown,
            "model_breakdown": model_breakdown,
            "cost": cost_summary,
        }

    @router.get(
        "/sessions/{session_id}/usage",
        response_model=SessionUsageSummary,
        dependencies=protected,
    )
    async def get_session_usage(session_id: NonBlankString):
        try:
            summary = await cayu_app.get_session_usage(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        return summary.model_dump()

    @router.post(
        "/sessions/{session_id}/cost",
        response_model=SessionCostSummary,
        dependencies=protected,
    )
    async def estimate_session_cost(session_id: NonBlankString, body: SessionCostBody):
        try:
            summary = await cayu_app.get_session_cost(
                session_id,
                body.pricing,
                currency=body.currency,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        return summary.model_dump(mode="json")

    @router.get(
        "/causal-budgets/{causal_budget_id}/usage",
        response_model=CausalBudgetUsageSummary,
        dependencies=protected,
    )
    async def get_causal_budget_usage(causal_budget_id: NonBlankString):
        try:
            summary = await cayu_app.get_causal_budget_usage(causal_budget_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Causal budget not found") from exc
        return summary.model_dump()

    @router.post(
        "/causal-budgets/{causal_budget_id}/cost",
        response_model=CausalBudgetCostSummary,
        dependencies=protected,
    )
    async def estimate_causal_budget_cost(
        causal_budget_id: NonBlankString,
        body: SessionCostBody,
    ):
        try:
            summary = await cayu_app.get_causal_budget_cost(
                causal_budget_id,
                body.pricing,
                currency=body.currency,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Causal budget not found") from exc
        return summary.model_dump(mode="json")

    @router.post(
        "/causal-budgets/{causal_budget_id}/summary",
        response_model=CausalBudgetSummaryResponse,
        dependencies=protected,
    )
    async def get_causal_budget_summary(
        causal_budget_id: NonBlankString,
        body: SessionCostBody,
    ):
        sessions = await _list_all_causal_sessions(causal_budget_id)
        if not sessions:
            raise HTTPException(status_code=404, detail="Causal budget not found")

        session_ids = [session.id for session in sessions]
        causal_event_records = await _query_all_causal_event_records(causal_budget_id)
        usage_event_records = [
            record
            for record in causal_event_records
            if record.event.type == EventType.MODEL_COMPLETED
        ]
        tool_event_records = [
            record
            for record in causal_event_records
            if record.event.type == EventType.TOOL_CALL_STARTED
        ]
        usage_events = [
            record.event
            for record in sorted(
                [*usage_event_records, *tool_event_records],
                key=lambda record: record.sequence,
            )
        ]
        usage_summary = causal_budget_usage_summary(
            causal_budget_id=causal_budget_id,
            session_ids=session_ids,
            events=usage_events,
        )
        cost_summary = build_causal_budget_cost_summary(
            causal_budget_id=causal_budget_id,
            session_ids=session_ids,
            events=[record.event for record in usage_event_records],
            pricing=body.pricing,
            currency=body.currency,
        )
        session_items = []
        for session in sessions:
            session_event_records = [
                record for record in causal_event_records if record.event.session_id == session.id
            ]
            outcome = session_outcome_from_records(
                session,
                session_event_records,
            )
            event_summary = event_summary_from_records(
                session.id,
                session_event_records,
            )
            session_items.append(
                {
                    "session": _serialize_session(session),
                    "outcome": _serialize_session_outcome(outcome),
                    "events": {
                        "total_events": event_summary.total_events,
                        "counts_by_type": event_summary.counts_by_type,
                        "latest_event": (
                            None
                            if event_summary.latest_event is None
                            else _serialize_event_record(event_summary.latest_event)
                        ),
                    },
                }
            )

        return {
            "causal_budget_id": causal_budget_id,
            "session_count": len(sessions),
            "sessions": session_items,
            "usage": usage_summary.model_dump(),
            "cost": cost_summary.model_dump(mode="json"),
        }

    @router.get(
        "/sessions/{session_id}/state",
        response_model=SessionStateResponse,
        dependencies=protected,
    )
    async def get_session_state(session_id: NonBlankString):
        state = await session_store.load_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")
        interruption_cascade = await cayu_app.interruption_cascade_status(session_id)
        return {
            "session_id": state.id,
            "status": state.status,
            "updated_at": state.updated_at.isoformat(),
            "last_activity_at": state.last_activity_at.isoformat(),
            "interruption_cascade": interruption_cascade,
        }

    @router.get(
        "/sessions/{session_id}/summary",
        response_model=SessionSummaryResponse,
        dependencies=protected,
    )
    async def get_session_summary(session_id: NonBlankString):
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        event_summary = await session_store.summarize_events(session_id)
        outcome = await session_store.summarize_outcome(session_id)
        transcript_page = await session_store.query_transcript(
            TranscriptQuery(session_id=session_id, limit=1)
        )
        usage_summary = await cayu_app.get_session_usage(session_id)

        return {
            "session": _serialize_session(session),
            "events": {
                "total_events": event_summary.total_events,
                "counts_by_type": event_summary.counts_by_type,
                "latest_event": (
                    None
                    if event_summary.latest_event is None
                    else _serialize_event_record(event_summary.latest_event)
                ),
            },
            "transcript": {
                "total_messages": transcript_page.total_records,
            },
            "outcome": _serialize_session_outcome(outcome),
            "usage": usage_summary.model_dump(),
        }

    async def _list_all_causal_sessions(causal_budget_id: str) -> list[Session]:
        sessions: list[Session] = []
        offset = 0
        while True:
            page = (
                await session_store.list_sessions(
                    SessionQuery(
                        causal_budget_id=causal_budget_id,
                        limit=1000,
                        offset=offset,
                        order_by=SessionOrder.CREATED_AT_ASC,
                    )
                )
            ).sessions
            if not page:
                return sessions
            sessions.extend(page)
            if len(page) < 1000:
                return sessions
            offset += len(page)

    async def _query_all_session_event_records(session_ids: list[str]) -> list[EventRecord]:
        if not session_ids:
            return []
        records: list[EventRecord] = []
        after_sequence = None
        while True:
            page = await session_store.query_events(
                EventQuery(
                    session_ids=tuple(session_ids),
                    after_sequence=after_sequence,
                    limit=5000,
                )
            )
            if not page:
                return records
            records.extend(page)
            if len(page) < 5000:
                return records
            after_sequence = page[-1].sequence

    async def _query_all_causal_event_records(causal_budget_id: str) -> list[EventRecord]:
        records: list[EventRecord] = []
        after_sequence = None
        while True:
            page = await session_store.query_events(
                EventQuery(
                    causal_budget_id=causal_budget_id,
                    after_sequence=after_sequence,
                    limit=5000,
                )
            )
            if not page:
                return records
            records.extend(page)
            if len(page) < 5000:
                return records
            after_sequence = page[-1].sequence

    @router.get(
        "/sessions/{session_id}/events",
        response_model=ListSessionEventsResponse,
        dependencies=protected,
    )
    async def list_session_events(
        session_id: NonBlankString,
        event_id: str | None = Query(
            default=None,
            description="Return the event with this exact session-scoped event ID, if it exists.",
        ),
        event_type: str | None = None,
        exclude_event_type: str | None = Query(
            default=None,
            description="Exclude one event type before applying pagination.",
        ),
        tool_name: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        workflow_name: str | None = None,
        after_sequence: int | None = Query(
            default=None,
            ge=0,
            description="Return only events with a greater durable sequence.",
        ),
        before_sequence: int | None = Query(
            default=None,
            ge=1,
            description="Return only events with a smaller durable sequence.",
        ),
        order_by: Annotated[
            EventOrder,
            Query(description="Return events in durable sequence order."),
        ] = EventOrder.SEQUENCE_ASC,
        limit: int = Query(default=100, ge=1, le=_EVENT_PAGE_LIMIT_MAX),
    ):
        state = await session_store.load_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")

        has_event_filters = any(
            value is not None
            for value in (
                event_id,
                event_type,
                exclude_event_type,
                tool_name,
                agent_name,
                environment_name,
                workflow_name,
            )
        )
        try:
            query = EventQuery(
                session_id=session_id,
                event_id=event_id,
                event_type=event_type,
                exclude_event_types=(exclude_event_type,) if exclude_event_type is not None else (),
                tool_name=tool_name,
                agent_name=agent_name,
                environment_name=environment_name,
                workflow_name=workflow_name,
                after_sequence=after_sequence,
                before_sequence=before_sequence,
                limit=limit + 1,
                order_by=order_by,
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=exc.errors(include_context=False, include_url=False),
            ) from exc

        raw_scan_sequence: int | None = None
        if before_sequence is None and has_event_filters:
            # Capture the raw high-water mark before the filtered read. Advancing
            # through a sequence observed after that read could skip a matching
            # event committed between the two queries.
            latest_records = await session_store.query_events(
                EventQuery(
                    session_id=session_id,
                    after_sequence=after_sequence,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            if latest_records:
                raw_scan_sequence = latest_records[0].sequence

        records = await session_store.query_events(query)
        page = records[:limit]
        has_more = len(records) > limit
        cursor = after_sequence if order_by == EventOrder.SEQUENCE_ASC else before_sequence
        next_sequence = page[-1].sequence if page else cursor

        scan_through_sequence: int | None = None
        if before_sequence is None:
            scan_candidates = [sequence for sequence in (after_sequence,) if sequence is not None]
            if order_by == EventOrder.SEQUENCE_ASC and has_more:
                # A matching event remains beyond this response, so the tail
                # cursor must stop at the last event actually returned.
                if page:
                    scan_candidates.append(page[-1].sequence)
            else:
                scan_candidates.extend(record.sequence for record in page)
                if raw_scan_sequence is not None:
                    scan_candidates.append(raw_scan_sequence)
            if scan_candidates:
                scan_through_sequence = max(scan_candidates)

        return {
            "session_id": session_id,
            "events": [_serialize_event_record(record) for record in page],
            "order_by": order_by,
            "next_sequence": next_sequence,
            "scan_through_sequence": scan_through_sequence,
            "has_more": has_more,
        }

    @router.get(
        "/sessions/{session_id}/transcript",
        response_model=SessionTranscriptResponse,
        dependencies=protected,
    )
    async def get_session_transcript(
        session_id: NonBlankString,
        role: MessageRole | None = None,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=_TRANSCRIPT_PAGE_LIMIT_MAX),
        include_thinking: bool = Query(default=True),
    ):
        state = await session_store.load_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")

        transcript_page = await session_store.query_transcript(
            TranscriptQuery(
                session_id=session_id,
                role=role,
                offset=offset,
                limit=limit,
                include_thinking=include_thinking,
            )
        )
        # Advance by the queried window size, not the returned record count: the
        # include_thinking filter can drop thinking-only records from a page, so
        # len(records) under-counts the messages consumed and would stall pagination.
        consumed = min(limit, max(0, transcript_page.total_records - offset))
        next_offset = offset + consumed

        return {
            "session_id": session_id,
            "messages": [
                _serialize_transcript_message(record.index, record.message)
                for record in transcript_page.records
            ],
            "offset": offset,
            "next_offset": next_offset,
            "has_more": next_offset < transcript_page.total_records,
            "total_messages": transcript_page.total_records,
        }

    @router.get(
        "/sessions/{session_id}",
        response_model=ApiSession,
        dependencies=protected,
    )
    async def get_session(session_id: NonBlankString):
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return _serialize_session(session)

    @router.delete("/sessions/{session_id}", status_code=204, dependencies=protected)
    async def delete_session(session_id: NonBlankString):
        try:
            await session_store.delete_session(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return None

    @router.patch(
        "/sessions/{session_id}/labels",
        dependencies=protected,
        response_model=ApiSession,
    )
    async def update_session_labels(
        session_id: NonBlankString,
        body: UpdateSessionLabelsBody,
    ):
        try:
            session = await session_store.update_labels(session_id, body.labels)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _serialize_session(session)

    @router.patch(
        "/sessions/{session_id}/metadata",
        dependencies=protected,
        response_model=ApiSession,
    )
    async def update_session_metadata(
        session_id: NonBlankString,
        body: UpdateSessionMetadataBody,
    ):
        try:
            session = await session_store.update_metadata(session_id, body.metadata)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _serialize_session(session)

    @router.get("/tasks", response_model=list[ApiTaskListItem], dependencies=protected)
    async def list_tasks(
        q: str | None = None,
        status: TaskStatus | None = None,
        task_type: str | None = Query(default=None, alias="type"),
        session_id: str | None = None,
        parent_task_id: str | None = None,
        assigned_agent_name: str | None = None,
        order_by: TaskOrder = TaskOrder.UPDATED_AT_DESC,
        limit: int = 50,
        offset: int = 0,
    ):
        if task_store is None:
            return []
        try:
            query = TaskQuery(
                q=q,
                status=status,
                type=task_type,
                session_id=session_id,
                parent_task_id=parent_task_id,
                assigned_agent_name=assigned_agent_name,
                order_by=order_by,
                limit=limit,
                offset=offset,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        tasks = await task_store.list_tasks(query)
        return [_serialize_task_list_item(t) for t in tasks]

    async def _require_task_store():
        if task_store is None:
            raise HTTPException(status_code=404, detail="Task store is not configured.")
        return task_store

    @router.get(
        "/tasks/{task_id}",
        response_model=ApiTaskDetail,
        dependencies=protected,
    )
    async def get_task(task_id: NonBlankString):
        store = await _require_task_store()
        task = await store.load_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return _serialize_task_detail(task)

    async def _apply_task_action(action, task_id: str):
        try:
            task = await action(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _serialize_task_detail(task)

    @router.post(
        "/tasks/{task_id}/pause",
        dependencies=protected,
        response_model=ApiTaskDetail,
    )
    async def pause_task(task_id: NonBlankString, body: TaskHoldBody | None = None):
        store = await _require_task_store()
        request_body = body or TaskHoldBody()
        return await _apply_task_action(
            lambda task_id: store.pause_task(
                task_id,
                reason=request_body.reason,
                payload=request_body.payload,
            ),
            task_id,
        )

    @router.post(
        "/tasks/{task_id}/block",
        dependencies=protected,
        response_model=ApiTaskDetail,
    )
    async def block_task(task_id: NonBlankString, body: TaskHoldBody | None = None):
        store = await _require_task_store()
        request_body = body or TaskHoldBody()
        return await _apply_task_action(
            lambda task_id: store.block_task(
                task_id,
                reason=request_body.reason,
                payload=request_body.payload,
            ),
            task_id,
        )

    @router.post(
        "/tasks/{task_id}/needs-attention",
        dependencies=protected,
        response_model=ApiTaskDetail,
    )
    async def mark_task_needs_attention(
        task_id: NonBlankString,
        body: TaskHoldBody | None = None,
    ):
        store = await _require_task_store()
        request_body = body or TaskHoldBody()
        return await _apply_task_action(
            lambda task_id: store.mark_task_needs_attention(
                task_id,
                reason=request_body.reason,
                payload=request_body.payload,
            ),
            task_id,
        )

    @router.post(
        "/tasks/{task_id}/resume",
        dependencies=protected,
        response_model=ApiTaskDetail,
    )
    async def resume_task(task_id: NonBlankString):
        store = await _require_task_store()
        return await _apply_task_action(store.resume_task, task_id)

    def _knowledge_review_workflow() -> KnowledgeReviewWorkflow:
        if knowledge_store is None:
            raise HTTPException(status_code=404, detail="Knowledge store is not configured.")
        return KnowledgeReviewWorkflow(
            knowledge_store,
            namespace=knowledge_review_namespace,
            labels=knowledge_review_labels,
            default_limit=50,
        )

    async def _apply_knowledge_review_action(action, entry_id: str):
        try:
            entry = await action(entry_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _serialize_reviewed_knowledge_entry(entry)

    @router.get(
        "/knowledge/pending",
        response_model=PendingKnowledgeListResponse,
        dependencies=protected,
    )
    async def list_pending_knowledge(
        namespace: str | None = None,
        label: Annotated[list[str] | None, Query()] = None,
        kind: Annotated[list[str] | None, Query()] = None,
        aspect: Annotated[list[str] | None, Query()] = None,
        visibility: Annotated[list[KnowledgeVisibility] | None, Query()] = None,
        source_type: str | None = None,
        source_id: str | None = None,
        limit: int = 50,
        max_bytes: int = 20_000,
    ):
        workflow = _knowledge_review_workflow()
        try:
            result = await workflow.list_pending(
                namespace=_clean_optional_query_value(namespace, "namespace"),
                labels=_parse_knowledge_label_filters(label),
                kinds=_parse_knowledge_string_filters(kind, "kind") if kind is not None else None,
                visibilities=visibility,
                aspects=_parse_knowledge_string_filters(aspect, "aspect"),
                source_type=_clean_optional_query_value(source_type, "source_type"),
                source_id=_clean_optional_query_value(source_id, "source_id"),
                limit=limit,
                max_bytes=max_bytes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "entries": [_serialize_knowledge_list_item(item) for item in result.entries],
            "truncated": result.truncated,
            "limit": result.limit,
            "max_bytes": result.max_bytes,
            "total_entries_known": result.total_entries_known,
        }

    @router.get(
        "/knowledge/pending/{entry_id}",
        response_model=PendingKnowledgeDetailResponse,
        dependencies=protected,
    )
    async def get_pending_knowledge(
        entry_id: NonBlankString,
        max_chunks: Annotated[
            int,
            Query(ge=1, le=_KNOWLEDGE_PENDING_DETAIL_MAX_CHUNKS),
        ] = _KNOWLEDGE_PENDING_DETAIL_MAX_CHUNKS,
        max_bytes: Annotated[
            int,
            Query(ge=1, le=_KNOWLEDGE_PENDING_DETAIL_MAX_BYTES),
        ] = _KNOWLEDGE_PENDING_DETAIL_MAX_BYTES,
    ):
        workflow = _knowledge_review_workflow()
        assert knowledge_store is not None
        try:
            entry = await workflow.get_pending(entry_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        try:
            chunks = await knowledge_store.read_chunks(
                entry.id,
                max_chunks=max_chunks,
                max_bytes=max_bytes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            **_serialize_knowledge_detail(entry),
            "chunks": [_serialize_knowledge_chunk(chunk) for chunk in chunks],
            "chunk_limit": max_chunks,
            "chunk_max_bytes": max_bytes,
        }

    @router.post(
        "/knowledge/{entry_id}/approve",
        dependencies=protected,
        response_model=ApiReviewedKnowledgeEntry,
    )
    async def approve_knowledge(entry_id: NonBlankString):
        workflow = _knowledge_review_workflow()
        return await _apply_knowledge_review_action(workflow.approve, entry_id)

    @router.post(
        "/knowledge/{entry_id}/reject",
        dependencies=protected,
        response_model=ApiReviewedKnowledgeEntry,
    )
    async def reject_knowledge(entry_id: NonBlankString):
        workflow = _knowledge_review_workflow()
        return await _apply_knowledge_review_action(workflow.reject, entry_id)

    @router.get("/health", response_model=HealthResponse)
    async def health():
        return {"ok": True}

    return router


def _normalize_api_path(path: str) -> str:
    value = path.strip()
    if not value:
        raise ValueError("api_path must not be blank.")
    if "?" in value or "#" in value or "://" in value:
        raise ValueError("api_path must be a URL path, not a URL.")
    value = "/" + value.strip("/")
    if value == "/":
        raise ValueError("api_path must not be the site root.")
    return value
