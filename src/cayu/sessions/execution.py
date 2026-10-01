"""Read-only session liveness; leases are observations, never run authority."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from inspect import isasyncgenfunction
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ExecutionOwnerKind = Literal[
    "in_process_runner", "server_stream", "task_worker", "recovery", "foreground_child_delivery"
]
ExecutionProgressKind = Literal["model_stream", "tool_call", "waiting_for_input", "publishing"]


class SessionExecutionConfig(BaseModel):
    """Bounded independent heartbeat: one small write per 15s, a 60s lease by default."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    heartbeat_interval_seconds: float = Field(default=15.0, ge=0.05, le=300)
    lease_seconds: float = Field(default=60.0, ge=0.15, le=1200)
    owner_label: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_interval(self):
        if (
            not math.isfinite(self.heartbeat_interval_seconds)
            or not math.isfinite(self.lease_seconds)
            or self.lease_seconds < 3 * self.heartbeat_interval_seconds
        ):
            raise ValueError(
                "Execution lease must cover at least three finite heartbeat intervals."
            )
        return self

    @field_validator("owner_label")
    @classmethod
    def validate_label(cls, value):
        if value is not None and (value.strip() != value or any(ord(c) < 32 for c in value)):
            raise ValueError("Execution owner label must be clean nonblank text.")
        return value


class SessionExecutionState(BaseModel):
    """Content-free liveness evaluated by the store clock; reading never renews it."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    schema_version: Literal[1] = 1
    session_id: str
    state: Literal["executing", "idle", "waiting", "owner_lost", "terminal", "unknown"]
    owner_kind: ExecutionOwnerKind | None = None
    owner_id: str | None = None
    owner_label: str | None = None
    local_owner: bool = False
    run_epoch: int = Field(ge=0)
    operation_id: str | None = None
    claimed_at: datetime | None = None
    heartbeat_at: datetime | None = None
    lease_expires_at: datetime | None = None
    last_progress_at: datetime | None = None
    last_progress_kind: ExecutionProgressKind | None = None

    @field_validator("claimed_at", "heartbeat_at", "lease_expires_at", "last_progress_at")
    @classmethod
    def aware(cls, value):
        if value is not None:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("Execution timestamps must be timezone-aware.")
            return value.astimezone(UTC)
        return value


class _ExecutionOwner(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    session_id: str
    session_instance_id: str
    run_epoch: int
    token: str
    owner_kind: ExecutionOwnerKind
    owner_id: str
    owner_label: str | None
    operation_id: str | None
    claimed_at: datetime
    heartbeat_at: datetime
    lease_expires_at: datetime
    last_progress_at: datetime
    last_progress_kind: ExecutionProgressKind = "publishing"
    released: bool = False


_OWNER_KIND: ContextVar[ExecutionOwnerKind] = ContextVar(
    "cayu_execution_owner_kind", default="in_process_runner"
)
_PROGRESS: ContextVar[dict[str, tuple[int, ExecutionProgressKind]] | None] = ContextVar(
    "cayu_execution_progress", default=None
)


@contextmanager
def execution_owner_kind(kind: ExecutionOwnerKind):
    token = _OWNER_KIND.set(kind)
    try:
        yield
    finally:
        _OWNER_KIND.reset(token)


def current_execution_owner_kind(*, task_id: str | None = None) -> ExecutionOwnerKind:
    kind = _OWNER_KIND.get()
    return "task_worker" if kind == "in_process_runner" and task_id is not None else kind


def bind_execution_progress(progress: dict[str, tuple[int, ExecutionProgressKind]]) -> None:
    _PROGRESS.set(progress)


def note_execution_progress(event) -> None:
    progress = _PROGRESS.get()
    if progress is None or event.session_id not in progress:
        return
    name = getattr(event.type, "value", event.type)
    kind: ExecutionProgressKind | None = None
    if name in {"model.started", "model.completed", "model.failed"}:
        kind = "model_stream"
    elif name in {"tool.call.started", "tool.call.completed", "tool.call.failed"}:
        kind = "tool_call"
    elif name in {"session.awaiting_user_input", "tool.call.approval_requested"}:
        kind = "waiting_for_input"
    elif name in {"session.started", "session.resumed", "session.checkpointed"}:
        kind = "publishing"
    if kind is not None:
        generation, _ = progress[event.session_id]
        progress[event.session_id] = generation + 1, kind


def execution_state(
    *,
    session_id,
    session_instance_id,
    run_epoch,
    status,
    waiting,
    waiting_for_child=False,
    owner,
    now,
):
    from cayu.sessions.base import SessionStatus

    current = (
        owner is not None
        and owner.session_instance_id == session_instance_id
        and owner.run_epoch == run_epoch
    )
    live = current and not owner.released and owner.lease_expires_at > now
    if status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
        state = "terminal"
    elif waiting:
        state = "waiting"
    elif live:
        state = "executing"
    elif waiting_for_child:
        state = "waiting"
    elif status == SessionStatus.INTERRUPTED:
        state = "terminal"
    elif status == SessionStatus.PENDING:
        state = "idle"
    elif current:
        state = "owner_lost"
    else:
        # Older binaries and custom stores cannot attest an owner. Do not infer
        # death from last_activity_at or grant recovery through this projection.
        state = "unknown"
    fields = {}
    if current and state in {"executing", "owner_lost"}:
        fields = owner.model_dump(
            exclude={"session_id", "session_instance_id", "run_epoch", "token", "released"}
        )
    return SessionExecutionState(session_id=session_id, state=state, run_epoch=run_epoch, **fields)


def execution_owned_by(kind: ExecutionOwnerKind):
    """Tag internal entry points without leaking context across yielded events."""

    def decorate(function):
        if isasyncgenfunction(function):

            @wraps(function)
            async def stream(*args, **kwargs):
                iterator = function(*args, **kwargs)
                try:
                    while True:
                        with execution_owner_kind(kind):
                            try:
                                event = await anext(iterator)
                            except StopAsyncIteration:
                                break
                        yield event
                finally:
                    with execution_owner_kind(kind):
                        await iterator.aclose()

            return stream

        @wraps(function)
        async def call(*args, **kwargs):
            with execution_owner_kind(kind):
                return await function(*args, **kwargs)

        return call

    return decorate
