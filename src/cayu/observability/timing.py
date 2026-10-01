"""Content-free, non-durable runtime timing; none of these records grants authority."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictInt

TimingPhaseName = Literal[
    "authorization",
    "admission",
    "started_persistence",
    "effect_state",
    "execution",
    "result_processing",
    "staging",
    "sibling_wait",
    "publication_queue_wait",
    "publication",
    "round_commit",
    "unattributed",
    "handoff",
    "context_policy",
    "recall",
    "counting",
    "preparation",
]
NonnegativeSeconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
TimingIdentifier = Annotated[str, Field(min_length=1, max_length=512)]
_CONFIG = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class RuntimeTimingConfig(BaseModel):
    """Bounded recent records and best-effort delivery; disable for the cheapest path."""

    model_config = _CONFIG
    enabled: bool = True
    recent_capacity: StrictInt = Field(default=64, ge=1, le=1024)
    max_calls_per_round: StrictInt = Field(default=128, ge=1, le=1024)
    sink_queue_capacity: StrictInt = Field(default=128, ge=1, le=1024)
    sink_timeout_seconds: float = Field(default=1, ge=0.01, le=30, allow_inf_nan=False)


class RuntimePhaseTiming(BaseModel):
    """Exclusive phase time; SQL execution includes database lock acquisition.

    ``first_started_at`` and ``last_completed_at`` bound the observed entries;
    a phase can be entered more than once, so the window can exceed its duration.
    """

    model_config = _CONFIG
    name: TimingPhaseName
    first_started_at: datetime | None = None
    last_completed_at: datetime | None = None
    duration_seconds: NonnegativeSeconds = 0
    store_transaction_count: StrictInt = Field(default=0, ge=0)
    store_lock_wait_seconds: NonnegativeSeconds = 0
    store_execution_seconds: NonnegativeSeconds = 0
    store_commit_seconds: NonnegativeSeconds = 0
    store_bytes_written: StrictInt = Field(default=0, ge=0)


class ToolCallTiming(BaseModel):
    """A call's observed phases, without arguments, results, prompts or exceptions."""

    model_config = _CONFIG
    session_id: TimingIdentifier
    tool_round_id: TimingIdentifier
    tool_call_id: TimingIdentifier
    tool_name: TimingIdentifier
    phases: tuple[RuntimePhaseTiming, ...] = Field(max_length=12)
    tool_effect_completed_at: datetime | None = None
    tool_terminal_staged_at: datetime | None = None
    tool_terminal_publication_started_at: datetime | None = None


class ToolRoundTiming(BaseModel):
    """Round totals include call phases; parallel elapsed time is not additive wall time."""

    model_config = _CONFIG
    schema_version: Literal[1] = 1
    kind: Literal["tool_round"] = "tool_round"
    session_id: TimingIdentifier
    tool_round_id: TimingIdentifier
    model_step_id: TimingIdentifier
    model_attempt_id: TimingIdentifier
    started_at: datetime
    completed_at: datetime
    duration_seconds: NonnegativeSeconds
    incomplete: bool
    recovered: bool = False
    calls_truncated: StrictInt = Field(default=0, ge=0)
    phases: tuple[RuntimePhaseTiming, ...] = Field(max_length=12)
    calls: tuple[ToolCallTiming, ...] = Field(max_length=1024)


class ModelStepPreparationTiming(BaseModel):
    """Observed gap to the next model start, split into exclusive preparation phases."""

    model_config = _CONFIG
    schema_version: Literal[1] = 1
    kind: Literal["model_step_preparation"] = "model_step_preparation"
    session_id: TimingIdentifier
    model_step_id: TimingIdentifier
    after_tool_round_id: TimingIdentifier | None
    started_at: datetime
    completed_at: datetime
    duration_seconds: NonnegativeSeconds
    incomplete: bool
    phases: tuple[RuntimePhaseTiming, ...] = Field(max_length=5)


RuntimeTimingRecord = ToolRoundTiming | ModelStepPreparationTiming


class RuntimeTimingSink(Protocol):
    """Receive best-effort observations without durable sink receipts or retries."""

    async def emit_timing(self, record: RuntimeTimingRecord) -> None: ...


class RuntimeTimingStatus(BaseModel):
    model_config = _CONFIG
    recent_records: int
    queued_records: int
    dropped_records: int
    failed_deliveries: int
