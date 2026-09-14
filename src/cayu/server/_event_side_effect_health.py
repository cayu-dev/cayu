"""Process-local recovery observations, isolated from event fan-out."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from cayu.runtime.event_side_effect_health import PersistedEventSideEffectHealth


class EventSideEffectRecoveryLoop(BaseModel):
    scope: Literal["process_local_reset_on_restart"] = "process_local_reset_on_restart"
    state: Literal["configured", "running", "stopped"] = "configured"
    started_at: datetime | None = None
    last_sweep_started_at: datetime | None = None
    last_sweep_completed_at: datetime | None = None
    last_success_at: datetime | None = None
    last_delivered_count: int | None = None
    last_duration_seconds: float | None = None
    interval_seconds: float
    batch_limit: int
    last_success_saturated: bool = False
    consecutive_failures: int = 0
    last_error: str | None = None
    last_error_at: datetime | None = None
    sweep_attempts: int = 0
    sweep_successes: int = 0
    sweep_failures: int = 0
    delivered_rows: int = 0
    saturated_batches: int = 0


class EventSideEffectHealthResponse(BaseModel):
    durable: PersistedEventSideEffectHealth
    recovery_loop: EventSideEffectRecoveryLoop | None = None
