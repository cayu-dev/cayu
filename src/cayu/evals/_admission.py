"""Launch-scoped CLI admission pacing, independent of case timeout budgets."""

from __future__ import annotations

import asyncio
import math
from asyncio import sleep
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from pydantic import BaseModel, ConfigDict, Field

from cayu.deadlines import current_execution_deadline
from cayu.evals._inspection_documents import write_process_document


class TrialAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    trial_number: int = Field(ge=1)
    admitted_at: datetime
    monotonic_seconds: float = Field(ge=0, allow_inf_nan=False)


class LaunchScheduling(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    stagger_seconds: float = Field(ge=0, allow_inf_nan=False)
    admissions: tuple[TrialAdmission, ...] = ()


def validate_stagger_seconds(value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("stagger_seconds must be finite and nonnegative (seconds).")
    return float(value)


_ACTIVE: ContextVar[LaunchAdmission | None] = ContextVar("eval_launch_admission", default=None)


@contextmanager
def admission_scope(admission: LaunchAdmission | None) -> Iterator[None]:
    token = _ACTIVE.set(admission)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def current_launch_admission() -> LaunchAdmission | None:
    return _ACTIVE.get()


class LaunchAdmission:
    def __init__(
        self, stagger_seconds: float, *, directory: Path | None = None, worker: int = 0
    ) -> None:
        self.stagger_seconds = validate_stagger_seconds(stagger_seconds)
        self.directory = directory
        self.worker = worker
        self._lock = asyncio.Lock()
        self._last: float | None = None
        self.admissions: list[TrialAdmission] = []

    def evidence(self) -> LaunchScheduling:
        return LaunchScheduling(
            stagger_seconds=self.stagger_seconds, admissions=tuple(self.admissions)
        )

    @staticmethod
    def _check_admission() -> None:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError
        current_execution_deadline().require_admission("eval_trial_admission")

    def _try_admit(self, case_id: str, trial_number: int) -> float:
        self._check_admission()
        now = monotonic()
        delay = 0 if self._last is None else self.stagger_seconds - (now - self._last)
        if delay > 0:
            return delay
        self._last = now
        self.admissions.append(
            TrialAdmission(
                case_id=case_id,
                trial_number=trial_number,
                admitted_at=datetime.now(UTC),
                monotonic_seconds=now,
            )
        )
        return 0

    def _try_shared_admission(self, case_id: str, trial_number: int) -> float:
        # POSIX workers share a host monotonic clock. Never block the event loop
        # on another process or hold the file lock while waiting for a time slot.
        import fcntl

        assert self.directory is not None
        with (self.directory / "admission.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 0.02
            try:
                state = self.directory / "admission-clock.json"
                if state.exists():
                    self._last = float(state.read_text(encoding="utf-8"))
                    if not math.isfinite(self._last) or self._last < 0:
                        raise ValueError("Invalid launch admission clock.")
                delay = self._try_admit(case_id, trial_number)
                if delay == 0:
                    write_process_document(state, self._last)
                    write_process_document(
                        self.directory / f"admissions-{self.worker}.json",
                        self.evidence().model_dump(mode="json"),
                    )
                return delay
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    async def admit(self, case_id: str, trial_number: int) -> None:
        async with self._lock:
            while True:
                self._check_admission()
                delay = (
                    self._try_admit(case_id, trial_number)
                    if self.directory is None
                    else self._try_shared_admission(case_id, trial_number)
                )
                if delay == 0:
                    return
                await sleep(delay)
