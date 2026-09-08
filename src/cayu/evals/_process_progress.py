"""Optional process-CLI observations; never execution or recovery authority."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cayu.evals._inspection_documents import write_process_document

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cayu.evals.models import EvalTrialResult
    from cayu.runtime.sessions import SessionStore

_ACTIVE: ContextVar[ProcessEvalProgress | None] = ContextVar("eval_process_progress", default=None)
_TRIAL: ContextVar[tuple[str, int] | None] = ContextVar("eval_observed_trial", default=None)


class ProcessEvalProgress:
    def __init__(
        self,
        directory: Path,
        *,
        launch_id: str,
        index: int,
        fingerprint: str,
        case_ids: tuple[str, ...],
    ) -> None:
        self.path = directory / f"progress-{index}.json"
        self.launch_id = launch_id
        self.index = index
        self.fingerprint = fingerprint
        self.case_ids = frozenset(case_ids)
        self.trials: dict[tuple[str, int], dict[str, Any]] = {}
        self.failed = False

    def write(self) -> None:
        if self.failed:
            return
        try:
            write_process_document(
                self.path,
                {
                    "schema_version": 1,
                    "launch_id": self.launch_id,
                    "index": self.index,
                    "fingerprint": self.fingerprint,
                    "observed_at": datetime.now(UTC).isoformat(),
                    "trials": list(self.trials.values()),
                },
            )
        except (OSError, ValueError, TypeError):
            # Observation publication must not change a scientific result, mask
            # cancellation, or convert a completed tool effect into a retry.
            # Missing/stale observations are explicitly unknown to inspection.
            self.failed = True

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _ACTIVE.set(self)
        try:
            self.write()
            yield
        finally:
            _ACTIVE.reset(token)


@contextmanager
def observe_eval_trial(case_id: str, trial_number: int) -> Iterator[None]:
    progress = _ACTIVE.get()
    if progress is None:
        yield
        return
    if _TRIAL.get() is not None or case_id not in progress.case_ids or trial_number != 1:
        # Nested application evaluations must not replace an outer trial's locator.
        nested_token = _ACTIVE.set(None)
        try:
            yield
        finally:
            _ACTIVE.reset(nested_token)
        return
    key = (case_id, trial_number)
    token = _TRIAL.set(key)
    progress.trials[key] = {
        "case_id": case_id,
        "trial_number": trial_number,
        "state": "started",
        "started_at": datetime.now(UTC).isoformat(),
    }
    progress.write()
    try:
        yield
    except BaseException as exc:
        progress.trials[key].update(state="interrupted", exception_type=type(exc).__name__)
        progress.write()
        raise
    finally:
        _TRIAL.reset(token)


def observe_eval_session(store: SessionStore, session_id: str) -> None:
    progress, key = _ACTIVE.get(), _TRIAL.get()
    if progress is None or key is None:
        return
    from cayu.storage.sqlite import SQLiteSessionStore

    reference: dict[str, Any] = {"session_id": session_id, "sqlite_path": None}
    # Only the exact built-in adapter has this known portable locator. Other
    # stores remain inspectable through inspect_eval_sessions(store, ...).
    if type(store) is SQLiteSessionStore:
        paths = store.durable_state_paths()
        if len(paths) == 1:
            reference["sqlite_path"] = str(paths[0])
    progress.trials[key]["session"] = reference
    progress.write()


def observe_eval_trial_result(result: EvalTrialResult) -> None:
    progress, key = _ACTIVE.get(), _TRIAL.get()
    if progress is None or key is None:
        return
    progress.trials[key].update(
        state="finished",
        completed_at=result.completed_at.isoformat(),
        status=result.status.value,
        score=result.score,
        error=None if result.error is None else result.error[:4096],
    )
    progress.write()
