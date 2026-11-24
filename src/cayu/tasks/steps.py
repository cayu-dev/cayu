"""Saved step results for task handlers, keyed by task id and step name.

A task handler that does more than one piece of work can save each step's
result as it finishes. When the same task runs again, for example after a lost
worker or a restart, :meth:`TaskSteps.run` returns the saved result instead of
calling the step again, so the handler pays only for steps that never finished.

The parts stand alone. :class:`TaskStepStore` is the storage contract,
:class:`InMemoryTaskStepStore` is a process-local store for tests, and the
SQLite and PostgreSQL stores in :mod:`cayu.storage` keep results in the same
database as the task store. :class:`TaskSteps` is a small helper bound to one
task id. Nothing in the task worker requires any of them.

The first saved result for a step wins: a later save of the same step returns
the stored record unchanged, so two runs that race on one step agree on a
single result. Results are JSON values of at most
:data:`TASK_STEP_RESULT_MAX_BYTES` compact UTF-8 bytes. Store a reference (an
artifact id, a path, a commit sha) for anything larger.

A saved result says the step finished; it does not undo or repeat the step's
external effects. A step that changes another system should use a stable
idempotency key there when it has one, because the worker can be lost after the
effect and before the save. When the other system has no idempotency key,
:meth:`TaskSteps.act_once` claims the step before the effect, and a later run
that finds the claim without a result reconciles it with a probe instead of
acting again. The claim is an ordinary saved step under a reserved name, so
every :class:`TaskStepStore` supports it unchanged.
"""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Final
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from cayu._validation import (
    DURABLE_METADATA_LIMITS,
    canonical_bounded_durable_json_bytes,
    require_durable_clean_nonblank,
)

#: Largest saved step result, in compact UTF-8 JSON bytes.
TASK_STEP_RESULT_MAX_BYTES = 256 * 1024
#: Largest task id or step name, in UTF-8 bytes.
TASK_STEP_KEY_MAX_BYTES = 512
#: Largest page :meth:`TaskStepStore.list_steps` returns.
TASK_STEP_LIST_MAX = 1000
#: Prefix of the reserved step names that hold :meth:`TaskSteps.begin` claims.
TASK_STEP_CLAIM_PREFIX: Final = "cayu:claim:"


class _NotDone:
    """Type of :data:`TASK_STEP_NOT_DONE`; compare with ``is``."""

    def __repr__(self) -> str:
        return "TASK_STEP_NOT_DONE"


#: What a reconcile probe returns when the claimed effect provably never happened.
TASK_STEP_NOT_DONE: Final = _NotDone()


class TaskStepRecord(BaseModel):
    """One saved step result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    step: str
    result: Any
    saved_at: datetime


def task_step_key(value: object, field_name: str) -> str:
    """Validate a task id or step name the way every store does."""

    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string.")
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > TASK_STEP_KEY_MAX_BYTES:
        raise ValueError(f"{field_name} must not exceed {TASK_STEP_KEY_MAX_BYTES} UTF-8 bytes.")
    return value


def task_step_ids(task_id: object, step: object) -> tuple[str, str]:
    """Validate one ``(task_id, step)`` key."""

    return task_step_key(task_id, "task_id"), task_step_key(step, "step")


def optional_task_step_key(value: object, field_name: str) -> str | None:
    """Validate an optional step name, such as a list cursor or a single-step delete."""

    return None if value is None else task_step_key(value, field_name)


def encode_task_step_result(result: Any) -> str:
    """Return the canonical stored JSON text for ``result``, refusing oversized values."""

    return canonical_bounded_durable_json_bytes(
        result,
        "task step result",
        max_bytes=TASK_STEP_RESULT_MAX_BYTES,
        max_nodes=DURABLE_METADATA_LIMITS.max_nodes,
    ).decode("utf-8")


def task_step_list_limit(value: object) -> int:
    if type(value) is not int or not 1 <= value <= TASK_STEP_LIST_MAX:
        raise ValueError(f"limit must be an integer between 1 and {TASK_STEP_LIST_MAX}.")
    return value


def task_step_saved_at_ms() -> int:
    """Store-side save time in UTC milliseconds, the precision every backend keeps."""

    return int(datetime.now(UTC).timestamp() * 1000)


def task_step_record(task_id: str, step: str, result_json: str, saved_at_ms: int) -> TaskStepRecord:
    """Rebuild a record from its stored columns; every read returns a fresh copy."""

    return TaskStepRecord(
        task_id=task_id,
        step=step,
        result=json.loads(result_json),
        saved_at=datetime.fromtimestamp(saved_at_ms / 1000, tz=UTC),
    )


class TaskStepClaim(BaseModel):
    """A claim on one step, saved before its external effect.

    ``token`` is unique to the claim; pass it to the other system when it can
    carry a client reference, so a probe can find the effect later. ``data`` is
    any JSON the caller saved with the claim. ``acquired`` is true only for the
    :meth:`TaskSteps.begin` call that made the claim.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    step: str
    token: str
    data: Any = None
    claimed_at: datetime
    acquired: bool = False


class TaskStepUnresolved(RuntimeError):
    """A step is claimed and has no result, and nothing settled whether its effect happened."""

    def __init__(self, claim: TaskStepClaim) -> None:
        super().__init__(
            f"Step {claim.step!r} of task {claim.task_id!r} was claimed and has no result; "
            "reconcile it with a probe, or release the claim once you know the effect "
            "never happened."
        )
        self.claim = claim


class TaskStepStore(ABC):
    """Storage contract for saved task step results.

    Implementations key records by ``(task_id, step)``, keep the first saved
    result for a key, and validate keys and results with
    :func:`task_step_key` and :func:`encode_task_step_result` so every backend
    accepts and returns the same values. The store does not check that the task
    exists or who owns it; callers pass the id of the task they are running.
    """

    @abstractmethod
    async def load_step(self, task_id: str, step: str) -> TaskStepRecord | None:
        """Return the saved result for one step, or ``None`` when there is none."""

    @abstractmethod
    async def save_step(self, task_id: str, step: str, result: Any) -> TaskStepRecord:
        """Save ``result`` unless the step already has one, and return the stored record."""

    @abstractmethod
    async def list_steps(
        self, task_id: str, *, after: str | None = None, limit: int = 100
    ) -> list[TaskStepRecord]:
        """Return one task's saved steps ordered by step name, starting after ``after``."""

    @abstractmethod
    async def delete_steps(self, task_id: str, *, step: str | None = None) -> int:
        """Delete one step, or every step of the task, and return how many were removed."""

    async def close(self) -> None:  # noqa: B027 - optional hook; stores without resources skip it
        """Release backend resources. The default store holds none."""


class InMemoryTaskStepStore(TaskStepStore):
    """Process-local store for tests, with the same rules as the durable stores."""

    def __init__(self) -> None:
        self._steps: dict[tuple[str, str], tuple[str, int]] = {}

    async def load_step(self, task_id: str, step: str) -> TaskStepRecord | None:
        key = task_step_ids(task_id, step)
        stored = self._steps.get(key)
        return None if stored is None else task_step_record(*key, *stored)

    async def save_step(self, task_id: str, step: str, result: Any) -> TaskStepRecord:
        key = task_step_ids(task_id, step)
        encoded = encode_task_step_result(result)
        stored = self._steps.setdefault(key, (encoded, task_step_saved_at_ms()))
        return task_step_record(*key, *stored)

    async def list_steps(
        self, task_id: str, *, after: str | None = None, limit: int = 100
    ) -> list[TaskStepRecord]:
        task_id = task_step_key(task_id, "task_id")
        after = optional_task_step_key(after, "after")
        limit = task_step_list_limit(limit)
        steps = sorted(
            step
            for owner, step in self._steps
            if owner == task_id and (after is None or step > after)
        )
        return [
            task_step_record(task_id, step, *self._steps[(task_id, step)]) for step in steps[:limit]
        ]

    async def delete_steps(self, task_id: str, *, step: str | None = None) -> int:
        task_id = task_step_key(task_id, "task_id")
        if step is not None:
            return 0 if self._steps.pop(task_step_ids(task_id, step), None) is None else 1
        keys = [key for key in self._steps if key[0] == task_id]
        for key in keys:
            del self._steps[key]
        return len(keys)


class TaskSteps:
    """Run and save the steps of one task.

    ``await steps.run("checkout", fn)`` returns the saved result when the step
    already finished, and otherwise calls ``fn`` (sync or async, no
    arguments), saves its result and returns it. Either way the value comes
    from the store, so a first run and a later run get equal values. The result
    must be JSON: dicts with string keys, lists, strings, numbers, booleans and
    ``None``. An exception from ``fn`` saves nothing.
    """

    def __init__(self, store: TaskStepStore, task_id: str) -> None:
        if not isinstance(store, TaskStepStore):
            raise TypeError("store must be a TaskStepStore.")
        self._store = store
        self._task_id = task_step_key(task_id, "task_id")

    @property
    def task_id(self) -> str:
        return self._task_id

    async def run(self, step: str, fn: Callable[[], Awaitable[Any] | Any]) -> Any:
        step = _user_step(step)
        saved = await self._store.load_step(self._task_id, step)
        if saved is not None:
            return saved.result
        value = fn()
        if inspect.isawaitable(value):
            value = await value
        return (await self._store.save_step(self._task_id, step, value)).result

    async def load(self, step: str) -> TaskStepRecord | None:
        return await self._store.load_step(self._task_id, _user_step(step))

    async def save(self, step: str, result: Any) -> TaskStepRecord:
        return await self._store.save_step(self._task_id, _user_step(step), result)

    async def saved(self, *, after: str | None = None, limit: int = 100) -> list[TaskStepRecord]:
        """Return this task's saved steps ordered by step name, leaving out claims."""

        limit = task_step_list_limit(limit)
        found: list[TaskStepRecord] = []
        while len(found) < limit:
            page = await self._store.list_steps(self._task_id, after=after, limit=limit)
            found.extend(r for r in page if not r.step.startswith(TASK_STEP_CLAIM_PREFIX))
            if len(page) < limit:
                break
            after = page[-1].step
        return found[:limit]

    async def forget(self, step: str) -> bool:
        """Delete one saved step and its claim so the next :meth:`run` calls it again."""

        # Validate here: the store reads step=None as "every step of the task".
        step = _user_step(step)
        if _claimable(step):  # a longer name can never have had a claim
            await self._store.delete_steps(self._task_id, step=_claim_step(step))
        return await self._store.delete_steps(self._task_id, step=step) == 1

    async def begin(self, step: str, data: Any = None) -> TaskStepClaim:
        """Claim ``step`` before its effect, or return the claim that already exists.

        The first claim wins, as with results: the returned claim has
        ``acquired=True`` only when this call made it. A caller that did not
        acquire the claim must not act; an earlier run may already have.
        """

        step = _user_step(step)
        token = uuid4().hex
        record = await self._store.save_step(
            self._task_id, _claim_step(step), {"token": token, "data": data}
        )
        return _claim(record, step, token)

    async def claim(self, step: str) -> TaskStepClaim | None:
        """Return the saved claim on ``step``, or ``None`` when it has none."""

        step = _user_step(step)
        record = await self._store.load_step(self._task_id, _claim_step(step))
        return None if record is None else _claim(record, step)

    async def release(self, step: str) -> bool:
        """Delete the claim on ``step``; do this only when its effect never happened."""

        step = _user_step(step)
        return await self._store.delete_steps(self._task_id, step=_claim_step(step)) == 1

    async def reconcile(
        self, step: str, probe: Callable[[TaskStepClaim], Awaitable[Any] | Any]
    ) -> Any:
        """Settle a claimed step that has no result, by asking ``probe`` what happened.

        Returns the saved result when the step already has one. Otherwise calls
        ``probe(claim)`` (sync or async). A JSON value means the effect
        happened: it is saved as the step's result and returned.
        :data:`TASK_STEP_NOT_DONE` means it provably never happened: the claim
        is released and :data:`TASK_STEP_NOT_DONE` is returned, so the caller
        may claim and act again. An exception from the probe leaves the claim
        in place. Raises :class:`LookupError` when the step has no claim.
        """

        step = _user_step(step)
        saved = await self._store.load_step(self._task_id, step)
        if saved is not None:
            return saved.result
        claim = await self.claim(step)
        if claim is None:
            raise LookupError(f"Step {step!r} of task {self._task_id!r} has no claim.")
        outcome = probe(claim)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        if outcome is TASK_STEP_NOT_DONE:
            await self.release(step)
            return TASK_STEP_NOT_DONE
        return (await self._store.save_step(self._task_id, step, outcome)).result

    async def act_once(
        self,
        step: str,
        act: Callable[[TaskStepClaim], Awaitable[Any] | Any],
        *,
        reconcile: Callable[[TaskStepClaim], Awaitable[Any] | Any] | None = None,
        data: Any = None,
    ) -> Any:
        """Claim, act, save; or reconcile when an earlier run claimed without a result.

        Returns the saved result when the step finished. Otherwise claims the
        step, calls ``act(claim)`` and saves what it returns. When a claim
        already exists, ``reconcile(claim)`` decides as in :meth:`reconcile`,
        and the step is claimed and acted on again only after it returns
        :data:`TASK_STEP_NOT_DONE`. Without ``reconcile``, or when another run
        claims first, or when ``reconcile`` raises, raises
        :class:`TaskStepUnresolved` (with the probe's exception as its cause).
        An exception from ``act`` keeps the claim, since the effect may have
        happened.
        """

        step = _user_step(step)
        saved = await self._store.load_step(self._task_id, step)
        if saved is not None:
            return saved.result
        claim = await self.begin(step, data)
        if not claim.acquired:
            if reconcile is None:
                raise TaskStepUnresolved(claim)
            try:
                outcome = await self.reconcile(step, reconcile)
            except Exception as error:
                raise TaskStepUnresolved(claim) from error
            if outcome is not TASK_STEP_NOT_DONE:
                return outcome
            claim = await self.begin(step, data)
            if not claim.acquired:
                raise TaskStepUnresolved(claim)
        value = act(claim)
        if inspect.isawaitable(value):
            value = await value
        return (await self._store.save_step(self._task_id, step, value)).result

    async def clear(self) -> int:
        """Delete every saved step of this task, for example after it completes."""

        return await self._store.delete_steps(self._task_id)


def _user_step(step: object) -> str:
    step = task_step_key(step, "step")
    if step.startswith(TASK_STEP_CLAIM_PREFIX):
        raise ValueError(f"step names starting with {TASK_STEP_CLAIM_PREFIX!r} are reserved.")
    return step


def _claimable(step: str) -> bool:
    return len((TASK_STEP_CLAIM_PREFIX + step).encode("utf-8")) <= TASK_STEP_KEY_MAX_BYTES


def _claim_step(step: str) -> str:
    if not _claimable(step):
        limit = TASK_STEP_KEY_MAX_BYTES - len(TASK_STEP_CLAIM_PREFIX)
        raise ValueError(f"A claimed step name must not exceed {limit} UTF-8 bytes.")
    return TASK_STEP_CLAIM_PREFIX + step


def _claim(record: TaskStepRecord, step: str, token: str | None = None) -> TaskStepClaim:
    stored = record.result
    if type(stored) is not dict or type(stored.get("token")) is not str:
        raise ValueError(f"The claim on step {step!r} is not a claim record.")
    return TaskStepClaim(
        task_id=record.task_id,
        step=step,
        token=stored["token"],
        data=stored.get("data"),
        claimed_at=record.saved_at,
        acquired=stored["token"] == token,
    )


def task_steps(store: TaskStepStore, task: Any) -> TaskSteps:
    """Return :class:`TaskSteps` for a :class:`~cayu.tasks.records.Task` or a task id."""

    task_id = task if type(task) is str else getattr(task, "id", None)
    if type(task_id) is not str:
        raise TypeError("task must be a Task or a task id string.")
    return TaskSteps(store, task_id)


__all__ = [
    "TASK_STEP_CLAIM_PREFIX",
    "TASK_STEP_KEY_MAX_BYTES",
    "TASK_STEP_LIST_MAX",
    "TASK_STEP_NOT_DONE",
    "TASK_STEP_RESULT_MAX_BYTES",
    "InMemoryTaskStepStore",
    "TaskStepClaim",
    "TaskStepRecord",
    "TaskStepStore",
    "TaskStepUnresolved",
    "TaskSteps",
    "encode_task_step_result",
    "optional_task_step_key",
    "task_step_ids",
    "task_step_key",
    "task_steps",
]
