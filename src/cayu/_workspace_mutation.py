"""Process-local quarantine for workspace mutations with uncertain settlement."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from cayu._exception_groups import exception_tree_contains, rebuild_exception_group

_BASE_EXCEPTION_ARGS_DESCRIPTOR = BaseException.__dict__["args"]
_MAX_PROCESS_SIGNAL_ARGS = 4


class WorkspaceMutationSettlementError(RuntimeError):
    """The runtime cannot prove that an invocation mutation has stopped."""


@dataclass(frozen=True, slots=True)
class _SettlementProbeOutcome:
    settled: bool
    process_signal: BaseException | None = None


@dataclass(eq=False, slots=True)
class _SettlementProbeOwner:
    scope: object
    probe: Callable[[], Awaitable[bool]]
    task: asyncio.Task[_SettlementProbeOutcome] | None = None
    active_waiters: int = 0
    process_signal_claimed: bool = False


class WorkspaceMutationProcessFence:
    """Retain every mutation uncertainty affecting one environment registration."""

    def __init__(
        self,
        *,
        _root: WorkspaceMutationProcessFence | None = None,
    ) -> None:
        if _root is None:
            self._root = self
            self._owners: list[_SettlementProbeOwner] = []
        else:
            self._root = _root._root
            self._owners = []
        self._scope = object()

    def child_fence(self) -> WorkspaceMutationProcessFence:
        """Return one concrete factory environment's settlement scope."""

        return WorkspaceMutationProcessFence(_root=self._root)

    def fail_closed(
        self,
        settlement_probe: Callable[[], Awaitable[bool]],
    ) -> None:
        """Quarantine this scope until every uncertain mutation proves quiescence."""

        if not callable(settlement_probe):
            raise TypeError("Workspace mutation settlement probe must be callable.")
        self._root._owners.append(
            _SettlementProbeOwner(
                scope=self._scope,
                probe=settlement_probe,
            )
        )

    async def wait_until_available(self) -> None:
        """Wait for retained positive evidence before admitting environment reuse."""

        while True:
            owners = self._selected_owners()
            if not owners:
                return
            current_loop = asyncio.get_running_loop()
            joined: list[tuple[_SettlementProbeOwner, asyncio.Task[_SettlementProbeOutcome]]] = []
            blocked_by_owned_outcome = False

            for owner in owners:
                task = owner.task
                if task is not None and not task.done() and task.get_loop() is not current_loop:
                    blocked_by_owned_outcome = True
                    continue
                if task is not None and task.done():
                    outcome = _settlement_probe_task_outcome(task)
                    if self._accept_probe_outcome(owner, task, outcome):
                        continue
                    if owner.active_waiters > 0:
                        # Another waiter already owns delivery of this exact
                        # completed generation. A later caller must not steal
                        # or duplicate its process-control outcome.
                        blocked_by_owned_outcome = True
                        continue
                    owner.task = None
                    owner.process_signal_claimed = False
                    task = None
                if task is None:
                    task = asyncio.create_task(
                        _run_settlement_probe(owner.probe),
                        name="cayu-workspace-mutation-settlement-probe",
                    )
                    owner.task = task
                    owner.process_signal_claimed = False
                owner.active_waiters += 1
                joined.append((owner, task))

            if blocked_by_owned_outcome:
                for owner, _task in joined:
                    owner.active_waiters -= 1
                raise _workspace_mutation_quarantine_error() from None
            if not joined:
                continue

            try:
                try:
                    outcomes = await asyncio.shield(
                        asyncio.gather(*(task for _owner, task in joined))
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    raise _workspace_mutation_quarantine_error() from None
                except BaseException:
                    # Probe-owned failures are captured inside
                    # _SettlementProbeOutcome. A control signal escaping this
                    # await therefore belongs to the supervising waiter and
                    # must remain authoritative while the same probe stays
                    # retained for a later join.
                    raise

                process_signals: list[BaseException] = []
                concurrent_failures: list[Exception] = []
                for (owner, task), outcome in zip(joined, outcomes, strict=True):
                    if self._accept_probe_outcome(owner, task, outcome):
                        continue
                    if outcome.process_signal is None:
                        concurrent_failures.append(_workspace_mutation_probe_failure())
                        continue
                    if owner.task is task and not owner.process_signal_claimed:
                        owner.process_signal_claimed = True
                        process_signals.append(
                            detached_workspace_mutation_process_signal(outcome.process_signal)
                            or _workspace_mutation_probe_failure()
                        )
                    else:
                        concurrent_failures.append(_workspace_mutation_probe_failure())

                if process_signals:
                    failures = [*process_signals, *concurrent_failures]
                    if len(failures) == 1:
                        raise process_signals[0] from None
                    raise BaseExceptionGroup(
                        "Workspace mutation settlement carried process-control failures.",
                        failures,
                    ) from None
                if concurrent_failures:
                    raise _workspace_mutation_quarantine_error() from None
            finally:
                for owner, _task in joined:
                    owner.active_waiters -= 1

    def require_available_nowait(self) -> None:
        """Reject unsafe finalization without waiting inside terminal cleanup."""

        for owner in self._selected_owners():
            task = owner.task
            if task is None or not task.done():
                raise _workspace_mutation_quarantine_error() from None
            outcome = _settlement_probe_task_outcome(task)
            if not self._accept_probe_outcome(owner, task, outcome):
                raise _workspace_mutation_quarantine_error() from None

    def _selected_owners(self) -> tuple[_SettlementProbeOwner, ...]:
        owners = self._root._owners
        if self is self._root:
            return tuple(owners)
        return tuple(owner for owner in owners if owner.scope is self._scope)

    def _accept_probe_outcome(
        self,
        owner: _SettlementProbeOwner,
        task: asyncio.Task[_SettlementProbeOutcome],
        outcome: _SettlementProbeOutcome,
    ) -> bool:
        owners = self._root._owners
        if owner not in owners:
            return outcome.settled
        if owner.task is not task or not outcome.settled:
            return False
        owners.remove(owner)
        owner.task = None
        return True


async def _run_settlement_probe(
    probe: Callable[[], Awaitable[bool]],
) -> _SettlementProbeOutcome:
    try:
        settled = await probe()
    except BaseException as exc:
        return _SettlementProbeOutcome(
            settled=False,
            process_signal=detached_workspace_mutation_process_signal(exc),
        )
    finally:
        del probe
    return _SettlementProbeOutcome(settled=settled is True)


def _settlement_probe_task_outcome(
    task: asyncio.Task[_SettlementProbeOutcome],
) -> _SettlementProbeOutcome:
    if task.cancelled():
        return _SettlementProbeOutcome(settled=False)
    try:
        outcome = task.result()
    except BaseException as exc:
        return _SettlementProbeOutcome(
            settled=False,
            process_signal=detached_workspace_mutation_process_signal(exc),
        )
    return outcome if type(outcome) is _SettlementProbeOutcome else _SettlementProbeOutcome(False)


def detached_workspace_mutation_process_signal(
    error: BaseException,
) -> BaseException | None:
    """Detach one extension-owned process signal from unsafe diagnostics."""

    if not exception_tree_contains(error, (KeyboardInterrupt, SystemExit)):
        return None
    if isinstance(error, BaseExceptionGroup):
        return rebuild_exception_group(
            error,
            group_message=("Workspace mutation settlement carried process-control failures."),
            leaf_mapper=_detached_process_signal_leaf,
            invalid_leaf_factory=lambda: RuntimeError(
                "Workspace mutation settlement probe failed concurrently."
            ),
        )
    return _detached_process_signal_leaf(error)


def _detached_process_signal_leaf(error: BaseException) -> BaseException:
    if isinstance(error, KeyboardInterrupt):
        return KeyboardInterrupt(*(_safe_process_signal_args(error) or ()))
    if isinstance(error, SystemExit):
        args = _safe_process_signal_args(error)
        # Unsafe extension-owned exit values must not cross the boundary, but
        # converting a non-empty SystemExit into code None would incorrectly
        # turn failure into a successful process exit.
        return SystemExit(*(args if args is not None else (1,)))
    return RuntimeError("Workspace mutation settlement probe failed concurrently.")


def _safe_process_signal_args(error: BaseException) -> tuple[object, ...] | None:
    try:
        args = _BASE_EXCEPTION_ARGS_DESCRIPTOR.__get__(error, BaseException)
    except BaseException:
        return None
    if type(args) is not tuple or len(args) > _MAX_PROCESS_SIGNAL_ARGS:
        return None
    if not all(value is None or type(value) in {bool, int, float} for value in args):
        return None
    return args


def _workspace_mutation_quarantine_error() -> WorkspaceMutationSettlementError:
    return WorkspaceMutationSettlementError("Workspace mutation settlement could not be proven.")


def _workspace_mutation_probe_failure() -> RuntimeError:
    return RuntimeError("Workspace mutation settlement probe failed concurrently.")


__all__ = [
    "WorkspaceMutationProcessFence",
    "WorkspaceMutationSettlementError",
    "detached_workspace_mutation_process_signal",
]
