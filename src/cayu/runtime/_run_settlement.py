"""Own the exit of one admitted invocation.

An admitted invocation holds its session's run fence until it settles. Any
yield, await or exception shape can end it, so ``RunSettlement`` owns every
exit with one rule: the body records what it decided in a ``RunLedger``; every
exit, however it happens, ends in one idempotent ``settle`` that reads durable
state and the ledger and finishes whatever is still owed, in a fixed order
(decide, carry out the decided outcome, environment, fence). The durable part
runs shielded from caller cancellation; the in-process part always runs in the
caller.
"""

from __future__ import annotations

import asyncio
import contextlib
import types
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any

from cayu.runtime._admission_outcome import invocation_command_proven_not_committed
from cayu.runtime._run_exit import RunExit, RunExitKind, classify_run_exit
from cayu.sessions._completion_finalization import pending_completion_finalization_from_checkpoint
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._invocation_scope import (
    _activate_session_interaction,
    _current_session_interaction_id,
    _deactivate_session_interaction,
)
from cayu.sessions._invocation_terminal_decision import (
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_decision_matches_active_profile,
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions._paired_writes import unsettled_completed_decision
from cayu.sessions.records import SessionStatus

if TYPE_CHECKING:
    from cayu.runtime._environment_lifecycle import EnvironmentLifecycle
    from cayu.runtime._invocation_lifecycle import InvocationContext
    from cayu.runtime._recovery_ownership import RecoveryOwnership
    from cayu.runtime._session_finalization import SessionFinalization
    from cayu.sessions._execution_profile_checkpoint import ActiveInvocationExecutionProfile
    from cayu.sessions.records import Session
    from cayu.sessions.store import SessionStore

_LIVE_STATUSES = frozenset(
    {SessionStatus.PENDING, SessionStatus.RUNNING, SessionStatus.INTERRUPTING}
)
# No write is legal after fence loss, and a process-level stop leaves the open
# invocation to recovery, as process loss would: settlement does nothing.
_UNSETTLED_EXITS = frozenset({RunExitKind.FENCED, RunExitKind.PROCESS_CONTROL})


def current_cancellation_requests() -> int:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return 0
    return 0 if task is None else task.cancelling()


class _Owed(Enum):
    NOTHING = "nothing"
    RECOVERY = "recovery"
    DECISION = "decision"
    COMPLETION = "completion"
    RELEASE = "release"


class OutcomeStage(IntEnum):
    """Monotonic settlement progress. ``settle`` resumes from the recorded stage."""

    ADMITTED = 0
    DECIDED = 1
    TERMINAL_PUBLISHED = 2
    FENCE_RELEASED = 3


@dataclass(slots=True)
class RunLedger:
    """Progress of one admitted invocation, shared by reference with its body.

    The body records progress synchronously at the point a durable write
    returns, never after a yield and never through a return value.
    """

    session_id: str
    run_epoch: int
    invocation_context: InvocationContext
    structured_output_pending: bool | None = None
    stage: OutcomeStage = OutcomeStage.ADMITTED
    remainder: Callable[[], Awaitable[bool | None]] | None = None
    failure_owned_by_body: bool = False
    """The body's failure handler owns an ordinary failure: never re-decide it."""
    before_abandonment: Callable[[], Awaitable[None]] | None = None
    """Publish state the body would publish on failure (deferred input)."""
    provider_free_exit: bool = False
    """The exit was rebuilt without provider state; settlement must not chain onto it."""
    completion: Callable[[Session], AsyncGenerator[Any, None] | None] | None = None
    """Publish a committed completion from its durable decision, for any exit."""
    cancellation_baseline: int = field(default_factory=current_cancellation_requests)
    local_release: list[Callable[[], None]] = field(default_factory=list)

    def rebind(self, context: InvocationContext) -> None:
        """Install a successor context synchronously, before any await."""

        if context.binding.session_id != self.session_id:
            raise ValueError("A successor context belongs to another session.")
        self.invocation_context = context
        self.run_epoch = context.binding.run_epoch

    def decided(self, remainder: Callable[[], Awaitable[bool | None]] | None = None) -> None:
        """Record a committed outcome decision and the work it still owes.

        ``remainder`` finishes the decided outcome (task, terminal event and
        hooks) when the body cannot. The body calls ``advance`` once it has
        finished that work itself.
        """

        self.remainder = remainder
        self.advance(OutcomeStage.DECIDED)

    def advance(self, stage: OutcomeStage) -> None:
        if stage > self.stage:
            self.stage = stage
        if stage >= OutcomeStage.TERMINAL_PUBLISHED:
            self.remainder = None

    def settlement_finishes_remainder(self, error: BaseException | None) -> bool:
        """Whether settlement, not the body's own cleanup, finishes this exit.

        A decided remainder is finished by settlement unless the exit is one
        settlement leaves alone (fence loss, process-level stop); the body then
        keeps its own environment and fence cleanup, as before a remainder.
        """

        return (
            self.remainder is not None
            and classify_run_exit(error, cancellation_baseline=self.cancellation_baseline).kind
            not in _UNSETTLED_EXITS
        )


async def drain_event_stream(stream: AsyncGenerator[Any, None]) -> bool:
    """Finish a publication whose events are persisted before they are yielded.

    Returns False when the generator had already ended: the body clears a
    remainder it finished, so that publication was ended by the exception the
    run exits with, inside one of its awaits, and must not be recorded as
    published. That exception, not a second one, reports it.
    """

    if isinstance(stream, types.AsyncGeneratorType) and stream.ag_frame is None:
        return False
    async with contextlib.aclosing(stream):
        async for _ in stream:
            pass
    return True


class RunSettlement:
    """Settle every exit of an admitted invocation through one path."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        session_finalization: SessionFinalization,
        environment_lifecycle: EnvironmentLifecycle,
        recovery_ownership: RecoveryOwnership,
        recovery_owns_failure: Callable[[BaseException, RunLedger], bool] = lambda error, ledger: (
            False
        ),
    ) -> None:
        self._recovery_owns_failure = recovery_owns_failure
        self._session_store = session_store
        self._session_finalization = session_finalization
        self._environment_lifecycle = environment_lifecycle
        self._recovery_ownership = recovery_ownership

    @contextlib.asynccontextmanager
    async def settling(self, ledger: RunLedger) -> AsyncIterator[RunLedger]:
        """Own every exit of the enclosed region, at any await or yield."""

        # A refusal recorded by an earlier owner of this session (a recovered
        # run, another invocation) is not evidence about this invocation.
        self._session_finalization.forget_refused_interaction_publications(ledger.session_id)
        error: BaseException | None = None
        try:
            yield ledger
        except BaseException as exit_error:
            error = exit_error
            raise
        finally:
            await self.finish(ledger, error)

    async def finish(self, ledger: RunLedger, error: BaseException | None) -> None:
        """Settle durably (shielded), then release in-process state."""

        run_exit = classify_run_exit(error, cancellation_baseline=ledger.cancellation_baseline)
        try:
            if run_exit.kind is not RunExitKind.RETURNED or ledger.remainder is not None:
                cause = None if error is None else (error.__cause__, error.__suppress_context__)
                failures = await self._recovery_ownership.run_cleanup_steps(
                    authoritative_failure=error,
                    steps=(("run settlement", lambda: self.settle(ledger, run_exit)),),
                    cancellation_baseline=ledger.cancellation_baseline,
                )
                if failures and ledger.provider_free_exit and error is not None:
                    # A settlement failure's frames hold the invocation context
                    # and its provider. Keep only the content-free notes.
                    assert cause is not None
                    error.__cause__, error.__suppress_context__ = cause
        finally:
            self.release_local(ledger)

    async def settle_lost_admission(
        self,
        error: BaseException,
        *,
        target_active_profile: ActiveInvocationExecutionProfile,
        ledger_for: Callable[[Session], RunLedger],
        cancellation_baseline: int,
    ) -> None:
        """Settle an admission whose call was interrupted after its commit.

        The exact target profile (unique interaction, next epoch) as the active
        profile proves this owner's command committed: a cancellation or close
        that arrived with the acknowledgement must not leave an admitted
        invocation that no owner settles. A failure the command proved did
        not commit (its receipt replay completed without a receipt) needs no
        re-read, so a rejected create never reads the session that won.
        """

        run_exit = classify_run_exit(error, cancellation_baseline=cancellation_baseline)
        if run_exit.kind in _UNSETTLED_EXITS or invocation_command_proven_not_committed(error):
            return
        admitted: list[RunLedger] = []

        async def reconcile() -> None:
            session_id = target_active_profile.session_id
            session = await self._session_store.load(session_id)
            if session is None or session.run_epoch != target_active_profile.run_epoch:
                return
            checkpoint = await self._session_store.load_checkpoint(session_id)
            if active_invocation_execution_profile_from_checkpoint(checkpoint) != (
                target_active_profile
            ):
                return
            ledger = ledger_for(session)
            admitted.append(ledger)
            await self.settle(ledger, run_exit)

        try:
            await self._recovery_ownership.run_cleanup_steps(
                authoritative_failure=error,
                steps=(("lost admission settlement", reconcile),),
                cancellation_baseline=cancellation_baseline,
            )
        finally:
            for ledger in admitted:
                self.release_local(ledger)

    async def settle(self, ledger: RunLedger, run_exit: RunExit) -> None:
        """Finish whatever this invocation still owes. Safe to call again."""

        if run_exit.kind in _UNSETTLED_EXITS or ledger.stage >= OutcomeStage.FENCE_RELEASED:
            return
        # A committed completion is finished from its decision whoever handled the error.
        owed = await self._owed_settlement(ledger)
        if owed is not _Owed.COMPLETION and ledger.stage < OutcomeStage.DECIDED:
            if run_exit.error is not None and self._recovery_owns_failure(run_exit.error, ledger):
                # The body deliberately left this invocation open for its recovery
                # owner (uncertain terminal evidence, an unsettled durable child
                # submission, a rejected interaction publication).
                return
            if ledger.failure_owned_by_body:
                # The body's own failure handler took this invocation's outcome; if
                # it could not finish, also because a cancellation interrupted it,
                # its durable state is left for recovery exactly as before.
                return
        if owed is _Owed.NOTHING:
            ledger.advance(OutcomeStage.FENCE_RELEASED)
            return
        if owed is _Owed.RECOVERY:
            return
        try:
            if owed is _Owed.DECISION:
                await self._decide_abandoned(ledger)
            # A completion that committed before the body recorded it publishes
            # what its decision owes before the release.
            if owed is _Owed.COMPLETION and await self._publish_owed_completion(ledger):
                ledger.advance(OutcomeStage.TERMINAL_PUBLISHED)
            if ledger.remainder is not None:
                published = await ledger.remainder()
                if published is False and run_exit.kind is RunExitKind.CANCELLED:
                    # A cancellation inside the publication ended it. Its decision
                    # still holds every event it owes, so a fresh publication
                    # finishes it; each step replays what is already durable.
                    published = await self._publish_owed_completion(ledger)
                if published is not False:
                    ledger.advance(OutcomeStage.TERMINAL_PUBLISHED)
        finally:
            # Abort the environment and release the fence even if the decision
            # or its remainder failed: the release itself keeps the fence when
            # no terminal outcome is durable. The failure is still reported.
            context = ledger.invocation_context
            try:
                await self._environment_lifecycle.abort_environment_setup(
                    session_id=ledger.session_id,
                    original_error=run_exit.error,
                    invocation_context=context,
                )
            finally:
                await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                    session_id=ledger.session_id,
                    invocation_context=context,
                )
        ledger.advance(OutcomeStage.FENCE_RELEASED)

    def release_local(self, ledger: RunLedger) -> None:
        """Retire this invocation's task-local state. Never awaits."""

        try:
            for release in reversed(ledger.local_release):
                release()
        finally:
            self._session_finalization.forget_refused_interaction_publications(ledger.session_id)
            _deactivate_session_interaction(ledger.session_id)

    async def _owed_settlement(self, ledger: RunLedger) -> _Owed:
        session = await self._session_store.load(ledger.session_id)
        binding = ledger.invocation_context.binding
        if (
            session is None
            or session.instance_id != binding.session_instance_id
            or session.run_epoch != ledger.run_epoch
        ):
            # Released, or re-fenced by another owner.
            return _Owed.NOTHING
        checkpoint = await self._session_store.load_checkpoint(ledger.session_id)
        active = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if active is None or active != ledger.invocation_context.active_profile:
            return _Owed.NOTHING
        if ledger.stage >= OutcomeStage.DECIDED:
            return _Owed.RELEASE
        completed = unsettled_completed_decision(checkpoint)
        if (
            ledger.completion is not None
            and session.status is SessionStatus.COMPLETED
            and completed is not None
            and invocation_terminal_decision_matches_active_profile(
                completed,
                session_id=session.id,
                session_instance_id=session.instance_id,
                run_epoch=session.run_epoch,
                interaction_id=active.interaction_id,
                execution_profile_fingerprint=active.profile.fingerprint,
            )
        ):
            return _Owed.COMPLETION
        settled = settled_invocation_terminal_decision_from_checkpoint(checkpoint)
        if (
            session.status not in _LIVE_STATUSES
            or invocation_terminal_decision_from_checkpoint(checkpoint) is not None
            or (settled is not None and settled.run_epoch == ledger.run_epoch)
            or pending_completion_finalization_from_checkpoint(checkpoint) is not None
        ):
            # The body, or another owner, committed this invocation's outcome
            # and owns the rest of it (its own release, terminal-decision
            # replay, completion finalization or recovery). Never re-decide it
            # and never retry its release on its behalf.
            return _Owed.RECOVERY
        return _Owed.DECISION

    async def _publish_owed_completion(self, ledger: RunLedger) -> bool:
        """Publish a completion from its decision while the session is still completed."""

        session = await self._session_store.load(ledger.session_id)
        if (
            ledger.completion is None
            or session is None
            or session.status is not SessionStatus.COMPLETED
            or unsettled_completed_decision(
                await self._session_store.load_checkpoint(ledger.session_id)
            )
            is None
        ):
            return False
        publication = ledger.completion(session)
        return publication is not None and await drain_event_stream(publication)

    async def _decide_abandoned(self, ledger: RunLedger) -> None:
        context = ledger.invocation_context
        if _current_session_interaction_id(ledger.session_id) is None:
            # Settlement may run where the body's attribution is not visible.
            _activate_session_interaction(ledger.session_id, context.binding.interaction_id)
        if ledger.before_abandonment is not None:
            await ledger.before_abandonment()
        await self._session_finalization.finalize_abandoned_session_by_id(
            ledger.session_id,
            registered_agent=context.registered_agent,
            registered_environment=context.registered_environment,
            execution_profile=context.profile,
            invocation_context=context,
            structured_output_pending=ledger.structured_output_pending,
        )
        ledger.advance(OutcomeStage.DECIDED)
