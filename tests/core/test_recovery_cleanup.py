from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from cayu import CayuApp
from cayu.core.messages import Message
from cayu.runtime._recovery_coordinator import _run_recovery_cleanup_steps
from cayu.runtime.recovery_cleanup import (
    RecoveryCleanupCapacityExceeded,
    RecoveryCleanupDeadlineExceeded,
    RecoveryCleanupDeadlineScope,
    RecoveryCleanupPolicy,
    RecoveryCleanupStep,
    RecoveryCleanupSupervisor,
)
from cayu.runtime.sessions import (
    CheckpointTransform,
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("step_timeout_seconds", 0),
        ("step_timeout_seconds", float("inf")),
        ("step_timeout_seconds", True),
        ("overall_timeout_seconds", -1),
        ("overall_timeout_seconds", float("nan")),
        ("overall_timeout_seconds", "1"),
        ("overall_timeout_seconds", 86_401),
        ("max_supervised_tasks", 0),
        ("max_supervised_tasks", True),
        ("max_supervised_tasks", 4097),
    ],
)
def test_recovery_cleanup_policy_rejects_unbounded_deadlines(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValidationError, ValueError)):
        RecoveryCleanupPolicy(**{field_name: value})


@pytest.mark.parametrize(
    "timeout_seconds",
    [0, -1, float("inf"), float("nan"), True, "1", 86_401],
)
def test_recovery_cleanup_drain_rejects_unbounded_deadline(timeout_seconds: object) -> None:
    async def scenario() -> None:
        supervisor = RecoveryCleanupSupervisor()
        with pytest.raises(ValueError, match="timeout_s"):
            await supervisor.drain(timeout_s=timeout_seconds)  # type: ignore[arg-type]

    asyncio.run(scenario())


def test_recovery_cleanup_rejects_unbounded_sequence_before_starting_work() -> None:
    async def scenario() -> None:
        started = False

        async def cleanup() -> None:
            nonlocal started
            started = True

        supervisor = RecoveryCleanupSupervisor()
        with pytest.raises(ValueError, match="64"):
            await supervisor.run_steps(
                steps=tuple((f"cleanup {index}", cleanup) for index in range(65)),
                shield_caller_cancellation=False,
            )
        assert started is False

    asyncio.run(scenario())


def test_recovery_cleanup_rejects_independent_step_without_predecessor() -> None:
    async def scenario() -> None:
        async def cleanup() -> None:
            raise AssertionError("invalid cleanup plan must not start")

        supervisor = RecoveryCleanupSupervisor()
        with pytest.raises(ValueError, match="cannot be independent"):
            await supervisor.run_steps(
                steps=(
                    RecoveryCleanupStep(
                        "invalid independent cleanup",
                        cleanup,
                        independent_with_previous=True,
                    ),
                ),
                shield_caller_cancellation=False,
            )

    asyncio.run(scenario())


def test_recovery_cleanup_timeout_attempts_later_independent_steps() -> None:
    async def scenario() -> None:
        release_hung_cleanup = asyncio.Event()
        cancellation_observed = asyncio.Event()
        attempted: list[str] = []

        async def hang_after_cancellation() -> None:
            while not release_hung_cleanup.is_set():
                try:
                    await release_hung_cleanup.wait()
                except asyncio.CancelledError:
                    cancellation_observed.set()

        def record_attempt(operation: str):
            async def run() -> None:
                attempted.append(operation)

            return run

        supervisor = RecoveryCleanupSupervisor(
            RecoveryCleanupPolicy(
                step_timeout_seconds=0.01,
                overall_timeout_seconds=0.1,
            )
        )
        failures = await _run_recovery_cleanup_steps(
            authoritative_failure=RuntimeError("authoritative runtime failure"),
            steps=(
                ("hung stream shutdown", hang_after_cancellation),
                RecoveryCleanupStep(
                    "independent finalizer shutdown",
                    record_attempt("finalizer"),
                    independent_with_previous=True,
                ),
                RecoveryCleanupStep(
                    "independent claim release",
                    record_attempt("claim"),
                    independent_with_previous=True,
                ),
                RecoveryCleanupStep(
                    "independent fence release",
                    record_attempt("fence"),
                    independent_with_previous=True,
                ),
                RecoveryCleanupStep(
                    "independent heartbeat stop",
                    record_attempt("heartbeat"),
                    independent_with_previous=True,
                ),
                RecoveryCleanupStep(
                    "independent environment cleanup",
                    record_attempt("environment"),
                    independent_with_previous=True,
                ),
                RecoveryCleanupStep(
                    "independent supervisor publication",
                    record_attempt("supervisor"),
                    independent_with_previous=True,
                ),
            ),
            supervisor=supervisor,
        )

        assert len(attempted) == 6
        assert set(attempted) == {
            "finalizer",
            "claim",
            "fence",
            "heartbeat",
            "environment",
            "supervisor",
        }
        assert len(failures) == 1
        operation, failure = failures[0]
        assert operation == "hung stream shutdown"
        assert isinstance(failure, RecoveryCleanupDeadlineExceeded)
        assert failure.code == "recovery_cleanup_deadline_exceeded"
        assert failure.scope is RecoveryCleanupDeadlineScope.STEP
        assert failure.operation == operation
        assert failure.evidence().model_dump(mode="json") == {
            "code": "recovery_cleanup_deadline_exceeded",
            "operation": "hung stream shutdown",
            "scope": "step",
            "timeout_seconds": 0.01,
            "outcome_unknown": True,
        }
        snapshot = supervisor.snapshot()
        assert snapshot.retained_tasks == 1
        assert len(snapshot.retained) == 1
        assert snapshot.retained[0].operation == operation
        assert snapshot.retained[0].scope is RecoveryCleanupDeadlineScope.STEP
        assert snapshot.retained[0].timeout_seconds == 0.01
        assert snapshot.retained[0].outcome_unknown is True
        assert snapshot.retained[0].caller_cancellation_observed is False
        await asyncio.wait_for(cancellation_observed.wait(), timeout=1)

        release_hung_cleanup.set()
        assert await supervisor.drain(timeout_s=1) is True
        snapshot = supervisor.snapshot()
        assert snapshot.retained_tasks == 0
        assert snapshot.timed_out_steps == 1
        assert snapshot.failed_after_timeout == 0

    asyncio.run(scenario())


def test_recovery_cleanup_independent_phase_reports_failures_in_declared_order() -> None:
    async def scenario() -> None:
        allow_first_failure = asyncio.Event()
        first_failure = RuntimeError("first cleanup failed")
        second_failure = ValueError("second cleanup failed")

        async def fail_first_after_second() -> None:
            await allow_first_failure.wait()
            raise first_failure

        async def fail_second_first() -> None:
            allow_first_failure.set()
            raise second_failure

        supervisor = RecoveryCleanupSupervisor()
        failures = await supervisor.run_steps(
            steps=(
                ("first independent cleanup", fail_first_after_second),
                RecoveryCleanupStep(
                    "second independent cleanup",
                    fail_second_first,
                    independent_with_previous=True,
                ),
            ),
            shield_caller_cancellation=False,
        )

        assert failures == (
            ("first independent cleanup", first_failure),
            ("second independent cleanup", second_failure),
        )

    asyncio.run(scenario())


def test_recovery_cleanup_completed_step_context_flows_to_later_step_and_caller() -> None:
    async def scenario() -> None:
        authority = ContextVar("test_recovery_cleanup_authority", default="initial")
        observed: list[str] = []

        async def replace_authority() -> None:
            authority.set("released")

        async def observe_authority() -> None:
            observed.append(authority.get())

        supervisor = RecoveryCleanupSupervisor(
            RecoveryCleanupPolicy(
                step_timeout_seconds=0.1,
                overall_timeout_seconds=0.2,
            )
        )
        assert (
            await supervisor.run_steps(
                steps=(
                    ("replace authority", replace_authority),
                    ("observe authority", observe_authority),
                ),
                shield_caller_cancellation=False,
            )
            == ()
        )
        assert observed == ["released"]
        assert authority.get() == "released"

    asyncio.run(scenario())


def test_recovery_cleanup_does_not_publish_timed_out_step_context() -> None:
    async def scenario() -> None:
        authority = ContextVar("test_timed_out_cleanup_authority", default="owned")
        release = asyncio.Event()
        observed: list[str] = []

        async def mutate_then_hang() -> None:
            authority.set("outcome-unknown")
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue

        async def observe_authority() -> None:
            observed.append(authority.get())

        supervisor = RecoveryCleanupSupervisor(
            RecoveryCleanupPolicy(
                step_timeout_seconds=0.01,
                overall_timeout_seconds=0.1,
            )
        )
        failures = await supervisor.run_steps(
            steps=(
                ("outcome-unknown authority mutation", mutate_then_hang),
                RecoveryCleanupStep(
                    "independent authority observation",
                    observe_authority,
                    independent_with_previous=True,
                ),
            ),
            shield_caller_cancellation=False,
        )

        assert isinstance(failures[0][1], RecoveryCleanupDeadlineExceeded)
        assert observed == ["owned"]
        assert authority.get() == "owned"

        release.set()
        assert await supervisor.drain(timeout_s=1) is True

    asyncio.run(scenario())


def test_recovery_cleanup_overall_deadline_retains_every_independent_phase_owner() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        started = [asyncio.Event(), asyncio.Event(), asyncio.Event()]
        cancelled = [asyncio.Event(), asyncio.Event(), asyncio.Event()]

        def step(index: int):
            async def run() -> None:
                started[index].set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        cancelled[index].set()
                        continue

            return run

        supervisor = RecoveryCleanupSupervisor(
            RecoveryCleanupPolicy(
                step_timeout_seconds=0.02,
                overall_timeout_seconds=0.01,
            )
        )
        failures = await supervisor.run_steps(
            steps=(
                ("cleanup 0", step(0)),
                RecoveryCleanupStep(
                    "cleanup 1",
                    step(1),
                    independent_with_previous=True,
                ),
                RecoveryCleanupStep(
                    "cleanup 2",
                    step(2),
                    independent_with_previous=True,
                ),
            ),
            shield_caller_cancellation=False,
        )

        assert len(failures) == 3
        assert all(isinstance(failure, RecoveryCleanupDeadlineExceeded) for _, failure in failures)
        assert any(
            failure.scope is RecoveryCleanupDeadlineScope.OVERALL
            for _, failure in failures
            if isinstance(failure, RecoveryCleanupDeadlineExceeded)
        )
        # Every explicitly independent owner in the active phase was attempted
        # before the sequence reached its overall deadline.
        await asyncio.sleep(0)
        assert all(signal.is_set() for signal in started)
        assert supervisor.snapshot().retained_tasks == 3
        await asyncio.wait_for(
            asyncio.gather(*(signal.wait() for signal in cancelled)),
            timeout=0.1,
        )

        release.set()
        assert await supervisor.drain(timeout_s=1) is True

    asyncio.run(scenario())


def test_recovery_cleanup_timeout_preserves_authoritative_failure() -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        async def hang() -> None:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue

        authoritative = RuntimeError("provider publication failed")
        supervisor = RecoveryCleanupSupervisor(
            RecoveryCleanupPolicy(
                step_timeout_seconds=0.01,
                overall_timeout_seconds=0.02,
            )
        )
        failures = await _run_recovery_cleanup_steps(
            authoritative_failure=authoritative,
            steps=(("provider stream shutdown", hang),),
            supervisor=supervisor,
        )

        assert len(failures) == 1
        assert isinstance(failures[0][1], RecoveryCleanupDeadlineExceeded)
        assert isinstance(authoritative.__cause__, BaseExceptionGroup)
        assert authoritative.__cause__.exceptions == (failures[0][1],)
        assert any(
            "original failure remains authoritative" in note for note in authoritative.__notes__
        )

        release.set()
        assert await supervisor.drain(timeout_s=1) is True

    asyncio.run(scenario())


def test_recovery_cleanup_caller_cancellation_is_bounded_and_retained() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        started = asyncio.Event()
        later_step_ran = False

        async def resist_cancellation() -> None:
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue

        async def later_step() -> None:
            nonlocal later_step_ran
            later_step_ran = True

        supervisor = RecoveryCleanupSupervisor(
            RecoveryCleanupPolicy(
                step_timeout_seconds=0.01,
                overall_timeout_seconds=0.1,
            )
        )
        cleanup_task = asyncio.create_task(
            _run_recovery_cleanup_steps(
                authoritative_failure=None,
                steps=(
                    ("cancellation-resistant stream close", resist_cancellation),
                    RecoveryCleanupStep(
                        "independent supervisor shutdown",
                        later_step,
                        independent_with_previous=True,
                    ),
                ),
                supervisor=supervisor,
            )
        )
        await started.wait()
        cleanup_task.cancel("operator cancelled cleanup")

        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(cleanup_task, timeout=0.2)
        assert raised.value.args == ("operator cancelled cleanup",)
        assert isinstance(raised.value.__cause__, RecoveryCleanupDeadlineExceeded)
        assert later_step_ran is True
        snapshot = supervisor.snapshot()
        assert snapshot.retained_tasks == 1
        assert snapshot.retained_after_cancellation == 1
        assert snapshot.retained[0].caller_cancellation_observed is True

        release.set()
        assert await supervisor.drain(timeout_s=1) is True

    asyncio.run(scenario())


def test_recovery_cleanup_capacity_fails_closed_without_starting_new_owner() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        rejected_step_started = False

        async def hang() -> None:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue

        async def rejected_step() -> None:
            nonlocal rejected_step_started
            rejected_step_started = True

        supervisor = RecoveryCleanupSupervisor(
            RecoveryCleanupPolicy(
                step_timeout_seconds=0.01,
                overall_timeout_seconds=0.1,
                max_supervised_tasks=1,
            )
        )
        failures = await supervisor.run_steps(
            steps=(
                ("retained owner", hang),
                RecoveryCleanupStep(
                    "capacity-rejected owner",
                    rejected_step,
                    independent_with_previous=True,
                ),
            ),
            shield_caller_cancellation=False,
        )

        assert len(failures) == 2
        failures_by_operation = dict(failures)
        assert isinstance(
            failures_by_operation["retained owner"],
            RecoveryCleanupDeadlineExceeded,
        )
        assert isinstance(
            failures_by_operation["capacity-rejected owner"],
            RecoveryCleanupCapacityExceeded,
        )
        assert rejected_step_started is False
        snapshot = supervisor.snapshot()
        assert snapshot.retained_tasks == 1
        assert snapshot.capacity_exhausted_steps == 1

        release.set()
        assert await supervisor.drain(timeout_s=1) is True

    asyncio.run(scenario())


def test_recovery_cleanup_capacity_is_reserved_before_concurrent_tasks_start() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        first_started = asyncio.Event()
        rejected_started = False

        async def active_cleanup() -> None:
            first_started.set()
            await release.wait()

        async def rejected_cleanup() -> None:
            nonlocal rejected_started
            rejected_started = True

        supervisor = RecoveryCleanupSupervisor(
            RecoveryCleanupPolicy(
                step_timeout_seconds=1,
                overall_timeout_seconds=2,
                max_supervised_tasks=1,
            )
        )
        first = asyncio.create_task(
            supervisor.run_steps(
                steps=(("active cleanup", active_cleanup),),
                shield_caller_cancellation=False,
            )
        )
        await first_started.wait()
        snapshot = supervisor.snapshot()
        assert snapshot.active_tasks == 1
        assert snapshot.retained_tasks == 0

        rejected = await supervisor.run_steps(
            steps=(("concurrent cleanup", rejected_cleanup),),
            shield_caller_cancellation=False,
        )
        assert len(rejected) == 1
        assert isinstance(rejected[0][1], RecoveryCleanupCapacityExceeded)
        assert rejected_started is False

        release.set()
        assert await first == ()
        assert supervisor.snapshot().active_tasks == 0

    asyncio.run(scenario())


def test_recovery_cleanup_drain_waits_for_active_supervised_work() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        started = asyncio.Event()

        async def active_cleanup() -> None:
            started.set()
            await release.wait()

        supervisor = RecoveryCleanupSupervisor(
            RecoveryCleanupPolicy(
                step_timeout_seconds=1,
                overall_timeout_seconds=2,
            )
        )
        cleanup = asyncio.create_task(
            supervisor.run_steps(
                steps=(("active cleanup", active_cleanup),),
                shield_caller_cancellation=False,
            )
        )
        await started.wait()

        assert await supervisor.drain(timeout_s=0.01) is False
        release.set()
        assert await supervisor.drain(timeout_s=1) is True
        assert await cleanup == ()
        assert supervisor.snapshot().active_tasks == 0

    asyncio.run(scenario())


def test_recovery_cleanup_defers_dependent_step_and_drain_follows_continuation() -> None:
    async def scenario() -> None:
        predecessor_started = asyncio.Event()
        predecessor_cancelled = asyncio.Event()
        release_predecessor = asyncio.Event()
        dependent_started = asyncio.Event()
        release_dependent = asyncio.Event()

        async def outcome_unknown_predecessor() -> None:
            predecessor_started.set()
            while not release_predecessor.is_set():
                try:
                    await release_predecessor.wait()
                except asyncio.CancelledError:
                    predecessor_cancelled.set()

        async def dependent_finalization() -> None:
            dependent_started.set()
            await release_dependent.wait()

        supervisor = RecoveryCleanupSupervisor(
            RecoveryCleanupPolicy(
                step_timeout_seconds=0.02,
                overall_timeout_seconds=0.2,
            )
        )
        failures = await supervisor.run_steps(
            steps=(
                ("outcome-unknown tool-round finalization", outcome_unknown_predecessor),
                ("dependent session finalization", dependent_finalization),
            ),
            shield_caller_cancellation=False,
        )

        assert isinstance(failures[0][1], RecoveryCleanupDeadlineExceeded)
        await asyncio.wait_for(predecessor_cancelled.wait(), timeout=1)
        assert dependent_started.is_set() is False

        drain = asyncio.create_task(supervisor.drain(timeout_s=0.5))
        release_predecessor.set()
        await asyncio.wait_for(dependent_started.wait(), timeout=1)
        await asyncio.sleep(0)
        assert drain.done() is False

        release_dependent.set()
        assert await drain is True
        assert supervisor.snapshot().active_tasks == 0
        assert supervisor.snapshot().retained_tasks == 0

    asyncio.run(scenario())


def test_recovery_cleanup_reissues_deadline_cancel_after_prior_step_consumes_caller_cancel() -> (
    None
):
    async def scenario() -> None:
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        second_cancelled = asyncio.Event()
        release_second = asyncio.Event()

        async def consume_caller_cancellation() -> None:
            first_started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                return

        async def hang_after_first_step() -> None:
            second_started.set()
            while not release_second.is_set():
                try:
                    await release_second.wait()
                except asyncio.CancelledError:
                    second_cancelled.set()

        supervisor = RecoveryCleanupSupervisor(
            RecoveryCleanupPolicy(
                step_timeout_seconds=0.02,
                overall_timeout_seconds=0.2,
            )
        )
        owner = asyncio.create_task(
            supervisor.run_steps(
                steps=(
                    ("caller-cancelled stream shutdown", consume_caller_cancellation),
                    ("later dependent shutdown", hang_after_first_step),
                ),
                shield_caller_cancellation=False,
            )
        )
        await first_started.wait()
        owner.cancel("operator cancelled cleanup")
        await asyncio.wait_for(second_started.wait(), timeout=1)

        failures = await asyncio.wait_for(owner, timeout=1)
        assert isinstance(failures[0][1], asyncio.CancelledError)
        assert failures[0][1].args == ("operator cancelled cleanup",)
        assert isinstance(failures[1][1], RecoveryCleanupDeadlineExceeded)
        await asyncio.wait_for(second_cancelled.wait(), timeout=1)
        assert supervisor.snapshot().retained_tasks == 1

        release_second.set()
        assert await supervisor.drain(timeout_s=1) is True

    asyncio.run(scenario())


def test_cayu_app_uses_one_configured_recovery_cleanup_supervisor() -> None:
    policy = RecoveryCleanupPolicy(
        step_timeout_seconds=1.5,
        overall_timeout_seconds=4.0,
        max_supervised_tasks=7,
    )
    app = CayuApp(recovery_cleanup_policy=policy, enable_logging=False)

    assert app._recovery_cleanup_policy == policy
    assert (
        app._recovery_coordinator._recovery_cleanup_supervisor
        is app._session_engine._recovery_cleanup_supervisor
        is app._recovery_cleanup_supervisor
    )
    assert app.recovery_cleanup_status().retained_tasks == 0


def test_timed_out_claim_release_converges_through_fresh_runtime() -> None:
    class BlockingClaimReleaseStore(InMemorySessionStore):
        def __init__(self, *, ownership_clock: Callable[[], datetime]) -> None:
            super().__init__(ownership_clock=ownership_clock)
            self.block_next_checkpoint_transform = False
            self.release_started = asyncio.Event()
            self.allow_release = asyncio.Event()

        async def transform_checkpoint(
            self,
            session_id: str,
            checkpoint_transform: CheckpointTransform,
        ) -> None:
            if self.block_next_checkpoint_transform:
                self.block_next_checkpoint_transform = False
                self.release_started.set()
                while not self.allow_release.is_set():
                    try:
                        await self.allow_release.wait()
                    except asyncio.CancelledError:
                        continue
            await super().transform_checkpoint(session_id, checkpoint_transform)

    async def scenario() -> None:
        current_time: dict[str, Any] = {
            "value": datetime(2026, 9, 2, tzinfo=UTC),
        }

        def clock() -> datetime:
            return current_time["value"]

        store = BlockingClaimReleaseStore(ownership_clock=clock)
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_timed_out_claim_release_restart",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        policy = RecoveryCleanupPolicy(
            step_timeout_seconds=0.01,
            overall_timeout_seconds=0.1,
        )
        original_app = CayuApp(
            session_store=store,
            recovery_cleanup_policy=policy,
            clock=clock,
            enable_logging=False,
        )
        original_claim = await original_app._recovery_coordinator._claim_incomplete_recovery(
            session=session,
            inactive_for_seconds=None,
        )
        assert original_claim is not None
        original_claim_id = original_claim.claim_id

        store.block_next_checkpoint_transform = True
        with pytest.raises(RecoveryCleanupDeadlineExceeded):
            await original_app._recovery_coordinator._cleanup_incomplete_recovery_claim(
                authority=original_claim.require_authority(),
                authoritative_failure=None,
            )
        await asyncio.wait_for(store.release_started.wait(), timeout=1)
        assert original_app.recovery_cleanup_status().retained_tasks == 1

        current_time["value"] += timedelta(minutes=6)
        current = await store.load(session.id)
        assert current is not None
        replacement_app = CayuApp(
            session_store=store,
            recovery_cleanup_policy=policy,
            clock=clock,
            enable_logging=False,
        )
        replacement_claim = await replacement_app._recovery_coordinator._claim_incomplete_recovery(
            session=current,
            inactive_for_seconds=None,
            required_expired_claim_id=original_claim_id,
        )
        assert replacement_claim is not None

        store.allow_release.set()
        assert await original_app.drain_recovery_cleanups(timeout_s=1) is True
        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is not None
        marker = checkpoint["incomplete_session_recovery_claim"]
        assert marker["claim_id"] == replacement_claim.claim_id

        await replacement_app._recovery_coordinator._cleanup_incomplete_recovery_claim(
            authority=replacement_claim.require_authority(),
            authoritative_failure=None,
        )
        final_checkpoint = await store.load_checkpoint(session.id)
        assert final_checkpoint is None or (
            "incomplete_session_recovery_claim" not in final_checkpoint
        )

    asyncio.run(scenario())
