from __future__ import annotations

import asyncio
import threading
import time
from contextvars import Context
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from cayu.core import Event, Message
from cayu.core.events import EventType
from cayu.core.thinking import ThinkingConfig
from cayu.runtime import CayuApp, InMemorySessionStore, RunRequest, SessionIdentity
from cayu.runtime import _recovery_coordinator as recovery_coordinator
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import sessions as sessions_module
from cayu.runtime._recovery_coordinator import (
    _effective_approval_budget_limits,
    _effective_approval_max_steps,
    _effective_approval_retry_policy,
    _effective_approval_run_limits,
    _effective_approval_thinking,
    _IncompleteRecoveryClaimLost,
    _interrupted_tool_round_results,
    _recovery_abandonment_signal,
    _retain_abandoned_unreplayable_tool_round,
    _run_recovery_cleanup_steps,
    _task_cancellation_count,
)
from cayu.runtime.approvals import PendingToolApproval, PendingToolCallApproval
from cayu.runtime.budgets import BudgetLimit
from cayu.runtime.build_provenance import (
    RuntimeBuildArtifactKind,
    RuntimeBuildProvenance,
    RuntimeBuildProvenanceOrigin,
)
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.execution_units import ToolRoundIdentity
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.sessions import (
    RUNTIME_BUILD_PROVENANCE_METADATA_KEY,
    CheckpointTransform,
    Session,
    SessionStatus,
)
from cayu.runtime.stop_policy import RunLimits


def _pending_approval(**kwargs) -> PendingToolApproval:
    return PendingToolApproval(
        approval_id="appr_1",
        model_step_id=f"mstep_{'1' * 32}",
        model_attempt_id=f"matt_{'2' * 32}",
        tool_round_id=f"tround_{'3' * 32}",
        tool_call_id="call_1",
        tool_name="side_effect",
        agent_name="assistant",
        publish_arguments=True,
        tool_calls=[PendingToolCallApproval(tool_call_id="call_1", tool_name="side_effect")],
        **kwargs,
    )


def _budget_limit(max_estimated_cost: str) -> BudgetLimit:
    return BudgetLimit(
        max_estimated_cost=Decimal(max_estimated_cost),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("10"),
                ),
            )
        ),
    )


def _tool_round_identity() -> ToolRoundIdentity:
    return ToolRoundIdentity(
        model_step_id=f"mstep_{'1' * 32}",
        model_attempt_id=f"matt_{'2' * 32}",
        tool_round_id=f"tround_{'3' * 32}",
    )


def test_abandoned_opaque_tool_round_history_is_idempotent_and_lossless() -> None:
    first = {"tool_round_id": "round-one"}
    second = {"tool_round_id": "round-two"}

    retained = _retain_abandoned_unreplayable_tool_round({}, first)
    assert _retain_abandoned_unreplayable_tool_round(retained, first) == retained

    retained = _retain_abandoned_unreplayable_tool_round(retained, second)
    abandoned = retained["abandoned_unreplayable_tool_round"]
    assert abandoned["tool_round"] == second
    assert abandoned["prior_tool_rounds"] == [first]


def test_effective_approval_run_config_prefers_override_then_recorded_settings() -> None:
    persisted = _pending_approval(
        max_steps=9,
        limits=RunLimits(max_tool_calls=2, scope="session"),
        budget_limits=(_budget_limit("1.00"),),
        retry_policy=RetryPolicy(max_attempts=4),
    )
    legacy = _pending_approval()

    assert _effective_approval_max_steps(max_steps=3, pending_approval=persisted) == 3
    assert _effective_approval_run_limits(
        limits=RunLimits(max_tool_calls=5),
        pending_approval=persisted,
    ) == RunLimits(max_tool_calls=5)
    assert (
        _effective_approval_budget_limits(
            budget_limits=(),
            pending_approval=persisted,
        )
        == ()
    )
    override_policy = RetryPolicy(max_attempts=2)
    assert (
        _effective_approval_retry_policy(
            retry_policy=override_policy,
            pending_approval=persisted,
        )
        is override_policy
    )

    assert _effective_approval_max_steps(max_steps=None, pending_approval=persisted) == 9
    assert _effective_approval_run_limits(
        limits=None,
        pending_approval=persisted,
    ) == RunLimits(max_tool_calls=2, scope="session")
    assert _effective_approval_budget_limits(
        budget_limits=None,
        pending_approval=persisted,
    ) == (_budget_limit("1.00"),)
    assert _effective_approval_retry_policy(
        retry_policy=None,
        pending_approval=persisted,
    ) == RetryPolicy(max_attempts=4)

    for override in (None, 16, 64):
        with pytest.raises(ValueError, match="requires recorded invocation max_steps"):
            _effective_approval_max_steps(max_steps=override, pending_approval=legacy)


def test_effective_approval_thinking_restores_pending_run_config() -> None:
    pending = _pending_approval(thinking=ThinkingConfig(effort="high"))

    restored = _effective_approval_thinking(thinking=None, pending_approval=pending)
    assert restored is not None
    assert restored.effort == "high"

    override = ThinkingConfig(effort="low")
    assert _effective_approval_thinking(thinking=override, pending_approval=pending) is override


def test_interrupted_tool_round_results_attaches_artifacts_by_tool_call_id() -> None:
    # Parallel cleanup artifacts stay with their producing call. The unkeyed
    # sequential fallback belongs only to the first unfinished call.
    a = runtime_records.ToolCallRequest(id="A", name="tool_a", arguments={})
    b = runtime_records.ToolCallRequest(id="B", name="tool_b", arguments={})

    keyed = _interrupted_tool_round_results(
        tool_calls=[a, b],
        completed_outcomes=[],
        tool_round_identity=_tool_round_identity(),
        cancellation_artifacts_by_id={"B": [{"producer": "B"}]},
    )
    by_id = {outcome.call.id: outcome for outcome in keyed}
    assert by_id["B"].result.artifacts == [{"producer": "B"}]
    assert by_id["A"].result.artifacts == []

    fallback = _interrupted_tool_round_results(
        tool_calls=[a, b],
        completed_outcomes=[],
        tool_round_identity=_tool_round_identity(),
        cancellation_artifacts=[{"producer": "unknown"}],
    )
    by_id = {outcome.call.id: outcome for outcome in fallback}
    assert by_id["A"].result.artifacts == [{"producer": "unknown"}]
    assert by_id["B"].result.artifacts == []


def test_recovery_abandonment_signal_finds_nested_grouped_cancellation() -> None:
    cancellation = asyncio.CancelledError("cancel recovery")
    grouped = BaseExceptionGroup(
        "recovery failed during cancellation",
        [
            GeneratorExit(),
            RuntimeError("cleanup failed"),
            BaseExceptionGroup("nested", [cancellation]),
        ],
    )

    assert _recovery_abandonment_signal(grouped) is cancellation
    assert (
        _recovery_abandonment_signal(ExceptionGroup("ordinary", [RuntimeError("failed")])) is None
    )


def test_recovery_abandonment_signal_handles_deep_group_iteratively() -> None:
    cancellation = asyncio.CancelledError("cancel recovery")
    grouped: BaseException = cancellation
    for _ in range(1_500):
        grouped = BaseExceptionGroup("nested recovery failure", [grouped])

    assert _recovery_abandonment_signal(grouped) is cancellation


def test_recovery_cleanup_preserves_ordered_failures_under_cancellation() -> None:
    async def scenario() -> None:
        cancellation = asyncio.CancelledError("cancel recovery")
        prior_cause = LookupError("primary failure cause")
        cancellation.__cause__ = prior_cause
        first = RuntimeError("claim release failed")
        second = ValueError("fence release failed")

        async def fail(error: BaseException) -> None:
            raise error

        failures = await _run_recovery_cleanup_steps(
            authoritative_failure=cancellation,
            steps=(
                ("claim release", lambda: fail(first)),
                ("fence release", lambda: fail(second)),
            ),
        )

        assert failures == (
            ("claim release", first),
            ("fence release", second),
        )
        assert isinstance(cancellation.__cause__, BaseExceptionGroup)
        assert cancellation.__cause__.exceptions == (first, second)
        assert cancellation.__cause__.__cause__ is prior_cause

    asyncio.run(scenario())


def test_recovery_cleanup_promotes_fatal_failure_from_grouped_cancellation() -> None:
    class FatalRecoverySignal(BaseException):
        pass

    async def scenario() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel("cancel grouped recovery")
        with pytest.raises(asyncio.CancelledError) as delivered:
            await asyncio.sleep(0)
        cancellation = delivered.value
        sibling = RuntimeError("provider recovery also failed")
        authoritative = BaseExceptionGroup(
            "grouped recovery cancellation",
            [cancellation, sibling],
        )
        fatal = FatalRecoverySignal("cleanup process control")

        async def fail() -> None:
            raise fatal

        with pytest.raises(FatalRecoverySignal) as raised:
            await _run_recovery_cleanup_steps(
                authoritative_failure=authoritative,
                steps=(("fatal cleanup", fail),),
            )

        assert raised.value is fatal
        assert fatal.__cause__ is authoritative
        assert authoritative.exceptions == (cancellation, sibling)
        assert task.cancelling() == 1

    asyncio.run(scenario())


def test_recovery_cleanup_preserves_cancellation_and_cleanup_cause_graphs() -> None:
    class FatalRecoverySignal(BaseException):
        pass

    async def scenario() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel("cancel recovery with prior evidence")
        with pytest.raises(asyncio.CancelledError) as delivered:
            await asyncio.sleep(0)
        cancellation = delivered.value
        prior_cancellation_cause = LookupError("provider publication failed")
        cancellation.__cause__ = prior_cancellation_cause
        child_cancellation = asyncio.CancelledError("cleanup child cancelled")
        fatal = FatalRecoverySignal("cleanup process control")
        cleanup_group = BaseExceptionGroup(
            "cleanup cancellation and process control",
            [child_cancellation, fatal],
        )
        prior_cleanup_cause = ValueError("cleanup had prior evidence")
        cleanup_group.__cause__ = prior_cleanup_cause

        async def fail() -> None:
            raise cleanup_group

        with pytest.raises(BaseExceptionGroup) as raised:
            await _run_recovery_cleanup_steps(
                authoritative_failure=cancellation,
                steps=(("fatal cleanup", fail),),
            )

        assert raised.value is cleanup_group
        assert cancellation.__cause__ is prior_cancellation_cause
        assert isinstance(cleanup_group.__cause__, BaseExceptionGroup)
        assert cleanup_group.__cause__.exceptions == (
            cancellation,
            prior_cleanup_cause,
        )
        assert cleanup_group.exceptions == (child_cancellation, fatal)
        assert task.cancelling() == 1

    asyncio.run(scenario())


def test_recovery_cleanup_preserves_fatal_implicit_context_under_cancellation() -> None:
    class FatalRecoverySignal(BaseException):
        pass

    async def scenario() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel("cancel recovery with implicit cleanup context")
        with pytest.raises(asyncio.CancelledError) as delivered:
            await asyncio.sleep(0)
        cancellation = delivered.value
        prior_cleanup_failure = RuntimeError("provider cleanup failed first")
        fatal = FatalRecoverySignal("cleanup process control")

        async def fail() -> None:
            try:
                raise prior_cleanup_failure
            except RuntimeError:
                raise fatal  # noqa: B904 - exercise Python's implicit exception context

        with pytest.raises(FatalRecoverySignal) as raised:
            await _run_recovery_cleanup_steps(
                authoritative_failure=cancellation,
                steps=(("fatal cleanup", fail),),
            )

        assert raised.value is fatal
        assert isinstance(fatal.__cause__, BaseExceptionGroup)
        assert fatal.__cause__.exceptions == (
            cancellation,
            prior_cleanup_failure,
        )
        assert fatal.__context__ is prior_cleanup_failure
        assert fatal.__suppress_context__ is True
        assert task.cancelling() == 1

    asyncio.run(scenario())


def test_recovery_cleanup_does_not_treat_hidden_context_as_visible_evidence() -> None:
    class FatalRecoverySignal(BaseException):
        pass

    async def scenario() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel("cancel recovery with hidden cleanup context")
        with pytest.raises(asyncio.CancelledError) as delivered:
            await asyncio.sleep(0)
        cancellation = delivered.value
        fatal = FatalRecoverySignal("cleanup process control")

        async def fail() -> None:
            try:
                raise cancellation
            except asyncio.CancelledError:
                raise fatal from None

        with pytest.raises(FatalRecoverySignal) as raised:
            await _run_recovery_cleanup_steps(
                authoritative_failure=cancellation,
                steps=(("fatal cleanup", fail),),
            )

        assert raised.value is fatal
        assert fatal.__context__ is cancellation
        assert fatal.__suppress_context__ is True
        assert fatal.__cause__ is cancellation
        assert task.cancelling() == 1

    asyncio.run(scenario())


def test_recovery_cancellation_generation_ignores_handled_prior_cancel() -> None:
    async def scenario() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)

        boundary = _task_cancellation_count()
        cancellation = asyncio.CancelledError("unrelated grouped cancellation")
        grouped = BaseExceptionGroup(
            "mixed failure",
            [cancellation, RuntimeError("fan-out failed")],
        )
        assert (
            _recovery_abandonment_signal(
                grouped,
                cancellation_baseline=boundary,
            )
            is None
        )

        task.cancel()
        assert _task_cancellation_count() > boundary
        assert _recovery_abandonment_signal(grouped, cancellation_baseline=boundary) is cancellation
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)

    asyncio.run(scenario())


def test_initial_incomplete_recovery_claim_cannot_fence_replacement_owner() -> None:
    class PausingClaimStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, *, ownership_clock) -> None:
            super().__init__(ownership_clock=ownership_clock)
            self.first_claim_ready = asyncio.Event()
            self.release_first_claim = asyncio.Event()
            self.first_claim_paused = False

        async def reserve_stalled_run_recovery(
            self,
            session_id: str,
            *,
            statuses: set[SessionStatus],
            checkpoint_transform,
            **kwargs,
        ):
            reserved = await super().reserve_stalled_run_recovery(
                session_id,
                statuses=statuses,
                checkpoint_transform=checkpoint_transform,
                **kwargs,
            )
            if not self.first_claim_paused:
                self.first_claim_paused = True
                self.first_claim_ready.set()
                await self.release_first_claim.wait()
            return reserved

        async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
            checkpoint = await super().load_checkpoint(session_id)
            task = asyncio.current_task()
            marker = (
                None if checkpoint is None else checkpoint.get("incomplete_session_recovery_claim")
            )
            if (
                not self.first_claim_paused
                and task is not None
                and task.get_name() == "initial-claimant"
                and type(marker) is dict
            ):
                # This hook exercises the pre-fix two-operation path too: it
                # pauses after that path verified its checkpoint lease but
                # before it separately advanced the run epoch.
                self.first_claim_paused = True
                self.first_claim_ready.set()
                await self.release_first_claim.wait()
            return checkpoint

    async def scenario() -> None:
        current_time = {"value": datetime(2026, 7, 20, tzinfo=UTC)}

        def clock() -> datetime:
            return current_time["value"]

        store = PausingClaimStore(ownership_clock=clock)
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_initial_claim_replacement_race",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        first_app = CayuApp(session_store=store, clock=clock, enable_logging=False)
        replacement_app = CayuApp(session_store=store, clock=clock, enable_logging=False)

        first_task = asyncio.create_task(
            first_app._recovery_coordinator._claim_incomplete_recovery(
                session=session,
                inactive_for_seconds=None,
            ),
            name="initial-claimant",
        )
        await asyncio.wait_for(store.first_claim_ready.wait(), timeout=5)
        first_checkpoint = await InMemorySessionStore.load_checkpoint(store, session.id)
        assert first_checkpoint is not None
        first_marker = first_checkpoint["incomplete_session_recovery_claim"]
        assert type(first_marker) is dict

        current_time["value"] += timedelta(minutes=6)
        current = await store.load(session.id)
        assert current is not None
        replacement_claim = await replacement_app._recovery_coordinator._claim_incomplete_recovery(
            session=current,
            inactive_for_seconds=None,
            required_expired_claim_id=first_marker["claim_id"],
        )
        assert replacement_claim is not None

        try:
            store.release_first_claim.set()
            assert await asyncio.wait_for(first_task, timeout=5) is None
            assert (
                sessions_module._current_session_run_epoch(session.id)
                == replacement_claim.session.run_epoch
            )

            durable = await store.load(session.id)
            assert durable is not None
            assert durable.run_epoch == replacement_claim.session.run_epoch
            checkpoint = await store.load_checkpoint(session.id)
            assert checkpoint is not None
            replacement_marker = checkpoint["incomplete_session_recovery_claim"]
            assert type(replacement_marker) is dict
            assert replacement_marker["claim_id"] == replacement_claim.claim_id

            # The replacement worker still owns the durable epoch and can
            # write; the expired caller did not fence it while unwinding.
            await store.update_metadata(session.id, {"replacement_owner_wrote": True})
        finally:
            store.release_first_claim.set()
            await replacement_app._recovery_coordinator._cleanup_incomplete_recovery_claim(
                authority=replacement_claim.require_authority(),
                authoritative_failure=None,
            )
            assert sessions_module._current_session_run_epoch(session.id) is None

    asyncio.run(scenario())


def test_incomplete_recovery_claim_finalization_transfers_across_tasks() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_transferable_recovery_claim",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)

        first = await app._recovery_coordinator._claim_incomplete_recovery(
            session=session,
            inactive_for_seconds=None,
        )
        assert first is not None
        assert sessions_module._current_session_run_epoch(session.id) == first.session.run_epoch

        # A supervising task can finalize the durable claim and the exact
        # process-local owner acquired by this task. Context copying must not
        # leave the acquiring task fenced at the released epoch.
        await asyncio.create_task(
            app._recovery_coordinator._cleanup_incomplete_recovery_claim(
                authority=first.require_authority(),
                authoritative_failure=None,
            ),
            context=Context(),
        )
        assert sessions_module._current_session_run_epoch(session.id) is None
        assert first.require_authority().retire() is False

        current = await store.load(session.id)
        assert current is not None
        assert current.run_epoch == first.session.run_epoch + 1
        second = await app._recovery_coordinator._claim_incomplete_recovery(
            session=current,
            inactive_for_seconds=None,
        )
        assert second is not None
        await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
            authority=first.require_authority(),
            authoritative_failure=None,
        )
        assert sessions_module._current_session_run_epoch(session.id) == second.session.run_epoch
        second_checkpoint = await store.load_checkpoint(session.id)
        assert second_checkpoint is not None
        second_marker = second_checkpoint["incomplete_session_recovery_claim"]
        assert second_marker["claim_id"] == second.claim_id
        await asyncio.create_task(
            app._recovery_coordinator._cleanup_incomplete_recovery_claim(
                authority=second.require_authority(),
                authoritative_failure=None,
            )
        )
        assert sessions_module._current_session_run_epoch(session.id) is None

        # Reconstruction has no task-local context to inherit. The durable
        # state remains immediately claimable by a new coordinator as well.
        reconstructed = CayuApp(session_store=store, enable_logging=False)
        reconstructed_session = await store.load(session.id)
        assert reconstructed_session is not None
        third = await reconstructed._recovery_coordinator._claim_incomplete_recovery(
            session=reconstructed_session,
            inactive_for_seconds=None,
        )
        assert third is not None
        await reconstructed._recovery_coordinator._cleanup_incomplete_recovery_claim(
            authority=third.require_authority(),
            authoritative_failure=None,
        )
        assert sessions_module._current_session_run_epoch(session.id) is None

    asyncio.run(scenario())


def test_incomplete_recovery_claim_finalization_is_exactly_once() -> None:
    class CountingReleaseStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.release_calls = 0

        async def release_run_fence(self, session_id: str) -> None:
            self.release_calls += 1
            await super().release_run_fence(session_id)

    async def scenario() -> None:
        store = CountingReleaseStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_exactly_once_recovery_finalization",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        claim = await app._recovery_coordinator._claim_incomplete_recovery(
            session=session,
            inactive_for_seconds=None,
        )
        assert claim is not None
        authority = claim.require_authority()

        async def finalize() -> None:
            await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
                authority=authority,
                authoritative_failure=None,
            )

        inherited = asyncio.create_task(finalize())
        independent = asyncio.create_task(finalize(), context=Context())
        await asyncio.gather(inherited, independent)
        await finalize()

        assert store.release_calls == 1
        assert sessions_module._current_session_run_epoch(session.id) is None
        current = await store.load(session.id)
        assert current is not None
        assert current.run_epoch == claim.session.run_epoch + 1
        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is None or "incomplete_session_recovery_claim" not in checkpoint

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure_mode",
    ["before_commit", "after_commit"],
)
def test_incomplete_recovery_claim_marker_release_failure_is_retryable(
    failure_mode: str,
) -> None:
    class FailOnceClaimReleaseStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.fail_claim_release = False
            self.claim_release_attempts = 0

        async def transform_checkpoint(
            self,
            session_id: str,
            checkpoint_transform: CheckpointTransform,
        ) -> None:
            self.claim_release_attempts += 1
            if not self.fail_claim_release:
                await super().transform_checkpoint(session_id, checkpoint_transform)
                return
            self.fail_claim_release = False
            if failure_mode == "after_commit":
                await super().transform_checkpoint(session_id, checkpoint_transform)
            raise RuntimeError(f"claim release failed {failure_mode.replace('_', ' ')}")

    async def scenario() -> None:
        store = FailOnceClaimReleaseStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_claim_release_{failure_mode}",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        claim = await app._recovery_coordinator._claim_incomplete_recovery(
            session=session,
            inactive_for_seconds=None,
        )
        assert claim is not None
        authority = claim.require_authority()

        attempts_before_cleanup = store.claim_release_attempts
        store.fail_claim_release = True
        failure_label = failure_mode.replace("_", " ")
        with pytest.raises(RuntimeError, match=f"claim release failed {failure_label}"):
            await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
                authority=authority,
                authoritative_failure=None,
            )

        checkpoint_after_failure = await store.load_checkpoint(session.id)
        if failure_mode == "before_commit":
            assert checkpoint_after_failure is not None
            assert (
                checkpoint_after_failure["incomplete_session_recovery_claim"]["claim_id"]
                == claim.claim_id
            )
        else:
            assert checkpoint_after_failure is None or (
                "incomplete_session_recovery_claim" not in checkpoint_after_failure
            )

        if failure_mode == "before_commit":
            assert authority.run_fence.retired is False
            await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
                authority=authority,
                authoritative_failure=None,
            )
            expected_release_attempts = 2
        else:
            # Reconciliation confirms the exact marker is already gone, so no
            # caller-owned retry is required to retire process-local authority.
            assert authority.run_fence.retired is True
            expected_release_attempts = 1
        await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
            authority=authority,
            authoritative_failure=None,
        )

        assert store.claim_release_attempts - attempts_before_cleanup == expected_release_attempts
        checkpoint_after_retry = await store.load_checkpoint(session.id)
        assert checkpoint_after_retry is None or (
            "incomplete_session_recovery_claim" not in checkpoint_after_retry
        )
        assert authority.run_fence.retired is True
        assert sessions_module._current_session_run_epoch(session.id) is None

    asyncio.run(scenario())


def test_claim_release_ack_loss_confirms_transfer_after_fence_failure() -> None:
    class LoseCleanupAcknowledgementStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.fail_fence_release = False
            self.lose_claim_release_acknowledgement = False

        async def release_run_fence(self, session_id: str) -> None:
            if self.fail_fence_release:
                self.fail_fence_release = False
                raise RuntimeError("recovery fence release unavailable")
            await super().release_run_fence(session_id)

        async def transform_checkpoint(
            self,
            session_id: str,
            checkpoint_transform: CheckpointTransform,
        ) -> None:
            await super().transform_checkpoint(session_id, checkpoint_transform)
            if self.lose_claim_release_acknowledgement:
                self.lose_claim_release_acknowledgement = False
                raise RuntimeError("claim release acknowledgement lost")

    async def scenario() -> None:
        store = LoseCleanupAcknowledgementStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_claim_release_ack_loss_after_fence_failure",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        first = await app._recovery_coordinator._claim_incomplete_recovery(
            session=session,
            inactive_for_seconds=None,
        )
        assert first is not None
        authority = first.require_authority()

        store.fail_fence_release = True
        store.lose_claim_release_acknowledgement = True
        with pytest.raises(RuntimeError, match="recovery fence release unavailable"):
            await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
                authority=authority,
                authoritative_failure=None,
            )

        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is None or "incomplete_session_recovery_claim" not in checkpoint
        assert authority.run_fence.retired is True
        assert sessions_module._current_session_run_epoch(session.id) is None

        # The first finalizer is gone. Durable reconciliation alone must make
        # the session immediately claimable without invoking it a second time.
        current = await store.load(session.id)
        assert current is not None
        replacement = await app._recovery_coordinator._claim_incomplete_recovery(
            session=current,
            inactive_for_seconds=None,
        )
        assert replacement is not None
        await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
            authority=replacement.require_authority(),
            authoritative_failure=None,
        )
        assert sessions_module._current_session_run_epoch(session.id) is None

    asyncio.run(scenario())


def test_claim_release_cancellation_after_commit_retires_authority() -> None:
    class BlockAfterClaimReleaseCommitStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.block_claim_release = False
            self.claim_release_committed = asyncio.Event()
            self.allow_claim_release_return = asyncio.Event()

        async def transform_checkpoint(
            self,
            session_id: str,
            checkpoint_transform: CheckpointTransform,
        ) -> None:
            await super().transform_checkpoint(session_id, checkpoint_transform)
            if self.block_claim_release:
                self.block_claim_release = False
                self.claim_release_committed.set()
                await self.allow_claim_release_return.wait()

    async def scenario() -> None:
        store = BlockAfterClaimReleaseCommitStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_claim_release_cancelled_after_commit",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        claim = await app._recovery_coordinator._claim_incomplete_recovery(
            session=session,
            inactive_for_seconds=None,
        )
        assert claim is not None
        authority = claim.require_authority()

        store.block_claim_release = True
        cleanup_task = asyncio.create_task(
            app._recovery_coordinator._cleanup_incomplete_recovery_claim(
                authority=authority,
                authoritative_failure=None,
            )
        )
        await asyncio.wait_for(store.claim_release_committed.wait(), timeout=5)
        cleanup_task.cancel("cancel after claim release commit")
        with pytest.raises(asyncio.CancelledError) as cancellation:
            await cleanup_task
        assert cancellation.value.args == ("cancel after claim release commit",)

        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is None or "incomplete_session_recovery_claim" not in checkpoint
        assert authority.run_fence.retired is True
        assert sessions_module._current_session_run_epoch(session.id) is None

        current = await store.load(session.id)
        assert current is not None
        replacement = await app._recovery_coordinator._claim_incomplete_recovery(
            session=current,
            inactive_for_seconds=None,
        )
        assert replacement is not None
        await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
            authority=replacement.require_authority(),
            authoritative_failure=None,
        )

    asyncio.run(scenario())


def test_incomplete_recovery_fence_release_failure_transfers_immediately() -> None:
    class FailOnceFenceReleaseStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.fail_fence_release = False

        async def release_run_fence(self, session_id: str) -> None:
            if self.fail_fence_release:
                self.fail_fence_release = False
                raise RuntimeError("recovery fence release unavailable")
            await super().release_run_fence(session_id)

    async def scenario() -> None:
        store = FailOnceFenceReleaseStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_failed_fence_release_transfer",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        first = await app._recovery_coordinator._claim_incomplete_recovery(
            session=session,
            inactive_for_seconds=None,
        )
        assert first is not None

        store.fail_fence_release = True
        with pytest.raises(RuntimeError, match="recovery fence release unavailable"):
            await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
                authority=first.require_authority(),
                authoritative_failure=None,
            )

        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is None or "incomplete_session_recovery_claim" not in checkpoint
        current = await store.load(session.id)
        assert current is not None
        replacement = await app._recovery_coordinator._claim_incomplete_recovery(
            session=current,
            inactive_for_seconds=None,
        )
        assert replacement is not None
        await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
            authority=replacement.require_authority(),
            authoritative_failure=None,
        )
        assert sessions_module._current_session_run_epoch(session.id) is None

    asyncio.run(scenario())


def test_initial_incomplete_recovery_claim_cannot_fence_after_its_lease_expires() -> None:
    class PausingClaimStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, *, ownership_clock) -> None:
            super().__init__(ownership_clock=ownership_clock)
            self.reservation_ready = asyncio.Event()
            self.release_reservation = asyncio.Event()

        async def reserve_stalled_run_recovery(self, *args, **kwargs):
            reserved = await super().reserve_stalled_run_recovery(*args, **kwargs)
            if not self.reservation_ready.is_set():
                self.reservation_ready.set()
                await self.release_reservation.wait()
            return reserved

    async def scenario() -> None:
        current_time = {"value": datetime(2026, 7, 20, tzinfo=UTC)}

        def clock() -> datetime:
            return current_time["value"]

        store = PausingClaimStore(ownership_clock=clock)
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_initial_claim_expired_before_fence",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, clock=clock, enable_logging=False)

        stale_claim = asyncio.create_task(
            app._recovery_coordinator._claim_incomplete_recovery(
                session=session,
                inactive_for_seconds=None,
            )
        )
        await asyncio.wait_for(store.reservation_ready.wait(), timeout=5)
        current_time["value"] += timedelta(minutes=6)
        store.release_reservation.set()

        assert await asyncio.wait_for(stale_claim, timeout=5) is None
        durable = await store.load(session.id)
        assert durable is not None
        assert durable.run_epoch == session.run_epoch
        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is None or "incomplete_session_recovery_claim" not in checkpoint

        retry = await app._recovery_coordinator._claim_incomplete_recovery(
            session=durable,
            inactive_for_seconds=None,
        )
        assert retry is not None
        await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
            authority=retry.require_authority(),
            authoritative_failure=None,
        )

    asyncio.run(scenario())


def test_incomplete_recovery_rejects_renewal_acknowledged_after_full_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            recovery_coordinator,
            "_INCOMPLETE_RECOVERY_CLAIM_LEASE",
            timedelta(milliseconds=50),
        )
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_recovery_renewal_ack_consumed_lease",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        coordinator = app._recovery_coordinator
        original_renew = coordinator._renew_incomplete_recovery_claim

        async def delayed_renew(session_id: str, claim_id: str):
            renewed = await original_renew(session_id, claim_id)
            await asyncio.sleep(0.06)
            return renewed

        monkeypatch.setattr(coordinator, "_renew_incomplete_recovery_claim", delayed_renew)
        with pytest.raises(
            _IncompleteRecoveryClaimLost,
            match="acknowledgement consumed its lease",
        ):
            await coordinator._claim_incomplete_recovery(
                session=session,
                inactive_for_seconds=None,
            )

        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is None or "incomplete_session_recovery_claim" not in checkpoint

    asyncio.run(scenario())


def test_terminal_finalization_claim_uses_store_time_under_worker_clock_skew() -> None:
    async def scenario() -> None:
        store_time = {"value": datetime(2026, 8, 1, tzinfo=UTC)}
        worker_time = {"value": datetime(2100, 1, 1, tzinfo=UTC)}
        store = InMemorySessionStore(ownership_clock=lambda: store_time["value"])
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_terminal_claim_store_time",
                messages=[Message.text("user", "interrupt")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        payload = {
            "reason": "operator",
            "interruption_type": "operator_requested",
            "interruption_request_id": "interrupt-store-time",
        }
        await store.checkpoint(
            session.id,
            {
                recovery_coordinator._PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY: payload,
            },
        )
        app = CayuApp(
            session_store=store,
            clock=lambda: worker_time["value"],
            enable_logging=False,
        )
        claim = await app._recovery_coordinator._claim_pending_terminal_evidence_finalization(
            session=session,
            expected_payload=payload,
        )
        assert claim is not None
        assert claim.claim_expires_at == store_time["value"] + timedelta(minutes=5)

        worker_time["value"] = datetime(2000, 1, 1, tzinfo=UTC)
        store_time["value"] += timedelta(minutes=6)
        assert (
            await app._recovery_coordinator._renew_terminal_evidence_finalization_claim(
                session=session,
                claim_id=claim.claim_id,
                expected_payload=payload,
            )
            is None
        )

    asyncio.run(scenario())


def test_terminal_finalization_claim_must_remain_live_at_commit_time() -> None:
    async def scenario() -> None:
        store_now = datetime(2026, 8, 1, tzinfo=UTC)
        store = InMemorySessionStore(ownership_clock=lambda: store_now)
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_terminal_claim_commit_expiry",
                messages=[Message.text("user", "interrupt")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        session = await store.update_status(created.id, SessionStatus.INTERRUPTED)
        claim_id = "terminal-claim-commit-expiry"
        claim_expires_at = store_now + timedelta(minutes=5)
        pending_payload = {
            "reason": "operator",
            "interruption_type": "operator_requested",
            "interruption_request_id": "interrupt-commit-expiry",
        }
        await store.checkpoint(
            session.id,
            {
                recovery_coordinator._PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY: pending_payload,
                recovery_coordinator._INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY: {
                    "version": 1,
                    "claim_id": claim_id,
                    "claimed_at": store_now.isoformat(),
                    "claim_expires_at": claim_expires_at.isoformat(),
                },
            },
        )
        app = CayuApp(session_store=store, enable_logging=False)

        async def expire_between_transform_and_commit(
            session_id: str,
            *,
            checkpoint_transform,
            commit_time_guard,
            **_kwargs,
        ):
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            transformed = checkpoint_transform(
                session,
                checkpoint,
                claim_expires_at - timedelta(microseconds=1),
            )
            assert recovery_coordinator._PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY not in transformed
            commit_time_guard(claim_expires_at)
            raise AssertionError("Expired finalization claim reached durable publication.")

        store.publish_checkpoint_and_events_with_store_time = (  # type: ignore[method-assign]
            expire_between_transform_and_commit
        )
        event = Event(
            type=EventType.SESSION_INTERRUPTED,
            session_id=session.id,
            payload=pending_payload,
        )
        with pytest.raises(
            _IncompleteRecoveryClaimLost,
            match="expired before durable publication",
        ):
            await app._session_engine._publish_terminal_event_under_finalization_claim(
                event=event,
                session=session,
                claim_id=claim_id,
                expected_payload=pending_payload,
            )
        retained = await store.load_checkpoint(session.id)
        assert retained is not None
        assert (
            retained[recovery_coordinator._PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY]
            == pending_payload
        )

    asyncio.run(scenario())


def test_expired_local_recovery_claim_never_starts_recovery_callback() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_expired_local_recovery_callback",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        claim = await app._recovery_coordinator._claim_incomplete_recovery(
            session=session,
            inactive_for_seconds=None,
        )
        assert claim is not None
        expired_claim = replace(claim, local_lease_deadline=time.monotonic() - 1)
        recovery_started = False

        async def recovery():
            nonlocal recovery_started
            recovery_started = True
            raise AssertionError("Expired recovery authority must not dispatch work.")

        with pytest.raises(_IncompleteRecoveryClaimLost):
            await app._recovery_coordinator._recover_incomplete_session_with_heartbeat(
                claim=expired_claim,
                recovery=recovery,
            )
        assert recovery_started is False
        await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
            authority=claim.require_authority(),
            authoritative_failure=None,
        )

    asyncio.run(scenario())


def test_initial_incomplete_recovery_reconciles_ambiguous_atomic_claim() -> None:
    class CommitThenRaiseClaimStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.lose_claim_acknowledgement = True

        async def fence_run_and_transform_checkpoint(
            self,
            session_id: str,
            *,
            statuses: set[SessionStatus],
            checkpoint_transform: CheckpointTransform,
            **kwargs,
        ) -> Session:
            fenced = await super().fence_run_and_transform_checkpoint(
                session_id,
                statuses=statuses,
                checkpoint_transform=checkpoint_transform,
                **kwargs,
            )
            if self.lose_claim_acknowledgement:
                self.lose_claim_acknowledgement = False
                raise RuntimeError("initial recovery claim acknowledgement lost")
            return fenced

    async def scenario() -> None:
        store = CommitThenRaiseClaimStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_initial_claim_acknowledgement_lost",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)

        with pytest.raises(
            RuntimeError,
            match="initial recovery claim acknowledgement lost",
        ):
            await app._recovery_coordinator._claim_incomplete_recovery(
                session=session,
                inactive_for_seconds=None,
            )

        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is None or "incomplete_session_recovery_claim" not in checkpoint

        current = await store.load(session.id)
        assert current is not None
        retry_claim = await app._recovery_coordinator._claim_incomplete_recovery(
            session=current,
            inactive_for_seconds=None,
        )
        assert retry_claim is not None
        await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
            authority=retry_claim.require_authority(),
            authoritative_failure=None,
        )

    asyncio.run(scenario())


def test_initial_incomplete_recovery_releases_ambiguous_store_time_reservation() -> None:
    class CommitThenRaiseReservationStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.lose_reservation_acknowledgement = True

        async def reserve_stalled_run_recovery(self, *args, **kwargs):
            reserved = await super().reserve_stalled_run_recovery(*args, **kwargs)
            if self.lose_reservation_acknowledgement:
                self.lose_reservation_acknowledgement = False
                raise RuntimeError("store-time reservation acknowledgement lost")
            return reserved

    async def scenario() -> None:
        store = CommitThenRaiseReservationStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_store_time_reservation_acknowledgement_lost",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)

        with pytest.raises(
            RuntimeError,
            match="store-time reservation acknowledgement lost",
        ):
            await app._recovery_coordinator._claim_incomplete_recovery(
                session=session,
                inactive_for_seconds=None,
            )

        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is None or "incomplete_session_recovery_claim" not in checkpoint
        current = await store.load(session.id)
        assert current is not None
        assert current.run_epoch == session.run_epoch

        retry_claim = await app._recovery_coordinator._claim_incomplete_recovery(
            session=current,
            inactive_for_seconds=None,
        )
        assert retry_claim is not None
        await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
            authority=retry_claim.require_authority(),
            authoritative_failure=None,
        )

    asyncio.run(scenario())


def test_initial_incomplete_recovery_rejects_build_provenance_change_before_fence() -> None:
    replacement_provenance = RuntimeBuildProvenance.from_artifact_digest(
        origin=RuntimeBuildProvenanceOrigin.EXPLICIT_MANIFEST,
        artifact_kind=RuntimeBuildArtifactKind.OTHER,
        artifact_digest="a" * 64,
    )

    class BuildProvenanceRaceStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.replace_build_provenance = True

        async def fence_run_and_transform_checkpoint(self, session_id, **kwargs):
            if self.replace_build_provenance:
                self.replace_build_provenance = False
                session = self._sessions[session_id]
                metadata = dict(session.metadata)
                metadata[RUNTIME_BUILD_PROVENANCE_METADATA_KEY] = replacement_provenance.model_dump(
                    mode="json"
                )
                self._sessions[session_id] = session.model_copy(update={"metadata": metadata})
            return await super().fence_run_and_transform_checkpoint(session_id, **kwargs)

    async def scenario() -> None:
        store = BuildProvenanceRaceStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_recovery_build_provenance_race",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)

        claim = await app._recovery_coordinator._claim_incomplete_recovery(
            session=session,
            inactive_for_seconds=None,
        )
        assert claim is None

        durable = await store.load(session.id)
        assert durable is not None
        assert durable.runtime_build_provenance == replacement_provenance
        assert durable.run_epoch == session.run_epoch
        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is None or "incomplete_session_recovery_claim" not in checkpoint

    asyncio.run(scenario())


def test_expired_incomplete_recovery_renewal_does_not_refresh_activity() -> None:
    async def scenario() -> None:
        current_time = {"value": datetime(2026, 7, 20, tzinfo=UTC)}

        def clock() -> datetime:
            return current_time["value"]

        store = InMemorySessionStore(ownership_clock=clock)
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_expired_recovery_renewal_no_activity",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        claim = await app._recovery_coordinator._claim_incomplete_recovery(
            session=session,
            inactive_for_seconds=None,
        )
        assert claim is not None

        before_session = await store.load(session.id)
        before_checkpoint = await store.load_checkpoint(session.id)
        assert before_session is not None
        assert before_checkpoint is not None

        current_time["value"] = claim.claim_expires_at
        renewed_until = await app._recovery_coordinator._renew_incomplete_recovery_claim(
            session.id,
            claim.claim_id,
        )
        assert renewed_until is None
        assert await store.load(session.id) == before_session
        assert await store.load_checkpoint(session.id) == before_checkpoint

        await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
            authority=claim.require_authority(),
            authoritative_failure=None,
        )

    asyncio.run(scenario())


def test_stalled_incomplete_recovery_renewal_stops_work_at_local_lease_deadline() -> None:
    class BlockingRenewalStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.block_renewal = False
            self.renewal_dispatched = asyncio.Event()
            self.release_renewal = asyncio.Event()
            self.renewal_finished = asyncio.Event()

        async def transform_checkpoint_with_store_time(
            self,
            session_id: str,
            checkpoint_transform,
        ) -> None:
            if self.block_renewal:
                self.renewal_dispatched.set()
                await self.release_renewal.wait()
            try:
                await super().transform_checkpoint_with_store_time(
                    session_id,
                    checkpoint_transform,
                )
            finally:
                if self.block_renewal:
                    self.renewal_finished.set()

    async def scenario() -> None:
        store = BlockingRenewalStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_stalled_recovery_renewal",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        claim = await app._recovery_coordinator._claim_incomplete_recovery(
            session=session,
            inactive_for_seconds=None,
        )
        assert claim is not None
        short_claim = replace(claim, local_lease_deadline=time.monotonic() + 0.1)
        store.block_renewal = True
        recovery_started = asyncio.Event()
        recovery_stopped = asyncio.Event()
        release_recovery = threading.Event()

        async def recovery():
            recovery_started.set()
            try:
                await asyncio.to_thread(release_recovery.wait)
            finally:
                recovery_stopped.set()

        owner = asyncio.create_task(
            app._recovery_coordinator._recover_incomplete_session_with_heartbeat(
                claim=short_claim,
                recovery=recovery,
            )
        )
        await asyncio.wait_for(recovery_started.wait(), timeout=1)
        await asyncio.wait_for(store.renewal_dispatched.wait(), timeout=1)
        await asyncio.sleep(0.15)
        assert owner.done() is False
        assert recovery_stopped.is_set() is False
        release_recovery.set()
        with pytest.raises(_IncompleteRecoveryClaimLost):
            await asyncio.wait_for(owner, timeout=1)
        assert recovery_stopped.is_set()
        assert not store.renewal_finished.is_set()

        store.release_renewal.set()
        await asyncio.wait_for(store.renewal_finished.wait(), timeout=1)
        await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
            authority=claim.require_authority(),
            authoritative_failure=None,
        )

    asyncio.run(scenario())


def test_incomplete_recovery_fatal_settlement_wins_over_lease_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[GeneratorExit, _IncompleteRecoveryClaimLost]:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_recovery_fatal_after_lease_loss",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        coordinator = app._recovery_coordinator
        claim = await coordinator._claim_incomplete_recovery(
            session=session,
            inactive_for_seconds=None,
        )
        assert claim is not None
        recovery_started = asyncio.Event()
        release_recovery = asyncio.Event()
        fatal = GeneratorExit("recovery settlement aborted")
        lease_failure = _IncompleteRecoveryClaimLost("recovery lease lost")

        async def lose_lease(**_kwargs) -> None:
            raise lease_failure

        async def recovery():
            recovery_started.set()
            await release_recovery.wait()
            raise fatal

        monkeypatch.setattr(
            coordinator,
            "_heartbeat_incomplete_recovery_claim",
            lose_lease,
        )
        owner = asyncio.create_task(
            coordinator._recover_incomplete_session_with_heartbeat(
                claim=claim,
                recovery=recovery,
            )
        )
        await recovery_started.wait()
        await asyncio.sleep(0)
        assert owner.done() is False
        release_recovery.set()
        with pytest.raises(GeneratorExit) as raised:
            await owner
        assert raised.value is fatal
        assert recovery_coordinator._exception_graph_contains_identity(fatal, lease_failure)
        return fatal, lease_failure

    fatal, lease_failure = asyncio.run(scenario())
    assert fatal is not lease_failure


def test_incomplete_recovery_renewal_preserves_owner_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingRenewalStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.block_renewal = False
            self.renewal_dispatched = asyncio.Event()
            self.release_renewal = asyncio.Event()
            self.renewal_finished = asyncio.Event()

        async def transform_checkpoint_with_store_time(
            self,
            session_id: str,
            checkpoint_transform,
        ) -> None:
            if self.block_renewal:
                self.renewal_dispatched.set()
                await self.release_renewal.wait()
            try:
                await super().transform_checkpoint_with_store_time(
                    session_id,
                    checkpoint_transform,
                )
            finally:
                if self.block_renewal:
                    self.renewal_finished.set()

    async def scenario() -> None:
        monkeypatch.setattr(
            recovery_coordinator,
            "_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = BlockingRenewalStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_cancelled_recovery_renewal",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        claim = await app._recovery_coordinator._claim_incomplete_recovery(
            session=session,
            inactive_for_seconds=None,
        )
        assert claim is not None
        store.block_renewal = True
        heartbeat = asyncio.create_task(
            app._recovery_coordinator._heartbeat_incomplete_recovery_claim(
                session_id=session.id,
                claim_id=claim.claim_id,
                local_lease_deadline=time.monotonic() + 10,
                stop=asyncio.Event(),
            )
        )
        await asyncio.wait_for(store.renewal_dispatched.wait(), timeout=1)

        assert heartbeat.cancel("stop recovery") is True
        await asyncio.sleep(0)
        assert not heartbeat.done()
        assert heartbeat.cancelling() == 0

        store.release_renewal.set()
        with pytest.raises(asyncio.CancelledError, match="stop recovery"):
            await heartbeat
        assert heartbeat.cancelled() is True
        assert heartbeat.cancelling() == 1
        assert store.renewal_finished.is_set()

        await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
            authority=claim.require_authority(),
            authoritative_failure=None,
        )

    asyncio.run(scenario())


def test_incomplete_recovery_owner_cancellation_drains_opaque_work() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_cancelled_opaque_recovery",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        coordinator = app._recovery_coordinator
        claim = await coordinator._claim_incomplete_recovery(
            session=session,
            inactive_for_seconds=None,
        )
        assert claim is not None
        work_started = threading.Event()
        release_work = threading.Event()

        def opaque_work() -> str:
            work_started.set()
            release_work.wait()
            return "settled"

        async def recovery() -> str:
            return await asyncio.to_thread(opaque_work)

        owner = asyncio.create_task(
            coordinator._recover_incomplete_session_with_heartbeat(
                claim=claim,
                recovery=recovery,
            )
        )
        assert await asyncio.to_thread(work_started.wait, 1)
        assert owner.cancel("stop recovery") is True
        await asyncio.sleep(0)
        assert owner.done() is False
        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is not None
        retained_claim = recovery_coordinator._incomplete_recovery_claim_from_checkpoint(checkpoint)
        assert retained_claim is not None
        assert retained_claim[0] == claim.claim_id

        release_work.set()
        with pytest.raises(asyncio.CancelledError, match="stop recovery"):
            await asyncio.wait_for(owner, timeout=1)
        assert owner.cancelled() is True
        assert owner.cancelling() == 1
        await coordinator._cleanup_incomplete_recovery_claim(
            authority=claim.require_authority(),
            authoritative_failure=None,
            recovery_work_quiescent=True,
        )

    asyncio.run(scenario())


def test_preclaimed_terminal_operation_cancellation_drains_opaque_work() -> None:
    async def scenario() -> None:
        app = CayuApp(enable_logging=False)
        work_started = threading.Event()
        release_work = threading.Event()

        async def heartbeat_forever() -> None:
            await asyncio.Event().wait()

        heartbeat = asyncio.create_task(heartbeat_forever())

        def opaque_work() -> str:
            work_started.set()
            release_work.wait()
            return "settled"

        owner = asyncio.create_task(
            app._recovery_coordinator._await_preclaimed_terminal_evidence_operation(
                heartbeat_task=heartbeat,
                operation=lambda: asyncio.to_thread(opaque_work),
                operation_name="terminal opaque work",
            )
        )
        assert await asyncio.to_thread(work_started.wait, 1)
        assert owner.cancel("stop finalization") is True
        await asyncio.sleep(0)
        assert owner.done() is False

        release_work.set()
        with pytest.raises(asyncio.CancelledError, match="stop finalization"):
            await asyncio.wait_for(owner, timeout=1)
        assert owner.cancelled() is True
        assert owner.cancelling() == 1
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)

    asyncio.run(scenario())
