from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from cayu.core import Message
from cayu.core.thinking import ThinkingConfig
from cayu.runtime import CayuApp, InMemorySessionStore, RunRequest, SessionIdentity
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._recovery_coordinator import (
    _DEFAULT_APPROVAL_MAX_STEPS,
    _effective_approval_budget_limits,
    _effective_approval_max_steps,
    _effective_approval_retry_policy,
    _effective_approval_run_limits,
    _effective_approval_thinking,
    _interrupted_tool_round_results,
    _recovery_abandonment_signal,
    _run_recovery_cleanup_steps,
    _task_cancellation_count,
)
from cayu.runtime.approvals import PendingToolApproval, PendingToolCallApproval
from cayu.runtime.budgets import BudgetLimit
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.sessions import CheckpointTransform, Session, SessionStatus
from cayu.runtime.stop_policy import RunLimits


def _pending_approval(**kwargs) -> PendingToolApproval:
    return PendingToolApproval(
        approval_id="appr_1",
        tool_call_id="call_1",
        tool_name="side_effect",
        agent_name="assistant",
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


def test_effective_approval_run_config_prefers_override_then_pending_then_default() -> None:
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

    assert (
        _effective_approval_max_steps(max_steps=None, pending_approval=legacy)
        == _DEFAULT_APPROVAL_MAX_STEPS
    )
    assert _effective_approval_run_limits(limits=None, pending_approval=legacy) == RunLimits()
    assert _effective_approval_budget_limits(budget_limits=None, pending_approval=legacy) == ()
    assert (
        _effective_approval_retry_policy(
            retry_policy=None,
            pending_approval=legacy,
        )
        is None
    )


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
        cancellation_artifacts_by_id={"B": [{"producer": "B"}]},
    )
    by_id = {outcome.call.id: outcome for outcome in keyed}
    assert by_id["B"].result.artifacts == [{"producer": "B"}]
    assert by_id["A"].result.artifacts == []

    fallback = _interrupted_tool_round_results(
        tool_calls=[a, b],
        completed_outcomes=[],
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
        def __init__(self) -> None:
            super().__init__()
            self.first_claim_ready = asyncio.Event()
            self.release_first_claim = asyncio.Event()
            self.first_claim_paused = False

        async def fence_run_and_transform_checkpoint(
            self,
            session_id: str,
            *,
            statuses: set[SessionStatus],
            checkpoint_transform: CheckpointTransform,
        ) -> Session:
            fenced = await super().fence_run_and_transform_checkpoint(
                session_id,
                statuses=statuses,
                checkpoint_transform=checkpoint_transform,
            )
            if not self.first_claim_paused:
                self.first_claim_paused = True
                self.first_claim_ready.set()
                await self.release_first_claim.wait()
            return fenced

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

        store = PausingClaimStore()
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
                inactive_before=None,
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
            inactive_before=None,
            required_expired_claim_id=first_marker["claim_id"],
        )
        assert replacement_claim is not None

        try:
            store.release_first_claim.set()
            assert await asyncio.wait_for(first_task, timeout=5) is None

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
                session_id=session.id,
                claim_id=replacement_claim.claim_id,
                authoritative_failure=None,
            )

    asyncio.run(scenario())


def test_initial_incomplete_recovery_reconciles_ambiguous_atomic_claim() -> None:
    class CommitThenRaiseClaimStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.lose_claim_acknowledgement = True

        async def fence_run_and_transform_checkpoint(
            self,
            session_id: str,
            *,
            statuses: set[SessionStatus],
            checkpoint_transform: CheckpointTransform,
        ) -> Session:
            fenced = await super().fence_run_and_transform_checkpoint(
                session_id,
                statuses=statuses,
                checkpoint_transform=checkpoint_transform,
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
                inactive_before=None,
            )

        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is None or "incomplete_session_recovery_claim" not in checkpoint

        current = await store.load(session.id)
        assert current is not None
        retry_claim = await app._recovery_coordinator._claim_incomplete_recovery(
            session=current,
            inactive_before=None,
        )
        assert retry_claim is not None
        await app._recovery_coordinator._cleanup_incomplete_recovery_claim(
            session_id=session.id,
            claim_id=retry_claim.claim_id,
            authoritative_failure=None,
        )

    asyncio.run(scenario())
