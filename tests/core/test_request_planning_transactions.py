"""Native planning transactions; public authorization is covered separately."""

import asyncio

import pytest
from tests.core.test_participant_identity import stores
from tests.core.test_request_planning_records import _records

from cayu.collaboration._contracts import CollaborationConflict, ExactConflict, ExactMatch
from cayu.collaboration._planning_store import (
    read_plan_in_transaction,
    retain_decision_in_transaction,
    retain_plan_in_transaction,
)
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.planning import (
    RequestPlanningDecline,
    evaluate_configured_request_policy,
    planning_policy_commitment,
)
from cayu.vaults.redaction import SecretRedactor

__all__ = ["stores"]
pytestmark = pytest.mark.anyio
REDACTOR = SecretRedactor()


async def _intent(store):
    initial, record = await _records(store)
    policy = record.receipt.policy.model_copy(update={"rules": ()})
    command = record.receipt.command.model_copy(
        update={"policy_sha256": planning_policy_commitment(policy, redactor=REDACTOR)}
    )
    return initial, command, policy


async def _retain(store, initial, command, policy):
    async with store._transaction(initial.binding.application_scope, write=True) as tx:
        return await retain_plan_in_transaction(
            store, tx, initial, command, policy, redactor=REDACTOR
        )


async def test_retained_plan_and_decision_replay_preserve_policy_and_capacity(stores):
    store = stores()
    initial, command, policy = await _intent(store)
    retained = await _retain(store, initial, command, policy)
    assert retained.state == "evaluating"
    replacement = policy.model_copy(
        update={"default": RequestPlanningDecline(reason="replacement")}
    )
    assert await _retain(stores(), initial, command, replacement) == retained
    proposal = evaluate_configured_request_policy(
        retained.receipt.policy, input_revision=0, redactor=REDACTOR
    )
    async with store._transaction(initial.binding.application_scope, write=True) as tx:
        before = await store._anchor(tx, initial, REDACTOR)
        decided = await retain_decision_in_transaction(
            store, tx, initial, command, proposal, redactor=REDACTOR
        )
        after = await store._anchor(tx, initial, REDACTOR)
    assert decided.state == "decided" and decided.decision == policy.default
    assert after.event_count == before.event_count + 1
    assert (
        after.retained_bytes + after.reserved_bytes <= before.retained_bytes + before.reserved_bytes
    )
    assert after.event_count + after.reserved_events == before.event_count + before.reserved_events
    reopened = stores()
    async with reopened._transaction(initial.binding.application_scope, write=True) as tx:
        assert (
            await retain_decision_in_transaction(
                reopened, tx, initial, command, proposal, redactor=REDACTOR
            )
            == decided
        )
        assert await reopened._anchor(tx, initial, REDACTOR) == after
        result = await read_plan_in_transaction(reopened, tx, initial, command, redactor=REDACTOR)
        assert isinstance(result, ExactMatch) and result.receipt == decided
        changed = command.model_copy(update={"deadline_at_ms": command.deadline_at_ms - 1})
        assert isinstance(
            await read_plan_in_transaction(reopened, tx, initial, changed, redactor=REDACTOR),
            ExactConflict,
        )
    with pytest.raises(CollaborationConflict):
        async with reopened._transaction(initial.binding.application_scope, write=True) as tx:
            await retain_decision_in_transaction(
                reopened, tx, initial, command, replacement.default, redactor=REDACTOR
            )
    async with reopened._transaction(initial.binding.application_scope, write=False) as tx:
        assert await reopened._anchor(tx, initial, REDACTOR) == after


async def test_competing_retention_and_late_failure_are_atomic(stores):
    first, second = stores(), stores()
    initial, command, policy = await _intent(first)
    async with first._transaction(initial.binding.application_scope, write=False) as tx:
        original = await first._anchor(tx, initial, REDACTOR)
    with pytest.raises(RuntimeError, match="before commit"):
        async with first._transaction(initial.binding.application_scope, write=True) as tx:
            await retain_plan_in_transaction(first, tx, initial, command, policy, redactor=REDACTOR)
            raise RuntimeError("before commit")
    async with first._transaction(initial.binding.application_scope, write=False) as tx:
        assert await first._anchor(tx, initial, REDACTOR) == original
        assert await tx.scan_pending_request_plans(after=None, limit=1) == []
    results = await asyncio.gather(
        _retain(first, initial, command, policy), _retain(second, initial, command, policy)
    )
    assert results[0] == results[1]
    async with second._transaction(initial.binding.application_scope, write=False) as tx:
        current = await second._anchor(tx, initial, REDACTOR)
        assert current.operation_count == original.operation_count + 1
        assert current.event_count == original.event_count + 1
        assert len(await tx.scan_pending_request_plans(after=None, limit=32)) == 1


async def test_stale_input_or_wrong_policy_rejects_without_retaining_intent(stores):
    store = stores()
    initial, command, policy = await _intent(store)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        original = await store._anchor(tx, initial, REDACTOR)
    for changed in (
        command.model_copy(update={"expected_input_revision": 1}),
        command.model_copy(update={"expected_input_sha256": "f" * 64}),
        command.model_copy(update={"policy_sha256": "f" * 64}),
        command.model_copy(update={"expected_revision": command.expected_revision + 1}),
    ):
        with pytest.raises(CollaborationConflict):
            await _retain(store, initial, changed, policy)
        async with store._transaction(initial.binding.application_scope, write=False) as tx:
            assert await store._anchor(tx, initial, REDACTOR) == original
            assert await tx.scan_pending_request_plans(after=None, limit=1) == []


async def test_readback_rejects_state_or_decision_diverging_from_events(stores):
    store = stores()
    initial, command, policy = await _intent(store)
    await _retain(store, initial, command, policy)
    async with store._transaction(initial.binding.application_scope, write=True) as tx:
        record = await retain_decision_in_transaction(
            store, tx, initial, command, policy.default, redactor=REDACTOR
        )
    operation = command.operation
    key = operation.namespace_incarnation, operation.generation, operation.caller_key
    from cayu.collaboration._contracts import CollaborationContractError

    for changes, failure in (
        ({"state": "cancelled"}, CollaborationUnavailable),
        ({"decision_commitment": "f" * 64}, CollaborationContractError),
    ):
        with pytest.raises(failure):
            async with store._transaction(initial.binding.application_scope, write=True) as tx:
                await tx.put("request_plans", key, record.model_copy(update=changes), insert=False)
                await read_plan_in_transaction(store, tx, initial, command, redactor=REDACTOR)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        found = await read_plan_in_transaction(store, tx, initial, command, redactor=REDACTOR)
        assert isinstance(found, ExactMatch) and found.receipt == record
