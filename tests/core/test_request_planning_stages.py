"""Local receiver fences; public source-authorized opening is tested separately."""

import pytest
from tests.core.test_participant_identity import stores
from tests.core.test_request_planning_transactions import _intent, _retain

from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import CollaborationConflict, CollaborationContractError
from cayu.collaboration._planning_records import (
    RequestPlanningStageIntent,
    RequestPlanningStageRecord,
)
from cayu.collaboration._planning_stages import (
    _PlannedStage,
    finish_stage,
    read_stage,
    retain_stage,
)
from cayu.collaboration._planning_store import admission_command, retain_decision_in_transaction
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration._request_arbitration import admit_in_transaction
from cayu.vaults.redaction import SecretRedactor

__all__ = ["stores"]
pytestmark = pytest.mark.anyio
REDACTOR = SecretRedactor()


async def _stage(store):
    initial, command, policy = await _intent(store)
    await _retain(store, initial, command, policy)
    async with store._transaction(initial.binding.application_scope, write=True) as tx:
        decided = await retain_decision_in_transaction(
            store, tx, initial, command, policy.default, redactor=REDACTOR
        )
        intent = RequestPlanningStageIntent(
            operation=initial.operation("stage-1"),
            plan=command.operation,
            plan_sha256=clarification_commitment(command, REDACTOR),
            ordinal=1,
            command=admission_command(decided, REDACTOR),
        )
        stage = await retain_stage(store, tx, initial, command, intent, redactor=REDACTOR)
    return initial, command, intent, stage


async def test_native_stage_scan_is_not_capped_at_cleanup_batch_size(stores):
    store = stores()
    initial, command, _, stage = await _stage(store)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        # The full validated frontier is read independently of the <=32 deletion
        # batch. Exercise both sides of that boundary and the actual scan ceiling.
        for limit in (1, 31, 32, 33, 128, 129):
            rows = await tx.scan_request_plan_stages(command.operation, limit=limit)
            assert len(rows) == 1
            assert prepare_contract(RequestPlanningStageRecord, rows[0], redactor=REDACTOR) == stage
        for limit in (0, True, 130):
            with pytest.raises(CollaborationContractError):
                await tx.scan_request_plan_stages(command.operation, limit=limit)
        rows = await tx.scan_request_plan_stages(command.operation, limit=129)
        assert prepare_contract(RequestPlanningStageRecord, rows[0], redactor=REDACTOR) == stage


@pytest.mark.parametrize("exclude_first", [False, True])
async def test_local_receiving_stage_fences_raw_calls_and_reconstructs_disposition(
    stores, exclude_first
):
    store = stores()
    initial, command, intent, registered = await _stage(store)
    scope = initial.binding.application_scope
    other = stores()
    async with other._transaction(scope, write=False) as tx:
        before = await other._anchor(tx, initial, REDACTOR)
    with pytest.raises(CollaborationConflict, match="retained planning stage"):
        async with other._transaction(scope, write=True) as tx:
            await admit_in_transaction(other, tx, initial, intent.command, redactor=REDACTOR)
    async with other._transaction(scope, write=False) as tx:
        assert await other._anchor(tx, initial, REDACTOR) == before
        assert await read_stage(tx, intent, redactor=REDACTOR) == registered
    planned = _PlannedStage(command, intent)
    if exclude_first:
        async with other._transaction(scope, write=True) as tx:
            final = await finish_stage(other, tx, initial, command, intent, None, redactor=REDACTOR)
        with pytest.raises(CollaborationConflict, match="no longer admits"):
            async with store._transaction(scope, write=True) as tx:
                await admit_in_transaction(
                    store, tx, initial, intent.command, redactor=REDACTOR, _planned_stage=planned
                )
        receipt = None
    else:
        async with other._transaction(scope, write=True) as tx:
            receipt = await admit_in_transaction(
                other, tx, initial, intent.command, redactor=REDACTOR, _planned_stage=planned
            )
            final = await read_stage(tx, intent, redactor=REDACTOR)
        with pytest.raises(CollaborationConflict, match="different disposition"):
            async with store._transaction(scope, write=True) as tx:
                await finish_stage(store, tx, initial, command, intent, None, redactor=REDACTOR)
    reopened = stores()
    async with reopened._transaction(scope, write=True) as tx:
        after = await reopened._anchor(tx, initial, REDACTOR)
        assert (
            await finish_stage(reopened, tx, initial, command, intent, receipt, redactor=REDACTOR)
            == final
        )
        assert await reopened._anchor(tx, initial, REDACTOR) == after
        if exclude_first:
            assert after.reserved_bytes == before.reserved_bytes - registered.reserved_bytes
        else:
            # Decline also settles the original request/permit reservations.
            assert after.reserved_bytes < before.reserved_bytes - registered.reserved_bytes
        assert after.reserved_events <= before.reserved_events - 1
        if receipt is not None:
            assert (
                await admit_in_transaction(
                    reopened, tx, initial, intent.command, redactor=REDACTOR, _planned_stage=planned
                )
                == receipt
            )


async def test_receiving_commit_and_stage_settlement_roll_back_together(stores):
    store = stores()
    initial, command, intent, stage = await _stage(store)
    scope = initial.binding.application_scope
    async with store._transaction(scope, write=False) as tx:
        before = await store._anchor(tx, initial, REDACTOR)
    with pytest.raises(RuntimeError, match="after native mutation"):
        async with store._transaction(scope, write=True) as tx:
            await admit_in_transaction(
                store,
                tx,
                initial,
                intent.command,
                redactor=REDACTOR,
                _planned_stage=_PlannedStage(command, intent),
            )
            raise RuntimeError("after native mutation")
    async with stores()._transaction(scope, write=False) as tx:
        assert await store._anchor(tx, initial, REDACTOR) == before
        assert await read_stage(tx, intent, redactor=REDACTOR) == stage
