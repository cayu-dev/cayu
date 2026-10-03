"""Reconstruct temporary service budget constraints from native admission.

This is a constraint on the existing registered budget receiver, not another
budget authority. Neither a question nor a caller-supplied service record grants
execution or spending permission.
"""

from __future__ import annotations

from cayu.budgets.binding import BudgetBinding, BudgetBindingError
from cayu.collaboration._contracts import ObjectRef
from cayu.collaboration._preparation import prepare_contract
from cayu.runtime._checkpoint_store import (
    load_runtime_session_checkpoint_snapshot,
    runtime_checkpoint_session_store,
)
from cayu.sessions._invocation_lifecycle import (
    _invocation_lifecycle_receipt_from_checkpoint,
)
from cayu.sessions._session_continuation import (
    CONTINUATION_SERVICE_PREFIX,
    ContinuationConflict,
    continuation_digest,
)
from cayu.sessions._session_continuation_store import ROOT_KEY, ContinuationRoot
from cayu.sessions._temporary_continuation import (
    TemporaryServiceAdmission,
    TemporaryServiceIntent,
    TemporaryServiceRecord,
)
from cayu.sessions._temporary_continuation_store import native_service_execution
from cayu.sessions._temporary_service_target import TARGET_PREFIX, TemporaryServiceTarget
from cayu.sessions.base import SessionStore
from cayu.vaults.redaction import SecretRedactor


async def admitted_service_budget_constraint(
    store: SessionStore, *, session_id: str
) -> TemporaryServiceIntent | None:
    """Read the active native run and its exact receiving-owned service record.

    The snapshot is atomic; a subsequent child transition conflicts rather than
    falling back to an unrelated budget. The runtime's usual invocation/stage
    guards still fence a stale caller before dispatch.
    """
    if not store._supports_session_continuation_protocol():
        # Such stores cannot admit temporary service in the first place.
        return None
    session, checkpoint = await load_runtime_session_checkpoint_snapshot(
        runtime_checkpoint_session_store(store), session_id
    )
    raw_root = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    root = (
        None
        if raw_root is None
        else prepare_contract(ContinuationRoot, raw_root, redactor=SecretRedactor())
    )
    if root is not None and (
        root.namespace.session_id != session.id
        or root.namespace.session_instance_id != session.instance_id
    ):
        raise ContinuationConflict("Service budget index belongs to another incarnation.")
    indexed = root is not None and (
        any(session.run_epoch in entry.service_receipt_epochs for entry in root.entries)
        or any(
            not ref.source_acknowledged and ref.expected_run_epoch + 1 == session.run_epoch
            for ref in root.target_services
        )
    )
    receipt = _invocation_lifecycle_receipt_from_checkpoint(
        checkpoint,
        command_identity=f"admit:{session.id}:{session.instance_id}:{session.run_epoch}",
    )
    key = None if receipt is None else receipt.temporary_service_operation_key
    if key is None:
        if indexed:
            raise ContinuationConflict("Service budget authority has lost its native admission.")
        return None
    if not indexed or root is None:
        raise ContinuationConflict("Service budget authority has lost its retention index.")
    target_key = TARGET_PREFIX + key.removeprefix(CONTINUATION_SERVICE_PREFIX)
    target_ref = next((ref for ref in root.target_services if ref.key == target_key), None)
    if target_ref is not None:
        raw = await store.load_session_operation(session_id, target_key)
        target = prepare_contract(TemporaryServiceTarget, raw, redactor=SecretRedactor())
        if continuation_digest(target) != target_ref.record_sha256:
            raise ContinuationConflict("Service budget target changed during reconstruction.")
        record = target.service
    else:
        raw = await store.load_session_operation(session_id, key)
        record = prepare_contract(TemporaryServiceRecord, raw, redactor=SecretRedactor())
        if await store._load_temporary_continuation_service(record.admission) != record:
            raise ContinuationConflict("Service budget source changed during reconstruction.")
    if (
        record.state != "admitted"
        or not isinstance(record.admission, TemporaryServiceAdmission)
        or native_service_execution(record.admission, checkpoint) != record.execution
        or record.execution is None
        or record.execution.session_id != session.id
        or record.execution.session_instance_id != session.instance_id
        or record.execution.run_epoch != session.run_epoch
    ):
        raise ContinuationConflict("Service budget lacks exact active receiving authority.")
    required_append = record.intent.required_peer_append
    if required_append is not None:
        # Callback completion is not evidence. Exact native reconciliation
        # verifies the complete request (including deadline, flags and target
        # boundary) after restart as well as in the original invocation.
        delivery = await store.read_peer_content_attempt(required_append)
        if delivery is None or delivery.status != "appended":
            raise ContinuationConflict(
                "Service requires its exact durable peer append before dispatch."
            )
    return record.intent


def require_service_budget_binding(
    intent: TemporaryServiceIntent | None, binding: BudgetBinding | None
) -> None:
    """Constrain registered authority before ledger registration/reservation."""
    if intent is None:
        return
    if binding is None or (
        binding.application_scope != intent.operation.application_scope
        or binding.authority_digest != intent.budget_authority_sha256
        or intent.budget_binding
        != ObjectRef(
            owner=intent.ticket.owner,
            kind="budget_binding",
            object_id=binding.binding_id,
            incarnation=binding.authority_digest,
            revision=1,
        )
    ):
        raise BudgetBindingError("Temporary service common-root budget authority conflicts.")
