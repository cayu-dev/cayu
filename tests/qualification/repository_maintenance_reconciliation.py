"""Operator-only settlement of a stopped, cancellation-fenced coding worker.

This is not task retry, product acceptance, or resource cleanup. Runtime recovery
must already have released the exact invocation. Unsupported effects stay fenced.
"""

import hashlib
import json
from datetime import UTC, datetime

from operations.maintenance_requests import (  # ty: ignore[unresolved-import]
    capture_accepted_request,
)

from cayu import (
    CodingProductArtifactRepository,
    CodingProductRunner,
    CodingSettlementPolicy,
    Message,
    ResolutionActor,
    ResolutionActorSource,
    TaskCancellationReconciliation,
    TaskCancellationReconciliationEvidence,
    TaskCancellationReconciliationOutcome,
    TaskCancellationReconciliationRequest,
    TaskStatus,
)
from cayu.guides.coding_host_evidence import (
    MaintenanceReconciliationUnavailable,
    _require_serial_check_quiescence,
)
from cayu.guides.coding_host_owner import inspect_stopped_worker
from cayu.sessions.transcript_input import session_input_messages_sha256
from tests.qualification.repository_maintenance_identity import copy_identity
from tests.qualification.repository_maintenance_intake import load_owned_coding_task
from tests.qualification.repository_maintenance_request import decode_request
from tests.qualification.repository_maintenance_results import coding_task_from_identity


async def _publication_forbidden(_authority):
    raise MaintenanceReconciliationUnavailable()


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode()
    ).hexdigest()


async def _owned(application, reservations, expected):
    expected = copy_identity(expected)
    current = await reservations.load_owned(
        tenant=expected.intent.tenant, public_id=expected.public_id
    )
    if current is None or copy_identity(current) != expected:
        raise MaintenanceReconciliationUnavailable()
    task = await load_owned_coding_task(application.app, expected)
    if task is None:
        raise MaintenanceReconciliationUnavailable()
    return expected, task


async def inspect_cancellation(application, reservations, expected, *, actor_subject):
    identity, task = await _owned(application, reservations, expected)
    if (
        task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        or task.status_reason != "cancellation_requested"
        or task.started_at is None
        or task.lease_expires_at is None
        or task.lease_expires_at > datetime.now(UTC)
        or task.interrupted_handoff_id is not None
    ):
        raise MaintenanceReconciliationUnavailable()
    owner = await inspect_stopped_worker(task.worker_id)
    accepted = decode_request(identity.intent.request_json)
    coding_task = coding_task_from_identity(identity)
    if await capture_accepted_request(application, coding_task) != accepted:
        raise MaintenanceReconciliationUnavailable()
    repository = CodingProductArtifactRepository(application.artifact_store)
    request = await repository.load_request(identity.product_run_id, session_id=identity.session_id)
    if (
        request.product_run_id != identity.product_run_id
        or request.session_id != identity.session_id
        or request.agent_name != application.agent_name
        or request.task.task_id != identity.task_id
        or request.task.instruction_sha256.removeprefix("sha256:")
        != session_input_messages_sha256([Message.text("user", accepted.instruction)])
        or request.source.workspace_id != accepted.source_workspace_id
        or request.source.origin_id != accepted.source_origin_id
        or request.source.destination_id != accepted.source_destination_id
        or request.source.git_baseline.head_revision != accepted.base_revision
        or request.settlement
        != CodingSettlementPolicy.model_validate_json(accepted.settlement_json)
        or request.parent_session_id != identity.workflow_session_id
        or request.causal_budget_id != identity.workflow_session_id
        or request.runtime.execution_profile_fingerprint.removeprefix("sha256:")
        != accepted.execution_profile_fingerprint
        or request.runtime.toolchain_profile_fingerprint != accepted.toolchain_profile_fingerprint
    ):
        raise MaintenanceReconciliationUnavailable()
    runner = CodingProductRunner(
        application.app,
        source_workspace=application.source_workspace,
        repository=repository,
        source_git_authority_validator=_publication_forbidden,
    )
    inspection = await runner.inspect_settled_execution(request)
    if inspection.request_fingerprint != request.fingerprint:
        raise MaintenanceReconciliationUnavailable()
    _require_serial_check_quiescence(
        inspection.events, tool_call_ordinals=inspection.tool_call_ordinals
    )
    fingerprint = _digest(
        {
            "schema": "maintenance.cancellation.v1",
            "actor": actor_subject,
            "identity": identity.model_dump(mode="json"),
            "task": task.model_dump(mode="json"),
            "owner": owner,
            "request": request.fingerprint,
            "release": inspection.release_fingerprint,
            "tool_call_ordinals": inspection.tool_call_ordinals,
            "events": [event.model_dump(mode="json") for event in inspection.events],
        }
    )
    return {
        "plan_fingerprint": "sha256:" + fingerprint,
        "task_id": task.id,
        "proposed_outcome": "cancelled",
    }


async def reconcile_cancellation(
    application, reservations, expected, *, actor_subject, plan_fingerprint, reconciliation_id
):
    identity, task = await _owned(application, reservations, expected)
    actor = ResolutionActor(
        subject=actor_subject, tenant=identity.intent.tenant, source=ResolutionActorSource.HTTP_AUTH
    )
    if task.status is TaskStatus.CANCELLED:
        if type(task.status_payload) is not dict:
            raise MaintenanceReconciliationUnavailable()
        retained = TaskCancellationReconciliation.model_validate(
            task.status_payload.get("cancellation_reconciliation")
        )
        if (
            retained.reconciliation_idempotency_key != reconciliation_id
            or retained.reconciled_by != actor
            or "sha256:" + retained.evidence.evidence_sha256 != plan_fingerprint
            or retained.evidence.validator_id != "maintenance-stopped-coding-worker"
            or retained.evidence.validator_version != "1"
            or retained.evidence.outcome is not TaskCancellationReconciliationOutcome.QUIESCENT
            or retained.evidence.evidence_id != identity.public_id
        ):
            raise MaintenanceReconciliationUnavailable()
        fields = retained.model_dump(mode="python", exclude={"request_sha256", "events"})
        request = TaskCancellationReconciliationRequest(**fields)
    else:
        plan = await inspect_cancellation(
            application, reservations, identity, actor_subject=actor_subject
        )
        if plan["plan_fingerprint"] != plan_fingerprint:
            raise MaintenanceReconciliationUnavailable()
        now = datetime.now(UTC)
        payload = task.status_payload
        request = TaskCancellationReconciliationRequest(
            task_id=task.id,
            original_worker_id=task.worker_id,
            original_lease_expires_at=task.lease_expires_at,
            cancellation_requested_at=payload["event"]["occurred_at"],
            cancellation_idempotency_key=payload["terminalization_idempotency_key"],
            reconciliation_idempotency_key=reconciliation_id,
            reconciliation_requested_at=now,
            reconciled_by=actor,
            evidence=TaskCancellationReconciliationEvidence(
                outcome=TaskCancellationReconciliationOutcome.QUIESCENT,
                validator_id="maintenance-stopped-coding-worker",
                validator_version="1",
                evidence_id=identity.public_id,
                evidence_sha256=plan_fingerprint.removeprefix("sha256:"),
                validated_at=now,
            ),
        )
    result = await application.app.task_store.reconcile_task_cancellation(request)
    return {
        "id": identity.public_id,
        "coding_task_status": result.task.status.value,
        "receipt_id": result.reconciliation.events[-1].id,
    }
