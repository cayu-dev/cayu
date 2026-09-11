"""Historical delivery evidence; never current remote observations or retry authority."""

from cayu import (
    GitHubDeliveryReconstructionRequiredError,
    GitHubDeliveryRepository,
    RemoteGitDeliveryReconstructionRequiredError,
    RemoteGitDeliveryRepository,
    RemoteGitDeliveryState,
    TaskStatus,
)
from tests.qualification.repository_maintenance_git_intake import (
    _completed_digest,
    _stored_delivery,
    _stored_preparation,
)
from tests.qualification.repository_maintenance_github_intake import _stored_github_delivery
from tests.qualification.repository_maintenance_identity import copy_identity
from tests.qualification.repository_maintenance_intake import MaintenanceTaskConflict


def _task_receipt(task, request, digest):
    if task is None:
        return "absent"
    if type(task.status) is not TaskStatus:
        return "conflicting"
    if task.status is not TaskStatus.COMPLETED:
        return "not_completed"
    if task.worker_id is not None or task.lease_expires_at is not None:
        return "conflicting"
    try:
        recorded = _completed_digest(task, request)
    except MaintenanceTaskConflict:
        return "conflicting"
    return "matches_latest" if recorded == digest else "different_result"


def _approval(result, approval):
    if result.approval_id is None and result.approval_fingerprint is None:
        return "not_recorded"
    if approval is None or (result.approval_id, result.approval_fingerprint) != (
        approval.approval_id,
        approval.fingerprint,
    ):
        raise MaintenanceTaskConflict()
    return "matches_saved_consent"


async def _git(application, identity):
    store = application.app.task_store
    preparation = await store.load_task(identity.git_preparation_task_id)
    delivery = await store.load_task(identity.git_delivery_task_id)
    if preparation is None and delivery is None:
        return {"evidence": "not_requested"}
    request = None
    approval = None
    if preparation is not None:
        request, _ = _stored_preparation(preparation, identity)
    if delivery is not None:
        approved_request, approval, _ = _stored_delivery(delivery, identity)
        if request is not None and request != approved_request:
            raise MaintenanceTaskConflict()
        request = approved_request
    if request is None:
        raise MaintenanceTaskConflict()
    repository = RemoteGitDeliveryRepository(application.artifact_store)
    receipts = await repository.load_lifecycle(request)
    if not receipts:
        return {"evidence": "absent", "request_fingerprint": request.fingerprint}
    receipt = receipts[-1]
    row = {
        "evidence": "recorded",
        "request_fingerprint": request.fingerprint,
        "state": receipt.state.value,
        "lifecycle_ordinal": receipt.ordinal,
        "result_evidence": "not_recorded",
        "cleanup_evidence": "not_recorded",
    }
    # PREPARED binds an intent fingerprint, not a result artifact. Cancellation
    # and an interrupted push can also have lifecycle-only evidence.
    if receipt.state is RemoteGitDeliveryState.PREPARED or receipt.evidence_sha256 is None:
        return row
    if receipt.state in {
        RemoteGitDeliveryState.PREPARING,
        RemoteGitDeliveryState.COMMITTING,
        RemoteGitDeliveryState.COMMITTED_LOCALLY,
        RemoteGitDeliveryState.PUSHING,
    }:
        raise MaintenanceTaskConflict()
    publication = await repository.load_result(request, receipt.evidence_sha256)
    result = publication.result
    if (result.state, result.tree, result.local_commit, result.reason_code) != (
        receipt.state,
        receipt.tree,
        receipt.commit,
        receipt.reason_code,
    ):
        raise MaintenanceTaskConflict()
    digest = publication.artifact.sha256
    return {
        **row,
        "result_evidence": "recorded",
        "result_digest": digest,
        "cleanup_evidence": "recorded_settled" if result.cleanup_settled else "recorded_unsettled",
        "commit": result.local_commit,
        "tree": result.tree,
        "approval_evidence": _approval(result, approval),
        "task_receipts": {
            "preparation": _task_receipt(preparation, request, digest),
            "delivery": _task_receipt(delivery, request, digest),
        },
    }


async def _github(application, identity):
    task = await application.app.task_store.load_task(identity.github_delivery_task_id)
    if task is None:
        return {"evidence": "not_requested"}
    request, approval, _ = _stored_github_delivery(task, identity)
    publication = await GitHubDeliveryRepository(application.artifact_store).latest(request)
    if publication is None:
        return {"evidence": "absent", "request_fingerprint": request.fingerprint}
    result = publication.result
    digest = publication.artifact.sha256
    return {
        "evidence": "recorded",
        "request_fingerprint": request.fingerprint,
        "state": result.state.value,
        "result_digest": digest,
        "head_commit": result.head_commit,
        "pull_request_number": None if result.pull_request is None else result.pull_request.number,
        "checks_state": None if result.checks_state is None else result.checks_state.value,
        "review_state": None if result.review_state is None else result.review_state.value,
        "next_poll_at": result.next_poll_at,
        "approval_evidence": _approval(result, approval),
        "task_receipt": _task_receipt(task, request, digest),
        "cleanup_evidence": "not_in_native_result",
    }


async def inspect_delivery_evidence(application, expected):
    """The host must authenticate and resolve the tenant-owned reservation first.

    Independent historical reads, not an atomic snapshot or business-success
    decision. Native readback validates content and authority before projection.
    """
    identity = copy_identity(expected)
    result: dict[str, object] = {"id": identity.public_id}
    for name, read in (("git", _git), ("github", _github)):
        try:
            result[name] = await read(application, identity)
        except (
            MaintenanceTaskConflict,
            GitHubDeliveryReconstructionRequiredError,
            RemoteGitDeliveryReconstructionRequiredError,
        ):
            result[name] = {"evidence": "conflicting"}
        except Exception:
            result[name] = {"evidence": "unavailable"}
    return result
