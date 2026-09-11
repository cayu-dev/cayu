"""Read an already completed coding result; never run coding or approve delivery."""

from domain.coding_product import CodingProductTask  # ty: ignore[unresolved-import]
from operations.maintenance_requests import (  # ty: ignore[unresolved-import]
    capture_accepted_request,
)

from cayu import CodingProductArtifactRepository, TaskStatus
from tests.qualification.repository_maintenance_identity import copy_identity
from tests.qualification.repository_maintenance_intake import load_owned_coding_task
from tests.qualification.repository_maintenance_request import decode_request


class MaintenanceResultUnavailable(ValueError):
    def __init__(self):
        super().__init__("Completed maintenance coding evidence is unavailable or conflicting.")


def coding_task_from_identity(identity):
    """Reconstruct saved task data, not authentication or dispatch permission."""
    identity = copy_identity(identity)
    accepted = decode_request(identity.intent.request_json)
    return CodingProductTask(
        product_run_id=identity.product_run_id,
        session_id=identity.session_id,
        task_id=identity.task_id,
        instruction=accepted.instruction,
        source_origin_id=accepted.source_origin_id,
        source_destination_id=accepted.source_destination_id,
        parent_session_id=identity.workflow_session_id,
        causal_budget_id=identity.workflow_session_id,
    )


async def load_verified_coding_result(application, reservations, expected):
    """Host-only readback after authorization; retain fresh delivery checks later.

    A matching identity does not authenticate a caller. This method cannot repair
    missing completion, authorize an external effect, or establish future quiescence.
    """
    expected = copy_identity(expected)
    identity = await reservations.load_owned(
        tenant=expected.intent.tenant, public_id=expected.public_id
    )
    if identity is None:
        raise MaintenanceResultUnavailable()
    identity = copy_identity(identity)
    if identity != expected:
        raise MaintenanceResultUnavailable()
    completed = await load_owned_coding_task(application.app, identity)
    if completed is None or completed.status is not TaskStatus.COMPLETED:
        raise MaintenanceResultUnavailable()
    result = completed.result
    if (
        type(result) is not dict
        or any(type(key) is not str for key in result)
        or set(result) != {"product_run_id", "result_digest"}
        or any(type(value) is not str for value in result.values())
        or result["product_run_id"] != identity.product_run_id
        or len(result["result_digest"]) != 64
        or any(char not in "0123456789abcdef" for char in result["result_digest"])
    ):
        raise MaintenanceResultUnavailable()
    digest = result["result_digest"]
    task = coding_task_from_identity(identity)
    if await capture_accepted_request(application, task) != decode_request(
        identity.intent.request_json
    ):
        raise MaintenanceResultUnavailable()
    repository = CodingProductArtifactRepository(application.artifact_store)
    admitted = await repository.load_request(task.product_run_id, session_id=task.session_id)
    publication = await repository.load_publication(
        request_fingerprint=admitted.fingerprint, digest=digest
    )
    await application.verify(task, publication)
    return task, publication
