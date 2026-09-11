"""Git phase handlers using the existing native broker and Runtime lease."""

import asyncio

from workflows.maintenance_delivery import (  # ty: ignore[unresolved-import]
    run_verified_git_delivery,
)

from cayu import (
    LocalArtifactStore,
    RemoteGitDeliveryBroker,
    complete_managed_task,
)
from tests.qualification.repository_maintenance_git_intake import (
    _configured_git_request,
    load_claimed_git_delivery,
    load_claimed_git_preparation,
)
from tests.qualification.repository_maintenance_intake import MaintenanceTaskConflict
from tests.qualification.repository_maintenance_lifetime import wait_owned_task
from tests.qualification.repository_maintenance_results import load_verified_coding_result


async def handle_git_preparation_task(application, reservations, claimed, worker_id, broker):
    _validate_broker(application, broker)
    identity, request = await load_claimed_git_preparation(
        application.app, reservations, claimed, worker_id
    )
    return await _run_claimed_git(
        application, reservations, claimed, worker_id, broker, identity, request, None
    )


async def handle_git_delivery_task(application, reservations, claimed, worker_id, broker):
    _validate_broker(application, broker)
    identity, request, approval = await load_claimed_git_delivery(
        application.app, reservations, claimed, worker_id
    )
    return await _run_claimed_git(
        application, reservations, claimed, worker_id, broker, identity, request, approval
    )


def _validate_broker(application, broker):
    if (
        type(broker) is not RemoteGitDeliveryBroker
        or type(application.artifact_store) is not LocalArtifactStore
    ):
        raise ValueError("Maintenance preparation requires native local-artifact delivery.")


async def _run_claimed_git(
    application, reservations, claimed, worker_id, broker, identity, request, approval
):
    """Retain native local-artifact work before returning control to task settlement.

    The enclosing role owns shared dependencies and must keep them open throughout
    this await. A native result is operation evidence, not delivery success.
    """
    task, product = await load_verified_coding_result(application, reservations, identity)
    expected = _configured_git_request(identity, product)
    if request != expected:
        raise MaintenanceTaskConflict()
    result = await wait_owned_task(
        asyncio.create_task(
            run_verified_git_delivery(
                application, task, product, broker, request, approval=approval
            )
        )
    )
    return await complete_managed_task(
        application.app.task_store,
        claimed,
        worker_id,
        {"request_fingerprint": request.fingerprint, "result_digest": result.artifact.sha256},
    )
