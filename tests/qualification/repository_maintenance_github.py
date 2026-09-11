"""Bounded observations within one worker lease; no approval or success inference."""

import asyncio
from contextlib import suppress
from datetime import datetime, timedelta

from workflows.maintenance_delivery import (  # ty: ignore[unresolved-import]
    run_verified_github_delivery,
)

from cayu import ExecutionDeadline, complete_managed_task, execution_deadline_scope
from tests.qualification.repository_maintenance_git_intake import load_verified_git_result
from tests.qualification.repository_maintenance_github_intake import (
    _configured_github_request,
    load_claimed_github_delivery,
)
from tests.qualification.repository_maintenance_intake import MaintenanceTaskConflict
from tests.qualification.repository_maintenance_lifetime import (
    raise_lifetime_failures,
    wait_owned_task,
)


async def handle_github_delivery_task(
    application, reservations, claimed, worker_id, connector_factory
):
    """Consume exact stored consent; retain the native connector until settlement."""
    identity, request, approval = await load_claimed_github_delivery(
        application.app, reservations, claimed, worker_id
    )
    task, product, remote = await load_verified_git_result(application, reservations, identity)
    if request != _configured_github_request(identity, product, remote):
        raise MaintenanceTaskConflict()
    # No resource acquisition before claim/source/config validation. The observer
    # owns this fresh connector's entire lifetime, including pre-dispatch failures.
    connector = connector_factory()
    publication = await observe_github_delivery(
        application, task, product, remote, connector, request, approval=approval
    )
    return await complete_managed_task(
        application.app.task_store,
        claimed,
        worker_id,
        {"request_fingerprint": request.fingerprint, "result_digest": publication.artifact.sha256},
    )


async def observe_github_delivery(
    application, task, product, remote, connector, request, *, approval=None
):
    """Return native outcome only after local connector settlement.

    Inputs are host-resolved authority, not raw route data. The enclosing worker
    must retain its lease and shared dependencies for this entire await. A returned
    failure or uncertainty is not verified delivery; no task is completed here.
    """
    stopped = asyncio.Event()

    def stop():
        stopped.set()
        connector.seal()

    async def lifetime():
        primary = None
        publication = None
        try:
            deadline = ExecutionDeadline(
                expires_at=datetime.fromisoformat(request.requested_at)
                + timedelta(seconds=request.limits.max_elapsed_seconds),
                source="maintenance",
                scope="github",
            )
            deadline.require_admission("github_observation")
            async with execution_deadline_scope(deadline):
                while not stopped.is_set():
                    publication = await run_verified_github_delivery(
                        application, task, product, remote, connector, request, approval=approval
                    )
                    delay = publication.result.next_poll_after_seconds
                    if delay is None:
                        break
                    with suppress(TimeoutError):
                        await asyncio.wait_for(stopped.wait(), timeout=delay)
        except BaseException as exc:
            primary = exc
        stop()
        try:
            while await connector.aclose(timeout_s=1) is not True:
                await asyncio.sleep(1)
        except BaseException as cleanup:
            if primary is not None:
                raise_lifetime_failures(primary, cleanup)
            raise
        if primary is not None:
            raise primary
        return publication

    return await wait_owned_task(asyncio.create_task(lifetime()), on_cancel=stop)
