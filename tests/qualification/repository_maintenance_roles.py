"""Named process adapters; Runtime and CLI retain their existing ownership."""

import asyncio
from uuid import uuid4

from app import AGENT  # ty: ignore[unresolved-import]
from integrations.maintenance_git_host import configured_git_broker  # ty: ignore[unresolved-import]
from integrations.maintenance_github_host import (  # ty: ignore[unresolved-import]
    configured_github_connector_factory,
)
from operations.maintenance_git_worker import (  # ty: ignore[unresolved-import]
    handle_git_delivery_task,
    handle_git_preparation_task,
)
from operations.maintenance_github import (  # ty: ignore[unresolved-import]
    handle_github_delivery_task,
)
from operations.maintenance_worker import handle_coding_task  # ty: ignore[unresolved-import]

from cayu import TaskQuery, run_task_worker
from tests.qualification.repository_maintenance_deployment import bind_maintenance_deployment
from tests.qualification.repository_maintenance_lifetime import (
    close_deployment,
    raise_lifetime_failures,
    wait_owned_task,
)


def _coding_handler(deployment):
    async def handle(_app, task, worker_id):
        await handle_coding_task(deployment.application, deployment.reservations, task, worker_id)

    return handle


def _git_handler(deployment, handler):
    broker = configured_git_broker(deployment.application.artifact_store)

    async def handle(_app, task, worker_id):
        await handler(deployment.application, deployment.reservations, task, worker_id, broker)

    return handle


def _github_handler(deployment):
    factory = configured_github_connector_factory(deployment.application.artifact_store)

    async def handle(_app, task, worker_id):
        await handle_github_delivery_task(
            deployment.application, deployment.reservations, task, worker_id, factory
        )

    return handle


async def _lifetime(deployment, stop, task_type, handler_builder):
    app = deployment.application.app
    primary = None
    try:
        await deployment.validate_startup_schema()
        handle = handler_builder(deployment)
        await run_task_worker(
            app,
            app.task_store,
            handle,
            worker_id=f"{task_type}-{uuid4().hex}",
            query=TaskQuery(type=task_type),
            stop=stop,
            recover_interrupted_handoffs=False,
        )
    except BaseException as exc:
        primary = exc
    stop.set()
    try:
        await close_deployment(deployment)
    except BaseException as cleanup:
        if primary is not None:
            raise_lifetime_failures(primary, cleanup)
        raise
    if primary is not None:
        raise primary


async def run_coding(app, stop):
    await _run_role(app, stop, "maintenance.coding", _coding_handler)


async def run_git_preparation(app, stop):
    await _run_role(
        app,
        stop,
        "maintenance.git_preparation",
        lambda owned: _git_handler(owned, handle_git_preparation_task),
    )


async def run_git_delivery(app, stop):
    await _run_role(
        app,
        stop,
        "maintenance.git_delivery",
        lambda owned: _git_handler(owned, handle_git_delivery_task),
    )


async def run_github_delivery(app, stop):
    await _run_role(app, stop, "maintenance.github_delivery", _github_handler)


async def _run_role(app, stop, task_type, handler_builder):
    """Stop cooperatively; preserve caller cancellation until owned work settles."""
    deployment = bind_maintenance_deployment(app, agent_name=AGENT.name)
    lifetime = asyncio.create_task(_lifetime(deployment, stop, task_type, handler_builder))
    await wait_owned_task(lifetime, on_cancel=stop.set)
