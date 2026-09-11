"""Authenticated ASGI host factory; no active services start during construction."""

from contextlib import asynccontextmanager

from app import AGENT, build_maintenance_app  # ty: ignore[unresolved-import]
from configuration.maintenance import configured_maintenance_access  # ty: ignore[unresolved-import]

from tests.qualification.repository_maintenance_asgi import MaintenanceASGI
from tests.qualification.repository_maintenance_deployment import bind_maintenance_deployment
from tests.qualification.repository_maintenance_http import build_maintenance_server
from tests.qualification.repository_maintenance_lifetime import (
    close_deployment,
    raise_lifetime_failures,
)


def build_api():
    access = configured_maintenance_access()
    app = build_maintenance_app()
    deployment = bind_maintenance_deployment(app, agent_name=AGENT.name)

    @asynccontextmanager
    async def lifespan(_server):
        primary = None
        try:
            await deployment.validate_startup_schema()
            yield
        except BaseException as exc:
            primary = exc
        try:
            await close_deployment(deployment)
        except BaseException as cleanup:
            if primary is not None:
                raise_lifetime_failures(primary, cleanup)
            raise
        if primary is not None:
            raise primary

    return MaintenanceASGI(
        build_maintenance_server(
            deployment.application, deployment.reservations, access, lifespan=lifespan
        )
    )
