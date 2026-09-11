"""Production construction and startup schema checks; caller owns the lifetime."""

import asyncio
from dataclasses import dataclass, field
from math import isfinite

from app import build_coding_product_application  # ty: ignore[unresolved-import]
from configuration.coding_storage import GENERATED_STORE_PROFILE  # ty: ignore[unresolved-import]
from configuration.settings import (  # ty: ignore[unresolved-import]
    configured_database_url,
    configured_public_authority_alias_codec,
)
from workflows.coding_product import CodingProductApplication  # ty: ignore[unresolved-import]

from cayu import (
    DockerCodingEnvironmentFactory,
    LocalWorkspace,
    PostgresBudgetLedger,
    PostgresKnowledgeStore,
    PostgresSessionStore,
    PostgresTaskStore,
    SubagentTool,
)
from cayu.storage.migrations import SchemaMode
from tests.qualification.repository_maintenance_budget import require_maintenance_budget
from tests.qualification.repository_maintenance_runs import PostgresMaintenanceRunStore


@dataclass
class _ShutdownOwner:
    task: asyncio.Task[bool] | None = None


def _retrieve_shutdown_failure(task):
    if not task.cancelled():
        task.exception()


@dataclass(frozen=True, repr=False)
class MaintenanceDeployment:
    """Reachable dependencies even if startup validation fails partway through.

    This object is not a worker, a lease, a quiescence receipt, or an automatic
    closer. The caller must retain it until owned activity has settled.
    """

    application: CodingProductApplication
    reservations: PostgresMaintenanceRunStore
    _shutdown: _ShutdownOwner = field(default_factory=_ShutdownOwner, init=False, compare=False)

    def _stores(self):
        app = self.application.app
        stores = (
            (app.session_store, PostgresSessionStore),
            (app.task_store, PostgresTaskStore),
            (app.knowledge_store, PostgresKnowledgeStore),
            (app.budget_ledger, PostgresBudgetLedger),
        )
        if any(type(store) is not expected for store, expected in stores):
            raise ValueError("Maintenance deployment requires its generated PostgreSQL stores.")
        return tuple(store for store, _expected in stores)

    async def validate_startup_schema(self):
        """Validate existing schemas; not a migration or periodic health probe."""
        require_maintenance_budget(self.application.app.budget_policy)
        for store in self._stores():
            await store.ensure_schema()
        await self.reservations.check_ready()

    async def aclose(self, *, timeout_s: float = 30.0) -> bool:
        """Observe owned shutdown after the caller permanently stops dispatch.

        Keep this bundle reachable on False, cancellation or error. An active
        close is never cancelled or replaced by another observer. Only a finished
        no-close drain failure can start a new attempt; close errors stay failed.
        Do not reuse the application after starting shutdown.
        """
        if type(timeout_s) not in {int, float} or not isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be a finite positive number.")
        task = self._shutdown.task
        if task is None or (
            task.done()
            and not task.cancelled()
            and task.exception() is None
            and task.result() is False
        ):
            task = asyncio.create_task(self._close_when_settled(timeout_s))
            self._shutdown.task = task
            task.add_done_callback(_retrieve_shutdown_failure)
        done, _pending = await asyncio.wait((task,), timeout=timeout_s)
        return task.result() if done else False

    async def _close_when_settled(self, timeout_s):
        app = self.application.app
        stores = self._stores()
        close_provider = getattr(app.get_provider(), "aclose", None)
        if not callable(close_provider):
            raise ValueError(
                "Maintenance provider requires an explicit asynchronous close capability."
            )
        if not await self.quiesce(timeout_s=timeout_s):
            return False
        # These awaits may exceed the observer's deadline. Their task remains
        # owned: cancellation of an observer must not strand a detached client.
        if await close_provider() is not None:
            raise RuntimeError("Maintenance provider closure did not confirm completion.")
        for store in stores:
            await store.close()
        return True

    async def quiesce(self, *, timeout_s: float = 30.0) -> bool:
        """Observe settled work while preserving dependencies for evidence reads.

        The caller must stop dispatch first and retain this deployment on False,
        cancellation or error. This is not closure, a recovery receipt, or proof
        about effects outside the Runtime's owned drains.
        """
        if type(timeout_s) not in {int, float} or not isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be a finite positive number.")
        app = self.application.app
        registries = {}
        for agent in app.describe().agents:
            for registered in app.get_agent(agent.name).tools.values():
                if isinstance(registered.tool, SubagentTool):
                    registry = registered.tool.background_task_registry
                    registries[id(registry)] = registry
        drains = (
            *(registry.drain for registry in registries.values()),
            app.drain_background_interruptions,
            app.drain_recovery_cleanups,
            app.drain_provider_operation_cancellations,
            app.drain_environment_cleanups,
            app.drain_knowledge_publications,
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        for drain in drains:
            remaining = deadline - loop.time()
            if remaining <= 0 or await drain(timeout_s=remaining) is not True:
                return False
        return True


def _deployment_dsn():
    if GENERATED_STORE_PROFILE != "postgres":
        raise ValueError("Maintenance deployment requires the PostgreSQL profile.")
    dsn = configured_database_url()
    if type(dsn) is not str or not dsn.strip():
        raise ValueError("Maintenance deployment requires CAYU_DATABASE_URL.")
    if configured_public_authority_alias_codec() is None:
        raise ValueError("Maintenance deployment requires persistent public authority alias keys.")
    return dsn


def bind_maintenance_deployment(app, *, agent_name):
    """Bind once to the configured factory's app; never construct a second runtime.

    This host-only adapter does not authenticate arbitrary injected applications
    or grant execution permission. The role owns the returned bundle's lifetime.
    """
    require_maintenance_budget(app.budget_policy)
    dsn = _deployment_dsn()
    app.get_agent(agent_name)
    factory = app.get_environment_factory()
    if type(factory) is not DockerCodingEnvironmentFactory:
        raise ValueError("Maintenance deployment requires its native Docker coding factory.")
    if type(factory.source_workspace) is not LocalWorkspace:
        raise ValueError("Maintenance deployment requires its local source workspace.")
    if factory.configured_artifact_store is None:
        raise ValueError("Maintenance deployment requires its configured artifact store.")
    application = CodingProductApplication(
        app,
        source_workspace=factory.source_workspace,
        artifact_store=factory.configured_artifact_store,
        toolchain_profile=factory.toolchain_profile,
        agent_name=agent_name,
        project_root=factory.source_workspace.root,
    )
    deployment = MaintenanceDeployment(application, PostgresMaintenanceRunStore(dsn))
    deployment._stores()
    return deployment


def build_maintenance_deployment(*, budget_policy, workspace_root=None):
    """Preserve generated durable identities and require explicit deployment config.

    Docker inspection and normal local artifact-root initialization belong to the
    generated builder. Database pools remain lazy; no migration or worker starts.
    """
    policy = require_maintenance_budget(budget_policy)
    dsn = _deployment_dsn()
    ledger = PostgresBudgetLedger(dsn, schema_mode=SchemaMode.VALIDATE)
    application = build_coding_product_application(
        workspace_root=workspace_root,
        budget_policy=policy,
        budget_ledger=ledger,
    )
    return MaintenanceDeployment(application, PostgresMaintenanceRunStore(dsn))
