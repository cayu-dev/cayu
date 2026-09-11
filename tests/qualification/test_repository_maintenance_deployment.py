"""Generated production wiring; mocked DB validation is not live PostgreSQL proof."""

import asyncio
import base64
import importlib
import json

import pytest

from cayu import (
    DockerImageIdentity,
    PostgresBudgetLedger,
    PostgresKnowledgeStore,
    PostgresSessionStore,
    PostgresTaskStore,
)
from cayu.cli.project import project_context
from tests.cli.test_scaffold_coding_budget import denial_policy
from tests.core.test_queued_session_messages import RecordingOneShotProvider
from tests.qualification.repository_maintenance_case import materialize_seed_repository
from tests.qualification.repository_maintenance_toolchain import maintenance_toolchain
from tests.qualification.test_repository_maintenance_application import project as project


@pytest.fixture
def deployment(project, tmp_path, monkeypatch):
    monkeypatch.setenv("CAYU_DATABASE_URL", "postgresql://fixture@127.0.0.1/maintenance")
    monkeypatch.setenv("CAYU_PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID", "fixture")
    monkeypatch.setenv(
        "CAYU_PUBLIC_AUTHORITY_ALIAS_KEYS",
        json.dumps({"fixture": base64.urlsafe_b64encode(b"a" * 32).decode().rstrip("=")}),
    )
    monkeypatch.setenv("CAYU_MODEL", "fake-model")
    source = tmp_path / "source"
    materialize_seed_repository(source)
    profile = maintenance_toolchain(
        image_identity=DockerImageIdentity(reference="fixture@sha256:" + "a" * 64),
        architecture="amd64",
        build_context_sha256="sha256:" + "b" * 64,
    )
    with project_context(project):
        operations = importlib.import_module("operations.coding")
        root = importlib.import_module("app")
        module = importlib.import_module("operations.maintenance_deployment")
        provider = RecordingOneShotProvider()
        monkeypatch.setattr(
            operations, "_configured_docker_authority", lambda _root: (profile, "/usr/bin/docker")
        )
        monkeypatch.setattr(root, "configured_provider", lambda: provider)
        yield module, source, provider


def test_deployment_preserves_generated_store_identity_without_connecting(deployment, monkeypatch):
    from psycopg_pool import AsyncConnectionPool

    module, source, provider = deployment

    async def forbidden(*args, **kwargs):
        pytest.fail("Construction connected to PostgreSQL")

    monkeypatch.setattr(AsyncConnectionPool, "open", forbidden)
    first = module.build_maintenance_deployment(
        budget_policy=denial_policy(), workspace_root=source
    )
    second = module.build_maintenance_deployment(
        budget_policy=denial_policy(), workspace_root=source
    )
    app = first.application.app
    assert type(app.session_store) is PostgresSessionStore
    assert type(app.task_store) is PostgresTaskStore
    assert type(app.knowledge_store) is PostgresKnowledgeStore
    assert type(app.budget_ledger) is PostgresBudgetLedger
    assert app.budget_policy == denial_policy()
    assert app.get_provider() is provider and not provider.requests
    assert (
        first.application.artifact_store.id
        == second.application.artifact_store.id
        == "coding-artifacts"
    )
    assert "postgresql" not in repr(first)

    # Runtime profile inspection legitimately reads task authority from storage.
    # Without PostgreSQL we can prove these declarations, not the full profile.
    for name in ("subagent", "subagent_result"):
        one = (
            app.get_agent(first.application.agent_name)
            .tools[name]
            .tool.spec.execution_profile_identity
        )
        two = (
            second.application.app.get_agent(second.application.agent_name)
            .tools[name]
            .tool.spec.execution_profile_identity
        )
        assert one is not None and one == two
    first_tool = app.get_agent(first.application.agent_name).tools["subagent"].tool
    second_tool = (
        second.application.app.get_agent(second.application.agent_name).tools["subagent"].tool
    )
    assert first_tool.background_task_registry is not second_tool.background_task_registry


def test_api_factory_builds_one_native_graph_without_connecting(deployment, monkeypatch):
    from psycopg_pool import AsyncConnectionPool

    _deployment_module, source, provider = deployment
    monkeypatch.setenv("CAYU_WORKSPACE_ROOT", str(source))
    monkeypatch.setenv("CAYU_MAINTENANCE_BUDGET_JSON", denial_policy().model_dump_json())
    monkeypatch.setenv(
        "CAYU_MAINTENANCE_ACCESS_JSON",
        json.dumps(
            {
                "operator_token": "operator",
                "product_tokens": {"product": {"tenant_id": "tenant", "subject_id": "subject"}},
            }
        ),
    )
    module = importlib.import_module("operations.maintenance_api")
    original = module.build_maintenance_app
    built = []

    def build():
        app = original()
        built.append(app)
        return app

    async def forbidden(*args, **kwargs):
        pytest.fail("ASGI factory connected to PostgreSQL")

    monkeypatch.setattr(module, "build_maintenance_app", build)
    monkeypatch.setattr(AsyncConnectionPool, "open", forbidden)
    result = module.build_api()
    assert callable(result) and len(built) == 1
    assert type(built[0].session_store) is PostgresSessionStore
    assert type(built[0].budget_ledger) is PostgresBudgetLedger
    assert built[0].budget_policy == denial_policy()
    assert built[0].get_provider() is provider and not provider.requests


def test_worker_binding_reuses_the_registered_graph_without_reconstruction(deployment, monkeypatch):
    from psycopg_pool import AsyncConnectionPool

    module, source, provider = deployment
    original = module.build_maintenance_deployment(
        budget_policy=denial_policy(), workspace_root=source
    ).application

    def forbidden(*args, **kwargs):
        pytest.fail("Worker binding rebuilt the application or connected to PostgreSQL")

    monkeypatch.setattr(module, "build_coding_product_application", forbidden)
    monkeypatch.setattr(AsyncConnectionPool, "open", forbidden)
    bound = module.bind_maintenance_deployment(original.app, agent_name=original.agent_name)
    factory = original.app.get_environment_factory()
    assert bound.application.app is original.app
    assert bound.application.app.get_provider() is provider and not provider.requests
    assert bound.application.source_workspace is original.source_workspace
    assert bound.application.artifact_store is original.artifact_store
    assert bound.application.toolchain_profile is factory.toolchain_profile
    assert bound.application.toolchain_profile == original.toolchain_profile
    assert bound.application.project_root == original.project_root
    assert bound.application.agent_name == original.agent_name
    assert bound._stores() == (
        original.app.session_store,
        original.app.task_store,
        original.app.knowledge_store,
        original.app.budget_ledger,
    )


@pytest.mark.parametrize("invalid", ["factory", "artifacts", "store", "agent"])
def test_worker_binding_rejects_unsupported_graph(deployment, monkeypatch, invalid):
    module, source, _provider = deployment
    original = module.build_maintenance_deployment(
        budget_policy=denial_policy(), workspace_root=source
    ).application
    agent = original.agent_name
    if invalid == "factory":
        monkeypatch.setattr(original.app, "get_environment_factory", lambda: object())
    elif invalid == "artifacts":
        # Public native factory construction, with no configured artifact store.
        from cayu import DockerCodingEnvironmentFactory

        factory = DockerCodingEnvironmentFactory(
            source_workspace=original.source_workspace,
            toolchain_profile=original.toolchain_profile,
        )
        monkeypatch.setattr(original.app, "get_environment_factory", lambda: factory)
    elif invalid == "store":
        monkeypatch.setattr(original.app, "task_store", None)
    else:
        agent = "not-registered"
    with pytest.raises((ValueError, KeyError)):
        module.bind_maintenance_deployment(original.app, agent_name=agent)


@pytest.mark.parametrize("backend", ["sqlite"])
def test_deployment_rejects_actual_sqlite_generated_profile(deployment, backend):
    module, source, provider = deployment
    assert backend == module.GENERATED_STORE_PROFILE
    with pytest.raises(ValueError, match="PostgreSQL profile"):
        module.build_maintenance_deployment(budget_policy=denial_policy(), workspace_root=source)
    assert not provider.requests


@pytest.mark.parametrize("missing", ["database", "alias", "profile"])
def test_deployment_rejects_missing_configuration_before_construction(
    deployment, monkeypatch, missing
):
    module, source, provider = deployment
    if missing == "database":
        monkeypatch.delenv("CAYU_DATABASE_URL")
    elif missing == "alias":
        monkeypatch.delenv("CAYU_PUBLIC_AUTHORITY_ALIAS_KEYS")
        monkeypatch.delenv("CAYU_PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID")
    else:
        monkeypatch.setattr(module, "GENERATED_STORE_PROFILE", "sqlite")

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid deployment reached construction")

    monkeypatch.setattr(module, "build_coding_product_application", forbidden)
    with pytest.raises(ValueError):
        module.build_maintenance_deployment(budget_policy=denial_policy(), workspace_root=source)
    assert not provider.requests


@pytest.mark.parametrize("failure", [None, "task", "cancel"])
def test_startup_schema_validation_preserves_failure_and_cancellation(
    deployment, monkeypatch, failure
):
    module, source, provider = deployment
    owned = module.build_maintenance_deployment(
        budget_policy=denial_policy(), workspace_root=source
    )
    app = owned.application.app
    calls = []
    original = RuntimeError("schema incompatible")

    async def scenario():
        entered = asyncio.Event()

        def validator(name):
            async def validate():
                calls.append(name)
                if name == "task" and failure == "task":
                    raise original
                if name == "task" and failure == "cancel":
                    entered.set()
                    await asyncio.Event().wait()

            return validate

        for name, store in (
            ("session", app.session_store),
            ("task", app.task_store),
            ("knowledge", app.knowledge_store),
            ("budget", app.budget_ledger),
        ):
            monkeypatch.setattr(store, "ensure_schema", validator(name))
        monkeypatch.setattr(owned.reservations, "check_ready", validator("reservations"))

        async def forbidden():
            pytest.fail("Startup initialized application schema")

        monkeypatch.setattr(owned.reservations, "initialize", forbidden)
        task = asyncio.create_task(owned.validate_startup_schema())
        try:
            if failure == "cancel":
                await asyncio.wait_for(entered.wait(), 5)
                task.cancel("startup-stop")
                with pytest.raises(asyncio.CancelledError, match="startup-stop"):
                    await task
                assert task.cancelled() and task.cancelling() == 1
            elif failure == "task":
                with pytest.raises(RuntimeError) as caught:
                    await task
                assert caught.value is original
            else:
                await task
            assert calls == (
                ["session", "task"]
                if failure
                else ["session", "task", "knowledge", "budget", "reservations"]
            )
            assert not provider.requests
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
