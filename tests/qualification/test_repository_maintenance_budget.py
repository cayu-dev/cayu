"""Generated product budget denial through Runtime; no live price or billing claim."""

import asyncio
import importlib
import importlib.util

from cayu import (
    CodingProductState,
    DockerImageIdentity,
    EventQuery,
    EventType,
    InMemoryBudgetLedger,
    InMemoryKnowledgeStore,
    InMemorySessionStore,
    InMemoryTaskStore,
    LocalArtifactStore,
)
from cayu.cli.project import project_context
from tests.cli.test_scaffold_coding_budget import denial_policy
from tests.core.test_queued_session_messages import RecordingOneShotProvider
from tests.qualification.repository_maintenance_case import materialize_seed_repository
from tests.qualification.repository_maintenance_toolchain import maintenance_toolchain
from tests.qualification.test_repository_maintenance_application import project as project


def test_generated_product_budget_denies_before_provider_dispatch(project, tmp_path, monkeypatch):
    monkeypatch.setenv("CAYU_MODEL", "fake-model")
    source = tmp_path / "source"
    materialize_seed_repository(source)
    target = tmp_path / "target"
    target.mkdir()
    profile = maintenance_toolchain(
        image_identity=DockerImageIdentity(reference="fixture@sha256:" + "a" * 64),
        architecture="amd64",
        build_context_sha256="sha256:" + "b" * 64,
    )
    with project_context(project):
        operations = importlib.import_module("operations.coding")
        module = importlib.import_module("app")
        domain = importlib.import_module("domain.coding_product")
        spec = importlib.util.spec_from_file_location(
            "maintenance_budget_harness", project / "tests/test_coding_composition.py"
        )
        assert spec is not None and spec.loader is not None
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)
        monkeypatch.setattr(
            operations, "_configured_docker_authority", lambda _root: (profile, "/usr/bin/docker")
        )
        runners = []
        harness._install_fake_docker_factory(monkeypatch, target=target, created_runners=runners)
        provider = RecordingOneShotProvider()
        application = module.build_coding_product_application(
            provider=provider,
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
            knowledge_store=InMemoryKnowledgeStore(access_scope=operations._knowledge_scope()),
            artifact_store=LocalArtifactStore(tmp_path / "artifacts", store_id="budget-artifacts"),
            workspace_root=source,
            budget_policy=denial_policy(),
            budget_ledger=InMemoryBudgetLedger(),
        )
        task = domain.CodingProductTask(
            product_run_id="budget-product",
            session_id="budget-session",
            task_id="budget-task",
            instruction="Repair the inclusive endpoint.",
        )

        async def scenario():
            publication = await application.run(task)
            assert publication.candidate.state is CodingProductState.CANCELLED
            assert not provider.requests
            for kind in (EventType.BUDGET_RESERVATION_FAILED, EventType.BUDGET_LIMIT_REACHED):
                records = await application.app.session_store.query_events(
                    EventQuery(session_id=task.session_id, event_type=kind)
                )
                assert len(records) == 1
                assert records[0].event.payload["maximum"] == "0.5"
                if kind is EventType.BUDGET_RESERVATION_FAILED:
                    assert records[0].event.payload["accepted"] is False
                    assert records[0].event.payload["requested"] == "1"
                    # Rejected reservations report projected usage, including
                    # the proposed dispatch, not already incurred charges.
                    assert records[0].event.payload["actual"] == "1"
            assert await application.app.drain_environment_cleanups() is True
            assert all(runner.closed for runner in runners)

        asyncio.run(scenario())
