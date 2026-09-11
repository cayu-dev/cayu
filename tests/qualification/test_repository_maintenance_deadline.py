"""Original workflow deadline across the generated product and persistent lookup."""

import asyncio
import importlib
import importlib.util
from contextlib import aclosing

import pytest

from cayu import (
    CodingProductArtifactRepository,
    DockerImageIdentity,
    EventQuery,
    EventType,
    ExecutionDeadline,
    ExecutionDeadlineExceeded,
    InMemoryKnowledgeStore,
    InMemorySessionStore,
    InMemoryTaskStore,
    LocalArtifactStore,
    SessionStatus,
)
from cayu.cli.project import project_context
from tests.core.test_queued_session_messages import RecordingOneShotProvider
from tests.qualification.repository_maintenance_case import materialize_seed_repository
from tests.qualification.repository_maintenance_toolchain import maintenance_toolchain
from tests.qualification.test_repository_maintenance_application import project as project


class WaitingProvider(RecordingOneShotProvider):
    """Cooperative local dispatch; no claims about opaque remote cancellation."""

    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.stopped = asyncio.Event()
        self.cancellation_count = 0

    async def stream(self, request):
        self.requests.append(request)
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            owner = asyncio.current_task()
            assert owner is not None
            self.cancellation_count = owner.cancelling()
            raise
        finally:
            self.stopped.set()
        # Preserve the provider's async-generator interface without a response.
        if False:
            yield


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("expired_before_start", [True, False])
def test_reserved_deadline_controls_generated_workflow(
    project, tmp_path, monkeypatch, backend, expired_before_start
):
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
        identities = importlib.import_module("domain.maintenance_identity")
        reservations = importlib.import_module("operations.maintenance_runs")
        workflow = importlib.import_module("workflows.maintenance_coding")
        requests = importlib.import_module("operations.maintenance_requests")
        request_domain = importlib.import_module("domain.maintenance_request")
        spec = importlib.util.spec_from_file_location(
            "maintenance_deadline_harness", project / "tests/test_coding_composition.py"
        )
        assert spec is not None and spec.loader is not None
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)
        monkeypatch.setattr(
            operations, "_configured_docker_authority", lambda _root: (profile, "/usr/bin/docker")
        )
        runners = []
        harness._install_fake_docker_factory(monkeypatch, target=target, created_runners=runners)
        provider = WaitingProvider()

        def build():
            if backend == "sqlite":
                return module.build_coding_product_application(
                    provider=provider, workspace_root=source
                )
            return module.build_coding_product_application(
                provider=provider,
                session_store=InMemorySessionStore(),
                task_store=InMemoryTaskStore(),
                knowledge_store=InMemoryKnowledgeStore(access_scope=operations._knowledge_scope()),
                artifact_store=LocalArtifactStore(tmp_path / "artifacts", store_id="deadline"),
                workspace_root=source,
            )

        async def close(application):
            assert await application.app.drain_environment_cleanups() is True
            if backend == "sqlite":
                await application.app.session_store.close()
                await application.app.task_store.close()
                await application.app.knowledge_store.close()

        async def scenario():
            application = build()
            owner = None
            try:
                original = ExecutionDeadline.after(
                    0 if expired_before_start else 10, source="maintenance", scope="coding"
                )
                path = tmp_path / "reservations.sqlite3"
                registry = reservations.SQLiteMaintenanceRunStore(path)
                await registry.initialize()
                provisional = domain.CodingProductTask(
                    product_run_id="inspection-product",
                    session_id="inspection-session",
                    task_id="inspection-task",
                    instruction="Repair the inclusive endpoint.",
                )
                accepted = await requests.capture_accepted_request(application, provisional)
                identity = await registry.reserve(
                    identities.MaintenanceRunIntent(
                        tenant="fixture",
                        subject="operator",
                        idempotency_key="deadline",
                        request_json=request_domain.encode_request(accepted),
                    ),
                    coding_expires_at=original.model_dump(mode="json")["expires_at"],
                )
                restored = await reservations.SQLiteMaintenanceRunStore(path).load_owned(
                    tenant="fixture", public_id=identity.public_id
                )
                assert restored == identity
                accepted = request_domain.decode_request(restored.intent.request_json)
                deadline = restored.coding_deadline()
                task = domain.CodingProductTask(
                    product_run_id=identity.product_run_id,
                    session_id=identity.session_id,
                    task_id=identity.task_id,
                    instruction=accepted.instruction,
                    parent_session_id=identity.workflow_session_id,
                    causal_budget_id=identity.workflow_session_id,
                )

                async def execute(app):
                    async with aclosing(
                        workflow.MaintenanceCodingWorkflow(app, task, accepted=accepted).execute(
                            identity.workflow_session_id, execution_deadline=deadline
                        )
                    ) as events:
                        async for event in events:
                            assert event.type is not EventType.WORKFLOW_COMPLETED

                owner = asyncio.create_task(execute(application))
                if not expired_before_start:
                    await asyncio.wait_for(provider.entered.wait(), timeout=15)
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(owner), timeout=20)
                assert owner.done() and not owner.cancelled() and owner.cancelling() == 0
                root = await application.app.session_store.load(identity.workflow_session_id)
                child = await application.app.session_store.load(identity.session_id)
                if expired_before_start:
                    assert root is None and child is None and not provider.requests
                    with pytest.raises(FileNotFoundError):
                        await CodingProductArtifactRepository(
                            application.artifact_store
                        ).load_request(identity.product_run_id, session_id=identity.session_id)
                else:
                    assert root is not None and child is not None
                    # Portable authority is UTC/source/scope. Reconstructed
                    # deadlines intentionally have fresh private monotonic anchors.
                    assert (
                        root.execution_deadline.model_dump()
                        == child.execution_deadline.model_dump()
                        == original.model_dump()
                    )
                    assert child.status is SessionStatus.INTERRUPTED
                    assert (
                        len(
                            await application.app.session_store.query_events(
                                EventQuery(
                                    session_id=identity.session_id,
                                    event_type=EventType.SESSION_INTERRUPTED,
                                )
                            )
                        )
                        == 1
                    )
                    assert provider.stopped.is_set() and provider.cancellation_count == 1
                    assert len(provider.requests) == 1
                assert (
                    await application.app.session_store.query_events(
                        EventQuery(
                            session_id=identity.workflow_session_id,
                            event_type=EventType.WORKFLOW_COMPLETED,
                        )
                    )
                    == []
                )
                assert all(runner.closed for runner in runners)
                if backend == "sqlite":
                    await close(application)
                    application = build()
                replay_identity = await reservations.SQLiteMaintenanceRunStore(path).load_owned(
                    tenant="fixture", public_id=identity.public_id
                )
                assert replay_identity == identity
                deadline = replay_identity.coding_deadline()
                assert deadline.model_dump() == original.model_dump() and deadline.expired
                with pytest.raises(ExecutionDeadlineExceeded):
                    await execute(application)
                assert len(provider.requests) == (0 if expired_before_start else 1)
            finally:
                if owner is not None:
                    if not owner.done():
                        owner.cancel()
                    await asyncio.gather(owner, return_exceptions=True)
                await close(application)

        asyncio.run(scenario())
