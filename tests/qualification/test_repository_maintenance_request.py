"""Accepted configuration through the emitted workflow's real read-only guard."""

import asyncio
import importlib
import warnings
from dataclasses import replace

import pytest

from cayu import (
    DockerImageIdentity,
    InMemoryKnowledgeStore,
    InMemorySessionStore,
    InMemoryTaskStore,
    LocalArtifactStore,
    LocalWorkspace,
)
from cayu.cli.project import project_context
from cayu.storage.sqlite import SQLiteTaskStore
from tests.cli.test_scaffold_coding_budget import denial_policy
from tests.core.test_queued_session_messages import RecordingOneShotProvider
from tests.qualification.repository_maintenance_case import materialize_seed_repository
from tests.qualification.repository_maintenance_toolchain import maintenance_toolchain
from tests.qualification.test_repository_maintenance_application import project as project


@pytest.fixture
def consumer(project, tmp_path, monkeypatch, request):
    source = tmp_path / "source"
    materialize_seed_repository(source)
    profile = maintenance_toolchain(
        image_identity=DockerImageIdentity(reference="fixture@sha256:" + "a" * 64),
        architecture="amd64",
        build_context_sha256="sha256:" + "b" * 64,
    )
    with project_context(project):
        operations = importlib.import_module("operations.coding")
        monkeypatch.setattr(
            operations, "_configured_docker_authority", lambda _root: (profile, "/usr/bin/docker")
        )
        provider = RecordingOneShotProvider()
        task_store = (
            SQLiteTaskStore(tmp_path / "worker-tasks.sqlite")
            if getattr(request, "param", "memory") == "sqlite"
            else InMemoryTaskStore()
        )
        application = importlib.import_module("app").build_coding_product_application(
            provider=provider,
            workspace_root=source,
            session_store=InMemorySessionStore(),
            task_store=task_store,
            knowledge_store=InMemoryKnowledgeStore(access_scope=operations._knowledge_scope()),
            artifact_store=LocalArtifactStore(tmp_path / "artifacts", store_id="requests"),
        )
        task = importlib.import_module("domain.coding_product").CodingProductTask(
            product_run_id="product",
            session_id="child",
            task_id="task",
            instruction="Repair the inclusive upper endpoint.",
            parent_session_id="root",
            causal_budget_id="root",
        )
        try:
            yield (
                application,
                task,
                provider,
                importlib.import_module("domain.maintenance_request"),
                importlib.import_module("operations.maintenance_requests"),
                importlib.import_module("workflows.maintenance_coding").MaintenanceCodingWorkflow,
            )
        finally:
            if isinstance(task_store, SQLiteTaskStore):
                asyncio.run(task_store.close())


def test_capture_roundtrip_and_bounded_instruction(consumer):
    app, task, provider, domain, requests, _ = consumer

    async def scenario():
        accepted = await requests.capture_accepted_request(app, task)
        assert domain.decode_request(domain.encode_request(accepted)) == accepted
        assert accepted.repository_root == str(app.project_root)
        assert accepted.source_workspace_id == "coding-source-workspace"
        assert accepted.settlement_json == domain.default_maintenance_settlement().model_dump_json()
        assert not provider.requests
        for instruction in ("", "  ", "a" * 4097, "é" * 2049, "\ud800", True):
            with pytest.raises(ValueError):
                await requests.capture_accepted_request(app, replace(task, instruction=instruction))
        maximum = await requests.capture_accepted_request(
            app, replace(task, instruction="é" * 2048)
        )
        assert maximum.instruction == "é" * 2048
        for invalid in ("{}", '{"instruction":"a","instruction":"b"}', "[]"):
            with pytest.raises(ValueError):
                domain.decode_request(invalid)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field",
    [
        "instruction",
        "repository_root",
        "source_workspace_id",
        "source_origin_id",
        "source_destination_id",
        "artifact_store_id",
        "toolchain_profile_fingerprint",
        "execution_profile_fingerprint",
        "corpus_fingerprint",
        "probe_fingerprint",
        "base_revision",
        "settlement_json",
    ],
)
def test_each_changed_expected_field_rejects_before_product_dispatch(consumer, field):
    app, task, provider, domain, requests, workflow = consumer

    async def scenario():
        accepted = await requests.capture_accepted_request(app, task)
        data = accepted.model_dump()
        if field == "execution_profile_fingerprint":
            data[field] = "0" * 64
        elif field.endswith("fingerprint"):
            data[field] = "sha256:" + "0" * 64
        elif field == "base_revision":
            data[field] = "0" * 40
        else:
            data[field] += "changed"
        changed = domain.MaintenanceAcceptedRequest.model_validate(data)
        with pytest.raises(ValueError, match="maintenance request"):
            async for _ in workflow(app, task, accepted=changed).execute("root"):
                pass
        assert not provider.requests
        assert await app.app.session_store.load(task.session_id) is None
        listed = await app.artifact_store.list(session_id=task.session_id)
        assert not listed.artifacts and listed.total_count == 0

    asyncio.run(scenario())


def test_changed_budget_and_default_only_settlement_reject(consumer):
    app, task, provider, domain, requests, workflow = consumer

    async def scenario():
        accepted = await requests.capture_accepted_request(app, task)
        app.app.budget_policy = denial_policy()
        with pytest.raises(ValueError, match="maintenance request"):
            async for _ in workflow(app, task, accepted=accepted).execute("root"):
                pass
        assert not provider.requests
        for changed in (
            replace(task, settlement=domain.default_maintenance_settlement()),
            replace(task, review_settlement=object()),
        ):
            with pytest.raises(ValueError, match="maintenance request"):
                await requests.capture_accepted_request(app, changed)

    asyncio.run(scenario())


def test_corrupt_request_rejected_without_serializing_canaries(consumer, caplog, capsys):
    app, task, _, domain, requests, _ = consumer
    canary = "request-secret-canary"

    class Hostile:
        def __repr__(self):
            return canary

        __str__ = __repr__

    async def scenario():
        accepted = await requests.capture_accepted_request(app, task)
        object.__setattr__(accepted, "instruction", canary)
        object.__setattr__(accepted, "artifact_store_id", Hostile())
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with pytest.raises(ValueError) as error:
                domain.encode_request(accepted)
            assert canary not in str(error.value) + repr(error.value)
        assert not captured

    asyncio.run(scenario())
    output = capsys.readouterr()
    assert canary not in caplog.text + output.out + output.err


def test_same_workspace_id_at_changed_repository_rejects(consumer, tmp_path):
    app, task, provider, _, requests, workflow = consumer

    async def scenario():
        accepted = await requests.capture_accepted_request(app, task)
        other = tmp_path / "other-source"
        materialize_seed_repository(other)
        app.source_workspace = LocalWorkspace(other, workspace_id=app.source_workspace.id)
        app.project_root = other.resolve()
        assert app.source_workspace.id == accepted.source_workspace_id
        with pytest.raises(ValueError, match="maintenance request"):
            async for _ in workflow(app, task, accepted=accepted).execute("root"):
                pass
        assert not provider.requests
        assert await app.app.session_store.load(task.session_id) is None

    asyncio.run(scenario())


def test_real_cancellation_during_inspection_propagates(consumer, monkeypatch):
    app, task, _, _, requests, _ = consumer

    async def scenario():
        entered = asyncio.Event()

        async def inspect(_task):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(app, "inspect_execution_profile", inspect)
        owner = asyncio.create_task(requests.capture_accepted_request(app, task))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            owner.cancel("request stopped")
            with pytest.raises(asyncio.CancelledError, match="request stopped"):
                await owner
            assert owner.cancelled() and owner.cancelling() == 1
        finally:
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())
