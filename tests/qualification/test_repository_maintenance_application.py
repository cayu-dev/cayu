"""Generated-consumer integration without claiming a live image or database."""

import ast
import asyncio
import importlib
import importlib.util
import json
import os
import subprocess
import tomllib
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256

import httpx
import pytest

from cayu import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CodingProductArtifactRepository,
    CodingProductState,
    CodingSettlementPolicy,
    DockerImageIdentity,
    EventQuery,
    EventType,
    ExecResult,
    InMemoryKnowledgeStore,
    InMemorySessionStore,
    InMemoryTaskStore,
    LocalArtifactStore,
    ModelPrice,
    ModelStreamEvent,
    PriceBook,
    ScriptedModelProvider,
)
from cayu.cli.project import project_context
from cayu.storage import SQLiteBudgetLedger
from tests.cli.test_scaffold_coding_budget import denial_policy
from tests.core.test_queued_session_messages import RecordingOneShotProvider
from tests.qualification.repository_maintenance_application import maintenance_project_files
from tests.qualification.repository_maintenance_case import (
    EXPECTED_RESPONSES,
    SEED_FILES,
    materialize_seed_repository,
)
from tests.qualification.repository_maintenance_delivery_case import (
    build_journey_http,
    exercise_local_delivery,
    local_git,
)
from tests.qualification.repository_maintenance_eval_case import (
    exercise_coding_eval,
    exercise_repeated_coding_eval,
)
from tests.qualification.repository_maintenance_probe import probe_program
from tests.qualification.repository_maintenance_toolchain import maintenance_toolchain
from tests.qualification.repository_maintenance_worker_case import exercise_coding_worker


def scripted_journey_budget():
    """Synthetic fixture pricing; never presented as paid-provider billing evidence."""
    return BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("1"),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="scripted",
                            model="maintenance-fixture",
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                    )
                ),
                reservation=BudgetReservation(max_input_tokens=65536, max_output_tokens=4096),
            ),
        )
    )


@pytest.fixture
def project(tmp_path, request):
    root = tmp_path / "maintenance-app"
    root.mkdir()
    backend = getattr(request.node, "callspec", None)
    database = (
        "sqlite"
        if backend is not None and backend.params.get("backend") == "sqlite"
        else "postgres"
    )
    for relative, content in maintenance_project_files(database=database).items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    # Snapshot-only fixture. This is not a resolved dependency lock or a build.
    (root / "uv.lock").write_text("version = 1\n")
    return root


def test_consumer_uses_public_imports_and_canonical_layout(project):
    configuration = tomllib.loads((project / "pyproject.toml").read_text())
    assert configuration["tool"]["cayu"]["scaffold"]["preset"] == "coding"
    for path in project.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith("tests.qualification")
                assert not module.startswith("cayu.runtime._")
    with project_context(project):
        domain = importlib.import_module("domain.maintenance_case")
        assert not hasattr(domain, "materialize_seed_repository")
        importlib.import_module("workflows.coding_product")
        importlib.import_module("app")


@pytest.mark.parametrize("database", ["postgres", "sqlite"])
def test_consumer_declares_server_runtime_dependency_without_service_preset(database):
    configuration = tomllib.loads(maintenance_project_files(database=database)["pyproject.toml"])
    assert configuration["tool"]["cayu"]["scaffold"]["preset"] == "coding"
    dependencies = configuration["project"]["dependencies"]
    expected = "cayu[postgres,server]==" if database == "postgres" else "cayu[server]=="
    assert len(dependencies) == 1 and dependencies[0].startswith(expected)


def test_image_builder_captures_probe_and_rejects_post_capture_changes(project):
    with project_context(project):
        builder = importlib.import_module("build_coding_image")
        snapshot = builder._trusted_build_context_snapshot()
        assert snapshot["environments/range_probe.py"] == probe_program()
        fingerprint = builder._trusted_build_context_sha256(snapshot)
        (project / "environments/range_probe.py").write_text("print('changed')\n")
        assert (
            builder._trusted_build_context_sha256(builder._trusted_build_context_snapshot())
            != fingerprint
        )
        with pytest.raises(RuntimeError, match="changed"):
            builder._require_project_inputs_unchanged(snapshot)
    assert (
        "COPY environments/range_probe.py /opt/cayu-acceptance/range_probe.py"
        in (project / "Dockerfile.coding").read_text()
    )


def test_generated_image_readback_constructs_target_specific_profile(project):
    with project_context(project):
        builder = importlib.import_module("build_coding_image")
        operations = importlib.import_module("operations.coding")
        snapshot = builder._trusted_build_context_snapshot()
        metadata = json.loads((project / "docker-coding-image.json").read_text())
        # Synthetic image metadata tests parsing only, not Docker admission.
        metadata.update(
            content_digest="sha256:" + "a" * 64,
            platform_architecture="amd64",
            dependency_inputs=builder._trusted_build_context_inputs(snapshot),
            trusted_build_context_sha256=builder._trusted_build_context_sha256(snapshot),
        )
        (project / "docker-coding-image.json").write_text(json.dumps(metadata))
        profile = operations._read_docker_toolchain_profile()
        assert profile.profile_id == metadata["profile_id"]
        assert profile.revision == metadata["profile_revision"]
        assert (
            tuple(check.name for check in operations._named_checks(profile))
            == operations._CHECK_NAMES
        )
        assert operations._COMMAND_SELECTOR_NAMES == ("python-version",)
        assert tuple(item.path for item in profile.dependency_inputs) == ("pyproject.toml",)
        assert {check.name for check in operations._named_checks(profile)} == {
            "format",
            "independent-range-probe",
            "lint",
            "test",
        }
        assert tuple(item.selector for item in profile.structured_command_authorities) == (
            "python-version",
        )
        assert profile.admission_probes[0].probe_id == "independent-probe-content"
        metadata["dependency_inputs"] = [
            item
            for item in metadata["dependency_inputs"]
            if item["path"] != "environments/range_probe.py"
        ]
        (project / "docker-coding-image.json").write_text(json.dumps(metadata))
        with pytest.raises(RuntimeError, match="toolchain identity"):
            operations._read_docker_toolchain_profile()


@pytest.mark.parametrize("wrong_base", [False, True])
@pytest.mark.parametrize("linked", [False, True])
def test_public_factory_constructs_and_prepares_the_bounded_workflow(
    project, tmp_path, monkeypatch, wrong_base, linked
):
    source = tmp_path / "target"
    base = materialize_seed_repository(source)
    if wrong_base:
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "-c",
                f"core.hooksPath={os.devnull}",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--allow-empty",
                "-m",
                "different base",
            ],
            check=True,
            capture_output=True,
            timeout=15,
        )
    profile = maintenance_toolchain(
        image_identity=DockerImageIdentity(reference="example.invalid/coding@sha256:" + "a" * 64),
        architecture="amd64",
        build_context_sha256="sha256:" + "b" * 64,
    )
    with project_context(project):
        operations = importlib.import_module("operations.coding")
        app_module = importlib.import_module("app")
        domain = importlib.import_module("domain.coding_product")
        # Only replace unavailable Docker inspection. No claim of admitted image
        # or execution; real factory construction and Runtime preparation remain.
        monkeypatch.setattr(
            operations, "_configured_docker_authority", lambda root: (profile, "/usr/bin/docker")
        )
        provider = RecordingOneShotProvider()
        application = app_module.build_coding_product_application(
            provider=provider,
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
            knowledge_store=InMemoryKnowledgeStore(access_scope=operations._knowledge_scope()),
            artifact_store=LocalArtifactStore(
                tmp_path / "artifacts", store_id="preparation-artifacts"
            ),
            workspace_root=source,
        )
        task = domain.CodingProductTask(
            product_run_id="bounded-product",
            session_id="bounded-session",
            task_id="bounded-task",
            instruction="Fix the inclusive upper endpoint and run all required checks.",
        )

        async def scenario(task=task):
            if linked:
                task = replace(
                    task,
                    parent_session_id="maintenance-root",
                    causal_budget_id="maintenance-budget",
                )
            inspected = await application.inspect_execution_profile(task)
            assert await application.inspect_execution_profile(task) == inspected
            allocated = replace(
                task,
                product_run_id="allocated-product",
                session_id="allocated-session",
                task_id="allocated-task",
                parent_session_id="allocated-root" if linked else None,
                causal_budget_id="allocated-root" if linked else None,
            )
            # Intake inspects configuration before reservation allocates IDs.
            # This selected composition must not use those IDs as policy identity.
            assert await application.inspect_execution_profile(allocated) == inspected
            assert await application.app.session_store.load(task.session_id) is None
            listed = await application.artifact_store.list(session_id=task.session_id)
            assert listed.total_count == 0 and not listed.artifacts and not listed.truncated
            with pytest.raises(FileNotFoundError):
                await CodingProductArtifactRepository(application.artifact_store).load_request(
                    task.product_run_id, session_id=task.session_id
                )
            assert not provider.requests
            application.app.budget_policy = denial_policy()
            try:
                budget_profile = await application.inspect_execution_profile(task)
                assert budget_profile != inspected
                assert await application.inspect_execution_profile(allocated) == budget_profile
            finally:
                application.app.budget_policy = None
            assert await application.inspect_execution_profile(task) == inspected
            if linked:
                from tests.evals.test_workflow_eval_target import _NoChildWorkflow

                assert await application.app.session_store.load("maintenance-root") is None
                async for _ in _NoChildWorkflow(application.app).run("maintenance-root"):
                    pass
                assert await application.inspect_execution_profile(task) == inspected
            weakened = replace(task, settlement=CodingSettlementPolicy(required_checks=("test",)))
            with pytest.raises(ValueError, match="fixed check set"):
                await application.run(weakened)
            with pytest.raises(FileNotFoundError):
                await CodingProductArtifactRepository(application.artifact_store).load_request(
                    task.product_run_id,
                    session_id=task.session_id,
                )
            if wrong_base:
                with pytest.raises(ValueError, match="fixed Git base"):
                    await application.run(task)
                with pytest.raises(FileNotFoundError):
                    await CodingProductArtifactRepository(application.artifact_store).load_request(
                        task.product_run_id,
                        session_id=task.session_id,
                    )
                assert not provider.requests
                return
            _, request, run = await application._prepare(task, require_existing=False)
            assert request.runtime.execution_profile_fingerprint == inspected
            assert request.parent_session_id == run.parent_session_id == task.parent_session_id
            assert request.causal_budget_id == run.causal_budget_id == task.causal_budget_id
            assert request.source.git_baseline.head_revision == base
            assert request.settlement.required_checks == (
                "format",
                "independent-range-probe",
                "lint",
                "test",
            )
            assert run.max_steps == 8
            assert run.limits.max_elapsed_seconds == 180
            assert run.limits.scope == "session"
            _, replay, replay_run = await application._prepare(task, require_existing=True)
            assert replay == request
            assert replay_run == run
            assert not provider.requests

        asyncio.run(scenario())


@pytest.mark.parametrize(
    (
        "backend",
        "incorrect_probe",
        "lose_push_ack",
        "lose_github_ack",
        "managed_worker",
        "workflow_eval",
        "restart_approval",
    ),
    [
        ("memory", False, False, False, False, False, False),
        ("memory", True, False, False, False, False, False),
        ("sqlite", False, False, False, False, False, False),
        ("sqlite", True, False, False, False, False, False),
        pytest.param("sqlite", False, True, False, False, False, False, id="sqlite-lost-push-ack"),
        pytest.param(
            "sqlite", False, False, True, False, False, False, id="sqlite-lost-github-ack"
        ),
        pytest.param("memory", False, False, False, True, False, False, id="memory-managed-worker"),
        pytest.param("sqlite", False, False, False, True, False, False, id="sqlite-managed-worker"),
        pytest.param(
            "sqlite", False, True, False, True, False, False, id="sqlite-managed-lost-push-ack"
        ),
        pytest.param(
            "sqlite", False, False, True, True, False, False, id="sqlite-managed-lost-github-ack"
        ),
        pytest.param(
            "sqlite",
            False,
            False,
            False,
            True,
            False,
            True,
            id="sqlite-approval-process-restart",
            marks=(
                pytest.mark.process,
                pytest.mark.sigkill_recovery,
                pytest.mark.skipif(
                    os.name != "posix", reason="SIGKILL process groups require POSIX"
                ),
            ),
        ),
        pytest.param(
            "sqlite", True, False, False, True, False, False, id="sqlite-managed-worker-rejected"
        ),
        pytest.param("memory", False, False, False, False, True, False, id="memory-workflow-eval"),
        pytest.param("sqlite", False, False, False, False, True, False, id="sqlite-workflow-eval"),
        pytest.param(
            "sqlite",
            False,
            False,
            False,
            False,
            "repeated",
            False,
            id="sqlite-workflow-eval-repeated",
        ),
        pytest.param(
            "sqlite",
            False,
            False,
            False,
            False,
            "dirty-repeated",
            False,
            id="sqlite-workflow-eval-dirty-source",
        ),
        pytest.param(
            "memory", True, False, False, False, True, False, id="memory-workflow-eval-rejected"
        ),
        pytest.param(
            "sqlite", True, False, False, False, True, False, id="sqlite-workflow-eval-rejected"
        ),
    ],
)
def test_public_workflow_seals_verifies_and_replays(
    project,
    tmp_path,
    monkeypatch,
    incorrect_probe,
    backend,
    lose_push_ack,
    lose_github_ack,
    managed_worker,
    workflow_eval,
    restart_approval,
):
    """Local binding integration, not proof of Docker isolation or model quality."""
    if managed_worker:
        monkeypatch.setenv("CAYU_MODEL", "maintenance-fixture")
    source = tmp_path / "source"
    materialize_seed_repository(source)
    remote = tmp_path / "remote.git"
    if backend == "sqlite":
        local_git(tmp_path, "clone", "--quiet", "--bare", str(source), str(remote))
    target = tmp_path / "target"
    target.mkdir()
    original = SEED_FILES["range_ops.py"]
    repaired = original.replace("lower <= value < upper", "lower <= value <= upper")
    profile = maintenance_toolchain(
        image_identity=DockerImageIdentity(reference="example.invalid/coding@sha256:" + "a" * 64),
        architecture="amd64",
        build_context_sha256="sha256:" + "b" * 64,
    )
    calls = [
        ("run_check", {"check": "test"}),
        (
            "apply_patch",
            {
                "operations": [
                    {
                        "type": "update",
                        "path": "range_ops.py",
                        "expected_revision": "sha256:" + sha256(original.encode()).hexdigest(),
                        "edits": [
                            {
                                "old_text": "lower <= value < upper",
                                "new_text": "lower <= value <= upper",
                            }
                        ],
                    }
                ]
            },
        ),
        *(
            ("run_check", {"check": name})
            for name in ("format", "independent-range-probe", "lint", "test")
        ),
        ("git_changes", {"mode": "diff", "scope": "all"}),
    ]
    scripts = [
        [
            ModelStreamEvent.tool_call(id=f"step-{i}", name=name, arguments=args),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    **(
                        {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
                        if managed_worker
                        else {}
                    ),
                }
            ),
        ]
        for i, (name, args) in enumerate(calls)
    ] + [
        [
            ModelStreamEvent.text_delta("Repair complete."),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    **(
                        {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
                        if managed_worker
                        else {}
                    ),
                }
            ),
        ]
    ]
    provider = ScriptedModelProvider(scripts)
    with project_context(project):
        operations = importlib.import_module("operations.coding")
        app_module = importlib.import_module("app")
        domain = importlib.import_module("domain.coding_product")
        acceptance = importlib.import_module("domain.maintenance_acceptance")
        spec = importlib.util.spec_from_file_location(
            "maintenance_local_harness", project / "tests/test_coding_composition.py"
        )
        assert spec is not None and spec.loader is not None
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)
        monkeypatch.setattr(
            operations, "_configured_docker_authority", lambda root: (profile, "/usr/bin/docker")
        )
        original_exec = harness._LocalDockerRunner.exec
        observed_tests = []

        async def execute(runner, command, **kwargs):
            argv = tuple(command.argv or ())
            if argv and argv[0].startswith("/opt/cayu-project/.venv/bin/"):
                content = (runner.root / "range_ops.py").read_text()
                assert content in (original, repaired)
                if argv[0].endswith("pytest"):
                    passed = content == repaired
                    observed_tests.append(passed)
                    return ExecResult(
                        stdout="passed" if passed else "failed", exit_code=0 if passed else 1
                    )
                if argv[0].endswith("python"):
                    responses = list(EXPECTED_RESPONSES)
                    if incorrect_probe:
                        responses[0] = not responses[0]
                    return ExecResult(stdout=json.dumps(responses), exit_code=0)
                return ExecResult(stdout="check complete", exit_code=0)
            return await original_exec(runner, command, **kwargs)

        monkeypatch.setattr(harness._LocalDockerRunner, "exec", execute)
        runners = []
        harness._install_fake_docker_factory(monkeypatch, target=target, created_runners=runners)

        def build_application(*, workspace_root=source, model_provider=provider):
            if backend == "sqlite":
                # Default generated stores carry the scaffold's portable identities;
                # opaque injected stores intentionally make delegation process-local.
                app = app_module.build_coding_product_application(
                    provider=model_provider,
                    workspace_root=workspace_root,
                    budget_policy=scripted_journey_budget() if managed_worker else None,
                    budget_ledger=SQLiteBudgetLedger(tmp_path / "journey-budget.sqlite")
                    if managed_worker
                    else None,
                )
                return app, app.app.session_store, app.app.task_store
            session_store = InMemorySessionStore()
            task_store = InMemoryTaskStore()
            app = app_module.build_coding_product_application(
                provider=model_provider,
                session_store=session_store,
                task_store=task_store,
                knowledge_store=InMemoryKnowledgeStore(access_scope=operations._knowledge_scope()),
                artifact_store=LocalArtifactStore(
                    tmp_path / "artifacts", store_id="workflow-artifacts"
                ),
                workspace_root=workspace_root,
                budget_policy=scripted_journey_budget() if managed_worker else None,
            )
            return app, session_store, task_store

        application, session_store, task_store = build_application()
        task = domain.CodingProductTask(
            product_run_id="workflow-product",
            session_id="workflow-session",
            task_id="workflow-task",
            instruction="Repair the inclusive endpoint.",
        )

        async def scenario(task=task):
            if workflow_eval in {"repeated", "dirty-repeated"}:
                dirty_first = workflow_eval == "dirty-repeated"
                await exercise_repeated_coding_eval(
                    build_application,
                    task,
                    scripts,
                    directory=tmp_path,
                    expected_model_steps=len(calls) + 1,
                    dirty_first=dirty_first,
                )
                assert observed_tests == (
                    [False, True] if dirty_first else [False, True, False, True]
                )
                assert (source / "range_ops.py").read_text() == original
                assert not provider.requests
                assert runners and all(runner.closed for runner in runners)
                return
            if managed_worker:
                reservations = importlib.import_module("operations.maintenance_runs")
                intake = importlib.import_module("operations.maintenance_intake")
                reservation_path = tmp_path / "maintenance-reservations.sqlite3"
                registry = reservations.SQLiteMaintenanceRunStore(reservation_path)
                await registry.initialize()
                api = build_journey_http(
                    application,
                    registry,
                    tenant="qualification-tenant",
                    subject="qualification-operator",
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=api), base_url="http://fixture"
                ) as client:
                    body = {
                        "instruction": task.instruction,
                        "idempotency_key": "inclusive-endpoint",
                    }
                    response = await client.post(
                        "/runs", headers={"authorization": "Bearer fixture-product"}, json=body
                    )
                    assert response.status_code == 202, response.text
                    replay = await client.post(
                        "/runs", headers={"authorization": "Bearer fixture-product"}, json=body
                    )
                    assert replay.status_code == 202 and replay.json() == response.json()
                    assert set(response.json()) == {"id", "coding_task_status"}
                identity = await registry.load_owned(
                    tenant="qualification-tenant", public_id=response.json()["id"]
                )
                assert identity is not None
                task = replace(
                    task,
                    product_run_id=identity.product_run_id,
                    session_id=identity.session_id,
                    task_id=identity.task_id,
                    parent_session_id=identity.workflow_session_id,
                    causal_budget_id=identity.workflow_session_id,
                )
                queued = await intake.ensure_coding_task(
                    application.app,
                    reservations.SQLiteMaintenanceRunStore(reservation_path),
                    identity,
                )
                publication = await exercise_coding_worker(
                    application,
                    task,
                    queued,
                    monkeypatch,
                    deadline=identity.coding_deadline(),
                    reservations=reservations.SQLiteMaintenanceRunStore(reservation_path),
                    rejected=incorrect_probe,
                )
                terminal = await application.app.task_store.load_task(task.task_id)
                usage = await application.app.get_session_usage(
                    application.app.project_session_id_for_exposure(task.session_id)
                )
                assert usage.model_steps == len(calls) + 1
                assert usage.usage.input_tokens == usage.usage.output_tokens == len(calls) + 1
                reserved = await application.app.session_store.query_events(
                    EventQuery(session_id=task.session_id, event_type=EventType.BUDGET_RESERVED)
                )
                reservation_ids = {item.event.payload["reservation_id"] for item in reserved}
                assert len(reservation_ids) == len(calls) + 1
                for reservation_id in reservation_ids:
                    record = await application.app.budget_ledger.load_reservation(reservation_id)
                    assert record is not None and record.status == "reconciled"
                    assert record.actual_amount == Decimal("0.000002")
                assert (
                    await intake.ensure_coding_task(
                        application.app,
                        reservations.SQLiteMaintenanceRunStore(reservation_path),
                        identity,
                    )
                    == terminal
                )
            elif workflow_eval:
                task, publication = await exercise_coding_eval(
                    application,
                    task,
                    expected_model_steps=len(calls) + 1,
                    reservation_path=tmp_path / "eval-reservations.sqlite3",
                    monkeypatch=monkeypatch,
                    rejected=incorrect_probe,
                )
            else:
                publication = await application.run(task)
            assert publication.candidate.state is CodingProductState.PATCH_READY_FOR_DELIVERY, (
                publication.candidate.model_dump_json(indent=2)
            )
            assert not publication.candidate.external_delivery_performed
            assert observed_tests == [False, True]
            assert (source / "range_ops.py").read_text() == repaired
            count = len(provider.requests)
            assert await application.run(task) == publication
            assert len(provider.requests) == count
            if incorrect_probe:
                with pytest.raises(acceptance.MaintenanceAcceptanceRejected):
                    await application.verify(task, publication)
            else:
                verified = await application.verify(task, publication)
                assert verified.result_digest == publication.result_reference.digest
                assert await application.verify(task, publication) == verified
            assert await application.app.drain_environment_cleanups() is True
            assert runners and all(runner.closed for runner in runners)

            if backend == "sqlite":
                await exercise_local_delivery(
                    application,
                    task,
                    publication,
                    root=tmp_path,
                    remote=remote,
                    incorrect_probe=incorrect_probe,
                    lose_push_ack=lose_push_ack,
                    lose_github_ack=lose_github_ack,
                    reservations=registry if managed_worker else None,
                    identity=identity if managed_worker else None,
                    restart_approval=restart_approval,
                )
                await session_store.close()
                await task_store.close()
                await application.app.knowledge_store.close()
                reconstructed, reopened_sessions, reopened_tasks = build_application()
                try:
                    if managed_worker:
                        assert reconstructed.app.budget_ledger is not application.app.budget_ledger
                        for reservation_id in reservation_ids:
                            record = await reconstructed.app.budget_ledger.load_reservation(
                                reservation_id
                            )
                            assert record is not None and record.status == "reconciled"
                            assert record.actual_amount == Decimal("0.000002")
                    assert reopened_sessions is not session_store
                    runner_count = len(runners)
                    assert await reconstructed.run(task) == publication
                    assert len(provider.requests) == count
                    assert len(runners) == runner_count
                    if incorrect_probe:
                        with pytest.raises(acceptance.MaintenanceAcceptanceRejected):
                            await reconstructed.verify(task, publication)
                    else:
                        assert await reconstructed.verify(task, publication) == verified
                    assert await reconstructed.app.drain_environment_cleanups() is True
                finally:
                    await reopened_sessions.close()
                    await reopened_tasks.close()
                    await reconstructed.app.knowledge_store.close()
                    if isinstance(reconstructed.app.budget_ledger, SQLiteBudgetLedger):
                        await reconstructed.app.budget_ledger.close()

        async def owned_scenario():
            try:
                await scenario()
            finally:
                if backend == "sqlite":
                    await session_store.close()
                    await task_store.close()
                    await application.app.knowledge_store.close()
                    if isinstance(application.app.budget_ledger, SQLiteBudgetLedger):
                        await application.app.budget_ledger.close()

        asyncio.run(owned_scenario())
