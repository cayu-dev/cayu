"""Sealed artifact/readback integration; controlled events are not Docker proof."""

import asyncio
import importlib
import json
from collections.abc import Mapping
from typing import cast

import pytest

from cayu import (
    ArtifactStoreHandle,
    CodingProductArtifactRepository,
    CodingProductState,
    DockerImageIdentity,
    ExecResult,
    LocalArtifactStore,
    LocalWorkspace,
    RunCheckTool,
    RunnerHandle,
    ToolContext,
    WorkspaceRevisionObservationLimits,
    compile_coding_product_candidate,
)
from cayu.cli.project import project_context
from cayu.workspaces.revisions import observe_deterministic_workspace
from tests.core.test_coding_products import _check_event, _request, _terminal_events, _tool_event
from tests.core.test_named_checks import RecordingRunner, _policy
from tests.qualification.repository_maintenance_acceptance import (
    MaintenanceAcceptanceRejected,
    verify_maintenance_result,
)
from tests.qualification.repository_maintenance_application import maintenance_project_files
from tests.qualification.repository_maintenance_case import (
    EXPECTED_RESPONSES,
    SEED_FILES,
    materialize_seed_repository,
)
from tests.qualification.repository_maintenance_probe import (
    PROBE_CHECK_NAME,
    PROBE_PYTHON,
    probe_check,
)
from tests.qualification.repository_maintenance_toolchain import (
    maintenance_checks,
    maintenance_toolchain,
)


async def _fixture(
    tmp_path, *, correct_answers=True, correct_check_profile=True, oversized_final=False
):
    source = tmp_path / "source"
    head = materialize_seed_repository(source)
    workspace = LocalWorkspace(
        source, workspace_id="source-workspace", excluded_directory_names=(".git",)
    )
    initial = await observe_deterministic_workspace(
        workspace,
        observer="cayu-coding-product-source",
        limits=WorkspaceRevisionObservationLimits(),
    )
    assert initial.revision is not None
    request = _request(baseline=initial.revision)
    toolchain = maintenance_toolchain(
        image_identity=DockerImageIdentity(reference="example.invalid/coding@sha256:" + "a" * 64),
        architecture="amd64",
        build_context_sha256="sha256:" + "b" * 64,
    )
    request = request.model_copy(
        update={
            "source": request.source.model_copy(
                update={
                    "git_baseline": request.source.git_baseline.model_copy(
                        update={"head_revision": head}
                    ),
                }
            ),
            "runtime": request.runtime.model_copy(
                update={
                    "toolchain_profile_id": toolchain.profile_id,
                    "toolchain_profile_revision": toolchain.revision,
                    "toolchain_profile_fingerprint": toolchain.fingerprint,
                    "image_fingerprint": toolchain.image_identity.fingerprint,
                    "dependency_identity": toolchain.dependency_identity,
                }
            ),
            "settlement": request.settlement.model_copy(
                update={
                    "required_checks": tuple(check.name for check in maintenance_checks()),
                }
            ),
        }
    )
    (source / "range_ops.py").write_text(
        SEED_FILES["range_ops.py"].replace("value < upper", "value <= upper")
        + ("#" * (16 * 1024) if oversized_final else "")
    )
    final = await observe_deterministic_workspace(
        workspace, observer="cayu-coding-product-source", limits=request.source.observation_limits
    )
    assert final.revision is not None
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="acceptance-artifacts")
    repository = CodingProductArtifactRepository(store)
    responses = list(EXPECTED_RESPONSES)
    if not correct_answers:
        responses[0] = not responses[0]
    check_result = await RunCheckTool(
        checks=(probe_check(),),
        command_policy=_policy(allowed=(PROBE_PYTHON,)),
        max_model_output_bytes=256,
    ).run(
        ToolContext(
            session_id=request.session_id,
            agent_name=request.agent_name,
            environment_name="coding",
            idempotency_key="probe-output",
            runner=cast(
                "RunnerHandle",
                RecordingRunner(ExecResult(stdout=json.dumps(responses), exit_code=0)),
            ),
            artifact_store=cast("ArtifactStoreHandle", store),
            workspace=workspace,
            workspace_id=workspace.id,
        ),
        {"check": PROBE_CHECK_NAME},
    )
    assert isinstance(check_result.structured, Mapping)
    events = []
    for check in maintenance_checks():
        if check.name == PROBE_CHECK_NAME:
            event = _tool_event("run_check", dict(check_result.structured), event_id="probe-result")
        else:
            event = _check_event(check.name, workspace_revision=final.revision)
            event.payload["result"]["structured"]["check_profile_fingerprint"] = (
                check.profile_fingerprint if correct_check_profile else "sha256:" + "f" * 64
            )
        events.append(event)
    candidate = await compile_coding_product_candidate(
        request,
        (
            *events,
            *_terminal_events(request=request, workspace_revision=final.revision, changed=True),
        ),
        initial_observation=initial,
        final_observation=final,
        repository=repository,
    )
    assert candidate.state is CodingProductState.PATCH_READY_FOR_DELIVERY
    publication = await repository.publish_candidate(candidate)
    return {
        "request": request,
        "reference": publication.result_reference,
        "repository": repository,
        "toolchain": toolchain,
        "workspace": workspace,
    }, source


def test_sealed_result_is_verified_without_dispatch_and_replays(tmp_path):
    async def scenario():
        inputs, _ = await _fixture(tmp_path)
        first = await verify_maintenance_result(**inputs)
        assert first == await verify_maintenance_result(**inputs)
        assert first.result_digest == inputs["reference"].digest
        assert first.request_fingerprint == inputs["request"].fingerprint

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure",
    ["wrong_answer", "changed_source", "wrong_request", "oversized_source", "wrong_check_profile"],
)
def test_check_success_cannot_replace_independent_acceptance(tmp_path, failure):
    async def scenario():
        inputs, source = await _fixture(
            tmp_path,
            correct_answers=failure != "wrong_answer",
            correct_check_profile=failure != "wrong_check_profile",
            oversized_final=failure == "oversized_source",
        )
        if failure == "changed_source":
            (source / "range_ops.py").write_text(SEED_FILES["range_ops.py"])
        elif failure == "wrong_request":
            request = inputs["request"]
            inputs["request"] = request.model_copy(
                update={"task": request.task.model_copy(update={"task_id": "other"})}
            )
        with pytest.raises(MaintenanceAcceptanceRejected):
            await verify_maintenance_result(**inputs)

    asyncio.run(scenario())


def test_missing_probe_artifact_cannot_reuse_a_prior_acceptance(tmp_path):
    async def scenario():
        inputs, _ = await _fixture(tmp_path)
        await verify_maintenance_result(**inputs)
        candidate = await inputs["repository"].read_candidate(inputs["reference"])
        probe = next(check for check in candidate.checks if check.check == PROBE_CHECK_NAME)
        await inputs["repository"].store.delete(probe.output_artifact.artifact_id)
        with pytest.raises(FileNotFoundError):
            await verify_maintenance_result(**inputs)

    asyncio.run(scenario())


def test_cancellation_during_artifact_read_remains_cancellation(tmp_path, monkeypatch):
    async def scenario():
        inputs, _ = await _fixture(tmp_path)
        started = asyncio.Event()

        async def blocked_read(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        with monkeypatch.context() as faults:
            faults.setattr(inputs["repository"].store, "read_bytes", blocked_read)
            task = asyncio.create_task(verify_maintenance_result(**inputs))
            await started.wait()
            task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
        await verify_maintenance_result(**inputs)

    asyncio.run(scenario())


def test_emitted_domain_verifies_the_same_retained_result(tmp_path):
    async def scenario():
        inputs, _ = await _fixture(tmp_path)
        consumer = tmp_path / "consumer"
        consumer.mkdir()
        for relative, content in maintenance_project_files().items():
            target = consumer / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        with project_context(consumer):
            domain = importlib.import_module("domain.maintenance_acceptance")
            accepted = await domain.verify_maintenance_result(**inputs)
            assert accepted.result_digest == inputs["reference"].digest
            workflow = importlib.import_module("workflows.coding_product")
            assert callable(workflow.CodingProductApplication.verify)

    asyncio.run(scenario())
