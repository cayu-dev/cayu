"""Live executable admission evidence for the Lambda MicroVM runner.

The real sidecar supervisor runs every probe locally through the agent lane;
only the guest boot-id read is answered by the harness, because Linux
``/proc`` is not available on every test host.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.runners.lambda_microvm_harness import (
    DEFAULT_GUEST_BOOT_ID,
    ClientTokenLambdaModel,
    SupervisorTransport,
    is_boot_id_read,
)

import cayu.runners.aws_lambda_microvm as lambda_microvm_module
from cayu import LambdaMicroVMRunner
from cayu.environments.admission import (
    EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS,
    ExecutionRequirements,
    ExecutionToolRequirement,
    evaluate_execution_admission,
)
from cayu.runners import LambdaMicroVMError, LambdaMicroVMOwnershipSuperseded
from cayu.runners.base import ExecCommand
from cayu.tools.base import ToolExecutableRequirement, ToolExecutionRequirement

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX guest probe semantics")


def _requirements(*alternatives: ToolExecutableRequirement) -> ExecutionRequirements:
    return ExecutionRequirements(
        tool_requirements=tuple(
            ExecutionToolRequirement(
                tool_name=f"tool-{index}",
                requirement=ToolExecutionRequirement(name="program", alternatives=(alternative,)),
            )
            for index, alternative in enumerate(alternatives)
        )
    )


def _guest_path(tmp_path: Path) -> str:
    bin_dir = tmp_path / "guest-bin"
    bin_dir.mkdir(exist_ok=True)
    tool = bin_dir / "fake-tool"
    tool.write_text('#!/bin/sh\n[ "$1" = --version ] && exit 3\nexit 0\n')
    tool.chmod(0o755)
    return f"{bin_dir}:/usr/bin:/bin"


def _runner(
    tmp_path: Path,
    *,
    model: ClientTokenLambdaModel | None = None,
    transport: SupervisorTransport | None = None,
) -> tuple[LambdaMicroVMRunner, ClientTokenLambdaModel, SupervisorTransport]:
    model = model or ClientTokenLambdaModel()
    root = tmp_path / "workspace"
    transport = transport or SupervisorTransport(root)
    response = model.run_microvm(imageIdentifier=model.image_arn)
    runner = LambdaMicroVMRunner(
        model,
        microvm_id=response["microvmId"],
        endpoint=response["endpoint"],
        image_identifier=response["imageArn"],
        image_version=response["imageVersion"],
        default_cwd=str(transport.supervisor.root),
        endpoint_transport=transport,
        poll_interval_s=0,
        env_overlay={"PATH": _guest_path(tmp_path)},
    )
    return runner, model, transport


@pytest.mark.anyio
async def test_present_executable_is_live_verified_and_missing_is_refused(tmp_path: Path) -> None:
    runner, _model, transport = _runner(tmp_path)
    requirements = _requirements(
        ToolExecutableRequirement(executable="fake-tool"),
        ToolExecutableRequirement(executable="absent-tool"),
    )
    observer = runner.execution_admission_observer(requirements)

    declared = runner.execution_admission_candidate_for(requirements)
    assert declared.candidate == "lambda-microvm"
    assert {claim.state for claim in declared.evidence.tool_requirements.executables} == {
        "declared"
    }
    before = datetime.now(UTC)
    candidate = await observer.collect()

    claims = {claim.executable: claim for claim in candidate.evidence.tool_requirements.executables}
    assert claims["fake-tool"].state == "live_verified"
    assert claims["fake-tool"].observed_at >= before
    assert claims["fake-tool"].valid_until - claims["fake-tool"].observed_at == timedelta(
        seconds=EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS
    )
    assert claims["absent-tool"].state == "unavailable"
    assert claims["absent-tool"].reason_code == "executable_unavailable"
    assert observer.snapshot() == candidate
    # Every probe and identity read used the unprivileged agent lane.
    assert transport.execution_profiles
    assert set(transport.execution_profiles) == {"agent"}
    probes = [payload["argv"] for payload in transport.payloads if not is_boot_id_read(payload)]
    assert [argv[-1] for argv in probes] == ["absent-tool", "fake-tool"]
    assert all(argv[:2] == ["/bin/sh", "-c"] for argv in probes)
    assert sum(is_boot_id_read(payload) for payload in transport.payloads) == 2

    decision = evaluate_execution_admission(
        candidate="lambda-microvm",
        requirements=requirements,
        evidence=candidate.evidence,
        stage="pre_exposure",
    )
    assert decision.status == "refused"
    assert {refusal.executable for refusal in decision.refusals} == {"absent-tool"}

    admitted_requirements = _requirements(ToolExecutableRequirement(executable="fake-tool"))
    admitted = await runner.execution_admission_observer(admitted_requirements).collect()
    assert (
        evaluate_execution_admission(
            candidate="lambda-microvm",
            requirements=admitted_requirements,
            evidence=admitted.evidence,
            stage="pre_exposure",
        ).status
        == "admitted"
    )
    await runner.close()


@pytest.mark.anyio
@pytest.mark.parametrize(("accepted", "live"), [((3,), True), ((0,), False)])
async def test_probe_arguments_invoke_the_executable_with_accepted_exit_codes(
    tmp_path: Path, accepted: tuple[int, ...], live: bool
) -> None:
    runner, _model, transport = _runner(tmp_path)
    requirement = ToolExecutableRequirement(
        executable="fake-tool", probe_arguments=("--version",), accepted_exit_codes=accepted
    )
    candidate = await runner.execution_admission_observer(_requirements(requirement)).collect()

    (claim,) = candidate.evidence.tool_requirements.executables
    assert claim.state == ("live_verified" if live else "unavailable")
    assert claim.requirement_fingerprint == requirement.fingerprint
    probes = [payload for payload in transport.payloads if not is_boot_id_read(payload)]
    assert [payload["argv"] for payload in probes] == [["fake-tool", "--version"]]
    assert probes[0]["execution_profile"] == "agent"
    await runner.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("image_version", "identity changed during admission"),
        ("boot_id", "identity changed during admission"),
        ("state", "is SUSPENDING"),
        ("endpoint", "different MicroVM identity"),
    ],
)
async def test_identity_drift_between_reads_fails_closed(
    tmp_path: Path, change: str, message: str
) -> None:
    boot_ids = iter([DEFAULT_GUEST_BOOT_ID, "ffffffff-4f50-4617-8293-a4b5c6d7e8f9"])
    root = tmp_path / "workspace"
    transport = SupervisorTransport(
        root, boot_id=(lambda: next(boot_ids)) if change == "boot_id" else DEFAULT_GUEST_BOOT_ID
    )
    runner, model, _ = _runner(tmp_path, transport=transport)
    original_get = model.get_microvm
    reads = 0

    def drifting_get(**kwargs):
        nonlocal reads
        reads += 1
        response = original_get(**kwargs)
        if reads > 1:
            if change == "image_version":
                response["imageVersion"] = "4"
            elif change == "state":
                response["state"] = "SUSPENDING"
            elif change == "endpoint":
                response["endpoint"] = "replacement.lambda-microvm.invalid"
        return response

    model.get_microvm = drifting_get
    runner.image_version = None  # The runner must not rely on its own record alone.
    observer = runner.execution_admission_observer(
        _requirements(ToolExecutableRequirement(executable="fake-tool"))
    )

    with pytest.raises(LambdaMicroVMError, match=message):
        await observer.collect()
    assert reads == 2

    # No evidence survives a failed observation.
    assert observer.snapshot().evidence.tool_requirements.executables[0].state == "declared"
    await runner.close()


@pytest.mark.anyio
async def test_runner_bound_image_mismatch_is_refused_before_probing(tmp_path: Path) -> None:
    runner, model, transport = _runner(tmp_path)
    model.microvms[runner.microvm_id]["imageVersion"] = "9"

    with pytest.raises(LambdaMicroVMError, match="different image"):
        await runner.execution_admission_observer(
            _requirements(ToolExecutableRequirement(executable="fake-tool"))
        ).collect()
    assert transport.payloads == []
    await runner.close()


@pytest.mark.anyio
async def test_superseded_owner_fails_admission_closed(tmp_path: Path) -> None:
    runner, model, transport = _runner(tmp_path)
    requirements = _requirements(ToolExecutableRequirement(executable="fake-tool"))
    observer = runner.execution_admission_observer(requirements)
    assert (await observer.collect()).evidence.tool_requirements.executables[0].state == (
        "live_verified"
    )

    successor = await LambdaMicroVMRunner.from_existing(
        runner.microvm_id,
        client=model,
        endpoint_transport=transport,
        default_cwd=runner.default_cwd,
        poll_interval_s=0,
        env_overlay=runner.env_overlay,
    )

    with pytest.raises(LambdaMicroVMOwnershipSuperseded):
        await observer.refresh()
    with pytest.raises(LambdaMicroVMOwnershipSuperseded):
        observer.snapshot()
    with pytest.raises(LambdaMicroVMOwnershipSuperseded):
        await runner.refresh_execution_admission()
    successor_candidate = await successor.execution_admission_observer(requirements).collect()
    assert successor_candidate.evidence.tool_requirements.executables[0].state == "live_verified"
    await successor.close()


@pytest.mark.anyio
async def test_suspended_or_closed_runner_refuses_collect_snapshot_and_refresh(
    tmp_path: Path,
) -> None:
    runner, _model, transport = _runner(tmp_path)
    requirements = _requirements(ToolExecutableRequirement(executable="fake-tool"))
    observer = runner.execution_admission_observer(requirements)
    await observer.collect()

    await runner.suspend()
    dispatched = len(transport.payloads)
    with pytest.raises(RuntimeError, match="suspended"):
        observer.snapshot()
    with pytest.raises(RuntimeError, match="suspended"):
        await observer.collect()
    with pytest.raises(RuntimeError, match="suspended"):
        await runner.refresh_execution_admission()
    assert len(transport.payloads) == dispatched

    await runner.resume()
    # Evidence from before the lifecycle transition is not reused.
    assert observer.snapshot().evidence.tool_requirements.executables[0].state == "declared"
    await observer.refresh()
    assert observer.snapshot().evidence.tool_requirements.executables[0].state == "live_verified"

    await runner.close()
    with pytest.raises(RuntimeError, match="closed"):
        await observer.collect()
    with pytest.raises(RuntimeError, match="closed"):
        observer.snapshot()


@pytest.mark.anyio
async def test_refresh_renews_evidence_with_a_new_observation_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _model, _transport = _runner(tmp_path)
    observer = runner.execution_admission_observer(
        _requirements(ToolExecutableRequirement(executable="fake-tool"))
    )
    first = (await observer.collect()).evidence.tool_requirements.executables[0]
    later = first.observed_at + timedelta(seconds=EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS + 1)

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return later

    monkeypatch.setattr(lambda_microvm_module, "datetime", _Clock)
    assert (
        evaluate_execution_admission(
            candidate="lambda-microvm",
            requirements=observer.requirements,
            evidence=observer.snapshot().evidence,
            now=later,
        ).status
        == "refused"
    )
    await observer.refresh()
    renewed = observer.snapshot().evidence.tool_requirements.executables[0]
    assert renewed.observed_at == later
    assert renewed.valid_until == later + timedelta(seconds=EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS)
    assert (
        evaluate_execution_admission(
            candidate="lambda-microvm",
            requirements=observer.requirements,
            evidence=observer.snapshot().evidence,
            now=later,
        ).status
        == "admitted"
    )
    await runner.close()


@pytest.mark.anyio
async def test_cancelled_probe_is_settled_by_runner_command_cleanup(tmp_path: Path) -> None:
    runner, _model, transport = _runner(tmp_path)
    blocked = tmp_path / "guest-bin" / "slow-tool"
    blocked.write_text("#!/bin/sh\nsleep 30\n")
    blocked.chmod(0o755)
    observer = runner.execution_admission_observer(
        _requirements(ToolExecutableRequirement(executable="slow-tool", probe_arguments=()))
    )
    task = asyncio.create_task(observer.collect())
    for _ in range(1000):
        if any(record.state == "running" for record in transport.supervisor._records.values()):
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("The probe never started in the guest.")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (probe,) = [payload for payload in transport.payloads if payload["argv"] == ["slow-tool"]]
    records = transport.supervisor._records
    assert {record.state for record in records.values()} <= {"completed", "cancelled"}
    assert any(record.state == "cancelled" for record in records.values())
    assert probe["execution_profile"] == "agent"
    # The runner remains usable only because cleanup positively settled.
    assert runner.lifecycle_state == "reusable"
    result = await runner.exec(ExecCommand.process("/bin/sh", "-c", "exit 0"))
    assert result.exit_code == 0
    await runner.close()


def test_requirements_without_executables_keep_the_no_claim_default(tmp_path: Path) -> None:
    runner, _model, transport = _runner(tmp_path)
    requirements = ExecutionRequirements()
    observer = runner.execution_admission_observer(requirements)
    assert type(observer).__name__ == "RunnerExecutionAdmissionObserver"
    assert runner.execution_admission_candidate_for(requirements) is None
    assert transport.payloads == []
