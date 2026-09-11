"""Probe transport tests using only literal, test-authored candidate fixtures."""

import asyncio
import json
import subprocess
import sys
from hashlib import sha256

import pytest

from cayu import ExecResult, LocalArtifactStore, RunCheckTool
from tests.core.test_named_checks import RecordingRunner, _policy, _run
from tests.qualification.repository_maintenance_case import (
    EXPECTED_RESPONSES,
    SEED_FILES,
    BehavioralOutcome,
)
from tests.qualification.repository_maintenance_probe import (
    PROBE_CHECK_NAME,
    PROBE_MAX_OUTPUT_BYTES,
    PROBE_MODEL_PREVIEW_BYTES,
    PROBE_PYTHON,
    evaluate_probe_stdout,
    probe_check,
    probe_fingerprint,
    probe_program,
)


@pytest.mark.parametrize("variant", ["seed", "fixed", "exit_zero", "false_success"])
def test_independent_parent_judges_responses_not_process_exit(tmp_path, variant):
    source = SEED_FILES["range_ops.py"]
    if variant == "fixed":
        source = source.replace("lower <= value < upper", "lower <= value <= upper")
    elif variant == "exit_zero":
        source = "raise SystemExit(0)\n"
    elif variant == "false_success":
        source = "print('all checks passed')\nraise SystemExit(0)\n"
    candidate = tmp_path / "range_ops.py"
    candidate.write_text(source, encoding="utf-8")
    program = tmp_path / "probe.py"
    program.write_bytes(probe_program())
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(program), str(candidate)],
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0
    expected = (
        BehavioralOutcome.PASSED
        if variant == "fixed"
        else BehavioralOutcome.INCORRECT
        if variant == "seed"
        else BehavioralOutcome.INVALID_RESPONSE
    )
    assert evaluate_probe_stdout(result.stdout) is expected


@pytest.mark.parametrize(
    "output",
    [
        b"",
        b"passed",
        b"true",
        b"[1]",
        b"\xff",
        b"[" * 5000,
        b" " * (PROBE_MAX_OUTPUT_BYTES + 1),
        "not bytes",
    ],
)
def test_probe_decoder_rejects_invalid_or_unbounded_output(output):
    assert evaluate_probe_stdout(output) is BehavioralOutcome.INVALID_RESPONSE


def test_probe_declaration_binds_program_and_retains_valid_output():
    check = probe_check()
    assert check.execution_profile_identity.implementation_version == probe_fingerprint()
    assert check.timeout_s == 30
    assert check.max_output_bytes == PROBE_MAX_OUTPUT_BYTES
    content = json.dumps(EXPECTED_RESPONSES, separators=(",", ":")).encode()
    assert PROBE_MODEL_PREVIEW_BYTES < len(content) <= PROBE_MAX_OUTPUT_BYTES
    assert evaluate_probe_stdout(content) is BehavioralOutcome.PASSED


def test_named_check_retains_complete_probe_output_for_parent_verification(tmp_path):
    check = probe_check()
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="probe-output")
    stdout = json.dumps(EXPECTED_RESPONSES, separators=(",", ":"))
    runner = RecordingRunner(ExecResult(stdout=stdout, exit_code=0))
    result = _run(
        RunCheckTool(
            checks=(check,),
            command_policy=_policy(allowed=(PROBE_PYTHON,)),
            max_model_output_bytes=PROBE_MODEL_PREVIEW_BYTES,
        ),
        runner,
        {"check": PROBE_CHECK_NAME},
        artifact_store=store,
    )
    assert result.structured["output_artifact_status"] == "stored"
    assert result.structured["stdout_projection_truncated"] is True
    assert result.structured["stdout_runner_truncated"] is False
    assert result.structured["check_profile_fingerprint"] == check.profile_fingerprint
    artifact = asyncio.run(store.read_bytes(result.artifacts[0]["id"]))
    assert result.structured["output_sha256"] == "sha256:" + sha256(artifact.content).hexdigest()
    record = json.loads(artifact.content)
    assert record["stdout"] == stdout
    assert record["check_profile_fingerprint"] == check.profile_fingerprint
    assert evaluate_probe_stdout(record["stdout"].encode()) is BehavioralOutcome.PASSED
    # This runner double proves output transport only, not Docker or revision binding.
