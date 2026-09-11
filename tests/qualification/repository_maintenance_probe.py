"""Fixed probe declaration and parent-side decoding for the maintained case.

The program belongs in the admitted image, outside the writable workspace.
This module does not create runners, import candidate code, or grant delivery.
"""

from __future__ import annotations

import hashlib
import json

from cayu import ExecCommand, ExecutionProfileBehaviorIdentity, NamedCheck
from tests.qualification.repository_maintenance_case import (
    PROBES,
    BehavioralOutcome,
    check_behavioral_responses,
    corpus_fingerprint,
)

PROBE_CHECK_NAME = "independent-range-probe"
PROBE_PROGRAM_PATH = "/opt/cayu-acceptance/range_probe.py"
PROBE_PYTHON = "/opt/cayu-project/.venv/bin/python"
PROBE_SOURCE_PATH = "/workspace/range_ops.py"
PROBE_MAX_OUTPUT_BYTES = 32 * 1024
PROBE_TIMEOUT_SECONDS = 30
# Valid output is at least 526 bytes. This preview bound makes RunCheckTool
# retain its complete output artifact, instead of relying on the event preview.
PROBE_MODEL_PREVIEW_BYTES = 256


def probe_program() -> bytes:
    """Return the image-owned probe; expected answers are not part of this code."""

    return (
        "import importlib.util\n"
        "import json\n"
        "import sys\n"
        "\n"
        "spec = importlib.util.spec_from_file_location('range_candidate', sys.argv[1])\n"
        "if spec is None or spec.loader is None:\n"
        "    raise RuntimeError('candidate module unavailable')\n"
        "candidate = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(candidate)\n"
        f"probes = {PROBES!r}\n"
        "answers = [candidate.in_closed_range(*probe) for probe in probes]\n"
        "print(json.dumps(answers, separators=(',', ':'), allow_nan=False))\n"
    ).encode()


def probe_fingerprint() -> str:
    return "sha256:" + hashlib.sha256(probe_program()).hexdigest()


def probe_check() -> NamedCheck:
    """Declare one fixed command; registration must retain Docker admission."""

    return NamedCheck(
        name=PROBE_CHECK_NAME,
        description="Collect responses for independent closed-range acceptance.",
        command=ExecCommand.process(
            PROBE_PYTHON,
            "-I",
            "-B",
            PROBE_PROGRAM_PATH,
            PROBE_SOURCE_PATH,
        ),
        timeout_s=PROBE_TIMEOUT_SECONDS,
        max_output_bytes=PROBE_MAX_OUTPUT_BYTES,
        required_executables=(PROBE_PYTHON,),
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="qualification:closed-range-probe",
            behavior_version=corpus_fingerprint(),
            implementation_version=probe_fingerprint(),
        ),
    )


def evaluate_probe_stdout(stdout: object) -> BehavioralOutcome:
    """Reject empty, oversized, malformed or falsely typed success responses."""

    if type(stdout) is not bytes or len(stdout) > PROBE_MAX_OUTPUT_BYTES:
        return BehavioralOutcome.INVALID_RESPONSE
    try:
        responses = json.loads(stdout.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError):
        return BehavioralOutcome.INVALID_RESPONSE
    return check_behavioral_responses(responses)
