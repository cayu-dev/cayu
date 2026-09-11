"""Target dependency and immutable probe declaration contracts, without Docker."""

import asyncio
import subprocess
import sys
from hashlib import sha256
from typing import Literal

import pytest
from pydantic import ValidationError

from cayu import (
    DockerCodingToolchainError,
    DockerCodingToolchainProfile,
    DockerImageIdentity,
    LocalWorkspace,
    verify_docker_coding_toolchain_dependencies,
)
from tests.qualification.repository_maintenance_case import SEED_FILES
from tests.qualification.repository_maintenance_probe import PROBE_PROGRAM_PATH, probe_program
from tests.qualification.repository_maintenance_toolchain import (
    maintenance_checks,
    maintenance_toolchain,
)


def _profile(
    *,
    image_identity: DockerImageIdentity | None = None,
    architecture: Literal["amd64", "arm64"] = "amd64",
    build_context_sha256: str = "sha256:" + "b" * 64,
):
    return maintenance_toolchain(
        image_identity=image_identity
        or DockerImageIdentity(reference="example.invalid/coding@sha256:" + "a" * 64),
        architecture=architecture,
        build_context_sha256=build_context_sha256,
    )


def test_profile_roundtrip_preserves_closed_check_authority():
    profile = _profile()
    rebuilt = DockerCodingToolchainProfile.model_validate_json(profile.model_dump_json())
    assert rebuilt == profile
    assert rebuilt.fingerprint == profile.fingerprint
    assert tuple(item.path for item in profile.dependency_inputs) == ("pyproject.toml",)
    assert profile.read_only_support_paths == ("/opt/cayu-acceptance", "/opt/cayu-project")
    for check in maintenance_checks():
        authority = profile.command_authority(check.name)
        assert authority is not None
        argv = check.command.argv
        assert argv is not None
        assert authority.command_argv() == tuple(argv)
        assert authority.timeout_seconds == check.timeout_s
        assert authority.max_output_bytes == check.max_output_bytes
        assert authority.max_model_output_bytes == 256
        with pytest.raises(ValueError):
            authority.validate_model_arguments(("--override",))


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "image_identity": DockerImageIdentity(
                reference="example.invalid/coding@sha256:" + "c" * 64
            )
        },
        {"architecture": "arm64"},
        {"build_context_sha256": "sha256:" + "d" * 64},
    ],
)
def test_changed_build_authority_changes_profile(overrides):
    assert _profile(**overrides).fingerprint != _profile().fingerprint


@pytest.mark.parametrize("value", ["", None, True, "latest"])
def test_missing_build_identity_is_not_a_valid_fallback(value):
    with pytest.raises((TypeError, ValidationError)):
        _profile(build_context_sha256=value)


@pytest.mark.parametrize("dependency_state", ["original", "changed", "missing"])
def test_target_dependency_admission_does_not_pin_editable_source(tmp_path, dependency_state):
    for relative, content in SEED_FILES.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    # A legitimate candidate edit must not require replacing the toolchain.
    (tmp_path / "range_ops.py").write_text(
        SEED_FILES["range_ops.py"].replace("value < upper", "value <= upper")
    )
    dependency = tmp_path / "pyproject.toml"
    if dependency_state == "changed":
        dependency.write_text("[project]\nname = 'changed'\n")
    elif dependency_state == "missing":
        dependency.unlink()
    operation = verify_docker_coding_toolchain_dependencies(_profile(), LocalWorkspace(tmp_path))
    if dependency_state == "original":
        asyncio.run(operation)
    else:
        with pytest.raises(DockerCodingToolchainError) as error:
            asyncio.run(operation)
        assert error.value.code == (
            "dependency_inputs_changed"
            if dependency_state == "changed"
            else "dependency_inputs_unavailable"
        )


def test_image_admission_probe_measures_program_bytes(tmp_path):
    probe = _profile().admission_probes[0]
    program = tmp_path / "range_probe.py"
    program.write_bytes(probe_program())
    # Only adapt the guest path/interpreter to test the declared hashing program.
    # This is not evidence of Docker admission or image construction.
    command = probe.argv[-1].replace(repr(PROBE_PROGRAM_PATH), repr(str(program)))

    def output_digest():
        result = subprocess.run(
            [sys.executable, "-I", "-c", command],
            check=True,
            capture_output=True,
            timeout=10,
        )
        return "sha256:" + sha256(result.stdout).hexdigest()

    assert output_digest() == probe.stdout_sha256
    program.write_bytes(b"print('false success')\n")
    assert output_digest() != probe.stdout_sha256
