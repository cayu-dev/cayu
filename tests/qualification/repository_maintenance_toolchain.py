"""Application-specific declarations for the fixed repository-maintenance case.

Image construction and admission remain separate: a profile is not evidence that
an image exists or that its admission probes have passed.
"""

from hashlib import sha256
from typing import Literal

from cayu import (
    DockerCodingAdmissionProbe,
    DockerCodingCommandAuthority,
    DockerCodingDependencyInput,
    DockerCodingToolchainProfile,
    DockerImageIdentity,
    ExecCommand,
    ExecutionProfileBehaviorIdentity,
    NamedCheck,
)
from tests.qualification.repository_maintenance_case import SEED_FILES, corpus_fingerprint
from tests.qualification.repository_maintenance_probe import (
    PROBE_MODEL_PREVIEW_BYTES,
    PROBE_PROGRAM_PATH,
    PROBE_PYTHON,
    probe_check,
    probe_fingerprint,
)


def maintenance_checks() -> tuple[NamedCheck, ...]:
    """Return fixed independent acceptance and project-test declarations."""

    return (
        NamedCheck(
            name="format",
            description="Verify formatting of the target repository without edits.",
            command=ExecCommand.process(
                "/opt/cayu-project/.venv/bin/ruff", "format", "--check", "--no-cache", "."
            ),
            timeout_s=30,
            max_output_bytes=32 * 1024,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="qualification:closed-range-format",
                behavior_version=corpus_fingerprint(),
                implementation_version="1",
            ),
        ),
        probe_check(),
        NamedCheck(
            name="lint",
            description="Run target repository static checks.",
            command=ExecCommand.process(
                "/opt/cayu-project/.venv/bin/ruff", "check", "--no-cache", "."
            ),
            timeout_s=30,
            max_output_bytes=32 * 1024,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="qualification:closed-range-lint",
                behavior_version=corpus_fingerprint(),
                implementation_version="1",
            ),
        ),
        NamedCheck(
            name="test",
            description="Run the admitted repository's project tests.",
            command=ExecCommand.process(
                "/opt/cayu-project/.venv/bin/pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_range_ops.py",
            ),
            timeout_s=30,
            max_output_bytes=32 * 1024,
            required_executables=("/opt/cayu-project/.venv/bin/pytest",),
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="qualification:closed-range-project-test",
                behavior_version=corpus_fingerprint(),
                implementation_version="1",
            ),
        ),
    )


def maintenance_toolchain(
    *,
    image_identity: DockerImageIdentity,
    architecture: Literal["amd64", "arm64"],
    build_context_sha256: str,
) -> DockerCodingToolchainProfile:
    """Bind separate immutable image inputs and editable target dependencies."""

    if type(build_context_sha256) is not str:
        raise TypeError("An explicit build context digest is required.")
    authorities = []
    for check in maintenance_checks():
        argv = check.command.argv
        if not argv:
            raise ValueError("Maintenance checks require a process command.")
        authorities.append(
            DockerCodingCommandAuthority(
                selector=check.name,
                revision="1",
                description=check.description,
                exposure="named_check",
                executable=argv[0],
                fixed_arguments=tuple(argv[1:]),
                max_arguments=0,
                timeout_seconds=check.timeout_s,
                max_output_bytes=check.max_output_bytes,
                max_model_output_bytes=PROBE_MODEL_PREVIEW_BYTES,
            )
        )
    authorities.append(
        DockerCodingCommandAuthority(
            selector="python-version",
            revision="1",
            description="Report the admitted interpreter version.",
            exposure="structured_command",
            executable=PROBE_PYTHON,
            fixed_arguments=("--version",),
            max_arguments=0,
            timeout_seconds=10,
            max_output_bytes=4096,
            max_model_output_bytes=PROBE_MODEL_PREVIEW_BYTES,
        )
    )
    return DockerCodingToolchainProfile(
        profile_id="repository-maintenance-python",
        revision="1",
        image_identity=image_identity,
        platform_architecture=architecture,
        read_only_support_paths=("/opt/cayu-acceptance", "/opt/cayu-project"),
        trusted_build_context_sha256=build_context_sha256,
        application_dependency_identity=corpus_fingerprint(),
        dependency_inputs=(
            DockerCodingDependencyInput(
                path="pyproject.toml",
                content_sha256="sha256:"
                + sha256(SEED_FILES["pyproject.toml"].encode()).hexdigest(),
                max_bytes=16 * 1024,
            ),
        ),
        command_authorities=tuple(sorted(authorities, key=lambda item: item.selector)),
        admission_probes=(
            DockerCodingAdmissionProbe(
                probe_id="independent-probe-content",
                argv=(
                    PROBE_PYTHON,
                    "-I",
                    "-c",
                    "from hashlib import sha256; from pathlib import Path; "
                    f"print('sha256:' + sha256(Path({PROBE_PROGRAM_PATH!r})"
                    ".read_bytes()).hexdigest())",
                ),
                stdout_sha256="sha256:" + sha256((probe_fingerprint() + "\n").encode()).hexdigest(),
                timeout_seconds=10,
                max_output_bytes=4096,
            ),
        ),
    )
