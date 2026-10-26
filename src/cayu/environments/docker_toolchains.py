"""Immutable admitted toolchain profiles for trusted-repository Docker coding.

``DockerCodingToolchainProfile`` is the Docker subclass of the backend-neutral
:class:`~cayu.environments.coding_toolchains.CodingToolchainProfile`: it adds
the exact image, the runtime user and the container restrictions. The
command-authority, dependency-input, probe and environment-variable classes are
the neutral ones under their original Docker names.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from cayu._validation import require_durable_clean_nonblank
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    ExecutionRequirements,
    evaluate_execution_admission,
)
from cayu.environments.coding_toolchains import (
    CODING_COMMAND_AUTHORITY_SCHEMA,
    MAX_CODING_DEPENDENCY_INPUT_BYTES,
    MAX_CODING_PROFILE_COMMANDS,
    MAX_CODING_PROFILE_DEPENDENCY_INPUTS,
    MAX_CODING_PROFILE_PROBES,
    CodingAdmissionProbe,
    CodingCommandAuthority,
    CodingDependencyInput,
    CodingFixedEnvironmentVariable,
    CodingToolchainError,
    CodingToolchainProfile,
    _AdmittedRunner,
    _fingerprint,
    _RenewableAdmissionRunner,
    verify_coding_toolchain_dependencies,
    verify_local_coding_toolchain_dependencies,
)
from cayu.environments.coding_toolchains import (
    MAX_STRUCTURED_COMMAND_ARG_BYTES as MAX_STRUCTURED_COMMAND_ARG_BYTES,
)
from cayu.environments.coding_toolchains import (
    MAX_STRUCTURED_COMMAND_ARGS as MAX_STRUCTURED_COMMAND_ARGS,
)
from cayu.environments.coding_toolchains import (
    MAX_STRUCTURED_COMMAND_TOTAL_ARG_BYTES as MAX_STRUCTURED_COMMAND_TOTAL_ARG_BYTES,
)
from cayu.runners.docker_workload import DockerImageIdentity, DockerWorkloadRestrictions

DOCKER_CODING_TOOLCHAIN_PROFILE_SCHEMA = "cayu.docker_coding_toolchain_profile.v1"
DOCKER_CODING_COMMAND_AUTHORITY_SCHEMA = CODING_COMMAND_AUTHORITY_SCHEMA
MAX_DOCKER_CODING_PROFILE_COMMANDS = MAX_CODING_PROFILE_COMMANDS
MAX_DOCKER_CODING_PROFILE_DEPENDENCY_INPUTS = MAX_CODING_PROFILE_DEPENDENCY_INPUTS
MAX_DOCKER_CODING_PROFILE_PROBES = MAX_CODING_PROFILE_PROBES
MAX_DOCKER_CODING_DEPENDENCY_INPUT_BYTES = MAX_CODING_DEPENDENCY_INPUT_BYTES

# The shared classes keep their Docker names; they are the same classes with the
# same schemas, so existing profiles and selector fingerprints are unchanged.
DockerCodingDependencyInput = CodingDependencyInput
DockerCodingFixedEnvironmentVariable = CodingFixedEnvironmentVariable
DockerCodingAdmissionProbe = CodingAdmissionProbe
DockerCodingCommandAuthority = CodingCommandAuthority

# Serialized key order of the Docker profile before the neutral base class
# existed. Pydantic lists inherited fields first, so the profile restores this
# order when it dumps itself; dumps stay byte-identical to earlier releases.
_DOCKER_PROFILE_FIELD_ORDER = (
    "schema_version",
    "profile_id",
    "revision",
    "image_identity",
    "platform_os",
    "platform_architecture",
    "runtime_user",
    "working_directory",
    "workspace_path",
    "read_only_support_paths",
    "restrictions",
    "command_authorities",
    "dependency_inputs",
    "trusted_build_context_sha256",
    "application_dependency_identity",
    "admission_probes",
    "prohibited_capabilities",
    "result_publication_max_bytes",
    "redaction_behavior_version",
    "adoption_behavior_version",
)


class DockerCodingToolchainError(CodingToolchainError):
    """A typed safe failure at a Docker toolchain profile boundary."""


class DockerCodingToolchainProfile(CodingToolchainProfile):
    """Versioned reconstructable authority for one exact Docker coding environment."""

    schema_version: Literal["cayu.docker_coding_toolchain_profile.v1"] = Field(
        default=DOCKER_CODING_TOOLCHAIN_PROFILE_SCHEMA,
        alias="schema",
    )
    image_identity: DockerImageIdentity
    platform_architecture: Literal["amd64", "arm64"]
    runtime_user: str = Field(default="1000:1000", max_length=64)
    restrictions: DockerWorkloadRestrictions = Field(default_factory=DockerWorkloadRestrictions)

    @field_validator("runtime_user")
    @classmethod
    def validate_runtime_user(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "runtime_user")
        parts = value.split(":")
        if len(parts) != 2 or any(not part.isascii() or not part.isdigit() for part in parts):
            raise ValueError("runtime_user must be a numeric uid:gid pair.")
        if any(int(part) <= 0 or int(part) > 2**31 - 1 for part in parts):
            raise ValueError("runtime_user uid and gid must be positive 32-bit integers.")
        return value

    @model_validator(mode="after")
    def validate_profile_relationships(self) -> DockerCodingToolchainProfile:
        if self.runtime_user != self.restrictions.user:
            raise ValueError("runtime_user must match the Docker restriction uid and gid.")
        self._validate_shared_relationships()
        return self

    @model_serializer(mode="wrap")
    def _serialize_in_profile_order(self, handler: SerializerFunctionWrapHandler) -> Any:
        dumped = handler(self)
        if type(dumped) is not dict:
            return dumped
        aliases = {"schema_version": "schema"}
        ordered: dict[str, Any] = {}
        for name in _DOCKER_PROFILE_FIELD_ORDER:
            for key in (name, aliases.get(name)):
                if key is not None and key in dumped:
                    ordered[key] = dumped[key]
        ordered.update((key, value) for key, value in dumped.items() if key not in ordered)
        return ordered

    @classmethod
    def from_toolchain(
        cls,
        profile: CodingToolchainProfile,
        *,
        image_identity: DockerImageIdentity,
        platform_architecture: Literal["amd64", "arm64"] | None = None,
        restrictions: DockerWorkloadRestrictions | None = None,
        extra_admission_probes: Sequence[CodingAdmissionProbe] = (),
    ) -> DockerCodingToolchainProfile:
        """Bind a neutral profile to one exact Docker image.

        ``platform_architecture`` is required unless the profile already names
        one, and must agree with it when both do. ``extra_admission_probes``
        add Docker-only checks, for example one that the image reports this
        architecture, without tying the neutral profile to a machine.
        """

        if type(profile) is not CodingToolchainProfile:
            raise TypeError("profile must be an exact CodingToolchainProfile.")
        if type(image_identity) is not DockerImageIdentity:
            raise TypeError("image_identity must be a DockerImageIdentity.")
        architecture = platform_architecture or profile.platform_architecture
        if architecture is None:
            raise ValueError(
                "A Docker toolchain profile needs platform_architecture; the neutral profile "
                "doesn't name one."
            )
        if (
            profile.platform_architecture is not None
            and platform_architecture is not None
            and profile.platform_architecture != platform_architecture
        ):
            raise ValueError(
                f"platform_architecture {platform_architecture!r} conflicts with the "
                f"profile's {profile.platform_architecture!r}."
            )
        owned_restrictions = restrictions or DockerWorkloadRestrictions()
        if type(owned_restrictions) is not DockerWorkloadRestrictions:
            raise TypeError("restrictions must be DockerWorkloadRestrictions or None.")
        probes = tuple(
            sorted(
                (*profile.admission_probes, *extra_admission_probes),
                key=lambda probe: probe.probe_id,
            )
        )
        shared = profile.model_dump(mode="python", by_alias=True)
        shared.pop("schema")
        return cls.model_validate(
            {
                **shared,
                "image_identity": image_identity,
                "platform_architecture": architecture,
                "runtime_user": owned_restrictions.user,
                "restrictions": owned_restrictions,
                "admission_probes": probes,
            }
        )

    def coding_toolchain(self) -> CodingToolchainProfile:
        """Return the backend-neutral profile: everything but the Docker image and container."""

        shared = self.model_dump(mode="python", by_alias=True)
        for name in ("schema", "image_identity", "runtime_user", "restrictions"):
            shared.pop(name)
        return CodingToolchainProfile.model_validate(shared)

    def evidence(self) -> dict[str, object]:
        """Return the bounded non-secret identity included in runtime receipts."""

        restrictions = self.restrictions.model_dump(mode="json")
        return {
            "toolchain_profile_id": self.profile_id,
            "toolchain_profile_revision": self.revision,
            "toolchain_profile_fingerprint": self.fingerprint,
            "toolchain_image_fingerprint": self.image_identity.fingerprint,
            "toolchain_image_content_digest": self.image_identity.content_digest,
            "toolchain_platform": f"{self.platform_os}/{self.platform_architecture}",
            "toolchain_runtime_user": self.runtime_user,
            "toolchain_home_environment": self.restrictions.home_environment,
            "toolchain_working_directory": self.working_directory,
            "toolchain_runtime_restrictions_identity": _fingerprint(
                restrictions,
                "toolchain_runtime_restrictions",
            ),
            "toolchain_resource_limits": {
                "pids_limit": self.restrictions.pids_limit,
                "memory_bytes": self.restrictions.memory_bytes,
                "memory_swap_bytes": self.restrictions.memory_swap_bytes,
                "cpu_period_us": self.restrictions.cpu_period_us,
                "cpu_quota_us": self.restrictions.cpu_quota_us,
                "shm_size_bytes": self.restrictions.shm_size_bytes,
            },
            "toolchain_prohibited_capabilities": list(self.prohibited_capabilities),
            "toolchain_dependency_identity": self.dependency_identity,
            "toolchain_result_publication_max_bytes": self.result_publication_max_bytes,
            "toolchain_redaction_behavior_version": self.redaction_behavior_version,
            "toolchain_adoption_behavior_version": self.adoption_behavior_version,
        }


async def ensure_docker_coding_toolchain_runner_admission(
    runner: object,
    *,
    profile: DockerCodingToolchainProfile,
) -> str | None:
    """Renew stale evidence, then independently validate the same authority."""

    failure = docker_coding_toolchain_runner_admission_failure(runner, profile=profile)
    if failure != "docker_admission_stale" or not isinstance(runner, _RenewableAdmissionRunner):
        return failure
    try:
        await runner.refresh_execution_admission()
    except Exception:
        # Admission probe diagnostics can contain deployment-private material.
        # Cancellation remains authoritative and is never swallowed here.
        return "docker_admission_unavailable"
    return docker_coding_toolchain_runner_admission_failure(runner, profile=profile)


def docker_coding_toolchain_runner_admission_failure(
    runner: object,
    *,
    profile: DockerCodingToolchainProfile,
) -> str | None:
    """Return a stable refusal code unless the active runner matches the profile."""

    if type(profile) is not DockerCodingToolchainProfile:
        raise TypeError("profile must be an exact DockerCodingToolchainProfile.")
    if not isinstance(runner, _AdmittedRunner):
        return "docker_admission_unavailable"
    try:
        candidate = runner.execution_admission_candidate()
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return "docker_admission_unavailable"
    if candidate is None:
        return "docker_admission_unavailable"
    if type(candidate) is not ExecutionAdmissionCandidate or candidate.candidate != "docker":
        return "docker_admission_mismatch"
    evidence = candidate.evidence
    if evidence.image_fingerprint != profile.image_identity.fingerprint:
        return "toolchain_image_drift"
    if evidence.toolchain_profile_fingerprint != profile.fingerprint:
        return "toolchain_profile_drift"
    network = evidence.claim_for("deny_by_default_network")
    if network is None or network.state != "live_verified" or network.observation != "denied":
        return "docker_network_admission_missing"
    requirements = evidence.tool_requirements
    if requirements is None:
        return "toolchain_executable_admission_missing"
    for executable in profile.required_executables:
        observed = requirements.executable_for(executable)
        if observed is None or observed.state != "live_verified":
            return "toolchain_executable_admission_missing"
    admission = evaluate_execution_admission(
        candidate=candidate.candidate,
        requirements=ExecutionRequirements.trusted(
            network_access="deny_by_default",
            minimum_evidence="live_verified",
            required_executables=profile.required_executables,
        ),
        evidence=evidence,
        stage="pre_exposure",
        now=datetime.now(UTC),
    )
    if admission.status != "admitted":
        if any(
            refusal.code in {"future_evidence", "overlong_evidence", "stale_evidence"}
            for refusal in admission.refusals
        ):
            return "docker_admission_stale"
        return "docker_admission_invalid"
    return None


async def verify_docker_coding_toolchain_dependencies(
    profile: DockerCodingToolchainProfile,
    workspace: Any,
) -> None:
    """Fail before dispatch when the active workspace no longer matches the profile."""

    if type(profile) is not DockerCodingToolchainProfile:
        raise TypeError("profile must be an exact DockerCodingToolchainProfile.")
    await verify_coding_toolchain_dependencies(profile, workspace)


def verify_local_docker_coding_toolchain_dependencies(
    profile: DockerCodingToolchainProfile,
    root: Path,
) -> None:
    """Verify source dependency inputs without allocating Docker resources."""

    if type(profile) is not DockerCodingToolchainProfile:
        raise TypeError("profile must be an exact DockerCodingToolchainProfile.")
    verify_local_coding_toolchain_dependencies(profile, root)


__all__ = [
    "DOCKER_CODING_COMMAND_AUTHORITY_SCHEMA",
    "DOCKER_CODING_TOOLCHAIN_PROFILE_SCHEMA",
    "DockerCodingAdmissionProbe",
    "DockerCodingCommandAuthority",
    "DockerCodingDependencyInput",
    "DockerCodingFixedEnvironmentVariable",
    "DockerCodingToolchainError",
    "DockerCodingToolchainProfile",
    "docker_coding_toolchain_runner_admission_failure",
    "verify_docker_coding_toolchain_dependencies",
    "verify_local_docker_coding_toolchain_dependencies",
]
