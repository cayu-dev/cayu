"""The backend-neutral toolchain profile, and Docker's profile as its exact subclass."""

from __future__ import annotations

import asyncio
import json
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from cayu.environments import (
    CodingAdmissionProbe,
    CodingCommandAuthority,
    CodingDependencyInput,
    CodingToolchainError,
    CodingToolchainProfile,
    DockerCodingAdmissionProbe,
    DockerCodingCommandAuthority,
    DockerCodingDependencyInput,
    DockerCodingEnvironmentFactory,
    DockerCodingFixedEnvironmentVariable,
    DockerCodingToolchainError,
    DockerCodingToolchainProfile,
    owned_coding_toolchain_profile,
    verify_coding_toolchain_dependencies,
    verify_docker_coding_toolchain_dependencies,
    verify_local_coding_toolchain_dependencies,
)
from cayu.runners import DockerImageIdentity, DockerWorkloadRestrictions, ExecCommand
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.tools import (
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    NamedCheck,
    RunCheckTool,
    RunCommandTool,
    StructuredCommandToolPolicy,
)
from cayu.workspaces import LocalWorkspace

_IMAGE = DockerImageIdentity(reference="golden-coding:local", content_digest="sha256:" + "a" * 64)


def _golden_profile() -> DockerCodingToolchainProfile:
    """A Docker profile that sets every field; its identity is pinned below."""

    restrictions = DockerWorkloadRestrictions(uid=1001, gid=1001, pids_limit=256)
    return DockerCodingToolchainProfile(
        profile_id="golden-python",
        revision="7",
        image_identity=_IMAGE,
        platform_architecture="arm64",
        runtime_user="1001:1001",
        restrictions=restrictions,
        read_only_support_paths=("/opt/golden",),
        command_authorities=(
            DockerCodingCommandAuthority(
                selector="focused-test",
                revision="1",
                description="Run one focused test target.",
                exposure="structured_command",
                executable="/opt/golden/.venv/bin/pytest",
                fixed_arguments=("-q", "-p", "no:cacheprovider"),
                allowed_flags=("--maxfail", "-x"),
                flags_with_values=("--maxfail",),
                allow_positional_arguments=True,
                positional_arguments_are_paths=True,
                positional_path_prefixes=("tests",),
                positional_path_suffixes=(".py",),
                allow_pytest_node_ids=True,
                max_arguments=4,
                fixed_environment=(
                    DockerCodingFixedEnvironmentVariable(name="PYTHONPATH", value="/workspace/src"),
                ),
                timeout_seconds=120,
                max_output_bytes=100_000,
                allowed_exit_codes=(0, 1, 5),
            ),
            DockerCodingCommandAuthority(
                selector="format",
                revision="2",
                description="Rewrite formatting.",
                exposure="structured_command",
                executable="/opt/golden/.venv/bin/ruff",
                fixed_arguments=("format",),
                effect="workspace_mutating",
                mutation_path_prefixes=("src", "tests"),
                approval="required",
                approval_expires_in_seconds=600,
            ),
            DockerCodingCommandAuthority(
                selector="test",
                revision="1",
                description="Run the project checks.",
                exposure="named_check",
                executable="/opt/golden/.venv/bin/pytest",
                fixed_arguments=("-q", "tests"),
                max_arguments=0,
                timeout_seconds=300,
            ),
        ),
        dependency_inputs=(
            DockerCodingDependencyInput(path="pyproject.toml", content_sha256="sha256:" + "b" * 64),
            DockerCodingDependencyInput(
                path="uv.lock", content_sha256="sha256:" + "c" * 64, max_bytes=1024
            ),
        ),
        trusted_build_context_sha256="sha256:" + "d" * 64,
        application_dependency_identity="golden-deps-1",
        admission_probes=(
            DockerCodingAdmissionProbe(
                probe_id="python-version",
                argv=("/opt/golden/.venv/bin/pytest", "--version"),
                expected_exit_codes=(0, 4),
                stdout_sha256="sha256:" + "e" * 64,
                timeout_seconds=5,
            ),
        ),
        result_publication_max_bytes=60_000,
        redaction_behavior_version="2",
        adoption_behavior_version="3",
    )


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class _AllowPolicy(CommandPolicy):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="golden.policy", behavior_version="1", implementation_version="1"
        )

    async def evaluate(self, ctx, request) -> CommandPolicyResult:
        del ctx, request
        return CommandPolicyResult(decision=CommandPolicyDecision.ALLOW)


def _run_check_tool(profile) -> RunCheckTool:
    authority = profile.command_authority("test")
    assert authority is not None
    return RunCheckTool(
        checks=[
            NamedCheck(
                name="test",
                description="Run the project checks.",
                command=ExecCommand.process(*authority.command_argv()),
                timeout_s=authority.timeout_seconds,
                max_output_bytes=authority.max_output_bytes,
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="golden.check", behavior_version="1", implementation_version="1"
                ),
            )
        ],
        command_policy=_AllowPolicy(),
        toolchain_profile=profile,
    )


def test_docker_profile_dump_and_identity_match_the_release_before_the_neutral_base() -> None:
    # Pinned from origin/main 53664bece, before DockerCodingToolchainProfile became
    # a subclass. Reconnect metadata, configuration fingerprints, policy identities
    # and receipts rebuild these from serialized data, so they must never move.
    profile = _golden_profile()

    assert (
        sha256(profile.model_dump_json(by_alias=True).encode()).hexdigest()
        == "24b5183a312581d9e7386825177e9d4c2b1b2581b887574653a07640e7af3cf7"
    )
    assert (
        sha256(profile.model_dump_json().encode()).hexdigest()
        == "2699f3030a145a325ef994147829e2938d8b75f7172f56af438f30ca48e1186c"
    )
    assert list(profile.model_dump(by_alias=True))[:4] == [
        "schema",
        "profile_id",
        "revision",
        "image_identity",
    ]
    assert profile.fingerprint == (
        "sha256:2abf90bba7ea63525c47cc4790ce220b96ba0e9fddfb01d716cad6e4022eadf8"
    )
    assert profile.dependency_identity == (
        "sha256:31dd69f0d41124e3b816d87a807f01531ec33b17e8968affbe7b0c79aa8f69c7"
    )
    assert _digest(profile.evidence()) == (
        "ab56a0eaac93d20d71750d95e223cb1107bdfcdb7f3de1714f7de938a62603bf"
    )
    assert [authority.fingerprint for authority in profile.command_authorities] == [
        "sha256:bbaebbc629fdd042f2c208517f4dbe54f46d49f5b868acb1228fdd9039c05ddd",
        "sha256:a313d4d5e345ef22a92c9edcd8360150cac1b48b3c7fc7fc28f792cd336b1906",
        "sha256:ea2c3287f386bb23dadf8c2c962d2fe450885c3a04b485403f72d0966f879158",
    ]
    assert (
        DockerCodingToolchainProfile.model_validate_json(profile.model_dump_json(by_alias=True))
        == profile
    )


def test_docker_typed_tool_and_factory_identities_match_the_release_before() -> None:
    profile = _golden_profile()
    command = RunCommandTool(toolchain_profile=profile)
    policy = StructuredCommandToolPolicy(toolchain_profile=profile)
    check = _run_check_tool(profile)

    # Presentation metadata is outside the execution contract pinned here.
    assert _digest(command.spec.model_dump(mode="json", exclude={"display"})) == (
        "38111b1966ff83a4878acd7e466d6ff019d29660177eecbe877cb435567ce9c5"
    )
    assert _digest(command._execution_profile_material()) == (
        "bb34dc41ed15af327763033b5ebf1a04304bb168e1c1e8dd790182a6e19a1ba6"
    )
    assert _digest(command.command_policy.execution_profile_identity.model_dump(mode="json")) == (
        "0d218d6b5290d2b80cddb01dcddf59536f951e573c2ca85052147bd85e24d877"
    )
    assert policy.execution_profile_identity is not None
    assert _digest(policy.execution_profile_identity.model_dump(mode="json")) == (
        "b876ea1f181c5666d42f6c351e4d6843800b539a19d45312e5a6b9330eccc472"
    )
    assert _digest(policy._execution_profile_material()) == (
        "fd6ea04e1c252e58d371b0989d3b84c9f9e01433a3214909658fb23520dc50d6"
    )
    assert _digest(check.spec.model_dump(mode="json", exclude={"display"})) == (
        "15007c30c7200026eb30a7daeb4763f484f3e71c0a143d08e8972cf6e3048dc5"
    )
    assert _digest(check._execution_profile_material()) == (
        "3eb7b2c8b0ebe299d478a18e3e0b2756ddedbb662bf7ea974ab2e95f4f27c502"
    )
    assert type(command.toolchain_profile) is DockerCodingToolchainProfile


def test_docker_factory_identity_matches_the_release_before(tmp_path: Path) -> None:
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(
            tmp_path, workspace_id="golden", excluded_directory_names=(".git",)
        ),
        toolchain_profile=_golden_profile(),
    )

    assert factory.execution_profile_identity.model_dump(mode="json") == {
        "name": "cayu.docker_coding_environment",
        "behavior_version": "15",
        "implementation_version": (
            "sha256:943bfe10f3b58af4cabc95d9b03c3c4d5e84ff8f967f398e2dc0fcf2d3ce6314"
        ),
    }


def test_shared_classes_keep_their_docker_names_and_schemas() -> None:
    assert DockerCodingCommandAuthority is CodingCommandAuthority
    assert DockerCodingDependencyInput is CodingDependencyInput
    assert DockerCodingAdmissionProbe is CodingAdmissionProbe
    assert issubclass(DockerCodingToolchainProfile, CodingToolchainProfile)
    assert issubclass(DockerCodingToolchainError, CodingToolchainError)
    authority = _golden_profile().command_authorities[0]
    assert authority.model_dump(by_alias=True)["schema"] == (
        "cayu.docker_coding_command_authority.v1"
    )


def test_neutral_profile_round_trips_through_docker_without_losing_authority() -> None:
    docker = _golden_profile()
    neutral = docker.coding_toolchain()

    assert type(neutral) is CodingToolchainProfile
    assert neutral.model_dump(by_alias=True)["schema"] == "cayu.coding_toolchain_profile.v1"
    assert neutral.command_authorities == docker.command_authorities
    assert neutral.required_executables == docker.required_executables
    assert neutral.dependency_identity == docker.dependency_identity
    assert neutral.fingerprint != docker.fingerprint
    assert "toolchain_image_fingerprint" not in neutral.evidence()
    assert neutral.evidence()["toolchain_platform"] == "linux/arm64"

    rebuilt = DockerCodingToolchainProfile.from_toolchain(
        neutral,
        image_identity=_IMAGE,
        restrictions=docker.restrictions,
    )
    assert rebuilt == docker
    assert rebuilt.fingerprint == docker.fingerprint


def test_from_toolchain_binds_architecture_and_docker_only_probes() -> None:
    neutral = (
        _golden_profile().coding_toolchain().model_copy(update={"platform_architecture": None})
    )
    platform = CodingAdmissionProbe(
        probe_id="platform",
        argv=("/opt/golden/.venv/bin/pytest", "--platform"),
    )

    with pytest.raises(ValueError, match="needs platform_architecture"):
        DockerCodingToolchainProfile.from_toolchain(neutral, image_identity=_IMAGE)
    docker = DockerCodingToolchainProfile.from_toolchain(
        neutral,
        image_identity=_IMAGE,
        platform_architecture="amd64",
        extra_admission_probes=(platform,),
    )
    assert docker.platform_architecture == "amd64"
    assert docker.runtime_user == docker.restrictions.user == "1000:1000"
    assert [probe.probe_id for probe in docker.admission_probes] == ["platform", "python-version"]
    assert neutral.admission_probes == _golden_profile().admission_probes

    pinned = neutral.model_copy(update={"platform_architecture": "arm64"})
    with pytest.raises(ValueError, match="conflicts"):
        DockerCodingToolchainProfile.from_toolchain(
            pinned, image_identity=_IMAGE, platform_architecture="amd64"
        )
    with pytest.raises(TypeError, match="exact CodingToolchainProfile"):
        DockerCodingToolchainProfile.from_toolchain(_golden_profile(), image_identity=_IMAGE)


def test_owned_profile_gate_accepts_only_the_two_exact_classes() -> None:
    docker = _golden_profile()
    neutral = docker.coding_toolchain()

    owned_docker = owned_coding_toolchain_profile(docker)
    owned_neutral = owned_coding_toolchain_profile(neutral)
    assert type(owned_docker) is DockerCodingToolchainProfile
    assert owned_docker == docker and owned_docker is not docker
    assert type(owned_neutral) is CodingToolchainProfile
    assert owned_neutral == neutral

    class WiderNeutral(CodingToolchainProfile):
        @property
        def required_executables(self) -> tuple[str, ...]:
            return ("/bin/sh",)

    class WiderDocker(DockerCodingToolchainProfile):
        pass

    wider_neutral = WiderNeutral.model_validate(neutral.model_dump(by_alias=True))
    wider_docker = WiderDocker.model_validate(docker.model_dump(by_alias=True))
    for value in (wider_neutral, wider_docker, neutral.model_dump(by_alias=True), None):
        with pytest.raises(TypeError, match="exact CodingToolchainProfile"):
            owned_coding_toolchain_profile(value)
    with pytest.raises(TypeError, match="exact CodingToolchainProfile"):
        RunCommandTool(toolchain_profile=wider_docker)
    with pytest.raises(TypeError, match="exact CodingToolchainProfile"):
        StructuredCommandToolPolicy(toolchain_profile=wider_neutral)
    with pytest.raises(TypeError, match="exact CodingToolchainProfile"):
        _run_check_tool(wider_neutral)


def test_neutral_profile_still_prohibits_network() -> None:
    fields = _golden_profile().coding_toolchain().model_dump(by_alias=True)
    fields["prohibited_capabilities"] = tuple(
        item for item in fields["prohibited_capabilities"] if item != "network"
    )

    with pytest.raises(ValidationError, match="Coding toolchain profiles cannot omit"):
        CodingToolchainProfile.model_validate(fields)


def test_validation_errors_name_the_selector_and_probe_at_fault() -> None:
    fields = _golden_profile().coding_toolchain().model_dump(by_alias=True)
    authorities = list(fields["command_authorities"])
    extra = {**authorities[0], "selector": "target-test"}

    with pytest.raises(ValidationError) as unsorted:
        CodingToolchainProfile.model_validate(
            {**fields, "command_authorities": [*authorities, extra]}
        )
    assert "'target-test' must come before 'test'" in str(unsorted.value)
    with pytest.raises(ValidationError) as repeated:
        CodingToolchainProfile.model_validate(
            {**fields, "command_authorities": [authorities[0], *authorities]}
        )
    assert "'focused-test' is repeated" in str(repeated.value)

    moved = [{**item, "executable": "/opt/target/.venv/bin/pytest"} for item in authorities]
    with pytest.raises(ValidationError) as probe:
        CodingToolchainProfile.model_validate({**fields, "command_authorities": moved})
    assert "probe 'python-version' runs '/opt/golden/.venv/bin/pytest'" in str(probe.value)


def _dependency_profile(content: bytes) -> CodingToolchainProfile:
    return CodingToolchainProfile(
        profile_id="neutral-deps",
        revision="1",
        command_authorities=(
            CodingCommandAuthority(
                selector="test",
                revision="1",
                description="Run tests.",
                exposure="named_check",
                executable="/usr/bin/true",
                max_arguments=0,
            ),
        ),
        dependency_inputs=(
            CodingDependencyInput(
                path="uv.lock", content_sha256="sha256:" + sha256(content).hexdigest()
            ),
        ),
    )


def test_neutral_dependency_checks_raise_the_neutral_error(tmp_path: Path) -> None:
    profile = _dependency_profile(b"locked\n")
    (tmp_path / "uv.lock").write_bytes(b"locked\n")

    verify_local_coding_toolchain_dependencies(profile, tmp_path)
    asyncio.run(verify_coding_toolchain_dependencies(profile, LocalWorkspace(tmp_path)))

    (tmp_path / "uv.lock").unlink()
    with pytest.raises(CodingToolchainError) as missing:
        verify_local_coding_toolchain_dependencies(profile, tmp_path)
    assert type(missing.value) is CodingToolchainError
    assert missing.value.code == "dependency_inputs_unavailable"
    assert missing.value.diagnostic() == {
        "code": "dependency_inputs_unavailable",
        "path_count": 1,
        "paths_fingerprint": missing.value.paths_fingerprint,
    }
    assert "uv.lock" not in str(missing.value)
    assert "1 of 1 declared inputs" in str(missing.value)


def test_docker_dependency_checks_keep_raising_the_docker_error(tmp_path: Path) -> None:
    docker = DockerCodingToolchainProfile.from_toolchain(
        _dependency_profile(b"locked\n"), image_identity=_IMAGE, platform_architecture="amd64"
    )
    (tmp_path / "uv.lock").write_bytes(b"changed\n")

    with pytest.raises(DockerCodingToolchainError) as changed:
        asyncio.run(verify_docker_coding_toolchain_dependencies(docker, LocalWorkspace(tmp_path)))
    assert changed.value.code == "dependency_inputs_changed"
    with pytest.raises(DockerCodingToolchainError):
        asyncio.run(verify_coding_toolchain_dependencies(docker, LocalWorkspace(tmp_path)))
    with pytest.raises(TypeError, match="exact DockerCodingToolchainProfile"):
        asyncio.run(
            verify_docker_coding_toolchain_dependencies(
                docker.coding_toolchain(),
                LocalWorkspace(tmp_path),
            )
        )
