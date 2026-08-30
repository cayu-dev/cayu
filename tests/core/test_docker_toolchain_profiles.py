from __future__ import annotations

import asyncio
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from cayu import (
    DockerCodingCommandAuthority,
    DockerCodingDependencyInput,
    DockerCodingToolchainError,
    DockerCodingToolchainProfile,
    DockerImageIdentity,
    LocalWorkspace,
    verify_docker_coding_toolchain_dependencies,
    verify_local_docker_coding_toolchain_dependencies,
)


def _digest(content: bytes) -> str:
    return "sha256:" + sha256(content).hexdigest()


def _image() -> DockerImageIdentity:
    return DockerImageIdentity(
        reference="registry.example/coding@sha256:" + ("a" * 64),
    )


def _profile(lock_content: bytes = b"locked\n") -> DockerCodingToolchainProfile:
    return DockerCodingToolchainProfile(
        profile_id="custom-rust",
        revision="2026-08-29",
        image_identity=_image(),
        platform_architecture="amd64",
        command_authorities=(
            DockerCodingCommandAuthority(
                selector="focused-test",
                revision="1",
                description="Run one admitted test source.",
                exposure="structured_command",
                executable="/usr/bin/cargo",
                fixed_arguments=("test",),
                allow_positional_arguments=True,
                positional_arguments_are_paths=True,
                positional_path_prefixes=("tests",),
                positional_path_suffixes=(".rs",),
                max_arguments=2,
            ),
        ),
        dependency_inputs=(
            DockerCodingDependencyInput(
                path="Cargo.lock",
                content_sha256=_digest(lock_content),
            ),
        ),
    )


def test_profile_is_closed_immutable_and_identity_changes_with_authority() -> None:
    profile = _profile()
    rebuilt = DockerCodingToolchainProfile.model_validate(
        profile.model_dump(mode="python", by_alias=True)
    )

    assert rebuilt == profile
    assert rebuilt.fingerprint == profile.fingerprint
    assert rebuilt.required_executables == ("/usr/bin/cargo",)
    assert rebuilt.command_authority("focused-test").command_argv(("tests/unit.rs",)) == (
        "/usr/bin/cargo",
        "test",
        "tests/unit.rs",
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DockerCodingToolchainProfile.model_validate(
            {**profile.model_dump(mode="python", by_alias=True), "image": "model-owned"}
        )


@pytest.mark.parametrize(
    "arguments",
    (
        ("../tests/unit.rs",),
        ("/tests/unit.rs",),
        ("tests/*.rs",),
        ("@response.txt",),
        ("--manifest-path", "../../Cargo.toml"),
        ("tests/unit.py",),
    ),
)
def test_selector_argument_grammar_fails_closed(arguments: tuple[str, ...]) -> None:
    authority = _profile().command_authority("focused-test")
    assert authority is not None
    with pytest.raises(ValueError):
        authority.validate_model_arguments(arguments)


def test_dependency_drift_fails_before_dispatch_on_local_and_workspace_paths(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "Cargo.lock"
    lock.write_bytes(b"locked\n")
    profile = _profile()

    verify_local_docker_coding_toolchain_dependencies(profile, tmp_path)
    asyncio.run(verify_docker_coding_toolchain_dependencies(profile, LocalWorkspace(tmp_path)))

    lock.write_bytes(b"changed\n")
    with pytest.raises(DockerCodingToolchainError) as local_error:
        verify_local_docker_coding_toolchain_dependencies(profile, tmp_path)
    assert local_error.value.code == "dependency_inputs_changed"
    assert local_error.value.paths == ("Cargo.lock",)
    with pytest.raises(DockerCodingToolchainError) as workspace_error:
        asyncio.run(verify_docker_coding_toolchain_dependencies(profile, LocalWorkspace(tmp_path)))
    assert workspace_error.value.code == "dependency_inputs_changed"


def test_explicit_empty_dependency_set_is_valid() -> None:
    profile = _profile().model_copy(update={"dependency_inputs": ()})

    verify_local_docker_coding_toolchain_dependencies(profile, Path.cwd())


def test_selector_required_flags_and_path_values_are_closed() -> None:
    authority = DockerCodingCommandAuthority(
        selector="package-test",
        revision="1",
        description="Run one admitted package manifest.",
        exposure="structured_command",
        executable="/usr/bin/cargo",
        fixed_arguments=("test",),
        allowed_flags=("--manifest-path", "--quiet"),
        required_flags=("--quiet",),
        flags_with_values=("--manifest-path",),
        path_value_flags=("--manifest-path",),
        positional_arguments_are_paths=True,
        allow_positional_arguments=True,
        positional_path_prefixes=("crates",),
        positional_path_suffixes=(".toml",),
        max_arguments=3,
    )

    assert authority.validate_model_arguments(
        ("--quiet", "--manifest-path", "crates/core/Cargo.toml")
    ) == ("--quiet", "--manifest-path", "crates/core/Cargo.toml")
    with pytest.raises(ValueError, match="required flag"):
        authority.validate_model_arguments(("--manifest-path", "crates/core/Cargo.toml"))
    with pytest.raises(ValueError, match="outside its admitted scope"):
        authority.validate_model_arguments(("--quiet", "--manifest-path", "vendor/core/Cargo.toml"))


def test_selector_finite_literal_allowlist_rejects_generic_positional_fallback() -> None:
    authority = DockerCodingCommandAuthority(
        selector="integration-test",
        revision="1",
        description="Run one admitted integration target.",
        exposure="structured_command",
        executable="/usr/bin/cargo",
        allowed_literals=("api", "storage"),
        allow_positional_arguments=True,
        min_arguments=1,
        max_arguments=1,
    )

    assert authority.validate_model_arguments(("api",)) == ("api",)
    with pytest.raises(ValueError, match="undeclared literal"):
        authority.validate_model_arguments(("not-declared",))


def test_local_dependency_verification_refuses_symlink_components(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    source = tmp_path / "source"
    outside.mkdir()
    source.mkdir()
    (outside / "Cargo.lock").write_bytes(b"locked\n")
    (source / "deps").symlink_to(outside, target_is_directory=True)
    profile = _profile().model_copy(
        update={
            "dependency_inputs": (
                DockerCodingDependencyInput(
                    path="deps/Cargo.lock",
                    content_sha256=_digest(b"locked\n"),
                ),
            )
        }
    )

    with pytest.raises(DockerCodingToolchainError) as caught:
        verify_local_docker_coding_toolchain_dependencies(profile, source)

    assert caught.value.code == "dependency_inputs_unavailable"
