"""Fixture seams for a generated Docker coding composition (``operations/coding.py``).

Tests use these instead of a recorded image and a Docker daemon. They replace
only the image receipt and the daemon's image check; the composition still
selects its backend, builds its factory and registers its environment.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Literal

from cayu.environments import CodingToolchainProfile, DockerCodingToolchainProfile
from cayu.runners import DockerImageIdentity

FIXTURE_IMAGE = DockerImageIdentity(reference="fixture@sha256:" + "a" * 64)
FIXTURE_BUILD_CONTEXT = "sha256:" + "c" * 64


def fixture_receipt(
    operations: ModuleType,
    *,
    image_identity: DockerImageIdentity = FIXTURE_IMAGE,
    platform_architecture: Literal["amd64", "arm64"] = "amd64",
    trusted_build_context_sha256: str = FIXTURE_BUILD_CONTEXT,
) -> Any:
    """A recorded-image receipt for the generated composition, without a build."""

    return operations.CodingImageReceipt(
        image_identity=image_identity,
        profile_revision="1",
        platform_architecture=platform_architecture,
        dependency_inputs=(),
        trusted_build_context_sha256=trusted_build_context_sha256,
    )


def generated_docker_toolchain_profile(
    operations: ModuleType,
    *,
    image_identity: DockerImageIdentity = FIXTURE_IMAGE,
    platform_architecture: Literal["amd64", "arm64"] = "amd64",
) -> DockerCodingToolchainProfile:
    """The generated profile bound to a fixture image, as the Docker backend runs it."""

    receipt = fixture_receipt(
        operations, image_identity=image_identity, platform_architecture=platform_architecture
    )
    return DockerCodingToolchainProfile.from_toolchain(
        operations._python_toolchain_profile(receipt),
        image_identity=image_identity,
        platform_architecture=platform_architecture,
    )


def admit_fixture_image(
    monkeypatch: Any,
    operations: ModuleType,
    *,
    toolchain_profile: CodingToolchainProfile | None = None,
    image_identity: DockerImageIdentity | None = None,
    platform_architecture: Literal["amd64", "arm64"] = "amd64",
    environ: dict[str, str] | None = None,
) -> Any:
    """Make the generated composition use a fixture image instead of a recorded one.

    A Docker-typed ``toolchain_profile`` supplies the image, architecture and
    build context and replaces the generated profile with its neutral part.
    The backend is selected from ``environ`` (empty by default: Docker).
    """

    build_context = FIXTURE_BUILD_CONTEXT
    if type(toolchain_profile) is DockerCodingToolchainProfile:
        image_identity = toolchain_profile.image_identity
        platform_architecture = toolchain_profile.platform_architecture
        build_context = toolchain_profile.trusted_build_context_sha256 or build_context
        toolchain_profile = toolchain_profile.coding_toolchain()
    elif toolchain_profile is not None and toolchain_profile.trusted_build_context_sha256:
        build_context = toolchain_profile.trusted_build_context_sha256
    receipt = fixture_receipt(
        operations,
        image_identity=image_identity or FIXTURE_IMAGE,
        platform_architecture=platform_architecture,
        trusted_build_context_sha256=build_context,
    )
    select = operations._coding_backend
    monkeypatch.setattr(operations, "read_coding_image_receipt", lambda: receipt)
    monkeypatch.setattr(
        operations,
        "_coding_backend",
        lambda profile, recorded: select(
            profile, recorded, environ=environ or {}, verify_local_image=False
        ),
    )
    if toolchain_profile is not None:
        neutral = toolchain_profile
        monkeypatch.setattr(operations, "_python_toolchain_profile", lambda recorded: neutral)
    return receipt
