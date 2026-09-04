"""Explicit Docker toolchain fixtures for environment-contract tests."""

from typing import Literal

from cayu.environments import (
    DockerCodingCommandAuthority,
    DockerCodingToolchainProfile,
)
from cayu.runners import DockerImageIdentity, DockerWorkloadRestrictions


def docker_toolchain_profile(
    *,
    image_identity: DockerImageIdentity,
    restrictions: DockerWorkloadRestrictions | None = None,
    required_executables: tuple[str, ...] = (),
    platform_architecture: Literal["amd64", "arm64"] = "amd64",
) -> DockerCodingToolchainProfile:
    restrictions = restrictions or DockerWorkloadRestrictions()
    return DockerCodingToolchainProfile(
        profile_id="test-toolchain",
        revision="1",
        image_identity=image_identity,
        platform_architecture=platform_architecture,
        restrictions=restrictions,
        runtime_user=restrictions.user,
        command_authorities=tuple(
            DockerCodingCommandAuthority(
                selector=f"check-{index}",
                revision="1",
                description="Test executable admission",
                exposure="named_check",
                executable=name if name.startswith("/") else f"/usr/bin/{name}",
                max_arguments=0,
            )
            for index, name in enumerate(required_executables)
        ),
    )
