"""DockerCodingEnvironmentFactory verifies a neutral toolchain profile against one image."""

from __future__ import annotations

from pathlib import Path

import pytest

from cayu.environments import (
    CodingAdmissionProbe,
    CodingCommandAuthority,
    CodingToolchainProfile,
    DockerCodingEnvironmentFactory,
    DockerCodingToolchainProfile,
)
from cayu.runners import DockerImageIdentity
from cayu.workspaces import LocalWorkspace

_IMAGE = DockerImageIdentity(reference="coding:local", content_digest="sha256:" + "a" * 64)


def _neutral() -> CodingToolchainProfile:
    return CodingToolchainProfile(
        profile_id="neutral",
        revision="1",
        command_authorities=(
            CodingCommandAuthority(
                selector="test",
                revision="1",
                description="Run tests.",
                exposure="named_check",
                executable="/opt/project/bin/pytest",
                max_arguments=0,
            ),
        ),
    )


def _source(tmp_path: Path) -> LocalWorkspace:
    return LocalWorkspace(tmp_path, workspace_id="source", excluded_directory_names=(".git",))


def test_neutral_profile_is_bound_to_the_image_and_reports_its_own_fingerprint(
    tmp_path: Path,
) -> None:
    neutral = _neutral()
    platform = CodingAdmissionProbe(probe_id="platform", argv=("/opt/project/bin/pytest", "-V"))
    factory = DockerCodingEnvironmentFactory(
        source_workspace=_source(tmp_path),
        toolchain_profile=neutral,
        image_identity=_IMAGE,
        platform_architecture="arm64",
        extra_admission_probes=(platform,),
    )

    assert type(factory.toolchain_profile) is DockerCodingToolchainProfile
    assert factory.toolchain_profile.image_identity == _IMAGE
    assert factory.toolchain_profile.platform_architecture == "arm64"
    assert [probe.probe_id for probe in factory.toolchain_profile.admission_probes] == ["platform"]
    assert factory.coding_toolchain_profile == neutral
    assert factory.image_identity == _IMAGE
    # Tools hold the neutral profile, so the runners report its fingerprint.
    candidate = factory.construction_admission_candidate()
    assert candidate.evidence.toolchain_profile_fingerprint == neutral.fingerprint
    assert candidate.evidence.image_fingerprint == _IMAGE.fingerprint

    docker_typed = DockerCodingEnvironmentFactory(
        source_workspace=_source(tmp_path),
        toolchain_profile=factory.toolchain_profile,
    )
    assert (
        docker_typed.construction_admission_candidate().evidence.toolchain_profile_fingerprint
        == (factory.toolchain_profile.fingerprint)
    )
    # The same image and authority reported under another fingerprint is another profile.
    assert docker_typed.execution_profile_identity != factory.execution_profile_identity


def test_profile_and_docker_options_must_agree(tmp_path: Path) -> None:
    neutral = _neutral()
    with pytest.raises(ValueError, match="needs image_identity"):
        DockerCodingEnvironmentFactory(
            source_workspace=_source(tmp_path), toolchain_profile=neutral
        )
    with pytest.raises(ValueError, match="needs platform_architecture"):
        DockerCodingEnvironmentFactory(
            source_workspace=_source(tmp_path), toolchain_profile=neutral, image_identity=_IMAGE
        )
    docker = DockerCodingToolchainProfile.from_toolchain(
        neutral, image_identity=_IMAGE, platform_architecture="amd64"
    )
    with pytest.raises(ValueError, match="already fixes its image"):
        DockerCodingEnvironmentFactory(
            source_workspace=_source(tmp_path), toolchain_profile=docker, image_identity=_IMAGE
        )

    class Wider(CodingToolchainProfile):
        pass

    with pytest.raises(TypeError, match="exact CodingToolchainProfile"):
        DockerCodingEnvironmentFactory(
            source_workspace=_source(tmp_path),
            toolchain_profile=Wider.model_validate(neutral.model_dump(by_alias=True)),
            image_identity=_IMAGE,
            platform_architecture="amd64",
        )
