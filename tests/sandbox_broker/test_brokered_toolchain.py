"""A brokered coding sandbox bound to a neutral toolchain profile, against the fake broker."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest
from tests.sandbox_broker.fake_broker import FakeBroker

from cayu.environments import (
    CodingAdmissionProbe,
    CodingCommandAuthority,
    CodingDependencyInput,
    CodingToolchainError,
    CodingToolchainProfile,
    DockerCodingToolchainProfile,
    EnvironmentFactoryRequest,
    coding_toolchain_runner_admission_failure,
)
from cayu.runners import DockerImageIdentity
from cayu.sandbox_broker import (
    BrokeredSandboxCodingEnvironmentFactory,
    BrokeredSandboxRunner,
    SandboxRecord,
)
from cayu.workspaces import LocalWorkspace

_LOCK = b"locked\n"
_CONTEXT = "sha256:" + "c" * 64
_PYTEST = "#!/bin/sh\ncase \"$1\" in --version) echo 'pytest 8.0.0';; *) echo ok;; esac\n"


def _toolchain(root: Path) -> Path:
    """A host directory standing in for the image's /opt/tools."""

    tools = root / "opt-tools"
    (tools / "bin").mkdir(parents=True)
    (tools / "bin" / "pytest").write_text(_PYTEST, encoding="utf-8")
    (tools / "bin" / "pytest").chmod(0o755)
    return tools


def _profile(**overrides) -> CodingToolchainProfile:
    fields = {
        "profile_id": "brokered-python",
        "revision": "1",
        "read_only_support_paths": ("/opt/tools",),
        "command_authorities": (
            CodingCommandAuthority(
                selector="test",
                revision="1",
                description="Run tests.",
                exposure="named_check",
                executable="/opt/tools/bin/pytest",
                fixed_arguments=("-q",),
                max_arguments=0,
            ),
        ),
        "dependency_inputs": (
            CodingDependencyInput(
                path="uv.lock", content_sha256="sha256:" + hashlib.sha256(_LOCK).hexdigest()
            ),
        ),
        "trusted_build_context_sha256": _CONTEXT,
        "admission_probes": (
            CodingAdmissionProbe(
                probe_id="pytest-version",
                argv=("/opt/tools/bin/pytest", "--version"),
                stdout_sha256="sha256:" + hashlib.sha256(b"pytest 8.0.0\n").hexdigest(),
            ),
        ),
    }
    fields.update(overrides)
    return CodingToolchainProfile(**fields)


def _source(tmp_path: Path) -> LocalWorkspace:
    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    (source / "uv.lock").write_bytes(_LOCK)
    (source / "calc.py").write_bytes(b"VALUE = 1\n")
    return LocalWorkspace(source, workspace_id="project", excluded_directory_names=(".git",))


def _broker(tmp_path: Path, *, identity: bool = True) -> FakeBroker:
    broker = FakeBroker(tmp_path / "cloud")
    broker.catalog.add("release")
    broker.guest_paths["/opt/tools"] = _toolchain(tmp_path)
    if identity:
        broker.image_identities["release"] = {
            "name": "release",
            "content_digest": "sha256:" + "d" * 64,
            "trusted_build_context_sha256": _CONTEXT,
            "platform_architecture": "arm64",
        }
    return broker


def _request() -> EnvironmentFactoryRequest:
    return EnvironmentFactoryRequest(
        session_id="coding", agent_name="coder", environment_name="coding"
    )


def _factory(broker: FakeBroker, tmp_path: Path, **overrides):
    options = {
        "broker": broker.client(),
        "source_workspace": _source(tmp_path),
        "egress": "none",
        "image": "release",
        "toolchain_profile": _profile(),
    }
    options.update(overrides)
    return BrokeredSandboxCodingEnvironmentFactory(**options)


def test_each_sandbox_is_bound_to_the_profile_before_use(tmp_path: Path) -> None:
    broker = _broker(tmp_path)
    profile = _profile()
    factory = _factory(broker, tmp_path, toolchain_profile=profile)

    async def scenario():
        result = await factory.create(_request())
        runner = result.environment.runner
        assert isinstance(runner, BrokeredSandboxRunner)
        evidence = runner.execution_admission_candidate().evidence
        failure = coding_toolchain_runner_admission_failure(runner, profile=profile)
        await result.release(None)
        return result.metadata, evidence, failure

    metadata, evidence, failure = asyncio.run(scenario())
    assert failure is None
    assert evidence.toolchain_profile_fingerprint == profile.fingerprint
    assert evidence.image_fingerprint is not None
    assert {item.state for item in evidence.tool_requirements.executables} == {"live_verified"}
    assert metadata["toolchain_profile_fingerprint"] == profile.fingerprint
    assert metadata["sandbox_image"] == "release"
    (sandbox,) = broker.sandboxes.values()
    assert sandbox.body["image"] == "release"
    assert ["/opt/tools/bin/pytest", "--version"] in broker.started
    assert broker.live() == []
    assert factory.construction_admission_candidate().evidence.toolchain_profile_fingerprint == (
        profile.fingerprint
    )


def test_a_catalog_image_without_the_build_identity_fails_admission_clearly(
    tmp_path: Path,
) -> None:
    broker = _broker(tmp_path, identity=False)
    factory = _factory(broker, tmp_path)

    with pytest.raises(CodingToolchainError) as refused:
        asyncio.run(factory.create(_request()))
    assert refused.value.code == "toolchain_image_unverified"
    assert "trusted build context" in str(refused.value)
    assert broker.live() == []


def test_probe_and_executable_failures_name_what_failed_and_leave_nothing(tmp_path: Path) -> None:
    broker = _broker(tmp_path)
    wrong_digest = _profile(
        admission_probes=(
            CodingAdmissionProbe(
                probe_id="pytest-version",
                argv=("/opt/tools/bin/pytest", "--version"),
                stdout_sha256="sha256:" + "0" * 64,
            ),
        )
    )
    with pytest.raises(CodingToolchainError) as probe:
        asyncio.run(_factory(broker, tmp_path, toolchain_profile=wrong_digest).create(_request()))
    assert probe.value.code == "admission_probe_mismatch"
    assert "'pytest-version'" in str(probe.value)

    missing = _profile(
        admission_probes=(),
        command_authorities=(
            *_profile().command_authorities,
            CodingCommandAuthority(
                selector="z-missing",
                revision="1",
                description="Not in the image.",
                exposure="named_check",
                executable="/opt/tools/bin/missing",
                max_arguments=0,
            ),
        ),
    )
    with pytest.raises(CodingToolchainError) as executables:
        asyncio.run(
            _factory(broker, tmp_path / "second", toolchain_profile=missing).create(_request())
        )
    assert executables.value.code == "toolchain_executables_unavailable"
    assert executables.value.path_count == 1
    assert broker.live() == []


def test_stale_source_dependencies_fail_before_allocation(tmp_path: Path) -> None:
    broker = _broker(tmp_path)
    factory = _factory(broker, tmp_path)
    (factory.source_workspace.root / "uv.lock").write_bytes(b"changed\n")

    with pytest.raises(CodingToolchainError) as stale:
        asyncio.run(factory.create(_request()))
    assert stale.value.code == "dependency_inputs_changed"
    assert broker.sandboxes == {}


def test_a_profile_needs_no_network_no_setup_and_a_neutral_type(tmp_path: Path) -> None:
    broker = _broker(tmp_path)
    profile = _profile()
    docker = DockerCodingToolchainProfile.from_toolchain(
        profile,
        image_identity=DockerImageIdentity(reference="x:1", content_digest="sha256:" + "a" * 64),
        platform_architecture="amd64",
    )
    with pytest.raises(TypeError, match=r"profile\.coding_toolchain\(\)"):
        _factory(broker, tmp_path, toolchain_profile=docker)
    with pytest.raises(ValueError, match="egress='none'"):
        _factory(broker, tmp_path, egress="public_web")
    with pytest.raises(ValueError, match="no setup_commands or workspace_setup_commands"):
        _factory(broker, tmp_path, setup_commands=["uv sync"])
    with pytest.raises(ValueError, match="no setup_commands or workspace_setup_commands"):
        _factory(broker, tmp_path, workspace_setup_commands=["uv sync"])
    with pytest.raises(ValueError, match="workspace is /workspace"):
        _factory(
            broker,
            tmp_path,
            toolchain_profile=_profile(working_directory="/src", workspace_path="/src"),
        )

    class Wider(CodingToolchainProfile):
        pass

    with pytest.raises(TypeError, match="exact CodingToolchainProfile"):
        _factory(
            broker,
            tmp_path,
            toolchain_profile=Wider.model_validate(profile.model_dump(by_alias=True)),
        )


def test_workspace_setup_commands_run_after_copy_in(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    source = _source(tmp_path)

    def factory(**options):
        return BrokeredSandboxCodingEnvironmentFactory(
            broker=broker.client(),
            source_workspace=source,
            egress="none",
            git_baseline=False,
            **options,
        )

    async def bind(selected):
        result = await selected.create(_request())
        environment = result.environment
        try:
            bound = await environment.binding.bind(
                environment.workspace, environment.runner, session_id="coding"
            )
            environment.binding.abandon(bound)
        finally:
            await result.release(None)

    # The project file exists only once the workspace is copied in.
    asyncio.run(bind(factory(workspace_setup_commands=["test -f calc.py"])))
    with pytest.raises(RuntimeError, match="setup command failed"):
        asyncio.run(bind(factory(setup_commands=["test -f calc.py"])))
    with pytest.raises(RuntimeError, match="workspace setup command failed"):
        asyncio.run(bind(factory(workspace_setup_commands=["test -f missing.py"])))
    assert broker.live() == []


def test_factory_identity_follows_the_profile_and_image(tmp_path: Path) -> None:
    broker = _broker(tmp_path)
    first = _factory(broker, tmp_path)
    same = _factory(broker, tmp_path / "other")
    other_image = _factory(broker, tmp_path / "third", image="coding-python")
    no_profile = _factory(broker, tmp_path / "fourth", toolchain_profile=None)

    assert first.execution_profile_identity.name == "cayu.brokered_sandbox_coding_environment"
    assert first.execution_profile_identity == same.execution_profile_identity
    assert first.execution_profile_identity != other_image.execution_profile_identity
    assert first.execution_profile_identity != no_profile.execution_profile_identity


def test_an_unrecognised_image_identity_is_ignored() -> None:
    record = {
        "id": "sbx_1",
        "image": "coding-python",
        "egress": {"mode": "none"},
        "status": "ready",
    }
    assert (
        SandboxRecord.model_validate({**record, "image_identity": "opaque"}).image_identity is None
    )
    assert (
        SandboxRecord.model_validate({**record, "image_identity": {"name": 5}}).image_identity
        is None
    )
    identity = SandboxRecord.model_validate(
        {**record, "image_identity": {"trusted_build_context_sha256": _CONTEXT, "new": 1}}
    ).image_identity
    assert identity is not None and identity.trusted_build_context_sha256 == _CONTEXT
