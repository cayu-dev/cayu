from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.docker_toolchain import docker_toolchain_profile
from tests.environments.test_docker_coding import (
    _image_identity,
    _LocalDockerRunner,
    _NoGitLocalDockerRunner,
)

from cayu._coding_product_authority import (
    CODING_PRODUCT_SOURCE_AUTHORITY_METADATA_KEY,
    CodingProductSourceCopyAuthority,
)
from cayu.environments.docker_coding import (
    DockerCodingEnvironmentFactory,
    DockerCodingWorkspaceBinding,
)
from cayu.environments.factory import EnvironmentFactoryRequest
from cayu.runners.base import ExecCommand
from cayu.workspaces.local import LocalWorkspace
from cayu.workspaces.revisions import WorkspaceRevisionObservationLimits
from cayu.workspaces.runner import RunnerWorkspace

_PROTECTED = (".cayu", ".git", ".runtime")
_GIT_IDENTITY = "-c user.name=Reviewer -c user.email=reviewer@example.invalid"
_CANDIDATE_PATCH = """\
diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1 +1 @@
-value = 1
+value = 2
"""
# Make the baseline the unpatched tree and leave the candidate as uncommitted changes.
_UNPATCHED_BASELINE = (
    "git apply -R candidate.patch",
    f"git {_GIT_IDENTITY} commit -q --amend --no-edit -a",
    "git apply candidate.patch",
)


def _factory(source: LocalWorkspace, **kwargs) -> DockerCodingEnvironmentFactory:
    return DockerCodingEnvironmentFactory(
        source_workspace=source,
        toolchain_profile=docker_toolchain_profile(image_identity=_image_identity()),
        **kwargs,
    )


def _source(root: Path) -> LocalWorkspace:
    return LocalWorkspace(root, workspace_id="setup-source", excluded_directory_names=_PROTECTED)


def _request(session_id: str, **kwargs) -> EnvironmentFactoryRequest:
    return EnvironmentFactoryRequest(
        session_id=session_id, agent_name="agent", environment_name="coding", **kwargs
    )


def _binding(
    factory: DockerCodingEnvironmentFactory, runner: _LocalDockerRunner, session_id: str
) -> DockerCodingWorkspaceBinding:
    target = RunnerWorkspace(
        runner, workspace_id="setup-target", excluded_directory_names=_PROTECTED
    )
    return factory.create_workspace_binding(_request(session_id), target_workspace=target)


async def _git(runner: _LocalDockerRunner, script: str) -> str:
    result = await runner.exec(ExecCommand.bash(script))
    assert result.exit_code == 0, result.stderr
    return result.stdout


def test_setup_commands_run_after_the_baseline_and_can_amend_it(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    # The candidate is already applied in the source, as a fix review receives it.
    (source_root / "calc.py").write_text("value = 2\n", encoding="utf-8")
    (source_root / "candidate.patch").write_text(_CANDIDATE_PATCH, encoding="utf-8")
    log = tmp_path / "setup.log"
    source = _source(source_root)
    factory = _factory(
        source, workspace_setup_commands=(*_UNPATCHED_BASELINE, f"printf 'ran\\n' >> {log}")
    )
    runner = _LocalDockerRunner(target_root)
    binding = _binding(factory, runner, "fix-review")

    async def run() -> None:
        bound = await binding.bind(source, runner, session_id="fix-review")
        assert log.read_text(encoding="utf-8") == "ran\n"
        assert await _git(runner, "git show HEAD:calc.py") == "value = 1\n"
        assert await _git(runner, "git rev-list --count HEAD") == "1\n"
        assert await _git(runner, "git diff --name-only") == "calc.py\n"
        # The binding records the amended baseline, not the one it committed.
        head = (await _git(runner, "git rev-parse HEAD")).strip()
        assert binding._bind_authority(bound).git is not None
        assert binding._bind_authority(bound).git.head_revision == head
        # A reviewer's stash comparison now sees the unpatched code.
        await _git(runner, f"git {_GIT_IDENTITY} stash -q")
        assert (target_root / "calc.py").read_text(encoding="utf-8") == "value = 1\n"
        await _git(runner, "git stash pop -q")
        assert (target_root / "calc.py").read_text(encoding="utf-8") == "value = 2\n"

        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is not None

    asyncio.run(run())

    # Setup never runs again on copy-back, and the source keeps the candidate.
    assert log.read_text(encoding="utf-8") == "ran\n"
    assert (source_root / "calc.py").read_text(encoding="utf-8") == "value = 2\n"
    assert not (source_root / ".git").exists()


def test_failing_setup_command_fails_the_bind_and_stops(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "code.py").write_text("value = 1\n", encoding="utf-8")
    source = _source(source_root)
    factory = _factory(
        source,
        workspace_setup_commands=(
            "printf first > first.txt",
            "echo setup-broke >&2; exit 3",
            "printf never > never.txt",
        ),
    )
    runner = _LocalDockerRunner(target_root)
    binding = _binding(factory, runner, "failing-setup")

    with pytest.raises(
        RuntimeError,
        match=r"Docker coding workspace setup command failed \(exit code 3\); "
        r"stderr: .*setup-broke",
    ):
        asyncio.run(binding.bind(source, runner, session_id="failing-setup"))

    assert (target_root / "first.txt").exists()
    assert not (target_root / "never.txt").exists()
    # The failed bind was abandoned, so nothing can finalize or publish it.
    assert binding._coding_authorities == {}
    assert not (source_root / "first.txt").exists()


def test_setup_command_timeout_fails_the_bind(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = _source(source_root)
    factory = _factory(source, workspace_setup_commands=("sleep 30",), setup_command_timeout_s=1)
    runner = _LocalDockerRunner(target_root)
    binding = _binding(factory, runner, "slow-setup")

    with pytest.raises(RuntimeError, match="setup command failed \\(timed out after 1 seconds\\)"):
        asyncio.run(binding.bind(source, runner, session_id="slow-setup"))


def test_setup_commands_run_without_a_git_baseline(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "input.csv").write_text("a,b\n", encoding="utf-8")
    source = _source(source_root)
    factory = _factory(
        source,
        git_baseline=False,
        workspace_setup_commands=("test -f input.csv && printf ready > prepared.txt",),
    )
    runner = _NoGitLocalDockerRunner(target_root)
    binding = _binding(factory, runner, "no-git-setup")

    async def run() -> None:
        bound = await binding.bind(source, runner, session_id="no-git-setup")
        assert not (target_root / ".git").exists()
        await binding.finalize(bound, outcome="completed")

    asyncio.run(run())

    # What setup writes in the workspace is copied back like any other change.
    assert (source_root / "prepared.txt").read_text(encoding="utf-8") == "ready"


def test_setup_commands_are_part_of_the_execution_profile(tmp_path: Path) -> None:
    source = LocalWorkspace(tmp_path, workspace_id="profile-source")
    default = _factory(source)
    empty = _factory(source, workspace_setup_commands=[])
    with_setup = _factory(source, workspace_setup_commands=["git status"])
    other_setup = _factory(source, workspace_setup_commands=["git status", "true"])
    patient = _factory(source, workspace_setup_commands=["git status"], setup_command_timeout_s=60)

    assert default.workspace_setup_commands == ()
    assert with_setup.workspace_setup_commands == ("git status",)
    assert empty.execution_profile_identity == default.execution_profile_identity
    assert with_setup.execution_profile_identity != default.execution_profile_identity
    assert other_setup.execution_profile_identity != with_setup.execution_profile_identity
    # The timeout is operational: raising it doesn't strand preserved sessions.
    assert patient.setup_command_timeout_s == 60
    assert patient.execution_profile_identity == with_setup.execution_profile_identity

    rebuilt = with_setup.with_source_workspace(LocalWorkspace(tmp_path, workspace_id="other"))
    assert rebuilt is not None
    assert rebuilt.workspace_setup_commands == ("git status",)
    assert rebuilt.execution_profile_identity == with_setup.execution_profile_identity

    with pytest.raises(TypeError, match="not one string"):
        _factory(source, workspace_setup_commands="git status")
    with pytest.raises(ValueError, match="nonblank shell commands"):
        _factory(source, workspace_setup_commands=["git status", "  "])
    with pytest.raises(ValueError, match="positive integer"):
        _factory(source, workspace_setup_commands=["true"], setup_command_timeout_s=0)


def test_coding_product_authority_refuses_setup_commands(tmp_path: Path) -> None:
    source = _source(tmp_path)
    factory = _factory(source, workspace_setup_commands=["true"])
    authority = CodingProductSourceCopyAuthority(
        request_fingerprint="sha256:" + ("a" * 64),
        source_workspace_id=source.id,
        baseline_revision="sha256:" + ("b" * 64),
        observation_limits=WorkspaceRevisionObservationLimits(),
    )
    request = _request(
        "product",
        metadata={CODING_PRODUCT_SOURCE_AUTHORITY_METADATA_KEY: authority.model_dump(mode="json")},
    )
    target = RunnerWorkspace(
        _LocalDockerRunner(tmp_path),
        workspace_id="product-target",
        excluded_directory_names=_PROTECTED,
    )

    with pytest.raises(ValueError, match="can't be combined with workspace_setup_commands"):
        factory.create_workspace_binding(request, target_workspace=target)
