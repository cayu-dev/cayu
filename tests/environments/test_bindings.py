from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import time
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from cayu.environments import (
    BoundWorkspace,
    GitRepositoryBinding,
    NativeBinding,
    NoWorkspaceBinding,
    SyncBinding,
    SyncBindingContext,
    WorkspaceBinding,
    WorkspaceSnapshot,
    copy_bound_workspace,
    copy_workspace_snapshot,
)
from cayu.environments.bindings import _reset_workspace_after_failed_clone
from cayu.runners import E2BRunner, ExecCommand, ExecResult, LocalRunner, Runner
from cayu.workspaces import (
    E2BWorkspace,
    LocalWorkspace,
    RunnerWorkspace,
    Workspace,
    WorkspaceListResult,
    WorkspaceReadResult,
)


class StubWorkspace(Workspace):
    id = "stub-workspace"

    async def read_bytes(
        self,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        return WorkspaceReadResult(content=b"", total_bytes=0)

    async def write_bytes(self, path: str, content: bytes) -> None:
        pass

    async def delete(self, path: str) -> None:
        pass

    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        return WorkspaceListResult(paths=(), total_count=0)


class StubRunner(Runner):
    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        return ExecResult(stdout="ok")


def _require_git() -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required for GitRepositoryBinding tests")


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _create_bare_origin(tmp_path: Path) -> tuple[Path, str]:
    _require_git()
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    origin.mkdir(parents=True)
    seed.mkdir(parents=True)
    _git(origin, "init", "--bare")
    _git(seed, "init")
    _git(seed, "checkout", "-b", "main")
    _git(seed, "config", "user.email", "tester@example.com")
    _git(seed, "config", "user.name", "Test User")
    (seed / "README.md").write_text("hello\n", encoding="utf-8")
    _git(seed, "add", "README.md")
    _git(seed, "commit", "-m", "initial")
    commit = _git(seed, "rev-parse", "HEAD")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "origin", "main")
    _git(origin, "symbolic-ref", "HEAD", "refs/heads/main")
    return origin, commit


def test_native_binding_passes_configured_workspace_and_runner_through() -> None:
    workspace = StubWorkspace()
    runner = StubRunner()
    metadata = {"mount": {"id": "mnt_1"}}

    bound = asyncio.run(
        NativeBinding(default_path="/workspace").bind(
            workspace,
            runner,
            session_id="sess_1",
            agent_name="agent",
            environment_name="env",
            metadata=metadata,
        )
    )

    assert bound.workspace is workspace
    assert bound.source_workspace is workspace
    assert bound.runner is runner
    assert bound.path == "/workspace"
    assert bound.metadata == {"mount": {"id": "mnt_1"}}

    metadata["mount"]["id"] = "mutated"
    assert bound.metadata == {"mount": {"id": "mnt_1"}}


def test_no_workspace_binding_hides_workspace() -> None:
    workspace = StubWorkspace()
    runner = StubRunner()

    bound = asyncio.run(
        NoWorkspaceBinding().bind(
            workspace,
            runner,
            session_id="sess_1",
            metadata={"reason": "api-only"},
        )
    )

    assert bound.workspace is None
    assert bound.source_workspace is workspace
    assert bound.runner is runner
    assert bound.path is None
    assert bound.metadata == {"reason": "api-only"}


def test_git_repository_binding_clones_local_origin_and_reports_snapshots(tmp_path) -> None:
    origin, commit = _create_bare_origin(tmp_path)
    target_root = tmp_path / "target"
    target_root.mkdir()
    workspace = LocalWorkspace(target_root, workspace_id="repo-workspace")

    async def run() -> tuple[BoundWorkspace, WorkspaceSnapshot | None]:
        binding = GitRepositoryBinding(repo_url=str(origin), ref="main", path="/workspace")
        bound = await binding.bind(
            workspace,
            None,
            session_id="sess_git",
            agent_name="assistant",
            environment_name="env",
            metadata={"request": "meta"},
        )
        (target_root / "README.md").write_text("changed\n", encoding="utf-8")
        final_snapshot = await binding.finalize(bound, outcome="completed")
        return bound, final_snapshot

    bound, final_snapshot = asyncio.run(run())

    assert (target_root / ".git").is_dir()
    assert (target_root / "README.md").read_text(encoding="utf-8") == "changed\n"
    assert bound.workspace is workspace
    assert bound.source_workspace is workspace
    assert bound.path == "/workspace"
    assert bound.metadata["request"] == "meta"
    assert bound.metadata["git_repository"]["repo_url"] == str(origin)
    assert bound.metadata["git_repository"]["ref"] == "main"
    assert bound.metadata["git_repository"]["commit"] == commit
    assert bound.metadata["git_repository"]["dirty"] is False
    assert bound.snapshot is not None
    assert bound.snapshot.source == "git"
    assert bound.snapshot.version == commit
    assert final_snapshot is not None
    assert final_snapshot.source == "git"
    assert final_snapshot.version == commit
    assert final_snapshot.metadata["git_repository"]["dirty"] is True
    assert final_snapshot.metadata["git_repository"]["outcome"] == "completed"


def test_git_repository_binding_fetches_and_checks_out_pull_request_ref(tmp_path) -> None:
    origin, base_commit = _create_bare_origin(tmp_path)
    seed = tmp_path / "seed"
    # Publish a PR-head commit under refs/pull/1/head, like GitHub does. The
    # default clone/fetch refspec (refs/heads/*) does not cover it.
    (seed / "feature.txt").write_text("pr change\n", encoding="utf-8")
    _git(seed, "add", "feature.txt")
    _git(seed, "commit", "-m", "pr head")
    pr_head = _git(seed, "rev-parse", "HEAD")
    _git(seed, "push", "origin", "HEAD:refs/pull/1/head")
    _git(seed, "reset", "--hard", base_commit)

    target_root = tmp_path / "target"
    target_root.mkdir()
    workspace = LocalWorkspace(target_root, workspace_id="pr-workspace")

    async def run() -> BoundWorkspace:
        binding = GitRepositoryBinding(
            repo_url=str(origin),
            ref="pr-1",
            fetch_refspecs=["+refs/pull/1/head:refs/heads/pr-1"],
        )
        return await binding.bind(workspace, None, session_id="sess_pr")

    bound = asyncio.run(run())

    assert pr_head != base_commit
    assert bound.metadata["git_repository"]["commit"] == pr_head
    assert bound.metadata["git_repository"]["ref"] == "pr-1"
    assert bound.metadata["git_repository"]["fetch_refspecs"] == [
        "+refs/pull/1/head:refs/heads/pr-1"
    ]
    assert (target_root / "feature.txt").read_text(encoding="utf-8") == "pr change\n"


def test_git_repository_binding_rejects_refspecs_when_fetch_disabled(tmp_path) -> None:
    origin, _commit = _create_bare_origin(tmp_path)

    with pytest.raises(ValueError, match="fetch_refspecs requires fetch=True"):
        GitRepositoryBinding(
            repo_url=str(origin),
            ref="pr-1",
            fetch=False,
            fetch_refspecs=["+refs/pull/1/head:refs/heads/pr-1"],
        )


def test_git_repository_binding_uses_runner_workspace(tmp_path) -> None:
    origin, commit = _create_bare_origin(tmp_path)
    runner_root = tmp_path / "runner"
    runner_root.mkdir()
    runner = LocalRunner(runner_root)
    workspace = RunnerWorkspace(runner, workspace_id="runner-repo")

    async def run() -> BoundWorkspace:
        return await GitRepositoryBinding(repo_url=str(origin), ref="main").bind(
            workspace,
            runner,
            session_id="sess_runner_git",
        )

    bound = asyncio.run(run())

    assert (runner_root / ".git").is_dir()
    assert (runner_root / "README.md").read_text(encoding="utf-8") == "hello\n"
    assert bound.workspace is workspace
    assert bound.runner is runner
    assert bound.snapshot is not None
    assert bound.snapshot.version == commit


def test_git_repository_binding_updates_existing_checkout_to_fetched_ref(tmp_path) -> None:
    origin, first_commit = _create_bare_origin(tmp_path)
    target_root = tmp_path / "target"
    target_root.mkdir()
    _git(target_root, "clone", str(origin), ".")
    assert _git(target_root, "rev-parse", "HEAD") == first_commit

    seed = tmp_path / "second-seed"
    _git(seed.parent, "clone", str(origin), seed.name)
    _git(seed, "config", "user.email", "tester@example.com")
    _git(seed, "config", "user.name", "Test User")
    (seed / "README.md").write_text("second\n", encoding="utf-8")
    _git(seed, "add", "README.md")
    _git(seed, "commit", "-m", "second")
    second_commit = _git(seed, "rev-parse", "HEAD")
    _git(seed, "push", "origin", "main")

    workspace = LocalWorkspace(target_root)

    async def run() -> BoundWorkspace:
        return await GitRepositoryBinding(repo_url=str(origin), ref="main").bind(
            workspace,
            None,
            session_id="sess_stale_git",
        )

    bound = asyncio.run(run())

    assert bound.snapshot is not None
    assert bound.snapshot.version == second_commit
    assert bound.metadata["git_repository"]["commit"] == second_commit
    assert _git(target_root, "rev-parse", "HEAD") == second_commit
    assert _git(target_root, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert (target_root / "README.md").read_text(encoding="utf-8") == "second\n"


def test_git_repository_binding_refuses_divergent_existing_branch(tmp_path) -> None:
    origin, _commit = _create_bare_origin(tmp_path)
    target_root = tmp_path / "target"
    target_root.mkdir()
    _git(target_root, "clone", str(origin), ".")
    _git(target_root, "config", "user.email", "tester@example.com")
    _git(target_root, "config", "user.name", "Test User")
    (target_root / "local.txt").write_text("local\n", encoding="utf-8")
    _git(target_root, "add", "local.txt")
    _git(target_root, "commit", "-m", "local")
    local_commit = _git(target_root, "rev-parse", "HEAD")

    seed = tmp_path / "second-seed"
    _git(seed.parent, "clone", str(origin), seed.name)
    _git(seed, "config", "user.email", "tester@example.com")
    _git(seed, "config", "user.name", "Test User")
    (seed / "remote.txt").write_text("remote\n", encoding="utf-8")
    _git(seed, "add", "remote.txt")
    _git(seed, "commit", "-m", "remote")
    _git(seed, "push", "origin", "main")

    workspace = LocalWorkspace(target_root)

    async def run() -> None:
        await GitRepositoryBinding(repo_url=str(origin), ref="main").bind(
            workspace,
            None,
            session_id="sess_diverged_git",
        )

    with pytest.raises(RuntimeError, match="ff-only"):
        asyncio.run(run())
    assert _git(target_root, "rev-parse", "HEAD") == local_commit


def test_git_repository_binding_refuses_non_empty_non_git_workspace(tmp_path) -> None:
    origin, _commit = _create_bare_origin(tmp_path)
    target_root = tmp_path / "target"
    target_root.mkdir()
    (target_root / "notes.txt").write_text("not git\n", encoding="utf-8")
    workspace = LocalWorkspace(target_root)

    async def run() -> None:
        await GitRepositoryBinding(repo_url=str(origin), ref="main").bind(
            workspace,
            None,
            session_id="sess_non_empty_git",
        )

    with pytest.raises(ValueError, match="empty workspace"):
        asyncio.run(run())


def test_git_repository_binding_refuses_directory_only_non_git_workspace(tmp_path) -> None:
    origin, _commit = _create_bare_origin(tmp_path)
    target_root = tmp_path / "target"
    target_root.mkdir()
    (target_root / "nested").mkdir()
    workspace = LocalWorkspace(target_root)

    async def run() -> None:
        await GitRepositoryBinding(repo_url=str(origin), ref="main").bind(
            workspace,
            None,
            session_id="sess_directory_only_git",
        )

    with pytest.raises(ValueError, match="empty workspace"):
        asyncio.run(run())


def test_git_repository_binding_refuses_dirty_existing_repo(tmp_path) -> None:
    origin, _commit = _create_bare_origin(tmp_path)
    target_root = tmp_path / "target"
    target_root.mkdir()
    _git(target_root, "clone", str(origin), ".")
    (target_root / "README.md").write_text("dirty\n", encoding="utf-8")
    workspace = LocalWorkspace(target_root)

    async def run() -> None:
        await GitRepositoryBinding(repo_url=str(origin), ref="main").bind(
            workspace,
            None,
            session_id="sess_dirty_git",
        )

    with pytest.raises(ValueError, match="dirty repository"):
        asyncio.run(run())


def test_git_repository_binding_refuses_unexpected_remote(tmp_path) -> None:
    origin, _commit = _create_bare_origin(tmp_path)
    other_origin, _other_commit = _create_bare_origin(tmp_path / "other")
    target_root = tmp_path / "target"
    target_root.mkdir()
    _git(target_root, "clone", str(origin), ".")
    workspace = LocalWorkspace(target_root)

    async def run() -> None:
        await GitRepositoryBinding(repo_url=str(other_origin), ref="main").bind(
            workspace,
            None,
            session_id="sess_wrong_remote",
        )

    with pytest.raises(ValueError, match="remote URL"):
        asyncio.run(run())


def test_git_repository_binding_rejects_credential_bearing_https_url() -> None:
    with pytest.raises(ValueError, match="embedded credentials"):
        GitRepositoryBinding(repo_url="https://token:secret@example.com/acme/app.git")


def test_git_repository_binding_rejects_option_like_git_inputs() -> None:
    with pytest.raises(ValueError, match="repo_url"):
        GitRepositoryBinding(repo_url="--upload-pack=bad")
    with pytest.raises(ValueError, match="ref"):
        GitRepositoryBinding(repo_url="https://example.com/acme/app.git", ref="--detach")
    with pytest.raises(ValueError, match="remote_name"):
        GitRepositoryBinding(repo_url="https://example.com/acme/app.git", remote_name="--tags")
    with pytest.raises(ValueError, match="git_executable"):
        GitRepositoryBinding(repo_url="https://example.com/acme/app.git", git_executable="-git")


def test_reset_workspace_after_failed_clone_clears_local_workspace(tmp_path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("x", encoding="utf-8")
    (root / "partial.txt").write_text("partial", encoding="utf-8")
    workspace = LocalWorkspace(root, workspace_id="ws")

    asyncio.run(
        _reset_workspace_after_failed_clone(workspace, timeout_s=None, output_limit_bytes=1024)
    )

    # Partial clone artifacts (including the .git directory and dotfiles) are removed, so the
    # workspace is empty again and a later bind is not permanently bricked.
    assert list(root.iterdir()) == []


def test_reset_workspace_after_failed_clone_propagates_cancellation(tmp_path) -> None:
    # Cleanup swallows ordinary errors, but a CancelledError arriving mid-cleanup must propagate
    # rather than being dropped (which would leave the task cancelled-but-not-delivered).
    runner_root = tmp_path / "runner"
    runner_root.mkdir()

    class CancellingRunner(LocalRunner):
        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            raise asyncio.CancelledError

    workspace = RunnerWorkspace(CancellingRunner(runner_root), workspace_id="ws")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            _reset_workspace_after_failed_clone(workspace, timeout_s=None, output_limit_bytes=1024)
        )


def test_git_repository_binding_resets_workspace_after_failed_clone(tmp_path) -> None:
    _require_git()
    origin, commit = _create_bare_origin(tmp_path)
    runner_root = tmp_path / "runner"
    runner_root.mkdir()

    class FlakyCloneRunner(LocalRunner):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self._root = root
            self.fail_next_clone = True

        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            if command.argv and "clone" in command.argv and self.fail_next_clone:
                self.fail_next_clone = False
                # Simulate a clone that dies mid-transfer, leaving partial non-work-tree artifacts.
                (self._root / ".git").mkdir(exist_ok=True)
                (self._root / "partial.txt").write_text("partial", encoding="utf-8")
                return ExecResult(stdout="", stderr="fatal: early EOF", exit_code=128)
            return await super().exec(command, **kwargs)

    runner = FlakyCloneRunner(runner_root)
    workspace = RunnerWorkspace(runner, workspace_id="runner-repo")
    binding = GitRepositoryBinding(repo_url=str(origin), ref="main")

    async def run() -> BoundWorkspace:
        with pytest.raises(RuntimeError):
            await binding.bind(workspace, runner, session_id="sess_clone_fail")
        # The failed clone's partial artifacts were reset, so the workspace is empty again and the
        # retry clones cleanly instead of being permanently bricked.
        assert not (runner_root / "partial.txt").exists()
        assert not (runner_root / ".git").exists()
        return await binding.bind(workspace, runner, session_id="sess_clone_retry")

    bound = asyncio.run(run())

    assert (runner_root / ".git").is_dir()
    assert bound.snapshot is not None
    assert bound.snapshot.version == commit


def test_git_repository_binding_resets_workspace_after_cancelled_clone(tmp_path) -> None:
    # A cancelled/interrupted clone leaves the same partial, bricking state as an ordinary failure,
    # so it must also reset the workspace (CancelledError is a BaseException, not an Exception).
    _require_git()
    origin, _ = _create_bare_origin(tmp_path)
    runner_root = tmp_path / "runner"
    runner_root.mkdir()

    class CancellingCloneRunner(LocalRunner):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self._root = root

        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            if command.argv and "clone" in command.argv:
                (self._root / ".git").mkdir(exist_ok=True)
                (self._root / "partial.txt").write_text("partial", encoding="utf-8")
                raise asyncio.CancelledError
            return await super().exec(command, **kwargs)

    runner = CancellingCloneRunner(runner_root)
    workspace = RunnerWorkspace(runner, workspace_id="repo")
    binding = GitRepositoryBinding(repo_url=str(origin), ref="main")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(binding.bind(workspace, runner, session_id="sess_clone_cancel"))

    # The cancellation propagated, but the workspace was still reset to empty (not bricked).
    assert not (runner_root / "partial.txt").exists()
    assert not (runner_root / ".git").exists()


def test_bind_request_rejects_invalid_values() -> None:
    invalid_workspace: Any = object()
    invalid_runner: Any = object()
    invalid_metadata: Any = []

    with pytest.raises(TypeError, match="workspace"):
        asyncio.run(NativeBinding().bind(invalid_workspace, None, session_id="sess_1"))
    with pytest.raises(TypeError, match="runner"):
        asyncio.run(NativeBinding().bind(None, invalid_runner, session_id="sess_1"))
    with pytest.raises(ValueError, match="session_id"):
        asyncio.run(NativeBinding().bind(None, None, session_id=" "))
    with pytest.raises(ValueError, match="agent_name"):
        asyncio.run(NativeBinding().bind(None, None, session_id="sess_1", agent_name=" "))
    with pytest.raises(ValueError, match="environment_name"):
        asyncio.run(NativeBinding().bind(None, None, session_id="sess_1", environment_name=" "))
    with pytest.raises(TypeError, match="metadata"):
        asyncio.run(
            NativeBinding().bind(None, None, session_id="sess_1", metadata=invalid_metadata)
        )
    with pytest.raises(ValueError, match="metadata"):
        asyncio.run(
            NativeBinding().bind(None, None, session_id="sess_1", metadata={"bad": object()})
        )


def test_binding_finalize_methods_are_noops() -> None:
    bound = BoundWorkspace()

    async def run() -> tuple[WorkspaceSnapshot | None, WorkspaceSnapshot | None]:
        return (
            await NativeBinding().finalize(bound, outcome="completed"),
            await NoWorkspaceBinding().finalize(
                bound,
                outcome="completed",
                metadata={"ok": True},
            ),
        )

    assert asyncio.run(run()) == (None, None)


def test_sync_binding_copies_source_to_target_and_syncs_back(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("before", encoding="utf-8")
    (source_root / "nested").mkdir()
    (source_root / "nested" / "b.txt").write_text("delete me", encoding="utf-8")
    (target_root / "stale.txt").write_text("stale", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")

    async def run() -> tuple[BoundWorkspace, WorkspaceSnapshot | None]:
        binding = SyncBinding(target_workspace=target, path="/workspace")
        bound = await binding.bind(
            source,
            None,
            session_id="sess_sync",
            agent_name="assistant",
            environment_name="env",
            metadata={"request": "meta"},
        )
        await target.write_bytes("a.txt", b"after")
        await target.delete("nested/b.txt")
        await target.write_bytes("new.txt", b"created")
        final_snapshot = await binding.finalize(
            bound,
            outcome="completed",
            metadata={"final": True},
        )
        return bound, final_snapshot

    bound, final_snapshot = asyncio.run(run())

    assert bound.workspace is target
    assert bound.source_workspace is source
    assert type(bound.state_key) is str
    assert bound.path == "/workspace"
    assert bound.snapshot is not None
    assert bound.snapshot.source == "sync"
    assert bound.metadata["request"] == "meta"
    assert "source_paths" not in bound.metadata["sync_binding"]
    assert "target_baseline_paths" not in bound.metadata["sync_binding"]
    assert "sync_state_id" not in bound.metadata["sync_binding"]
    assert bound.metadata["sync_binding"]["cleaned_target_files"] == 1
    assert not (target_root / "stale.txt").exists()
    assert (source_root / "a.txt").read_text(encoding="utf-8") == "after"
    assert not (source_root / "nested" / "b.txt").exists()
    assert (source_root / "new.txt").read_text(encoding="utf-8") == "created"
    assert final_snapshot is not None
    assert final_snapshot.workspace_id == "source"
    assert final_snapshot.source == "sync"
    assert final_snapshot.metadata["copied_files"] == 2
    assert final_snapshot.metadata["deleted_files"] == 1
    assert "deleted_paths" not in final_snapshot.metadata
    assert final_snapshot.metadata["final"] is True


def test_sync_binding_can_use_target_workspace_factory(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("before", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    calls: list[SyncBindingContext] = []

    async def factory(context: SyncBindingContext) -> Workspace:
        calls.append(context)
        return target

    async def run() -> BoundWorkspace:
        return await SyncBinding(target_workspace_factory=factory).bind(
            source,
            None,
            session_id="sess_sync_factory",
            agent_name="assistant",
            environment_name="env",
            metadata={"request": "meta"},
        )

    bound = asyncio.run(run())

    assert bound.workspace is target
    assert bound.source_workspace is source
    assert len(calls) == 1
    assert calls[0].source_workspace is source
    assert calls[0].runner is None
    assert calls[0].session_id == "sess_sync_factory"
    assert calls[0].agent_name == "assistant"
    assert calls[0].environment_name == "env"
    assert calls[0].metadata == {"request": "meta"}
    assert (target_root / "a.txt").read_text(encoding="utf-8") == "before"


def test_sync_binding_rejects_source_as_target_workspace(tmp_path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = LocalWorkspace(source_root, workspace_id="source")

    async def run() -> None:
        await SyncBinding(target_workspace=source).bind(
            source,
            None,
            session_id="sess_sync_same_workspace",
        )

    with pytest.raises(ValueError, match="different"):
        asyncio.run(run())


def test_sync_binding_rejects_target_with_same_workspace_id(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = LocalWorkspace(source_root, workspace_id="same")
    target = LocalWorkspace(target_root, workspace_id="same")

    async def run() -> None:
        await SyncBinding(target_workspace=target).bind(
            source,
            None,
            session_id="sess_sync_same_workspace_id",
        )

    with pytest.raises(ValueError, match="different"):
        asyncio.run(run())


def test_sync_binding_rejects_target_with_same_local_root(tmp_path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    source = LocalWorkspace(root, workspace_id="source")
    target = LocalWorkspace(root, workspace_id="target")

    async def run() -> None:
        await SyncBinding(target_workspace=target).bind(
            source,
            None,
            session_id="sess_sync_same_local_root",
        )

    with pytest.raises(ValueError, match="different"):
        asyncio.run(run())


def test_sync_binding_rejects_target_subclass_with_same_local_root(tmp_path) -> None:
    class CustomLocalWorkspace(LocalWorkspace):
        pass

    root = tmp_path / "workspace"
    root.mkdir()
    source = CustomLocalWorkspace(root, workspace_id="source")
    target = LocalWorkspace(root, workspace_id="target")

    async def run() -> None:
        await SyncBinding(target_workspace=target).bind(
            source,
            None,
            session_id="sess_sync_same_local_root_subclass",
        )

    with pytest.raises(ValueError, match="different"):
        asyncio.run(run())


def test_sync_binding_rejects_target_with_same_runner_cwd(tmp_path) -> None:
    root = tmp_path / "runner"
    root.mkdir()
    runner = LocalRunner(root)
    source = RunnerWorkspace(runner, cwd=".", workspace_id="source")
    target = RunnerWorkspace(runner, cwd=".", workspace_id="target")

    async def run() -> None:
        await SyncBinding(target_workspace=target).bind(
            source,
            runner,
            session_id="sess_sync_same_runner_cwd",
        )

    with pytest.raises(ValueError, match="different"):
        asyncio.run(run())


class _OpaqueWorkspace(StubWorkspace):
    """Custom Workspace with no stable identity (``resource_key`` defaults to None)."""

    def __init__(self, workspace_id: str) -> None:
        self.id = workspace_id


class _KeyedWorkspace(StubWorkspace):
    """Custom Workspace that reports a stable identity via ``resource_key``."""

    def __init__(self, workspace_id: str, key: str) -> None:
        self.id = workspace_id
        self._key = key

    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("keyed", self._key)


def test_sync_binding_refuses_indeterminate_custom_workspace_identity() -> None:
    source = _OpaqueWorkspace("source")
    target = _OpaqueWorkspace("target")

    async def run() -> None:
        await SyncBinding(target_workspace=target).bind(
            source,
            None,
            session_id="sess_sync_opaque",
        )

    with pytest.raises(ValueError, match="does not define resource_key"):
        asyncio.run(run())


def test_sync_binding_rejects_custom_workspace_with_matching_resource_key() -> None:
    source = _KeyedWorkspace("source", "shared")
    target = _KeyedWorkspace("target", "shared")

    async def run() -> None:
        await SyncBinding(target_workspace=target).bind(
            source,
            None,
            session_id="sess_sync_keyed_same",
        )

    with pytest.raises(ValueError, match="different"):
        asyncio.run(run())


def test_sync_binding_allows_custom_workspace_with_distinct_resource_key() -> None:
    source = _KeyedWorkspace("source", "src")
    target = _KeyedWorkspace("target", "dst")

    async def run() -> None:
        await SyncBinding(target_workspace=target).bind(
            source,
            None,
            session_id="sess_sync_keyed_diff",
        )

    asyncio.run(run())


def test_sync_binding_rejects_local_workspace_aliased_by_runner_workspace(tmp_path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    local = LocalWorkspace(root, workspace_id="local")
    runner_view = RunnerWorkspace(LocalRunner(root), cwd=None, workspace_id="runner")

    async def run(source, target) -> None:
        await SyncBinding(target_workspace=target).bind(source, None, session_id="sess_alias")

    # A LocalRunner-backed RunnerWorkspace addresses the same host dir as the LocalWorkspace, so the
    # canonical "local" key must match in both directions (pre-fix these differed and the source was wiped).
    with pytest.raises(ValueError, match="different"):
        asyncio.run(run(local, runner_view))
    with pytest.raises(ValueError, match="different"):
        asyncio.run(run(runner_view, local))


def test_sync_binding_allows_local_and_runner_workspace_over_distinct_roots(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = LocalWorkspace(source_root, workspace_id="local")
    target = RunnerWorkspace(LocalRunner(target_root), cwd=None, workspace_id="runner")

    async def run() -> None:
        await SyncBinding(target_workspace=target).bind(
            source,
            None,
            session_id="sess_distinct_roots",
        )

    asyncio.run(run())


def test_sync_binding_refuses_runner_workspace_with_indeterminate_runner() -> None:
    # StubRunner exposes none of the probed identity attrs, so its resource_key is indeterminate.
    # Two distinct stub runners over the "same" external resource must fail closed, not pass on object id.
    source = RunnerWorkspace(StubRunner(), workspace_id="source")
    target = RunnerWorkspace(StubRunner(), workspace_id="target")

    async def run() -> None:
        await SyncBinding(target_workspace=target).bind(
            source,
            None,
            session_id="sess_opaque_runner",
        )

    with pytest.raises(ValueError, match="does not define resource_key"):
        asyncio.run(run())


def _offline_e2b_runner(default_cwd: str = "/home/user/workspace") -> E2BRunner:
    return E2BRunner(object(), sandbox_id="sbx_same", default_cwd=default_cwd, e2b_module=object())


def test_runner_workspace_resource_key_matches_native_e2b_wrapper() -> None:
    runner = _offline_e2b_runner()
    native = E2BWorkspace(runner, workspace_id="e2b")
    runner_view = RunnerWorkspace(runner, cwd=None, workspace_id="runner")

    # A RunnerWorkspace over a remote runner addresses the runner's default_cwd; its key must resolve to
    # that absolute guest path so it matches the native wrapper (pre-fix the RunnerWorkspace path was ".").
    assert runner_view.resource_key == native.resource_key
    assert runner_view.resource_key[2] == "/home/user/workspace"


def test_sync_binding_rejects_native_e2b_wrapper_aliased_by_runner_workspace() -> None:
    runner = _offline_e2b_runner()
    native = E2BWorkspace(runner, workspace_id="e2b")
    runner_view = RunnerWorkspace(runner, cwd=None, workspace_id="runner")

    async def run(source, target) -> None:
        await SyncBinding(target_workspace=target).bind(
            source, runner, session_id="sess_remote_alias"
        )

    with pytest.raises(ValueError, match="different"):
        asyncio.run(run(native, runner_view))
    with pytest.raises(ValueError, match="different"):
        asyncio.run(run(runner_view, native))


def test_sync_binding_allows_native_e2b_wrapper_over_distinct_dir() -> None:
    runner = _offline_e2b_runner(default_cwd="/home/user/workspace")
    native = E2BWorkspace(runner, root="/home/user/other", workspace_id="e2b")
    runner_view = RunnerWorkspace(runner, cwd=None, workspace_id="runner")

    # Same sandbox, genuinely different guest directories -> distinct keys -> not aliased.
    assert native.resource_key != runner_view.resource_key


def test_sync_binding_can_finalize_from_copied_bound_workspace(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("before", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(target_workspace=target)

    async def run() -> WorkspaceSnapshot | None:
        bound = await binding.bind(source, None, session_id="sess_sync_copy")
        copied_bound = copy_bound_workspace(bound)
        await target.write_bytes("a.txt", b"after")
        return await binding.finalize(copied_bound, outcome="completed")

    final_snapshot = asyncio.run(run())

    assert final_snapshot is not None
    assert (source_root / "a.txt").read_text(encoding="utf-8") == "after"
    assert binding._states == {}


def test_sync_binding_rejects_concurrent_bind_on_fixed_target(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("x", encoding="utf-8")
    (target_root / "stale.txt").write_text("stale", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(target_workspace=target)

    async def run() -> None:
        await binding.bind(source, None, session_id="sess_a")
        # sess_a still holds the shared fixed target; a concurrent second bind must be rejected
        # instead of interleaving clear/copy over it.
        with pytest.raises(ValueError, match="already bound by an active session"):
            await binding.bind(source, None, session_id="sess_b")

    asyncio.run(run())
    # The reservation was taken before any mutating await, so the rejected bind never touched the
    # target, and sess_a's reservation is still held.
    assert binding._active_fixed_target_ids == {"target"}


def test_sync_binding_allows_sequential_reuse_of_fixed_target(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("x", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(target_workspace=target)

    async def run() -> None:
        bound = await binding.bind(source, None, session_id="sess_a")
        await binding.finalize(bound, outcome="completed")
        # finalize released the reservation, so a later session reuses the same target cleanly.
        await binding.bind(source, None, session_id="sess_b")

    asyncio.run(run())
    assert binding._active_fixed_target_ids == {"target"}


def test_sync_binding_abandon_releases_fixed_target(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("x", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(target_workspace=target)

    async def run() -> None:
        bound = await binding.bind(source, None, session_id="sess_a")
        binding.abandon(bound)
        # abandon released the reservation, so a re-bind of the same target succeeds.
        await binding.bind(source, None, session_id="sess_b")

    asyncio.run(run())
    assert binding._active_fixed_target_ids == {"target"}


def test_sync_binding_factory_targets_are_not_reservation_gated(tmp_path) -> None:
    # The concurrency guard applies only to a shared fixed target; factory targets are per-bind.
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "a.txt").write_text("x", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    made: list[LocalWorkspace] = []

    async def factory(context: SyncBindingContext) -> Workspace:
        root = tmp_path / f"target_{len(made)}"
        root.mkdir()
        target = LocalWorkspace(root, workspace_id=f"target_{len(made)}")
        made.append(target)
        return target

    binding = SyncBinding(target_workspace_factory=factory)

    async def run() -> None:
        await binding.bind(source, None, session_id="sess_a")
        await binding.bind(source, None, session_id="sess_b")

    asyncio.run(run())
    assert len(made) == 2
    assert binding._active_fixed_target_ids == set()


def test_sync_binding_releases_fixed_target_when_bind_fails(tmp_path) -> None:
    # A bind that fails mid-way must release its reservation (via the except path), or the fixed
    # target would be stuck rejecting every later bind after one transient failure.
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("x", encoding="utf-8")
    (target_root / "stale.txt").write_text("stale", encoding="utf-8")

    class ExplodingClearWorkspace(LocalWorkspace):
        fail_clear = True

        async def delete(self, path: str) -> None:
            if self.fail_clear:
                raise RuntimeError("clear failed")
            await super().delete(path)

    source = LocalWorkspace(source_root, workspace_id="source")
    target = ExplodingClearWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(target_workspace=target)

    async def run() -> None:
        with pytest.raises(RuntimeError, match="clear failed"):
            await binding.bind(source, None, session_id="sess_fail")
        # The failed bind released its reservation, so a retry can bind the same target.
        assert binding._active_fixed_target_ids == set()
        target.fail_clear = False
        await binding.bind(source, None, session_id="sess_retry")
        assert binding._active_fixed_target_ids == {"target"}

    asyncio.run(run())


def test_sync_binding_keeps_state_when_finalize_fails(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "removed.txt").write_text("delete me", encoding="utf-8")

    class FlakyDeleteWorkspace(LocalWorkspace):
        fail_delete = True

        async def delete(self, path: str) -> None:
            if self.fail_delete and path == "removed.txt":
                raise RuntimeError("delete failed")
            await super().delete(path)

    source = FlakyDeleteWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(target_workspace=target)

    async def run() -> None:
        bound = await binding.bind(source, None, session_id="sess_sync_retry")
        await target.delete("removed.txt")
        with pytest.raises(RuntimeError, match="delete failed"):
            await binding.finalize(bound, outcome="completed")
        assert len(binding._states) == 1
        source.fail_delete = False
        await binding.finalize(bound, outcome="completed")

    asyncio.run(run())

    assert binding._states == {}
    assert not (source_root / "removed.txt").exists()


def test_sync_binding_respects_sync_back_and_delete_options(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "keep.txt").write_text("source", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(
        target_workspace=target,
        sync_back="on_success",
        delete_missing=False,
    )

    async def run() -> WorkspaceSnapshot | None:
        bound = await binding.bind(source, None, session_id="sess_sync_policy")
        assert len(binding._states) == 1
        await target.delete("keep.txt")
        return await binding.finalize(bound, outcome="failed")

    final_snapshot = asyncio.run(run())

    assert final_snapshot is None
    assert binding._states == {}
    assert (source_root / "keep.txt").read_text(encoding="utf-8") == "source"


def test_sync_binding_clean_target_never_does_not_sync_target_baseline_files(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "shared.txt").write_text("source value", encoding="utf-8")
    (target_root / "cache.txt").write_text("target cache", encoding="utf-8")
    (target_root / "shared.txt").write_text("old target value", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")

    async def run() -> WorkspaceSnapshot | None:
        binding = SyncBinding(target_workspace=target, clean_target="never")
        bound = await binding.bind(source, None, session_id="sess_sync_baseline")
        await target.write_bytes("cache.txt", b"mutated cache")
        await target.write_bytes("shared.txt", b"updated shared")
        await target.write_bytes("created.txt", b"created during run")
        return await binding.finalize(bound, outcome="completed")

    final_snapshot = asyncio.run(run())

    assert final_snapshot is not None
    assert final_snapshot.metadata["copied_files"] == 2
    assert not (source_root / "cache.txt").exists()
    assert (source_root / "shared.txt").read_text(encoding="utf-8") == "updated shared"
    assert (source_root / "created.txt").read_text(encoding="utf-8") == "created during run"


def test_sync_binding_rejects_truncated_file_copy(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "large.txt").write_bytes(b"abcdef")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")

    async def run() -> None:
        await SyncBinding(target_workspace=target, max_file_bytes=3).bind(
            source,
            None,
            session_id="sess_sync_limit",
        )

    with pytest.raises(RuntimeError, match="large.txt"):
        asyncio.run(run())


class _CountingLocalRunner(LocalRunner):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.exec_calls = 0

    async def exec(self, *args: Any, **kwargs: Any) -> ExecResult:
        self.exec_calls += 1
        return await super().exec(*args, **kwargs)


def test_sync_binding_bulk_transfers_runner_workspace_files(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("alpha", encoding="utf-8")
    (source_root / "b.txt").write_text("bravo", encoding="utf-8")
    (source_root / "nested").mkdir()
    (source_root / "nested" / "c.txt").write_text("charlie", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    runner = _CountingLocalRunner(target_root, inherit_env=False)
    target = RunnerWorkspace(
        runner,
        workspace_id="target",
        python_executable=sys.executable,
    )
    binding = SyncBinding(target_workspace=target)

    async def run() -> tuple[int, WorkspaceSnapshot | None]:
        bound = await binding.bind(source, None, session_id="sess_bulk")
        bind_execs = runner.exec_calls
        await target.write_bytes("a.txt", b"changed")
        await target.write_bytes("new.txt", b"created")
        final_snapshot = await binding.finalize(bound, outcome="completed")
        return bind_execs, final_snapshot

    bind_execs, final_snapshot = asyncio.run(run())

    # Bind costs one exec to list the target for cleaning plus one bulk tar
    # write, independent of how many files are copied in.
    assert bind_execs == 2
    # Two manual writes plus finalize's list + bulk tar read.
    assert runner.exec_calls == 6
    assert (target_root / "nested" / "c.txt").read_text(encoding="utf-8") == "charlie"
    assert (source_root / "a.txt").read_text(encoding="utf-8") == "changed"
    assert (source_root / "new.txt").read_text(encoding="utf-8") == "created"
    assert final_snapshot is not None
    assert final_snapshot.metadata["copied_files"] == 4
    assert final_snapshot.metadata["copied_bytes"] == len("changed") + len("created") + len(
        "bravo"
    ) + len("charlie")
    assert binding._states == {}


def test_sync_binding_bulk_transfer_respects_max_file_bytes(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "big.txt").write_text("too large", encoding="utf-8")
    source = RunnerWorkspace(
        LocalRunner(source_root, inherit_env=False),
        workspace_id="source",
        python_executable=sys.executable,
    )
    target = LocalWorkspace(target_root, workspace_id="target")

    async def run() -> None:
        await SyncBinding(target_workspace=target, max_file_bytes=3).bind(
            source,
            None,
            session_id="sess_bulk_limit",
        )

    with pytest.raises(RuntimeError, match="exceeds max_file_bytes=3"):
        asyncio.run(run())


def test_sync_binding_abandon_releases_state_without_syncing(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("before", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(target_workspace=target)

    async def run() -> None:
        bound = await binding.bind(source, None, session_id="sess_abandon")
        assert len(binding._states) == 1
        binding.abandon(bound)
        assert binding._states == {}
        with pytest.raises(ValueError, match="in-process bind state"):
            await binding.finalize(bound, outcome="completed")

    asyncio.run(run())

    invalid_bound: Any = object()
    with pytest.raises(TypeError, match="BoundWorkspace"):
        binding.abandon(invalid_bound)
    assert (source_root / "a.txt").read_text(encoding="utf-8") == "before"


def test_sync_binding_rebind_replaces_leaked_state_for_same_session(tmp_path) -> None:
    # Concurrent sessions must use a factory (a shared fixed target rejects overlapping binds), so
    # the leaked-state scenario is exercised through per-session factory targets.
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "a.txt").write_text("before", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    made: list[LocalWorkspace] = []

    async def factory(context: SyncBindingContext) -> Workspace:
        root = tmp_path / f"target_{len(made)}"
        root.mkdir()
        target = LocalWorkspace(root, workspace_id=f"target_{len(made)}")
        made.append(target)
        return target

    binding = SyncBinding(target_workspace_factory=factory)

    async def run() -> None:
        first = await binding.bind(source, None, session_id="sess_leak")
        await binding.bind(source, None, session_id="sess_other")
        rebound = await binding.bind(source, None, session_id="sess_leak")
        assert len(binding._states) == 2
        assert first.state_key not in binding._states
        assert rebound.state_key in binding._states
        with pytest.raises(ValueError, match="in-process bind state"):
            await binding.finalize(first, outcome="completed")

    asyncio.run(run())


def test_sync_binding_same_session_rebind_reuses_fixed_target(tmp_path) -> None:
    # A same-session rebind of a fixed target is not a concurrent bind: the prior state is pruned
    # (releasing its reservation) before the new bind re-reserves the target.
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("before", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(target_workspace=target)

    async def run() -> None:
        first = await binding.bind(source, None, session_id="sess_x")
        rebound = await binding.bind(source, None, session_id="sess_x")
        assert len(binding._states) == 1
        assert first.state_key not in binding._states
        assert rebound.state_key in binding._states

    asyncio.run(run())
    assert binding._active_fixed_target_ids == {"target"}


def test_sync_binding_prunes_expired_states_on_bind(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("before", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(target_workspace=target, state_ttl_s=60)

    async def run() -> None:
        stale = await binding.bind(source, None, session_id="sess_stale")
        assert stale.state_key is not None
        binding._states[stale.state_key] = replace(
            binding._states[stale.state_key],
            created_at=time.monotonic() - 120.0,
        )
        fresh = await binding.bind(source, None, session_id="sess_fresh")
        assert fresh.state_key is not None
        assert set(binding._states) == {fresh.state_key}
        assert binding._states[fresh.state_key].session_id == "sess_fresh"

    asyncio.run(run())


def test_sync_binding_rejects_reserved_metadata_key(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")

    async def run() -> None:
        await SyncBinding(target_workspace=target).bind(
            source,
            None,
            session_id="sess_sync_reserved_metadata",
            metadata={"sync_binding": {"caller": "value"}},
        )

    with pytest.raises(ValueError, match="reserved"):
        asyncio.run(run())


def test_sync_binding_rejects_reserved_finalize_metadata_key(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(target_workspace=target)

    async def run() -> None:
        bound = await binding.bind(source, None, session_id="sess_sync_reserved_finalize")
        with pytest.raises(ValueError, match="reserved"):
            await binding.finalize(
                bound,
                outcome="completed",
                metadata={"copied_files": 999},
            )

    asyncio.run(run())


def test_binding_finalize_rejects_invalid_values() -> None:
    invalid_bound: Any = object()
    invalid_metadata: Any = []

    async def run() -> None:
        binding = NativeBinding()

        with pytest.raises(TypeError, match="BoundWorkspace"):
            await binding.finalize(invalid_bound)
        with pytest.raises(ValueError, match="outcome"):
            await binding.finalize(BoundWorkspace(), outcome=" ")
        with pytest.raises(TypeError, match="metadata"):
            await binding.finalize(BoundWorkspace(), metadata=invalid_metadata)
        with pytest.raises(ValueError, match="metadata"):
            await binding.finalize(BoundWorkspace(), metadata={"bad": object()})

    asyncio.run(run())


def test_bound_workspace_validates_shape_and_copies_metadata() -> None:
    workspace = StubWorkspace()
    runner = StubRunner()
    metadata = {"nested": {"value": 1}}
    snapshot = WorkspaceSnapshot(
        snapshot_id="snap_1",
        workspace_id=workspace.id,
        version="v1",
        source="git",
        metadata={"branch": "main"},
    )

    bound = BoundWorkspace(
        workspace=workspace,
        runner=runner,
        path="/workspace",
        metadata=metadata,
        snapshot=snapshot,
    )

    metadata["nested"]["value"] = 2
    snapshot.metadata["branch"] = "dev"
    assert bound.workspace is workspace
    assert bound.source_workspace is None
    assert bound.runner is runner
    assert bound.path == "/workspace"
    assert bound.metadata == {"nested": {"value": 1}}
    assert bound.snapshot is not snapshot
    assert bound.snapshot is not None
    assert bound.snapshot.snapshot_id == snapshot.snapshot_id
    assert bound.snapshot.metadata == {"branch": "main"}

    with pytest.raises(FrozenInstanceError):
        bound.__setattr__("path", "/other")


def test_bound_workspace_rejects_invalid_values() -> None:
    invalid_workspace: Any = object()
    invalid_runner: Any = object()
    invalid_path: Any = 123
    invalid_metadata: Any = []
    invalid_snapshot: Any = object()

    with pytest.raises(TypeError, match="workspace"):
        BoundWorkspace(workspace=invalid_workspace)
    with pytest.raises(TypeError, match="source_workspace"):
        BoundWorkspace(source_workspace=invalid_workspace)
    with pytest.raises(TypeError, match="runner"):
        BoundWorkspace(runner=invalid_runner)
    with pytest.raises(TypeError, match="path"):
        BoundWorkspace(path=invalid_path)
    with pytest.raises(ValueError, match="state_key"):
        BoundWorkspace(state_key=" ")
    with pytest.raises(ValueError, match="path"):
        BoundWorkspace(path=" ")
    with pytest.raises(TypeError, match="metadata"):
        BoundWorkspace(metadata=invalid_metadata)
    with pytest.raises(ValueError, match="metadata"):
        BoundWorkspace(metadata={"bad": object()})
    with pytest.raises(TypeError, match="snapshot"):
        BoundWorkspace(snapshot=invalid_snapshot)


def test_workspace_snapshot_validates_shape_and_copies_metadata() -> None:
    metadata = {"nested": {"value": 1}}

    snapshot = WorkspaceSnapshot(
        snapshot_id="snap_1",
        workspace_id="workspace_1",
        version="v1",
        source="git",
        metadata=metadata,
    )

    metadata["nested"]["value"] = 2
    assert snapshot.snapshot_id == "snap_1"
    assert snapshot.workspace_id == "workspace_1"
    assert snapshot.version == "v1"
    assert snapshot.source == "git"
    assert snapshot.metadata == {"nested": {"value": 1}}

    with pytest.raises(FrozenInstanceError):
        snapshot.__setattr__("version", "v2")


def test_workspace_snapshot_rejects_invalid_values() -> None:
    invalid_metadata: Any = []

    with pytest.raises(ValueError, match="snapshot_id"):
        WorkspaceSnapshot(snapshot_id=" ")
    with pytest.raises(ValueError, match="workspace_id"):
        WorkspaceSnapshot(snapshot_id="snap_1", workspace_id=" ")
    with pytest.raises(ValueError, match="version"):
        WorkspaceSnapshot(snapshot_id="snap_1", version=" ")
    with pytest.raises(ValueError, match="source"):
        WorkspaceSnapshot(snapshot_id="snap_1", source=" ")
    with pytest.raises(TypeError, match="metadata"):
        WorkspaceSnapshot(snapshot_id="snap_1", metadata=invalid_metadata)
    with pytest.raises(ValueError, match="metadata"):
        WorkspaceSnapshot(snapshot_id="snap_1", metadata={"bad": object()})


def test_binding_constructors_validate_values() -> None:
    invalid_path: Any = 123
    invalid_clean_target: Any = "sometimes"
    invalid_sync_back: Any = "sometimes"
    invalid_delete_missing: Any = "yes"

    with pytest.raises(TypeError, match="default_path"):
        NativeBinding(default_path=invalid_path)
    with pytest.raises(ValueError, match="default_path"):
        NativeBinding(default_path=" ")
    with pytest.raises(TypeError, match="target_workspace"):
        SyncBinding(target_workspace=invalid_path)
    with pytest.raises(TypeError, match="target_workspace_factory"):
        SyncBinding(target_workspace_factory=invalid_path)
    with pytest.raises(ValueError, match="either target_workspace or target_workspace_factory"):
        SyncBinding(
            target_workspace=StubWorkspace(), target_workspace_factory=lambda _ctx: StubWorkspace()
        )
    with pytest.raises(ValueError, match="path"):
        SyncBinding(path=" ")
    with pytest.raises(ValueError, match="max_files"):
        SyncBinding(max_files=0)
    with pytest.raises(ValueError, match="clean_target"):
        SyncBinding(clean_target=invalid_clean_target)
    with pytest.raises(ValueError, match="sync_back"):
        SyncBinding(sync_back=invalid_sync_back)
    with pytest.raises(TypeError, match="delete_missing"):
        SyncBinding(delete_missing=invalid_delete_missing)
    with pytest.raises(ValueError, match="state_ttl_s"):
        SyncBinding(state_ttl_s=0)
    with pytest.raises(TypeError, match="state_ttl_s"):
        SyncBinding(state_ttl_s=invalid_delete_missing)


def test_copy_bound_workspace_defensively_copies_metadata_and_snapshot() -> None:
    bound = BoundWorkspace(
        metadata={"token": {"cursor": "a"}},
        snapshot=WorkspaceSnapshot(
            snapshot_id="snap_1",
            metadata={"nested": {"value": 1}},
        ),
    )

    copied = copy_bound_workspace(bound)
    bound.metadata["token"]["cursor"] = "b"
    assert bound.snapshot is not None
    bound.snapshot.metadata["nested"]["value"] = 2

    assert copied is not bound
    assert copied.metadata == {"token": {"cursor": "a"}}
    assert copied.snapshot is not None
    assert copied.snapshot.metadata == {"nested": {"value": 1}}


def test_copy_workspace_snapshot_defensively_copies_metadata() -> None:
    snapshot = WorkspaceSnapshot(snapshot_id="snap_1", metadata={"token": {"cursor": "a"}})

    copied = copy_workspace_snapshot(snapshot)
    snapshot.metadata["token"]["cursor"] = "b"

    assert copied is not snapshot
    assert copied is not None
    assert copied.metadata == {"token": {"cursor": "a"}}
    assert copy_workspace_snapshot(None) is None


def test_copy_bound_workspace_rejects_invalid_value() -> None:
    invalid_bound: Any = object()

    with pytest.raises(TypeError, match="BoundWorkspace"):
        copy_bound_workspace(invalid_bound)


def test_copy_workspace_snapshot_rejects_invalid_value() -> None:
    invalid_snapshot: Any = object()

    with pytest.raises(TypeError, match="WorkspaceSnapshot"):
        copy_workspace_snapshot(invalid_snapshot)


def test_workspace_binding_is_abstract() -> None:
    abstract_cls: Any = WorkspaceBinding

    with pytest.raises(TypeError):
        abstract_cls()
