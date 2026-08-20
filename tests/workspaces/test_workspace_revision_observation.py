from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys

import pytest
from pydantic import ValidationError

from cayu.environments import (
    BoundWorkspace,
    DeterministicWorkspaceBinding,
    GitRepositoryBinding,
    NativeBinding,
)
from cayu.runners import ExecCommand, ExecResult, Runner
from cayu.workspaces import (
    LocalWorkspace,
    RunnerWorkspace,
    Workspace,
    WorkspaceDirectMutationReconciliation,
    WorkspaceForkLineage,
    WorkspaceForkLineageStatus,
    WorkspaceIdentity,
    WorkspaceListResult,
    WorkspaceMutationAttribution,
    WorkspaceMutationAttributionConfidence,
    WorkspacePathRevision,
    WorkspaceReadResult,
    WorkspaceRevisionDeltaStatus,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationLimits,
    WorkspaceRevisionObservationStatus,
    WorkspaceWriterIsolationEvidence,
    WorkspaceWriterIsolationStatus,
    compare_workspace_revisions,
)


def test_workspace_writer_isolation_defaults_to_unknown(tmp_path) -> None:
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace-1")
    binding = DeterministicWorkspaceBinding()

    async def bind():
        return await binding.bind(workspace, None, session_id="session-1")

    evidence = binding.observe_writer_isolation(asyncio.run(bind()))

    assert evidence == WorkspaceWriterIsolationEvidence()
    assert evidence.status is WorkspaceWriterIsolationStatus.UNKNOWN


def test_workspace_writer_isolation_requires_inspectable_exclusive_generation() -> None:
    with pytest.raises(ValidationError, match="requires a mechanism and generation"):
        WorkspaceWriterIsolationEvidence(
            status=WorkspaceWriterIsolationStatus.EXCLUSIVE,
            detail_code=None,
        )


def test_workspace_writer_isolation_evidence_is_text_bounded() -> None:
    with pytest.raises(ValidationError):
        WorkspaceWriterIsolationEvidence(
            status=WorkspaceWriterIsolationStatus.EXCLUSIVE,
            mechanism="m" * 257,
            generation="generation-1",
            detail_code=None,
        )


def test_workspace_attribution_rejects_false_exclusive_causality() -> None:
    with pytest.raises(ValidationError, match="requires exclusive writer isolation"):
        WorkspaceMutationAttribution(
            confidence=WorkspaceMutationAttributionConfidence.EXCLUSIVE_TOOL,
            writer_isolation=WorkspaceWriterIsolationStatus.UNKNOWN,
            direct_reconciliation=WorkspaceDirectMutationReconciliation.CONSISTENT,
            detail_code="invalid_test_claim",
        )


def test_workspace_fork_lineage_only_allows_revision_for_proven_derivation() -> None:
    shared = WorkspaceForkLineage(
        status=WorkspaceForkLineageStatus.SHARED_OR_AMBIGUOUS,
        detail_code="shared_live_workspace_not_isolated",
    )
    assert shared.source_workspace_revision is None

    with pytest.raises(ValidationError, match="Only demonstrably derived"):
        WorkspaceForkLineage(
            status=WorkspaceForkLineageStatus.SHARED_OR_AMBIGUOUS,
            source_workspace_revision="revision-1",
            detail_code="invalid_shared_revision",
        )


def test_workspace_observation_rejects_duplicate_paths() -> None:
    identity = WorkspaceIdentity(workspace_id="workspace-1", observer="custom")

    with pytest.raises(ValidationError, match="paths must be unique"):
        WorkspaceRevisionObservation(
            identity=identity,
            status=WorkspaceRevisionObservationStatus.SUPPORTED,
            revision="sha256:" + "a" * 64,
            paths=(
                WorkspacePathRevision(path="same.txt", content_sha256="a" * 64),
                WorkspacePathRevision(path="same.txt", content_sha256="b" * 64),
            ),
            total_paths=2,
        )


def test_workspace_observation_rejects_noncanonical_path_aliases() -> None:
    with pytest.raises(ValidationError, match="relative and traversal-free"):
        WorkspacePathRevision(path="directory/./same.txt")


def test_deterministic_workspace_binding_fails_closed_on_noncanonical_list_path() -> None:
    class NoncanonicalListWorkspace(Workspace):
        id = "noncanonical-workspace"

        def bounded_read_limit(self, max_bytes: int) -> int:
            return max_bytes

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            del path, offset, max_bytes
            raise AssertionError("Unsafe workspace path must be rejected before reading.")

        async def write_bytes(self, path: str, content: bytes) -> None:
            del path, content

        async def delete(self, path: str) -> None:
            del path

        async def create_bytes(self, path: str, content: bytes):
            del path, content
            raise NotImplementedError

        async def replace_bytes(self, path: str, content: bytes, *, expected_revision: str):
            del path, content, expected_revision
            raise NotImplementedError

        async def delete_if_revision(self, path: str, *, expected_revision: str):
            del path, expected_revision
            raise NotImplementedError

        async def list(
            self,
            pattern: str = "**/*",
            *,
            limit: int | None = None,
        ) -> WorkspaceListResult:
            del pattern, limit
            return WorkspaceListResult(paths=("directory/./same.txt",), total_count=1)

    workspace = NoncanonicalListWorkspace()
    binding = DeterministicWorkspaceBinding()
    observation = asyncio.run(
        binding.observe_revision(BoundWorkspace(workspace=workspace, source_workspace=workspace))
    )

    assert observation.status is WorkspaceRevisionObservationStatus.FAILED
    assert observation.detail_code == "unsafe_workspace_path"


@pytest.mark.parametrize(
    ("malformed_operation", "detail_code"),
    [
        ("list", "workspace_list_failed"),
        ("read", "workspace_file_read_failed"),
    ],
)
def test_deterministic_workspace_binding_classifies_malformed_workspace_results(
    malformed_operation,
    detail_code,
) -> None:
    class MalformedResultWorkspace(Workspace):
        id = "malformed-result-workspace"

        def bounded_read_limit(self, max_bytes: int) -> int:
            return max_bytes

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            del path, offset, max_bytes
            if malformed_operation == "read":
                return object()  # type: ignore[return-value]
            return WorkspaceReadResult(content=b"safe", total_bytes=4)

        async def write_bytes(self, path: str, content: bytes) -> None:
            del path, content

        async def delete(self, path: str) -> None:
            del path

        async def create_bytes(self, path: str, content: bytes):
            del path, content
            raise NotImplementedError

        async def replace_bytes(self, path: str, content: bytes, *, expected_revision: str):
            del path, content, expected_revision
            raise NotImplementedError

        async def delete_if_revision(self, path: str, *, expected_revision: str):
            del path, expected_revision
            raise NotImplementedError

        async def list(
            self,
            pattern: str = "**/*",
            *,
            limit: int | None = None,
        ) -> WorkspaceListResult:
            del pattern, limit
            if malformed_operation == "list":
                return object()  # type: ignore[return-value]
            return WorkspaceListResult(paths=("safe.txt",), total_count=1)

    workspace = MalformedResultWorkspace()
    binding = DeterministicWorkspaceBinding()
    observation = asyncio.run(
        binding.observe_revision(BoundWorkspace(workspace=workspace, source_workspace=workspace))
    )

    assert observation.status in {
        WorkspaceRevisionObservationStatus.FAILED,
        WorkspaceRevisionObservationStatus.INCOMPLETE,
    }
    assert observation.detail_code == detail_code
    assert observation.paths == ()


def test_workspace_binding_reports_revision_observation_as_unsupported(tmp_path) -> None:
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace-1")
    binding = NativeBinding()

    async def observe():
        bound = await binding.bind(workspace, None, session_id="session-1")
        return await binding.observe_revision(bound)

    observation = asyncio.run(observe())

    assert observation.status is WorkspaceRevisionObservationStatus.UNSUPPORTED
    assert observation.identity.workspace_id == "workspace-1"
    assert observation.identity.observer == "NativeBinding"
    assert observation.revision is None
    assert observation.paths == ()
    assert observation.total_paths == 0
    assert observation.detail_code == "revision_observation_unsupported"


def test_deterministic_workspace_binding_observes_changes_and_no_change(tmp_path) -> None:
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace-1")
    binding = DeterministicWorkspaceBinding()
    (tmp_path / "existing.txt").write_text("before\n", encoding="utf-8")

    async def observe_changes():
        bound = await binding.bind(workspace, None, session_id="session-1")
        before = await binding.observe_revision(bound)
        (tmp_path / "existing.txt").write_text("after\n", encoding="utf-8")
        (tmp_path / "added.bin").write_bytes(b"\x00\x01\x02")
        after = await binding.observe_revision(bound)
        unchanged = await binding.observe_revision(bound)
        return before, after, unchanged

    before, after, unchanged = asyncio.run(observe_changes())

    assert before.status is WorkspaceRevisionObservationStatus.SUPPORTED
    assert after.status is WorkspaceRevisionObservationStatus.SUPPORTED
    assert before.revision != after.revision
    assert [path.path for path in after.paths] == ["added.bin", "existing.txt"]

    changed = compare_workspace_revisions(before, after)
    assert changed.status is WorkspaceRevisionDeltaStatus.CHANGED
    assert [(path.path, path.change) for path in changed.paths] == [
        ("added.bin", "added"),
        ("existing.txt", "modified"),
    ]

    no_change = compare_workspace_revisions(after, unchanged)
    assert no_change.status is WorkspaceRevisionDeltaStatus.NO_CHANGE
    assert no_change.paths == ()
    assert no_change.total_paths == 0


def test_deterministic_workspace_binding_enforces_aggregate_file_byte_limit(tmp_path) -> None:
    (tmp_path / "a.txt").write_bytes(b"aaa")
    (tmp_path / "b.txt").write_bytes(b"bbb")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace-1")
    binding = DeterministicWorkspaceBinding(
        observation_limits=WorkspaceRevisionObservationLimits(
            max_file_bytes=4,
            max_total_file_bytes=5,
        )
    )

    async def observe():
        bound = await binding.bind(workspace, None, session_id="session-1")
        return await binding.observe_revision(bound)

    observation = asyncio.run(observe())

    assert observation.status is WorkspaceRevisionObservationStatus.TRUNCATED
    assert observation.detail_code == "total_file_byte_limit_exceeded"
    assert observation.paths == ()


def test_deterministic_workspace_binding_allows_empty_file_at_exact_aggregate_limit(
    tmp_path,
) -> None:
    (tmp_path / "a.txt").write_bytes(b"aaa")
    (tmp_path / "z-empty.txt").write_bytes(b"")
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace-1")
    binding = DeterministicWorkspaceBinding(
        observation_limits=WorkspaceRevisionObservationLimits(
            max_file_bytes=4,
            max_total_file_bytes=3,
        )
    )

    async def observe():
        bound = await binding.bind(workspace, None, session_id="session-1")
        return await binding.observe_revision(bound)

    observation = asyncio.run(observe())

    assert observation.status is WorkspaceRevisionObservationStatus.SUPPORTED
    assert [path.path for path in observation.paths] == ["a.txt", "z-empty.txt"]


@pytest.mark.parametrize(
    "binding_factory",
    [
        lambda limits: DeterministicWorkspaceBinding(observation_limits=limits),
        lambda limits: GitRepositoryBinding(
            repo_url="https://example.invalid/repository.git",
            fetch=False,
            observation_limits=limits,
        ),
    ],
)
def test_workspace_observation_limits_are_revalidated_at_binding_boundary(
    binding_factory,
) -> None:
    forged = WorkspaceRevisionObservationLimits().model_copy(update={"max_total_file_bytes": 0})

    with pytest.raises(ValueError):
        binding_factory(forged)


def test_mutated_workspace_observation_limits_are_revalidated_before_collection(
    tmp_path,
) -> None:
    workspace = LocalWorkspace(tmp_path, workspace_id="workspace-1")
    binding = DeterministicWorkspaceBinding()
    binding.observation_limits = WorkspaceRevisionObservationLimits().model_copy(
        update={"max_total_file_bytes": True}
    )
    bound = BoundWorkspace(workspace=workspace, source_workspace=workspace)

    with pytest.raises(TypeError, match="limit fields must be integers"):
        asyncio.run(binding.observe_revision(bound))


def test_git_binding_observes_bounded_worktree_and_index_mutations(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.rstrip()

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    git("config", "status.renames", "false")
    for name in ("tracked.txt", "deleted.txt", "rename.txt"):
        (tmp_path / name).write_text(f"{name}\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "baseline")

    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        require_clean=False,
        verify_remote_url=False,
    )

    async def observe_changes():
        bound = await binding.bind(workspace, None, session_id="session-1")
        before = await binding.observe_revision(bound)
        (tmp_path / "tracked.txt").write_text("unstaged\n", encoding="utf-8")
        (tmp_path / "staged.txt").write_text("staged\n", encoding="utf-8")
        git("add", "staged.txt")
        (tmp_path / "untracked.bin").write_bytes(b"\x00secret-binary\xff")
        (tmp_path / "deleted.txt").unlink()
        git("mv", "rename.txt", "renamed.txt")
        index_before_observation = (tmp_path / ".git" / "index").read_bytes()
        after = await binding.observe_revision(bound)
        index_after_observation = (tmp_path / ".git" / "index").read_bytes()
        return before, after, index_before_observation, index_after_observation

    before, after, index_before_observation, index_after_observation = asyncio.run(
        observe_changes()
    )
    delta = compare_workspace_revisions(before, after)

    assert before.status is WorkspaceRevisionObservationStatus.SUPPORTED
    assert after.status is WorkspaceRevisionObservationStatus.SUPPORTED
    assert [(path.path, path.change, path.renamed_from) for path in delta.paths] == [
        ("deleted.txt", "deleted", None),
        ("renamed.txt", "renamed", "rename.txt"),
        ("staged.txt", "added", None),
        ("tracked.txt", "modified", None),
        ("untracked.bin", "added", None),
    ]
    assert "secret-binary" not in after.model_dump_json()
    assert index_after_observation == index_before_observation
    assert git("status", "--porcelain=v1") == (
        " D deleted.txt\n"
        "D  rename.txt\n"
        "A  renamed.txt\n"
        "A  staged.txt\n"
        " M tracked.txt\n"
        "?? untracked.bin"
    )


def test_git_binding_observes_dirty_baseline_and_head_branch_movement(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.rstrip()

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    (tmp_path / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "baseline")
    (tmp_path / "tracked.txt").write_text("first dirty state\n", encoding="utf-8")

    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        require_clean=False,
        verify_remote_url=False,
    )

    async def observe_changes():
        bound = await binding.bind(workspace, None, session_id="session-1")
        dirty_before = await binding.observe_revision(bound)
        (tmp_path / "tracked.txt").write_text("second dirty state\n", encoding="utf-8")
        dirty_after = await binding.observe_revision(bound)
        git("checkout", "-b", "receipt-branch")
        git("add", "tracked.txt")
        git("commit", "-m", "move head")
        moved = await binding.observe_revision(bound)
        return dirty_before, dirty_after, moved

    dirty_before, dirty_after, moved = asyncio.run(observe_changes())

    dirty_delta = compare_workspace_revisions(dirty_before, dirty_after)
    assert dirty_delta.status is WorkspaceRevisionDeltaStatus.CHANGED
    assert [(path.path, path.change) for path in dirty_delta.paths] == [("tracked.txt", "modified")]

    moved_delta = compare_workspace_revisions(dirty_after, moved)
    assert moved_delta.status is WorkspaceRevisionDeltaStatus.CHANGED
    assert moved_delta.head_changed is True
    assert moved_delta.branch_changed is True
    assert [(path.path, path.change) for path in moved_delta.paths] == [("tracked.txt", "modified")]
    assert moved.branch == "receipt-branch"


def test_git_binding_attributes_clean_to_committed_tracked_mutation(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    (tmp_path / "tracked.txt").write_text("before\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "baseline")

    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        require_clean=False,
        verify_remote_url=False,
    )

    async def observe_changes():
        bound = await binding.bind(workspace, None, session_id="session-1")
        before = await binding.observe_revision(bound)
        (tmp_path / "tracked.txt").write_text("after\n", encoding="utf-8")
        git("add", "tracked.txt")
        git("commit", "-m", "tool commit")
        after = await binding.observe_revision(bound)
        return before, after

    before, after = asyncio.run(observe_changes())
    delta = compare_workspace_revisions(before, after)

    assert before.status is WorkspaceRevisionObservationStatus.SUPPORTED
    assert after.status is WorkspaceRevisionObservationStatus.SUPPORTED
    assert delta.status is WorkspaceRevisionDeltaStatus.CHANGED
    assert delta.head_changed is True
    assert [(path.path, path.change) for path in delta.paths] == [("tracked.txt", "modified")]


def test_git_binding_attributes_committed_executable_bit_change(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    git("config", "core.fileMode", "true")
    script = tmp_path / "script.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o644)
    git("add", "script.sh")
    git("commit", "-m", "baseline")

    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        require_clean=False,
        verify_remote_url=False,
    )

    async def observe_changes():
        bound = await binding.bind(workspace, None, session_id="session-1")
        before = await binding.observe_revision(bound)
        git("update-index", "--chmod=+x", "script.sh")
        git("commit", "-m", "make executable")
        after = await binding.observe_revision(bound)
        return before, after

    before, after = asyncio.run(observe_changes())
    delta = compare_workspace_revisions(before, after)

    assert next(path for path in before.paths if path.path == "script.sh").index_mode == "100644"
    assert next(path for path in after.paths if path.path == "script.sh").index_mode == "100755"
    assert [(path.path, path.change) for path in delta.paths] == [("script.sh", "modified")]


@pytest.mark.skipif(sys.platform == "win32", reason="Win32 does not expose POSIX executable bits")
def test_git_binding_ignores_core_filemode_false_when_observing_worktree(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    script = tmp_path / "script.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o644)
    git("add", "script.sh")
    git("commit", "-m", "baseline")
    git("config", "core.fileMode", "false")

    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        require_clean=False,
        verify_remote_url=False,
    )

    async def observe_changes():
        bound = await binding.bind(workspace, None, session_id="session-1")
        before = await binding.observe_revision(bound)
        script.chmod(0o755)
        assert git("status", "--porcelain=v1") == ""
        after = await binding.observe_revision(bound)
        return before, after

    before, after = asyncio.run(observe_changes())
    delta = compare_workspace_revisions(before, after)

    assert after.status is WorkspaceRevisionObservationStatus.SUPPORTED
    assert next(path for path in after.paths if path.path == "script.sh").working_tree == "M"
    assert [(path.path, path.change) for path in delta.paths] == [("script.sh", "modified")]


def test_git_binding_attributes_clean_to_committed_rename(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    (tmp_path / "before.txt").write_text("same content\n", encoding="utf-8")
    git("add", "before.txt")
    git("commit", "-m", "baseline")

    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        require_clean=False,
        verify_remote_url=False,
    )

    async def observe_changes():
        bound = await binding.bind(workspace, None, session_id="session-1")
        before = await binding.observe_revision(bound)
        git("mv", "before.txt", "after.txt")
        git("commit", "-m", "rename")
        after = await binding.observe_revision(bound)
        return before, after

    before, after = asyncio.run(observe_changes())
    delta = compare_workspace_revisions(before, after)

    assert delta.status is WorkspaceRevisionDeltaStatus.CHANGED
    assert [(path.path, path.change, path.renamed_from) for path in delta.paths] == [
        ("after.txt", "renamed", "before.txt")
    ]


def test_git_binding_does_not_reuse_explicit_rename_source_for_identical_addition(
    tmp_path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    (tmp_path / "before.txt").write_text("same content\n", encoding="utf-8")
    git("add", "before.txt")
    git("commit", "-m", "baseline")

    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        require_clean=False,
        verify_remote_url=False,
    )

    async def observe_changes():
        bound = await binding.bind(workspace, None, session_id="session-1")
        before = await binding.observe_revision(bound)
        git("mv", "before.txt", "renamed.txt")
        (tmp_path / "added.txt").write_text("same content\n", encoding="utf-8")
        git("add", "added.txt")
        after = await binding.observe_revision(bound)
        return before, after

    before, after = asyncio.run(observe_changes())
    delta = compare_workspace_revisions(before, after)

    assert {path.path for path in delta.paths} == {"added.txt", "renamed.txt"}
    assert sorted(path.change for path in delta.paths) == ["added", "renamed"]
    renamed = next(path for path in delta.paths if path.change == "renamed")
    added = next(path for path in delta.paths if path.change == "added")
    assert renamed.renamed_from == "before.txt"
    assert added.renamed_from is None


def test_git_binding_distinguishes_restore_from_delete_and_existing_rename_edit(
    tmp_path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.rstrip()

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    (tmp_path / "restore.txt").write_text("baseline\n", encoding="utf-8")
    (tmp_path / "rename.txt").write_text("rename baseline\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "baseline")
    (tmp_path / "restore.txt").write_text("dirty\n", encoding="utf-8")
    git("mv", "rename.txt", "renamed.txt")

    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        require_clean=False,
        verify_remote_url=False,
    )

    async def observe_changes():
        bound = await binding.bind(workspace, None, session_id="session-1")
        before = await binding.observe_revision(bound)
        git("checkout", "--", "restore.txt")
        (tmp_path / "renamed.txt").write_text("edited after rename\n", encoding="utf-8")
        after = await binding.observe_revision(bound)
        return before, after

    before, after = asyncio.run(observe_changes())
    delta = compare_workspace_revisions(before, after)

    assert delta.status is WorkspaceRevisionDeltaStatus.CHANGED
    assert [(path.path, path.change) for path in delta.paths] == [
        ("renamed.txt", "modified"),
        ("restore.txt", "modified"),
    ]


def test_git_binding_excludes_ignored_tree_from_observation_limits(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.rstrip()

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    (tmp_path / ".gitignore").write_text("ignored.log\ncache/\n", encoding="utf-8")
    (tmp_path / "deleted.txt").write_text("delete me\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "baseline")

    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        require_clean=False,
        verify_remote_url=False,
        observation_limits=WorkspaceRevisionObservationLimits(max_paths=3),
    )

    async def observe_changes():
        bound = await binding.bind(workspace, None, session_id="session-1")
        before = await binding.observe_revision(bound)
        (tmp_path / "deleted.txt").unlink()
        git("add", "deleted.txt")
        (tmp_path / "visible.txt").write_text("visible and attributable\n", encoding="utf-8")
        (tmp_path / "ignored.log").write_text("ignored but attributable\n", encoding="utf-8")
        (tmp_path / "cache").mkdir()
        (tmp_path / "cache" / "item.txt").write_text("ignored directory file\n", encoding="utf-8")
        after = await binding.observe_revision(bound)
        (tmp_path / "ignored.log").unlink()
        (tmp_path / "cache" / "item.txt").unlink()
        removed = await binding.observe_revision(bound)
        return before, after, removed

    before, after, removed = asyncio.run(observe_changes())
    delta = compare_workspace_revisions(before, after)
    removed_delta = compare_workspace_revisions(after, removed)

    assert after.status is WorkspaceRevisionObservationStatus.SUPPORTED
    assert [(path.path, path.change) for path in delta.paths] == [
        ("deleted.txt", "deleted"),
        ("visible.txt", "added"),
    ]
    assert all(path.ignored is False for path in after.paths)
    assert {path.path for path in after.paths} == {
        ".gitignore",
        "deleted.txt",
        "visible.txt",
    }
    assert removed_delta.status is WorkspaceRevisionDeltaStatus.NO_CHANGE
    assert removed_delta.paths == ()


def test_git_binding_disables_repository_fsmonitor_during_observation(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.rstrip()

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    (tmp_path / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "baseline")
    fsmonitor = tmp_path / "fsmonitor.sh"
    fsmonitor.write_text("#!/bin/sh\ntouch fsmonitor-canary\nprintf '{}\\n'\n", encoding="utf-8")
    fsmonitor.chmod(0o755)
    git("config", "core.fsmonitor", "./fsmonitor.sh")

    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    bound = BoundWorkspace(workspace=workspace, source_workspace=workspace)
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        verify_remote_url=False,
    )

    observation = asyncio.run(binding.observe_revision(bound))

    assert observation.status is WorkspaceRevisionObservationStatus.SUPPORTED
    assert not (tmp_path / "fsmonitor-canary").exists()


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_git_binding_fails_closed_when_index_flags_hide_worktree_changes(
    tmp_path,
    index_flag,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "baseline")
    git("update-index", index_flag, "tracked.txt")
    tracked.write_text("hidden mutation\n", encoding="utf-8")
    assert git("status", "--porcelain=v1") == ""

    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        verify_remote_url=False,
    )

    observation = asyncio.run(
        binding.observe_revision(BoundWorkspace(workspace=workspace, source_workspace=workspace))
    )

    assert observation.status is WorkspaceRevisionObservationStatus.INCOMPLETE
    assert observation.detail_code == "git_index_visibility_flags_unsupported"
    assert observation.total_paths == 1


@pytest.mark.parametrize("trace_variable", ["GIT_TRACE", "GIT_TRACE_REFS", "GIT_TRACE2_EVENT"])
def test_git_binding_disables_ambient_trace_files_during_observation(
    tmp_path,
    monkeypatch,
    trace_variable,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    trace_path = tmp_path / "observer-trace.log"
    monkeypatch.setenv(trace_variable, str(trace_path))
    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    bound = BoundWorkspace(workspace=workspace, source_workspace=workspace)
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        verify_remote_url=False,
    )

    observation = asyncio.run(binding.observe_revision(bound))

    assert observation.status is WorkspaceRevisionObservationStatus.SUPPORTED
    assert not trace_path.exists()


def test_git_binding_enforces_aggregate_worktree_byte_limit(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "a.txt").write_bytes(b"aaa")
    (tmp_path / "b.txt").write_bytes(b"bbb")
    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    bound = BoundWorkspace(workspace=workspace, source_workspace=workspace)
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        verify_remote_url=False,
        observation_limits=WorkspaceRevisionObservationLimits(
            max_file_bytes=4,
            max_total_file_bytes=5,
        ),
    )

    observation = asyncio.run(binding.observe_revision(bound))

    assert observation.status is WorkspaceRevisionObservationStatus.TRUNCATED
    assert observation.detail_code == "total_file_byte_limit_exceeded"


def test_git_binding_allows_empty_file_at_exact_aggregate_limit(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "a.txt").write_bytes(b"aaa")
    (tmp_path / "z-empty.txt").write_bytes(b"")
    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    bound = BoundWorkspace(workspace=workspace, source_workspace=workspace)
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        verify_remote_url=False,
        observation_limits=WorkspaceRevisionObservationLimits(
            max_file_bytes=4,
            max_total_file_bytes=3,
        ),
    )

    observation = asyncio.run(binding.observe_revision(bound))

    assert observation.status is WorkspaceRevisionObservationStatus.SUPPORTED
    assert [path.path for path in observation.paths] == ["a.txt", "z-empty.txt"]


def test_git_binding_marks_dirty_tracked_symlink_observation_incomplete(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")
    try:
        (tmp_path / "target-one").write_text("one", encoding="utf-8")
        (tmp_path / "link").symlink_to("target-one")
    except OSError:
        pytest.skip("symlinks are unavailable")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    git("add", "link", "target-one")
    git("commit", "-m", "baseline")
    (tmp_path / "target-two").write_text("two", encoding="utf-8")
    (tmp_path / "link").unlink()
    (tmp_path / "link").symlink_to("target-two")

    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    bound = BoundWorkspace(workspace=workspace, source_workspace=workspace)
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        verify_remote_url=False,
    )

    observation = asyncio.run(binding.observe_revision(bound))

    assert observation.status is WorkspaceRevisionObservationStatus.INCOMPLETE
    assert observation.detail_code == "git_worktree_special_path_unsupported"


def test_git_binding_observes_repository_without_initial_commit_and_enforces_limits(
    tmp_path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    bound = BoundWorkspace(workspace=workspace, source_workspace=workspace)

    async def observe():
        complete = await GitRepositoryBinding(
            repo_url="https://example.invalid/repository.git",
            fetch=False,
            verify_remote_url=False,
        ).observe_revision(bound)
        truncated = await GitRepositoryBinding(
            repo_url="https://example.invalid/repository.git",
            fetch=False,
            verify_remote_url=False,
            observation_limits=WorkspaceRevisionObservationLimits(max_paths=1),
        ).observe_revision(bound)
        file_limited = await GitRepositoryBinding(
            repo_url="https://example.invalid/repository.git",
            fetch=False,
            verify_remote_url=False,
            observation_limits=WorkspaceRevisionObservationLimits(max_file_bytes=2),
        ).observe_revision(bound)
        path_limited = await GitRepositoryBinding(
            repo_url="https://example.invalid/repository.git",
            fetch=False,
            verify_remote_url=False,
            observation_limits=WorkspaceRevisionObservationLimits(max_path_bytes=2),
        ).observe_revision(bound)
        return complete, truncated, file_limited, path_limited

    complete, truncated, file_limited, path_limited = asyncio.run(observe())

    assert complete.status is WorkspaceRevisionObservationStatus.SUPPORTED
    assert complete.head_revision is None
    assert complete.branch is not None
    assert [path.path for path in complete.paths] == ["one.txt", "two.txt"]
    assert truncated.status is WorkspaceRevisionObservationStatus.TRUNCATED
    assert truncated.detail_code == "path_count_limit_exceeded"
    assert truncated.total_paths == 2
    assert file_limited.status is WorkspaceRevisionObservationStatus.TRUNCATED
    assert file_limited.detail_code == "file_byte_limit_exceeded"
    assert file_limited.total_paths == 2
    assert path_limited.status is WorkspaceRevisionObservationStatus.TRUNCATED
    assert path_limited.detail_code == "path_byte_limit_exceeded"
    assert path_limited.total_paths == 2


@pytest.mark.parametrize(
    ("failing_command", "expected_message"),
    [
        ("head", "Git HEAD identity could not be read."),
        ("branch", "Git branch identity could not be read."),
    ],
)
def test_git_binding_does_not_treat_ambiguous_exit_one_as_snapshot_identity(
    failing_command,
    expected_message,
) -> None:
    class AmbiguousSnapshotRunner(Runner):
        default_cwd = "/workspace"

        async def exec(
            self,
            command: ExecCommand,
            *,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            env_remove: tuple[str, ...] = (),
            timeout_s: int | None = None,
            stdin: str | None = None,
            output_limit_bytes: int | None = None,
        ) -> ExecResult:
            del cwd, env, env_remove, timeout_s, stdin, output_limit_bytes
            argv = command.argv or ()
            if "--is-inside-work-tree" in argv:
                return ExecResult(stdout="true\n")
            if "status" in argv:
                return ExecResult(stdout="")
            if "--verify" in argv and "HEAD" in argv:
                if failing_command == "head":
                    return ExecResult(exit_code=1, stderr="PRIVATE_GIT_FAILURE_CANARY")
                return ExecResult(stdout="a" * 40 + "\n")
            if "symbolic-ref" in argv:
                if failing_command == "branch":
                    return ExecResult(exit_code=1, stderr="PRIVATE_GIT_FAILURE_CANARY")
                return ExecResult(stdout="main\n")
            raise AssertionError(f"Unexpected Git command: {argv!r}")

    runner = AmbiguousSnapshotRunner()
    workspace = RunnerWorkspace(
        runner,
        cwd=None,
        workspace_id="runner-workspace",
    )
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        verify_remote_url=False,
    )

    with pytest.raises(RuntimeError, match=expected_message) as raised:
        asyncio.run(binding.bind(workspace, runner, session_id="session-1"))

    assert "PRIVATE_GIT_FAILURE_CANARY" not in str(raised.value)


@pytest.mark.parametrize(
    ("failing_command", "detail_code"),
    [
        ("rev-parse", "git_head_observation_failed"),
        ("symbolic-ref", "git_branch_observation_failed"),
    ],
)
def test_git_binding_rejects_failed_head_or_branch_observation(
    failing_command, detail_code
) -> None:
    class FailedAuthorityRunner(Runner):
        default_cwd = "/workspace"

        async def exec(
            self,
            command: ExecCommand,
            *,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            env_remove: tuple[str, ...] = (),
            timeout_s: int | None = None,
            stdin: str | None = None,
            output_limit_bytes: int | None = None,
        ) -> ExecResult:
            del cwd, env, env_remove, timeout_s, stdin, output_limit_bytes
            argv = command.argv or ()
            if "--is-inside-work-tree" in argv:
                return ExecResult(stdout="true\n")
            if failing_command in argv:
                return ExecResult(exit_code=128, stderr="authority unavailable")
            if "rev-parse" in argv:
                return ExecResult(stdout="a" * 40 + "\n")
            if "symbolic-ref" in argv:
                return ExecResult(stdout="main\n")
            raise AssertionError(f"Unexpected Git command: {argv!r}")

    workspace = RunnerWorkspace(
        FailedAuthorityRunner(),
        cwd=None,
        workspace_id="runner-workspace",
    )
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        verify_remote_url=False,
    )

    observation = asyncio.run(
        binding.observe_revision(BoundWorkspace(workspace=workspace, source_workspace=workspace))
    )

    assert observation.status is WorkspaceRevisionObservationStatus.FAILED
    assert observation.detail_code == detail_code


def test_git_binding_ignores_submodule_ignore_configuration(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    child = tmp_path / "child"
    parent = tmp_path / "parent"
    child.mkdir()
    parent.mkdir()

    def git(cwd, *args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )

    for repository in (child, parent):
        git(repository, "init")
        git(repository, "config", "user.email", "tester@example.com")
        git(repository, "config", "user.name", "Test User")
    (child / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    git(child, "add", "tracked.txt")
    git(child, "commit", "-m", "child baseline")
    git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "nested",
    )
    git(parent, "commit", "-m", "parent baseline")
    git(parent, "config", "submodule.nested.ignore", "all")
    (parent / "nested" / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    workspace = LocalWorkspace(parent, workspace_id="git-workspace")
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        require_clean=False,
        verify_remote_url=False,
    )

    observation = asyncio.run(
        binding.observe_revision(BoundWorkspace(workspace=workspace, source_workspace=workspace))
    )

    assert observation.status is WorkspaceRevisionObservationStatus.INCOMPLETE
    assert observation.detail_code == "git_worktree_special_path_unsupported"


def test_git_binding_fails_closed_on_symlink_escape(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("outside-secret", encoding="utf-8")
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "escape").symlink_to(outside)
    workspace = LocalWorkspace(tmp_path, workspace_id="git-workspace")
    bound = BoundWorkspace(workspace=workspace, source_workspace=workspace)
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        verify_remote_url=False,
    )

    observation = asyncio.run(binding.observe_revision(bound))

    assert observation.status is WorkspaceRevisionObservationStatus.INCOMPLETE
    assert observation.detail_code == "git_worktree_path_unreadable"
    assert "outside-secret" not in observation.model_dump_json()


class _UnsafeGitStatusRunner(Runner):
    default_cwd = "/workspace"

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        del cwd, env, env_remove, timeout_s, stdin, output_limit_bytes
        argv = command.argv or ()
        if "--is-inside-work-tree" in argv:
            return ExecResult(stdout="true\n")
        if "status" in argv:
            return ExecResult(stdout="?? ../escape\x00")
        if "ls-files" in argv:
            return ExecResult(stdout="")
        if "symbolic-ref" in argv:
            return ExecResult(stdout="main\n")
        if "rev-parse" in argv:
            return ExecResult(stdout="a" * 40 + "\n")
        raise AssertionError(f"Unexpected Git command: {argv!r}")


def test_git_binding_rejects_traversal_reported_by_runner() -> None:
    workspace = RunnerWorkspace(
        _UnsafeGitStatusRunner(),
        cwd=None,
        workspace_id="runner-workspace",
    )
    bound = BoundWorkspace(workspace=workspace, source_workspace=workspace)
    binding = GitRepositoryBinding(
        repo_url="https://example.invalid/repository.git",
        fetch=False,
        verify_remote_url=False,
    )

    observation = asyncio.run(binding.observe_revision(bound))

    assert observation.status is WorkspaceRevisionObservationStatus.FAILED
    assert observation.detail_code == "unsafe_git_path"
    assert observation.paths == ()
