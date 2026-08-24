from __future__ import annotations

import asyncio
import hashlib
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from tests.workspaces.branch_conformance import (
    verify_atomic_publication,
    verify_bound_rollback_and_cleanup,
    verify_branch_isolation_and_net_changes,
    verify_conflict_is_all_or_none,
)
from tests.workspaces.conformance import (
    verify_bounded_reads_and_result_isolation,
    verify_paging_and_conditional_mutations,
    verify_relative_path_safety,
    verify_round_trip,
)

from cayu.workspaces import (
    LocalWorkspace,
    Workspace,
    WorkspaceBranchChangeSet,
    WorkspaceBranchClosedError,
    WorkspaceBranchFencedError,
    WorkspaceBranchLifecycleStatus,
    WorkspaceBranchLimits,
    WorkspaceBranchOutcomeStatus,
    WorkspaceBranchPublicationError,
    WorkspaceBranchPublicationRequest,
    WorkspaceBranchRequest,
    WorkspaceBranchResourceExhaustedError,
    WorkspaceListResult,
    WorkspaceMutationResult,
    WorkspaceReadResult,
)
from cayu.workspaces._local_branch import LocalWorkspaceBranch
from cayu.workspaces.branches import (
    workspace_branch_change_set_digest,
    workspace_branch_evidence,
)
from cayu.workspaces.revisions import (
    WorkspaceIdentity,
    WorkspaceRevisionObservationLimitExceeded,
    WorkspaceRevisionObservationLimits,
    observe_deterministic_workspace,
)

_BRANCH_PUBLIC_EXPORTS = (
    "RemoteWorkspaceBranchAuthorityProvider",
    "WorkspaceBranch",
    "WorkspaceBranchAuthority",
    "WorkspaceBranchBindingAuthority",
    "WorkspaceBranchBindingAuthorityClaimScope",
    "WorkspaceBranchCapabilities",
    "WorkspaceBranchChange",
    "WorkspaceBranchChangeSet",
    "WorkspaceBranchClosedError",
    "WorkspaceBranchConflict",
    "WorkspaceBranchContentIdentity",
    "WorkspaceBranchCreationResult",
    "WorkspaceBranchDurableState",
    "WorkspaceBranchEvidence",
    "WorkspaceBranchFencedError",
    "WorkspaceBranchLifecycleInspection",
    "WorkspaceBranchLifecycleStatus",
    "WorkspaceBranchLifecycleSummary",
    "WorkspaceBranchLimits",
    "WorkspaceBranchOperationConflict",
    "WorkspaceBranchOutcomeStatus",
    "WorkspaceBranchPublicationError",
    "WorkspaceBranchPublicationRequest",
    "WorkspaceBranchPublicationResult",
    "WorkspaceBranchPublicationStrength",
    "WorkspaceBranchRecoveryStrength",
    "WorkspaceBranchRecoveryRequest",
    "WorkspaceBranchRecoveryResult",
    "WorkspaceBranchRequest",
    "WorkspaceBranchResourceExhaustedError",
    "WorkspaceBranchRollbackRequest",
    "WorkspaceBranchRollbackResult",
    "WorkspaceBranchRetentionStrength",
    "WorkspaceBranchStore",
    "WorkspaceBranchStoreDurability",
)


async def _observation(workspace: Workspace):
    return await observe_deterministic_workspace(
        workspace,
        observer="local-branch-tests",
        limits=WorkspaceRevisionObservationLimits(),
    )


async def _source_and_request(
    root: Path,
    *,
    limits: WorkspaceBranchLimits | None = None,
) -> tuple[LocalWorkspace, WorkspaceBranchRequest]:
    source = LocalWorkspace(root, workspace_id="source-workspace")
    observation = await _observation(source)
    return source, WorkspaceBranchRequest(
        baseline=observation,
        limits=limits or WorkspaceBranchLimits(),
    )


async def _created_branch(
    source: LocalWorkspace,
    request: WorkspaceBranchRequest,
) -> LocalWorkspaceBranch:
    result = await source.create_branch(request)
    assert result.status is WorkspaceBranchOutcomeStatus.CREATED
    assert isinstance(result.branch, LocalWorkspaceBranch)
    return result.branch


def test_local_workspace_reports_actual_attached_branch_lifecycle(tmp_path: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        assert source.branch_lifecycle_summary().model_dump(mode="json") == {
            "attached_count": 0,
            "statuses": [],
            "truncated": False,
        }
        branch = await _created_branch(source, request)
        assert source.branch_lifecycle_summary().model_dump(mode="json") == {
            "attached_count": 1,
            "statuses": ["active"],
            "truncated": False,
        }

        await branch.rollback()

        assert source.branch_lifecycle_summary().model_dump(mode="json") == {
            "attached_count": 1,
            "statuses": ["rolled_back"],
            "truncated": False,
        }

    asyncio.run(scenario())


def _filesystem_is_case_insensitive(root: Path) -> bool:
    probe = root / ".cayu-case-probe"
    alternate = root / ".CAYU-CASE-PROBE"
    probe.write_bytes(b"probe")
    try:
        return alternate.exists() and os.path.samefile(probe, alternate)
    finally:
        probe.unlink()


@pytest.fixture
def populated_root(tmp_path: Path) -> Path:
    (tmp_path / "original.txt").write_bytes(b"original")
    (tmp_path / "deleted.txt").write_bytes(b"delete-me")
    return tmp_path


def test_workspace_branch_contracts_are_top_level_public_exports() -> None:
    import cayu
    import cayu.workspaces as workspaces

    for name in _BRANCH_PUBLIC_EXPORTS:
        assert getattr(cayu, name) is getattr(workspaces, name)
        assert name in cayu.__all__
        assert name in workspaces.__all__


def test_local_branches_pass_isolation_and_net_change_conformance(
    populated_root: Path,
) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        first = await _created_branch(source, request)
        second = await _created_branch(source, request)
        await verify_branch_isolation_and_net_changes(source, first, second)
        assert (await first.rollback()).status is WorkspaceBranchOutcomeStatus.ROLLED_BACK
        assert (await second.rollback()).status is WorkspaceBranchOutcomeStatus.ROLLED_BACK

    asyncio.run(scenario())


def test_local_branch_private_storage_uses_the_source_filesystem_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    source_root = tmp_path / "workspace"
    source_root.mkdir()
    (source_root / "source.txt").write_bytes(b"source")
    original_mkdtemp = branch_module.tempfile.mkdtemp
    requested_directories: list[Path | None] = []

    def record_mkdtemp(*args, **kwargs):
        directory = kwargs.get("dir")
        requested_directories.append(None if directory is None else Path(directory))
        return original_mkdtemp(*args, **kwargs)

    async def scenario() -> None:
        source, request = await _source_and_request(source_root)
        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module.tempfile, "mkdtemp", record_mkdtemp)
            branch = await _created_branch(source, request)

        assert requested_directories == [source_root.parent]
        assert branch._private_root.parent.samefile(source_root.parent)
        assert branch._private_root.stat().st_dev == source_root.stat().st_dev
        assert not branch._private_root.is_relative_to(source_root)
        await branch.rollback()

    asyncio.run(scenario())


def test_local_branch_rejects_mismatched_private_path_semantics_before_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    source_root = tmp_path / "workspace"
    source_root.mkdir()
    source_file = source_root / "source.txt"
    source_file.write_bytes(b"source")
    siblings_before = set(tmp_path.iterdir())
    casefolded = branch_module._DirectoryLookupSemantics.UNICODE_CASEFOLDED
    case_sensitive = branch_module._DirectoryLookupSemantics.CASE_SENSITIVE

    def mismatched_path_semantics(path: Path):
        return casefolded if path.samefile(source_root) else case_sensitive

    async def scenario() -> None:
        source, request = await _source_and_request(source_root)
        with monkeypatch.context() as scoped:
            scoped.setattr(
                branch_module,
                "_directory_lookup_semantics",
                mismatched_path_semantics,
            )
            result = await source.create_branch(request)

        assert result.status is WorkspaceBranchOutcomeStatus.UNSUPPORTED
        assert result.branch is None
        assert result.evidence.detail_code == "private_storage_path_semantics_mismatch"
        assert source_file.read_bytes() == b"source"
        assert set(tmp_path.iterdir()) == siblings_before

    asyncio.run(scenario())


def test_local_branch_rejects_unknown_path_semantics_before_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    source_root = tmp_path / "workspace"
    source_root.mkdir()
    source_file = source_root / "source.txt"
    source_file.write_bytes(b"source")
    siblings_before = set(tmp_path.iterdir())

    async def scenario() -> None:
        source, request = await _source_and_request(source_root)
        with monkeypatch.context() as scoped:
            scoped.setattr(
                branch_module,
                "_directory_lookup_semantics",
                lambda _path: branch_module._DirectoryLookupSemantics.UNKNOWN,
            )
            result = await source.create_branch(request)

        assert result.status is WorkspaceBranchOutcomeStatus.UNSUPPORTED
        assert result.branch is None
        assert result.evidence.detail_code == "path_semantics_unavailable"
        assert source_file.read_bytes() == b"source"
        assert set(tmp_path.iterdir()) == siblings_before

    asyncio.run(scenario())


def test_local_branch_rejects_mixed_source_directory_path_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    source_root = tmp_path / "workspace"
    nested = source_root / "nested"
    nested.mkdir(parents=True)
    nested_file = nested / "source.txt"
    nested_file.write_bytes(b"source")
    siblings_before = set(tmp_path.iterdir())
    casefolded = branch_module._DirectoryLookupSemantics.UNICODE_CASEFOLDED
    case_sensitive = branch_module._DirectoryLookupSemantics.CASE_SENSITIVE

    def mixed_path_semantics(path: Path):
        return casefolded if path.samefile(nested) else case_sensitive

    async def scenario() -> None:
        source, request = await _source_and_request(source_root)
        with monkeypatch.context() as scoped:
            scoped.setattr(
                branch_module,
                "_directory_lookup_semantics",
                mixed_path_semantics,
            )
            result = await source.create_branch(request)

        assert result.status is WorkspaceBranchOutcomeStatus.UNSUPPORTED
        assert result.branch is None
        assert result.evidence.detail_code == "source_contains_mixed_path_semantics"
        assert nested_file.read_bytes() == b"source"
        assert set(tmp_path.iterdir()) == siblings_before

    asyncio.run(scenario())


def test_local_branch_publication_rejects_source_path_semantics_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    source_root = tmp_path / "workspace"
    nested = source_root / "nested"
    nested.mkdir(parents=True)
    source_semantics = branch_module._DirectoryLookupSemantics.CASE_SENSITIVE

    def current_path_semantics(path: Path):
        if path.samefile(nested):
            return source_semantics
        return branch_module._DirectoryLookupSemantics.CASE_SENSITIVE

    async def scenario() -> None:
        nonlocal source_semantics

        source, request = await _source_and_request(source_root)
        with monkeypatch.context() as scoped:
            scoped.setattr(
                branch_module,
                "_directory_lookup_semantics",
                current_path_semantics,
            )
            branch = await _created_branch(source, request)
            await branch.write_bytes("nested/branch-only.txt", b"candidate")
            changes = await branch.changes()
            source_semantics = branch_module._DirectoryLookupSemantics.UNICODE_CASEFOLDED
            result = await branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                )
            )

        assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        assert not (nested / "branch-only.txt").exists()
        assert (await branch.rollback()).status is WorkspaceBranchOutcomeStatus.ROLLED_BACK

    asyncio.run(scenario())


def test_local_branch_publication_rejects_new_source_parent_semantics_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    source_root = tmp_path / "workspace"
    source_root.mkdir()
    new_source_parent = source_root / "newdir"
    source_parent_semantics = branch_module._DirectoryLookupSemantics.CASE_SENSITIVE

    def current_path_semantics(path: Path):
        if new_source_parent.exists() and path.samefile(new_source_parent):
            return source_parent_semantics
        return branch_module._DirectoryLookupSemantics.CASE_SENSITIVE

    async def scenario() -> None:
        nonlocal source_parent_semantics

        source, request = await _source_and_request(source_root)
        with monkeypatch.context() as scoped:
            scoped.setattr(
                branch_module,
                "_directory_lookup_semantics",
                current_path_semantics,
            )
            branch = await _created_branch(source, request)
            await branch.write_bytes("newdir/branch-only.txt", b"candidate")
            changes = await branch.changes()
            new_source_parent.mkdir()
            source_parent_semantics = branch_module._DirectoryLookupSemantics.UNICODE_CASEFOLDED
            result = await branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                )
            )

        assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        assert not (new_source_parent / "branch-only.txt").exists()
        assert (await branch.rollback()).status is WorkspaceBranchOutcomeStatus.ROLLED_BACK

    asyncio.run(scenario())


def test_local_branch_passes_ordinary_workspace_conformance(tmp_path: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        await verify_round_trip(branch)
        await verify_relative_path_safety(branch)
        await verify_bounded_reads_and_result_isolation(branch)
        await verify_paging_and_conditional_mutations(branch)
        await branch.rollback()

    asyncio.run(scenario())


def test_private_read_rejects_growth_before_or_during_bounded_read(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    class RejectUnboundedRead:
        def read(self, size: int = -1) -> bytes:
            raise AssertionError(f"private content was read with size {size}")

    class GrowAfterStat:
        def __init__(self, content: bytes) -> None:
            self.content = content
            self.read_sizes: list[int] = []

        def read(self, size: int = -1) -> bytes:
            self.read_sizes.append(size)
            return self.content[:size]

    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        expected = branch._baseline["original.txt"]
        real_open = branch_module.open_regular_for_read

        @contextmanager
        def oversized_before_read(root, path):
            if root == branch._baseline_root and path == "original.txt":
                yield RejectUnboundedRead(), expected.bytes + 1
                return
            with real_open(root, path) as opened:
                yield opened

        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module, "open_regular_for_read", oversized_before_read)
            with pytest.raises(WorkspaceBranchFencedError, match="content identity"):
                await branch.read_bytes("original.txt")

        growing = GrowAfterStat(b"original" + b"!")

        @contextmanager
        def grows_after_stat(root, path):
            if root == branch._baseline_root and path == "original.txt":
                yield growing, expected.bytes
                return
            with real_open(root, path) as opened:
                yield opened

        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module, "open_regular_for_read", grows_after_stat)
            with pytest.raises(WorkspaceBranchFencedError, match="changed during read"):
                await branch.read_bytes("original.txt")

        assert growing.read_sizes == [expected.bytes + 1]
        await branch.rollback()

    asyncio.run(scenario())


def test_publication_rejects_oversized_private_content_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    class RejectRead:
        def read(self, _size: int = -1) -> bytes:
            raise AssertionError("oversized private content was read")

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        await branch.write_bytes("created.txt", b"created")
        changes = await branch.changes()
        expected = branch._overlay["created.txt"]
        real_open = branch_module.open_regular_for_read

        @contextmanager
        def oversized_overlay(root, path):
            if root == branch._overlay_root and path == "created.txt":
                yield RejectRead(), expected.bytes + 1
                return
            with real_open(root, path) as opened:
                yield opened

        publication = WorkspaceBranchPublicationRequest(
            branch_id=branch.branch_id,
            baseline_revision=changes.baseline_revision,
            change_set_digest=changes.digest,
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module, "open_regular_for_read", oversized_overlay)
            with pytest.raises(WorkspaceBranchPublicationError) as raised:
                await branch.publish(publication)

        assert isinstance(raised.value.__cause__, WorkspaceBranchFencedError)
        assert not (tmp_path / "created.txt").exists()
        await branch.rollback()

    asyncio.run(scenario())


def test_publication_rollback_rejects_oversized_private_baseline_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    (tmp_path / "original.txt").write_bytes(b"original")

    class RejectRead:
        def read(self, _size: int = -1) -> bytes:
            raise AssertionError("oversized private baseline was read")

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"published")
        changes = await branch.changes()
        expected = branch._baseline["original.txt"]
        real_open = branch_module.open_regular_for_read
        real_replace = branch_module.replace_regular_if_revision

        @contextmanager
        def oversized_baseline(root, path):
            if root == branch._baseline_root and path == "original.txt":
                yield RejectRead(), expected.bytes + 1
                return
            with real_open(root, path) as opened:
                yield opened

        def replace_then_raise(*args, **kwargs):
            real_replace(*args, **kwargs)
            raise OSError("lost publication acknowledgement")

        publication = WorkspaceBranchPublicationRequest(
            branch_id=branch.branch_id,
            baseline_revision=changes.baseline_revision,
            change_set_digest=changes.digest,
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module, "open_regular_for_read", oversized_baseline)
            scoped.setattr(branch_module, "replace_regular_if_revision", replace_then_raise)
            with pytest.raises(WorkspaceBranchPublicationError) as raised:
                await branch.publish(publication)

        assert isinstance(raised.value.__cause__, BaseExceptionGroup)
        assert [type(error) for error in raised.value.__cause__.exceptions] == [
            OSError,
            WorkspaceBranchFencedError,
        ]
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED
        assert (tmp_path / "original.txt").read_bytes() == b"published"

    asyncio.run(scenario())


def test_local_branch_passes_bound_rollback_and_cleanup_conformance(tmp_path: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(
            tmp_path,
            limits=WorkspaceBranchLimits(max_file_bytes=3),
        )
        branch = await _created_branch(source, request)
        await verify_bound_rollback_and_cleanup(source, branch)

    asyncio.run(scenario())


def test_local_branch_publishes_complete_change_set(populated_root: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        assert request.baseline.revision is not None
        await verify_atomic_publication(source, branch, request.baseline.revision)
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.COMMITTED

    asyncio.run(scenario())


def test_local_branch_conflict_applies_none(populated_root: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        assert request.baseline.revision is not None
        await verify_conflict_is_all_or_none(source, branch, request.baseline.revision)
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.ACTIVE
        await branch.rollback()

    asyncio.run(scenario())


def test_created_path_publication_conflict_does_not_read_source_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        await branch.write_bytes("created.txt", b"branch")
        changes = await branch.changes()
        (tmp_path / "created.txt").write_bytes(b"source-conflict")

        def reject_content_read(_descriptor: int, _size: int) -> bytes:
            raise AssertionError("created-path conflict read source content")

        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module.os, "read", reject_content_read)
            result = await branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                )
            )

        assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        assert [(conflict.path, conflict.actual_kind) for conflict in result.conflicts] == [
            ("created.txt", "file")
        ]
        assert result.conflicts[0].actual is None
        await branch.rollback()

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["modified", "deleted"])
def test_size_mismatch_publication_conflict_does_not_read_source_content(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        path = "original.txt" if operation == "modified" else "deleted.txt"
        if operation == "modified":
            await branch.write_bytes(path, b"branch")
        else:
            await branch.delete(path)
        changes = await branch.changes()
        (populated_root / path).write_bytes(b"different-source-size")

        def reject_content_read(_descriptor: int, _size: int) -> bytes:
            raise AssertionError("size-mismatch conflict read source content")

        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module.os, "read", reject_content_read)
            result = await branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                )
            )

        assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        assert [(conflict.path, conflict.actual_kind) for conflict in result.conflicts] == [
            (path, "file")
        ]
        assert result.conflicts[0].actual is None
        await branch.rollback()

    asyncio.run(scenario())


def test_same_size_publication_candidate_is_hashed_before_conflict(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"branch")
        changes = await branch.changes()
        (populated_root / "original.txt").write_bytes(b"changed!")
        real_read = branch_module.os.read
        read_calls = 0

        def count_content_read(descriptor: int, size: int) -> bytes:
            nonlocal read_calls
            read_calls += 1
            return real_read(descriptor, size)

        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module.os, "read", count_content_read)
            result = await branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                )
            )

        assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        assert read_calls > 0
        assert result.conflicts[0].actual is not None
        assert result.conflicts[0].actual.bytes == len(b"changed!")
        await branch.rollback()

    asyncio.run(scenario())


def test_local_branch_creation_requires_current_exact_baseline(populated_root: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        await source.write_bytes("original.txt", b"new-source")
        result = await source.create_branch(request)
        assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        assert result.branch is None
        assert tuple(conflict.path for conflict in result.conflicts) == ("original.txt",)

    asyncio.run(scenario())


def test_creation_conflicts_stop_at_the_evidence_allocation_boundary(tmp_path: Path) -> None:
    for index in range(12):
        directory = tmp_path / f"directory-{index:02d}-{'x' * 40}"
        directory.mkdir()
        (directory / "source.txt").write_bytes(b"baseline")

    async def scenario() -> None:
        import cayu.workspaces._local_branch as branch_module

        source = LocalWorkspace(tmp_path, workspace_id="source-workspace")
        baseline = await _observation(source)
        for path in sorted(tmp_path.rglob("source.txt")):
            path.write_bytes(b"changed")

        probe = await source.create_branch(WorkspaceBranchRequest(baseline=baseline))
        assert probe.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        exact_bytes = 2 + len(probe.evidence.model_dump_json().encode("utf-8"))
        exact_bytes += sum(
            len(conflict.model_dump_json().encode("utf-8")) for conflict in probe.conflicts
        )
        exact_bytes += len(probe.conflicts) - 1
        assert exact_bytes > 1024

        constructed = 0
        original_conflict = branch_module.WorkspaceBranchConflict

        def count_conflict(*args, **kwargs):
            nonlocal constructed
            constructed += 1
            return original_conflict(*args, **kwargs)

        with pytest.MonkeyPatch.context() as scoped:
            scoped.setattr(branch_module, "WorkspaceBranchConflict", count_conflict)
            exhausted = await source.create_branch(
                WorkspaceBranchRequest(
                    baseline=baseline,
                    limits=WorkspaceBranchLimits(
                        max_active_branches=1,
                        max_evidence_bytes=1024,
                    ),
                )
            )
        assert exhausted.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert exhausted.evidence.detail_code == "conflict_evidence_limit_exceeded"
        assert constructed < len(probe.conflicts)

        exact = await source.create_branch(
            WorkspaceBranchRequest(
                baseline=baseline,
                limits=WorkspaceBranchLimits(
                    max_active_branches=1,
                    max_evidence_bytes=exact_bytes,
                ),
            )
        )
        assert exact.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        assert exact.conflicts == probe.conflicts

    asyncio.run(scenario())


def test_publication_ignores_unaffected_source_change(populated_root: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"branch")
        changes = await branch.changes()
        await source.write_bytes("deleted.txt", b"unaffected")
        result = await branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=branch.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )
        assert result.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (await source.read_bytes("original.txt")).content == b"branch"
        assert (await source.read_bytes("deleted.txt")).content == b"unaffected"

    asyncio.run(scenario())


def test_local_branch_rejects_stale_change_set_identity(populated_root: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        stale = await branch.changes()
        await branch.write_bytes("original.txt", b"later")
        with pytest.raises(ValueError, match="changed after"):
            await branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=stale.baseline_revision,
                    change_set_digest=stale.digest,
                )
            )
        assert (await source.read_bytes("original.txt")).content == b"original"
        await branch.rollback()

    asyncio.run(scenario())


def test_local_branch_rollback_is_idempotent_and_source_neutral(populated_root: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"branch")
        await source.write_bytes("original.txt", b"source")
        first = await branch.rollback()
        second = await branch.rollback()
        assert first == second
        assert (await source.read_bytes("original.txt")).content == b"source"
        with pytest.raises(WorkspaceBranchClosedError):
            await branch.read_bytes("original.txt")

    asyncio.run(scenario())


def test_local_branch_publication_retry_returns_same_commit(populated_root: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"published")
        changes = await branch.changes()
        publication = WorkspaceBranchPublicationRequest(
            branch_id=branch.branch_id,
            baseline_revision=changes.baseline_revision,
            change_set_digest=changes.digest,
        )
        first = await branch.publish(publication)
        second = await branch.publish(publication)
        assert first == second
        assert (await source.read_bytes("original.txt")).content == b"published"
        for conflicting in (
            WorkspaceBranchPublicationRequest(
                branch_id="another-branch",
                baseline_revision=publication.baseline_revision,
                change_set_digest=publication.change_set_digest,
            ),
            WorkspaceBranchPublicationRequest(
                branch_id=publication.branch_id,
                baseline_revision="sha256:" + "0" * 64,
                change_set_digest=publication.change_set_digest,
            ),
            WorkspaceBranchPublicationRequest(
                branch_id=publication.branch_id,
                baseline_revision=publication.baseline_revision,
                change_set_digest="sha256:" + "0" * 64,
            ),
        ):
            with pytest.raises(WorkspaceBranchClosedError, match="different publication"):
                await branch.publish(conflicting)

    asyncio.run(scenario())


def test_public_branch_results_do_not_alias_private_authority(populated_root: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        created = await source.create_branch(request)
        assert isinstance(created.branch, LocalWorkspaceBranch)
        branch = created.branch
        original_source = request.baseline.identity

        object.__setattr__(created.evidence.source, "workspace_id", "spoofed-creation")
        first_changes = await branch.changes()
        assert first_changes.source == original_source
        object.__setattr__(first_changes.source, "observer", "spoofed-change-set")
        second_changes = await branch.changes()
        assert second_changes.source == original_source
        assert second_changes.digest == first_changes.digest

        publication = WorkspaceBranchPublicationRequest(
            branch_id=branch.branch_id,
            baseline_revision=second_changes.baseline_revision,
            change_set_digest=second_changes.digest,
        )
        first_commit = await branch.publish(publication)
        forged_digest = "sha256:" + "0" * 64
        object.__setattr__(first_commit.evidence, "change_set_digest", forged_digest)
        object.__setattr__(first_commit.evidence.source, "workspace_id", "spoofed-commit")
        replay = await branch.publish(publication)
        assert replay is not first_commit
        assert replay.evidence.source == original_source
        assert replay.evidence.change_set_digest == publication.change_set_digest
        with pytest.raises(WorkspaceBranchClosedError, match="different publication"):
            await branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=publication.baseline_revision,
                    change_set_digest=forged_digest,
                )
            )

        rollback_branch = await _created_branch(source, request)
        first_rollback = await rollback_branch.rollback()
        object.__setattr__(first_rollback.evidence.source, "observer", "spoofed-rollback")
        object.__setattr__(first_rollback.evidence, "detail_code", "spoofed-detail")
        rollback_replay = await rollback_branch.rollback()
        assert rollback_replay is not first_rollback
        assert rollback_replay.evidence.source == original_source
        assert rollback_replay.evidence.detail_code == "workspace_branch_rolled_back"

    asyncio.run(scenario())


def test_conflict_results_do_not_alias_branch_authority(populated_root: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"branch")
        changes = await branch.changes()
        await source.write_bytes("original.txt", b"source")
        conflicted = await branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=branch.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )
        assert conflicted.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        object.__setattr__(conflicted.evidence.source, "workspace_id", "spoofed-conflict")
        object.__setattr__(conflicted.evidence, "change_set_digest", "sha256:" + "0" * 64)
        object.__setattr__(conflicted.conflicts[0], "path", "spoofed.txt")

        current = await branch.changes()
        assert current.source == request.baseline.identity
        assert current.digest == changes.digest
        await branch.rollback()

    asyncio.run(scenario())


def test_verified_publication_records_commit_before_source_lock_teardown(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._mutations as mutation_module

    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"published-before-teardown")
        changes = await branch.changes()
        publication = WorkspaceBranchPublicationRequest(
            branch_id=branch.branch_id,
            baseline_revision=changes.baseline_revision,
            change_set_digest=changes.digest,
        )
        original_lock = mutation_module.cooperative_path_lock
        failed = False

        @contextmanager
        def fail_source_lock_teardown(*args, **kwargs):
            nonlocal failed
            with original_lock(*args, **kwargs):
                yield
            if kwargs.get("lock_directory_name") == "cayu-workspace-source-locks" and not failed:
                failed = True
                raise OSError("injected source lock teardown failure")

        monkeypatch.setattr(
            mutation_module,
            "cooperative_path_lock",
            fail_source_lock_teardown,
        )
        with pytest.raises(OSError, match="source lock teardown"):
            await branch.publish(publication)

        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.COMMITTED
        assert (populated_root / "original.txt").read_bytes() == b"published-before-teardown"
        replay = await branch.publish(publication)
        assert replay.status is WorkspaceBranchOutcomeStatus.COMMITTED
        with pytest.raises(WorkspaceBranchFencedError):
            await source.read_bytes("original.txt")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("limits", "operation", "detail_code"),
    [
        (WorkspaceBranchLimits(max_file_bytes=3), "write", "file_byte_limit_exceeded"),
        (WorkspaceBranchLimits(max_overlay_bytes=3), "write", "overlay_byte_limit_exceeded"),
        (WorkspaceBranchLimits(max_changed_paths=1), "changes", "changed_path_limit_exceeded"),
        (WorkspaceBranchLimits(max_files=2), "files", "file_count_limit_exceeded"),
    ],
)
def test_local_branch_enforces_mutation_limits_before_change(
    tmp_path: Path,
    limits: WorkspaceBranchLimits,
    operation: str,
    detail_code: str,
) -> None:
    (tmp_path / "base.txt").write_bytes(b"a")

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path, limits=limits)
        branch = await _created_branch(source, request)
        if operation == "write":
            action = branch.write_bytes("base.txt", b"four")
        elif operation == "changes":
            await branch.write_bytes("base.txt", b"b")
            action = branch.write_bytes("second.txt", b"c")
        else:
            await branch.write_bytes("second.txt", b"b")
            action = branch.write_bytes("third.txt", b"c")
        with pytest.raises(WorkspaceBranchResourceExhaustedError) as raised:
            await action
        assert raised.value.detail_code == detail_code
        await branch.rollback()

    asyncio.run(scenario())


def test_local_branch_enforces_path_and_evidence_byte_limits(tmp_path: Path) -> None:
    async def path_limit_scenario() -> None:
        source, request = await _source_and_request(
            tmp_path,
            limits=WorkspaceBranchLimits(max_path_bytes=8),
        )
        branch = await _created_branch(source, request)
        with pytest.raises(WorkspaceBranchResourceExhaustedError) as raised:
            await branch.write_bytes("too-long-path.txt", b"x")
        assert raised.value.detail_code == "path_byte_limit_exceeded"
        await branch.rollback()

    asyncio.run(path_limit_scenario())

    async def path_count_scenario() -> None:
        source, request = await _source_and_request(
            tmp_path,
            limits=WorkspaceBranchLimits(max_paths=2),
        )
        branch = await _created_branch(source, request)
        with pytest.raises(WorkspaceBranchResourceExhaustedError) as raised:
            await branch.write_bytes("one/two/file.txt", b"x")
        assert raised.value.detail_code == "path_count_limit_exceeded"
        await branch.rollback()

    asyncio.run(path_count_scenario())

    async def evidence_scenario() -> None:
        source, request = await _source_and_request(
            tmp_path,
            limits=WorkspaceBranchLimits(max_evidence_bytes=1024),
        )
        branch = await _created_branch(source, request)
        exhausted = None
        for index in range(20):
            try:
                await branch.write_bytes(f"evidence/{index:02d}-{'x' * 40}.txt", b"x")
            except WorkspaceBranchResourceExhaustedError as error:
                exhausted = error
                break
        assert exhausted is not None
        assert exhausted.detail_code == "change_evidence_limit_exceeded"
        assert len((await branch.changes()).model_dump_json().encode()) <= 1024
        await branch.rollback()

    asyncio.run(evidence_scenario())


def test_local_branch_rejects_impossible_file_directory_views(tmp_path: Path) -> None:
    (tmp_path / "directory").mkdir()
    (tmp_path / "directory" / "child.txt").write_bytes(b"child")
    (tmp_path / "file.txt").write_bytes(b"file")

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        with pytest.raises(IsADirectoryError):
            await branch.write_bytes("directory", b"file-over-directory")
        with pytest.raises(NotADirectoryError):
            await branch.write_bytes("file.txt/child.txt", b"child-under-file")
        with pytest.raises(IsADirectoryError):
            await branch.delete("directory")
        assert (await branch.changes()).changes == ()
        await branch.rollback()

    asyncio.run(scenario())


def test_local_branch_rejects_filesystem_alias_of_baseline_directory(tmp_path: Path) -> None:
    if not _filesystem_is_case_insensitive(tmp_path):
        pytest.skip("The backing filesystem preserves case-distinct paths.")
    (tmp_path / "CaseSensitive").mkdir()

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)

        with pytest.raises(ValueError, match="filesystem-equivalent path"):
            await branch.write_bytes("casesensitive/new.txt", b"content")

        assert (await branch.changes()).changes == ()
        assert tuple(path.name for path in tmp_path.iterdir()) == ("CaseSensitive",)
        assert not any((tmp_path / "CaseSensitive").iterdir())
        await branch.rollback()

    asyncio.run(scenario())


def test_local_branch_rejects_filesystem_alias_between_overlay_directories(
    tmp_path: Path,
) -> None:
    if not _filesystem_is_case_insensitive(tmp_path):
        pytest.skip("The backing filesystem preserves case-distinct paths.")

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)

        await branch.write_bytes("CaseSensitive/one.txt", b"one")
        with pytest.raises(ValueError, match="filesystem-equivalent path"):
            await branch.write_bytes("casesensitive/two.txt", b"two")

        changes = await branch.changes()
        assert tuple(change.path for change in changes.changes) == ("CaseSensitive/one.txt",)
        assert not any(tmp_path.iterdir())
        await branch.rollback()

    asyncio.run(scenario())


def test_local_branch_serializes_concurrent_filesystem_alias_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _filesystem_is_case_insensitive(tmp_path):
        pytest.skip("The backing filesystem preserves case-distinct paths.")

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        validation_barrier = threading.Barrier(2)
        original_validate = branch._validated_path

        def rendezvous_before_state_lock(path: str) -> str:
            relative = original_validate(path)
            validation_barrier.wait(timeout=5)
            return relative

        monkeypatch.setattr(branch, "_validated_path", rendezvous_before_state_lock)
        results = await asyncio.gather(
            branch.write_bytes("CaseSensitive/one.txt", b"one"),
            branch.write_bytes("casesensitive/two.txt", b"two"),
            return_exceptions=True,
        )

        assert sum(result is None for result in results) == 1
        assert (
            sum(
                isinstance(result, ValueError) and "filesystem-equivalent path" in str(result)
                for result in results
            )
            == 1
        )
        changed_paths = tuple(change.path for change in (await branch.changes()).changes)
        assert changed_paths in {
            ("CaseSensitive/one.txt",),
            ("casesensitive/two.txt",),
        }
        assert not any(tmp_path.iterdir())
        await branch.rollback()

    asyncio.run(scenario())


def test_local_branch_prunes_transient_overlay_directories(tmp_path: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(
            tmp_path,
            limits=WorkspaceBranchLimits(max_paths=2),
        )
        branch = await _created_branch(source, request)
        for index in range(5):
            path = f"temporary-{index}/file.txt"
            await branch.write_bytes(path, b"temporary")
            await branch.delete(path)
            assert (await branch.changes()).changes == ()
            assert not (branch._overlay_root / f"temporary-{index}").exists()

        await branch.write_bytes("temporary-0", b"now-a-file")
        assert (await branch.read_bytes("temporary-0")).content == b"now-a-file"
        await branch.rollback()

    asyncio.run(scenario())


def test_failed_private_write_prunes_partial_overlay_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        original_write = branch_module.write_regular
        failed = False

        def make_parents_then_fail(root, path, content):
            nonlocal failed
            if root == branch._overlay_root and not failed:
                failed = True
                (root / "partial" / "nested").mkdir(parents=True)
                raise OSError("injected private write failure")
            return original_write(root, path, content)

        monkeypatch.setattr(branch_module, "write_regular", make_parents_then_fail)
        with pytest.raises(OSError, match="private write failure"):
            await branch.write_bytes("partial/nested/file.txt", b"content")
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.ACTIVE
        assert not (branch._overlay_root / "partial").exists()
        assert (await branch.changes()).changes == ()

        await branch.write_bytes("partial", b"retry")
        assert (await branch.read_bytes("partial")).content == b"retry"
        await branch.rollback()

    asyncio.run(scenario())


def test_local_branch_publishes_and_reverses_file_to_directory_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = tmp_path / "node"
    node.write_bytes(b"baseline-file")

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        await branch.delete("node")
        await branch.write_bytes("node/nested/child.txt", b"child")
        changes = await branch.changes()
        assert [(change.path, change.operation) for change in changes.changes] == [
            ("node", "deleted"),
            ("node/nested/child.txt", "created"),
        ]

        def fail_after_apply(_change_set):
            raise OSError("injected post-apply verification failure")

        monkeypatch.setattr(branch, "_published_identity_conflicts", fail_after_apply)
        with pytest.raises(WorkspaceBranchPublicationError):
            await branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                )
            )
        assert node.is_file()
        assert node.read_bytes() == b"baseline-file"
        await branch.rollback()

        replacement = await _created_branch(source, request)
        await replacement.delete("node")
        await replacement.write_bytes("node/nested/child.txt", b"child")
        replacement_changes = await replacement.changes()
        committed = await replacement.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=replacement.branch_id,
                baseline_revision=replacement_changes.baseline_revision,
                change_set_digest=replacement_changes.digest,
            )
        )
        assert committed.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (node / "nested" / "child.txt").read_bytes() == b"child"

    asyncio.run(scenario())


def test_creation_evidence_stays_bounded_independent_of_baseline(tmp_path: Path) -> None:
    for index in range(20):
        (tmp_path / f"baseline-{index:02d}.txt").write_bytes(b"baseline")

    async def scenario() -> None:
        source, request = await _source_and_request(
            tmp_path,
            limits=WorkspaceBranchLimits(max_evidence_bytes=1024),
        )
        result = await source.create_branch(request)
        assert result.status is WorkspaceBranchOutcomeStatus.CREATED
        assert result.evidence.affected_path_sha256 == ()
        assert len(result.evidence.model_dump_json().encode()) <= 1024
        assert result.branch is not None
        await result.branch.rollback()

    asyncio.run(scenario())


def test_creation_rejects_fixed_identity_that_cannot_fit_evidence(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace_id = "workspace-" + "w" * 2000
        source = LocalWorkspace(tmp_path, workspace_id=workspace_id)
        baseline = await observe_deterministic_workspace(
            source,
            observer="observer-" + "o" * 2000,
            limits=WorkspaceRevisionObservationLimits(),
        )
        request = WorkspaceBranchRequest(
            baseline=baseline,
            limits=WorkspaceBranchLimits(max_evidence_bytes=1024),
        )
        first = await source.create_branch(request)
        second = await source.create_branch(request)
        assert first == second
        assert first.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert first.branch is None
        assert first.evidence.detail_code in {
            "change_evidence_limit_exceeded",
            "result_evidence_limit_exceeded",
        }
        assert len(first.evidence.model_dump_json().encode("utf-8")) <= 1024
        assert first.evidence.source.workspace_id.startswith("sha256:")
        assert first.evidence.source.observer.startswith("sha256:")

    asyncio.run(scenario())


def test_creation_preflights_largest_minimal_terminal_evidence(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace_id = "workspace-" + "w" * 420
        observer = "observer-" + "o" * 420
        source = LocalWorkspace(tmp_path, workspace_id=workspace_id)
        baseline = await observe_deterministic_workspace(
            source,
            observer=observer,
            limits=WorkspaceRevisionObservationLimits(),
        )
        assert baseline.revision is not None
        branch_id = "wsb_" + "0" * 32
        identity = WorkspaceIdentity(workspace_id=workspace_id, observer=observer)
        digest = workspace_branch_change_set_digest(
            branch_id=branch_id,
            source=identity,
            baseline_revision=baseline.revision,
            changes=(),
        )
        change_set = WorkspaceBranchChangeSet(
            branch_id=branch_id,
            source=identity,
            baseline_revision=baseline.revision,
            changes=(),
            digest=digest,
        )
        terminal_evidence = workspace_branch_evidence(
            source=identity,
            baseline_revision=baseline.revision,
            branch_id=branch_id,
            outcome=WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            change_set_digest=digest,
            detail_code="result_evidence_limit_exceeded",
        )
        terminal_bytes = len(terminal_evidence.model_dump_json().encode("utf-8"))
        assert terminal_bytes >= 1024
        assert len(change_set.model_dump_json().encode("utf-8")) <= terminal_bytes - 1

        rejected = await source.create_branch(
            WorkspaceBranchRequest(
                baseline=baseline,
                limits=WorkspaceBranchLimits(max_evidence_bytes=terminal_bytes - 1),
            )
        )
        assert rejected.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert rejected.evidence.detail_code == "result_evidence_limit_exceeded"
        assert len(rejected.evidence.model_dump_json().encode("utf-8")) <= terminal_bytes - 1

        exact = await source.create_branch(
            WorkspaceBranchRequest(
                baseline=baseline,
                limits=WorkspaceBranchLimits(max_evidence_bytes=terminal_bytes),
            )
        )
        assert exact.status is WorkspaceBranchOutcomeStatus.CREATED
        assert exact.branch is not None
        changes = await exact.branch.changes()
        committed = await exact.branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=changes.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )
        assert committed.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert len(committed.evidence.model_dump_json().encode("utf-8")) <= terminal_bytes

    asyncio.run(scenario())


def test_local_branch_creation_returns_typed_baseline_limit(populated_root: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(
            populated_root,
            limits=WorkspaceBranchLimits(max_baseline_bytes=4),
        )
        result = await source.create_branch(request)
        assert result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert result.evidence.detail_code == "baseline_byte_limit_exceeded"

    asyncio.run(scenario())


def test_local_branch_capture_bounds_growth_by_remaining_baseline_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    source_path = tmp_path / "growing.txt"
    source_path.write_bytes(b"x")
    original_copy = branch_module._copy_one_regular
    rejected_target_sizes: list[int] = []
    grew_source = False

    def grow_after_stat(*args, **kwargs):
        nonlocal grew_source
        if not grew_source:
            source_path.write_bytes(b"grown")
            grew_source = True
        target = args[2]
        try:
            return original_copy(*args, **kwargs)
        except WorkspaceBranchResourceExhaustedError:
            rejected_target_sizes.append(target.stat().st_size)
            raise

    async def scenario() -> None:
        source, request = await _source_and_request(
            tmp_path,
            limits=WorkspaceBranchLimits(
                max_file_bytes=16,
                max_baseline_bytes=2,
            ),
        )
        monkeypatch.setattr(branch_module, "_copy_one_regular", grow_after_stat)

        result = await source.create_branch(request)

        assert result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert result.evidence.detail_code == "baseline_byte_limit_exceeded"
        assert result.branch is None
        assert rejected_target_sizes
        assert max(rejected_target_sizes) <= request.limits.max_baseline_bytes

    asyncio.run(scenario())


def test_private_branch_files_remain_owner_readable_independent_of_source_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    source_path = tmp_path / "shared.txt"
    source_path.write_bytes(b"shared")
    source_info = source_path.stat()
    original_fstat = branch_module.os.fstat

    def report_nonowner_read_mode(descriptor: int):
        info = original_fstat(descriptor)
        if (info.st_dev, info.st_ino) != (source_info.st_dev, source_info.st_ino):
            return info
        values = list(info)
        values[0] = (info.st_mode & ~0o7777) | 0o040
        return os.stat_result(values)

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module.os, "fstat", report_nonowner_read_mode)
            branch = await _created_branch(source, request)

        baseline_path = branch._baseline_root / "shared.txt"
        assert baseline_path.stat().st_mode & 0o7777 == 0o600
        assert (await branch.read_bytes("shared.txt")).content == b"shared"

        await branch.write_bytes("shared.txt", b"private")
        overlay_path = branch._overlay_root / "shared.txt"
        assert overlay_path.stat().st_mode & 0o7777 == 0o600
        assert (await branch.read_bytes("shared.txt")).content == b"private"
        await branch.rollback()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_local_branch_creation_fails_closed_for_non_regular_source(
    tmp_path: Path,
    kind: str,
) -> None:
    (tmp_path / "regular.txt").write_bytes(b"regular")
    source = LocalWorkspace(tmp_path, workspace_id="source-workspace")

    async def scenario() -> None:
        observation = await _observation(source)
        if kind == "symlink":
            (tmp_path / "unsafe").symlink_to(tmp_path / "regular.txt")
        else:
            os.mkfifo(tmp_path / "unsafe")
        result = await source.create_branch(WorkspaceBranchRequest(baseline=observation))
        assert result.status is WorkspaceBranchOutcomeStatus.UNSUPPORTED
        assert result.evidence.detail_code in {
            "source_contains_symlink",
            "source_contains_special_file",
        }

    asyncio.run(scenario())


@pytest.mark.skipif(os.name == "nt", reason="Backslash is a path separator on Windows.")
def test_local_branch_preserves_distinct_posix_paths(tmp_path: Path) -> None:
    (tmp_path / "ambiguous").mkdir()
    (tmp_path / "ambiguous" / "name.txt").write_bytes(b"nested")
    (tmp_path / "ambiguous\\name.txt").write_bytes(b"backslash")

    async def scenario() -> None:
        source = LocalWorkspace(tmp_path, workspace_id="ambiguous-source")
        observation = await _observation(source)
        result = await source.create_branch(WorkspaceBranchRequest(baseline=observation))
        assert result.status is WorkspaceBranchOutcomeStatus.CREATED
        assert isinstance(result.branch, LocalWorkspaceBranch)
        branch = result.branch

        assert (await branch.read_bytes("ambiguous/name.txt")).content == b"nested"
        assert (await branch.read_bytes("ambiguous\\name.txt")).content == b"backslash"
        await branch.write_bytes("ambiguous/name.txt", b"nested-branch")
        await branch.write_bytes("ambiguous\\name.txt", b"backslash-branch")
        changes = await branch.changes()
        assert tuple(change.path for change in changes.changes) == (
            "ambiguous/name.txt",
            "ambiguous\\name.txt",
        )

        published = await branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=branch.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )
        assert published.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (tmp_path / "ambiguous" / "name.txt").read_bytes() == b"nested-branch"
        assert (tmp_path / "ambiguous\\name.txt").read_bytes() == b"backslash-branch"

    asyncio.run(scenario())


def test_publication_uses_actual_filesystem_alias_semantics(tmp_path: Path) -> None:
    canonical = "Case/item.txt"
    alias = "case/ITEM.TXT"
    (tmp_path / "Case").mkdir()
    (tmp_path / canonical).write_bytes(b"baseline")

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        await branch.write_bytes(canonical, b"branch")
        changes = await branch.changes()
        await source.write_bytes(alias, b"late-alias")
        aliases = os.path.samefile(tmp_path / "Case", tmp_path / "case")

        result = await branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=branch.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )

        if aliases:
            assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
            assert tuple(conflict.path for conflict in result.conflicts) == (canonical,)
            assert (await source.read_bytes(canonical)).content == b"late-alias"
            await branch.rollback()
        else:
            assert result.status is WorkspaceBranchOutcomeStatus.COMMITTED
            assert (await source.read_bytes(canonical)).content == b"branch"
            assert (await source.read_bytes(alias)).content == b"late-alias"

    asyncio.run(scenario())


def test_publication_uses_actual_unicode_filesystem_alias_semantics(tmp_path: Path) -> None:
    import unicodedata

    requested_directory = "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    (tmp_path / requested_directory).mkdir()
    actual_directory = next(path.name for path in tmp_path.iterdir())
    (tmp_path / actual_directory / "item.txt").write_bytes(b"baseline")
    alternate_directory = unicodedata.normalize("NFD", actual_directory)
    if alternate_directory == actual_directory:
        alternate_directory = unicodedata.normalize("NFC", actual_directory)
    assert alternate_directory != actual_directory
    canonical = f"{actual_directory}/item.txt"
    alias = f"{alternate_directory}/item.txt"

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        await branch.write_bytes(canonical, b"branch")
        changes = await branch.changes()
        await source.write_bytes(alias, b"late-alias")
        aliases = os.path.samefile(
            tmp_path / actual_directory,
            tmp_path / alternate_directory,
        )

        result = await branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=branch.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )

        if aliases:
            assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
            assert tuple(conflict.path for conflict in result.conflicts) == (canonical,)
            assert (await source.read_bytes(canonical)).content == b"late-alias"
            await branch.rollback()
        else:
            assert result.status is WorkspaceBranchOutcomeStatus.COMMITTED
            assert (await source.read_bytes(canonical)).content == b"branch"
            assert (await source.read_bytes(alias)).content == b"late-alias"

    asyncio.run(scenario())


def test_publication_rejects_branch_only_source_alias_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    canonical = "Case/branch-only.txt"
    source_alias = tmp_path / "case" / "source-only.txt"

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        await branch.write_bytes(canonical, b"branch")
        changes = await branch.changes()
        source_alias.parent.mkdir()
        source_alias.write_bytes(b"source")

        def injected_alias(root: Path, first: str, second: str) -> bool:
            assert root.samefile(tmp_path)
            return {first, second} == {"Case", "case"}

        def reject_source_creation(*_args, **_kwargs) -> None:
            raise AssertionError("publication began mutating the source")

        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module, "_filesystem_paths_alias", injected_alias)
            scoped.setattr(branch_module, "create_regular", reject_source_creation)
            result = await branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                )
            )

        assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        assert tuple(conflict.path for conflict in result.conflicts) == ("case",)
        assert source_alias.read_bytes() == b"source"
        assert not (tmp_path / canonical).exists()
        assert (await branch.rollback()).status is WorkspaceBranchOutcomeStatus.ROLLED_BACK

    asyncio.run(scenario())


def test_publication_alias_scan_stops_at_the_configured_path_bound(tmp_path: Path) -> None:
    (tmp_path / "source.txt").write_bytes(b"baseline")

    async def scenario() -> None:
        source, request = await _source_and_request(
            tmp_path,
            limits=WorkspaceBranchLimits(max_paths=1, max_files=1),
        )
        branch = await _created_branch(source, request)
        await branch.write_bytes("source.txt", b"branch")
        changes = await branch.changes()
        await source.write_bytes("unrelated.txt", b"unrelated")

        result = await branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=branch.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )

        assert result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert result.evidence.detail_code == "source_alias_scan_path_limit_exceeded"
        assert (await source.read_bytes("source.txt")).content == b"baseline"
        await branch.rollback()

    asyncio.run(scenario())


def test_publication_alias_scan_errors_stop_at_the_evidence_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    components = tuple(f"d{index:02d}" for index in range(24))
    parent = tmp_path.joinpath(*components)
    parent.mkdir(parents=True)
    changed_path = "/".join((*components, "created.txt"))

    async def scenario() -> None:
        source, request = await _source_and_request(
            tmp_path,
            limits=WorkspaceBranchLimits(
                max_paths=64,
                max_files=64,
                max_evidence_bytes=1024,
            ),
        )
        branch = await _created_branch(source, request)
        await branch.write_bytes(changed_path, b"branch")
        changes = await branch.changes()
        scans = 0

        def reject_scan(_directory_fd):
            nonlocal scans
            scans += 1
            raise OSError("injected alias enumeration failure")

        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module.os, "scandir", reject_scan)
            result = await branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                )
            )

        assert result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert result.evidence.detail_code == "conflict_evidence_limit_exceeded"
        assert scans < len(components) + 1
        assert not (tmp_path / Path(*components) / "created.txt").exists()
        await branch.rollback()

    asyncio.run(scenario())


def test_publication_physical_aliases_do_not_duplicate_content_conflicts(tmp_path: Path) -> None:
    path_count = 6

    def populate(root: Path) -> None:
        root.mkdir(exist_ok=True)
        for index in range(path_count):
            directory = root / f"Case{index:02d}"
            directory.mkdir()
            (directory / "item.txt").write_bytes(b"baseline")

    async def prepare(root: Path):
        source, request = await _source_and_request(
            root,
            limits=WorkspaceBranchLimits(max_changed_paths=path_count * 2),
        )
        branch = await _created_branch(source, request)
        for index in range(path_count):
            canonical = f"Case{index:02d}/item.txt"
            alias = f"case{index:02d}/ITEM.TXT"
            await branch.write_bytes(canonical, b"branch")
            await source.write_bytes(canonical, b"source-changed")
            await source.write_bytes(alias, b"late-alias")
        return source, branch, await branch.changes()

    async def scenario() -> None:
        populate(tmp_path)
        _source, branch, changes = await prepare(tmp_path)
        result = await branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=branch.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )
        assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        assert tuple(conflict.path for conflict in result.conflicts) == tuple(
            f"Case{index:02d}/item.txt" for index in range(path_count)
        )
        await branch.rollback()

    asyncio.run(scenario())


def test_publication_rejects_late_symlink_ancestor_without_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_bytes(b"outside")

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        await branch.write_bytes("redirect/secret.txt", b"branch")
        changes = await branch.changes()
        (tmp_path / "redirect").symlink_to(outside, target_is_directory=True)
        result = await branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=branch.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )
        assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        assert [(conflict.path, conflict.actual_kind) for conflict in result.conflicts] == [
            ("redirect", "symlink")
        ]
        assert outside_file.read_bytes() == b"outside"
        await branch.rollback()

    asyncio.run(scenario())


def test_publication_fails_closed_after_source_root_replacement(tmp_path: Path) -> None:
    (tmp_path / "source.txt").write_bytes(b"baseline")

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        await branch.write_bytes("source.txt", b"branch")
        changes = await branch.changes()
        moved = tmp_path.with_name(f"{tmp_path.name}-moved")
        tmp_path.rename(moved)
        tmp_path.mkdir()
        (tmp_path / "source.txt").write_bytes(b"replacement-root")
        with pytest.raises(WorkspaceBranchFencedError, match="root identity"):
            await branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                )
            )
        assert (tmp_path / "source.txt").read_bytes() == b"replacement-root"
        assert (moved / "source.txt").read_bytes() == b"baseline"
        await branch.rollback()

    asyncio.run(scenario())


def test_local_branch_expires_and_releases_capacity(populated_root: Path) -> None:
    async def scenario() -> None:
        limits = WorkspaceBranchLimits(lifetime_ms=20, max_active_branches=1)
        source, request = await _source_and_request(populated_root, limits=limits)
        branch = await _created_branch(source, request)
        await asyncio.sleep(0.1)
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.ROLLED_BACK
        replacement = await source.create_branch(request)
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None
        await replacement.branch.rollback()

    asyncio.run(scenario())


def test_cancelled_creation_drains_snapshot_and_releases_capacity(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    entered = threading.Event()
    release = threading.Event()
    original = branch_module._capture_baseline

    def delayed_capture(*args):
        entered.set()
        assert release.wait(timeout=5)
        return original(*args)

    monkeypatch.setattr(branch_module, "_capture_baseline", delayed_capture)

    async def scenario() -> None:
        source, request = await _source_and_request(
            populated_root,
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )
        task = asyncio.create_task(source.create_branch(request))
        await asyncio.to_thread(entered.wait, 5)
        assert task.cancel()
        await asyncio.sleep(0)
        assert task.cancelling() == 1
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        replacement = await source.create_branch(request)
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None
        await replacement.branch.rollback()

    asyncio.run(scenario())


def test_cancelled_creation_retains_capacity_during_async_snapshot_cleanup(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    capture_entered = threading.Event()
    release_capture = threading.Event()
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    original_capture = branch_module._capture_baseline
    original_remove = branch_module._remove_private_tree

    def delayed_capture(*args):
        capture_entered.set()
        assert release_capture.wait(timeout=5)
        return original_capture(*args)

    remove_calls = 0

    def delayed_remove(path):
        nonlocal remove_calls
        remove_calls += 1
        if remove_calls == 1:
            cleanup_entered.set()
            assert release_cleanup.wait(timeout=5)
        return original_remove(path)

    monkeypatch.setattr(branch_module, "_capture_baseline", delayed_capture)
    monkeypatch.setattr(branch_module, "_remove_private_tree", delayed_remove)

    async def scenario() -> None:
        source, request = await _source_and_request(
            populated_root,
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )
        task = asyncio.create_task(source.create_branch(request))
        await asyncio.to_thread(capture_entered.wait, 5)
        assert task.cancel()
        release_capture.set()
        await asyncio.to_thread(cleanup_entered.wait, 5)

        heartbeat = asyncio.create_task(asyncio.sleep(0.01))
        await asyncio.wait_for(heartbeat, timeout=1)
        blocked = await source.create_branch(request)
        assert blocked.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert blocked.evidence.detail_code == "active_branch_limit_exceeded"
        assert not task.done()

        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        replacement = await source.create_branch(request)
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None
        await replacement.branch.rollback()

    asyncio.run(scenario())


def test_failed_cancelled_creation_cleanup_retains_capacity_for_retry_owner(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    capture_entered = threading.Event()
    release_capture = threading.Event()
    original_capture = branch_module._capture_baseline
    original_remove = branch_module._remove_private_tree
    scheduled: list[tuple[Path, object]] = []

    def delayed_capture(*args):
        capture_entered.set()
        assert release_capture.wait(timeout=5)
        return original_capture(*args)

    remove_calls = 0

    def fail_once(path):
        nonlocal remove_calls
        remove_calls += 1
        if remove_calls == 1:
            raise OSError("injected abandoned snapshot cleanup failure")
        return original_remove(path)

    def retain_retry(path, capacity_lease):
        scheduled.append((path, capacity_lease))

    monkeypatch.setattr(branch_module, "_capture_baseline", delayed_capture)
    monkeypatch.setattr(branch_module, "_remove_private_tree", fail_once)
    monkeypatch.setattr(branch_module, "_schedule_private_tree_cleanup", retain_retry)

    async def scenario() -> None:
        source, request = await _source_and_request(
            populated_root,
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )
        task = asyncio.create_task(source.create_branch(request))
        await asyncio.to_thread(capture_entered.wait, 5)
        assert task.cancel()
        assert task.cancelling() == 1
        release_capture.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert task.cancelled()
        assert task.cancelling() == 1
        assert isinstance(raised.value.__cause__, OSError)
        assert len(scheduled) == 1
        assert len(branch_module._PRIVATE_TREE_CLEANUPS) == 1

        replacement = await source.create_branch(request)
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None
        assert not branch_module._PRIVATE_TREE_CLEANUPS
        scheduled.clear()
        await replacement.branch.rollback()

    asyncio.run(scenario())


def test_creation_setup_failure_removes_owned_private_tree(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        private_roots: list[Path] = []
        original_mkdtemp = branch_module.tempfile.mkdtemp
        original_chmod = branch_module.os.chmod

        def record_mkdtemp(*args, **kwargs):
            path = Path(original_mkdtemp(*args, **kwargs))
            private_roots.append(path)
            return str(path)

        def fail_private_chmod(path, mode):
            if Path(path) in private_roots:
                raise OSError("injected private setup failure")
            return original_chmod(path, mode)

        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module.tempfile, "mkdtemp", record_mkdtemp)
            scoped.setattr(branch_module.os, "chmod", fail_private_chmod)
            with pytest.raises(OSError, match="private setup failure"):
                await source.create_branch(request)

        assert private_roots
        assert all(not path.exists() for path in private_roots)
        replacement = await source.create_branch(request)
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None
        await replacement.branch.rollback()

    asyncio.run(scenario())


def test_creation_timer_failure_cleans_private_tree_without_blocking_event_loop(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    private_roots: list[Path] = []
    original_mkdtemp = branch_module.tempfile.mkdtemp
    original_remove = branch_module._remove_private_tree

    def record_mkdtemp(*args, **kwargs):
        path = Path(original_mkdtemp(*args, **kwargs))
        private_roots.append(path)
        return str(path)

    def block_cleanup(path):
        cleanup_entered.set()
        assert release_cleanup.wait(timeout=5)
        return original_remove(path)

    def reject_timer_start(_timer):
        raise RuntimeError("injected branch timer failure")

    async def scenario() -> None:
        source, request = await _source_and_request(
            populated_root,
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module.tempfile, "mkdtemp", record_mkdtemp)
            scoped.setattr(branch_module, "_remove_private_tree", block_cleanup)
            scoped.setattr(threading.Timer, "start", reject_timer_start)
            creation = asyncio.create_task(source.create_branch(request))
            assert await asyncio.to_thread(cleanup_entered.wait, 5)

            loop_responsive = asyncio.Event()
            asyncio.get_running_loop().call_soon(loop_responsive.set)
            await asyncio.wait_for(loop_responsive.wait(), timeout=1)
            assert not creation.done()

            release_cleanup.set()
            with pytest.raises(RuntimeError, match="branch timer failure"):
                await creation

        assert private_roots
        assert all(not path.exists() for path in private_roots)
        replacement = await source.create_branch(request)
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None
        await replacement.branch.rollback()

    asyncio.run(scenario())


def test_creation_lock_teardown_failure_removes_captured_private_tree(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module
    import cayu.workspaces._mutations as mutation_module

    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        private_roots: list[Path] = []
        original_mkdtemp = branch_module.tempfile.mkdtemp
        original_lock = mutation_module.cooperative_path_lock
        failed = False

        def record_mkdtemp(*args, **kwargs):
            path = Path(original_mkdtemp(*args, **kwargs))
            private_roots.append(path)
            return str(path)

        @contextmanager
        def fail_source_lock_teardown(*args, **kwargs):
            nonlocal failed
            with original_lock(*args, **kwargs):
                yield
            if kwargs.get("lock_directory_name") == "cayu-workspace-source-locks" and not failed:
                failed = True
                raise OSError("injected capture lock teardown failure")

        with monkeypatch.context() as scoped:
            scoped.setattr(branch_module.tempfile, "mkdtemp", record_mkdtemp)
            scoped.setattr(
                mutation_module,
                "cooperative_path_lock",
                fail_source_lock_teardown,
            )
            with pytest.raises(OSError, match="capture lock teardown"):
                await source.create_branch(request)

        assert private_roots
        assert all(not path.exists() for path in private_roots)
        replacement = await source.create_branch(request)
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None
        await replacement.branch.rollback()

    asyncio.run(scenario())


def test_cancelled_publication_drains_to_terminal_commit(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module
    import cayu.workspaces.local as local_module

    entered = threading.Event()
    release = threading.Event()
    contender_entered = threading.Event()
    original = branch_module.replace_regular_if_revision
    original_write = local_module._write_file

    def delayed_replace(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(timeout=5)
        return result

    def observed_write(*args, **kwargs):
        contender_entered.set()
        return original_write(*args, **kwargs)

    monkeypatch.setattr(branch_module, "replace_regular_if_revision", delayed_replace)
    monkeypatch.setattr(local_module, "_write_file", observed_write)

    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"published")
        changes = await branch.changes()
        publication = WorkspaceBranchPublicationRequest(
            branch_id=branch.branch_id,
            baseline_revision=changes.baseline_revision,
            change_set_digest=changes.digest,
        )
        task = asyncio.create_task(branch.publish(publication))
        await asyncio.to_thread(entered.wait, 5)
        assert task.cancel()
        await asyncio.sleep(0)
        assert task.cancelling() == 1
        assert not task.done()
        contender = asyncio.create_task(source.write_bytes("deleted.txt", b"after-publication"))
        await asyncio.to_thread(contender_entered.wait, 5)
        await asyncio.sleep(0.02)
        assert not contender.done()
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:  # pragma: no cover - cancellation must remain caller-visible
            raise AssertionError("Publication cancellation was lost.")
        assert task.cancelled()
        await contender
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.COMMITTED
        assert (await source.read_bytes("original.txt")).content == b"published"
        assert (await source.read_bytes("deleted.txt")).content == b"after-publication"
        replay = await branch.publish(publication)
        assert replay.status is WorkspaceBranchOutcomeStatus.COMMITTED

    asyncio.run(scenario())


def test_waiting_source_call_is_rejected_when_publication_fences_source(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module
    import cayu.workspaces.local as local_module

    rollback_entered = threading.Event()
    release_rollback = threading.Event()
    contender_entered = threading.Event()

    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        private_root = branch._private_root
        await branch.write_bytes("original.txt", b"uncertain")
        changes = await branch.changes()

        original_replace = branch_module.replace_regular_if_revision

        def commit_then_raise(*args, **kwargs):
            original_replace(*args, **kwargs)
            raise OSError("primary publication failure")

        def block_then_fail_restore(*_args, **_kwargs):
            rollback_entered.set()
            assert release_rollback.wait(timeout=5)
            raise OSError("rollback restoration failure")

        original_read = local_module._read_file_locked

        def observed_read(*args, **kwargs):
            contender_entered.set()
            return original_read(*args, **kwargs)

        monkeypatch.setattr(branch_module, "replace_regular_if_revision", commit_then_raise)
        monkeypatch.setattr(branch_module, "restore_regular", block_then_fail_restore)
        publication = asyncio.create_task(
            branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                )
            )
        )
        assert await asyncio.to_thread(rollback_entered.wait, 5)

        monkeypatch.setattr(local_module, "_read_file_locked", observed_read)
        contender = asyncio.create_task(source.read_bytes("original.txt"))
        assert await asyncio.to_thread(contender_entered.wait, 5)
        await asyncio.sleep(0)
        assert not contender.done()

        release_rollback.set()
        with pytest.raises(WorkspaceBranchPublicationError):
            await publication
        with pytest.raises(WorkspaceBranchFencedError):
            await asyncio.wait_for(contender, timeout=2)
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED
        assert not private_root.exists()

    asyncio.run(scenario())


def test_publication_failure_restores_every_applied_path(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    original = branch_module.replace_regular_if_revision
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(branch_module, "replace_regular_if_revision", fail_second)

    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("deleted.txt", b"branch-one")
        await branch.write_bytes("original.txt", b"branch-two")
        changes = await branch.changes()
        with pytest.raises(WorkspaceBranchPublicationError):
            await branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                )
            )
        assert (await source.read_bytes("deleted.txt")).content == b"delete-me"
        assert (await source.read_bytes("original.txt")).content == b"original"
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.ACTIVE
        await branch.rollback()

    asyncio.run(scenario())


@pytest.mark.parametrize("first_operation", ["modified", "deleted"])
def test_publication_rollback_preserves_prepublication_source_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_operation: str,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    target = tmp_path / "a-target.txt"
    later = tmp_path / "z-later.txt"
    target.write_bytes(b"target-baseline")
    later.write_bytes(b"later-baseline")
    target.chmod(0o644)

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        if first_operation == "modified":
            await branch.write_bytes("a-target.txt", b"target-branch")
        else:
            await branch.delete("a-target.txt")
        await branch.write_bytes("z-later.txt", b"later-branch")
        changes = await branch.changes()

        target.chmod(0o600)
        mutation_calls = 0
        original_replace = branch_module.replace_regular_if_revision
        original_delete = branch_module.delete_regular_if_revision

        def fail_second(operation):
            def wrapped(*args, **kwargs):
                nonlocal mutation_calls
                mutation_calls += 1
                if mutation_calls == 2:
                    raise OSError("injected publication failure")
                return operation(*args, **kwargs)

            return wrapped

        with monkeypatch.context() as scoped:
            scoped.setattr(
                branch_module,
                "replace_regular_if_revision",
                fail_second(original_replace),
            )
            scoped.setattr(
                branch_module,
                "delete_regular_if_revision",
                fail_second(original_delete),
            )
            with pytest.raises(WorkspaceBranchPublicationError):
                await branch.publish(
                    WorkspaceBranchPublicationRequest(
                        branch_id=branch.branch_id,
                        baseline_revision=changes.baseline_revision,
                        change_set_digest=changes.digest,
                    )
                )

        assert target.read_bytes() == b"target-baseline"
        assert target.stat().st_mode & 0o7777 == 0o600
        assert later.read_bytes() == b"later-baseline"
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.ACTIVE
        await branch.rollback()

    asyncio.run(scenario())


def test_publication_prepares_bounded_commit_evidence_before_mutation(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"must-not-publish")
        changes = await branch.changes()
        original_evidence = branch._evidence

        def fail_commit_evidence(outcome, *args, **kwargs):
            if outcome is WorkspaceBranchOutcomeStatus.COMMITTED:
                raise WorkspaceBranchResourceExhaustedError("result_evidence_limit_exceeded")
            return original_evidence(outcome, *args, **kwargs)

        monkeypatch.setattr(branch, "_evidence", fail_commit_evidence)
        result = await branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=branch.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )
        assert result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert (await source.read_bytes("original.txt")).content == b"original"
        await branch.rollback()

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal_constructor", ["result", "receipt"])
def test_publication_prepares_complete_terminal_state_before_mutation(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_constructor: str,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"prepared-terminal")
        changes = await branch.changes()
        publication = WorkspaceBranchPublicationRequest(
            branch_id=branch.branch_id,
            baseline_revision=changes.baseline_revision,
            change_set_digest=changes.digest,
        )
        source_mutations = 0
        original_replace = branch_module.replace_regular_if_revision

        def record_replace(*args, **kwargs):
            nonlocal source_mutations
            source_mutations += 1
            return original_replace(*args, **kwargs)

        monkeypatch.setattr(branch_module, "replace_regular_if_revision", record_replace)
        with monkeypatch.context() as scoped:
            if terminal_constructor == "result":
                original_result = branch_module.WorkspaceBranchPublicationResult

                def fail_result(*args, **kwargs):
                    if kwargs.get("status") is WorkspaceBranchOutcomeStatus.COMMITTED:
                        raise MemoryError("injected result allocation failure")
                    return original_result(*args, **kwargs)

                scoped.setattr(branch_module, "WorkspaceBranchPublicationResult", fail_result)
            else:

                def fail_receipt(*args, **kwargs):
                    raise MemoryError("injected receipt allocation failure")

                scoped.setattr(branch_module, "_PublicationReceipt", fail_receipt)

            with pytest.raises(MemoryError, match="allocation failure"):
                await branch.publish(publication)

        assert source_mutations == 0
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.ACTIVE
        assert (await source.read_bytes("original.txt")).content == b"original"

        committed = await branch.publish(publication)
        assert committed.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert source_mutations == 1
        assert (await source.read_bytes("original.txt")).content == b"prepared-terminal"

    asyncio.run(scenario())


def test_publication_commit_then_raise_restores_ambiguous_attempt(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    original = branch_module.replace_regular_if_revision

    def commit_then_raise(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("lost publication acknowledgement")

    monkeypatch.setattr(branch_module, "replace_regular_if_revision", commit_then_raise)

    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"ambiguous")
        changes = await branch.changes()
        with pytest.raises(WorkspaceBranchPublicationError):
            await branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                )
            )
        assert (await source.read_bytes("original.txt")).content == b"original"
        await branch.rollback()

    asyncio.run(scenario())


def test_publication_staging_cleanup_failure_fences_and_retains_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module
    import cayu.workspaces._local_guard as guard_module

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        await branch.write_bytes("created.txt", b"private-branch-content")
        changes = await branch.changes()
        original_unlink = guard_module.os.unlink

        def reject_staging_unlink(path, *args, **kwargs):
            if str(path).startswith(".created.txt.cayu-"):
                raise OSError("injected staging cleanup failure")
            return original_unlink(path, *args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(guard_module.os, "unlink", reject_staging_unlink)
            scoped.setattr(branch_module, "_schedule_source_staging_cleanup", lambda _error: None)
            with pytest.raises(WorkspaceBranchPublicationError, match="rollback was incomplete"):
                await branch.publish(
                    WorkspaceBranchPublicationRequest(
                        branch_id=branch.branch_id,
                        baseline_revision=changes.baseline_revision,
                        change_set_digest=changes.digest,
                    )
                )
            assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED
            assert not (tmp_path / "created.txt").exists()
            staging_paths = tuple(tmp_path.glob(".created.txt.cayu-*"))
            assert len(staging_paths) == 1
            assert staging_paths[0].read_bytes() == b"private-branch-content"
            retained = tuple(branch_module._SOURCE_STAGING_CLEANUPS)
            assert len(retained) == 1

        branch_module._retry_source_staging_cleanup(retained[0])
        assert not tuple(tmp_path.glob(".created.txt.cayu-*"))
        assert not branch_module._SOURCE_STAGING_CLEANUPS
        with pytest.raises(WorkspaceBranchFencedError):
            await source.read_bytes("created.txt")

    asyncio.run(scenario())


def test_source_staging_cleanup_timer_failure_is_assisted_on_next_entrance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module
    import cayu.workspaces._local_guard as guard_module

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        await branch.write_bytes("created.txt", b"private-branch-content")
        changes = await branch.changes()
        original_unlink = guard_module.os.unlink

        def reject_staging_unlink(path, *args, **kwargs):
            if str(path).startswith(".created.txt.cayu-"):
                raise OSError("injected staging cleanup failure")
            return original_unlink(path, *args, **kwargs)

        def reject_timer_start(_timer) -> None:
            raise RuntimeError("injected timer start failure")

        with monkeypatch.context() as scoped:
            scoped.setattr(guard_module.os, "unlink", reject_staging_unlink)
            scoped.setattr(threading.Timer, "start", reject_timer_start)
            with pytest.raises(WorkspaceBranchPublicationError, match="rollback was incomplete"):
                await branch.publish(
                    WorkspaceBranchPublicationRequest(
                        branch_id=branch.branch_id,
                        baseline_revision=changes.baseline_revision,
                        change_set_digest=changes.digest,
                    )
                )

        staging_paths = tuple(tmp_path.glob(".created.txt.cayu-*"))
        assert len(staging_paths) == 1
        retained = tuple(
            (error, record)
            for error, record in branch_module._SOURCE_STAGING_CLEANUPS.items()
            if record.source_key == source.resource_key
        )
        assert len(retained) == 1
        assert retained[0][1].claimed is False

        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(request)

        assert not tuple(tmp_path.glob(".created.txt.cayu-*"))
        assert retained[0][0] not in branch_module._SOURCE_STAGING_CLEANUPS

    asyncio.run(scenario())


def test_local_create_preserves_primary_and_fatal_staging_cleanup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_guard as guard_module

    (tmp_path / "existing.txt").write_bytes(b"original")
    original_unlink = guard_module.os.unlink
    cleanup_calls = 0

    def interrupt_staging_cleanup(path, *args, **kwargs):
        nonlocal cleanup_calls
        if str(path).startswith(".existing.txt.cayu-"):
            cleanup_calls += 1
            if cleanup_calls == 1:
                raise GeneratorExit("first cleanup control signal")
            if cleanup_calls == 2:
                original_unlink(path, *args, **kwargs)
                raise SystemExit("second cleanup control signal")
        return original_unlink(path, *args, **kwargs)

    async def scenario() -> None:
        source = LocalWorkspace(tmp_path)
        with monkeypatch.context() as scoped:
            scoped.setattr(guard_module.os, "unlink", interrupt_staging_cleanup)
            with pytest.raises(BaseExceptionGroup) as raised:
                await source.create_bytes("existing.txt", b"replacement")

        assert tuple(type(error) for error in raised.value.exceptions) == (
            FileExistsError,
            GeneratorExit,
            SystemExit,
        )
        assert [str(error) for error in raised.value.exceptions[1:]] == [
            "first cleanup control signal",
            "second cleanup control signal",
        ]
        assert (tmp_path / "existing.txt").read_bytes() == b"original"
        assert not tuple(tmp_path.glob(".existing.txt.cayu-*"))

    asyncio.run(scenario())


def test_branch_mutation_preserves_fatal_signal_after_staging_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_guard as guard_module

    original_unlink = guard_module.os.unlink
    interrupted = False

    def interrupt_after_staging_cleanup(path, *args, **kwargs):
        nonlocal interrupted
        if str(path).startswith(".created.txt.cayu-") and not interrupted:
            interrupted = True
            original_unlink(path, *args, **kwargs)
            raise GeneratorExit("cleanup control signal")
        return original_unlink(path, *args, **kwargs)

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        with monkeypatch.context() as scoped:
            scoped.setattr(guard_module.os, "unlink", interrupt_after_staging_cleanup)
            with pytest.raises(GeneratorExit, match="cleanup control signal"):
                await branch.write_bytes("created.txt", b"created")

        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.ACTIVE
        assert (await branch.read_bytes("created.txt")).content == b"created"
        changes = await branch.changes()
        assert [(change.path, change.operation) for change in changes.changes] == [
            ("created.txt", "created")
        ]
        await branch.rollback()

    asyncio.run(scenario())


def test_private_overlay_mutations_reconcile_commit_then_raise(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)

        original_write = branch_module.write_regular
        write_failed = False

        def write_then_raise(*args, **kwargs):
            nonlocal write_failed
            result = original_write(*args, **kwargs)
            if args[0] == branch._overlay_root and not write_failed:
                write_failed = True
                raise OSError("lost private write acknowledgement")
            return result

        monkeypatch.setattr(branch_module, "write_regular", write_then_raise)
        with pytest.raises(OSError, match="private write acknowledgement"):
            await branch.write_bytes("original.txt", b"reconciled")
        assert (await branch.read_bytes("original.txt")).content == b"reconciled"

        await branch.create_bytes("created.txt", b"created")
        original_delete = branch_module.delete_regular
        delete_failed = False

        def delete_then_raise(*args, **kwargs):
            nonlocal delete_failed
            result = original_delete(*args, **kwargs)
            if args[0] == branch._overlay_root and not delete_failed:
                delete_failed = True
                raise OSError("lost private delete acknowledgement")
            return result

        monkeypatch.setattr(branch_module, "delete_regular", delete_then_raise)
        with pytest.raises(OSError, match="private delete acknowledgement"):
            await branch.delete("created.txt")
        with pytest.raises(FileNotFoundError):
            await branch.read_bytes("created.txt")
        assert [(change.path, change.operation) for change in (await branch.changes()).changes] == [
            ("original.txt", "modified")
        ]
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.ACTIVE
        await branch.rollback()

    asyncio.run(scenario())


def test_private_staging_cleanup_failure_fences_and_transfers_tree_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module
    import cayu.workspaces._local_guard as guard_module

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)
        branch = await _created_branch(source, request)
        private_root = branch._private_root
        original_unlink = guard_module.os.unlink

        def reject_staging_unlink(path, *args, **kwargs):
            if str(path).startswith(".private.txt.cayu-"):
                raise OSError("injected private staging cleanup failure")
            return original_unlink(path, *args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(guard_module.os, "unlink", reject_staging_unlink)
            scoped.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_args: None)
            with pytest.raises(
                guard_module._LocalGuardStagingCleanupError,
                match="staging cleanup did not complete",
            ):
                await branch.write_bytes("private.txt", b"private-content")
            assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED
            assert private_root.exists()
            assert len(branch_module._PRIVATE_TREE_CLEANUPS) == 1

        branch_module._retry_retained_private_tree_cleanup(
            private_root,
            branch._capacity_lease,
        )
        assert not private_root.exists()
        assert not branch_module._PRIVATE_TREE_CLEANUPS
        with pytest.raises(WorkspaceBranchFencedError):
            await branch.read_bytes("private.txt")

    asyncio.run(scenario())


def test_committed_result_survives_private_cleanup_failure(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    original = branch_module._remove_private_tree
    calls = 0

    def fail_once(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected private cleanup failure")
        return original(path)

    monkeypatch.setattr(branch_module, "_remove_private_tree", fail_once)

    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"committed")
        changes = await branch.changes()
        publication = WorkspaceBranchPublicationRequest(
            branch_id=branch.branch_id,
            baseline_revision=changes.baseline_revision,
            change_set_digest=changes.digest,
        )
        first = await branch.publish(publication)
        second = await branch.publish(publication)
        assert first == second
        assert first.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (await source.read_bytes("original.txt")).content == b"committed"
        branch._cleanup_committed()

    asyncio.run(scenario())


def test_committed_cleanup_survives_retry_timer_start_failure(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    original_remove = branch_module._remove_private_tree
    original_timer_start = threading.Timer.start
    remove_calls = 0

    def fail_once(path):
        nonlocal remove_calls
        remove_calls += 1
        if remove_calls == 1:
            raise OSError("injected committed cleanup failure")
        return original_remove(path)

    def reject_timer_start(self):
        raise RuntimeError("can't start new thread")

    async def scenario() -> None:
        source, request = await _source_and_request(
            populated_root,
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"committed")
        changes = await branch.changes()
        publication = WorkspaceBranchPublicationRequest(
            branch_id=branch.branch_id,
            baseline_revision=changes.baseline_revision,
            change_set_digest=changes.digest,
        )

        monkeypatch.setattr(branch_module, "_remove_private_tree", fail_once)
        monkeypatch.setattr(threading.Timer, "start", reject_timer_start)
        first = await branch.publish(publication)
        assert first.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert len(branch_module._PRIVATE_TREE_CLEANUPS) == 1

        replay = await branch.publish(publication)
        assert replay == first
        assert not branch_module._PRIVATE_TREE_CLEANUPS

        monkeypatch.setattr(threading.Timer, "start", original_timer_start)
        replacement = await source.create_branch(
            WorkspaceBranchRequest(
                baseline=await _observation(source),
                limits=request.limits,
            )
        )
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None
        await replacement.branch.rollback()

    asyncio.run(scenario())


def test_abandoned_creation_cleanup_survives_retry_timer_start_failure(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    capture_entered = threading.Event()
    release_capture = threading.Event()
    original_capture = branch_module._capture_baseline
    original_remove = branch_module._remove_private_tree
    original_timer_start = threading.Timer.start
    remove_calls = 0

    def delayed_capture(*args):
        capture_entered.set()
        assert release_capture.wait(timeout=5)
        return original_capture(*args)

    def fail_once(path):
        nonlocal remove_calls
        remove_calls += 1
        if remove_calls == 1:
            raise OSError("injected abandoned cleanup failure")
        return original_remove(path)

    def reject_cleanup_timer_start(self):
        if self.interval == 1.0:
            raise RuntimeError("can't start new thread")
        return original_timer_start(self)

    monkeypatch.setattr(branch_module, "_capture_baseline", delayed_capture)
    monkeypatch.setattr(branch_module, "_remove_private_tree", fail_once)
    monkeypatch.setattr(threading.Timer, "start", reject_cleanup_timer_start)

    async def scenario() -> None:
        source, request = await _source_and_request(
            populated_root,
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )
        task = asyncio.create_task(source.create_branch(request))
        await asyncio.to_thread(capture_entered.wait, 5)
        assert task.cancel()
        assert task.cancelling() == 1
        release_capture.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert task.cancelled()
        assert task.cancelling() == 1
        assert isinstance(raised.value.__cause__, OSError)
        assert len(branch_module._PRIVATE_TREE_CLEANUPS) == 1

        replacement = await source.create_branch(request)
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None
        assert not branch_module._PRIVATE_TREE_CLEANUPS
        await replacement.branch.rollback()

    asyncio.run(scenario())


def test_rollback_cleanup_failure_is_retryable(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    original = branch_module._remove_private_tree
    calls = 0

    def fail_once(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected rollback cleanup failure")
        return original(path)

    monkeypatch.setattr(branch_module, "_remove_private_tree", fail_once)

    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"discard")
        with pytest.raises(OSError, match="rollback cleanup"):
            await branch.rollback()
        result = await branch.rollback()
        assert result.status is WorkspaceBranchOutcomeStatus.ROLLED_BACK
        assert (await source.read_bytes("original.txt")).content == b"original"

    asyncio.run(scenario())


def test_rollback_retry_joins_retained_cleanup_owner(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    original_remove = branch_module._remove_private_tree
    original_release = branch_module._release_branch_attachment
    background_entered = threading.Event()
    rollback_join_entered = threading.Event()
    release_background = threading.Event()
    removal_lock = threading.Lock()
    removal_calls = 0
    active_removals = 0
    max_active_removals = 0
    existing_removals = 0
    release_calls = 0

    def controlled_remove(path):
        nonlocal removal_calls, active_removals, max_active_removals, existing_removals
        with removal_lock:
            removal_calls += 1
            call = removal_calls
            active_removals += 1
            max_active_removals = max(max_active_removals, active_removals)
        try:
            if call == 1:
                raise OSError("injected rollback cleanup failure")
            if call == 2:
                background_entered.set()
                assert release_background.wait(timeout=5)
            if path.exists():
                existing_removals += 1
            return original_remove(path)
        finally:
            with removal_lock:
                active_removals -= 1

    def count_release(owner, *, terminal):
        nonlocal release_calls
        release_calls += 1
        original_release(owner, terminal=terminal)

    monkeypatch.setattr(branch_module, "_remove_private_tree", controlled_remove)
    monkeypatch.setattr(branch_module, "_release_branch_attachment", count_release)
    monkeypatch.setattr(branch_module, "_schedule_private_tree_cleanup", lambda *_args: None)

    async def scenario() -> None:
        source, request = await _source_and_request(
            populated_root,
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )
        branch = await _created_branch(source, request)
        private_root = branch._private_root
        cleanup_lock = branch._capacity_lease._cleanup_settlement_lock
        cleanup_lock_entries = 0
        cleanup_lock_entries_guard = threading.Lock()

        class ObservedCleanupLock:
            def __enter__(self):
                nonlocal cleanup_lock_entries
                with cleanup_lock_entries_guard:
                    cleanup_lock_entries += 1
                    if cleanup_lock_entries == 3:
                        rollback_join_entered.set()
                cleanup_lock.acquire()
                return self

            def __exit__(self, *_args):
                cleanup_lock.release()

        monkeypatch.setattr(
            branch._capacity_lease,
            "_cleanup_settlement_lock",
            ObservedCleanupLock(),
        )
        await branch.write_bytes("original.txt", b"discard")
        with pytest.raises(OSError, match="rollback cleanup"):
            await branch.rollback()
        assert len(branch_module._PRIVATE_TREE_CLEANUPS) == 1

        background_errors: list[BaseException] = []

        def run_retained_cleanup() -> None:
            try:
                branch_module._retry_retained_private_tree_cleanup(
                    private_root,
                    branch._capacity_lease,
                )
            except BaseException as error:
                background_errors.append(error)

        background = threading.Thread(target=run_retained_cleanup)
        background.start()
        assert await asyncio.to_thread(background_entered.wait, 5)
        rollback = asyncio.create_task(branch.rollback())
        assert await asyncio.to_thread(rollback_join_entered.wait, 5)
        assert not rollback.done()
        assert removal_calls == 2
        assert max_active_removals == 1

        release_background.set()
        result = await rollback
        await asyncio.to_thread(background.join, 5)
        assert not background.is_alive()
        assert not background_errors
        assert result.status is WorkspaceBranchOutcomeStatus.ROLLED_BACK
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.ROLLED_BACK
        assert not private_root.exists()
        assert existing_removals == 1
        assert max_active_removals == 1
        assert release_calls == 1
        assert not branch_module._PRIVATE_TREE_CLEANUPS
        assert await branch.rollback() == result

        replacement = await source.create_branch(request)
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None
        await replacement.branch.rollback()

    asyncio.run(scenario())


def test_publication_rollback_failure_fences_source(
    populated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("deleted.txt", b"branch-one")
        await branch.write_bytes("original.txt", b"branch-two")
        changes = await branch.changes()

        original_replace = branch_module.replace_regular_if_revision
        replace_calls = 0

        def fail_second_replace(*args, **kwargs):
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls == 2:
                raise OSError("primary publication failure")
            return original_replace(*args, **kwargs)

        restore_calls = 0

        monkeypatch.setattr(
            branch_module,
            "replace_regular_if_revision",
            fail_second_replace,
        )
        original_restore = branch_module.restore_regular

        def fail_first_restore(*args, **kwargs):
            nonlocal restore_calls
            restore_calls += 1
            if restore_calls == 1:
                raise OSError("rollback restoration failure")
            return original_restore(*args, **kwargs)

        monkeypatch.setattr(branch_module, "restore_regular", fail_first_restore)
        with pytest.raises(WorkspaceBranchPublicationError) as raised:
            await branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                )
            )
        assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED
        assert isinstance(raised.value.__cause__, BaseExceptionGroup)
        evidence = raised.value.__cause__.exceptions
        assert [str(error) for error in evidence] == [
            "primary publication failure",
            "rollback restoration failure",
        ]
        from cayu.workspaces import WorkspaceBranchFencedError

        with pytest.raises(WorkspaceBranchFencedError):
            await source.read_bytes("original.txt")

    asyncio.run(scenario())


def test_change_set_content_identities_are_deterministic(populated_root: Path) -> None:
    async def scenario() -> None:
        source, request = await _source_and_request(populated_root)
        branch = await _created_branch(source, request)
        await branch.write_bytes("original.txt", b"changed")
        first = await branch.changes()
        second = await branch.changes()
        assert first == second
        change = first.changes[0]
        assert change.before is not None
        assert change.before.sha256 == hashlib.sha256(b"original").hexdigest()
        assert change.after is not None
        assert change.after.sha256 == hashlib.sha256(b"changed").hexdigest()
        assert "changed" not in first.model_dump_json()
        await branch.rollback()

    asyncio.run(scenario())


class _UnsupportedWorkspace(Workspace):
    id = "unsupported"

    async def read_bytes(self, path: str, *, offset: int = 0, max_bytes: int | None = None):
        return WorkspaceReadResult(content=b"", total_bytes=0)

    def bounded_read_limit(self, max_bytes: int) -> int:
        return max_bytes

    async def write_bytes(self, path: str, content: bytes) -> None:
        return None

    async def delete(self, path: str) -> None:
        return None

    async def create_bytes(self, path: str, content: bytes) -> WorkspaceMutationResult:
        raise NotImplementedError

    async def replace_bytes(
        self,
        path: str,
        content: bytes,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        raise NotImplementedError

    async def delete_if_revision(
        self,
        path: str,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        raise NotImplementedError

    async def list(self, pattern: str = "**/*", *, limit: int | None = None):
        return WorkspaceListResult(paths=(), total_count=0)


def test_workspace_compatibility_default_is_typed_unsupported(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = LocalWorkspace(tmp_path, workspace_id="unsupported")
        observation = await _observation(source)
        result = await _UnsupportedWorkspace().create_branch(
            WorkspaceBranchRequest(baseline=observation)
        )
        assert result.status is WorkspaceBranchOutcomeStatus.UNSUPPORTED
        assert result.branch is None

    asyncio.run(scenario())


def test_workspace_compatibility_default_bounds_long_identity_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces.branches as branch_contracts

    original_evidence = branch_contracts.workspace_branch_evidence
    constructed_sources: list[WorkspaceIdentity] = []

    def record_bounded_evidence(**kwargs):
        source = kwargs["source"]
        constructed_sources.append(source)
        assert len(source.workspace_id) <= 71
        assert len(source.observer) <= 71
        return original_evidence(**kwargs)

    monkeypatch.setattr(branch_contracts, "workspace_branch_evidence", record_bounded_evidence)

    async def scenario() -> None:
        source = LocalWorkspace(tmp_path, workspace_id="workspace-" + "w" * 2000)
        observation = await observe_deterministic_workspace(
            source,
            observer="observer-" + "o" * 2000,
            limits=WorkspaceRevisionObservationLimits(),
        )
        result = await _UnsupportedWorkspace().create_branch(
            WorkspaceBranchRequest(
                baseline=observation,
                limits=WorkspaceBranchLimits(max_evidence_bytes=1024),
            )
        )
        assert result.status is WorkspaceBranchOutcomeStatus.UNSUPPORTED
        assert len(result.evidence.model_dump_json().encode("utf-8")) <= 1024
        assert result.evidence.source.workspace_id.startswith("sha256:")
        assert result.evidence.source.observer.startswith("sha256:")
        assert constructed_sources == [result.evidence.source]

    asyncio.run(scenario())


def test_bounded_evidence_projection_matches_exact_serialized_boundary() -> None:
    import cayu.workspaces.branches as branch_contracts

    source = WorkspaceIdentity(
        workspace_id='workspace-"-\\-雪',
        observer="observer-\n-value",
    )
    branch_id = "wsb_" + "0" * 32
    baseline_revision = "sha256:" + "1" * 64
    change_set = WorkspaceBranchChangeSet(
        branch_id=branch_id,
        source=source,
        baseline_revision=baseline_revision,
        changes=(),
        digest=workspace_branch_change_set_digest(
            branch_id=branch_id,
            source=source,
            baseline_revision=baseline_revision,
            changes=(),
        ),
    )
    assert branch_contracts._workspace_branch_empty_change_set_json_size(
        branch_id=branch_id,
        source=source,
        baseline_revision=baseline_revision,
    ) == len(change_set.model_dump_json().encode("utf-8"))

    expected = workspace_branch_evidence(
        source=source,
        outcome=WorkspaceBranchOutcomeStatus.UNSUPPORTED,
        baseline_revision=baseline_revision,
        paths=("nested/file.txt",),
        detail_code="workspace_branching_unsupported",
    )
    exact_size = len(expected.model_dump_json().encode("utf-8"))

    assert (
        branch_contracts._bounded_workspace_branch_evidence(
            source=source,
            outcome=WorkspaceBranchOutcomeStatus.UNSUPPORTED,
            baseline_revision=baseline_revision,
            paths=("nested/file.txt",),
            detail_code="workspace_branching_unsupported",
            max_bytes=exact_size,
        )
        == expected
    )
    with pytest.raises(
        WorkspaceBranchResourceExhaustedError,
        match="result_evidence_limit_exceeded",
    ):
        branch_contracts._bounded_workspace_branch_evidence(
            source=source,
            outcome=WorkspaceBranchOutcomeStatus.UNSUPPORTED,
            baseline_revision=baseline_revision,
            paths=("nested/file.txt",),
            detail_code="workspace_branching_unsupported",
            max_bytes=exact_size - 1,
        )


def test_branch_request_limits_precede_full_baseline_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces.branches as branch_contracts

    (tmp_path / "first.txt").write_bytes(b"first")
    (tmp_path / "second.txt").write_bytes(b"second")

    async def scenario() -> None:
        source = LocalWorkspace(tmp_path, workspace_id="bounded-copy")
        baseline = await _observation(source)
        request = WorkspaceBranchRequest(
            baseline=baseline,
            limits=WorkspaceBranchLimits(max_paths=1),
        )
        copy_called = False

        def reject_copy(*args, **kwargs):
            nonlocal copy_called
            copy_called = True
            raise AssertionError("full baseline copy must not run")

        monkeypatch.setattr(
            branch_contracts,
            "copy_bounded_workspace_revision_observation",
            reject_copy,
        )
        exhausted = await source.create_branch(request)
        assert exhausted.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert exhausted.evidence.detail_code == "path_count_limit_exceeded"
        assert not copy_called

        unsupported = await _UnsupportedWorkspace().create_branch(request)
        assert unsupported.status is WorkspaceBranchOutcomeStatus.UNSUPPORTED
        assert not copy_called

    asyncio.run(scenario())


def test_local_branch_translates_hard_baseline_copy_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces.branches as branch_contracts

    async def scenario() -> None:
        source, request = await _source_and_request(tmp_path)

        def exceed_hard_limit(*args, **kwargs):
            raise WorkspaceRevisionObservationLimitExceeded

        monkeypatch.setattr(
            branch_contracts,
            "copy_bounded_workspace_revision_observation",
            exceed_hard_limit,
        )
        result = await source.create_branch(request)
        assert result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert result.evidence.detail_code == "baseline_observation_limit_exceeded"

    asyncio.run(scenario())


def test_local_branch_bounds_authority_before_resource_evidence_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.workspaces.branches as branch_contracts

    original_evidence = branch_contracts.workspace_branch_evidence
    constructed_sources: list[WorkspaceIdentity] = []

    def record_bounded_evidence(**kwargs):
        source = kwargs["source"]
        constructed_sources.append(source)
        assert len(source.workspace_id) <= 71
        return original_evidence(**kwargs)

    monkeypatch.setattr(branch_contracts, "workspace_branch_evidence", record_bounded_evidence)

    async def scenario() -> None:
        source, request = await _source_and_request(
            tmp_path,
            limits=WorkspaceBranchLimits(max_evidence_bytes=1024),
        )
        oversized_workspace_id = "workspace-" + "w" * 60_000
        source.id = oversized_workspace_id
        object.__setattr__(
            request.baseline.identity,
            "workspace_id",
            oversized_workspace_id,
        )

        result = await source.create_branch(request)
        assert result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert result.evidence.detail_code == "change_evidence_limit_exceeded"
        assert result.evidence.source.workspace_id.startswith("sha256:")
        assert constructed_sources == [result.evidence.source]

    asyncio.run(scenario())


def test_local_workspace_subclass_must_prove_branching(populated_root: Path) -> None:
    class ExtensionWorkspace(LocalWorkspace):
        pass

    async def scenario() -> None:
        exact = LocalWorkspace(populated_root, workspace_id="extension")
        observation = await _observation(exact)
        extension = ExtensionWorkspace(populated_root, workspace_id="extension")
        assert extension.branch_capabilities().isolation is False
        result = await extension.create_branch(WorkspaceBranchRequest(baseline=observation))
        assert result.status is WorkspaceBranchOutcomeStatus.UNSUPPORTED

    asyncio.run(scenario())
