"""SyncBinding copies back only the files a run changed (``copy_back_scope``)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from cayu.environments.bindings import SyncBinding, SyncBindingSourceConflictError
from cayu.runners import LocalRunner
from cayu.workspaces import LocalWorkspace, RunnerWorkspace

_FILES = {f"pkg/module_{index}.py": f"VALUE = {index}\n" for index in range(12)}


def _tree(tmp_path: Path) -> tuple[Path, Path, LocalWorkspace, RunnerWorkspace]:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    (source_root / "pkg").mkdir(parents=True)
    target_root.mkdir()
    for path, content in _FILES.items():
        (source_root / path).write_text(content, encoding="utf-8")
    (source_root / "run.sh").write_text("echo hi\n", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = RunnerWorkspace(LocalRunner(target_root), cwd=None, workspace_id="target")
    return source_root, target_root, source, target


def _binding(target, **options) -> SyncBinding:
    # The coding factories' policy: revision-checked, Git-mode-preserving copy-back.
    options.setdefault("source_conflict_policy", "require_revision")
    options.setdefault("preserve_git_modes", True)
    return SyncBinding(target_workspace=target, max_file_bytes=1024 * 1024, **options)


def _tree_bytes(root: Path) -> dict[str, tuple[bytes, bool]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), os.access(path, os.X_OK))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize("policy", ["require_revision", "overwrite"])
def test_copy_back_sends_only_changed_created_and_mode_changed_files(tmp_path, policy) -> None:
    source_root, target_root, source, target = _tree(tmp_path)

    async def run():
        binding = _binding(target, source_conflict_policy=policy)
        bound = await binding.bind(source, None, session_id="changed")
        assert bound.metadata["sync_binding"]["copy_back_scope"] == "changed"
        (target_root / "pkg/module_1.py").write_text("VALUE = 'edited'\n", encoding="utf-8")
        (target_root / "pkg/module_2.py").unlink()
        (target_root / "pkg/new.py").write_text("NEW = True\n", encoding="utf-8")
        (target_root / "run.sh").chmod(0o755)
        # Rewritten with the same bytes: not a change.
        (target_root / "pkg/module_3.py").write_text("VALUE = 3\n", encoding="utf-8")
        return await binding.finalize(bound, outcome="completed")

    snapshot = asyncio.run(run())

    assert snapshot is not None
    assert snapshot.metadata["copy_back_scope"] == "changed"
    assert snapshot.metadata["copied_files"] == 3
    assert snapshot.metadata["deleted_files"] == 1
    assert snapshot.metadata["unchanged_files"] == len(_FILES) - 2
    assert _tree_bytes(source_root) == _tree_bytes(target_root)
    assert os.access(source_root / "run.sh", os.X_OK)


def test_copy_back_scope_all_copies_every_file(tmp_path) -> None:
    source_root, target_root, source, target = _tree(tmp_path)

    async def run():
        binding = _binding(target, copy_back_scope="all")
        bound = await binding.bind(source, None, session_id="all")
        (target_root / "pkg/module_1.py").write_text("VALUE = 'edited'\n", encoding="utf-8")
        return await binding.finalize(bound, outcome="completed")

    snapshot = asyncio.run(run())

    assert snapshot is not None
    assert snapshot.metadata["copy_back_scope"] == "all"
    assert snapshot.metadata["copied_files"] == len(_FILES) + 1
    assert snapshot.metadata["unchanged_files"] == 0
    assert _tree_bytes(source_root) == _tree_bytes(target_root)


def test_target_that_cannot_hash_its_files_copies_everything(tmp_path) -> None:
    source_root, target_root, source, _ = _tree(tmp_path)
    target = LocalWorkspace(target_root, workspace_id="local-target")

    async def run():
        binding = _binding(target)
        bound = await binding.bind(source, None, session_id="local")
        await target.write_bytes("pkg/module_1.py", b"VALUE = 'edited'\n")
        return await binding.finalize(bound, outcome="completed")

    snapshot = asyncio.run(run())

    assert snapshot is not None
    assert snapshot.metadata["copy_back_scope"] == "all"
    assert snapshot.metadata["copied_files"] == len(_FILES) + 1
    assert (source_root / "pkg/module_1.py").read_bytes() == b"VALUE = 'edited'\n"


@pytest.mark.parametrize("failing_capture", [1, 2])
def test_a_failed_measurement_copies_everything(tmp_path, failing_capture) -> None:
    source_root, target_root, source, target = _tree(tmp_path)
    calls = 0
    capture = target.capture_content_manifest

    async def flaky_capture(**limits):
        nonlocal calls
        calls += 1
        if calls == failing_capture:
            raise RuntimeError("Workspace changed during content observation.")
        return await capture(**limits)

    target.capture_content_manifest = flaky_capture  # type: ignore[method-assign]

    async def run():
        binding = _binding(target)
        bound = await binding.bind(source, None, session_id="flaky")
        (target_root / "pkg/module_1.py").write_text("VALUE = 'edited'\n", encoding="utf-8")
        return await binding.finalize(bound, outcome="completed")

    snapshot = asyncio.run(run())

    assert snapshot is not None
    assert snapshot.metadata["copy_back_scope"] == "all"
    assert snapshot.metadata["copied_files"] == len(_FILES) + 1
    assert _tree_bytes(source_root) == _tree_bytes(target_root)


def test_later_passes_republish_files_reverted_after_an_earlier_pass(tmp_path) -> None:
    source_root, target_root, source, target = _tree(tmp_path)

    async def run():
        binding = _binding(target)
        bound = await binding.bind(source, None, session_id="passes")
        (target_root / "pkg/module_1.py").write_text("VALUE = 'edited'\n", encoding="utf-8")
        (target_root / "pkg/module_2.py").unlink()
        first = await binding._copy_back_while_bound(bound)
        # Back to the bytes copied in: the source now holds the edit, so this
        # is a change again, as is a deleted file recreated as it was.
        (target_root / "pkg/module_1.py").write_text("VALUE = 1\n", encoding="utf-8")
        (target_root / "pkg/module_2.py").write_text("VALUE = 2\n", encoding="utf-8")
        second = await binding._copy_back_while_bound(bound)
        third = await binding.finalize(bound, outcome="completed")
        return first, second, third

    first, second, third = asyncio.run(run())

    assert (first.metadata["copied_files"], first.metadata["deleted_files"]) == (1, 1)
    assert second.metadata["copied_files"] == 2
    assert second.metadata["copy_back_scope"] == "changed"
    # Files published by an earlier pass are compared no longer, only copied.
    assert third.metadata["copied_files"] == 2
    assert third.metadata["unchanged_files"] == len(_FILES) - 1
    assert _tree_bytes(source_root) == _tree_bytes(target_root)


def test_retry_after_a_partly_applied_copy_back_republishes_applied_files(tmp_path) -> None:
    source_root, target_root, _, target = _tree(tmp_path)

    class FailOnce(LocalWorkspace):
        failed = False

        async def replace_bytes_with_git_mode(self, path, content, **options):
            if path == "pkg/module_5.py" and not self.failed:
                self.failed = True
                raise OSError("injected copy-back failure")
            return await super().replace_bytes_with_git_mode(path, content, **options)

    source = FailOnce(source_root, workspace_id="source")

    async def run():
        binding = _binding(target)
        bound = await binding.bind(source, None, session_id="retry")
        (target_root / "pkg/module_1.py").write_text("VALUE = 'edited'\n", encoding="utf-8")
        (target_root / "pkg/module_5.py").write_text("VALUE = 'mine'\n", encoding="utf-8")
        with pytest.raises(SyncBindingSourceConflictError) as conflict:
            await binding.finalize(bound, outcome="completed")
        assert conflict.value.applied_paths == ("pkg/module_1.py",)
        # The run puts back the bytes copied in: the source holds the edit, so
        # the retry must publish the revert, not skip it as unchanged.
        (target_root / "pkg/module_1.py").write_text("VALUE = 1\n", encoding="utf-8")
        return await binding.finalize(bound, outcome="completed")

    snapshot = asyncio.run(run())

    assert snapshot is not None
    assert snapshot.metadata["copied_files"] == 2
    assert _tree_bytes(source_root) == _tree_bytes(target_root)


def test_recovered_copy_back_copies_everything(tmp_path) -> None:
    source_root, target_root, source, target = _tree(tmp_path)

    async def run():
        original = _binding(target)
        bound = await original.bind(source, None, session_id="original")
        (target_root / "pkg/module_1.py").write_text("VALUE = 'edited'\n", encoding="utf-8")
        state = original._completion_finalization_recovery_state(bound)
        original.abandon(bound)
        recovered_binding = _binding(target)
        recovered = await recovered_binding._recover_completion_finalization(
            source,
            None,
            session_id="recovered",
            agent_name=None,
            environment_name=None,
            recovery_state=state,
        )
        return await recovered_binding.finalize(recovered, outcome="completed")

    snapshot = asyncio.run(run())

    assert snapshot is not None
    assert snapshot.metadata["copy_back_scope"] == "all"
    assert snapshot.metadata["copied_files"] == len(_FILES) + 1
    assert _tree_bytes(source_root) == _tree_bytes(target_root)


def test_copy_back_scope_is_validated() -> None:
    with pytest.raises(ValueError, match="copy_back_scope"):
        SyncBinding(copy_back_scope="some")  # type: ignore[arg-type]
