"""SyncBinding copies in only the files a kept target doesn't already hold (``copy_in_scope``)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from cayu.environments.bindings import SyncBinding
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
    (source_root / "run.sh").chmod(0o755)
    source = LocalWorkspace(source_root, workspace_id="source")
    target = RunnerWorkspace(LocalRunner(target_root), cwd=None, workspace_id="target")
    return source_root, target_root, source, target


def _binding(target, **options) -> SyncBinding:
    options.setdefault("source_conflict_policy", "require_revision")
    options.setdefault("preserve_git_modes", True)
    options.setdefault("copy_in_scope", "changed")
    return SyncBinding(target_workspace=target, max_file_bytes=1024 * 1024, **options)


def _tree_bytes(root: Path) -> dict[str, tuple[bytes, bool]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), os.access(path, os.X_OK))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_copy_in_sends_only_what_the_target_lacks_and_removes_the_rest(tmp_path) -> None:
    source_root, target_root, source, target = _tree(tmp_path)

    async def run():
        binding = _binding(target, sync_back="never")
        first = await binding.bind(source, None, session_id="first")
        await binding.finalize(first, outcome="completed")
        # What a run left in the kept target, and a change made to the source meanwhile.
        (target_root / "pkg/module_1.py").write_text("VALUE = 'edited'\n", encoding="utf-8")
        (target_root / "pkg/module_2.py").unlink()
        (target_root / "pkg/new.py").write_text("NEW = True\n", encoding="utf-8")
        (target_root / "run.sh").chmod(0o644)
        (source_root / "pkg/module_4.py").write_text("VALUE = 'source'\n", encoding="utf-8")
        second = await binding.bind(source, None, session_id="second")
        await binding.finalize(second, outcome="completed")
        return first, second

    first, second = asyncio.run(run())
    assert first.metadata["sync_binding"]["transferred_files"] == len(_FILES) + 1
    sync = second.metadata["sync_binding"]
    assert sync["copy_in_scope"] == "changed"
    # module_1, module_2, module_4 and run.sh crossed; new.py went.
    assert (sync["transferred_files"], sync["unchanged_files"]) == (4, len(_FILES) - 3)
    assert sync["copied_files"] == len(_FILES) + 1
    # The clean removed new.py and the three target files that differed.
    assert sync["cleaned_target_files"] == 4
    assert _tree_bytes(source_root) == _tree_bytes(target_root)
    assert os.access(target_root / "run.sh", os.X_OK)


def test_copy_back_after_a_delta_copy_in_sends_only_the_runs_changes(tmp_path) -> None:
    source_root, target_root, source, target = _tree(tmp_path)

    async def run():
        binding = _binding(target)
        first = await binding.bind(source, None, session_id="first")
        await binding.finalize(first, outcome="completed")
        second = await binding.bind(source, None, session_id="second")
        (target_root / "pkg/module_5.py").write_text("VALUE = 'run'\n", encoding="utf-8")
        snapshot = await binding.finalize(second, outcome="completed")
        return second, snapshot

    second, snapshot = asyncio.run(run())
    assert second.metadata["sync_binding"]["transferred_files"] == 0
    assert snapshot is not None and snapshot.metadata["copy_back_scope"] == "changed"
    assert snapshot.metadata["copied_files"] == 1
    assert (source_root / "pkg/module_5.py").read_text(encoding="utf-8") == "VALUE = 'run'\n"


def test_copy_in_scope_all_copies_every_file_into_a_kept_target(tmp_path) -> None:
    _, _, source, target = _tree(tmp_path)

    async def run():
        binding = _binding(target, copy_in_scope="all")
        first = await binding.bind(source, None, session_id="first")
        await binding.finalize(first, outcome="completed")
        second = await binding.bind(source, None, session_id="second")
        await binding.finalize(second, outcome="completed")
        return second

    sync = asyncio.run(run()).metadata["sync_binding"]
    assert "copy_in_scope" not in sync
    assert sync["cleaned_target_files"] == len(_FILES) + 1


def test_copy_in_scope_needs_source_hashes_and_git_modes(tmp_path) -> None:
    _, _, _, target = _tree(tmp_path)
    with pytest.raises(ValueError, match="copy_in_scope"):
        SyncBinding(copy_in_scope="some")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="require_revision"):
        _binding(target, source_conflict_policy="overwrite")
    with pytest.raises(ValueError, match="preserve_git_modes"):
        _binding(target, preserve_git_modes=False)
