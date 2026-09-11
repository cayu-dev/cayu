from __future__ import annotations

import asyncio
import os
import sys

import pytest
from tests.workspaces.test_runner_workspace import _ListResultRunner

from cayu.runners import LocalRunner
from cayu.workspaces import LocalWorkspace, RunnerWorkspace
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservationLimits,
    observe_deterministic_workspace,
)


def observe(workspace, **limits):
    return asyncio.run(
        observe_deterministic_workspace(
            workspace, observer="test", limits=WorkspaceRevisionObservationLimits(**limits)
        )
    )


class CountingRunner(LocalRunner):
    calls = 0

    async def exec(self, *args, **kwargs):
        self.calls += 1
        return await super().exec(*args, **kwargs)


@pytest.mark.parametrize("count", [0, 10, 548, 1000])
def test_bulk_revision_matches_regular_file_identity_in_one_call(tmp_path, count):
    for index in range(count):
        path = tmp_path / f"file-{index}.txt"
        path.write_bytes(b"" if index == 0 else b"content\n")
        if index == 1:
            path.chmod(0o755)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/private").write_text("excluded")
    (tmp_path / "ignored.txt").write_text("excluded")
    (tmp_path / "link").symlink_to("../not-readable")
    if hasattr(os, "mkfifo"):
        os.mkfifo(tmp_path / "fifo")
    options = dict(
        workspace_id="same",
        excluded_directory_names=(".git",),
        excluded_path_patterns=("ignored.txt",),
    )
    runner = CountingRunner(tmp_path, inherit_env=False)
    remote = RunnerWorkspace(runner, python_executable=sys.executable, **options)
    result = observe(remote)
    assert runner.calls == 1
    assert result == observe(LocalWorkspace(tmp_path, **options))
    assert result.status == "supported" and result.total_paths == count


@pytest.mark.parametrize(
    ("limits", "code"),
    [
        ({"max_paths": 1}, "path_count_limit_exceeded"),
        ({"max_path_bytes": 1}, "path_byte_limit_exceeded"),
        ({"max_file_bytes": 2}, "file_byte_limit_exceeded"),
        ({"max_total_file_bytes": 4}, "total_file_byte_limit_exceeded"),
    ],
)
def test_bulk_limits_never_return_partial_revision(tmp_path, limits, code):
    (tmp_path / "aa").write_bytes(b"abc")
    (tmp_path / "bb").write_bytes(b"def")
    runner = CountingRunner(tmp_path, inherit_env=False)
    result = observe(RunnerWorkspace(runner, python_executable=sys.executable), **limits)
    assert runner.calls == 1
    assert result.status == "truncated" and result.detail_code == code
    assert result.revision is None and result.paths == ()


def test_bulk_exact_byte_limit_allows_trailing_empty_file(tmp_path):
    (tmp_path / "a").write_bytes(b"abc")
    (tmp_path / "z").write_bytes(b"")
    result = observe(
        RunnerWorkspace(LocalRunner(tmp_path), python_executable=sys.executable),
        max_file_bytes=3,
        max_total_file_bytes=3,
    )
    assert result.status == "supported" and result.total_paths == 2


def test_bulk_canonical_manifest_limit(tmp_path):
    for index in range(15):
        (tmp_path / str(index)).write_bytes(b"")
    result = observe(
        RunnerWorkspace(LocalRunner(tmp_path), python_executable=sys.executable),
        max_manifest_bytes=1024,
    )
    assert result.status == "truncated"
    assert result.detail_code == "manifest_byte_limit_exceeded"
    assert result.revision is None


@pytest.mark.parametrize(
    "payload",
    [
        {"ok": True, "entries": [["../escape", "a" * 64, 0, "100644"]], "total_bytes": 0},
        {"ok": True, "entries": [[".git/private", "a" * 64, 0, "100644"]], "total_bytes": 0},
        {"ok": True, "entries": [["link", "a" * 64, 0, "120000"]], "total_bytes": 0},
        {"ok": True, "entries": [["a", "a" * 64, 0, "100644"]] * 2, "total_bytes": 0},
        {"ok": True, "entries": [], "total_bytes": True},
        {"ok": True, "limit": "arbitrary"},
        {"ok": True, "limit": "file_byte_limit_exceeded", "entries": []},
    ],
)
def test_invalid_bulk_evidence_fails_closed_without_serial_fallback(payload):
    runner = _ListResultRunner(payload)
    result = observe(RunnerWorkspace(runner, excluded_directory_names=(".git",)))
    assert result.status == "incomplete" and result.revision is None
    assert runner.exec_calls == 1


def test_bulk_cancellation_propagates():
    class CancelledRunner(_ListResultRunner):
        async def exec(self, *args, **kwargs):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        observe(RunnerWorkspace(CancelledRunner({})))


def test_bulk_detects_same_size_change_after_hash(tmp_path, monkeypatch):
    import cayu.workspaces.runner as module

    (tmp_path / "file").write_bytes(b"before")
    program = module._RUNNER_WORKSPACE_PROGRAM.replace(
        "    if listing() != before:",
        '    with open("file", "wb") as changed:\n        changed.write(b"after!")\n    if listing() != before:',
    )
    monkeypatch.setattr(module, "_RUNNER_WORKSPACE_PROGRAM", program)
    result = observe(RunnerWorkspace(LocalRunner(tmp_path), python_executable=sys.executable))
    assert result.status == "incomplete" and result.revision is None


def test_subclass_custom_read_semantics_are_not_bypassed(tmp_path):
    class RedactedWorkspace(RunnerWorkspace):
        async def read_bytes(self, *args, **kwargs):
            raise RuntimeError("Custom read policy")

    (tmp_path / "file").write_bytes(b"private")
    result = observe(RedactedWorkspace(LocalRunner(tmp_path), python_executable=sys.executable))
    assert result.status == "incomplete"
    assert result.detail_code == "workspace_file_read_failed"


def test_bulk_rejects_symlink_swap_before_read(tmp_path, monkeypatch):
    import cayu.workspaces.runner as module

    (tmp_path / "file").write_bytes(b"before")
    program = module._RUNNER_WORKSPACE_PROGRAM.replace(
        "    before = listing()",
        '    before = listing()\n    os.unlink("file")\n    os.symlink("../outside", "file")',
    )
    monkeypatch.setattr(module, "_RUNNER_WORKSPACE_PROGRAM", program)
    result = observe(RunnerWorkspace(LocalRunner(tmp_path), python_executable=sys.executable))
    assert result.status == "incomplete" and result.revision is None


def test_bulk_rejects_tree_membership_change_after_hash(tmp_path, monkeypatch):
    import cayu.workspaces.runner as module

    (tmp_path / "file").write_bytes(b"before")
    program = module._RUNNER_WORKSPACE_PROGRAM.replace(
        "    if listing() != before:",
        '    os.rename("file", "renamed")\n    if listing() != before:',
    )
    monkeypatch.setattr(module, "_RUNNER_WORKSPACE_PROGRAM", program)
    result = observe(RunnerWorkspace(LocalRunner(tmp_path), python_executable=sys.executable))
    assert result.status == "incomplete" and result.revision is None
