from __future__ import annotations

import asyncio
import errno
import os
import shutil
from collections import Counter
from pathlib import Path

import pytest

from cayu.workspaces import LocalWorkspace
from cayu.workspaces import local as local_workspace

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX descriptor traversal")

# Names around "/" in code-point order ("-", ".", " ", "!" sort before it and
# "0" after it), so a plain per-directory sort would give the wrong order.
_TRICKY_FILES = (
    "a-b/x.txt",
    "a/b.txt",
    "a.txt",
    "a/c/d.txt",
    "a b/e.txt",
    "a.b/f",
    "a!/g",
    "a0/h",
    "ab",
    ".hidden/q",
    "deep/1/2/3/4/r",
    "z",
)


def _write(root: Path, relative_paths: tuple[str, ...]) -> None:
    for relative_path in relative_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")


def _list(workspace: LocalWorkspace, pattern: str = "**/*", limit: int | None = None):
    return asyncio.run(workspace.list(pattern, limit=limit))


def _count_calls(monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for name in names:
        original = getattr(os, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(os, name, counted)
    return counts


def _record_directory_opens(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    opened: list[str] = []
    original = os.open

    def recording_open(path, flags, *args, **kwargs):
        if flags & os.O_DIRECTORY:
            opened.append(os.fsdecode(path))
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)
    return opened


@pytest.mark.parametrize("exclusions", [False, True])
def test_list_returns_paths_in_sorted_order_for_every_limit(tmp_path, exclusions) -> None:
    _write(tmp_path, _TRICKY_FILES)
    excluded = ("unused-name",) if exclusions else ()
    workspace = LocalWorkspace(tmp_path, excluded_directory_names=excluded)
    expected = tuple(sorted(_TRICKY_FILES))

    full = _list(workspace)
    assert full.paths == expected
    assert full.total_count == len(expected)
    assert full.truncated is False

    for limit in range(1, len(expected)):
        page = _list(workspace, limit=limit)
        assert page.paths == expected[:limit]
        assert page.total_count is None
        assert page.truncated is True

    exact = _list(workspace, limit=len(expected))
    assert exact.paths == expected
    assert exact.total_count == len(expected)
    assert exact.truncated is False


def test_list_limit_applies_after_pattern_filtering(tmp_path) -> None:
    _write(tmp_path, ("a/1.py", "a/2.txt", "b/3.py", "c/4.txt", "d/5.py"))
    workspace = LocalWorkspace(tmp_path)

    page = _list(workspace, "**/*.py", limit=2)

    assert page.paths == ("a/1.py", "b/3.py")
    assert page.truncated is True
    assert _list(workspace, "**/*.py", limit=3).truncated is False


def test_list_stops_walking_once_the_limit_is_settled(tmp_path, monkeypatch) -> None:
    _write(tmp_path, tuple(f"dir{index:02}/file.txt" for index in range(50)))
    workspace = LocalWorkspace(tmp_path)
    opened = _record_directory_opens(monkeypatch)

    page = _list(workspace, limit=2)

    assert page.paths == ("dir00/file.txt", "dir01/file.txt")
    assert page.truncated is True
    walked = [path for path in opened if path.startswith("dir")]
    # Two directories fill the page and the third proves there is more.
    assert walked == ["dir00", "dir01", "dir02"]


def test_list_reads_each_directory_once_without_per_file_stat(tmp_path, monkeypatch) -> None:
    small = tmp_path / "small"
    large = tmp_path / "large"
    for root, count in ((small, 3), (large, 300)):
        _write(root, tuple(f"pkg/sub/file{index:03}.txt" for index in range(count)))

    measured = {}
    for root in (small, large):
        workspace = LocalWorkspace(root)
        with monkeypatch.context() as patch:
            counts = _count_calls(patch, ("stat", "lstat", "open", "scandir", "readlink"))
            listed = _list(workspace)
        assert listed.truncated is False
        measured[root.name] = dict(counts)

    # Filesystem calls depend on the number of directories, not files.
    assert measured["small"] == measured["large"]
    assert measured["large"].get("lstat", 0) == 0
    assert measured["large"]["scandir"] == 3


def test_list_skips_symlinks_and_special_files(tmp_path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    _write(root, ("kept/inner.txt", "top.txt"))
    _write(outside, ("secret.txt", "nested/secret.txt"))
    (root / "dir-link").symlink_to(outside, target_is_directory=True)
    (root / "file-link").symlink_to(outside / "secret.txt")
    (root / "kept" / "inside-link").symlink_to(root / "top.txt")
    (root / "broken-link").symlink_to(root / "missing")
    os.mkfifo(root / "fifo")
    (root / "empty").mkdir()

    for excluded in ((), ("unused-name",)):
        listed = _list(LocalWorkspace(root, excluded_directory_names=excluded))
        assert listed.paths == ("kept/inner.txt", "top.txt")


def test_list_does_not_follow_a_directory_swapped_for_a_symlink(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    _write(root, ("a.txt", "victim/own.txt", "z.txt"))
    _write(outside, ("secret.txt",))
    workspace = LocalWorkspace(root)
    original = os.open
    swapped: list[bool] = []

    def swapping_open(path, flags, *args, **kwargs):
        if os.fsdecode(path) == "victim" and not swapped:
            swapped.append(True)
            shutil.rmtree(root / "victim")
            (root / "victim").symlink_to(outside, target_is_directory=True)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swapping_open)

    listed = _list(workspace)

    assert swapped
    assert listed.paths == ("a.txt", "z.txt")


def test_list_rejects_a_root_replaced_by_a_symlink(tmp_path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    _write(root, ("a.txt",))
    _write(outside, ("secret.txt",))
    workspace = LocalWorkspace(root)
    root.rename(tmp_path / "moved")
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes the workspace root"):
        _list(workspace)


def test_list_never_opens_excluded_directories(tmp_path, monkeypatch) -> None:
    _write(
        tmp_path,
        (
            "node_modules/pkg/index.js",
            "src/Node_Modules./deep.js",
            "src/node_modules",
            "src/gen/out.py",
            "src/keep/gen/kept.py",
            "src/main.py",
        ),
    )
    workspace = LocalWorkspace(
        tmp_path,
        excluded_directory_names=("node_modules",),
        excluded_path_patterns=("src/gen",),
    )
    opened = _record_directory_opens(monkeypatch)

    listed = _list(workspace)

    assert listed.paths == ("src/keep/gen/kept.py", "src/main.py")
    root_components = len(workspace.root.parts)
    assert opened[root_components:] == ["src", "keep", "gen"]


def test_unreadable_directory_is_skipped_without_exclusions_and_raised_with_them(
    tmp_path, monkeypatch
) -> None:
    _write(tmp_path, ("a.txt", "b.txt", "locked/hidden.txt"))
    original = os.open

    def denying_open(path, flags, *args, **kwargs):
        if os.fsdecode(path) == "locked":
            raise PermissionError(errno.EACCES, "Permission denied", "locked")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", denying_open)

    assert _list(LocalWorkspace(tmp_path)).paths == ("a.txt", "b.txt")
    with pytest.raises(PermissionError):
        _list(LocalWorkspace(tmp_path, excluded_directory_names=("unused-name",)))
    # A page that is settled before the walk reaches the directory succeeds.
    settled = _list(LocalWorkspace(tmp_path, excluded_directory_names=("unused-name",)), limit=1)
    assert settled.paths == ("a.txt",)
    assert settled.truncated is True


def test_list_closes_every_directory_descriptor(tmp_path, monkeypatch) -> None:
    _write(tmp_path, tuple(f"d{index}/e/f/file.txt" for index in range(5)))
    workspace = LocalWorkspace(tmp_path)
    opened: list[int] = []
    original_open = os.open

    def tracking_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if flags & os.O_DIRECTORY:
            opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", tracking_open)

    for limit in (None, 1, 3):
        opened.clear()
        _list(workspace, limit=limit)
        assert opened
        for descriptor in opened:
            with pytest.raises(OSError) as raised:
                os.fstat(descriptor)
            assert raised.value.errno == errno.EBADF


def _track_open_directories(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the most directory descriptors the walk held open at once."""
    live: set[int] = set()
    peak = [0]
    original_open = os.open
    original_close = os.close

    def tracking_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if flags & os.O_DIRECTORY:
            live.add(descriptor)
            peak[0] = max(peak[0], len(live))
        return descriptor

    def tracking_close(descriptor):
        live.discard(descriptor)
        return original_close(descriptor)

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(os, "close", tracking_close)
    return peak


def test_list_bounds_open_directories_on_deep_trees(tmp_path, monkeypatch) -> None:
    deep = "/".join(["d"] * 100)
    files = (
        "a.txt",
        f"{deep}/deep.txt",
        "d/d/d/sibling/s.txt",
        "d/d/d/zz.txt",
        "d/e/f/g/h/i/j.txt",
        "d/z.txt",
        "z.txt",
    )
    _write(tmp_path, files)
    expected = tuple(sorted(files))
    monkeypatch.setattr(local_workspace, "_LIST_OPEN_DIRECTORY_LIMIT", 3)
    peak = _track_open_directories(monkeypatch)

    for excluded in ((), ("unused-name",)):
        workspace = LocalWorkspace(tmp_path, excluded_directory_names=excluded)
        assert _list(workspace).paths == expected
        for limit in range(1, len(expected)):
            page = _list(workspace, limit=limit)
            assert page.paths == expected[:limit]
            assert page.truncated is True

    # The root stays open, and opening the root or reopening a closed ancestor
    # briefly holds one extra descriptor.
    assert 0 < peak[0] <= 3 + 2


@pytest.mark.parametrize("excluded", [(), ("unused-name",)])
def test_list_fails_instead_of_shortening_when_descriptors_run_out(
    tmp_path, monkeypatch, excluded
) -> None:
    _write(tmp_path, ("a.txt", "b/c.txt", "z.txt"))
    original = os.open

    def exhausted_open(path, flags, *args, **kwargs):
        if os.fsdecode(path) == "b":
            raise OSError(errno.EMFILE, "Too many open files")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", exhausted_open)

    with pytest.raises(OSError) as raised:
        _list(LocalWorkspace(tmp_path, excluded_directory_names=excluded))
    assert raised.value.errno == errno.EMFILE


def test_list_applies_backslash_separated_directory_exclusions(tmp_path) -> None:
    _write(
        tmp_path,
        (
            "kept.txt",
            "prefix\\node_modules\\secret.txt",
            "dir\\Node_Modules/inner.txt",
        ),
    )
    workspace = LocalWorkspace(tmp_path, excluded_directory_names=("node_modules",))

    assert _list(workspace).paths == ("kept.txt",)


def test_list_rejects_a_root_ancestor_replaced_by_a_symlink(tmp_path) -> None:
    root = tmp_path / "parent" / "workspace"
    _write(root, ("a.txt",))
    _write(tmp_path / "other" / "workspace", ("secret.txt",))
    workspace = LocalWorkspace(root)
    (tmp_path / "parent").rename(tmp_path / "moved")
    (tmp_path / "parent").symlink_to(tmp_path / "other", target_is_directory=True)

    with pytest.raises(ValueError, match="escapes the workspace root"):
        _list(workspace)


def test_list_needs_only_search_permission_on_root_ancestors(tmp_path) -> None:
    root = tmp_path / "parent" / "workspace"
    _write(root, ("a.txt",))
    workspace = LocalWorkspace(root)
    (tmp_path / "parent").chmod(0o111)
    try:
        assert _list(workspace).paths == ("a.txt",)
    finally:
        (tmp_path / "parent").chmod(0o755)


class _FailingEntry:
    def __init__(self, entry, error: OSError) -> None:
        self._entry = entry
        self._error = error
        self.name = entry.name

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        raise self._error

    def is_file(self, *, follow_symlinks: bool = True) -> bool:
        raise self._error


@pytest.mark.parametrize("excluded", [(), ("unused-name",)])
def test_list_fails_when_an_entry_type_cannot_be_read(tmp_path, monkeypatch, excluded) -> None:
    _write(tmp_path, ("a.txt", "sub/b.txt", "z.txt"))
    original = os.scandir

    class FailingScandir:
        def __init__(self, path) -> None:
            self._iterator = original(path)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> None:
            self._iterator.close()

        def __iter__(self):
            for entry in self._iterator:
                if entry.name == "sub":
                    yield _FailingEntry(entry, OSError(errno.EIO, "Input/output error"))
                else:
                    yield entry

    monkeypatch.setattr(os, "scandir", FailingScandir)

    with pytest.raises(OSError) as raised:
        _list(LocalWorkspace(tmp_path, excluded_directory_names=excluded))
    assert raised.value.errno == errno.EIO
