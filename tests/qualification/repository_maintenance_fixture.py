"""Read-only capture of an explicitly authorized qualification Git base."""

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tests.qualification.repository_maintenance_case import ALLOWED_CHANGE_PATHS, SEED_FILES


@dataclass(frozen=True)
class MaintenanceRepositoryFixture:
    base_revision: str
    # Path, SHA-256, byte length, Git mode; no repository contents in generated code.
    files: tuple[tuple[str, str, int, str], ...]

    def __post_init__(self) -> None:
        if (
            type(self.base_revision) is not str
            or re.fullmatch(r"[0-9a-f]{40}", self.base_revision) is None
        ):
            raise ValueError("Invalid qualification Git base.")
        if type(self.files) is not tuple or not 1 <= len(self.files) <= 1024:
            raise ValueError("Invalid qualification manifest.")
        paths = []
        total = 0
        for entry in self.files:
            if type(entry) is not tuple or len(entry) != 4:
                raise ValueError("Invalid qualification manifest entry.")
            path, digest, size, mode = entry
            if (
                type(path) is not str
                or path in {"", "."}
                or len(path.encode()) > 4096
                or str(PurePosixPath(path)) != path
                or PurePosixPath(path).is_absolute()
                or any(part in {"..", ".git"} for part in PurePosixPath(path).parts)
                or "\\" in path
                or any(ord(c) < 32 or ord(c) == 127 for c in path)
                or type(digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or type(size) is not int
                or not 0 <= size <= 8 * 1024 * 1024
                or type(mode) is not str
                or mode not in {"100644", "100755"}
            ):
                raise ValueError("Invalid qualification manifest entry.")
            total += size
            paths.append(path)
        if (
            paths != sorted(set(paths))
            or total > 64 * 1024 * 1024
            or sum(len(path.encode()) for path in paths) > 64 * 1024
        ):
            raise ValueError("Invalid qualification manifest bounds.")
        entries = {path: (digest, size, mode) for path, digest, size, mode in self.files}
        for path in ALLOWED_CHANGE_PATHS:
            content = SEED_FILES[path].encode()
            if entries.get(path) != (hashlib.sha256(content).hexdigest(), len(content), "100644"):
                raise ValueError("Qualification behavioral seed differs from the fixed case.")


def capture_repository_fixture(root: Path, *, expected_base: str) -> MaintenanceRepositoryFixture:
    """Require a clean complete checkout; never fetch, reset, stage or mutate Git."""
    if type(expected_base) is not str or re.fullmatch(r"[0-9a-f]{40}", expected_base) is None:
        raise ValueError("An exact Git base is required.")
    root = root.resolve(strict=True)
    environment = {
        "PATH": os.defpath,
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
    }

    def git(*args: str) -> bytes:
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=" + os.devnull, "-C", str(root), *args],
            env=environment,
            capture_output=True,
            timeout=30,
        )
        if result.returncode:
            raise ValueError("Qualification Git base inspection failed.")
        return result.stdout

    def check_clean() -> None:
        if git("rev-parse", "HEAD").decode().strip() != expected_base or git(
            "status", "--porcelain=v1", "--untracked-files=all", "--ignored"
        ):
            raise ValueError("Qualification checkout must match its clean authorized base.")

    check_clean()
    records = git("ls-tree", "-rz", "--full-tree", expected_base).split(b"\0")[:-1]
    if not 1 <= len(records) <= 1024:
        raise ValueError("Qualification manifest exceeds its path bound.")
    entries = []
    total = 0
    for record in records:
        header, raw_path = record.split(b"\t", 1)
        mode, kind, object_id = header.decode("ascii").split()
        path = raw_path.decode("utf-8")
        relative = PurePosixPath(path)
        if (
            mode not in {"100644", "100755"}
            or kind != "blob"
            or relative.is_absolute()
            or any(p in {"..", ".git"} for p in relative.parts)
            or str(relative) != path
        ):
            raise ValueError("Unsupported qualification path or Git mode.")
        size = int(git("cat-file", "-s", object_id))
        total += size
        if size > 8 * 1024 * 1024 or total > 64 * 1024 * 1024:
            raise ValueError("Qualification manifest exceeds its byte bound.")
        content = git("cat-file", "blob", object_id)
        target = root / path
        if target.resolve(strict=True) != target or not target.is_file():
            raise ValueError("Qualification checkout contains an indirect path.")
        with target.open("rb") as stream:
            current = stream.read(size + 1)
        current_mode = "100755" if target.stat().st_mode & 0o111 else "100644"
        if current != content or current_mode != mode:
            raise ValueError("Qualification checkout conflicts with its Git tree.")
        if path in ALLOWED_CHANGE_PATHS and content != SEED_FILES[path].encode():
            raise ValueError("Qualification behavioral seed differs from the fixed case.")
        entries.append((path, hashlib.sha256(content).hexdigest(), size, mode))
    if not set(ALLOWED_CHANGE_PATHS) <= {entry[0] for entry in entries}:
        raise ValueError("Qualification behavioral seed is missing.")
    check_clean()
    return MaintenanceRepositoryFixture(expected_base, tuple(sorted(entries)))
