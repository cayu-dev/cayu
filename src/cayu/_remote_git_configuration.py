"""Reject repository-controlled Git behavior before invoking the Git executable."""

from __future__ import annotations

import configparser
import os
import stat
from hashlib import sha256
from pathlib import Path


def executable_digest(path: Path) -> str:
    """Hash an admitted executable with bounded reads and a finite total cap."""
    digest = sha256()
    total = 0
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            total += len(chunk)
            if total > 128 << 20:
                raise ValueError("Remote Git executable exceeds its identity byte bound.")
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def require_inert_repository_config(root: Path) -> None:
    """Admit only Git init's bookkeeping, not local execution/transport policy."""
    repository = root / ".git"
    if not repository.exists() and not repository.is_symlink():
        return
    if repository.is_symlink() or not repository.is_dir():
        raise ValueError("Remote Git repository identity is unsafe.")
    descriptor = os.open(repository / "config", os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("Remote Git repository configuration is unsafe.")
        content = os.read(descriptor, 8193)
    finally:
        os.close(descriptor)
    if len(content) > 8192:
        raise ValueError("Remote Git repository configuration exceeds its byte bound.")
    parser = configparser.RawConfigParser(strict=True)
    try:
        parser.read_string(content.decode("utf-8"))
    except (ValueError, configparser.Error) as error:
        raise ValueError("Remote Git repository configuration is malformed.") from error
    allowed = {
        "core": {
            "repositoryformatversion": {"0", "1"},
            "filemode": {"true", "false"},
            "bare": {"false"},
            "logallrefupdates": {"true"},
            "ignorecase": {"true", "false"},
            "precomposeunicode": {"true", "false"},
        },
        "extensions": {"objectformat": {"sha256"}},
    }
    if parser.defaults() or "core" not in parser:
        raise ValueError("Remote Git repository configuration is not implementation-owned.")
    for section in parser.sections():
        for key, value in parser.items(section):
            if value not in allowed.get(section, {}).get(key, set()):
                raise ValueError(
                    "Remote Git repository configuration contains unapproved behavior."
                )
