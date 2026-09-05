"""Implementation-owned cleanup entry point, run under LocalRunner's deadline."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main(arguments: list[str]) -> int:
    if len(arguments) != 1:
        return 2
    root = Path(arguments[0])
    if not root.is_absolute() or not root.name.startswith("delivery-") or root.is_symlink():
        return 2
    if not root.exists():
        return 0
    if not root.is_dir():
        return 2
    shutil.rmtree(root)
    return 0 if not root.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
