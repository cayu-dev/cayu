"""Fixed check program whose `--` delimiter semantics are part of this fixture."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main() -> int:
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "--help":
        print("fixture help")
        return 0
    if arguments and arguments[0].startswith("--junitxml="):
        Path(arguments[0].split("=", 1)[1]).write_text("report\n", encoding="utf-8")
        return 0
    if arguments and arguments[0].startswith("--create="):
        Path(arguments[0].split("=", 1)[1]).write_text("created\n", encoding="utf-8")
        return 0
    if arguments and arguments[0].startswith("--overwrite="):
        Path(arguments[0].split("=", 1)[1]).write_text("overwritten\n", encoding="utf-8")
        return 0
    if arguments and arguments[0].startswith("--remove="):
        Path(arguments[0].split("=", 1)[1]).unlink()
        return 0
    if not arguments or arguments[0] != "--":
        return 4

    selectors = arguments[1:]
    selected_paths = [selector.split("::", 1)[0] for selector in selectors]
    if "tests/slow.py" in selected_paths:
        time.sleep(2)
    if "tests/writes.py" in selected_paths:
        (Path.cwd().parent / "outside-output.txt").write_text(
            "unexpected write\n", encoding="utf-8"
        )
    if "tests/mkdir.py" in selected_paths:
        (Path.cwd() / "internal-empty-dir").mkdir()
    if "tests/rmdir.py" in selected_paths:
        (Path.cwd() / "internal-empty-dir").rmdir()
    if "tests/symlink.py" in selected_paths:
        (Path.cwd() / "internal-link").symlink_to("tests/pass.py")
    if "tests/retarget_symlink.py" in selected_paths:
        link = Path.cwd() / "internal-link"
        link.unlink()
        link.symlink_to("tests/fail.py")

    tests_executed = 0 if "tests/zero.py" in selected_paths else len(selected_paths)
    print(json.dumps({"tests_executed": tests_executed}))
    return 1 if "tests/fail.py" in selected_paths else 0


if __name__ == "__main__":
    raise SystemExit(main())
