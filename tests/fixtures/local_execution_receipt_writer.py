"""Crash fixture that pauses after a local-attempt receipt staging fsync."""

from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path

# Production launches this stdlib-only supervisor by file path. Match that
# boundary without importing the complete SDK just to write a staging receipt.
supervisor = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "src/cayu/runtime/_local_execution_supervisor.py")
)


def main() -> None:
    payload_path = Path(sys.argv[1])
    receipt_path = Path(sys.argv[2])
    ready_path = Path(sys.argv[3])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise TypeError("receipt payload must be an object")

    def pause_before_rename(_source: os.PathLike[str], _target: os.PathLike[str]) -> None:
        ready_path.write_text("staged", encoding="ascii")
        while True:
            time.sleep(1)

    supervisor["os"].replace = pause_before_rename
    supervisor["_atomic_receipt"](receipt_path, payload)


if __name__ == "__main__":
    main()
