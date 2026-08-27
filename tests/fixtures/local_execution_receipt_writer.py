"""Crash fixture that pauses after a local-attempt receipt staging fsync."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from cayu.runtime import _local_execution_supervisor as supervisor


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

    supervisor.os.replace = pause_before_rename  # type: ignore[assignment]
    supervisor._atomic_receipt(receipt_path, payload)


if __name__ == "__main__":
    main()
