from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_core_import_does_not_initialize_optional_campaigns():
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/smoke_built_wheel_basic_imports.py")],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "supported lazy public imports passed" in completed.stdout
