"""Compatibility wrapper for the self-contained dashboard API generator."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any, cast

_SCRIPT = Path(__file__).parents[1] / "dashboard" / "scripts" / "generate-api-types.py"
_NAMESPACE = runpy.run_path(str(_SCRIPT))
main = cast("Any", _NAMESPACE["main"])


if __name__ == "__main__":
    raise SystemExit(main())
