"""Bounded JSON object files shared by CLI inputs and startup configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cayu._validation import copy_durable_json_object

MAX_INPUT_BYTES = 1024 * 1024


def json_object_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError("Input exceeds 1 MiB.")
        value = json.loads(raw)
        return copy_durable_json_object(value, "CLI JSON input")
    except (OSError, ValueError, TypeError):
        raise ValueError("Input must be a readable JSON object file of at most 1 MiB.") from None
