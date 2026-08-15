"""Dependency-neutral durable authority values shared with storage adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from cayu._validation import canonical_durable_json_bytes

_SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


@dataclass(frozen=True)
class CheckpointValueAuthority:
    """Content authority for one exact durable checkpoint value."""

    sha256: str

    def __post_init__(self) -> None:
        if type(self.sha256) is not str or _SHA256_HEX_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("Checkpoint value authority must be a lowercase SHA-256 digest.")


def checkpoint_value_authority(value: Any, field_name: str) -> CheckpointValueAuthority:
    """Authenticate one portable checkpoint value without exposing its contents."""

    encoded = canonical_durable_json_bytes(value, field_name)
    return CheckpointValueAuthority(sha256=sha256(encoded).hexdigest())
