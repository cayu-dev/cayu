"""Bounded policy JSON codec, independent of inference and financial contracts."""

from __future__ import annotations

import json
import re
from contextlib import suppress
from typing import Any

MAX_POLICY_BYTES = 65_536
MAX_EVIDENCE_BYTES = 65_536
MAX_NODES = 4096
MAX_DEPTH = 12
MAX_INTEGER = 2**53 - 1


class PolicyContractError(ValueError):
    """Fixed, payload-free diagnostic for invalid or contradictory evidence."""


def require(condition: bool) -> None:
    if not condition:
        raise PolicyContractError("Model policy contract validation failed.")


def identifier(value: object) -> None:
    require(type(value) is str and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", value) is not None)


def canonical(value: Any, *, max_bytes: int = MAX_POLICY_BYTES) -> bytes:
    """Reject unsafe types before copying/serializing; never format rejected values."""
    pending = [(value, 0)]
    nodes = 0
    # Count scalar bytes while walking, before allocating the full encoding.
    scalar_bytes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        require(depth <= MAX_DEPTH and nodes <= MAX_NODES)
        kind = type(current)
        if kind is dict:
            require(len(current) * 2 + nodes + len(pending) <= MAX_NODES)
            require(all(type(key) is str for key in current))
            pending.extend((part, depth + 1) for item in current.items() for part in item)
        elif kind is list:
            require(len(current) + nodes + len(pending) <= MAX_NODES)
            pending.extend((item, depth + 1) for item in current)
        elif kind is str:
            require(len(current) <= max_bytes)
            encoded = None
            with suppress(UnicodeError):
                encoded = current.encode("utf-8")
            require(encoded is not None)
            assert encoded is not None
            scalar_bytes += len(encoded)
            require(scalar_bytes <= max_bytes)
        elif kind is int:
            require(0 <= current <= MAX_INTEGER)
        else:
            require(current is None or kind is bool)
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    require(len(encoded) <= max_bytes)
    return encoded


def decode(wire: bytes, *, max_bytes: int = MAX_POLICY_BYTES) -> dict[str, Any]:
    require(type(wire) is bytes and 0 < len(wire) <= max_bytes)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result)
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        require(False)

    value = None
    with suppress(ValueError, UnicodeError, RecursionError):
        value = json.loads(
            wire.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
            parse_float=reject_constant,
        )
    # Raise outside the handler: schema/decoder exceptions must not retain the
    # rejected document in a diagnostic cause/context chain.
    require(type(value) is dict)
    assert isinstance(value, dict)
    canonical(value, max_bytes=max_bytes)
    return value
