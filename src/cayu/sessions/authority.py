"""Dependency-neutral durable authority values shared with storage adapters."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from cayu._validation import (
    DURABLE_DOCUMENT_LIMITS,
    MAX_DURABLE_JSON_INTEGER,
    MAX_DURABLE_JSON_NESTING,
    DurableJsonLimits,
    canonical_durable_json_bytes,
    copy_durable_record,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank

_SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


class SessionRunFenced(RuntimeError):
    """A durable write was rejected because its run no longer owns the session epoch."""


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


def _require_raw_sha256_digest(value: Any) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError


def _canonical_runtime_publication_digest(
    value: Any, *, max_nesting: int = MAX_DURABLE_JSON_NESTING
) -> str:
    encoded = canonical_durable_json_bytes(value, "runtime_publication", max_nesting=max_nesting)
    return sha256(encoded).hexdigest()


CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS = 32


@dataclass(frozen=True)
class CheckpointRootFieldProjection:
    """Bounded opaque evidence for one runtime-selected root field."""

    is_present: bool
    json_type: str | None
    scalar_text: str | None
    scalar_text_truncated: bool = False


CheckpointRootFieldProjectionValidator = Callable[
    [str, CheckpointRootFieldProjection],
    None,
]


@dataclass(frozen=True)
class CheckpointRootFieldGuard:
    """One runtime-owned validator to run inside a store read snapshot."""

    key: str
    validate: CheckpointRootFieldProjectionValidator

    def __post_init__(self) -> None:
        key = require_clean_nonblank(self.key, "checkpoint root field key")
        if any(
            char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
            for char in key
        ):
            raise ValueError(
                "checkpoint root field key must contain only ASCII letters, digits, "
                "and underscores."
            )
        if not callable(self.validate):
            raise TypeError("checkpoint root field validator must be callable.")


def checkpoint_root_field_projection(
    checkpoint: dict[str, Any] | None,
    key: str,
) -> CheckpointRootFieldProjection:
    """Project one root field without copying arbitrary checkpoint contents."""

    if checkpoint is None or key not in checkpoint:
        return CheckpointRootFieldProjection(
            is_present=False,
            json_type=None,
            scalar_text=None,
        )
    value = checkpoint[key]
    if value is None:
        json_type = "null"
        scalar_text = None
    elif type(value) is bool:
        json_type = "boolean"
        scalar_text = None
    elif type(value) is int:
        json_type = "integer"
        scalar_text = str(value)
    elif type(value) is float:
        json_type = "number"
        scalar_text = None
    elif type(value) is str:
        json_type = "string"
        scalar_text = None
    elif type(value) is list:
        json_type = "array"
        scalar_text = None
    elif type(value) is dict:
        json_type = "object"
        scalar_text = None
    else:
        json_type = "other"
        scalar_text = None
    truncated = (
        scalar_text is not None and len(scalar_text) > CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS
    )
    return CheckpointRootFieldProjection(
        is_present=True,
        json_type=json_type,
        scalar_text=(
            None if scalar_text is None else scalar_text[:CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS]
        ),
        scalar_text_truncated=truncated,
    )


def checkpoint_root_field_projection_from_storage(
    *,
    json_type: str | None,
    scalar_text: Any,
) -> CheckpointRootFieldProjection:
    """Normalize bounded root-field evidence projected by a durable store."""

    json_type_aliases = {
        "false": "boolean",
        "real": "number",
        "text": "string",
        "true": "boolean",
    }
    normalized_json_type = (
        None if json_type is None else json_type_aliases.get(json_type, json_type)
    )
    normalized_scalar = None if scalar_text is None else str(scalar_text)
    return CheckpointRootFieldProjection(
        is_present=json_type is not None,
        json_type=normalized_json_type,
        scalar_text=(
            None
            if normalized_scalar is None
            else normalized_scalar[:CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS]
        ),
        scalar_text_truncated=(
            normalized_scalar is not None
            and len(normalized_scalar) > CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS
        ),
    )


def _runtime_publication_json_equal(
    left: Any, right: Any, *, max_nesting: int = MAX_DURABLE_JSON_NESTING
) -> bool:
    """Compare validated JSON material under the durable canonicalization contract."""

    limits = DurableJsonLimits(
        DURABLE_DOCUMENT_LIMITS.max_bytes, DURABLE_DOCUMENT_LIMITS.max_nodes, max_nesting
    )
    return _validated_runtime_publication_json_equal(
        copy_durable_record(left, "runtime_publication", limits=limits),
        copy_durable_record(right, "runtime_publication", limits=limits),
    )


def _validated_runtime_publication_json_equal(left: Any, right: Any) -> bool:
    """Compare detached durable JSON without encoding and hashing it again.

    Inputs must have passed durable copying, including integral-float
    normalization. Python container equality alone conflates booleans/numbers.
    """
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _validated_runtime_publication_json_equal(value, right[key])
            for key, value in left.items()
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _validated_runtime_publication_json_equal(a, b)
            for a, b in zip(left, right, strict=True)
        )
    return left == right


def _validate_required_runtime_fence(value: int, field_name: str) -> int:
    validated = _validate_optional_runtime_publication_fence(value, field_name)
    if validated is None:
        raise TypeError(f"{field_name} must be an integer.")
    return validated


def _validate_optional_runtime_publication_fence(
    value: int | None,
    field_name: str,
) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer or None.")
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.")
    if value > MAX_DURABLE_JSON_INTEGER:
        raise ValueError(f"{field_name} must be at most {MAX_DURABLE_JSON_INTEGER}.")
    return value
