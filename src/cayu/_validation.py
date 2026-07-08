from __future__ import annotations

from math import isfinite
from typing import Any

_MAX_LABEL_KEY_LENGTH = 128
_MAX_LABEL_VALUE_LENGTH = 512
_RESERVED_LABEL_PREFIX = "cayu:"


def require_nonblank(value: str, field_name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"`{field_name}` must be a string.")
    if not value.strip():
        raise ValueError(f"`{field_name}` cannot be blank.")
    return value


def require_clean_nonblank(value: str, field_name: str) -> str:
    value = require_nonblank(value, field_name)
    if value != value.strip():
        raise ValueError(f"`{field_name}` must not start or end with whitespace.")
    return value


def require_nonblank_keys(value: dict[str, Any], field_name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"`{field_name}` must be a dictionary.")
    for key in value:
        if type(key) is not str:
            raise ValueError(f"`{field_name}` keys must be strings.")
        require_nonblank(key, f"{field_name} key")
    return value


def require_clean_nonblank_keys(value: dict[str, Any], field_name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"`{field_name}` must be a dictionary.")
    for key in value:
        if type(key) is not str:
            raise ValueError(f"`{field_name}` keys must be strings.")
        require_clean_nonblank(key, f"{field_name} key")
    return value


def copy_label_map(
    value: Any,
    field_name: str,
    *,
    allow_reserved: bool = True,
) -> dict[str, str]:
    if value is None:
        return {}
    if type(value) is not dict:
        raise ValueError(f"`{field_name}` must be a dictionary.")
    copied: dict[str, str] = {}
    for key, item in value.items():
        if type(key) is not str:
            raise ValueError(f"`{field_name}` keys must be strings.")
        clean_key = require_clean_nonblank(key, f"{field_name} key")
        if len(clean_key) > _MAX_LABEL_KEY_LENGTH:
            raise ValueError(
                f"`{field_name}` keys must be at most {_MAX_LABEL_KEY_LENGTH} characters."
            )
        if not allow_reserved and clean_key.startswith(_RESERVED_LABEL_PREFIX):
            raise ValueError(
                f"`{field_name}` keys starting with `{_RESERVED_LABEL_PREFIX}` "
                "are reserved for Cayu."
            )
        if type(item) is not str:
            raise ValueError(f"`{field_name}.{clean_key}` must be a string.")
        clean_value = require_clean_nonblank(item, f"{field_name}.{clean_key}")
        if len(clean_value) > _MAX_LABEL_VALUE_LENGTH:
            raise ValueError(
                f"`{field_name}` values must be at most {_MAX_LABEL_VALUE_LENGTH} characters."
            )
        copied[clean_key] = clean_value
    return copied


def require_finite(value: float, field_name: str) -> float:
    if not isfinite(value):
        raise ValueError(f"`{field_name}` must be finite.")
    return value


def escape_json_pointer_segment(key: str) -> str:
    """Escape one JSON-pointer reference token per RFC 6901 (`~`→`~0`, `/`→`~1`).

    Used to build the `$`-rooted, `/`-separated error paths shared by the
    runtime's structured-output validation and provider schema preflights.
    """
    return key.replace("~", "~0").replace("/", "~1")


def unescape_json_pointer_segment(segment: str) -> str:
    """Decode one JSON-pointer reference token per RFC 6901 (`~1`→`/`, `~0`→`~`).

    The replace order is load-bearing: `~1` must be decoded before `~0` so
    that `~01` becomes `~1`, not `/` (RFC 6901 section 4).
    """
    return segment.replace("~1", "/").replace("~0", "~")


def copy_json_value(value: Any, field_name: str) -> Any:
    return _copy_json_value(value, field_name, set())


def copy_json_object(value: Any, field_name: str) -> dict[str, Any]:
    copied = copy_json_value(value, field_name)
    if type(copied) is not dict:
        raise ValueError(f"`{field_name}` must be a JSON object.")
    return copied


def validate_json_value(value: Any, field_name: str) -> None:
    _copy_json_value(value, field_name, set())


def _copy_json_value(value: Any, field_name: str, seen: set[int]) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if isfinite(value):
            return value
        raise ValueError(f"`{field_name}` must contain finite JSON numbers.")
    if type(value) is list:
        value_id = id(value)
        if value_id in seen:
            raise ValueError(f"`{field_name}` must not contain circular references.")
        seen.add(value_id)
        try:
            return [
                _copy_json_value(item, f"{field_name}[{index}]", seen)
                for index, item in enumerate(value)
            ]
        finally:
            seen.remove(value_id)
    if type(value) is dict:
        value_id = id(value)
        if value_id in seen:
            raise ValueError(f"`{field_name}` must not contain circular references.")
        seen.add(value_id)
        try:
            copied: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError(f"`{field_name}` JSON object keys must be strings.")
                copied[key] = _copy_json_value(item, f"{field_name}.{key}", seen)
            return copied
        finally:
            seen.remove(value_id)
    raise ValueError(f"`{field_name}` must contain JSON-compatible values.")
