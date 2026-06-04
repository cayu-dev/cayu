from __future__ import annotations

from math import isfinite
from typing import Any


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


def require_finite(value: float, field_name: str) -> float:
    if not isfinite(value):
        raise ValueError(f"`{field_name}` must be finite.")
    return value


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
