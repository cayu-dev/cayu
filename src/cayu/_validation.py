from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from math import isfinite
from types import MappingProxyType
from typing import Any, Never

from pydantic import BaseModel

_MAX_LABEL_KEY_LENGTH = 128
_MAX_LABEL_VALUE_LENGTH = 512
_RESERVED_LABEL_PREFIX = "cayu:"


class FrozenJsonDict(Mapping[str, Any]):
    """An immutable JSON object with mapping-compatible reads."""

    _data: Mapping[str, Any]
    __slots__ = ("_data",)

    def __init__(self, values: Mapping[str, Any] | Iterable[tuple[str, Any]] = ()) -> None:
        object.__setattr__(self, "_data", MappingProxyType(dict(values)))

    def __setattr__(self, name: str, value: Any) -> Never:
        _raise_frozen_json_mutation()

    def __delattr__(self, name: str) -> Never:
        _raise_frozen_json_mutation()

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(dict(self._data))

    def __delitem__(self, key: str, /) -> None:
        _raise_frozen_json_mutation()

    def __ior__(self, value: Any, /) -> Never:
        _raise_frozen_json_mutation()

    def __setitem__(self, key: str, value: Any, /) -> None:
        _raise_frozen_json_mutation()

    def clear(self) -> None:
        _raise_frozen_json_mutation()

    def pop(self, key: str, default: Any = None, /) -> Any:
        _raise_frozen_json_mutation()

    def popitem(self) -> tuple[str, Any]:
        _raise_frozen_json_mutation()

    def setdefault(self, key: str, default: Any = None, /) -> Any:
        _raise_frozen_json_mutation()

    def update(self, *args: Any, **kwargs: Any) -> None:
        _raise_frozen_json_mutation()

    def __copy__(self) -> FrozenJsonDict:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenJsonDict:
        return self

    def __reduce__(self):
        return type(self), (tuple(self._data.items()),)


class FrozenJsonList(Sequence[Any]):
    """An immutable JSON array with sequence-compatible reads."""

    _items: tuple[Any, ...]
    __slots__ = ("_items",)

    def __init__(self, values: Iterable[Any] = ()) -> None:
        object.__setattr__(self, "_items", tuple(values))

    def __setattr__(self, name: str, value: Any) -> Never:
        _raise_frozen_json_mutation()

    def __delattr__(self, name: str) -> Never:
        _raise_frozen_json_mutation()

    def __getitem__(self, index: int | slice) -> Any:
        value = self._items[index]
        if isinstance(index, slice):
            return type(self)(value)
        return value

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __repr__(self) -> str:
        return repr(list(self._items))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence) or isinstance(other, str | bytes | bytearray):
            return False
        return tuple(self._items) == tuple(other)

    def __delitem__(self, key: Any, /) -> None:
        _raise_frozen_json_mutation()

    def __iadd__(self, value: Any, /) -> Never:
        _raise_frozen_json_mutation()

    def __imul__(self, value: Any, /) -> Never:
        _raise_frozen_json_mutation()

    def __setitem__(self, key: Any, value: Any, /) -> None:
        _raise_frozen_json_mutation()

    def append(self, value: Any, /) -> None:
        _raise_frozen_json_mutation()

    def clear(self) -> None:
        _raise_frozen_json_mutation()

    def extend(self, values: Any, /) -> None:
        _raise_frozen_json_mutation()

    def insert(self, index: Any, value: Any, /) -> None:
        _raise_frozen_json_mutation()

    def pop(self, index: Any = -1, /) -> Any:
        _raise_frozen_json_mutation()

    def remove(self, value: Any, /) -> None:
        _raise_frozen_json_mutation()

    def reverse(self) -> None:
        _raise_frozen_json_mutation()

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        _raise_frozen_json_mutation()

    def __copy__(self) -> FrozenJsonList:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenJsonList:
        return self

    def __reduce__(self):
        return type(self), (self._items,)


def _raise_frozen_json_mutation() -> Never:
    raise TypeError("Frozen JSON values cannot be mutated.")


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


def require_unicode_scalar_text(value: str, field_name: str) -> str:
    """Reject lone UTF-16 surrogate code points from a validated string."""

    if type(value) is not str:
        raise ValueError(f"`{field_name}` must be a string.")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError(f"`{field_name}` must not contain Unicode surrogate code points.")
    return value


def require_unicode_scalar_json(value: Any, field_name: str) -> Any:
    """Reject lone UTF-16 surrogates recursively from a JSON-compatible value."""

    if type(value) is str:
        return require_unicode_scalar_text(value, field_name)
    if type(value) is list:
        for index, item in enumerate(value):
            require_unicode_scalar_json(item, f"{field_name}[{index}]")
        return value
    if type(value) is dict:
        for key, item in value.items():
            require_unicode_scalar_text(key, f"{field_name} key")
            require_unicode_scalar_json(item, f"{field_name}.{key}")
        return value
    return value


def freeze_json_value(value: Any) -> Any:
    """Recursively freeze an already validated JSON-compatible value."""

    if type(value) is FrozenJsonDict or type(value) is FrozenJsonList:
        return value
    if type(value) is dict:
        return FrozenJsonDict({key: freeze_json_value(item) for key, item in value.items()})
    if type(value) is list:
        return FrozenJsonList(freeze_json_value(item) for item in value)
    return value


def thaw_json_value(value: Any) -> Any:
    """Return ordinary JSON containers from recursively frozen JSON data."""

    if type(value) is FrozenJsonDict:
        return {key: thaw_json_value(item) for key, item in value.items()}
    if type(value) is FrozenJsonList:
        return [thaw_json_value(item) for item in value]
    if type(value) is dict:
        return {key: thaw_json_value(item) for key, item in value.items()}
    if type(value) is list:
        return [thaw_json_value(item) for item in value]
    return value


class JsonUtf8SizeCounter:
    """Count compact JSON UTF-8 bytes without building the serialized value."""

    def __init__(self, limit: int, *, ensure_ascii: bool = False) -> None:
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a non-negative integer.")
        if type(ensure_ascii) is not bool:
            raise TypeError("ensure_ascii must be a boolean.")
        self.remaining = limit
        self.ensure_ascii = ensure_ascii

    def _consume(self, count: int) -> bool:
        self.remaining -= count
        return self.remaining >= 0

    def _string(self, value: str) -> bool:
        if not self._consume(2):  # opening and closing quotes
            return False
        for character in value:
            codepoint = ord(character)
            if character in {'"', "\\"} or character in "\b\f\n\r\t":
                size = 2
            elif codepoint < 0x20 or (self.ensure_ascii and 0x7F <= codepoint < 0x10000):
                size = 6
            elif self.ensure_ascii and codepoint >= 0x10000:
                size = 12
            elif codepoint < 0x80:
                size = 1
            elif codepoint < 0x800:
                size = 2
            elif codepoint < 0x10000:
                size = 3
            else:
                size = 4
            if not self._consume(size):
                return False
        return True

    def value(self, value: Any) -> bool:
        if value is None:
            return self._consume(4)
        if value is True:
            return self._consume(4)
        if value is False:
            return self._consume(5)
        if isinstance(value, str):
            return self._string(value)
        if isinstance(value, datetime):
            # ``+00:00`` is five bytes longer than Pydantic's UTC ``Z`` form, so
            # this intentionally provides a conservative upper bound.
            return self._string(value.isoformat())
        if type(value) in {int, float}:
            return self._consume(len(str(value).encode("utf-8")))
        if isinstance(value, BaseModel):
            fields = type(value).model_fields
            if not self._consume(2):
                return False
            for index, (name, field) in enumerate(fields.items()):
                if index and not self._consume(1):
                    return False
                key = field.serialization_alias or field.alias or name
                if not self._string(key) or not self._consume(1):
                    return False
                if not self.value(getattr(value, name)):
                    return False
            return True
        if isinstance(value, Mapping):
            if not self._consume(2):
                return False
            for index, (key, item) in enumerate(value.items()):
                if index and not self._consume(1):
                    return False
                if not self._string(str(key)) or not self._consume(1):
                    return False
                if not self.value(item):
                    return False
            return True
        if isinstance(value, (list, tuple)):
            if not self._consume(2):
                return False
            for index, item in enumerate(value):
                if index and not self._consume(1):
                    return False
                if not self.value(item):
                    return False
            return True
        return False


def json_utf8_size_within_limit(
    value: Any,
    max_bytes: int,
    *,
    ensure_ascii: bool = False,
) -> bool:
    """Whether compact, unescaped JSON for ``value`` fits ``max_bytes``.

    The walk stops as soon as the limit is exceeded and never allocates a
    serialized copy. Pydantic models and datetimes are supported for internal
    response-size guards; ordinary JSON containers cover transport payloads.
    """
    return JsonUtf8SizeCounter(max_bytes, ensure_ascii=ensure_ascii).value(value)


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


def collision_safe_json_object(
    items: Iterable[tuple[str, Any]],
    *,
    preserve_input_order: bool,
) -> dict[str, Any]:
    """Build a deterministic JSON object without dropping transformed key collisions."""

    indexed_items = [
        (
            index,
            key,
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            value,
        )
        for index, (key, value) in enumerate(items)
    ]
    assigned: list[tuple[int, str, Any]] = []
    counts: dict[str, int] = {}
    used_keys: set[str] = set()
    for index, key, _, value in sorted(indexed_items, key=lambda item: (item[1], item[2])):
        count = counts.get(key, 0) + 1
        candidate = key if count == 1 else f"{key}_{count}"
        while candidate in used_keys:
            count += 1
            candidate = f"{key}_{count}"
        counts[key] = count
        used_keys.add(candidate)
        assigned.append((index, candidate, value))

    if preserve_input_order:
        assigned.sort(key=lambda item: item[0])
    return {key: value for _, key, value in assigned}


def _copy_json_value(value: Any, field_name: str, seen: set[int]) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if isfinite(value):
            return value
        raise ValueError(f"`{field_name}` must contain finite JSON numbers.")
    if type(value) in {list, FrozenJsonList}:
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
    if type(value) in {dict, FrozenJsonDict}:
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
