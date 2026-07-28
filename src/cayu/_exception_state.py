"""Runtime-owned state storage for exceptions crossing extension boundaries."""

from __future__ import annotations

from typing import Any

_BASE_EXCEPTION_DICT_DESCRIPTOR = BaseException.__dict__["__dict__"]


def exception_state(
    error: BaseException,
    attribute_name: str,
) -> object | None:
    """Read instance state without invoking exception-subclass descriptors."""

    namespace = _exception_namespace(error)
    if namespace is None:
        return None
    try:
        return dict.get(namespace, attribute_name)
    except BaseException:
        return None


def exception_state_contains(
    error: BaseException,
    attribute_name: str,
) -> bool:
    """Check instance state without invoking exception-subclass descriptors."""

    namespace = _exception_namespace(error)
    if namespace is None:
        return False
    try:
        return dict.__contains__(namespace, attribute_name)
    except BaseException:
        return False


def set_exception_state(
    error: BaseException,
    attribute_name: str,
    value: Any,
) -> bool:
    """Write instance state without invoking exception-subclass descriptors."""

    namespace = _exception_namespace(error)
    if namespace is None:
        return False
    try:
        dict.__setitem__(namespace, attribute_name, value)
    except BaseException:
        return False
    return True


def pop_exception_state(
    error: BaseException,
    attribute_name: str,
) -> object | None:
    """Remove instance state without invoking exception-subclass descriptors."""

    namespace = _exception_namespace(error)
    if namespace is None:
        return None
    try:
        return dict.pop(namespace, attribute_name, None)
    except BaseException:
        return None


def _exception_namespace(error: BaseException) -> dict[str, Any] | None:
    if not isinstance(error, BaseException):
        return None
    try:
        namespace = _BASE_EXCEPTION_DICT_DESCRIPTOR.__get__(error, BaseException)
    except BaseException:
        return None
    return namespace if type(namespace) is dict else None


__all__ = [
    "exception_state",
    "exception_state_contains",
    "pop_exception_state",
    "set_exception_state",
]
