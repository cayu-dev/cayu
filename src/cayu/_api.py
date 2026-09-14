"""Resolve explicitly declared public exports without eager package imports."""

from importlib import import_module
from typing import Any


def resolve_export(
    name: str,
    namespace: dict[str, Any],
    exports: dict[str, tuple[str, str]],
) -> Any:
    target = exports.get(name)
    if target is None:
        raise AttributeError(f"module {namespace['__name__']!r} has no attribute {name!r}")
    module_name, symbol = target
    value = getattr(import_module(module_name), symbol)
    namespace[name] = value
    return value
