"""Resolve explicitly declared public exports without eager package imports."""

from collections.abc import Iterable
from importlib import import_module
from typing import Any

# Modules that import an optional extra (Postgres or server packages) at import time.
# Their exports stay importable by name, which reports the extra to install, but are
# left out of ``from package import *`` so a wildcard import works on a plain install.
OPTIONAL_EXTRA_MODULES = frozenset(
    {
        "cayu.storage.work_context_postgres",
        "cayu.storage.tasks_postgres",
        "cayu.storage.budget_postgres",
        "cayu.storage.collaboration_postgres",
        "cayu.storage.event_watchers_postgres",
        "cayu.storage.evals_postgres",
        "cayu.storage.postgres",
        "cayu.storage.product_operations_postgres",
        "cayu.storage.product_operations_sqlite",
    }
)


def wildcard_names(
    public_names: Iterable[str],
    exports: dict[str, tuple[str, str]],
) -> list[str]:
    """Return the public names that ``from package import *`` can load without extras."""

    return [name for name in public_names if exports[name][0] not in OPTIONAL_EXTRA_MODULES]


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
