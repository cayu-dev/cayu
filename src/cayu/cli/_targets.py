from __future__ import annotations

import importlib
from typing import Any


class TargetResolutionError(ValueError):
    """A target string cannot resolve to the requested Python object."""


def load_target(
    target: str,
    *,
    label: str = "Target",
    normalize_errors: bool = False,
) -> Any:
    """Load a ``module:attribute`` target without invoking it."""
    if ":" not in target:
        raise TargetResolutionError(f"{label} must use module:attribute syntax.")
    module_name, attr_path = target.split(":", 1)
    if not module_name or not attr_path:
        raise TargetResolutionError(f"{label} must use module:attribute syntax.")

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Normalize a missing target, but preserve failures from the target's own imports.
        missing_requested_module = exc.name == module_name or module_name.startswith(f"{exc.name}.")
        if not normalize_errors or not missing_requested_module:
            raise
        raise TargetResolutionError(f"{label} module was not found: {module_name}.") from exc
    value: Any = module
    for part in attr_path.split("."):
        if not part:
            raise TargetResolutionError(f"{label} attribute path contains an empty segment.")
        try:
            value = getattr(value, part)
        except AttributeError as exc:
            if not normalize_errors:
                raise
            raise TargetResolutionError(
                f"{label} attribute was not found: {module_name}:{attr_path}."
            ) from exc
    return value
