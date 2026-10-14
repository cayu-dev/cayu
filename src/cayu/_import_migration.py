"""Replacement paths for removed imports, without loading their implementations."""

import json
from functools import cache
from importlib.resources import files


@cache
def _replacements() -> dict[str, dict[str, str]]:
    grouped = json.loads(files("cayu.data").joinpath("public_api_imports.json").read_bytes())
    return {
        package: {name: f"{module}.{name}" for module, names in owners.items() for name in names}
        for package, owners in grouped.items()
    }


def import_replacement(package: str, name: str) -> str | None:
    """Look up a removed name; ordinary package imports do not read the map."""
    if package not in {"cayu", "cayu.runtime"} or name.startswith("_"):
        return None
    return _replacements()[package].get(name)


def replacement_import(target: str) -> str:
    module, name = target.rsplit(".", 1)
    return f"from {module} import {name}"
