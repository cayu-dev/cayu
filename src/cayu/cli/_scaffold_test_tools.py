"""Where a generated project's `pyproject.toml` lists its test tools."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import cast

_REQUIREMENT_NAME = re.compile(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")


def extra_only_dev_requirements(document: Mapping[str, object]) -> tuple[str, ...]:
    """The `[project.optional-dependencies].dev` requirements a plain `uv sync` leaves out.

    A requirement counts as installed when `[dependency-groups].dev`, or a group it
    includes with `{include-group = ...}`, names the same package.
    """

    extra = _entry(document, "project", "optional-dependencies", "dev")
    if not isinstance(extra, list):
        return ()
    grouped = _group_package_names(document.get("dependency-groups"), "dev")
    return tuple(
        item
        for item in extra
        if isinstance(item, str) and (name := _package_name(item)) and name not in grouped
    )


def _group_package_names(groups: object, name: str) -> set[str]:
    names: set[str] = set()
    if not isinstance(groups, Mapping):
        return names
    # Group names compare normalized, like package names.
    by_name = {_normalized(str(key)): value for key, value in groups.items()}
    pending = [name]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        items = by_name.get(current)
        for item in items if isinstance(items, list) else ():
            if isinstance(item, str) and (package := _package_name(item)):
                names.add(package)
            elif isinstance(item, Mapping):
                included = cast("Mapping[str, object]", item).get("include-group")
                if isinstance(included, str):
                    pending.append(_normalized(included))
    return names


def _package_name(requirement: str) -> str | None:
    match = _REQUIREMENT_NAME.match(requirement)
    return _normalized(match.group(1)) if match else None


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _entry(table: object, *keys: str) -> object:
    for key in keys:
        if not isinstance(table, Mapping):
            return None
        table = cast("Mapping[str, object]", table).get(key)
    return table
