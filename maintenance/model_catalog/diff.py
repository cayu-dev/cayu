"""Deterministic, reviewer-friendly diffs for ``ModelCatalog`` snapshots."""

from __future__ import annotations

import json
import re
from typing import Any

from cayu import ModelCatalog, ModelInfo


def _key(model: ModelInfo) -> tuple[str, str]:
    return model.provider_name, model.model


def markdown_code_span(value: str) -> str:
    """Render arbitrary text in a CommonMark code span without delimiter breakout."""

    if not value:
        return "<code></code>"
    longest_run = max((len(match.group()) for match in re.finditer(r"`+", value)), default=0)
    delimiter = "`" * (longest_run + 1)
    if longest_run == 0:
        return f"{delimiter}{value}{delimiter}"
    return f"{delimiter} {value} {delimiter}"


def _flatten(value: Any, *, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten(value[key], prefix=path))
        return flattened
    return {prefix: value}


def catalog_diff(before: ModelCatalog, after: ModelCatalog) -> dict[str, Any]:
    """Return added, removed, and field-level changed records."""

    old = {_key(model): model for model in before.models}
    new = {_key(model): model for model in after.models}
    added = [f"{provider}/{model}" for provider, model in sorted(new.keys() - old.keys())]
    removed = [f"{provider}/{model}" for provider, model in sorted(old.keys() - new.keys())]
    changed: list[dict[str, Any]] = []

    for key in sorted(old.keys() & new.keys()):
        old_fields = _flatten(old[key].model_dump(mode="json"))
        new_fields = _flatten(new[key].model_dump(mode="json"))
        fields = {
            field: {"before": old_fields.get(field), "after": new_fields.get(field)}
            for field in sorted(old_fields.keys() | new_fields.keys())
            if old_fields.get(field) != new_fields.get(field)
        }
        if fields:
            changed.append({"model": f"{key[0]}/{key[1]}", "fields": fields})
    return {"added": added, "removed": removed, "changed": changed}


def format_catalog_diff(before: ModelCatalog, after: ModelCatalog) -> str:
    """Render a compact Markdown report suitable for an automated PR body."""

    diff = catalog_diff(before, after)
    lines = ["## Model catalog refresh", ""]
    lines.append(
        f"{len(diff['added'])} added, {len(diff['removed'])} removed, "
        f"{len(diff['changed'])} changed."
    )

    for heading, key in (("Added", "added"), ("Removed", "removed")):
        if diff[key]:
            lines.extend(["", f"### {heading}", ""])
            lines.extend(f"- {markdown_code_span(identity)}" for identity in diff[key])

    if diff["changed"]:
        lines.extend(["", "### Changed", ""])
        for record in diff["changed"]:
            lines.append(f"- {markdown_code_span(record['model'])}")
            for field, values in record["fields"].items():
                before_value = json.dumps(values["before"], sort_keys=True)
                after_value = json.dumps(values["after"], sort_keys=True)
                lines.append(
                    f"  - {markdown_code_span(field)}: {markdown_code_span(before_value)} "
                    f"→ {markdown_code_span(after_value)}"
                )

    lines.extend(
        [
            "",
            "Review the official provenance URL and evidence for every changed price or capability.",
            "This workflow never changes a published release directly.",
            "",
        ]
    )
    return "\n".join(lines)
