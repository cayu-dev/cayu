from __future__ import annotations

import json
import runpy
from contextlib import nullcontext
from pathlib import Path

import pytest

from cayu import _validation
from cayu._dashboard_artifacts import API_TYPE_FILES, API_TYPES_ROOT
from cayu.sessions import queries

_ROOT = Path(__file__).parents[2]
_SCRIPT = _ROOT / "dashboard/scripts/generate-api-types.py"
_CASES = json.loads((_ROOT / "dashboard/tests/fixtures/label-text-boundaries.json").read_text())


def test_dashboard_label_limits_follow_domain_declarations(monkeypatch):
    namespace = runpy.run_path(str(_SCRIPT))
    assert namespace["LABEL_LIMITS"].read_bytes() == namespace["_label_limits"]()
    for name, value in {"LABEL_MAX_KEY_CHARS": 37, "LABEL_MAX_VALUE_CHARS": 193}.items():
        monkeypatch.setattr(_validation, name, value)
    for name, value in {
        "MAX_AGGREGATE_LABEL_FILTERS": 7,
        "MAX_AGGREGATE_LABEL_SELECTORS": 11,
        "MAX_AGGREGATE_LABEL_SELECTOR_VALUES": 23,
    }.items():
        monkeypatch.setattr(queries, name, value)
    rendered = namespace["_label_limits"]().decode()
    for field, value in {
        "key": 37,
        "value": 193,
        "filters": 7,
        "selectors": 11,
        "selectorValues": 23,
    }.items():
        assert f"  {field}: {value}," in rendered
    assert f'unit: "{_validation.LABEL_TEXT_LENGTH_UNIT}"' in rendered
    labels = {"😀" * 37: "😀" * 193}
    assert _validation.copy_label_map(labels, "labels") == labels
    for invalid in ({"😀" * 38: "value"}, {"key": "😀" * 194}):
        with pytest.raises(ValueError, match="at most"):
            _validation.copy_label_map(invalid, "labels")


@pytest.mark.parametrize("damage", ["value", "count", "unit", "missing", "server-version"])
def test_api_command_rejects_and_repairs_label_and_version_drift(tmp_path, monkeypatch, damage):
    main = runpy.run_path(str(_SCRIPT))["main"]
    namespace = main.__globals__
    generated = tmp_path / "generated/server-api"
    limits = generated.parent / "label-limits.ts"
    metadata = tmp_path / "release-metadata.ts"

    def generate(*, schema_path, output_dir):
        assert schema_path.read_text() == "{}\n"
        output_dir.mkdir(parents=True)
        for path in API_TYPE_FILES:
            (output_dir / Path(path).relative_to(API_TYPES_ROOT)).write_text("export {}\n")

    for key, value in {
        "GENERATED_SOURCE_ROOT": generated.parent,
        "GENERATED_DIR": generated,
        "OPENAPI_BASELINE": tmp_path / "openapi.json",
        "RELEASE_METADATA": metadata,
        "EVAL_TEXT_LIMITS": generated.parent / "eval-text-limits.ts",
        "LABEL_LIMITS": limits,
        "_openapi_json": lambda: "{}\n",
        "_run_generator": generate,
    }.items():
        monkeypatch.setitem(namespace, key, value)
    assert main([]) == 0
    assert main(["--check"]) == 0
    target = metadata if damage == "server-version" else limits
    if damage == "missing":
        target.unlink()
    else:
        original = target.read_text()
        if damage == "server-version":
            from cayu._server_contract_version import SERVER_CONTRACT_VERSION

            changed = original.replace(
                f'SUPPORTED_SERVER_CONTRACT_VERSION = "{SERVER_CONTRACT_VERSION}"',
                'SUPPORTED_SERVER_CONTRACT_VERSION = "unexpected-version"',
            )
        else:
            old, new = {
                "value": (f"key: {_validation.LABEL_MAX_KEY_CHARS},", "key: 1,"),
                "count": (f"filters: {queries.MAX_AGGREGATE_LABEL_FILTERS},", "filters: 1,"),
                "unit": (_validation.LABEL_TEXT_LENGTH_UNIT, "utf16_code_units"),
            }[damage]
            changed = original.replace(old, new)
        assert changed != original
        target.write_text(changed)
    message = (
        "release metadata is stale" if damage == "server-version" else "label limits are stale"
    )
    with pytest.raises(SystemExit, match=message):
        main(["--check"])
    assert main([]) == 0
    assert main(["--check"]) == 0


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["id"])
def test_label_models_preserve_shared_text_boundaries(case):
    text = case["text"] * case["repeat"]
    key, value = (text, "value") if case["field"] == "key" else ("key", text)
    labels = {key: value}
    factories = [
        lambda: _validation.copy_label_map(labels, "labels", allow_reserved=False),
        lambda: queries.SessionAggregateFilter(labels=labels),
    ]
    for operator in ("in", "not_in"):
        factories.append(
            lambda operator=operator: queries.SessionAggregateFilter(
                label_selectors=[{"key": key, "operator": operator, "values": [value]}]
            )
        )
    if case["field"] == "key":
        for operator in ("exists", "not_exists"):
            factories.append(
                lambda operator=operator: queries.SessionAggregateFilter(
                    label_selectors=[{"key": key, "operator": operator}]
                )
            )
    for create in factories:
        with nullcontext() if case["accepted"] else pytest.raises(ValueError):
            create()
