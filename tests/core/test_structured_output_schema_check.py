"""Structured output checks each distinct JSON Schema against the meta-schema once."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import cayu.context.structured_output as structured_output
from cayu.context.structured_output import StructuredOutputSpec


def test_a_schema_is_meta_checked_once_and_invalid_ones_always_fail(monkeypatch) -> None:
    checked: list[dict] = []
    original = structured_output.Draft202012Validator.check_schema

    def counting(schema):
        checked.append(schema)
        return original(schema)

    monkeypatch.setattr(structured_output.Draft202012Validator, "check_schema", counting)
    monkeypatch.setattr(
        structured_output, "_CHECKED_SCHEMAS", type(structured_output._CHECKED_SCHEMAS)()
    )
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    for _ in range(3):
        spec = StructuredOutputSpec(json_schema=dict(schema))
        assert spec.json_schema == schema
    # The same schema with its keys in another order is the same schema.
    StructuredOutputSpec(
        json_schema={"properties": {"answer": {"type": "string"}}, "type": "object"}
    )
    assert len(checked) == 1
    for _ in range(2):
        with pytest.raises(ValidationError, match="Invalid structured output JSON Schema"):
            StructuredOutputSpec(json_schema={"type": 5})
    assert len(checked) == 3
    # A schema that differs in one value is checked on its own.
    StructuredOutputSpec(json_schema={"type": "object", "minProperties": 1})
    StructuredOutputSpec(json_schema={"type": "object", "minProperties": 2})
    assert len(checked) == 5
