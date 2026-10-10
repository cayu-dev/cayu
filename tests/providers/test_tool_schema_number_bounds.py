"""Tool-schema number bounds serialize the same way on every request.

OpenAI renders an integer bound on a ``number`` field as ``100`` on some
upstream replicas and ``100.0`` on others. Durable copies collapse ``100.0`` to
``100``, so the OpenAI adapters write ``number`` bounds as floats.
"""

from __future__ import annotations

import json

from cayu import Message
from cayu.providers import ModelRequest
from cayu.providers._tool_schemas import float_number_bounds
from cayu.providers.chat_completions import build_chat_completions_payload
from cayu.providers.openai import build_openai_payload
from cayu.tools.base import ToolSpec

_SCHEMA = {
    "type": "object",
    "properties": {
        "confidence": {"type": "number", "minimum": 0, "maximum": 100.0},
        "ratio": {"type": ["number", "null"], "exclusiveMinimum": 0, "multipleOf": 0.5},
        "count": {"type": "integer", "minimum": 0, "maximum": 100},
        "either": {"type": ["integer", "number"], "minimum": 1},
        # Property names that happen to be schema keywords are not bounds.
        "minimum": {"type": "string", "default": "0"},
        "items": {
            "type": "array",
            "items": {"type": "number", "minimum": -5, "default": 3},
        },
    },
    "required": ["confidence"],
}


def _request() -> ModelRequest:
    spec = ToolSpec(name="score", description="Record a score.", input_schema=_SCHEMA)
    return ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "Score it.")],
        tools=[
            {"name": spec.name, "description": spec.description, "input_schema": spec.input_schema}
        ],
    )


def test_durable_request_copies_store_integral_number_bounds_as_integers() -> None:
    # This is the normalization the adapters undo at the provider boundary.
    bounds = _request().tools[0]["input_schema"]["properties"]["confidence"]
    assert type(bounds["maximum"]) is int


def test_openai_number_bounds_are_floats_and_byte_stable() -> None:
    payloads = [build_openai_payload(_request()) for _ in range(3)]
    serialized = {json.dumps(payload["tools"]) for payload in payloads}
    assert len(serialized) == 1

    properties = payloads[0]["tools"][0]["parameters"]["properties"]
    assert properties["confidence"] == {"type": "number", "minimum": 0.0, "maximum": 100.0}
    assert '"minimum": 0.0, "maximum": 100.0' in serialized.pop()
    assert type(properties["ratio"]["exclusiveMinimum"]) is float
    assert properties["ratio"]["multipleOf"] == 0.5
    assert type(properties["count"]["minimum"]) is int
    assert type(properties["count"]["maximum"]) is int
    assert type(properties["either"]["minimum"]) is int
    assert properties["minimum"] == {"type": "string", "default": "0"}
    assert properties["items"]["items"] == {"type": "number", "minimum": -5.0, "default": 3}
    assert type(properties["items"]["items"]["default"]) is int


def test_chat_completions_number_bounds_are_floats() -> None:
    for clean_schemas in (True, False):
        payload = build_chat_completions_payload(_request(), clean_schemas=clean_schemas)
        properties = payload["tools"][0]["function"]["parameters"]["properties"]
        assert type(properties["confidence"]["minimum"]) is float
        assert type(properties["confidence"]["maximum"]) is float
        assert type(properties["count"]["maximum"]) is int


def test_float_number_bounds_does_not_mutate_its_input() -> None:
    schema = {"type": "number", "minimum": 1}
    assert float_number_bounds(schema) == {"type": "number", "minimum": 1.0}
    assert type(schema["minimum"]) is int
    assert float_number_bounds({"type": "number", "minimum": True}) == {
        "type": "number",
        "minimum": True,
    }


def test_float_number_bounds_leaves_literal_data_and_inexact_integers_alone() -> None:
    literal = {"type": "number", "minimum": 1}
    schema = {
        "type": "object",
        "default": literal,
        "examples": [literal],
        "properties": {
            "huge": {"type": "number", "maximum": 2**53 + 1},
            "exact": {"type": "number", "maximum": 2**53},
            "choice": {"anyOf": [{"type": "number", "minimum": 2}, {"const": literal}]},
        },
        "$defs": {"ratio": {"type": "number", "maximum": 1}},
    }

    converted = float_number_bounds(schema)

    assert converted["default"] == literal
    assert type(converted["default"]["minimum"]) is int
    assert type(converted["examples"][0]["minimum"]) is int
    assert converted["properties"]["huge"]["maximum"] == 2**53 + 1
    assert type(converted["properties"]["huge"]["maximum"]) is int
    assert type(converted["properties"]["exact"]["maximum"]) is float
    assert type(converted["properties"]["choice"]["anyOf"][0]["minimum"]) is float
    assert type(converted["properties"]["choice"]["anyOf"][1]["const"]["minimum"]) is int
    assert type(converted["$defs"]["ratio"]["maximum"]) is float
