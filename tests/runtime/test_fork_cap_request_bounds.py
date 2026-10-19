"""A capped fork's input bound covers everything a request carries to its adapter."""

from __future__ import annotations

from cayu.messages import Message
from cayu.providers.base import ModelRequest
from cayu.providers.hosted import OpenAIWebSearch
from cayu.runtime._fork_cap_requests import request_input_token_upper_bound

_SCHEMA_BYTES = 60_000


def _request(**fields) -> ModelRequest:
    return ModelRequest(model="fake-model", messages=[Message.text("user", "Answer.")], **fields)


def test_a_native_structured_output_schema_counts_toward_the_bound() -> None:
    # The runtime puts the schema in options, and a native adapter sends it
    # (for example as OpenAI's text.format); it is never copied into messages.
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "description": "x" * _SCHEMA_BYTES}},
    }
    request = _request(options={"structured_output": {"json_schema": schema, "strategy": "native"}})
    bound = request_input_token_upper_bound(request)
    assert bound is not None and bound > _SCHEMA_BYTES


def test_tool_projections_count_toward_the_bound() -> None:
    tool = {
        "name": "lookup",
        "description": "y" * _SCHEMA_BYTES,
        "input_schema": {"type": "object", "properties": {}},
    }
    request = _request(
        targeted_tool_projection={"marker_id": "sha256:" + "0" * 64, "tools": [tool]}
    )
    bound = request_input_token_upper_bound(request)
    assert bound is not None and bound > _SCHEMA_BYTES


def test_hosted_tools_cannot_be_bounded_by_request_bytes() -> None:
    # A provider-run search adds input the request itself never carries.
    assert request_input_token_upper_bound(_request(hosted_tools=(OpenAIWebSearch(),))) is None


def test_a_plain_request_is_still_bounded_by_its_bytes() -> None:
    bound = request_input_token_upper_bound(_request())
    assert bound is not None and len("Answer.") < bound < 1_000
