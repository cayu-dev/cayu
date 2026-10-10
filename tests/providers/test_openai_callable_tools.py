"""A stable tool catalogue with per-step callability on OpenAI Responses."""

from __future__ import annotations

import json

import pytest

from cayu import Message
from cayu.providers import ModelRequest
from cayu.providers.anthropic import build_anthropic_payload
from cayu.providers.openai import build_openai_payload
from cayu.providers.openai_subscription import OpenAISubscriptionProvider
from tests.core.test_openai_subscription_provider import StaticSubscriptionAuth

_TOOLS = [
    {
        "name": name,
        "description": f"Run {name}.",
        "input_schema": {"type": "object", "properties": {}},
    }
    for name in ("controller_sync_work_plan", "controller_update_work_plan", "lookup")
]


def _request(callable_names: list[str] | None, **openai_options) -> ModelRequest:
    options: dict = {}
    if callable_names is not None:
        options["callable_tool_names"] = callable_names
    if openai_options:
        options["openai"] = openai_options
    return ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "Plan the work.")],
        tools=_TOOLS,
        options=options,
    )


def _selectors(*names: str) -> list[dict[str, str]]:
    return [{"type": "function", "name": name} for name in names]


def test_exposure_changes_only_the_allowed_tools_choice() -> None:
    before = build_openai_payload(_request(["controller_sync_work_plan", "lookup"]))
    after = build_openai_payload(_request(["controller_update_work_plan", "lookup"]))

    assert json.dumps(before["tools"]) == json.dumps(after["tools"])
    assert [tool["name"] for tool in before["tools"]] == [tool["name"] for tool in _TOOLS]
    assert before["tool_choice"] == {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": _selectors("controller_sync_work_plan", "lookup"),
    }
    assert after["tool_choice"]["tools"] == _selectors("controller_update_work_plan", "lookup")


def test_without_callable_names_the_payload_is_unchanged() -> None:
    payload = build_openai_payload(_request(None))
    assert "tool_choice" not in payload


def test_configured_tool_choice_applies_within_the_callable_subset() -> None:
    required = build_openai_payload(_request(["lookup"], tool_choice="required"))
    assert required["tool_choice"] == {
        "type": "allowed_tools",
        "mode": "required",
        "tools": _selectors("lookup"),
    }
    named = build_openai_payload(
        _request(["lookup"], tool_choice={"type": "function", "name": "lookup"})
    )
    assert named["tool_choice"] == {"type": "function", "name": "lookup"}
    assert build_openai_payload(_request([]))["tool_choice"] == "none"
    with pytest.raises(ValueError, match="unavailable in the native request"):
        build_openai_payload(
            _request(
                ["lookup"],
                tool_choice={"type": "function", "name": "controller_update_work_plan"},
            )
        )


def test_callable_names_must_be_request_tools() -> None:
    with pytest.raises(ValueError, match="absent from the request"):
        build_openai_payload(_request(["missing"]))
    with pytest.raises(ValueError, match="cannot repeat"):
        build_openai_payload(_request(["lookup", "lookup"]))


def test_adapters_without_an_allow_list_send_the_full_catalogue() -> None:
    # Anthropic has no per-request allow-list: the catalogue stays fixed and the
    # runtime refuses calls outside the exposure.
    payload = build_anthropic_payload(_request(["lookup"]))
    assert [tool["name"] for tool in payload["tools"]] == [tool["name"] for tool in _TOOLS]
    assert "tool_choice" not in payload


def test_subscription_fingerprints_distinguish_callable_tool_subsets() -> None:
    provider = OpenAISubscriptionProvider(auth=StaticSubscriptionAuth())
    sync = _request(["controller_sync_work_plan"])
    update = _request(["controller_update_work_plan"])

    assert provider.request_fingerprint_options(sync) != provider.request_fingerprint_options(
        update
    )
    assert provider.request_fingerprint_options(sync)["openai"]["tool_choice"] == {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": _selectors("controller_sync_work_plan"),
    }
    assert provider.request_footprint_options(sync) != provider.request_footprint_options(update)
