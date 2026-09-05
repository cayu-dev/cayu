from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.core.test_browser_session import _context, _FakeBrowserBackend

from cayu import Message
from cayu.providers.base import ModelRequest
from cayu.providers.chat_completions import build_chat_completions_payload
from cayu.providers.openai import build_openai_payload
from cayu.tools.browser_session import BrowserSessionTool

_PAGE = {"session_id": "bs_test", "page_id": "bp_test"}
_ACTION = {**_PAGE, "expected_revision": "br_test", "expected_control_epoch": 1}
_OPERATIONS = {
    "navigate": {"url": "https://example.test/"},
    "observe": _PAGE,
    "observe_visual": _PAGE,
    "click_visual_target": {
        **_ACTION,
        "visual_revision": "vr_" + "1" * 32,
        "visual_ref": "vt_" + "2" * 32,
    },
    "click_visual_point": {
        **_ACTION,
        "visual_revision": "vr_" + "1" * 32,
        "screenshot_sha256": "3" * 64,
        "x": 0.5,
        "y": 0.5,
    },
    "click": {**_ACTION, "ref": "ref_test"},
    "fill": {**_ACTION, "ref": "ref_test", "value": "hello"},
    "select": {**_ACTION, "ref": "ref_test", "value": "hello"},
    "press": {**_ACTION, "ref": "ref_test", "key": "Enter"},
    "wait": {**_ACTION, "wait_ms": 1},
    "back": _ACTION,
    "forward": _ACTION,
    "reload": _ACTION,
    "scroll": {**_ACTION, "direction": "down", "amount": "page", "repeat_count": 1},
    "hover": {**_ACTION, "ref": "ref_test"},
    "upload": {**_ACTION, "ref": "ref_test", "artifact_ids": ["art_" + "4" * 32]},
    "screenshot": _ACTION,
    "download": {**_ACTION, "ref": "ref_test"},
    "list_pages": {"session_id": "bs_test"},
    "switch_page": _PAGE,
    "close_page": _PAGE,
    "close": {"session_id": "bs_test"},
}


def test_browser_schema_is_portable_through_both_openai_payload_builders() -> None:
    tool = BrowserSessionTool()
    request = ModelRequest(
        model="gpt-4.1-mini",
        messages=[Message.text("user", "Inspect the local shop.")],
        tools=[
            {
                "name": tool.spec.name,
                "description": tool.spec.description,
                "input_schema": tool.schema,
            }
        ],
    )
    chat = build_chat_completions_payload(request)["tools"][0]["function"]["parameters"]
    responses = build_openai_payload(request)["tools"][0]["parameters"]
    for schema in (chat, responses):
        assert schema == tool.schema
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert not {"allOf", "anyOf", "oneOf", "not", "enum", "const"} & schema.keys()
        assert set(schema["properties"]["operation"]["enum"]) == set(_OPERATIONS)


@pytest.mark.parametrize("operation,fields", _OPERATIONS.items())
def test_portable_browser_schema_keeps_runtime_required_fields(
    operation: str,
    fields: dict[str, object],
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        backend = _FakeBrowserBackend()
        tool = BrowserSessionTool(_backend=backend)
        context = _context(tmp_path)
        complete = {"operation": operation, "operation_id": "schema-check", **fields}
        for missing in complete:
            incomplete = {key: value for key, value in complete.items() if key != missing}
            result = await tool.run(context, incomplete)
            assert result.is_error, (operation, missing)
            assert result.structured is not None
            assert result.structured["error"] == "invalid_arguments", (operation, missing)
            assert result.structured["execution"]["dispatch"] == "not_started"
            assert backend.calls == []

        # A schema-known field belonging to another operation must not become
        # usable merely because the provider sees the union of all properties.
        extra = {"ref": "ref_test"} if operation == "navigate" else {"url": "https://example.test/"}
        result = await tool.run(context, {**complete, **extra})
        assert result.is_error
        assert result.structured is not None
        assert result.structured["error"] == "invalid_arguments"
        assert result.structured["execution"]["dispatch"] == "not_started"
        assert backend.calls == []

    asyncio.run(exercise())
