from __future__ import annotations

import asyncio
from datetime import date
from typing import cast

import pytest

from cayu import (
    ToolContext,
    ToolEffect,
    ToolResult,
    WebSearchAdapter,
    WebSearchAdapterRequest,
    WebSearchRestrictions,
    WebSearchTool,
)


class _SearchAdapter:
    def __init__(self, result: ToolResult) -> None:
        self.result = result
        self.calls: list[tuple[ToolContext, WebSearchAdapterRequest]] = []

    async def search(
        self,
        ctx: ToolContext,
        request: WebSearchAdapterRequest,
    ) -> ToolResult:
        self.calls.append((ctx, request))
        return self.result


def test_web_search_exposes_closed_application_bounded_contract() -> None:
    expected = ToolResult(content="search result", structured={"query": "cayu runtime"})
    adapter = _SearchAdapter(expected)
    context = ToolContext(session_id="sess_search")
    tool = WebSearchTool(
        adapter=adapter,
        default_results=3,
        max_results=7,
        max_snippet_bytes=321,
        max_total_snippet_bytes=654,
        timeout_seconds=9,
    )

    result = asyncio.run(
        tool.run(
            context,
            {"query": "  cayu\n\t runtime  ", "num_results": 6},
        )
    )

    assert result == expected
    assert adapter.calls == [
        (
            context,
            WebSearchAdapterRequest(
                query="cayu runtime",
                max_results=6,
                max_snippet_bytes=321,
                max_total_snippet_bytes=654,
                timeout_seconds=9.0,
            ),
        )
    ]
    assert tool.name == "web_search"
    assert tool.spec.effect is ToolEffect.NONE
    assert tool.schema == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 4096,
                "pattern": r"\S",
            },
            "num_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 7,
                "default": 3,
            },
        },
        "required": ["query"],
    }


def test_web_search_carries_application_owned_restrictions_outside_model_schema() -> None:
    adapter = _SearchAdapter(ToolResult(content="ok"))
    restrictions = WebSearchRestrictions(
        include_domains=("EXAMPLE.COM",),
        exclude_domains=("docs.example.net",),
        published_on_or_after=date(2026, 1, 2),
        country="us",
        locale="en-US",
        content_types=("APPLICATION/PDF",),
    )
    tool = WebSearchTool(adapter=adapter, restrictions=restrictions)

    result = asyncio.run(
        tool.run(ToolContext(session_id="sess_restricted_search"), {"query": "cayu"})
    )

    assert result.content == "ok"
    assert adapter.calls[0][1].restrictions == WebSearchRestrictions(
        include_domains=("example.com",),
        exclude_domains=("docs.example.net",),
        published_on_or_after=date(2026, 1, 2),
        country="US",
        locale="en-US",
        content_types=("application/pdf",),
    )
    assert set(tool.schema["properties"]) == {"query", "num_results"}


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "query", "unexpected": True},
        {"query": "query", "num_results": True},
        {"query": "query", "num_results": 0},
        {"query": "query", "num_results": 8},
        {"query": "bad\x00query"},
        {"query": "\ud800"},
        {"query": "é" * 2049},
    ],
)
def test_web_search_rejects_invalid_arguments_before_dispatch(args: dict[str, object]) -> None:
    adapter = _SearchAdapter(ToolResult(content="must not run"))

    result = asyncio.run(
        WebSearchTool(adapter=adapter, max_results=7).run(
            ToolContext(session_id="sess_invalid_search"),
            args,  # type: ignore[arg-type]
        )
    )

    assert result.structured == {"error": "invalid_arguments"}
    assert result.is_error is True
    assert adapter.calls == []


def test_web_search_uses_default_result_count() -> None:
    adapter = _SearchAdapter(ToolResult(content="ok"))

    asyncio.run(
        WebSearchTool(adapter=adapter, default_results=4, max_results=6).run(
            ToolContext(session_id="sess_search_default"),
            {"query": "query"},
        )
    )

    assert adapter.calls[0][1].max_results == 4


def test_web_search_validates_configuration_and_adapter_contract() -> None:
    adapter = _SearchAdapter(ToolResult(content="ok"))

    with pytest.raises(TypeError, match="WebSearchAdapter"):
        WebSearchTool(adapter=cast("WebSearchAdapter", object()))
    with pytest.raises(ValueError, match="default_results"):
        WebSearchTool(adapter=adapter, default_results=2, max_results=1)
    with pytest.raises(ValueError, match="max_results"):
        WebSearchTool(adapter=adapter, max_results=101)
    with pytest.raises(ValueError, match="max_snippet_bytes"):
        WebSearchTool(adapter=adapter, max_snippet_bytes=0)
    with pytest.raises(ValueError, match="max_total_snippet_bytes"):
        WebSearchTool(adapter=adapter, max_total_snippet_bytes=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        WebSearchTool(adapter=adapter, timeout_seconds=float("inf"))


def test_web_search_requires_exact_tool_result() -> None:
    class InvalidAdapter:
        async def search(
            self,
            ctx: ToolContext,
            request: WebSearchAdapterRequest,
        ) -> object:
            del ctx, request
            return object()

    with pytest.raises(TypeError, match="must return ToolResult"):
        asyncio.run(
            WebSearchTool(adapter=cast("WebSearchAdapter", InvalidAdapter())).run(
                ToolContext(session_id="sess_invalid_result"),
                {"query": "query"},
            )
        )


def test_web_search_timeout_is_bounded_and_cancellation_remains_authoritative() -> None:
    started = asyncio.Event()

    class BlockingAdapter:
        async def search(
            self,
            ctx: ToolContext,
            request: WebSearchAdapterRequest,
        ) -> ToolResult:
            del ctx, request
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def timeout_scenario() -> ToolResult:
        return await WebSearchTool(
            adapter=BlockingAdapter(),
            timeout_seconds=0.01,
        ).run(ToolContext(session_id="sess_search_timeout"), {"query": "query"})

    timeout_result = asyncio.run(timeout_scenario())
    assert timeout_result.structured == {"error": "timeout"}

    async def cancellation_scenario() -> None:
        task = asyncio.create_task(
            WebSearchTool(adapter=BlockingAdapter()).run(
                ToolContext(session_id="sess_search_cancel"),
                {"query": "query"},
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancellation_scenario())
