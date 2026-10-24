"""Authenticated read-only HTTP assembly for tool usage."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from cayu.server.contracts import ApiErrorResponse, ToolCallsResponse, ToolUsageResponse
from cayu.sessions.event_queries import EventQueryResultTooLarge
from cayu.sessions.tool_usage import InvalidToolUsageQuery, ToolUsageConflict, ToolUsageReader


def register_tool_usage(
    router: APIRouter, *, protected: list[Any], reader: Callable[[], ToolUsageReader]
) -> None:
    responses: dict[int | str, dict[str, Any]] = {
        409: {
            "model": ApiErrorResponse,
            "description": "The retained source or cursor binding changed; restart the listing.",
        },
        413: {
            "model": ApiErrorResponse,
            "description": "An individual event exceeds the bounded read size.",
        },
        501: {
            "model": ApiErrorResponse,
            "description": "The store does not implement the required bounded reads.",
        },
    }

    @router.get(
        "/tools/usage",
        response_model=ToolUsageResponse,
        dependencies=protected,
        responses=responses,
    )
    async def tool_usage():
        try:
            return await reader().totals()
        except EventQueryResultTooLarge:
            raise HTTPException(
                413, "A retained tool event exceeds the bounded read size."
            ) from None
        except NotImplementedError:
            raise HTTPException(
                501, "This store does not support bounded tool usage reads."
            ) from None
        except ToolUsageConflict as exc:
            raise HTTPException(409, str(exc)) from None

    @router.get(
        "/tools/calls",
        response_model=ToolCallsResponse,
        dependencies=protected,
        responses=responses,
    )
    async def tool_calls(
        name: str | None = Query(default=None, max_length=4096),
        cursor: str | None = Query(default=None, max_length=8192),
        limit: int = Query(default=20, ge=1, le=100),
    ):
        try:
            return await reader().recent(name=name, cursor=cursor, limit=limit)
        except EventQueryResultTooLarge:
            raise HTTPException(
                413, "A retained tool event exceeds the bounded read size."
            ) from None
        except NotImplementedError:
            raise HTTPException(
                501, "This store does not support bounded tool usage reads."
            ) from None
        except ToolUsageConflict as exc:
            raise HTTPException(409, str(exc)) from None
        except InvalidToolUsageQuery:
            raise HTTPException(
                422,
                [
                    {
                        "loc": ["query", "cursor"],
                        "msg": "Invalid tool-call cursor.",
                        "type": "value_error",
                    }
                ],
            ) from None
