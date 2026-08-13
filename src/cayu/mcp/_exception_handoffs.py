"""Authenticated in-process handoffs carried by MCP transport failures."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from cayu._exception_state import exception_state, set_exception_state

MCP_SESSION_CLOSE_TASK_ATTRIBUTE = "_cayu_mcp_session_close_task"
MCP_HTTP_SETTLEMENT_TASK_ATTRIBUTE = "_cayu_mcp_http_settlement_task"

_MCP_SESSION_CLOSE_TASK_TOKEN = object()
_MCP_HTTP_SETTLEMENT_TASK_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _McpFailureTaskHandoff:
    task: asyncio.Task[None]
    token: object


_MCP_FAILURE_HANDOFF_TOKENS = {
    MCP_SESSION_CLOSE_TASK_ATTRIBUTE: _MCP_SESSION_CLOSE_TASK_TOKEN,
    MCP_HTTP_SETTLEMENT_TASK_ATTRIBUTE: _MCP_HTTP_SETTLEMENT_TASK_TOKEN,
}


def attach_mcp_session_close_task(
    error: BaseException,
    task: asyncio.Task[None],
) -> None:
    _attach_mcp_failure_task(
        error,
        attribute_name=MCP_SESSION_CLOSE_TASK_ATTRIBUTE,
        task=task,
        token=_MCP_SESSION_CLOSE_TASK_TOKEN,
    )


def mcp_session_close_task(error: BaseException) -> asyncio.Task[None] | None:
    return _mcp_failure_task(
        error,
        attribute_name=MCP_SESSION_CLOSE_TASK_ATTRIBUTE,
        token=_MCP_SESSION_CLOSE_TASK_TOKEN,
    )


def attach_mcp_http_settlement_task(
    error: BaseException,
    task: asyncio.Task[None],
) -> None:
    _attach_mcp_failure_task(
        error,
        attribute_name=MCP_HTTP_SETTLEMENT_TASK_ATTRIBUTE,
        task=task,
        token=_MCP_HTTP_SETTLEMENT_TASK_TOKEN,
    )


def mcp_http_settlement_task(error: BaseException) -> asyncio.Task[None] | None:
    return _mcp_failure_task(
        error,
        attribute_name=MCP_HTTP_SETTLEMENT_TASK_ATTRIBUTE,
        token=_MCP_HTTP_SETTLEMENT_TASK_TOKEN,
    )


def copy_mcp_failure_handoffs(source: BaseException, target: BaseException) -> None:
    """Copy only handoffs authenticated by their runtime-owned attribute token."""

    for attribute_name, token in _MCP_FAILURE_HANDOFF_TOKENS.items():
        handoff = _authenticated_mcp_failure_task_handoff(
            exception_state(source, attribute_name),
            token=token,
        )
        if handoff is not None:
            set_exception_state(target, attribute_name, handoff)


def _attach_mcp_failure_task(
    error: BaseException,
    *,
    attribute_name: str,
    task: asyncio.Task[None],
    token: object,
) -> None:
    set_exception_state(
        error,
        attribute_name,
        _McpFailureTaskHandoff(task=task, token=token),
    )


def _mcp_failure_task(
    error: BaseException,
    *,
    attribute_name: str,
    token: object,
) -> asyncio.Task[None] | None:
    handoff = _authenticated_mcp_failure_task_handoff(
        exception_state(error, attribute_name),
        token=token,
    )
    if handoff is not None:
        return handoff.task
    return None


def _authenticated_mcp_failure_task_handoff(
    handoff: object,
    *,
    token: object,
) -> _McpFailureTaskHandoff | None:
    if (
        type(handoff) is _McpFailureTaskHandoff
        and handoff.token is token
        and isinstance(handoff.task, asyncio.Task)
    ):
        return handoff
    return None
