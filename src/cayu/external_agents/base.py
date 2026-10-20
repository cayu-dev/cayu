"""Vendor harness runs, separate from Cayu's model-step provider interface."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from cayu.deadlines import ExecutionDeadline


@dataclass(frozen=True)
class ExternalAgentEvent:
    """One SDK observation. Usage is incremental, never a cumulative total."""

    kind: Literal["text", "tool_call", "tool_result", "usage", "result"]
    payload: dict[str, Any]
    id: str
    model: str | None = None
    usage_dialect: Literal["anthropic", "openai"] | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "text",
            "tool_call",
            "tool_result",
            "usage",
            "result",
        }:
            raise ValueError("Unknown external observation kind.")
        if not isinstance(self.id, str) or not self.id or len(self.id) > 4096:
            raise ValueError("External observation requires a bounded identity.")
        if not isinstance(self.payload, dict):
            raise ValueError("External observation payload must be an object.")
        if self.usage_dialect is not None and (
            self.kind != "usage" or self.usage_dialect not in {"anthropic", "openai"}
        ):
            raise ValueError("Usage dialect requires a supported usage observation.")


@dataclass(frozen=True)
class ExternalAgentContext:
    session_id: str
    prompt: str
    execution_deadline: ExecutionDeadline
    output_schema: dict[str, Any] | None = None
    authorize: Callable[[str, dict[str, Any], str], Awaitable[bool]] | None = None
    check_budget: Callable[[], Awaitable[None]] | None = None
    protect_secret: Callable[[str], None] | None = None
    contains_secret: Callable[[Any], bool] | None = None
    record_model_usage: Callable[[ExternalAgentEvent], Awaitable[None]] | None = None
    claim_model_usage: Callable[[], None] | None = None
    model_request_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    pending_model_usage: dict[str, ExternalAgentEvent] = field(default_factory=dict, repr=False)

    async def allow_tool(self, name: str, arguments: dict[str, Any], call_id: str) -> bool:
        self.execution_deadline.require_admission("external tool authorization")
        if self.authorize is None or (
            self.contains_secret is not None
            and self.contains_secret({"name": name, "arguments": arguments, "id": call_id})
        ):
            return False
        if self.check_budget is not None:
            await self.check_budget()
        return await self.authorize(name, arguments, call_id) is True


class ExternalAgentAdapter(Protocol):
    """A whole vendor loop. Closing the stream must join its process cleanup."""

    name: str
    provider_name: str
    model: str

    def stream(self, context: ExternalAgentContext) -> AsyncGenerator[ExternalAgentEvent]: ...


@dataclass(frozen=True)
class ExternalAgentResult:
    session_id: str
    task_id: str
    output: Any


@dataclass(frozen=True)
class ExternalAgentLaunch:
    """Prepared sandbox stdio launcher and its complete cleanup authority.

    The launcher executes the *whole CLI* inside a separately owned sandbox.
    ``close`` must stop and join all guest descendants, not just its stdio client.
    Model keys belong to the service proxy; ``environment`` contains only guest
    configuration and narrowly scoped proxy/MCP capabilities.
    """

    executable: str
    cwd: str
    close: Callable[[], Awaitable[None]]
    environment: Mapping[str, str] = field(default_factory=dict)
    secret_values: tuple[str, ...] = ()
    mcp_servers: Mapping[str, Any] = field(default_factory=dict)
    codex_config_overrides: tuple[str, ...] = ()
    # Install before the SDK starts the CLI so launchers can register credentials
    # created by individual processes before forwarding their output.
    bind_secret_protector: Callable[[Callable[[str], None]], None] | None = None

    def protect_secrets(self, protector: Callable[[str], None] | None) -> None:
        if protector is not None:
            for secret in self.secret_values:
                protector(secret)
            if self.bind_secret_protector is not None:
                self.bind_secret_protector(protector)


class ExternalAgentError(RuntimeError):
    """Content-free external harness failure; raw SDK exceptions stay private."""


class ExternalAgentBudgetExceeded(ExternalAgentError):
    pass
