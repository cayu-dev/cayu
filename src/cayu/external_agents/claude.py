"""Run Claude Code through an owned launcher.

See ``cayu guide external-agents`` for setup and examples.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, aclosing, suppress
from decimal import Decimal
from importlib import import_module
from typing import Any

from jsonschema import Draft202012Validator

from cayu.external_agents._cleanup import close_external_guest
from cayu.external_agents.base import (
    ExternalAgentContext,
    ExternalAgentError,
    ExternalAgentEvent,
    ExternalAgentLaunch,
)
from cayu.external_agents.environment import join_cleanup, scrub_external_environment

_COUNTERS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


async def _owned_messages(sdk, options, prompt):
    """Keep SDK task-group ownership and disconnect in one explicitly joined task."""
    queue = asyncio.Queue(maxsize=1)
    accepted = asyncio.Event()

    async def receive() -> None:
        async with sdk.ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                await queue.put(message)
                await accepted.wait()
                accepted.clear()

    owner = asyncio.create_task(receive())
    pending = None
    try:
        while True:
            pending = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait((pending, owner), return_when=asyncio.FIRST_COMPLETED)
            if pending not in done:
                await owner
                return
            message = pending.result()
            pending = None
            yield message
            accepted.set()
    finally:

        async def close() -> None:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            if not owner.done():
                owner.cancel()
            with suppress(asyncio.CancelledError):
                await owner

        await join_cleanup(close())


def _usage(value: dict[str, Any]) -> dict[str, int]:
    aliases = {
        "input_tokens": "inputTokens",
        "output_tokens": "outputTokens",
        "cache_read_input_tokens": "cacheReadInputTokens",
        "cache_creation_input_tokens": "cacheCreationInputTokens",
    }
    result = {key: value.get(key, value.get(alias, 0)) for key, alias in aliases.items()}
    if any(type(item) is not int or item < 0 for item in result.values()):
        raise ExternalAgentError("Claude returned invalid token usage.")
    return result


class ClaudeAgentAdapter:
    """Run Claude Code as a Cayu session. See ``cayu guide external-agents``."""

    name = "claude-agent-sdk"
    provider_name = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        launch: Callable[[ExternalAgentContext], AbstractAsyncContextManager[ExternalAgentLaunch]],
        effort: str | None = None,
        tools: tuple[str, ...] = ("Read", "Grep", "Glob"),
        mcp_servers: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.launch = launch
        self.effort = effort
        self.tools = tools
        self.mcp_servers = mcp_servers or {}

    async def stream(self, context: ExternalAgentContext) -> AsyncGenerator[ExternalAgentEvent]:
        try:
            sdk = import_module("claude_agent_sdk")
        except ImportError:
            raise ImportError("Install cayu[claude-agent-sdk] to use ClaudeAgentAdapter.") from None
        decisions: dict[str, tuple[str, dict[str, Any], bool]] = {}

        async def authorize(name: str, arguments: dict[str, Any], call_id: str) -> bool:
            previous = decisions.get(call_id)
            if previous is not None:
                if previous[:2] != (name, arguments):
                    return False
                return previous[2]
            if name == "StructuredOutput" and context.output_schema is not None:
                # The SDK injects this result carrier independently of the
                # caller's executable tool allowlist. Its payload still crosses
                # the run's deadline, secret, budget and access boundaries.
                context.execution_deadline.require_admission("external structured output")
                allowed = not (
                    context.contains_secret is not None
                    and context.contains_secret(
                        {"name": name, "arguments": arguments, "id": call_id}
                    )
                )
                if allowed and context.check_budget is not None:
                    await context.check_budget()
                allowed = allowed and Draft202012Validator(context.output_schema).is_valid(
                    arguments
                )
            else:
                allowed = await context.allow_tool(name, arguments, call_id)
            decisions[call_id] = name, dict(arguments), allowed
            return allowed

        async def pre_tool(data, tool_use_id, hook_context):
            del hook_context
            allowed = await authorize(
                data["tool_name"], data["tool_input"], tool_use_id or "missing"
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow" if allowed else "deny",
                }
            }

        async def permission(name, arguments, permission_context):
            call_id = getattr(permission_context, "tool_use_id", None)
            allowed = call_id is not None and await authorize(name, arguments, call_id)
            return (
                sdk.PermissionResultAllow(updated_input=arguments)
                if allowed
                else sdk.PermissionResultDeny(message="Cayu tool authorization denied.")
            )

        async def prompt():
            yield {"type": "user", "message": {"role": "user", "content": context.prompt}}

        totals: dict[str, dict[str, int]] = {}
        messages: dict[str, dict[str, int]] = {}
        streams: dict[str | None, tuple[str, str, dict[str, Any]]] = {}
        final_usage: set[str | None] = set()
        async with self.launch(context) as launch:
            try:
                launch.protect_secrets(context.protect_secret)
                options = sdk.ClaudeAgentOptions(
                    model=self.model,
                    cwd=launch.cwd,
                    cli_path=launch.executable,
                    system_prompt={"type": "preset", "preset": "claude_code"},
                    tools=list(self.tools),
                    permission_mode="default",
                    allowed_tools=[],
                    can_use_tool=permission,
                    hooks={"PreToolUse": [sdk.HookMatcher(hooks=[pre_tool])]},
                    setting_sources=[],
                    include_partial_messages=True,
                    # Guest diagnostics bypass recorder policy and can contain
                    # scoped capabilities or private tool data. Never publish them.
                    stderr=lambda _line: None,
                    strict_mcp_config=True,
                    mcp_servers={**self.mcp_servers, **launch.mcp_servers},
                    env=scrub_external_environment(launch.environment),
                    effort=self.effort,
                    output_format=None
                    if context.output_schema is None
                    else {"type": "json_schema", "schema": context.output_schema},
                    extra_args={"no-session-persistence": None},
                )
                sequence = 0
                async with aclosing(_owned_messages(sdk, options, prompt())) as stream:
                    async for message in stream:
                        sequence += 1
                        if isinstance(message, sdk.AssistantMessage):
                            for index, block in enumerate(message.content):
                                event_id = f"{message.uuid or sequence}:block:{index}"
                                if isinstance(block, sdk.TextBlock):
                                    yield ExternalAgentEvent("text", {"text": block.text}, event_id)
                                elif isinstance(block, sdk.ToolUseBlock):
                                    yield ExternalAgentEvent(
                                        "tool_call",
                                        {
                                            "name": block.name,
                                            "arguments": block.input,
                                            "call_id": block.id,
                                        },
                                        event_id,
                                    )
                        elif isinstance(message, sdk.StreamEvent):
                            event = message.event
                            key = message.parent_tool_use_id
                            kind = event.get("type")
                            if kind == "message_start":
                                raw = event["message"]
                                final_usage.discard(key)
                                streams[key] = (
                                    raw["id"],
                                    raw["model"],
                                    dict(raw.get("usage") or {}),
                                )
                            elif kind == "message_delta" and key in streams:
                                streams[key][2].update(event.get("usage") or {})
                                if "output_tokens" in (event.get("usage") or {}):
                                    final_usage.add(key)
                            elif kind == "message_stop" and key in streams:
                                identity, model, raw = streams.pop(key)
                                if key not in final_usage:
                                    yield ExternalAgentEvent(
                                        "usage", {}, f"{identity}:unknown", model
                                    )
                                    continue
                                final_usage.discard(key)
                                current = _usage(raw)
                                previous = messages.get(identity)
                                if previous is not None:
                                    if previous != current:
                                        raise ExternalAgentError(
                                            "Claude final message usage conflicts."
                                        )
                                    continue
                                messages[identity] = current
                                total = totals.setdefault(model, dict.fromkeys(_COUNTERS, 0))
                                for counter in _COUNTERS:
                                    total[counter] += current[counter]
                                yield ExternalAgentEvent(
                                    "usage", current, f"{identity}:usage", model
                                )
                        elif isinstance(message, sdk.UserMessage):
                            if isinstance(message.content, list):
                                for index, block in enumerate(message.content):
                                    if isinstance(block, sdk.ToolResultBlock):
                                        yield ExternalAgentEvent(
                                            "tool_result",
                                            {
                                                "call_id": block.tool_use_id,
                                                "content": block.content,
                                                "is_error": block.is_error,
                                            },
                                            f"result:{sequence}:{index}",
                                        )
                        elif isinstance(message, sdk.ResultMessage):
                            # SDK total keys may alias API message model IDs.
                            # Reconcile aggregate counters first so a model alias
                            # cannot charge the same completed calls twice.
                            if message.model_usage:
                                final = {
                                    model: _usage(raw) for model, raw in message.model_usage.items()
                                }
                                current = {
                                    key: sum(value[key] for value in final.values())
                                    for key in _COUNTERS
                                }
                                previous = {
                                    key: sum(value[key] for value in totals.values())
                                    for key in _COUNTERS
                                }
                                if any(current[key] < previous[key] for key in _COUNTERS):
                                    raise ExternalAgentError(
                                        "Claude final usage conflicts with step usage."
                                    )
                                residual = {key: current[key] - previous[key] for key in _COUNTERS}
                                if any(residual.values()):
                                    if len(final) == 1:
                                        model = next(iter(final))
                                        yield ExternalAgentEvent(
                                            "usage", residual, "final:usage", model
                                        )
                                    elif set(totals).issubset(final):
                                        for model, value in final.items():
                                            seen = totals.get(model, dict.fromkeys(_COUNTERS, 0))
                                            delta = {
                                                key: value[key] - seen[key] for key in _COUNTERS
                                            }
                                            if any(amount < 0 for amount in delta.values()):
                                                raise ExternalAgentError(
                                                    "Claude model usage counters conflict."
                                                )
                                            if any(delta.values()):
                                                yield ExternalAgentEvent(
                                                    "usage", delta, f"final:{model}", model
                                                )
                                    else:
                                        # Partial multi-model receipts with aliases
                                        # cannot be priced without guessing which
                                        # model incurred the missing calls.
                                        yield ExternalAgentEvent(
                                            "usage", {}, "final:ambiguous-model-usage", self.model
                                        )
                            elif not messages and message.usage:
                                yield ExternalAgentEvent(
                                    "usage", _usage(message.usage), "final:usage", self.model
                                )
                            elif not messages:
                                yield ExternalAgentEvent(
                                    "usage", {}, "final:missing-usage", self.model
                                )
                            if message.total_cost_usd is not None:
                                cost = Decimal(str(message.total_cost_usd))
                                if not cost.is_finite() or not 0 <= cost < Decimal("100000000"):
                                    raise ExternalAgentError(
                                        "Claude returned invalid reported cost."
                                    )
                                yield ExternalAgentEvent(
                                    "usage",
                                    {
                                        "input_tokens": 0,
                                        "output_tokens": 0,
                                        "cost": f"{cost:.9f}",
                                        "cost_currency": "USD",
                                        "cost_status": "reported",
                                    },
                                    "final:reported-cost",
                                    self.model,
                                )
                            if message.is_error:
                                raise ExternalAgentError("Claude harness returned an error result.")
                            if (
                                context.output_schema is not None
                                and message.structured_output is None
                            ):
                                raise ExternalAgentError("Claude returned no structured output.")
                            yield ExternalAgentEvent(
                                "result",
                                {
                                    "output": message.structured_output
                                    if context.output_schema is not None
                                    else message.result
                                },
                                "final:result",
                            )
            finally:
                await close_external_guest(launch.close)
