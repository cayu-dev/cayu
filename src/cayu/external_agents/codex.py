"""Optional Codex app-server SDK adapter, with deny-by-default approvals."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, suppress
from importlib import import_module
from typing import Any

from cayu.external_agents._cleanup import close_external_guest
from cayu.external_agents._codex_transport import BoundedCodexTransport
from cayu.external_agents.base import (
    ExternalAgentContext,
    ExternalAgentError,
    ExternalAgentEvent,
    ExternalAgentLaunch,
)
from cayu.external_agents.environment import (
    CODEX_RUN_OVERRIDES,
    codex_proxy_overrides,
    join_cleanup,
    scrub_external_environment,
)


class CodexAgentAdapter:
    """Run Codex as a Cayu session. See ``cayu guide external-agents``."""

    name = "codex"
    provider_name = "openai"

    def __init__(
        self,
        *,
        model: str,
        launch: Callable[[ExternalAgentContext], AbstractAsyncContextManager[ExternalAgentLaunch]],
        base_url: str | None = None,
        effort: str | None = None,
        sandbox: str = "read-only",
        config_overrides: tuple[str, ...] = (),
    ) -> None:
        """``base_url`` routes model calls through a Cayu gateway capability.

        Without it Codex uses its own provider configuration and credentials,
        for example an existing login with ``local_agent_launch``.
        """
        self.model = model
        self.launch = launch
        self.effort = effort
        self.sandbox = sandbox
        provider = CODEX_RUN_OVERRIDES if base_url is None else codex_proxy_overrides(base_url)
        self.overrides = (*provider, *config_overrides)

    async def stream(self, context: ExternalAgentContext) -> AsyncGenerator[ExternalAgentEvent]:
        try:
            sdk = import_module("openai_codex.client")
        except ImportError:
            raise ImportError("Install cayu[codex] to use CodexAgentAdapter.") from None
        loop = asyncio.get_running_loop()
        closing = threading.Event()
        futures: set[Any] = set()
        futures_lock = threading.Lock()
        approval_tasks: set[asyncio.Task] = set()

        async def allow(name, arguments, call_id):
            if closing.is_set():
                return False
            task = asyncio.current_task()
            assert task is not None
            approval_tasks.add(task)
            try:
                return await context.allow_tool(name, arguments, call_id)
            finally:
                approval_tasks.discard(task)

        def approval(method, parameters):
            params = parameters or {}
            names = {
                "item/commandExecution/requestApproval": "codex_command",
                "item/fileChange/requestApproval": "codex_file_change",
            }
            if closing.is_set() or method not in names:
                return {"decision": "decline"}
            call_id = str(params.get("itemId") or params.get("callId") or "missing")
            future = asyncio.run_coroutine_threadsafe(allow(names[method], params, call_id), loop)
            with futures_lock:
                futures.add(future)
                if closing.is_set():
                    future.cancel()
            try:
                allowed = future.result(timeout=context.execution_deadline.remaining_seconds())
            except BaseException:
                future.cancel()
                allowed = False
            finally:
                with futures_lock:
                    futures.discard(future)
            return {"decision": "accept" if allowed else "decline"}

        async with self.launch(context) as launch:
            client = None
            pending: asyncio.Task[Any] | None = None
            try:
                launch.protect_secrets(context.protect_secret)

                class Client(BoundedCodexTransport, sdk.CodexClient):
                    pass

                client = Client(
                    sdk.CodexConfig(
                        codex_bin=launch.executable,
                        config_overrides=(*self.overrides, *launch.codex_config_overrides),
                        cwd=launch.cwd,
                        env=scrub_external_environment(launch.environment),
                    ),
                    approval_handler=approval,
                )

                # start() only spawns the local stdio tunnel. Keep it on the owning
                # thread so cancellation cannot race a later process creation.
                async def call(function, *args):
                    nonlocal pending
                    pending = asyncio.create_task(asyncio.to_thread(function, *args))
                    return await asyncio.shield(pending)

                client.start()
                await call(client.initialize)
                thread = await call(
                    client.thread_start,
                    {
                        "model": self.model,
                        "cwd": launch.cwd,
                        "ephemeral": True,
                        "approvalPolicy": "untrusted",
                        "sandbox": "danger-full-access"
                        if self.sandbox == "external"
                        else self.sandbox,
                    },
                )
                parameters: dict[str, Any] = {}
                if self.sandbox == "external":
                    # The launcher owns filesystem and network containment.
                    # Do not nest an unavailable bwrap sandbox inside it.
                    parameters["sandboxPolicy"] = {
                        "type": "externalSandbox",
                        "networkAccess": "enabled",
                    }
                if self.effort is not None:
                    parameters["effort"] = self.effort
                if context.output_schema is not None:
                    parameters["outputSchema"] = context.output_schema
                turn = await call(client.turn_start, thread.thread.id, context.prompt, parameters)
                counters: dict[str, int] = {}
                final_text: str | None = None
                fallback_text: str | None = None
                sequence = 0
                while True:
                    notification = await call(client.next_turn_notification, turn.turn.id)
                    payload = notification.payload.model_dump(mode="json", by_alias=True)
                    method = notification.method
                    sequence += 1
                    if method == "thread/tokenUsage/updated":
                        total = payload["tokenUsage"]["total"]
                        current = {
                            "input_tokens": total["inputTokens"],
                            "output_tokens": total["outputTokens"],
                            "cached_input_tokens": total.get("cachedInputTokens", 0),
                            "cache_write_input_tokens": total.get("cacheWriteInputTokens") or 0,
                            "reasoning_output_tokens": total.get("reasoningOutputTokens", 0),
                        }
                        if any(
                            type(value) is not int or value < counters.get(key, 0)
                            for key, value in current.items()
                        ):
                            raise ExternalAgentError(
                                "Codex token counters regressed or were invalid."
                            )
                        delta = {
                            key: value - counters.get(key, 0) for key, value in current.items()
                        }
                        counters = current
                        yield ExternalAgentEvent(
                            "usage",
                            {
                                "input_tokens": delta["input_tokens"],
                                "output_tokens": delta["output_tokens"],
                                "input_tokens_details": {
                                    "cached_tokens": delta["cached_input_tokens"],
                                    "cache_write_tokens": delta["cache_write_input_tokens"],
                                },
                                "output_tokens_details": {
                                    "reasoning_tokens": delta["reasoning_output_tokens"]
                                },
                            },
                            f"usage:{sequence}",
                            self.model,
                        )
                    elif method == "item/agentMessage/delta":
                        yield ExternalAgentEvent(
                            "text", {"text": payload["delta"]}, f"text:{sequence}"
                        )
                    elif method in {"item/started", "item/completed"}:
                        item = payload["item"]
                        if method == "item/completed" and item.get("type") == "agentMessage":
                            if item.get("phase") == "final_answer":
                                final_text = item["text"]
                            elif item.get("phase") is None:
                                fallback_text = item["text"]
                        if item.get("type") in {
                            "commandExecution",
                            "fileChange",
                            "mcpToolCall",
                            "dynamicToolCall",
                        }:
                            yield ExternalAgentEvent(
                                "tool_call" if method == "item/started" else "tool_result",
                                item,
                                f"item:{sequence}",
                            )
                    elif method == "turn/completed":
                        if not counters:
                            yield ExternalAgentEvent("usage", {}, "final:missing-usage", self.model)
                        if payload["turn"]["status"] != "completed":
                            raise ExternalAgentError("Codex turn did not complete.")
                        output = final_text if final_text is not None else fallback_text
                        if output is None:
                            raise ExternalAgentError("Codex returned no final response.")
                        if context.output_schema is not None:
                            try:
                                output = json.loads(output)
                            except ValueError:
                                raise ExternalAgentError(
                                    "Codex returned invalid structured output."
                                ) from None
                        yield ExternalAgentEvent("result", {"output": output}, "final:result")
                        break
            finally:

                async def close() -> None:
                    closing.set()
                    with futures_lock:
                        for future in tuple(futures):
                            future.cancel()
                    tasks = tuple(approval_tasks)
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    try:
                        # Removing the guest releases blocked stdin writers before
                        # the SDK closes that same buffered pipe and joins them.
                        await close_external_guest(launch.close)
                    finally:
                        if client is not None:
                            await asyncio.to_thread(client.stop_transport)
                            try:
                                with suppress(BrokenPipeError):
                                    await asyncio.to_thread(client.close)
                            finally:
                                if pending is not None:
                                    await asyncio.gather(pending, return_exceptions=True)

                await join_cleanup(close())
