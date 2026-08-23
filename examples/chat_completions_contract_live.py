from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from _live_checks import require, require_equal
from cayu import (
    AgentSpec,
    CayuApp,
    ChatCompletionsProvider,
    Environment,
    EnvironmentSpec,
    Event,
    EventType,
    LocalRunner,
    LocalWorkspace,
    Message,
    ProviderStatePart,
    ReadFileTool,
    RunRequest,
    StructuredOutputSpec,
    WriteFileTool,
)
from cayu.providers import ChatCompletionsTransport, HttpxChatCompletionsTransport

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["passed"]},
        "adapter": {"type": "string", "enum": ["chat_completions"]},
    },
    "required": ["status", "adapter"],
    "additionalProperties": False,
}


async def main() -> None:
    selection = _live_selection()
    if selection is None:
        return

    provider_name, model = selection
    with tempfile.TemporaryDirectory(prefix="cayu-chat-completions-contract-") as directory:
        root = Path(directory)
        await _run_tool_contract(provider_name, model, root)
        await _run_structured_output_contract(provider_name, model, root)

    print("completed")


async def _run_tool_contract(provider_name: str, model: str, root: Path) -> None:
    session_id = f"chat_contract_tools_{uuid4().hex}"
    workspace = LocalWorkspace(root, workspace_id="chat-contract-tools")
    transport = _RecordingChatCompletionsTransport() if provider_name == "openrouter" else None
    app = _app(
        provider_name=provider_name,
        workspace=workspace,
        runner=LocalRunner(root, inherit_env=False),
        transport=transport,
    )
    app.register_agent(
        AgentSpec(
            name="assistant",
            model=model,
            system_prompt=(
                "You are verifying the Chat Completions adapter. Use tools exactly "
                "when the user requests file work. Do not claim a tool ran unless it did."
            ),
            provider_options=(
                {"openrouter": {"reasoning": {"enabled": True}}}
                if provider_name == "openrouter"
                else {}
            ),
        ),
        tools=[WriteFileTool(), ReadFileTool()],
    )

    events = await _collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[
                Message.text(
                    "user",
                    (
                        "Use write_file to create contract/result.txt with exactly "
                        "'chat contract ok'. Then use read_file to read contract/result.txt. "
                        "After the tool result is returned, give a short final answer."
                    ),
                )
            ],
            max_steps=5,
        ),
    )

    _require_completed(events)
    _require_model_usage(events)
    _require_finish_reason(events, "tool_calls")
    _require_tool_completed(events, "write_file")
    _require_tool_completed(events, "read_file")
    require_equal(
        (root / "contract" / "result.txt").read_text(encoding="utf-8"),
        "chat contract ok",
        "written file",
    )
    usage = await app.get_session_usage(session_id)
    require(usage.usage.total_tokens > 0, "missing total token usage")
    if transport is not None:
        _require_openrouter_route_evidence(events, requested_model=model)
        transcript = await app.session_store.load_transcript(session_id)
        reasoning_sequences = [
            part.state["details"]
            for message in transcript
            for part in message.content
            if type(part) is ProviderStatePart
            and part.provider == "chat_completions"
            and part.state.get("type") == "reasoning_details"
        ]
        require(bool(reasoning_sequences), "OpenRouter returned no reasoning_details")
        replayed_sequences = [
            message["reasoning_details"]
            for call in transport.calls[1:]
            for message in call["messages"]
            if isinstance(message, dict) and "reasoning_details" in message
        ]
        require(
            all(
                sequence in replayed_sequences
                for sequence in reasoning_sequences[:-1] or reasoning_sequences
            ),
            "OpenRouter reasoning_details were not replayed losslessly",
        )
    print("tool_contract verified")


async def _run_structured_output_contract(provider_name: str, model: str, root: Path) -> None:
    session_id = f"chat_contract_structured_{uuid4().hex}"
    app = _app(
        provider_name=provider_name,
        workspace=LocalWorkspace(root, workspace_id="chat-contract-structured"),
        runner=LocalRunner(root, inherit_env=False),
    )
    app.register_agent(
        AgentSpec(
            name="assistant",
            model=model,
            system_prompt=(
                "Use the structured-output tool for the final answer. Do not return "
                "JSON as plain text."
            ),
        )
    )

    events = await _collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[
                Message.text(
                    "user",
                    (
                        "Return structured output with status='passed' and "
                        "adapter='chat_completions'."
                    ),
                )
            ],
            structured_output=StructuredOutputSpec(
                name="chat_completions_contract",
                json_schema=STRUCTURED_SCHEMA,
                max_retries=2,
                repair_prompt=(
                    "Call the structured-output tool with an output object containing "
                    "status='passed' and adapter='chat_completions'."
                ),
            ),
            max_steps=4,
        ),
    )

    _require_completed(events)
    _require_model_usage(events)
    _require_finish_reason(events, "tool_calls")
    validated = [event for event in events if event.type == EventType.STRUCTURED_OUTPUT_VALIDATED]
    require(bool(validated), "structured output was not validated")
    require_equal(
        validated[-1].payload.get("output"),
        {"status": "passed", "adapter": "chat_completions"},
        "structured output",
    )
    usage = await app.get_session_usage(session_id)
    require(usage.usage.total_tokens > 0, "missing total token usage")
    print("structured_output_contract verified")


def _live_selection() -> tuple[str, str] | None:
    selected = os.environ.get("CAYU_PROVIDER", "gemini").strip().lower()
    if selected == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("Set OPENROUTER_API_KEY to run the live OpenRouter contract check.")
            return None
        model = os.environ.get("CAYU_OPENROUTER_MODEL")
        if not model or not model.strip():
            print("Set CAYU_OPENROUTER_MODEL to an explicit reasoning-capable model slug.")
            return None
        return selected, model.strip()
    if selected != "gemini":
        raise RuntimeError("CAYU_PROVIDER must be gemini or openrouter for this live check.")
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY to run this live Chat Completions contract check.")
        return None
    return selected, os.environ.get(
        "CAYU_CHAT_COMPLETIONS_CONTRACT_MODEL",
        "gemini-3.1-flash-lite",
    )


class _RecordingChatCompletionsTransport:
    def __init__(self) -> None:
        self._delegate = HttpxChatCompletionsTransport()
        self.calls: list[dict[str, Any]] = []

    async def aclose(self) -> None:
        await self._delegate.aclose()

    async def stream_chat_completions(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        self.calls.append(deepcopy(dict(payload)))
        async for event in self._delegate.stream_chat_completions(
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
            stream_idle_timeout_s=stream_idle_timeout_s,
        ):
            yield event


def _app(
    *,
    provider_name: str,
    workspace: LocalWorkspace,
    runner: LocalRunner,
    transport: ChatCompletionsTransport | None = None,
) -> CayuApp:
    app = CayuApp(enable_logging=False)
    if provider_name == "openrouter":
        provider = ChatCompletionsProvider(
            name="openrouter",
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
            openrouter_router_metadata=True,
            transport=transport,
        )
    else:
        provider = ChatCompletionsProvider(
            name="google",
            api_key_env="GEMINI_API_KEY",
            base_url=GEMINI_BASE_URL,
            document_encoding="image_url",
        )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-contract"),
            workspace=workspace,
            runner=runner,
        ),
        default=True,
    )
    return app


async def _collect_events(app: CayuApp, request: RunRequest) -> list[Event]:
    events: list[Event] = []
    async for event in app.run(request):
        print(
            event.type,
            event.environment_name or "-",
            event.tool_name or "-",
            event.payload,
        )
        events.append(event)
    return events


def _require_completed(events: list[Event]) -> None:
    require(
        any(event.type == EventType.SESSION_COMPLETED for event in events),
        "session did not complete",
    )


def _require_model_usage(events: list[Event]) -> None:
    model_completed = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    require(bool(model_completed), "missing model.completed event")
    require(
        any(
            event.payload.get("usage_metrics", {}).get("total_tokens", 0) > 0
            for event in model_completed
        ),
        "model.completed events did not include token usage",
    )


def _require_finish_reason(events: list[Event], reason: str) -> None:
    reasons = [
        event.payload.get("completion", {}).get("finish_reason")
        for event in events
        if event.type == EventType.MODEL_COMPLETED
    ]
    require(reason in reasons, f"missing normalized finish_reason {reason!r}: {reasons!r}")


def _require_openrouter_route_evidence(events: list[Event], *, requested_model: str) -> None:
    completed = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    require(
        any(
            event.payload.get("requested_model") == requested_model
            and isinstance(event.payload.get("model"), str)
            and bool(event.payload["model"])
            and isinstance(event.payload.get("upstream_provider"), str)
            and bool(event.payload["upstream_provider"])
            and isinstance(event.payload.get("openrouter_metadata"), dict)
            for event in completed
        ),
        "OpenRouter completion did not retain effective model and bounded route evidence",
    )


def _require_tool_completed(events: list[Event], tool_name: str) -> None:
    require(
        any(
            event.type == EventType.TOOL_CALL_COMPLETED and event.tool_name == tool_name
            for event in events
        ),
        f"tool {tool_name!r} did not complete",
    )


if __name__ == "__main__":
    asyncio.run(main())
