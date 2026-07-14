from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
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
    ReadFileTool,
    RunRequest,
    StructuredOutputSpec,
    WriteFileTool,
)

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
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY to run this live Chat Completions contract check.")
        return

    model = os.environ.get("CAYU_CHAT_COMPLETIONS_CONTRACT_MODEL", "gemini-3.1-flash-lite")
    with tempfile.TemporaryDirectory(prefix="cayu-chat-completions-contract-") as directory:
        root = Path(directory)
        await _run_tool_contract(model, root)
        await _run_structured_output_contract(model, root)

    print("completed")


async def _run_tool_contract(model: str, root: Path) -> None:
    session_id = f"chat_contract_tools_{uuid4().hex}"
    workspace = LocalWorkspace(root, workspace_id="chat-contract-tools")
    app = _app(workspace=workspace, runner=LocalRunner(root, inherit_env=False))
    app.register_agent(
        AgentSpec(
            name="assistant",
            model=model,
            system_prompt=(
                "You are verifying the Chat Completions adapter. Use tools exactly "
                "when the user requests file work. Do not claim a tool ran unless it did."
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
    print("tool_contract verified")


async def _run_structured_output_contract(model: str, root: Path) -> None:
    session_id = f"chat_contract_structured_{uuid4().hex}"
    app = _app(
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


def _app(*, workspace: LocalWorkspace, runner: LocalRunner) -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ChatCompletionsProvider(
            name="google",
            api_key_env="GEMINI_API_KEY",
            base_url=GEMINI_BASE_URL,
            document_encoding="image_url",
        ),
        default=True,
    )
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
