"""Run a local dashboard seeded with session inspection and pending-action examples.

Usage:
    PYTHONPATH=src .venv/bin/python examples/dashboard_pending_actions.py
    # Open http://127.0.0.1:8001/cayu/
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

import uvicorn

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeStatus,
    Message,
    ModelPricing,
    PricingCatalog,
    RunRequest,
    SQLiteKnowledgeStore,
    SQLiteSessionStore,
    SQLiteTaskStore,
    StaticToolPolicy,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.server import create_server
from cayu.tools import UserInputTool

WORKSPACE = Path(__file__).parent / ".examples-workspaces" / "dashboard-pending-actions"
DB_DIR = WORKSPACE / ".cayu"

DEMO_PRICING = PricingCatalog(
    prices=(
        ModelPricing(
            provider_name="fake",
            model="fake-model",
            input_per_million=Decimal("1.00"),
            output_per_million=Decimal("3.00"),
        ),
    )
)


class DashboardDemoProvider(ModelProvider):
    """Deterministic provider that creates dashboard-friendly session states."""

    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        prompt = _request_text(request).lower()
        if _has_tool_result(request):
            yield ModelStreamEvent.text_delta(_final_text(prompt))
            yield ModelStreamEvent.completed(
                _completed_payload("stop", input_tokens=220, output_tokens=48)
            )
            return

        if "provider failure" in prompt:
            raise RuntimeError("Provider stream failed while preparing the model response.")

        if "blocked" in prompt:
            yield ModelStreamEvent.tool_call(
                id="call_blocked_deploy",
                name="deploy_service",
                arguments={
                    "service": "checkout-api",
                    "environment": "production",
                    "risk": "demo policy blocks production deployment",
                },
            )
            yield ModelStreamEvent.completed(
                _completed_payload("tool_calls", input_tokens=180, output_tokens=18)
            )
            return

        if "approval" in prompt:
            yield ModelStreamEvent.tool_call(
                id="call_approval_deploy",
                name="deploy_service",
                arguments={
                    "service": "checkout-api",
                    "environment": "production",
                    "risk": "writes external production state",
                },
            )
            yield ModelStreamEvent.completed(
                _completed_payload("tool_calls", input_tokens=185, output_tokens=20)
            )
            return

        if "question" in prompt or "user input" in prompt:
            yield ModelStreamEvent.tool_call(
                id="call_choose_environment",
                name="ask_user",
                arguments={
                    "question": "Which environment should the deployment target?",
                    "options": ["staging", "production"],
                },
            )
            yield ModelStreamEvent.completed(
                _completed_payload("tool_calls", input_tokens=170, output_tokens=16)
            )
            return

        if "failure" in prompt:
            yield ModelStreamEvent.tool_call(
                id="call_fail_health_check",
                name="failing_health_check",
                arguments={"service": "billing-worker"},
            )
            yield ModelStreamEvent.completed(
                _completed_payload("tool_calls", input_tokens=165, output_tokens=14)
            )
            return

        yield ModelStreamEvent.tool_call(
            id="call_echo_summary",
            name="echo",
            arguments={"text": "Collected run telemetry for dashboard inspection."},
        )
        yield ModelStreamEvent.completed(
            _completed_payload("tool_calls", input_tokens=150, output_tokens=12)
        )


class EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo text for a successful dashboard demo run.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content=args["text"], structured={"agent": ctx.agent_name})


class DeployServiceTool(Tool):
    spec = ToolSpec(
        name="deploy_service",
        description="Pretend to deploy a service.",
        input_schema={
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "environment": {"type": "string"},
                "risk": {"type": "string"},
            },
            "required": ["service", "environment"],
        },
        effect=ToolEffect.EXTERNAL,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=f"Deployment approved for {args['service']} in {args['environment']}.",
            structured={"approved_by": ctx.metadata.get("actor", "dashboard-demo")},
        )


class FailingHealthCheckTool(Tool):
    spec = ToolSpec(
        name="failing_health_check",
        description="Raise a deterministic failure for dashboard debugging.",
        input_schema={
            "type": "object",
            "properties": {"service": {"type": "string"}},
            "required": ["service"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        raise RuntimeError(f"Health check failed for {args['service']}: timeout after 30s")


def _request_text(request: ModelRequest) -> str:
    chunks: list[str] = []
    for message in request.messages:
        for part in message.content:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def _has_tool_result(request: ModelRequest) -> bool:
    return any(message.role == "tool" for message in request.messages)


def _completed_payload(finish_reason: str, *, input_tokens: int, output_tokens: int) -> dict:
    return {
        "finish_reason": finish_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _final_text(prompt: str) -> str:
    if "approval" in prompt:
        return "Approved deployment completed. The audit event is recorded."
    if "question" in prompt or "user input" in prompt:
        return "Thanks. The selected environment was applied to the deployment plan."
    return "Dashboard demo session completed with a tool call and final model response."


async def _drain(app: CayuApp, request: RunRequest) -> str:
    events = []
    async for event in app.run(request):
        events.append(event)
    final_type = str(events[-1].type) if events else "no-events"
    return f"{request.session_id}: {final_type}{_seed_note(events)}"


def _seed_note(events) -> str:
    for event in events:
        event_type = str(event.type)
        if event_type == "tool.call.approval_requested":
            approval = event.payload.get("approval") if isinstance(event.payload, dict) else None
            if isinstance(approval, dict):
                return f" ({approval.get('tool_name')} approval pending)"
        if event_type == "session.awaiting_user_input":
            return " (user input pending)"
        if event_type == "tool.call.failed":
            return f" (failed tool: {event.tool_name})"
        if event_type == "tool.call.blocked":
            return f" (blocked tool: {event.tool_name})"
        if event_type == "session.failed":
            error_type = (
                event.payload.get("error_type") if isinstance(event.payload, dict) else None
            )
            return f" ({error_type or 'session failed'})"
    return ""


async def seed_sessions(app: CayuApp) -> None:
    seeded = [
        await _drain(
            app,
            RunRequest(
                agent_name="demo-completed",
                session_id="sess_dashboard_completed",
                messages=[Message.text("user", "Create a completed dashboard session.")],
            ),
        ),
        await _drain(
            app,
            RunRequest(
                agent_name="demo-approval",
                session_id="sess_dashboard_awaiting_approval",
                messages=[Message.text("user", "Create an approval pending dashboard session.")],
            ),
        ),
        await _drain(
            app,
            RunRequest(
                agent_name="demo-user-input",
                session_id="sess_dashboard_awaiting_user_input",
                messages=[Message.text("user", "Create a user input question dashboard session.")],
            ),
        ),
        await _drain(
            app,
            RunRequest(
                agent_name="demo-failure",
                session_id="sess_dashboard_failed_tool",
                messages=[Message.text("user", "Create a failure dashboard session.")],
            ),
        ),
        await _drain(
            app,
            RunRequest(
                agent_name="demo-blocked-tool",
                session_id="sess_dashboard_blocked_tool",
                messages=[Message.text("user", "Create a blocked tool dashboard session.")],
            ),
        ),
        await _drain(
            app,
            RunRequest(
                agent_name="demo-session-failed",
                session_id="sess_dashboard_session_failed",
                messages=[Message.text("user", "Create a provider failure dashboard session.")],
            ),
        ),
    ]
    print("Seeded dashboard sessions:", flush=True)
    for line in seeded:
        print(f"- {line}", flush=True)


async def seed_pending_knowledge(store: SQLiteKnowledgeStore) -> None:
    entries = [
        KnowledgeEntry(
            id="pending_remote_git_credentials",
            title="Remote sandbox Git credentials",
            text=(
                "Remote sandbox Git pushes should use a brokered credential proxy so raw "
                "GitHub tokens never enter sandbox environment variables, files, process "
                "arguments, or command output."
            ),
            namespace="project:cayu",
            labels={"project": "cayu", "area": "sandbox-git"},
            kind="procedure",
            status=KnowledgeStatus.PENDING,
            created_by_type=KnowledgeActorType.MODEL,
            created_by="demo-knowledge-agent",
            source_type="tool",
            source_id="remember_knowledge",
            aspects=["git", "credentials", "remote-sandbox"],
            impact_targets=["sandbox.git.push"],
            confidence=0.92,
        ),
        KnowledgeEntry(
            id="pending_failed_tool_triage",
            title="Failed tool triage",
            text=(
                "When a tool call fails, inspect the stored tool.call.failed event payload "
                "before retrying so policy-denied calls and external side effects are not "
                "mistaken for transient runtime errors."
            ),
            namespace="project:cayu",
            labels={"project": "cayu", "area": "operations"},
            kind="procedure",
            status=KnowledgeStatus.PENDING,
            created_by_type=KnowledgeActorType.MODEL,
            created_by="demo-knowledge-agent",
            source_type="tool",
            source_id="remember_knowledge",
            aspects=["tools", "debugging"],
            impact_targets=["tool.review"],
            confidence=0.84,
        ),
    ]
    for entry in entries:
        await store.put_entry_with_chunks(
            entry,
            [
                KnowledgeChunk(
                    id=f"{entry.id}:0",
                    entry_id=entry.id,
                    chunk_index=0,
                    text=entry.text,
                    metadata={"demo": True},
                )
            ],
        )
    print(f"Seeded pending knowledge entries: {len(entries)}", flush=True)


def build_app() -> CayuApp:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(exist_ok=True)
    for name in ("sessions.db", "tasks.db", "knowledge.db"):
        for path in DB_DIR.glob(f"{name}*"):
            path.unlink(missing_ok=True)

    knowledge_store = SQLiteKnowledgeStore(DB_DIR / "knowledge.db")
    app = CayuApp(
        session_store=SQLiteSessionStore(DB_DIR / "sessions.db"),
        task_store=SQLiteTaskStore(DB_DIR / "tasks.db"),
        knowledge_store=knowledge_store,
        enable_logging=False,
    )
    app.register_provider(DashboardDemoProvider(), default=True)

    shared_tools = [EchoTool(), DeployServiceTool(), FailingHealthCheckTool(), UserInputTool()]
    app.register_agent(AgentSpec(name="demo-completed", model="fake-model"), tools=shared_tools)
    app.register_agent(
        AgentSpec(name="demo-approval", model="fake-model"),
        tools=shared_tools,
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=["deploy_service"]),
    )
    app.register_agent(AgentSpec(name="demo-user-input", model="fake-model"), tools=shared_tools)
    app.register_agent(AgentSpec(name="demo-failure", model="fake-model"), tools=shared_tools)
    app.register_agent(
        AgentSpec(name="demo-blocked-tool", model="fake-model"),
        tools=shared_tools,
        tool_policy=StaticToolPolicy(deny=["deploy_service"]),
    )
    app.register_agent(
        AgentSpec(name="demo-session-failed", model="fake-model"),
        tools=shared_tools,
    )
    return app


def main() -> None:
    app = build_app()
    asyncio.run(seed_sessions(app))
    knowledge_store = app.knowledge_store
    assert isinstance(knowledge_store, SQLiteKnowledgeStore)
    asyncio.run(seed_pending_knowledge(knowledge_store))
    server = create_server(
        app,
        dev=True,
        dashboard_config={"pricingCatalog": DEMO_PRICING.model_dump(mode="json")},
    )
    host = os.environ.get("CAYU_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("CAYU_DASHBOARD_PORT", "8001"))
    print(f"Dashboard demo ready: http://{host}:{port}/cayu/", flush=True)
    uvicorn.run(server, host=host, port=port)


if __name__ == "__main__":
    main()
