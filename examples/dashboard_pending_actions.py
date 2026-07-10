"""Run a local dashboard seeded with session inspection and pending-action examples.

Usage:
    PYTHONPATH=src .venv/bin/python examples/dashboard_pending_actions.py
    # Open http://127.0.0.1:8001/cayu/
"""

from __future__ import annotations

import asyncio
import os
import shutil
import struct
import zlib
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

import uvicorn

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    ArtifactScope,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeStatus,
    LocalArtifactStore,
    Message,
    ModelPricing,
    PricingCatalog,
    RunRequest,
    SQLiteKnowledgeStore,
    SQLiteSessionStore,
    SQLiteTaskStore,
    StaticToolPolicy,
    TaskCreate,
    TaskQuery,
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


def _demo_png_bytes() -> bytes:
    """Build a simple visible PNG fixture without image dependencies."""
    width = 640
    height = 360

    def color_at(x: int, y: int) -> tuple[int, int, int]:
        if 56 <= y <= 136 and 44 <= x <= 596:
            if x < 204:
                return (224, 242, 254)
            if x < 400:
                return (204, 251, 241)
            return (255, 237, 213)
        if 216 <= y <= 304 and 144 <= x <= 496:
            if x < 320:
                return (207, 250, 254)
            return (237, 233, 254)
        if abs(y - 180) <= 2 or abs(x - 320) <= 2:
            return (51, 65, 85)
        return (248, 250, 252)

    raw_rows = bytearray()
    for y in range(height):
        raw_rows.append(0)
        for x in range(width):
            raw_rows.extend(color_at(x, y))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw_rows), level=9))
        + chunk(b"IEND", b"")
    )


def _demo_pdf_bytes() -> bytes:
    """Build a tiny valid PDF without adding a test dependency."""
    stream = (
        b"BT\n/F1 20 Tf\n72 720 Td\n(Cayu dashboard demo PDF artifact) Tj\n"
        b"0 -32 Td\n(Use Open raw or Download to inspect native bytes.) Tj\nET\n"
    )
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"5 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode()
        + stream
        + b"endstream\nendobj\n",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for item in objects:
        offsets.append(len(output))
        output.extend(item)
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(output)


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

        if "manual recovery" in prompt:
            yield ModelStreamEvent.tool_call(
                id="call_external_side_effect",
                name="external_side_effect",
                arguments={
                    "operation": "refund_invoice",
                    "invoice_id": "inv_demo_recovery",
                    "amount": 1280,
                },
            )
            yield ModelStreamEvent.completed(
                _completed_payload("tool_calls", input_tokens=175, output_tokens=15)
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


class ExternalSideEffectTool(Tool):
    spec = ToolSpec(
        name="external_side_effect",
        description="Pretend to perform an external side effect for manual recovery demos.",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "invoice_id": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["operation", "invoice_id", "amount"],
        },
        effect=ToolEffect.EXTERNAL,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=f"External operation {args['operation']} accepted for {args['invoice_id']}.",
            structured={"operation": args["operation"], "invoice_id": args["invoice_id"]},
        )


class ManualRecoveryDemoSessionStore(SQLiteSessionStore):
    """Drop one terminal event to leave a real pending tool-round recovery case."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._dropped_terminal_once = False

    async def append_events(self, session_id: str, events: list) -> None:
        if not self._dropped_terminal_once and any(
            event.type == EventType.TOOL_CALL_COMPLETED
            and event.tool_name == "external_side_effect"
            for event in events
        ):
            self._dropped_terminal_once = True
            raise RuntimeError("terminal tool event unavailable for manual recovery demo")
        await super().append_events(session_id, events)


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
        await _drain(
            app,
            RunRequest(
                agent_name="demo-manual-recovery",
                session_id="sess_dashboard_manual_recovery",
                messages=[Message.text("user", "Create a manual recovery dashboard session.")],
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


async def seed_tasks(app: CayuApp) -> None:
    if app.task_store is None:
        return
    await app.create_task(
        TaskCreate(
            task_id="queue_review_invoice",
            type="review",
            title="Review invoice exception",
            assigned_agent_name="demo-completed",
            input={"invoice_id": "inv_demo_42", "amount": 1280},
        )
    )
    await app.create_task(
        TaskCreate(
            task_id="queue_billing_export",
            type="sync",
            title="Wait for billing export",
            assigned_agent_name="demo-failure",
            input={"system": "billing"},
        )
    )
    await app.block_task(
        "queue_billing_export",
        reason="Waiting on upstream billing export",
        payload={"dependency": "billing_export_2026_07_08"},
    )
    await app.create_task(
        TaskCreate(
            task_id="queue_missing_target",
            type="deploy",
            title="Resolve deployment target",
            assigned_agent_name="demo-approval",
            input={"service": "payments-api"},
        )
    )
    await app.mark_task_needs_attention(
        "queue_missing_target",
        reason="Operator must choose staging or production target",
        payload={"options": ["staging", "production"]},
    )
    await app.create_task(
        TaskCreate(
            task_id="queue_paused_audit",
            type="audit",
            title="Paused nightly audit",
            assigned_agent_name="demo-user-input",
            input={"scope": "workspace"},
        )
    )
    await app.pause_task("queue_paused_audit", reason="Paused for dashboard demo")
    await app.create_task(
        TaskCreate(
            task_id="queue_claimed_worker",
            type="worker_claimed",
            title="Claimed by worker",
            assigned_agent_name="demo-completed",
        )
    )
    await app.task_store.claim_task(
        "worker-demo-1",
        TaskQuery(type="worker_claimed"),
        lease_seconds=300,
    )
    await app.create_task(
        TaskCreate(
            task_id="queue_running_session",
            type="run",
            title="Attached approval session",
            assigned_agent_name="demo-approval",
        )
    )
    await app.task_store.start_task(
        "queue_running_session",
        session_id="sess_dashboard_awaiting_approval",
    )
    await app.create_task(
        TaskCreate(
            task_id="queue_completed_cleanup",
            type="cleanup",
            title="Completed cleanup",
            assigned_agent_name="demo-completed",
        )
    )
    await app.task_store.complete_task("queue_completed_cleanup", {"status": "ok"})
    await app.create_task(
        TaskCreate(
            task_id="queue_failed_check",
            type="health_check",
            title="Failed health check",
            assigned_agent_name="demo-failure",
        )
    )
    await app.task_store.fail_task("queue_failed_check", {"error": "dependency timeout"})
    print("Seeded task queue examples: 8", flush=True)


async def seed_artifacts(app: CayuApp) -> None:
    environment = app.get_environment("demo-local").environment
    artifact_store = environment.artifact_store
    assert isinstance(artifact_store, LocalArtifactStore)
    await artifact_store.put_bytes(
        b"checkout-api deploy log\nstatus=ok\nrelease=2026.07.09\n",
        filename="checkout-deploy.log",
        content_type="text/plain",
        scope=ArtifactScope.SESSION,
        session_id="sess_dashboard_completed",
        agent_name="demo-completed",
        environment_name="demo-local",
        metadata={"demo": True, "source": "deploy_service"},
    )
    await artifact_store.put_bytes(
        b'{"invoice_id":"inv_demo_42","status":"needs_review","amount":1280}\n',
        filename="invoice-review.json",
        content_type="application/json",
        scope=ArtifactScope.SESSION,
        session_id="sess_dashboard_awaiting_approval",
        agent_name="demo-approval",
        environment_name="demo-local",
        metadata={"demo": True, "source": "approval_queue"},
    )
    await artifact_store.put_bytes(
        b"workspace=cayu-demo\nrunner=local\nartifact_store=demo-artifacts\n",
        filename="environment-summary.txt",
        content_type="text/plain",
        scope=ArtifactScope.ENVIRONMENT,
        environment_name="demo-local",
        metadata={"demo": True, "purpose": "control-plane inventory"},
    )
    await artifact_store.put_bytes(
        _demo_png_bytes(),
        filename="dashboard-session-map.png",
        content_type="image/png",
        scope=ArtifactScope.SESSION,
        session_id="sess_dashboard_completed",
        agent_name="demo-completed",
        environment_name="demo-local",
        metadata={"demo": True, "source": "visual_fixture"},
    )
    await artifact_store.put_bytes(
        _demo_pdf_bytes(),
        filename="dashboard-run-report.pdf",
        content_type="application/pdf",
        scope=ArtifactScope.SESSION,
        session_id="sess_dashboard_completed",
        agent_name="demo-completed",
        environment_name="demo-local",
        metadata={"demo": True, "source": "pdf_fixture"},
    )
    print("Seeded artifact examples: 5", flush=True)


def build_app() -> CayuApp:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(exist_ok=True)
    for name in ("sessions.db", "tasks.db", "knowledge.db"):
        for path in DB_DIR.glob(f"{name}*"):
            path.unlink(missing_ok=True)
    shutil.rmtree(DB_DIR / "artifacts", ignore_errors=True)

    knowledge_store = SQLiteKnowledgeStore(DB_DIR / "knowledge.db")
    app = CayuApp(
        session_store=ManualRecoveryDemoSessionStore(DB_DIR / "sessions.db"),
        task_store=SQLiteTaskStore(DB_DIR / "tasks.db"),
        knowledge_store=knowledge_store,
        enable_logging=False,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(
                name="demo-local",
                metadata={"tier": "demo", "region": "local"},
            ),
            artifact_store=LocalArtifactStore(DB_DIR / "artifacts", store_id="demo-artifacts"),
            knowledge_store=knowledge_store,
            workspace_instructions="Use the dashboard demo workspace for local control-plane checks.",
        ),
        default=True,
    )
    app.register_provider(DashboardDemoProvider(), default=True)

    shared_tools = [
        EchoTool(),
        DeployServiceTool(),
        FailingHealthCheckTool(),
        ExternalSideEffectTool(),
        UserInputTool(),
    ]
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
    app.register_agent(
        AgentSpec(name="demo-manual-recovery", model="fake-model"),
        tools=shared_tools,
    )
    return app


def main() -> None:
    app = build_app()
    asyncio.run(seed_sessions(app))
    asyncio.run(seed_tasks(app))
    asyncio.run(seed_artifacts(app))
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
