"""Verified browser contract for the packaged Cayu dashboard."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import parse_qs, urlencode, urlsplit

import uvicorn
from playwright.async_api import (
    BrowserContext,
    Page,
    Request,
    async_playwright,
    expect,
)

from _live_checks import is_superseded_browser_read_abort, require, require_equal
from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    CorpusTarget,
    Environment,
    EnvironmentSpec,
    EvalPlan,
    EvaluationEvidencePolicySpec,
    Event,
    EventType,
    InMemoryTaskStore,
    LocalArtifactStore,
    LocalWorkspace,
    Message,
    MessageRole,
    ModelJudgeTarget,
    ModelPrice,
    PriceBook,
    RetryPolicy,
    RunLimits,
    RunRequest,
    SQLiteEvalStore,
    TaskCreate,
    TextPart,
    ThinkingPart,
    Tool,
    ToolCapabilityCeiling,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolResultPart,
    ToolSpec,
)
from cayu._server_contract_version import SERVER_CONTRACT_VERSION
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    bedrock_billing_identity,
    completed_bedrock_billing_identity,
)
from cayu.runtime import (
    EventQuery,
    InMemorySessionStore,
    SessionIdentity,
    SessionStatus,
)
from cayu.runtime import (
    _execution_profile_admission as execution_profile_admission,
)
from cayu.server import (
    BasicAuth,
    DashboardConfig,
    EvalsConfig,
    EvaluationPromotionConfig,
    ServerConfig,
    create_server,
)

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send
    from starlette.types import Message as AsgiMessage

SESSION_ID = "dashboard-contract-session"
PROMOTION_SESSION_ID = "dashboard-contract-eval-promotion"
APPROVAL_SESSION_ID = "dashboard-contract-approval"
INTERRUPT_SESSION_ID = "dashboard-contract-interrupt"
INTERRUPT_FAILURE_SESSION_ID = "dashboard-contract-interrupt-failure"
RESUME_INTERRUPT_SESSION_ID = "dashboard-contract-resume-interrupt"
REOBSERVE_SESSION_ID = "dashboard-contract-reobserve"
BEDROCK_SESSION_ID = "dashboard-contract-bedrock-billing"
FILTERED_SESSION_ID = "dashboard-contract-filtered-session"
PAGINATED_SESSION_PREFIX = "dashboard-contract-page"
PAGINATED_SESSION_COUNT = 101
DISCOVERY_ENVIRONMENT = "dashboard-contract-production"
DISCOVERY_BUDGET_ID = "dashboard-contract-budget"
SLOW_SESSION_QUERY = "dashboard-contract-slow-query"
SLOW_USAGE_AGENT = "dashboard-contract-slow-usage-agent"
DASHBOARD_MODEL_STEP_ID = f"mstep_{'1' * 32}"
DASHBOARD_MODEL_ATTEMPT_ID = f"matt_{'2' * 32}"
AGENT_NAME = "dashboard-contract-agent"
AUTH_USERNAME = "dashboard-contract-operator"
AUTH_PASSWORD = "dashboard-contract-password"
PROVIDER_NAME = "contract-provider"
MODEL_NAME = "contract-model"
PAYLOAD_MARKER = "dashboard-contract-usage"
WORKFLOW_ROOT_SESSION_ID = "dashboard-workflow-root"
WORKFLOW_PARENT_SESSION_ID = "dashboard-workflow-parent"
WORKFLOW_FOCUS_SESSION_ID = "dashboard-workflow-focus"
WORKFLOW_ACTIVE_SESSION_ID = "dashboard-workflow-child-active"
WORKFLOW_FAILED_SESSION_ID = "dashboard-workflow-child-failed"
WORKFLOW_INTERRUPTED_SESSION_ID = "dashboard-workflow-child-interrupted"
WORKFLOW_CHILD_SESSION_PREFIX = "dashboard-workflow-child"
WORKFLOW_CHILD_SESSION_COUNT = 27
WORKFLOW_BUDGET_ID = "dashboard-workflow-budget"
WORKFLOW_ENVIRONMENT = "dashboard-workflow-production"
WORKFLOW_PROVIDER_NAME = "dashboard-workflow-provider"
WORKFLOW_MODEL_NAME = "dashboard-workflow-model"
WORKFLOW_SECONDARY_MODEL_NAME = "dashboard-workflow-secondary-model"
WORKFLOW_PARENT_TASK_ID = "dashboard-workflow-task-parent"
WORKFLOW_BLOCKED_TASK_ID = "dashboard-workflow-task-blocked"
WORKFLOW_LINKED_TASK_PREFIX = "dashboard-workflow-linked-task"
WORKFLOW_LINKED_TASK_COUNT = 26
WORKFLOW_CHILD_TASK_PREFIX = "dashboard-workflow-child-task"
WORKFLOW_CHILD_TASK_COUNT = 26
EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
_LIVE_CREDENTIAL_ENV = (
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
)

ObserverAbort = tuple[str, str | None, str]


class DashboardContractProvider(ModelProvider):
    name = "contract-provider"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.judge_requests: list[ModelRequest] = []
        self.direct_requests: list[ModelRequest] = []
        self.direct_completions = 0
        self.recovery_requests: list[ModelRequest] = []
        self.replay_markers: list[str] = []
        self.promotion_outputs = 0
        self._approval_seeded = False
        self._scenario_approval_seeded = False
        self._direct_started = asyncio.Condition()
        self._direct_releases: asyncio.Queue[None] = asyncio.Queue()
        self._replay_releases: asyncio.Queue[str] = asyncio.Queue()
        self.block_next_promotion_run = False
        self.blocked_promotion_run_started = asyncio.Event()

    async def wait_for_direct_requests(self, count: int) -> None:
        async with self._direct_started:
            await asyncio.wait_for(
                self._direct_started.wait_for(lambda: len(self.direct_requests) >= count),
                timeout=10,
            )

    def release_direct(self) -> None:
        self._direct_releases.put_nowait(None)

    def release_after_replay(self, marker: str) -> None:
        self._replay_releases.put_nowait(marker)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        request_text = "\n".join(
            part.text
            for message in request.messages
            for part in message.content
            if isinstance(part, TextPart)
        )
        if "criterion ids must appear exactly once" in request_text.lower():
            self.judge_requests.append(request)
            yield ModelStreamEvent.text_delta(
                json.dumps(
                    {
                        "criteria": [
                            {
                                "criterion_id": "correctness",
                                "score": 1,
                                "explanation": "The known output satisfies the fixed task.",
                            }
                        ]
                    },
                    separators=(",", ":"),
                )
            )
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 120,
                        "output_tokens": 24,
                        "total_tokens": 144,
                    },
                }
            )
            return
        if "promote this captured dashboard run" in request_text.lower():
            if self.block_next_promotion_run:
                self.block_next_promotion_run = False
                self.blocked_promotion_run_started.set()
                await asyncio.Event().wait()
            if (
                "verify the queued production follow-up" in request_text.lower()
                and not self._scenario_approval_seeded
            ):
                self._scenario_approval_seeded = True
                yield ModelStreamEvent.tool_call(
                    id="dashboard-scenario-approval-call",
                    name="dashboard_contract_tool",
                    arguments={"operation": "verify-scenario"},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            self.promotion_outputs += 1
            yield ModelStreamEvent.text_delta(
                f"dashboard eval promotion output {self.promotion_outputs}"
            )
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        if "exercise one fresh control plane-authored evaluation" in request_text.lower():
            if not any(
                isinstance(part, ToolResultPart)
                for message in request.messages
                for part in message.content
            ):
                yield ModelStreamEvent.tool_call(
                    id="dashboard-eval-search-call",
                    name="dashboard_eval_search",
                    arguments={"query": "cayu", "limit": 5},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            yield ModelStreamEvent.text_delta("dashboard authored evaluation output")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        if "recover after" not in request_text:
            if "seed the dashboard approval" in request_text.lower() and not self._approval_seeded:
                self._approval_seeded = True
                yield ModelStreamEvent.tool_call(
                    id="dashboard-approval-call",
                    name="dashboard_contract_tool",
                    arguments={"operation": "verify"},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            async with self._direct_started:
                self.direct_requests.append(request)
                self._direct_started.notify_all()
            await asyncio.wait_for(self._direct_releases.get(), timeout=30)
            self.direct_completions += 1
            yield ModelStreamEvent.text_delta("dashboard session mutation completed")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return

        self.recovery_requests.append(request)
        marker = await asyncio.wait_for(self._replay_releases.get(), timeout=10)
        self.replay_markers.append(marker)
        yield ModelStreamEvent.text_delta("dashboard mutation recovery completed")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class DashboardContractTool(Tool):
    spec = ToolSpec(
        name="dashboard_contract_tool",
        description="Exercise dashboard approval resolution without external side effects.",
        input_schema={
            "type": "object",
            "properties": {"operation": {"type": "string"}},
            "required": ["operation"],
        },
        effect=ToolEffect.NONE,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=f"Dashboard contract operation {args['operation']} completed.",
            structured={"agent": ctx.agent_name},
        )


class DashboardEvalSearchTool(Tool):
    spec = ToolSpec(
        name="dashboard_eval_search",
        description="Return deterministic public facts for the installed Evals contract.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query", "limit"],
        },
        parallel_safe=False,
        effect=ToolEffect.IDEMPOTENT,
        workspace_mutation=True,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        if ctx.workspace is None or ctx.artifact_store is None:
            return ToolResult(
                content="The dashboard eval fixture requires workspace and artifact storage.",
                is_error=True,
            )
        output = b'{"source":"dashboard_eval_search","status":"ready"}\n'
        await ctx.workspace.write_bytes("dashboard-eval-output.json", output)
        await ctx.artifact_store.put_bytes(
            output,
            artifact_id=("art_" + hashlib.sha256(ctx.session_id.encode("utf-8")).hexdigest()[:32]),
            filename="dashboard-eval-report.json",
            content_type="application/json",
            session_id=ctx.session_id,
            agent_name=ctx.agent_name,
            environment_name=ctx.environment_name,
        )
        return ToolResult(
            content=f"Found public results for {args['query']}.",
            structured={"status": "ok", "count": min(args["limit"], 2)},
        )


class MutationDisconnectFaults:
    """Close two initial run observers without cancelling detached execution."""

    def __init__(self, app: ASGIApp, provider: DashboardContractProvider) -> None:
        self.app = app
        self.provider = provider
        self.initial_run_requests = 0
        self.initial_mutation_ids: list[str] = []
        self.initial_mutation_requests: dict[str, int] = {}
        self.replay_requests: list[tuple[str, str]] = []
        self.slow_summary_started = asyncio.Event()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        headers = dict(scope.get("headers", []))
        is_summary_post = (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/api/sessions/summary"
        )
        query = parse_qs(bytes(scope.get("query_string", b"")).decode("utf-8", errors="replace"))
        if is_summary_post and query.get("q") == [SLOW_SESSION_QUERY]:
            self.slow_summary_started.set()
            await asyncio.sleep(1)
        is_run_post = (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/api/run"
        )
        mutation_id = headers.get(b"cayu-mutation-id", b"").decode("ascii", errors="replace")
        replay_header = headers.get(b"last-event-id")
        if (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and replay_header is None
            and mutation_id
        ):
            path = str(scope.get("path", ""))
            self.initial_mutation_requests[path] = self.initial_mutation_requests.get(path, 0) + 1
        if is_run_post and replay_header is not None:
            replay_marker = replay_header.decode("ascii", errors="replace")
            self.replay_requests.append((replay_marker, mutation_id))
            self.provider.release_after_replay(replay_marker)
            await self.app(scope, receive, send)
            return

        inject = is_run_post and replay_header is None and self.initial_run_requests < 2
        if not inject:
            await self.app(scope, receive, send)
            return

        fault_index = self.initial_run_requests
        self.initial_run_requests += 1
        self.initial_mutation_ids.append(mutation_id)
        response_finished = False

        async def fault_send(message: AsgiMessage) -> None:
            nonlocal response_finished
            if response_finished:
                return
            if message["type"] != "http.response.body":
                await send(message)
                return

            body = message.get("body", b"")
            more_body = message.get("more_body", False)
            if fault_index == 0:
                # Preserve HTTP acceptance but close before the first SSE frame.
                if body or not more_body:
                    response_finished = True
                    # A comment flushes the response headers without creating an
                    # SSE event. An empty final ASGI body can be coalesced with
                    # the headers by the HTTP server, which would model a request
                    # that never opened rather than an accepted observer close.
                    await send(
                        {
                            "type": "http.response.body",
                            "body": b": injected observer close\n\n",
                            "more_body": False,
                        }
                    )
                return

            # Deliver one durable SSE response chunk, then close the observer.
            # The client must reconnect from the exact durable marker established
            # by that frame or its bounded REST reconciliation.
            if body:
                response_finished = True
                await send({"type": "http.response.body", "body": body, "more_body": False})
            elif not more_body:
                response_finished = True
                await send(message)

        await self.app(scope, receive, fault_send)


def _profiled_session_identity(
    app: CayuApp,
    *,
    provider_name: str,
    model: str,
    max_steps: int,
) -> SessionIdentity:
    """Build the exact profile used by low-level resumable dashboard fixtures."""

    runtime_version = version("cayu")
    registered_agent = app._agents[AGENT_NAME]
    engine = app._session_engine
    return SessionIdentity(
        provider_name=provider_name,
        model=model,
        runtime_name="cayu",
        runtime_version=runtime_version,
        execution_profile=execution_profile_admission.resolve_execution_profile_identity(
            registered_agent=registered_agent,
            runtime_name="cayu",
            runtime_version=runtime_version,
            provider_name=provider_name,
            model=model,
            durable_system_prompt=registered_agent.spec.system_prompt,
            redactor=app._secret_redactor,
            process_identity=app._execution_profile_process_identity,
            runtime_hooks=engine._runtime_hooks,
            loop_policies=engine._loop_policies,
            loop_policy_identities=engine._loop_policy_execution_profile_identities,
            registered_provider=app._providers.get(provider_name),
            finalization=execution_profile_admission.model_finalization_material(
                max_steps=max_steps,
                limits=RunLimits(),
                retry_policy=RetryPolicy(),
            ),
        ),
    )


async def main() -> None:
    require_equal(
        DashboardContractTool.spec.effect,
        ToolEffect.NONE,
        "dashboard contract tool must remain mutation-free",
    )
    app, provider, store, task_store, runtime_directory = await _seed_app()
    price_book = _dashboard_price_book()
    judge_app = CayuApp(enable_logging=False)
    judge_app.register_provider(provider, default=True)
    judge_app.register_agent(AgentSpec(name="dashboard-contract-judge", model=MODEL_NAME))
    judge = ModelJudgeTarget(
        key="dashboard-quality-judge",
        label="Dashboard quality judge",
        app=judge_app,
        agent_name="dashboard-contract-judge",
        max_input_tokens=3_072,
        max_output_tokens=1_024,
        max_total_tokens=4_096,
        max_estimated_cost="0.01",
        price_book=price_book,
        allow_same_model=True,
    )
    evals_directory = tempfile.TemporaryDirectory(prefix="cayu-dashboard-evals-")
    eval_store = SQLiteEvalStore(Path(evals_directory.name) / "evals.sqlite")
    provider.block_next_promotion_run = True
    server_app = MutationDisconnectFaults(
        create_server(
            app,
            config=ServerConfig.protected(
                BasicAuth(username=AUTH_USERNAME, password=AUTH_PASSWORD),
                dashboard=DashboardConfig(
                    runtime_config={"priceBook": price_book.model_dump(mode="json")}
                ),
                evaluation_promotion=EvaluationPromotionConfig(
                    target_key="dashboard.regressions",
                    source_agent_name=AGENT_NAME,
                    application_release_id="dashboard-browser-contract",
                    evidence_policy=EvaluationEvidencePolicySpec.create(
                        include_tool_results=True,
                        include_artifact_text=True,
                    ),
                ),
                evals=EvalsConfig(
                    target=CorpusTarget(
                        key="dashboard.regressions",
                        app=app,
                        request_base=RunRequest(agent_name=AGENT_NAME, messages=[]),
                        application_release_id="dashboard-browser-contract",
                        price_book=price_book,
                        model_judges=(judge,),
                        evidence_policy=EvaluationEvidencePolicySpec.create(
                            include_tool_results=True,
                            include_artifact_text=True,
                        ),
                    ),
                    store=eval_store,
                    poll_interval_seconds=0.05,
                ),
            ),
        ),
        provider,
    )
    listener = _loopback_listener()
    port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(
        uvicorn.Config(
            server_app,
            log_level="warning",
            lifespan="on",
        )
    )
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await _wait_for_server(server, server_task)
        evidence = await _run_browser_contract(
            base_url,
            provider,
            server_app,
            store,
            task_store,
        )
        require_equal(
            server_app.initial_run_requests,
            2,
            "browser recovery must inject both initial observer disconnects",
        )
        require_equal(
            len(provider.recovery_requests),
            2,
            "browser recovery must execute each dashboard run exactly once",
        )
        require_equal(
            len(provider.direct_requests),
            3,
            "resume, approval, and interrupted-resume browser flows must each execute once",
        )
        require_equal(
            provider.direct_completions,
            2,
            "only the explicitly released resume and approval provider calls may complete",
        )
        require_equal(
            server_app.initial_mutation_requests,
            {
                "/api/run": 2,
                "/api/resume": 2,
                "/api/tool-approvals/resolve": 1,
                f"/api/sessions/{INTERRUPT_SESSION_ID}/interrupt": 1,
                f"/api/sessions/{RESUME_INTERRUPT_SESSION_ID}/interrupt": 1,
            },
            "browser session mutations must each be submitted exactly once",
        )
        require_equal(
            len(server_app.replay_requests),
            2,
            "each injected observer close must produce one explicit replay request",
        )
        replay_markers = [marker for marker, _mutation_id in server_app.replay_requests]
        replay_mutation_ids = [mutation_id for _marker, mutation_id in server_app.replay_requests]
        require_equal(
            replay_mutation_ids,
            server_app.initial_mutation_ids,
            "replay requests must preserve the exact submitted mutation identities",
        )
        require_equal(
            provider.replay_markers,
            replay_markers,
            "provider execution must remain blocked until its browser replay is observed",
        )
        require_equal(
            len(set(replay_mutation_ids)),
            2,
            "each dashboard run must use a distinct mutation identity",
        )
        for marker in replay_markers:
            marker_parts = marker.split(":", maxsplit=1)
            require_equal(len(marker_parts), 2, "Last-Event-ID must contain a session and event id")
            session_id, event_id = marker_parts
            require(bool(session_id and event_id), "Last-Event-ID must name a durable event")
            require(
                event_id.startswith("cayu_event_") and event_id[11:].isdigit(),
                "Last-Event-ID must use the server's public event identity",
            )
            event_sequence = int(event_id[11:])
            records = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    after_sequence=event_sequence - 1,
                    limit=1,
                )
            )
            require_equal(
                len(records),
                1,
                "Last-Event-ID must be the exact identity of an existing durable event",
            )
            require_equal(
                records[0].sequence,
                event_sequence,
                "Last-Event-ID must identify the exact durable event sequence",
            )
        evidence["mutation_provider_requests"] = len(provider.requests)
        evidence["injected_initial_disconnects"] = server_app.initial_run_requests
        evidence["mutation_replay_requests"] = len(server_app.replay_requests)
        print(f"{EVIDENCE_PREFIX}{json.dumps(evidence, sort_keys=True)}")
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=10)
        except TimeoutError:
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)
        await eval_store.close()
        evals_directory.cleanup()
        runtime_directory.cleanup()


async def _seed_app() -> tuple[
    CayuApp,
    DashboardContractProvider,
    InMemorySessionStore,
    InMemoryTaskStore,
    tempfile.TemporaryDirectory[str],
]:
    store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(session_store=store, task_store=task_store, enable_logging=False)
    runtime_directory = tempfile.TemporaryDirectory(prefix="cayu-dashboard-runtime-")
    runtime_root = Path(runtime_directory.name)
    workspace_root = runtime_root / "workspace"
    workspace_root.mkdir()
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=LocalWorkspace(workspace_root, workspace_id="dashboard-workspace"),
            artifact_store=LocalArtifactStore(
                runtime_root / "artifacts",
                store_id="dashboard-artifacts",
            ),
        ),
        default=True,
    )
    provider = DashboardContractProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model=MODEL_NAME))
    app.register_agent(
        AgentSpec(name=AGENT_NAME, model=MODEL_NAME),
        tools=[DashboardContractTool(), DashboardEvalSearchTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=["dashboard_contract_tool"]),
    )
    contract_tool_capability_ceiling = ToolCapabilityCeiling(
        tool_names=(DashboardContractTool.spec.name, DashboardEvalSearchTool.spec.name)
    )
    contract_session_identity = _profiled_session_identity(
        app,
        provider_name=PROVIDER_NAME,
        model=MODEL_NAME,
        max_steps=20,
    )
    await store.create(
        RunRequest(
            agent_name=AGENT_NAME,
            session_id=SESSION_ID,
            labels={"stage": "initial"},
            metadata={
                "customer": {"id": "dashboard-customer"},
                "cayu:dashboard_fixture": {"marker": "runtime-dashboard-marker"},
            },
            messages=[Message.text("user", "Show the dashboard contract session.")],
            max_steps=20,
            tool_capability_ceiling=contract_tool_capability_ceiling,
        ),
        identity=contract_session_identity,
    )
    await store.append_events(
        SESSION_ID,
        [
            Event(
                id="dashboard-session-started",
                type=EventType.SESSION_STARTED,
                session_id=SESSION_ID,
                agent_name=AGENT_NAME,
            ),
            Event(
                id="dashboard-model-started",
                type=EventType.MODEL_STARTED,
                session_id=SESSION_ID,
                agent_name=AGENT_NAME,
                interaction_id="dashboard-contract-interaction",
                payload={
                    "provider": PROVIDER_NAME,
                    "model": MODEL_NAME,
                    "step": 1,
                    "attempt": 1,
                    "max_attempts": 1,
                    "model_step_id": DASHBOARD_MODEL_STEP_ID,
                    "model_attempt_id": DASHBOARD_MODEL_ATTEMPT_ID,
                },
            ),
            Event(
                id="dashboard-model-completed",
                type=EventType.MODEL_COMPLETED,
                session_id=SESSION_ID,
                agent_name=AGENT_NAME,
                interaction_id="dashboard-contract-interaction",
                payload={
                    "contract_marker": PAYLOAD_MARKER,
                    "provider_name": PROVIDER_NAME,
                    "requested_model": MODEL_NAME,
                    "step": 1,
                    "attempt": 1,
                    "max_attempts": 1,
                    "model_step_id": DASHBOARD_MODEL_STEP_ID,
                    "model_attempt_id": DASHBOARD_MODEL_ATTEMPT_ID,
                    "usage_metrics": {
                        "provider_name": PROVIDER_NAME,
                        "requested_model": MODEL_NAME,
                        "model": MODEL_NAME,
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "total_tokens": 15,
                    },
                },
            ),
            Event(
                id="dashboard-tool-failed",
                type=EventType.TOOL_CALL_FAILED,
                session_id=SESSION_ID,
                agent_name=AGENT_NAME,
                tool_name="browser_contract_tool",
                payload={"error": "dashboard contract tool failure"},
            ),
            Event(
                id="dashboard-session-completed",
                type=EventType.SESSION_COMPLETED,
                session_id=SESSION_ID,
                agent_name=AGENT_NAME,
            ),
        ],
    )
    await store.append_transcript_messages(
        SESSION_ID,
        [
            Message.text("user", "dashboard transcript user marker"),
            Message(
                role=MessageRole.ASSISTANT,
                content=(ThinkingPart(text="dashboard transcript thinking marker"),),
            ),
            Message.text("assistant", "dashboard transcript assistant marker"),
        ],
    )
    await store.update_status(SESSION_ID, SessionStatus.COMPLETED)

    async for _ in app.run(
        RunRequest(
            agent_name=AGENT_NAME,
            session_id=PROMOTION_SESSION_ID,
            messages=[Message.text("user", "Promote this captured dashboard run.")],
        )
    ):
        pass

    bedrock_identity = completed_bedrock_billing_identity(
        bedrock_billing_identity(
            invoked_model="global.anthropic.claude-sonnet-4-6",
            source_region="us-east-1",
            resource_type="inference_profile",
            profile_scope="global",
            requested_service_tier="default",
        ),
        effective_service_tier="default",
    )
    await store.create(
        RunRequest(
            agent_name=AGENT_NAME,
            session_id=BEDROCK_SESSION_ID,
            labels={"billing": "bedrock"},
            messages=[Message.text("user", "Verify the Bedrock billing breakdown.")],
        ),
        identity=SessionIdentity(provider_name="bedrock", model=bedrock_identity.resource_id),
    )
    await store.append_events(
        BEDROCK_SESSION_ID,
        [
            Event(
                id="dashboard-bedrock-model-completed",
                type=EventType.MODEL_COMPLETED,
                session_id=BEDROCK_SESSION_ID,
                agent_name=AGENT_NAME,
                payload={
                    "usage_metrics": {
                        "provider_name": "bedrock",
                        "requested_model": bedrock_identity.resource_id,
                        "model": bedrock_identity.resource_id,
                        "billing_identity": bedrock_identity.model_dump(mode="json"),
                        "input_tokens": 1_000_000,
                        "output_tokens": 0,
                        "total_tokens": 1_000_000,
                    },
                    "billing_identity": bedrock_identity.model_dump(mode="json"),
                },
            ),
            Event(
                id="dashboard-bedrock-session-completed",
                type=EventType.SESSION_COMPLETED,
                session_id=BEDROCK_SESSION_ID,
                agent_name=AGENT_NAME,
            ),
        ],
    )
    await store.update_status(BEDROCK_SESSION_ID, SessionStatus.COMPLETED)

    async def seed_completed_session(
        session_id: str,
        prompt: str,
        *,
        parent_session_id: str | None = None,
        causal_budget_id: str | None = None,
        environment_name: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        await store.create(
            RunRequest(
                agent_name=AGENT_NAME,
                session_id=session_id,
                parent_session_id=parent_session_id,
                causal_budget_id=causal_budget_id,
                environment_name=environment_name,
                labels=labels or {},
                messages=[Message.text("user", prompt)],
                max_steps=20,
                tool_capability_ceiling=contract_tool_capability_ceiling,
            ),
            identity=contract_session_identity,
        )
        await store.append_transcript_messages(session_id, [Message.text("user", prompt)])
        await store.append_events(
            session_id,
            [
                Event(
                    id=f"{session_id}-completed",
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                    agent_name=AGENT_NAME,
                )
            ],
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)

    await seed_completed_session(
        RESUME_INTERRUPT_SESSION_ID,
        "Resume this session, then interrupt its active dashboard observation.",
    )
    await seed_completed_session(
        REOBSERVE_SESSION_ID,
        "Retry an incomplete dashboard mutation observation.",
    )

    for session_id in (INTERRUPT_SESSION_ID, INTERRUPT_FAILURE_SESSION_ID):
        await store.create(
            RunRequest(
                agent_name=AGENT_NAME,
                session_id=session_id,
                messages=[Message.text("user", "Wait for a dashboard interruption.")],
                max_steps=20,
                tool_capability_ceiling=contract_tool_capability_ceiling,
            ),
            identity=contract_session_identity,
        )

    approval_events = []
    async for event in app.run(
        RunRequest(
            agent_name=AGENT_NAME,
            session_id=APPROVAL_SESSION_ID,
            messages=[Message.text("user", "Seed the dashboard approval contract.")],
        )
    ):
        approval_events.append(event)
    require_equal(
        approval_events[-1].type,
        EventType.SESSION_INTERRUPTED,
        "the browser approval fixture must pause through the real runtime contract",
    )
    require(
        any(event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED for event in approval_events),
        "the browser approval fixture must publish a durable approval request",
    )

    for index in range(PAGINATED_SESSION_COUNT):
        await seed_completed_session(
            f"{PAGINATED_SESSION_PREFIX}-{index:03d}",
            f"Exercise bounded session page {index}.",
            parent_session_id=SESSION_ID,
            causal_budget_id=DISCOVERY_BUDGET_ID,
            environment_name=DISCOVERY_ENVIRONMENT,
            labels={"tenant": "acme", "region": "us", "tier": "critical"},
        )
    await seed_completed_session(
        FILTERED_SESSION_ID,
        "Exercise every server-authoritative session discovery filter.",
        parent_session_id=SESSION_ID,
        causal_budget_id=DISCOVERY_BUDGET_ID,
        environment_name=DISCOVERY_ENVIRONMENT,
        labels={"tenant": "acme", "region": "us", "tier": "critical"},
    )

    async def seed_workflow_session(
        session_id: str,
        *,
        parent_session_id: str | None,
        status: SessionStatus,
        usage: tuple[int, int] | None = None,
        model: str = WORKFLOW_MODEL_NAME,
    ) -> None:
        await store.create(
            RunRequest(
                agent_name=AGENT_NAME,
                session_id=session_id,
                parent_session_id=parent_session_id,
                causal_budget_id=WORKFLOW_BUDGET_ID,
                environment_name=WORKFLOW_ENVIRONMENT,
                labels={"workflow": "dashboard-contract"},
                messages=[Message.text("user", f"Inspect Workflow node {session_id}.")],
            ),
            identity=SessionIdentity(
                provider_name=WORKFLOW_PROVIDER_NAME,
                model=model,
            ),
        )
        events: list[Event] = []
        if usage is not None:
            input_tokens, output_tokens = usage
            events.append(
                Event(
                    id=f"{session_id}-model-completed",
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    agent_name=AGENT_NAME,
                    environment_name=WORKFLOW_ENVIRONMENT,
                    payload={
                        "usage_metrics": {
                            "provider_name": WORKFLOW_PROVIDER_NAME,
                            "requested_model": model,
                            "model": model,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        }
                    },
                )
            )
        terminal_type = {
            SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
            SessionStatus.FAILED: EventType.SESSION_FAILED,
            SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
        }.get(status)
        events.append(
            Event(
                id=f"{session_id}-{status}",
                type=terminal_type or EventType.SESSION_STARTED,
                session_id=session_id,
                agent_name=AGENT_NAME,
                environment_name=WORKFLOW_ENVIRONMENT,
                payload={"error_type": "workflow_demo_failure"}
                if status is SessionStatus.FAILED
                else {},
            )
        )
        await store.append_events(session_id, events)
        await store.update_status(session_id, status)

    await seed_workflow_session(
        WORKFLOW_ROOT_SESSION_ID,
        parent_session_id=None,
        status=SessionStatus.COMPLETED,
    )
    await seed_workflow_session(
        WORKFLOW_PARENT_SESSION_ID,
        parent_session_id=WORKFLOW_ROOT_SESSION_ID,
        status=SessionStatus.COMPLETED,
    )
    await seed_workflow_session(
        WORKFLOW_FOCUS_SESSION_ID,
        parent_session_id=WORKFLOW_PARENT_SESSION_ID,
        status=SessionStatus.COMPLETED,
        usage=(40, 10),
    )
    await seed_workflow_session(
        WORKFLOW_FAILED_SESSION_ID,
        parent_session_id=WORKFLOW_FOCUS_SESSION_ID,
        status=SessionStatus.FAILED,
        usage=(20, 5),
        model=WORKFLOW_SECONDARY_MODEL_NAME,
    )
    await seed_workflow_session(
        WORKFLOW_INTERRUPTED_SESSION_ID,
        parent_session_id=WORKFLOW_FOCUS_SESSION_ID,
        status=SessionStatus.INTERRUPTED,
    )
    for index in range(WORKFLOW_CHILD_SESSION_COUNT - 3):
        await seed_workflow_session(
            f"{WORKFLOW_CHILD_SESSION_PREFIX}-{index:03d}",
            parent_session_id=WORKFLOW_FOCUS_SESSION_ID,
            status=SessionStatus.COMPLETED,
        )
    # Keep the running child beyond the first page so browser verification
    # proves that routine refresh reconciles retained continuation pages.
    await seed_workflow_session(
        WORKFLOW_ACTIVE_SESSION_ID,
        parent_session_id=WORKFLOW_FOCUS_SESSION_ID,
        status=SessionStatus.RUNNING,
    )

    workflow_invocation = await store.load_invocation_snapshot(WORKFLOW_FOCUS_SESSION_ID)
    if workflow_invocation is None:
        raise AssertionError("Workflow focus session disappeared during dashboard seeding.")
    await app.create_task(
        TaskCreate(
            task_id=WORKFLOW_PARENT_TASK_ID,
            type="workflow",
            title="Coordinate dashboard Workflow verification",
            session_id=WORKFLOW_FOCUS_SESSION_ID,
            assigned_agent_name=AGENT_NAME,
        )
    )
    await task_store.start_task(
        WORKFLOW_PARENT_TASK_ID,
        session_invocation=workflow_invocation,
    )
    for index in range(WORKFLOW_LINKED_TASK_COUNT - 1):
        task_id = (
            WORKFLOW_BLOCKED_TASK_ID if index == 0 else f"{WORKFLOW_LINKED_TASK_PREFIX}-{index:03d}"
        )
        await app.create_task(
            TaskCreate(
                task_id=task_id,
                type="workflow_step",
                title=f"Linked Workflow step {index}",
                session_id=WORKFLOW_FOCUS_SESSION_ID,
                assigned_agent_name=AGENT_NAME,
            )
        )
        if task_id == WORKFLOW_BLOCKED_TASK_ID:
            await task_store.block_task(task_id, reason="Waiting for a reviewed dependency.")
        elif index == 1:
            await task_store.fail_task(task_id, {"error_type": "workflow_demo_failure"})
        else:
            await task_store.complete_task(task_id, {"verified": True})

    for index in range(WORKFLOW_CHILD_TASK_COUNT):
        task_id = f"{WORKFLOW_CHILD_TASK_PREFIX}-{index:03d}"
        await app.create_task(
            TaskCreate(
                task_id=task_id,
                type="workflow_substep",
                title=f"Child Workflow step {index}",
                parent_task_id=WORKFLOW_PARENT_TASK_ID,
                assigned_agent_name=AGENT_NAME,
            )
        )
        if index == 0:
            await task_store.fail_task(task_id, {"error_type": "workflow_demo_failure"})
        else:
            await task_store.complete_task(task_id, {"verified": True})

    return app, provider, store, task_store, runtime_directory


def _dashboard_price_book() -> PriceBook:
    return PriceBook(
        price_book_version="dashboard-browser-contract",
        generated_at="2026-07-21",
        prices=(
            ModelPrice.fixed(
                provider_name=PROVIDER_NAME,
                model=MODEL_NAME,
                match="exact",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
            ModelPrice.fixed(
                provider_name="bedrock",
                model="global.anthropic.claude-sonnet-4-6",
                match="exact",
                pricing_context={
                    "source_region": ("us-east-1",),
                    "service_tier": ("default",),
                },
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
            ModelPrice.fixed(
                provider_name=WORKFLOW_PROVIDER_NAME,
                model=WORKFLOW_MODEL_NAME,
                match="exact",
                input_per_million=Decimal("0.002"),
                output_per_million=Decimal("0.002"),
            ),
            ModelPrice.fixed(
                provider_name=WORKFLOW_PROVIDER_NAME,
                model=WORKFLOW_SECONDARY_MODEL_NAME,
                match="exact",
                input_per_million=Decimal("0.004"),
                output_per_million=Decimal("0.004"),
                currency="CAD",
            ),
        ),
    )


def build_release_acceptance_eval_plan() -> EvalPlan:
    """Build the local side of the dashboard-to-CI installed-package proof."""

    app = CayuApp(enable_logging=False)
    runtime_root = Path(tempfile.mkdtemp(prefix="cayu-dashboard-local-evals-"))
    workspace_root = runtime_root / "workspace"
    workspace_root.mkdir()
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=LocalWorkspace(workspace_root, workspace_id="dashboard-workspace"),
            artifact_store=LocalArtifactStore(
                runtime_root / "artifacts",
                store_id="dashboard-artifacts",
            ),
        ),
        default=True,
    )
    app.register_provider(DashboardContractProvider(), default=True)
    app.register_agent(
        AgentSpec(name=AGENT_NAME, model=MODEL_NAME),
        tools=[DashboardContractTool(), DashboardEvalSearchTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=["dashboard_contract_tool"]),
    )
    return EvalPlan(
        corpus_target=CorpusTarget(
            key="dashboard.regressions",
            app=app,
            request_base=RunRequest(agent_name=AGENT_NAME, messages=[]),
            application_release_id="dashboard-local-ci-contract",
            price_book=_dashboard_price_book(),
            evidence_policy=EvaluationEvidencePolicySpec.create(
                include_tool_results=True,
                include_artifact_text=True,
            ),
        )
    )


async def _run_browser_contract(
    base_url: str,
    provider: DashboardContractProvider,
    faults: MutationDisconnectFaults,
    session_store: InMemorySessionStore,
    task_store: InMemoryTaskStore,
) -> dict[str, object]:
    browser_failures: dict[str, list[str]] = {
        "console_errors": [],
        "page_errors": [],
        "request_failures": [],
        "api_errors": [],
    }
    expected_observer_aborts: list[ObserverAbort] = []
    expected_query_aborts: list[str] = []
    superseded_read_aborts: list[str] = []
    expected_edit_rejections: list[str] = []
    expected_edit_console_errors: list[str] = []
    expected_usage_rejections: list[str] = []
    expected_usage_console_errors: list[str] = []
    expected_observer_abort_paths = {
        "/api/run",
        "/api/resume",
        "/api/tool-approvals/resolve",
        f"/api/sessions/{INTERRUPT_SESSION_ID}/interrupt",
        f"/api/sessions/{INTERRUPT_FAILURE_SESSION_ID}/interrupt",
        f"/api/sessions/{RESUME_INTERRUPT_SESSION_ID}/interrupt",
    }
    diagnostics_dir = Path(
        os.environ.get(
            "CAYU_DASHBOARD_DIAGNOSTICS_DIR",
            str(Path(tempfile.gettempdir()) / "cayu-dashboard-behavior"),
        )
    )

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1440, "height": 1000},
            http_credentials={"username": AUTH_USERNAME, "password": AUTH_PASSWORD},
            locale="en-US",
        )
        await context.grant_permissions(["clipboard-read", "clipboard-write"], origin=base_url)
        await context.tracing.start(screenshots=True, snapshots=True)
        page = await context.new_page()
        _record_browser_failures(
            page,
            browser_failures,
            expected_observer_aborts,
            expected_observer_abort_paths,
            expected_query_aborts,
            superseded_read_aborts,
            expected_edit_rejections,
            expected_edit_console_errors,
            expected_usage_rejections,
            expected_usage_console_errors,
        )
        try:
            await _exercise_contract_version_gate(page, base_url)
            await _exercise_capability_contract(page, base_url)
            await _exercise_system_page(page, base_url)
            await _exercise_dashboard(
                page,
                base_url,
                provider,
                faults,
                session_store,
                task_store,
                expected_query_aborts,
            )
            _require_no_browser_failures(browser_failures)
            run_observer_aborts = [
                detail
                for path, last_event_id, detail in expected_observer_aborts
                if path == "/api/run" and last_event_id is not None
            ]
            require_equal(
                len(run_observer_aborts),
                2,
                "each recovered browser run must close exactly one replay SSE observer; "
                f"observed={run_observer_aborts!r}",
            )
            eval_query_aborts = [
                detail for detail in expected_query_aborts if "/api/evals/runs?" in detail
            ]
            eval_catalog_root_query_aborts = [
                detail for detail in expected_query_aborts if "/api/evals/corpora?" in detail
            ]
            eval_catalog_projection_aborts = [
                detail for detail in expected_query_aborts if "/api/evals/corpora/" in detail
            ]
            require_equal(
                len(expected_query_aborts)
                - len(eval_query_aborts)
                - len(eval_catalog_root_query_aborts)
                - len(eval_catalog_projection_aborts),
                3,
                "the superseded session, usage, and Workflow queries must each abort one "
                "browser request",
            )
            require(
                bool(eval_query_aborts),
                "leaving a refreshing Evals run list must abort its superseded browser read",
            )
            require(
                bool(eval_catalog_root_query_aborts),
                "leaving an importing Evals page must abort its superseded catalog read",
            )
            require_equal(
                len(expected_edit_rejections),
                1,
                "the injected session edit rejection must remain an expected API response",
            )
            require_equal(
                len(expected_edit_console_errors),
                len(expected_edit_rejections),
                "each expected edit rejection must produce one classified Chromium resource error",
            )
            require_equal(
                len(expected_usage_rejections),
                1,
                "the injected usage refresh failure must remain an expected API response",
            )
            require_equal(
                len(expected_usage_console_errors),
                len(expected_usage_rejections),
                "each expected usage rejection must produce one classified Chromium resource error",
            )
        except BaseException:
            await _capture_diagnostics(context, page, diagnostics_dir, browser_failures)
            raise
        else:
            await context.tracing.stop()
        finally:
            await browser.close()

    return {
        "browser": "chromium",
        "base_url": base_url,
        "session_id": SESSION_ID,
        "interactions": [
            "new_run_agent_selection",
            "mutation_pre_frame_recovery",
            "contract_version_gate",
            "capability_aware_routes",
            "evals_readiness_shell",
            "usage_without_default_pricing",
            "unavailable_mutation_controls",
            "overview_read_only_controls",
            "manual_system_snapshot",
            "mutation_post_frame_recovery",
            "session_resume",
            "approval_resolution",
            "session_interrupt",
            "active_resume_interrupt",
            "interrupt_failure_dismissal",
            "manual_mutation_reobservation",
            "sessions_list",
            "session_detail",
            "captured_evaluation_preview_and_assertion_authoring",
            "scenario_authoring_and_controlled_execution",
            "authored_eval_suite_creation_duplication_and_subset_launch",
            "authored_scenario_structured_input",
            "structured_judge_authoring_and_fixed_evidence_calibration",
            "authored_eval_suite_reuse",
            "captured_result_save_and_baseline",
            "eval_result_catalog_navigation",
            "eval_catalog_import_and_download",
            "eval_durable_launch_and_cancellation",
            "eval_result_inspection_and_reports",
            "eval_compatible_comparison",
            "eval_local_rerun_and_ci",
            "session_annotation_editing",
            "event_detail",
            "event_filters",
            "exact_event_lookup",
            "filtered_failure_diagnostics",
            "transcript_filters",
            "history_navigation",
            "session_cursor_pagination",
            "session_filter_url_state",
            "session_query_cancellation",
            "overview_operational_snapshot",
            "usage_aggregate_scope",
            "usage_filter_url_state",
            "workflow_topology",
            "workflow_branch_pagination",
            "workflow_url_restoration",
            "workflow_keyboard_navigation",
            "workflow_refresh_policy",
            "workflow_without_pricing",
            "workflow_unavailable",
        ],
        "console_errors": 0,
        "page_errors": 0,
        "mutation_observer_aborts": len(expected_observer_aborts),
        "query_aborts": len(expected_query_aborts),
        "superseded_read_aborts": len(superseded_read_aborts),
        "session_edit_rejections": len(expected_edit_rejections),
        "usage_refresh_rejections": len(expected_usage_rejections),
        "request_failures": 0,
        "api_errors": 0,
    }


async def _exercise_dashboard(
    page: Page,
    base_url: str,
    provider: DashboardContractProvider,
    faults: MutationDisconnectFaults,
    session_store: InMemorySessionStore,
    task_store: InMemoryTaskStore,
    expected_query_aborts: list[str],
) -> None:
    session_event_records = await session_store.query_events(
        EventQuery(session_id=SESSION_ID, limit=100)
    )

    def projected_event_id(event_type: EventType) -> str:
        matches = [record for record in session_event_records if record.event.type is event_type]
        require_equal(
            len(matches),
            1,
            f"the dashboard fixture must contain one {event_type} event",
        )
        record = matches[0]
        return f"cayu_event_{record.sequence}"

    model_completed_event_id = projected_event_id(EventType.MODEL_COMPLETED)
    tool_failed_event_id = projected_event_id(EventType.TOOL_CALL_FAILED)

    await _exercise_operational_scope(page, base_url)
    await _exercise_run_agent_inventory(page, base_url)
    await _exercise_mutation_recovery(page, base_url)
    await page.goto(f"{base_url}/cayu/sessions", wait_until="networkidle")
    require((await page.locator("body").inner_text()).strip() != "", "dashboard rendered blank")

    await expect(page.get_by_role("heading", name="Sessions").first).to_be_visible()
    await _exercise_session_discovery(page, faults)
    session_link = page.get_by_role("link", name=SESSION_ID)
    await expect(session_link).to_be_visible()
    session_row = page.get_by_role("row").filter(has_text=SESSION_ID)
    await expect(session_row.get_by_text("completed", exact=True)).to_be_visible()
    await session_link.click()

    await expect(page).to_have_url(re.compile(rf"/cayu/sessions/{SESSION_ID}$"))
    await expect(page.get_by_role("heading", name=SESSION_ID)).to_be_visible()
    token_stat = page.get_by_text("Tokens", exact=True).locator("..")
    await expect(token_stat.get_by_text("15", exact=True)).to_be_visible()
    await _exercise_session_annotations(page)

    completed_event = page.get_by_role("button", name=re.compile(r"model\.completed"))
    await expect(completed_event).to_be_visible()
    await completed_event.click()

    await expect(page.get_by_text("Event Detail", exact=True)).to_be_visible()
    await expect(page.get_by_text("model.completed", exact=True).last).to_be_visible()
    await expect(page.locator("pre").filter(has_text=PAYLOAD_MARKER)).to_be_visible()

    event_type_filter = page.get_by_label("Filter events by exact event type")
    await event_type_filter.fill("model.completed")
    await page.get_by_role("button", name="Apply filters").click()
    await expect(page).to_have_url(re.compile(r"[?&]event_type=model\.completed(?:&|$)"))
    await expect(page.get_by_role("button", name=re.compile(r"model\.completed"))).to_be_visible()
    await expect(page.get_by_role("button", name=re.compile(r"model\.started"))).to_have_count(0)
    await expect(page.get_by_text("Tool failed: browser_contract_tool", exact=True)).to_be_visible()
    await page.get_by_role("button", name="Inspect event").click()
    await expect(page).to_have_url(
        re.compile(rf"[?&]event_id={re.escape(tool_failed_event_id)}(?:&|$)")
    )
    await expect(page.get_by_text(tool_failed_event_id, exact=True)).to_be_visible()

    event_id_filter = page.get_by_label("Filter events by exact event ID")
    await event_id_filter.fill(model_completed_event_id)
    await event_type_filter.fill("model.completed")
    await page.get_by_role("button", name="Apply filters").click()
    await expect(page).to_have_url(
        re.compile(rf"[?&]event_id={re.escape(model_completed_event_id)}(?:&|$)")
    )
    await expect(page.get_by_text(model_completed_event_id, exact=True)).to_be_visible()

    transcript_role_filter = page.get_by_label("Filter transcript by role")
    await transcript_role_filter.select_option("assistant")
    await expect(page).to_have_url(re.compile(r"[?&]transcript_role=assistant(?:&|$)"))
    await page.reload(wait_until="networkidle")
    await expect(event_type_filter).to_have_value("model.completed")
    await expect(event_id_filter).to_have_value(model_completed_event_id)
    await expect(transcript_role_filter).to_have_value("assistant")
    await expect(
        page.get_by_text("dashboard transcript assistant marker", exact=True)
    ).to_be_visible()
    await expect(page.get_by_text("dashboard transcript user marker", exact=True)).to_have_count(0)
    thinking_payload = page.locator("pre").filter(has_text="dashboard transcript thinking marker")
    await expect(thinking_payload).to_be_visible()

    include_thinking = page.get_by_label("Include thinking")
    await include_thinking.uncheck()
    await expect(page).to_have_url(re.compile(r"[?&]include_thinking=false(?:&|$)"))
    await expect(thinking_payload).to_have_count(0)
    await expect(
        page.get_by_text("dashboard transcript assistant marker", exact=True)
    ).to_be_visible()

    await page.go_back()
    await expect(include_thinking).to_be_checked()
    await expect(thinking_payload).to_be_visible()
    await _exercise_existing_session_mutations(page, base_url, provider)
    await _exercise_captured_evaluation(
        page,
        base_url,
        provider,
        expected_query_aborts,
    )
    await _exercise_workflow(
        page,
        base_url,
        session_store,
        task_store,
        expected_query_aborts,
    )


async def _exercise_captured_evaluation(
    page: Page,
    base_url: str,
    provider: DashboardContractProvider,
    expected_query_aborts: list[str],
) -> None:
    await page.goto(
        f"{base_url}/cayu/sessions/{WORKFLOW_ACTIVE_SESSION_ID}",
        wait_until="networkidle",
    )
    await expect(page.get_by_test_id("evaluate-session")).to_have_count(0)

    await page.goto(
        f"{base_url}/cayu/sessions/{PROMOTION_SESSION_ID}",
        wait_until="networkidle",
    )
    evaluate = page.get_by_test_id("evaluate-session")
    await expect(evaluate).to_be_visible()
    await evaluate.focus()
    await page.keyboard.press("Enter")

    sheet = page.get_by_test_id("promotion-sheet")
    await expect(sheet).to_be_visible()
    await expect(sheet.get_by_text("Captured evidence", exact=True)).to_be_visible()
    await expect(
        sheet.get_by_text("This score matches the current edits.", exact=True)
    ).to_be_visible()
    await expect(sheet.get_by_test_id("scenario-authoring")).to_be_visible()

    export = sheet.get_by_test_id("promotion-export")
    await expect(export).to_be_enabled()
    case_name = sheet.get_by_label("Case name", exact=True)
    await case_name.fill("Captured dashboard regression")
    await sheet.get_by_label("Assertion quick-add type").select_option("final_output_contains")
    await sheet.get_by_role("button", name="Add observed", exact=True).click()
    await expect(export).to_be_disabled()
    await expect(
        sheet.get_by_text("Edit detected. Preview again before export.", exact=True)
    ).to_be_visible()

    await sheet.get_by_test_id("promotion-preview").click()
    await expect(
        sheet.get_by_text("This score matches the current edits.", exact=True)
    ).to_be_visible()
    await expect(export).to_be_enabled()

    scenario_editor = sheet.get_by_test_id("scenario-authoring")
    await scenario_editor.get_by_role("button", name="Queue input", exact=True).click()
    queued_event = scenario_editor.get_by_test_id("scenario-event-1")
    await expect(queued_event).to_be_visible()
    queued_text = queued_event.locator("textarea")
    require_equal(
        await queued_text.count(),
        1,
        "a newly queued scenario event must expose exactly one text-part editor",
    )
    await queued_text.fill("Verify the queued production follow-up.")
    await scenario_editor.get_by_role("button", name="Approval", exact=True).click()
    approval_event = scenario_editor.get_by_test_id("scenario-event-2")
    await expect(approval_event).to_be_visible()
    await approval_event.get_by_label("Current tool name", exact=True).fill(
        "dashboard_contract_tool"
    )
    await scenario_editor.get_by_role(
        "spinbutton",
        name="Max tool calls per trial",
    ).fill("5")
    await scenario_editor.get_by_role(
        "button",
        name="Check readiness",
        exact=True,
    ).click()
    await expect(
        scenario_editor.get_by_text(
            "Current launch requirements are ready",
            exact=True,
        )
    ).to_be_visible()
    await scenario_editor.get_by_role("button", name="Save scenario", exact=True).click()
    await expect(scenario_editor.get_by_text(re.compile(r"Saved scenario .+\."))).to_be_visible()
    # A later cancellation check intentionally blocks the next promoted run.
    # Keep this scenario run deterministic, then re-arm that one-shot fixture.
    provider.block_next_promotion_run = False
    await scenario_editor.get_by_role("button", name="Run scenario", exact=True).click()
    await expect(
        scenario_editor.get_by_text(
            re.compile(r"Opened eval run .+ \((queued|running)\)\. Follow it in the Runs tab\."),
        )
    ).to_be_visible()
    scenario_runs_response = await page.request.get(
        f"{base_url}/api/evals/runs?target_key=dashboard.regressions&limit=100"
    )
    require_equal(
        scenario_runs_response.status,
        200,
        "the admitted scenario must be discoverable in the durable run catalog",
    )
    scenario_runs = [
        run
        for run in (await scenario_runs_response.json())["items"]
        if run["spec"]["invocation"].get("scenario") is not None
    ]
    require_equal(
        len(scenario_runs),
        1,
        "the explicit Run scenario action must admit exactly one scenario run",
    )
    scenario_run_id = scenario_runs[0]["spec"]["run_id"]

    async with page.expect_download() as download_info:
        await export.click()
    download = await download_info.value
    require_equal(
        download.suggested_filename,
        "dashboard.regressions-captured.eval.json",
        "captured export must retain the server-owned portable target filename",
    )
    download_path = await download.path()
    require(download_path is not None, "promotion export must produce a browser download")
    captured_corpus = json.loads(Path(download_path).read_text(encoding="utf-8"))
    require_equal(
        captured_corpus["target_key"],
        "dashboard.regressions",
        "captured export must retain the configured target key",
    )
    require_equal(
        captured_corpus["cases"][0]["name"],
        "Captured dashboard regression",
        "captured export must contain the exact rescored case edit",
    )
    require_equal(
        captured_corpus["cases"][0]["input"],
        None,
        "captured export must not invent runnable session input",
    )
    require(
        PROMOTION_SESSION_ID not in json.dumps(captured_corpus, sort_keys=True),
        "captured export must not disclose its source session identity",
    )

    save = sheet.get_by_test_id("promotion-save")
    await expect(save).to_be_enabled()
    await save.click()
    await expect(sheet.get_by_text(re.compile(r"Saved result .* to Evals\."))).to_be_visible()
    await sheet.get_by_role("button", name="Approve baseline", exact=True).click()
    await expect(sheet.get_by_text("Baseline approved", exact=True)).to_be_visible()
    await sheet.get_by_role("link", name="Open Evals").click()

    await expect(page).to_have_url(re.compile(r"/cayu/evals\?"))
    await expect(page.get_by_role("heading", name="Evals")).to_be_visible()
    await expect(page.locator("#evals-panel-catalog")).to_have_count(1)
    await expect(page.locator("#evals-panel-results")).to_have_count(1)
    await expect(page.locator("#evals-panel-runs")).to_have_count(1)
    await expect(page.get_by_role("tab", name="Results", exact=True)).to_have_attribute(
        "aria-selected", "true"
    )
    await expect(page.get_by_text("Explainable result", exact=True)).to_be_visible()
    await expect(page.get_by_text("Baseline", exact=True)).to_be_visible()

    await page.get_by_role("tab", name="Runs", exact=True).click()
    retained_corpus_filter = page.get_by_role("tabpanel", name="Runs").get_by_role(
        "button", name=re.compile(r"^Corpus ")
    )
    if await retained_corpus_filter.count() == 1:
        await retained_corpus_filter.click()
    scenario_run_button = page.get_by_title(scenario_run_id, exact=True)
    await expect(scenario_run_button).to_be_visible()
    await scenario_run_button.click()
    scenario_progress = page.get_by_test_id("scenario-run-progress")
    await expect(scenario_progress).to_contain_text("awaiting approval", timeout=20_000)
    await expect(scenario_progress).to_contain_text(
        "Fresh approval required for dashboard_contract_tool."
    )
    await scenario_progress.get_by_role("button", name="Approve", exact=True).click()
    await expect(
        page.get_by_text("Approved trial 1's fresh tool request.", exact=True)
    ).to_be_visible()
    await expect(page.get_by_test_id("eval-run-status-announcement")).to_have_text(
        "Eval run status: completed.",
        timeout=20_000,
    )
    await expect(scenario_progress).to_contain_text("completed")
    await expect(
        page.locator('[data-slot="card-title"]').filter(has_text="Published result")
    ).to_be_visible()
    async with page.expect_download() as scenario_json_download_info:
        await page.get_by_role("button", name="JSON", exact=True).click()
    scenario_json_download = await scenario_json_download_info.value
    require_equal(
        scenario_json_download.suggested_filename,
        f"{scenario_run_id}.eval-result.json",
        "the scenario result must retain the ordinary eval-result filename",
    )
    scenario_result_path_text = await scenario_json_download.path()
    require(
        scenario_result_path_text is not None,
        "the downloaded scenario result must be readable by the installed CLI",
    )
    scenario_result_path = Path(scenario_result_path_text)
    scenario_report = json.loads(scenario_result_path.read_text(encoding="utf-8"))
    require_equal(
        scenario_report["result"]["run"]["status"],
        "passed",
        "the queued-input and fresh-approval scenario must publish an ordinary passing result",
    )
    async with page.expect_download() as scenario_html_download_info:
        await page.get_by_role("button", name="HTML", exact=True).click()
    scenario_html_download = await scenario_html_download_info.value
    require_equal(
        scenario_html_download.suggested_filename,
        f"{scenario_run_id}.eval-report.html",
        "the scenario run must publish the ordinary HTML report",
    )

    await page.get_by_test_id("new-evaluation").click()
    authoring = page.get_by_test_id("eval-suite-authoring-sheet")
    await expect(authoring).to_be_visible()
    await authoring.get_by_test_id("authored-suite-id").fill("authored-dashboard-regressions")
    await authoring.get_by_test_id("authored-suite-name").fill("Authored dashboard regressions")
    await authoring.get_by_test_id("authored-case-id").fill("authored-primary")
    await authoring.get_by_test_id("authored-case-name").fill("Authored primary behavior")
    await authoring.get_by_role(
        "textbox",
        name="Case input message 1",
        exact=True,
    ).fill("Exercise one fresh Control Plane-authored evaluation.")
    await authoring.get_by_role("button", name="Add same-model AI judge", exact=True).click()
    await expect(authoring.get_by_test_id("judge-profile-summary")).to_contain_text(
        re.compile(r"same model as candidate", re.IGNORECASE)
    )
    memory_authoring = authoring.get_by_test_id("eval-memory-authoring")
    await memory_authoring.get_by_role("button", name="Require memory exposure", exact=True).click()
    memory_assertion = authoring.get_by_test_id("promotion-assertion").last
    await memory_assertion.get_by_label("Minimum admitted items", exact=True).fill("0")
    await memory_assertion.get_by_label("Maximum admitted items", exact=True).fill("0")
    await memory_assertion.get_by_label("Minimum provider exposures", exact=True).fill("0")
    await memory_assertion.get_by_label("Maximum provider exposures", exact=True).fill("0")
    await memory_authoring.get_by_role(
        "button", name="Add reference-backed judge", exact=True
    ).click()
    memory_judge = authoring.get_by_test_id("structured-judge-editor").last
    await expect(memory_judge.get_by_label("Criterion 1 ID", exact=True)).to_have_value(
        "reference-correctness"
    )
    await memory_judge.get_by_label("Expected facts (one per line)", exact=True).fill(
        "The fresh run should finish without admitting memory."
    )
    await expect(
        memory_authoring.get_by_text(
            "Replace the blank expected fact in the memory-use judge with trusted reference "
            "truth before checking or saving the suite.",
            exact=True,
        )
    ).to_have_count(0)
    await memory_judge.get_by_role("button", name="Remove AI judge", exact=True).click()
    assertion_kind = authoring.get_by_label("Assertion quick-add type", exact=True)
    await assertion_kind.select_option("tool_arguments_contain")
    await authoring.get_by_role("button", name="Add expectation", exact=True).click()
    argument_assertion = authoring.get_by_test_id("promotion-assertion").last
    await argument_assertion.get_by_label("Tool name", exact=True).fill("dashboard_eval_search")
    await argument_assertion.get_by_label(
        "Expected argument JSON subset",
        exact=True,
    ).fill('{"query":"cayu"}')
    await assertion_kind.select_option("tool_result_contains")
    await authoring.get_by_role("button", name="Add expectation", exact=True).click()
    result_assertion = authoring.get_by_test_id("promotion-assertion").last
    await result_assertion.get_by_label("Tool name", exact=True).fill("dashboard_eval_search")
    await result_assertion.get_by_label(
        "Expected result JSON subset",
        exact=True,
    ).fill('{"structured":{"status":"ok"}}')
    await assertion_kind.select_option("process_event")
    await authoring.get_by_role("button", name="Add expectation", exact=True).click()
    process_count_assertion = authoring.get_by_test_id("promotion-assertion").last
    await process_count_assertion.get_by_label("Process event", exact=True).select_option(
        "tool_call_started"
    )
    await process_count_assertion.get_by_label("Maximum count", exact=True).fill("1")
    await assertion_kind.select_option("process_events_in_order")
    await authoring.get_by_role("button", name="Add expectation", exact=True).click()
    process_order_assertion = authoring.get_by_test_id("promotion-assertion").last
    await process_order_assertion.get_by_label(
        "Expected process event 1",
        exact=True,
    ).select_option("session_started")
    await process_order_assertion.get_by_label(
        "Expected process event 2",
        exact=True,
    ).select_option("tool_call_started")
    await process_order_assertion.get_by_role("button", name="Add event", exact=True).click()
    await process_order_assertion.get_by_label(
        "Expected process event 3",
        exact=True,
    ).select_option("tool_call_completed")
    await process_order_assertion.get_by_role("button", name="Add event", exact=True).click()
    await process_order_assertion.get_by_label(
        "Expected process event 4",
        exact=True,
    ).select_option("session_completed")
    await assertion_kind.select_option("workspace_file")
    await authoring.get_by_role("button", name="Add expectation", exact=True).click()
    workspace_assertion = authoring.get_by_test_id("promotion-assertion").last
    await workspace_assertion.get_by_label("Relative workspace path", exact=True).fill(
        "dashboard-eval-output.json"
    )
    await workspace_assertion.get_by_label("Minimum bytes", exact=True).fill("1")
    await assertion_kind.select_option("artifact")
    await authoring.get_by_role("button", name="Add expectation", exact=True).click()
    artifact_assertion = authoring.get_by_test_id("promotion-assertion").last
    await artifact_assertion.get_by_label("Filename (optional)", exact=True).fill(
        "dashboard-eval-report.json"
    )
    await artifact_assertion.get_by_label("Content type (optional)", exact=True).fill(
        "application/json"
    )
    await artifact_assertion.get_by_label(
        "Public artifact text contains (optional)",
        exact=True,
    ).fill('"status":"ready"')
    case_list = authoring.get_by_test_id("authored-suite-cases")
    case_id_input = authoring.get_by_test_id("authored-case-id")
    case_name_input = authoring.get_by_test_id("authored-case-name")
    primary_assertion_count = await authoring.get_by_test_id("promotion-assertion").count()
    await authoring.get_by_role("button", name="Duplicate", exact=True).click()
    await expect(case_id_input).to_have_value("authored-primary-copy")
    await expect(case_name_input).to_have_value("Authored primary behavior copy")
    await expect(authoring.get_by_test_id("promotion-assertion")).to_have_count(
        primary_assertion_count
    )
    await authoring.get_by_label("Remove case authored-primary-copy", exact=True).click()
    await expect(case_list.locator('input[type="checkbox"]:checked')).to_have_count(1)
    await authoring.get_by_role("button", name="Add case", exact=True).click()
    await case_id_input.fill("authored-primary")
    await expect(case_name_input).to_have_value("Case 2")
    await expect(case_list.locator('input[type="checkbox"]:checked')).to_have_count(2)
    await case_id_input.fill("authored-scenario")
    await case_list.get_by_role("button").filter(has_text="Authored primary behavior").click()
    await expect(case_id_input).to_have_value("authored-primary")
    await case_list.get_by_role("button").filter(has_text="Case 2").click()
    await expect(case_id_input).to_have_value("authored-scenario")
    await authoring.get_by_role("button", name="Multi-stage scenario", exact=True).click()
    await expect(authoring.get_by_test_id("authored-suite-scenario-lock")).to_be_visible()
    await expect(authoring.get_by_role("button", name="New", exact=True)).to_be_disabled()
    await expect(authoring.get_by_role("button", name="Add case", exact=True)).to_be_disabled()
    await expect(authoring.get_by_test_id("authored-case-id")).to_be_disabled()
    await authoring.get_by_role("button", name="Discard scenario edits", exact=True).click()
    await expect(authoring.get_by_test_id("authored-simple-input")).to_be_visible()
    await expect(authoring.get_by_test_id("authored-suite-scenario-lock")).not_to_be_visible()
    await expect(authoring.get_by_role("button", name="Add case", exact=True)).to_be_enabled()
    await authoring.get_by_role("button", name="Multi-stage scenario", exact=True).click()
    authored_scenario = authoring.get_by_test_id("scenario-authoring")
    authored_initial_event = authored_scenario.get_by_test_id("scenario-event-0")
    await authored_initial_event.locator("textarea").fill(
        "Exercise one Control Plane-authored multi-stage scenario."
    )
    await authored_initial_event.get_by_role("button", name="Add part", exact=True).click()
    await authored_initial_event.locator("select").last.select_option("json")
    await authored_initial_event.locator("textarea").last.fill(
        '{"release":"candidate","structured":true}'
    )
    await authored_scenario.get_by_role("button", name="Check readiness", exact=True).click()
    await expect(
        authored_scenario.get_by_text("Current launch requirements are ready", exact=True)
    ).to_be_visible()
    await authored_scenario.get_by_role("button", name="Save scenario", exact=True).click()
    await expect(
        authoring.get_by_text(re.compile(r"Saved scenario .+ for authored-scenario\."))
    ).to_be_visible()
    await assertion_kind.select_option("tool_arguments_contain")
    await authoring.get_by_role("button", name="Add expectation", exact=True).click()
    scenario_argument_assertion = authoring.get_by_test_id("promotion-assertion").last
    await scenario_argument_assertion.get_by_label("Tool name", exact=True).fill(
        "dashboard_contract_tool"
    )
    await scenario_argument_assertion.get_by_label(
        "Expected argument JSON subset",
        exact=True,
    ).fill('{"operation":"verify-scenario"}')
    await assertion_kind.select_option("tool_result_contains")
    await authoring.get_by_role("button", name="Add expectation", exact=True).click()
    scenario_result_assertion = authoring.get_by_test_id("promotion-assertion").last
    await scenario_result_assertion.get_by_label("Tool name", exact=True).fill(
        "dashboard_contract_tool"
    )
    await scenario_result_assertion.get_by_label(
        "Expected result JSON subset",
        exact=True,
    ).fill('{"structured":{"agent":"dashboard-contract-agent"}}')
    await authoring.get_by_label("Maximum cost per trial", exact=True).fill("0.1")
    await authoring.get_by_label("Select Case 2 for launch", exact=True).uncheck()
    await authoring.get_by_test_id("authored-suite-preview").click()
    await expect(authoring.get_by_text("Suite is ready to save", exact=True)).to_be_visible()
    await case_list.get_by_role("button").filter(has_text="Authored primary behavior").click()
    calibration = authoring.get_by_test_id("judge-calibration")
    await expect(calibration).to_be_visible()
    await calibration.get_by_label("Evidence source ID", exact=True).fill(
        "dashboard-reviewed-known-output"
    )
    known_output = calibration.get_by_label("Known candidate output", exact=True)
    await known_output.fill("dashboard authored evaluation output")
    await calibration.get_by_label("Repeated judge calls", exact=True).fill("2")
    runs_before_calibration_response = await page.request.get(
        f"{base_url}/api/evals/runs?target_key=dashboard.regressions&limit=100"
    )
    require_equal(
        runs_before_calibration_response.status,
        200,
        "the calibration contract must read the durable run catalog",
    )
    runs_before_calibration = len((await runs_before_calibration_response.json())["items"])
    judge_requests_before_calibration = len(provider.judge_requests)
    await calibration.get_by_role("button", name="Check calibration", exact=True).click()
    await expect(calibration.get_by_text("Calibration is ready", exact=True)).to_be_visible()
    await expect(calibration.get_by_text(re.compile(r"2 judge calls"))).to_be_visible()
    await expect(calibration.get_by_text(re.compile(r"Same-model judge:"))).to_be_visible()
    await calibration.get_by_role("button", name="Run calibration", exact=True).click()
    await expect(calibration.get_by_text(re.compile(r"Calibration complete"))).to_be_visible(
        timeout=20_000
    )
    await expect(calibration.get_by_text(re.compile(r"Trial 1: passed"))).to_be_visible()
    await expect(calibration.get_by_text(re.compile(r"Trial 2: passed"))).to_be_visible()
    await expect(
        calibration.get_by_text(re.compile(r"correctness: judge 1 · human 1"))
    ).to_have_count(2)
    await expect(calibration.get_by_text(re.compile(r"Usage: 1 model step"))).to_have_count(2)
    await expect(calibration.get_by_text(re.compile(r"Cost: .* USD"))).to_have_count(2)
    require_equal(
        len(provider.judge_requests),
        judge_requests_before_calibration + 2,
        "fixed-evidence calibration must invoke only the two requested judge trials",
    )
    runs_after_calibration_response = await page.request.get(
        f"{base_url}/api/evals/runs?target_key=dashboard.regressions&limit=100"
    )
    require_equal(
        runs_after_calibration_response.status,
        200,
        "the calibration contract must re-read the durable run catalog",
    )
    require_equal(
        len((await runs_after_calibration_response.json())["items"]),
        runs_before_calibration,
        "fixed-evidence calibration must not admit a candidate eval run",
    )
    await known_output.fill("edited evidence invalidates the completed report")
    await expect(calibration.get_by_text(re.compile(r"Calibration complete"))).to_have_count(0)
    suite_save_started = asyncio.Event()
    release_suite_save = asyncio.Event()

    async def delay_authored_suite_save(route, request) -> None:
        if request.method != "POST":
            await route.continue_()
            return
        suite_save_started.set()
        await release_suite_save.wait()
        await route.continue_()

    authored_suite_path = "**/api/evals/suites"
    await page.route(authored_suite_path, delay_authored_suite_save)
    try:
        await authoring.get_by_test_id("authored-suite-save").click()
        await asyncio.wait_for(suite_save_started.wait(), timeout=5)
        try:
            await expect(authoring.get_by_role("button", name="New", exact=True)).to_be_disabled()
            await expect(
                authoring.get_by_role("button", name="Add case", exact=True)
            ).to_be_disabled()
            await expect(
                authoring.get_by_label(
                    "Select Authored primary behavior for launch",
                    exact=True,
                )
            ).to_be_disabled()
        finally:
            release_suite_save.set()
        await expect(
            authoring.get_by_text(re.compile(r"Saved immutable suite .+\."))
        ).to_be_visible()
    finally:
        release_suite_save.set()
        await page.unroute(authored_suite_path, delay_authored_suite_save)
    await authoring.get_by_test_id("authored-suite-run-preview").click()
    await expect(authoring.get_by_text("1 durable run ready", exact=True)).to_be_visible()
    await expect(authoring.get_by_test_id("authored-suite-exposure")).to_contain_text("0.1 USD")
    await expect(authoring.get_by_test_id("authored-suite-exposure")).to_contain_text(
        "candidate cost not hard bounded"
    )
    await authoring.get_by_test_id("authored-suite-launch").click()
    await expect(
        authoring.get_by_text(
            "Started 1 durable eval run for 1 selected case.",
            exact=True,
        )
    ).to_be_visible()
    authored_run_ids = parse_qs(urlsplit(page.url).query).get("run", [])
    require_equal(
        len(authored_run_ids),
        1,
        "the authored subset launch must open its exact durable run",
    )
    authored_run_id = authored_run_ids[0]
    await page.keyboard.press("Escape")
    await expect(page.get_by_test_id("eval-run-status-announcement")).to_have_text(
        "Eval run status: completed.",
        timeout=20_000,
    )
    await expect(
        page.locator('[data-slot="card-title"]').filter(has_text="Published result")
    ).to_be_visible()
    await expect(page.get_by_text("root_status", exact=True)).to_be_visible()
    await expect(page.get_by_text("tool_arguments_contain", exact=True)).to_be_visible()
    await expect(page.get_by_text("tool_result_contains", exact=True)).to_be_visible()
    await expect(page.get_by_text("process_event", exact=True)).to_be_visible()
    await expect(page.get_by_text("process_events_in_order", exact=True)).to_be_visible()
    await expect(page.get_by_text("workspace_file", exact=True)).to_be_visible()
    await expect(page.get_by_title("artifact", exact=True)).to_be_visible()
    await expect(page.get_by_text("memory_attribution", exact=True)).to_be_visible()
    await expect(page.get_by_text("Dashboard quality judge", exact=False)).to_be_visible()
    await expect(page.get_by_text("same model · explicitly allowed", exact=True)).to_be_visible()
    await expect(page.get_by_text("correctness", exact=True)).to_be_visible()
    await expect(
        page.get_by_text("The known output satisfies the fixed task.", exact=True)
    ).to_be_visible()
    authored_result_response = await page.request.get(
        f"{base_url}/api/evals/runs/{authored_run_id}/result"
    )
    require_equal(
        authored_result_response.status,
        200,
        "the authored structured run must publish an explainable result",
    )
    authored_result_body = await authored_result_response.json()
    authored_assertions = authored_result_body["result"]["run"]["cases"][0]["trials"][0][
        "assertions"
    ]
    authored_tool_assertions = {
        item["detail"]["kind"]: item
        for item in authored_assertions
        if "tool_" in item["detail"]["kind"]
    }
    require_equal(
        sorted(authored_tool_assertions),
        ["tool_arguments_contain", "tool_result_contains"],
        "fresh authoring must publish both tool JSON assertion details",
    )
    require_equal(
        [item["outcome"] for item in authored_tool_assertions.values()],
        ["passed", "passed"],
        "fresh tool JSON assertions must pass through the canonical evaluator",
    )
    authored_process_assertions = {
        item["detail"]["kind"]: item
        for item in authored_assertions
        if item["detail"]["kind"] in {"process_event", "process_events_in_order"}
    }
    require_equal(
        sorted(authored_process_assertions),
        ["process_event", "process_events_in_order"],
        "fresh authoring must publish both portable process assertion details",
    )
    require_equal(
        [item["outcome"] for item in authored_process_assertions.values()],
        ["passed", "passed"],
        "fresh process assertions must pass through the canonical evaluator",
    )
    require_equal(
        authored_process_assertions["process_events_in_order"]["detail"]["expected"],
        [
            "session_started",
            "tool_call_started",
            "tool_call_completed",
            "session_completed",
        ],
        "published process order must retain the exact closed-vocabulary expectation",
    )
    authored_structural_assertions = {
        item["detail"]["kind"]: item
        for item in authored_assertions
        if item["detail"]["kind"] in {"workspace_file", "artifact"}
    }
    require_equal(
        sorted(authored_structural_assertions),
        ["artifact", "workspace_file"],
        "fresh authoring must publish both structural assertion details",
    )
    require_equal(
        [item["outcome"] for item in authored_structural_assertions.values()],
        ["passed", "passed"],
        "fresh structural assertions must pass through the canonical evaluator",
    )
    structural_result_json = json.dumps(authored_structural_assertions, sort_keys=True)
    require(
        '"artifact_id"' not in structural_result_json,
        "published structural details must omit private artifact identities",
    )
    authored_memory_assertion = next(
        item for item in authored_assertions if item["detail"]["kind"] == "memory_attribution"
    )
    require_equal(
        authored_memory_assertion["outcome"],
        "passed",
        "fresh zero-memory evidence must satisfy the explicit zero-count expectation",
    )
    require_equal(
        (
            authored_memory_assertion["detail"]["admitted_item_count"],
            authored_memory_assertion["detail"]["provider_exposure_count"],
        ),
        (0, 0),
        "published memory detail must retain exact complete structural counts",
    )
    authored_presentation = authored_result_body["presentation"]
    authored_tool_presentations = [
        item["tool_json"]
        for item in authored_presentation["cases"][0]["trials"][0]["assertions"]
        if item.get("tool_json") is not None
    ]
    require_equal(
        [item["observation_state"] for item in authored_tool_presentations],
        ["available", "available"],
        "the canonical presentation must retain safe tool JSON detail",
    )
    await expect(page.get_by_test_id("eval-tool-json-detail")).to_have_count(2)
    await expect(page.get_by_text("Observed safe value", exact=True)).to_have_count(2)
    await expect(page.get_by_test_id("eval-process-detail")).to_have_count(2)
    await expect(page.get_by_test_id("eval-process-expected-order")).to_contain_text(
        "Session started → Tool call started → Tool call completed → Session completed"
    )
    await expect(page.get_by_test_id("eval-structure-detail")).to_have_count(2)
    await expect(page.get_by_test_id("eval-memory-evidence")).to_be_visible()
    await expect(page.get_by_test_id("eval-memory-assertion-detail")).to_be_visible()
    await page.get_by_text("Inspect bounded structural source evidence", exact=True).click()
    await expect(page.get_by_text("Recall receipts", exact=True)).to_be_visible()
    await expect(page.get_by_text("Exposure states", exact=True)).to_be_visible()
    await expect(page.get_by_text("Causal contribution", exact=True)).to_be_visible()
    authored_structural_presentations = [
        item["structure"]
        for item in authored_presentation["cases"][0]["trials"][0]["assertions"]
        if item.get("structure") is not None
    ]
    require_equal(
        sorted(item["kind"] for item in authored_structural_presentations),
        ["artifact", "workspace_file"],
        "the canonical presentation must retain both safe structural details",
    )
    require_equal(
        [item["observation_state"] for item in authored_structural_presentations],
        ["available", "available"],
        "the canonical presentation must distinguish available structural evidence",
    )
    require_equal(
        authored_presentation["dimensions"]["evaluator_health"],
        "healthy",
        "the canonical result presentation must keep evaluator health distinct",
    )
    authored_judgment = next(
        item["structured_judge"]
        for item in authored_presentation["cases"][0]["trials"][0]["assertions"]
        if item["kind"] == "structured_model_judge"
    )
    require_equal(
        authored_judgment["criteria"][0]["weighted_contribution"],
        "1",
        "the canonical presentation must expose Cayu's exact criterion contribution",
    )
    authored_comparison_response = await page.request.post(
        f"{base_url}/api/evals/result-comparisons",
        data={
            "baseline_result_revision": authored_presentation["result_revision"],
            "current_result_revision": authored_presentation["result_revision"],
            "score_tolerance": 0,
        },
    )
    require_equal(
        authored_comparison_response.status,
        200,
        "the protected comparison route must accept exact structured result identities",
    )
    authored_comparison = (await authored_comparison_response.json())["comparison"]
    require_equal(
        authored_comparison["structured_judge_comparison_state"],
        "compared",
        "structured comparison must pair exact retained trial identity",
    )
    require_equal(
        authored_comparison["structured_judgments"][0]["criteria"][0]["score_delta"],
        "0",
        "structured comparison must expose exact criterion deltas",
    )
    require_equal(
        authored_comparison["tool_json_comparison_state"],
        "compared",
        "tool JSON comparison must pair exact retained trial identity",
    )
    require_equal(
        [item["observed_value_change"] for item in authored_comparison["tool_json_assertions"]],
        ["unchanged", "unchanged"],
        "tool JSON comparison must preserve bounded observed values",
    )
    async with page.expect_download() as authored_json_download_info:
        await page.get_by_role("button", name="JSON", exact=True).click()
    authored_result_path_text = await (await authored_json_download_info.value).path()
    require(
        authored_result_path_text is not None,
        "the explainable authored-result JSON report must be readable",
    )
    authored_result_path = Path(authored_result_path_text)
    async with page.expect_download() as authored_html_download_info:
        await page.get_by_role("button", name="HTML", exact=True).click()
    authored_html_path = await (await authored_html_download_info.value).path()
    require(
        authored_html_path is not None,
        "the explainable authored-result HTML report must be readable",
    )
    authored_html = Path(authored_html_path).read_text(encoding="utf-8")
    require(
        "The known output satisfies the fixed task." in authored_html
        and "0 item(s) admitted" in authored_html
        and "0 provider exposure(s)" in authored_html
        and "weighted_contribution" not in authored_html
        and "judge_output" not in authored_html,
        "HTML must render readable structured evidence without raw judge payload fields",
    )

    await page.get_by_test_id("new-evaluation").click()
    authoring = page.get_by_test_id("eval-suite-authoring-sheet")
    await authoring.get_by_role(
        "button",
        name=re.compile(r"^Authored dashboard regressions\b"),
    ).click()
    await expect(authoring.get_by_text(re.compile(r"Loaded immutable suite .+\."))).to_be_visible()
    await authoring.get_by_test_id("authored-suite-preview").click()
    await expect(authoring.get_by_text("Suite is ready to save", exact=True)).to_be_visible()
    await authoring.get_by_test_id("authored-suite-run-preview").click()
    await expect(authoring.get_by_text("2 durable runs ready", exact=True)).to_be_visible()
    scenario_case_button = (
        authoring.get_by_test_id("authored-suite-cases")
        .get_by_role("button")
        .filter(has_text="Case 2")
    )
    await scenario_case_button.click()
    revised_scenario = authoring.get_by_test_id("scenario-authoring")
    scenario_preview_started = asyncio.Event()
    release_scenario_preview = asyncio.Event()

    async def delay_clean_scenario_preview(route, request) -> None:
        if request.method != "POST":
            await route.continue_()
            return
        scenario_preview_started.set()
        await release_scenario_preview.wait()
        await route.continue_()

    authored_scenario_preview_path = "**/api/evals/scenarios/preview"
    await page.route(authored_scenario_preview_path, delay_clean_scenario_preview)
    try:
        await revised_scenario.get_by_role("button", name="Check readiness", exact=True).click()
        await asyncio.wait_for(scenario_preview_started.wait(), timeout=5)
        try:
            await expect(authoring.get_by_test_id("authored-suite-preview")).to_be_disabled()
            await expect(authoring.get_by_test_id("authored-suite-save")).to_be_disabled()
            await expect(authoring.get_by_test_id("authored-suite-run-preview")).to_be_disabled()
            await expect(authoring.get_by_test_id("authored-suite-launch")).to_be_disabled()
        finally:
            release_scenario_preview.set()
        await expect(
            revised_scenario.get_by_text("Current launch requirements are ready", exact=True)
        ).to_be_visible()
    finally:
        release_scenario_preview.set()
        await page.unroute(authored_scenario_preview_path, delay_clean_scenario_preview)

    revised_scenario_input = revised_scenario.get_by_test_id("scenario-event-0").locator("textarea")
    await revised_scenario_input.fill(
        "Exercise revised Control Plane-authored multi-stage behavior."
    )
    await expect(authoring.get_by_test_id("authored-suite-scenario-lock")).to_be_visible()
    await expect(authoring.get_by_role("button", name="New", exact=True)).to_be_disabled()
    await expect(authoring.get_by_role("button", name="Add case", exact=True)).to_be_disabled()
    await expect(scenario_case_button).to_be_disabled()
    await expect(authoring.get_by_test_id("authored-case-id")).to_be_disabled()
    await expect(authoring.get_by_test_id("authored-suite-preview")).to_be_disabled()
    await expect(authoring.get_by_test_id("authored-suite-save")).to_be_disabled()
    await expect(authoring.get_by_test_id("authored-suite-run-preview")).to_be_disabled()
    await expect(authoring.get_by_test_id("authored-suite-launch")).to_be_disabled()
    await authoring.get_by_role("button", name="Discard scenario edits", exact=True).click()
    await expect(revised_scenario_input).to_have_value(
        "Exercise one Control Plane-authored multi-stage scenario."
    )
    await expect(authoring.get_by_test_id("authored-suite-scenario-lock")).not_to_be_visible()
    await expect(scenario_case_button).to_be_enabled()
    await expect(authoring.get_by_test_id("authored-case-id")).to_be_enabled()
    await revised_scenario_input.fill(
        "Exercise revised Control Plane-authored multi-stage behavior."
    )
    await expect(authoring.get_by_test_id("authored-suite-scenario-lock")).to_be_visible()

    await revised_scenario.get_by_role("button", name="Check readiness", exact=True).click()
    await expect(
        revised_scenario.get_by_text("Current launch requirements are ready", exact=True)
    ).to_be_visible()
    scenario_save_started = asyncio.Event()
    release_scenario_save = asyncio.Event()

    async def delay_revised_scenario_save(route, request) -> None:
        scenario_save_started.set()
        await release_scenario_save.wait()
        await route.continue_()

    authored_scenario_path = "**/api/evals/scenarios"
    await page.route(authored_scenario_path, delay_revised_scenario_save)
    try:
        await revised_scenario.get_by_role("button", name="Save scenario", exact=True).click()
        await asyncio.wait_for(scenario_save_started.wait(), timeout=5)
        try:
            await expect(authoring.get_by_role("button", name="New", exact=True)).to_be_disabled()
            await expect(
                authoring.get_by_role("button", name="Add case", exact=True)
            ).to_be_disabled()
            await expect(scenario_case_button).to_be_disabled()
            await expect(authoring.get_by_test_id("authored-suite-name")).to_be_disabled()
            await expect(authoring.get_by_test_id("authored-case-id")).to_be_disabled()
        finally:
            release_scenario_save.set()
        await expect(
            authoring.get_by_text(re.compile(r"Saved scenario .+ for authored-scenario\."))
        ).to_be_visible()
    finally:
        release_scenario_save.set()
        await page.unroute(authored_scenario_path, delay_revised_scenario_save)

    await expect(authoring.get_by_test_id("authored-suite-scenario-lock")).not_to_be_visible()
    await expect(authoring.get_by_test_id("authored-suite-preview")).to_be_enabled()
    await expect(authoring.get_by_test_id("authored-suite-run-preview")).to_be_disabled()
    await expect(authoring.get_by_test_id("authored-suite-launch")).to_be_disabled()
    await authoring.get_by_test_id("authored-suite-preview").click()
    await expect(authoring.get_by_text("Suite is ready to save", exact=True)).to_be_visible()
    await authoring.get_by_test_id("authored-suite-save").click()
    await expect(authoring.get_by_text(re.compile(r"Saved immutable suite .+\."))).to_be_visible()
    await authoring.get_by_test_id("authored-suite-run-preview").click()
    await expect(authoring.get_by_text("2 durable runs ready", exact=True)).to_be_visible()
    await authoring.get_by_test_id("authored-suite-name").fill(
        "Authored dashboard regressions revised"
    )
    await authoring.get_by_test_id("authored-suite-preview").click()
    await expect(authoring.get_by_text("Suite is ready to save", exact=True)).to_be_visible()
    await expect(authoring.get_by_test_id("authored-suite-run-preview")).to_be_disabled()
    await expect(authoring.get_by_test_id("authored-suite-launch")).to_be_disabled()
    await page.keyboard.press("Escape")
    require(
        authored_run_id in page.url,
        "reviewing a reusable authored suite must not replace the selected run",
    )
    provider.block_next_promotion_run = True

    await page.get_by_role("tab", name="Results", exact=True).click()
    captured_result_row = page.get_by_role("row").filter(has_text="Captured")
    await expect(captured_result_row).to_have_count(1)
    await captured_result_row.get_by_role("button").click()
    await page.get_by_role("button", name="Open corpus", exact=True).click()
    scenario_row = (
        page.get_by_test_id("scenario-catalog")
        .get_by_role("row")
        .filter(has_text="Captured dashboard regression")
    )
    await expect(scenario_row).to_be_visible()
    await expect(scenario_row.get_by_text("3 events")).to_be_visible()

    runnable_preview_response = await page.request.post(
        f"{base_url}/api/evals/promotion/sessions/{PROMOTION_SESSION_ID}/preview",
        data={},
    )
    require_equal(
        runnable_preview_response.status,
        200,
        "the compatibility promotion endpoint must still build a runnable corpus candidate",
    )
    runnable_candidate = (await runnable_preview_response.json())["candidate"]
    runnable_export_response = await page.request.post(
        f"{base_url}/api/evals/promotion/sessions/{PROMOTION_SESSION_ID}/export",
        data={
            "expected_candidate_revision": runnable_candidate["revision"],
            "candidate": runnable_candidate,
        },
    )
    require_equal(
        runnable_export_response.status,
        200,
        "the compatibility promotion endpoint must export its runnable corpus",
    )
    corpus = await runnable_export_response.json()
    suite_name = corpus["suites"][0]["name"]
    suite_id = corpus["suites"][0]["id"]

    await page.get_by_test_id("eval-import-file").set_input_files(
        {
            "name": "dashboard-browser-contract.eval.json",
            "mimeType": "application/json",
            "buffer": json.dumps(corpus).encode(),
        }
    )
    await expect(page.get_by_text(re.compile(r"Imported corpus .*\."))).to_be_visible()
    async with page.expect_download() as corpus_download_info:
        await page.get_by_role("button", name="Download JSON", exact=True).click()
    corpus_download = await corpus_download_info.value
    require_equal(
        corpus_download.suggested_filename,
        f"{corpus['target_key']}-{corpus['revision'][7:19]}.eval.json",
        "the dashboard must preserve the server-owned corpus filename",
    )
    await page.wait_for_load_state("networkidle")

    async def launch_saved_suite() -> str:
        await page.get_by_role(
            "button",
            name=f"Run suite {suite_name} ({suite_id}) on current app",
            exact=True,
        ).click()
        await expect(page).to_have_url(re.compile(r"[?&]tab=runs(?:&|$)"))
        run_ids = parse_qs(urlsplit(page.url).query).get("run", [])
        require_equal(len(run_ids), 1, "a dashboard launch must select its exact durable run")
        await expect(
            page.get_by_text(
                re.compile(
                    r"^Opened eval run .+ "
                    r"\((queued|running|cancelling|completed|failed|cancelled)\)\.$"
                )
            )
        ).to_be_visible()
        return run_ids[0]

    tab_keyboard_contract_checked = False

    async def reopen_catalog() -> None:
        nonlocal tab_keyboard_contract_checked
        runs_tab = page.get_by_role("tab", name="Runs", exact=True)
        catalog_tab = page.get_by_role("tab", name="Catalog", exact=True)
        await runs_tab.focus()
        if tab_keyboard_contract_checked:
            await page.keyboard.press("Home")
            await expect(catalog_tab).to_have_attribute("aria-selected", "true")
            await expect(page.get_by_role("button", name=suite_name, exact=True)).to_be_visible()
            await page.wait_for_load_state("networkidle")
            return

        await page.keyboard.press("ArrowRight")
        await expect(catalog_tab).to_have_attribute("aria-selected", "true")
        await expect(page.get_by_role("button", name=suite_name, exact=True)).to_be_visible()
        await page.wait_for_load_state("networkidle")
        await catalog_tab.focus()
        await page.keyboard.press("ArrowLeft")
        await expect(runs_tab).to_have_attribute("aria-selected", "true")
        await page.keyboard.press("Home")
        await expect(catalog_tab).to_have_attribute("aria-selected", "true")
        await expect(page.get_by_role("button", name=suite_name, exact=True)).to_be_visible()
        await page.wait_for_load_state("networkidle")
        await page.keyboard.press("End")
        await expect(runs_tab).to_have_attribute("aria-selected", "true")
        await page.keyboard.press("Home")
        await expect(catalog_tab).to_have_attribute("aria-selected", "true")
        await expect(page.get_by_role("button", name=suite_name, exact=True)).to_be_visible()
        await page.wait_for_load_state("networkidle")
        tab_keyboard_contract_checked = True

    cancelled_run_id = await launch_saved_suite()
    await asyncio.wait_for(provider.blocked_promotion_run_started.wait(), timeout=10)
    run_status = page.get_by_test_id("eval-run-status-announcement")
    await expect(run_status).to_have_text("Eval run status: running.")
    status_mutations = await run_status.evaluate(
        """
        element => new Promise(resolve => {
          let mutations = 0
          const observer = new MutationObserver(records => {
            mutations += records.length
          })
          observer.observe(element, { characterData: true, childList: true, subtree: true })
          window.setTimeout(() => {
            observer.disconnect()
            resolve(mutations)
          }, 1800)
        })
        """
    )
    require_equal(
        status_mutations,
        0,
        "background polling must not mutate the run-status live region",
    )
    cancel = page.get_by_role("button", name="Cancel run", exact=True)
    await expect(cancel).to_be_visible(timeout=15_000)
    await cancel.click()
    await expect(page.get_by_role("cell", name="cancelled", exact=True)).to_be_visible(
        timeout=15_000
    )
    await expect(run_status).to_have_text("Eval run status: cancelled.")
    require(
        cancelled_run_id in page.url,
        "cancellation must retain the selected durable run identity",
    )

    await reopen_catalog()
    baseline_run_id = await launch_saved_suite()
    published_result = page.locator('[data-slot="card-title"]').filter(has_text="Published result")
    await expect(published_result).to_be_visible(timeout=20_000)
    await expect(
        page.locator("pre").filter(has_text="dashboard eval promotion output")
    ).to_be_visible()
    await expect(page.get_by_text("Estimated cost", exact=True)).to_be_visible()

    async with page.expect_download() as json_download_info:
        await page.get_by_role("button", name="JSON", exact=True).click()
    json_download = await json_download_info.value
    require_equal(
        json_download.suggested_filename,
        f"{baseline_run_id}.eval-result.json",
        "the dashboard must preserve the server-owned eval JSON filename",
    )
    json_result_path = await json_download.path()
    require(
        json_result_path is not None,
        "the dashboard eval JSON download must produce a readable local file",
    )
    baseline_report = json.loads(Path(json_result_path).read_text(encoding="utf-8"))
    baseline_result_revision = baseline_report["result"]["revision"]
    async with page.expect_download() as html_download_info:
        await page.get_by_role("button", name="HTML", exact=True).click()
    html_download = await html_download_info.value
    require_equal(
        html_download.suggested_filename,
        f"{baseline_run_id}.eval-report.html",
        "the dashboard must preserve the server-owned eval HTML filename",
    )

    await reopen_catalog()
    current_run_id = await launch_saved_suite()
    await expect(published_result).to_be_visible(timeout=20_000)
    require(
        current_run_id != baseline_run_id,
        "two explicit dashboard launches must create distinct durable runs",
    )
    await page.get_by_label("Baseline result revision").fill(baseline_result_revision)
    await page.get_by_role("button", name="Compare", exact=True).click()
    await expect(page.get_by_text("These results are comparable.", exact=True)).to_be_visible()
    await expect(page.get_by_text("No compatible-result regressions.", exact=True)).to_be_visible()
    await _exercise_local_eval_acceptance(
        corpus=corpus,
        dashboard_result_path=Path(json_result_path),
        scenario_result_path=scenario_result_path,
        authored_result_path=authored_result_path,
    )

    await reopen_catalog()
    await page.get_by_role("button", name=suite_name, exact=True).click()

    async def expose_all_catalog_pagination(route, request) -> None:
        if request.method != "GET":
            await route.continue_()
            return
        response = await route.fetch()
        body = await response.json()
        if not isinstance(body, dict) or not isinstance(body.get("items"), list):
            await route.fulfill(response=response)
            return
        body["has_more"] = True
        body["next_cursor"] = "dashboard-next-page"
        await route.fulfill(response=response, json=body)

    corpus_path = "**/api/evals/corpora*"
    nested_catalog_path = "**/api/evals/corpora/**"
    await page.route(corpus_path, expose_all_catalog_pagination)
    await page.route(nested_catalog_path, expose_all_catalog_pagination)
    try:
        await page.reload(wait_until="networkidle")
        for scope in ("corpus catalog", "corpus suites", "suite cases"):
            await expect(
                page.get_by_role("navigation", name=f"{scope} pagination", exact=True)
            ).to_be_visible()
            await expect(
                page.get_by_role("button", name=f"First {scope} page", exact=True)
            ).to_be_visible()
            await expect(
                page.get_by_role("button", name=f"Next {scope} page", exact=True)
            ).to_be_visible()
    finally:
        await page.unroute(nested_catalog_path, expose_all_catalog_pagination)
        await page.unroute(corpus_path, expose_all_catalog_pagination)

    catalog_refresh_started = asyncio.Event()
    release_catalog_refresh = asyncio.Event()
    catalog_refresh_continued = asyncio.Event()

    async def delay_abandoned_catalog_refresh(route, request) -> None:
        if request.method != "GET":
            await route.continue_()
            return
        catalog_refresh_started.set()
        await release_catalog_refresh.wait()
        try:
            await route.continue_()
        except Exception:
            pass
        finally:
            catalog_refresh_continued.set()

    aborted_before = len(expected_query_aborts)
    await page.route(corpus_path, delay_abandoned_catalog_refresh)
    try:
        await page.get_by_test_id("eval-import-file").set_input_files(
            {
                "name": "dashboard-abandoned-import.eval.json",
                "mimeType": "application/json",
                "buffer": json.dumps(corpus).encode(),
            }
        )
        await asyncio.wait_for(catalog_refresh_started.wait(), timeout=5)
        await page.get_by_role("link", name="Sessions", exact=True).click()
        await expect(page).to_have_url(re.compile(r"/cayu/sessions$"))
        await expect(page.get_by_role("heading", name="Sessions").first).to_be_visible()
    finally:
        release_catalog_refresh.set()
        if catalog_refresh_started.is_set():
            await asyncio.wait_for(catalog_refresh_continued.wait(), timeout=5)
        await page.unroute(corpus_path, delay_abandoned_catalog_refresh)

    deadline = asyncio.get_running_loop().time() + 5
    while len(expected_query_aborts) == aborted_before:
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("leaving Evals did not abort the delayed catalog refresh")
        await asyncio.sleep(0.05)
    await page.wait_for_timeout(250)
    await expect(page).to_have_url(re.compile(r"/cayu/sessions$"))


async def _exercise_local_eval_acceptance(
    *,
    corpus: dict[str, object],
    dashboard_result_path: Path,
    scenario_result_path: Path,
    authored_result_path: Path,
) -> None:
    """Prove that dashboard exports are the exact local reporting and CI inputs."""

    application_root = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment.pop("CAYU_PROVIDER", None)
    for name in _LIVE_CREDENTIAL_ENV:
        environment.pop(name, None)

    def run_command(arguments: list[str]) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "cayu", "eval", *arguments],
            cwd=application_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "installed-package eval command failed "
                f"with exit {completed.returncode}: {completed.stdout}{completed.stderr}"
            )

    with tempfile.TemporaryDirectory(prefix="cayu-evals-local-acceptance-") as temporary:
        output_root = Path(temporary)
        corpus_path = output_root / "dashboard-runnable.eval.json"
        corpus_path.write_text(json.dumps(corpus, sort_keys=True), encoding="utf-8")
        inspection_path = output_root / "corpus-inspection.json"
        local_result_path = output_root / "local-result.json"
        local_report_path = output_root / "local-report.html"
        dashboard_report_path = output_root / "dashboard-report.html"
        scenario_report_path = output_root / "scenario-report.html"
        scenario_comparison_path = output_root / "scenario-comparison.json"
        authored_report_path = output_root / "authored-report.html"
        authored_comparison_path = output_root / "authored-comparison.json"
        comparison_path = output_root / "comparison.json"
        target = "dashboard_behavior_live:build_release_acceptance_eval_plan"

        await asyncio.to_thread(run_command, ["validate", str(corpus_path)])
        await asyncio.to_thread(
            run_command,
            ["inspect", str(corpus_path), "--json", "--output", str(inspection_path)],
        )
        await asyncio.to_thread(
            run_command,
            [
                "run",
                target,
                "--corpus",
                str(corpus_path),
                "--output",
                str(local_result_path),
                "--html-output",
                str(local_report_path),
            ],
        )
        await asyncio.to_thread(
            run_command,
            [
                "report",
                str(dashboard_result_path),
                "--output",
                str(dashboard_report_path),
            ],
        )
        await asyncio.to_thread(
            run_command,
            ["report", str(scenario_result_path), "--output", str(scenario_report_path)],
        )
        await asyncio.to_thread(
            run_command,
            ["report", str(authored_result_path), "--output", str(authored_report_path)],
        )
        await asyncio.to_thread(
            run_command,
            [
                "compare",
                str(authored_result_path),
                str(authored_result_path),
                "--json",
                "--output",
                str(authored_comparison_path),
            ],
        )
        await asyncio.to_thread(
            run_command,
            [
                "compare",
                str(scenario_result_path),
                str(scenario_result_path),
                "--json",
                "--output",
                str(scenario_comparison_path),
            ],
        )
        await asyncio.to_thread(
            run_command,
            [
                "compare",
                str(dashboard_result_path),
                str(local_result_path),
                "--json",
                "--output",
                str(comparison_path),
            ],
        )

        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        scenario_comparison = json.loads(scenario_comparison_path.read_text(encoding="utf-8"))
        authored_comparison = json.loads(authored_comparison_path.read_text(encoding="utf-8"))
        require_equal(
            inspection["target_key"],
            "dashboard.regressions",
            "local corpus inspection must preserve the dashboard target contract",
        )
        require_equal(
            comparison["compatibility"]["comparable"],
            True,
            "dashboard and local results must share one comparison contract",
        )
        require_equal(
            comparison["regressions"],
            [],
            "the deterministic local rerun must pass its dashboard CI baseline",
        )
        require(
            comparison["baseline"]["application_release_id"]
            != comparison["current"]["application_release_id"],
            "comparison acceptance must prove that release changes remain comparable",
        )
        require(
            "Cayu Eval Report" in dashboard_report_path.read_text(encoding="utf-8"),
            "the local CLI must render a downloaded dashboard result",
        )
        require(
            "Cayu Eval Report" in scenario_report_path.read_text(encoding="utf-8"),
            "the local CLI must render a downloaded scenario result",
        )
        authored_report = authored_report_path.read_text(encoding="utf-8")
        require(
            "The known output satisfies the fixed task." in authored_report
            and "judge_output" not in authored_report,
            "the installed CLI must render explainable structured evidence without raw output",
        )
        require_equal(
            authored_comparison["structured_judge_comparison_state"],
            "compared",
            "the installed CLI must preserve exact structured-comparison semantics",
        )
        require_equal(
            scenario_comparison["regressions"],
            [],
            "the downloaded scenario result must pass the stable CLI comparison gate",
        )


async def _exercise_workflow(
    page: Page,
    base_url: str,
    session_store: InMemorySessionStore,
    task_store: InMemoryTaskStore,
    expected_query_aborts: list[str],
) -> None:
    topology_path = f"/api/sessions/{WORKFLOW_FOCUS_SESSION_ID}/topology"
    workflow_url = f"{base_url}/cayu/sessions/{WORKFLOW_FOCUS_SESSION_ID}/workflow"
    observed_requests: list[tuple[str, str, dict[str, object] | None]] = []

    def record_workflow_request(request: Request) -> None:
        path = urlsplit(request.url).path
        if not path.startswith("/api/"):
            return
        body: dict[str, object] | None = None
        if request.method == "POST":
            try:
                parsed_body = request.post_data_json
                if isinstance(parsed_body, dict):
                    body = parsed_body
            except Exception:
                body = None
        observed_requests.append((request.method, path, body))

    def request_count(method: str, path: str) -> int:
        return sum(
            1
            for observed_method, observed_path, _body in observed_requests
            if observed_method == method and observed_path == path
        )

    async def wait_for_request_count(method: str, path: str, minimum: int) -> None:
        deadline = asyncio.get_running_loop().time() + 10
        while request_count(method, path) < minimum:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(
                    f"Timed out waiting for {minimum} {method} {path} requests; "
                    f"observed={observed_requests!r}"
                )
            await asyncio.sleep(0.05)

    async def wait_for_request_quiescence(
        method: str,
        path: str,
        *,
        quiet_seconds: float = 1.0,
    ) -> int:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 10
        observed_count = request_count(method, path)
        quiet_since = loop.time()
        while True:
            await asyncio.sleep(0.05)
            current_count = request_count(method, path)
            now = loop.time()
            if current_count != observed_count:
                observed_count = current_count
                quiet_since = now
            elif now - quiet_since >= quiet_seconds:
                return current_count
            if now >= deadline:
                raise AssertionError(
                    f"Timed out waiting for {method} {path} request quiescence; "
                    f"observed={observed_requests!r}"
                )

    page.on("request", record_workflow_request)
    try:
        await page.goto(
            f"{base_url}/cayu/sessions/{WORKFLOW_FOCUS_SESSION_ID}",
            wait_until="networkidle",
        )
        workflow_link = page.get_by_role("link", name="Workflow", exact=True)
        await expect(workflow_link).to_be_visible()
        observed_requests.clear()
        await workflow_link.click()
        await expect(page).to_have_url(re.compile(rf"{re.escape(workflow_url)}$"))
        await expect(page.get_by_role("heading", name="Workflow", exact=True)).to_be_visible()
        await expect(
            page.get_by_text(
                "Bounded operational topology and causal-budget usage for one focus session.",
                exact=True,
            )
        ).to_be_visible()
        await expect(page.get_by_text("25 loaded child sessions", exact=True)).to_be_visible()
        await expect(page.get_by_text("25 loaded linked tasks", exact=True)).to_be_visible()
        topology_list = page.get_by_role("list", name="Loaded Workflow topology")
        await expect(
            topology_list.get_by_text(WORKFLOW_ACTIVE_SESSION_ID, exact=True)
        ).to_have_count(0)
        await expect(
            topology_list.get_by_text(WORKFLOW_FAILED_SESSION_ID, exact=True)
        ).to_be_visible()
        await expect(
            topology_list.get_by_text(WORKFLOW_INTERRUPTED_SESSION_ID, exact=True)
        ).to_be_visible()
        await expect(page.get_by_text("Causal-budget usage", exact=True)).to_be_visible()
        await expect(page.get_by_text("75", exact=True)).to_be_visible()
        await expect(page.get_by_text("<USD\u00a00.0001", exact=True).first).to_be_visible()
        await expect(page.get_by_text("<CAD\u00a00.0001", exact=True).first).to_be_visible()
        await expect(
            page.get_by_text(
                "2 returned session groups are outside the loaded topology and are not mapped to rows here.",
                exact=True,
            )
        ).to_be_visible()

        require_equal(
            request_count("POST", topology_path),
            1,
            "initial Workflow navigation must issue one coalesced topology request",
        )
        require_equal(
            request_count("POST", "/api/usage/rollup"),
            1,
            "initial Workflow navigation must issue one independent usage request",
        )
        initial_usage_body = next(
            body
            for method, path, body in observed_requests
            if method == "POST" and path == "/api/usage/rollup"
        )
        if initial_usage_body is None:
            raise AssertionError("the initial Workflow usage request body must be a JSON object")
        initial_usage_end = initial_usage_body.get("end_at")
        if not isinstance(initial_usage_end, str):
            raise AssertionError("the initial Workflow usage request must have a string end_at")
        initial_topology_body = next(
            body
            for method, path, body in observed_requests
            if method == "POST" and path == topology_path
        )
        if initial_topology_body is None:
            raise AssertionError("the Workflow topology request body must be a JSON object")
        require_equal(
            initial_topology_body.get("expanded_parent_ids"),
            [WORKFLOW_FOCUS_SESSION_ID],
            "the initial Workflow read must expand only the focus session branch",
        )
        require_equal(
            initial_topology_body.get("linked_task_session_ids"),
            [WORKFLOW_FOCUS_SESSION_ID],
            "the initial Workflow read must batch the focus task linkage",
        )

        await page.get_by_role("button", name="Load more child sessions", exact=True).click()
        await expect(page.get_by_text("27 loaded child sessions", exact=True)).to_be_visible()
        await expect(
            topology_list.get_by_text(WORKFLOW_ACTIVE_SESSION_ID, exact=True)
        ).to_be_visible()

        session_detail_link = topology_list.locator(
            f'a[href="/cayu/sessions/{WORKFLOW_ACTIVE_SESSION_ID}"]'
        )
        await expect(session_detail_link).to_have_count(1)
        task_detail_link = topology_list.locator(
            f'a[href="/cayu/tasks?q={WORKFLOW_PARENT_TASK_ID}&task_id={WORKFLOW_PARENT_TASK_ID}"]'
        )
        await expect(task_detail_link).to_have_count(1)

        slow_topology_started = asyncio.Event()
        release_slow_topology = asyncio.Event()
        slow_topology_continued = asyncio.Event()

        async def delay_superseded_topology(route, request) -> None:
            body = request.post_data_json
            expanded_ids = body.get("expanded_parent_ids", []) if isinstance(body, dict) else []
            if not (
                WORKFLOW_ACTIVE_SESSION_ID in expanded_ids
                and WORKFLOW_FAILED_SESSION_ID not in expanded_ids
            ):
                await route.continue_()
                return
            slow_topology_started.set()
            await release_slow_topology.wait()
            try:
                await route.continue_()
            except Exception:
                pass
            finally:
                slow_topology_continued.set()

        aborted_before = len(expected_query_aborts)
        await page.route(f"**{topology_path}", delay_superseded_topology)
        try:
            await page.get_by_role(
                "button",
                name=f"Expand session {WORKFLOW_ACTIVE_SESSION_ID}",
                exact=True,
            ).click()
            await asyncio.wait_for(slow_topology_started.wait(), timeout=5)
            await page.get_by_role(
                "button",
                name=f"Expand session {WORKFLOW_FAILED_SESSION_ID}",
                exact=True,
            ).click()
            await expect(page.get_by_text("refreshing", exact=True)).to_have_count(0)
        finally:
            release_slow_topology.set()
            if slow_topology_started.is_set():
                await asyncio.wait_for(slow_topology_continued.wait(), timeout=5)
            await page.unroute(f"**{topology_path}", delay_superseded_topology)
        deadline = asyncio.get_running_loop().time() + 5
        while len(expected_query_aborts) == aborted_before:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("the superseded Workflow topology request was not aborted")
            await asyncio.sleep(0.05)
        require_equal(
            len(expected_query_aborts),
            aborted_before + 1,
            "one newer Workflow expansion must abort exactly one superseded topology read",
        )

        task_expand = page.get_by_role(
            "button",
            name=f"Expand child tasks for {WORKFLOW_PARENT_TASK_ID}",
            exact=True,
        )
        await task_expand.focus()
        await page.keyboard.press("Enter")
        await expect(
            page.get_by_role(
                "button",
                name=f"Collapse child tasks for {WORKFLOW_PARENT_TASK_ID}",
                exact=True,
            )
        ).to_be_visible()
        await expect(page.get_by_text("25 loaded child tasks", exact=True)).to_be_visible()

        await page.get_by_role("button", name="Load more linked tasks", exact=True).click()
        await expect(page.get_by_text("26 loaded linked tasks", exact=True)).to_be_visible()
        await page.get_by_role("button", name="Load more child tasks", exact=True).click()
        await expect(page.get_by_text("26 loaded child tasks", exact=True)).to_be_visible()

        restored_query = parse_qs(urlsplit(page.url).query)
        require_equal(
            set(restored_query.get("expanded_session_id", [])),
            {WORKFLOW_ACTIVE_SESSION_ID, WORKFLOW_FAILED_SESSION_ID},
            "expanded Workflow session branches must round-trip through the URL",
        )
        require_equal(
            restored_query.get("expanded_task_id"),
            [WORKFLOW_PARENT_TASK_ID],
            "expanded Workflow task branches must round-trip through the URL",
        )
        require(
            not any("cursor" in key for key in restored_query),
            "opaque Workflow continuation cursors must not enter shareable URL state",
        )
        await page.get_by_role(
            "button",
            name=f"Collapse session {WORKFLOW_ACTIVE_SESSION_ID}",
            exact=True,
        ).click()
        await expect(page.get_by_text("refreshing", exact=True)).to_have_count(0)
        collapsed_query = parse_qs(urlsplit(page.url).query)
        require_equal(
            collapsed_query.get("expanded_session_id"),
            [WORKFLOW_FAILED_SESSION_ID],
            "collapsing a Workflow session must remove only that durable URL expansion",
        )
        topology_before_reload = request_count("POST", topology_path)
        usage_before_reload = request_count("POST", "/api/usage/rollup")
        await page.reload(wait_until="networkidle")
        await expect(page.get_by_text("25 loaded child sessions", exact=True)).to_be_visible()
        await expect(page.get_by_text("25 loaded child tasks", exact=True)).to_be_visible()
        require_equal(
            request_count("POST", topology_path),
            topology_before_reload + 1,
            "restoring a Workflow URL must replay one bounded first-page topology request",
        )
        require_equal(
            request_count("POST", "/api/usage/rollup"),
            usage_before_reload + 1,
            "restoring a Workflow URL must issue one independently authoritative usage request",
        )

        filter_summary = page.locator("summary").filter(has_text="Loaded-node filters")
        await filter_summary.click()
        failed_filter = page.get_by_label("failed", exact=True)
        await failed_filter.check()
        await page.get_by_role("button", name="Apply view", exact=True).click()
        await expect(page).to_have_url(re.compile(r"[?&]status=failed(?:&|$)"))
        await expect(
            topology_list.get_by_text(WORKFLOW_FAILED_SESSION_ID, exact=True)
        ).to_be_visible()
        await expect(
            topology_list.get_by_text(WORKFLOW_ACTIVE_SESSION_ID, exact=True)
        ).to_have_count(0)
        await page.reload(wait_until="networkidle")
        await expect(page.get_by_label("failed", exact=True)).to_be_checked()
        await expect(
            topology_list.get_by_text(WORKFLOW_FAILED_SESSION_ID, exact=True)
        ).to_be_visible()
        await page.get_by_role("button", name="Clear loaded-node filters", exact=True).click()
        await expect(page).not_to_have_url(re.compile(r"[?&]status="))
        await expect(
            topology_list.get_by_text(WORKFLOW_ACTIVE_SESSION_ID, exact=True)
        ).to_have_count(0)
        await page.get_by_role("button", name="Load more child sessions", exact=True).click()
        await expect(page.get_by_text("27 loaded child sessions", exact=True)).to_be_visible()
        await expect(
            topology_list.get_by_text(WORKFLOW_ACTIVE_SESSION_ID, exact=True)
        ).to_be_visible()

        await page.evaluate(
            """() => {
                Object.defineProperty(document, "visibilityState", {
                    configurable: true,
                    get: () => "hidden",
                });
                document.dispatchEvent(new Event("visibilitychange"));
            }"""
        )
        hidden_topology_count = request_count("POST", topology_path)
        hidden_usage_count = request_count("POST", "/api/usage/rollup")
        await page.wait_for_timeout(5_500)
        require_equal(
            request_count("POST", topology_path),
            hidden_topology_count,
            "a hidden Workflow page must not poll topology",
        )
        require_equal(
            request_count("POST", "/api/usage/rollup"),
            hidden_usage_count,
            "a hidden Workflow page must not poll usage",
        )

        retained_refresh_start = len(observed_requests)
        await page.evaluate(
            """() => {
                Object.defineProperty(document, "visibilityState", {
                    configurable: true,
                    get: () => "visible",
                });
                document.dispatchEvent(new Event("visibilitychange"));
            }"""
        )
        await wait_for_request_count("POST", topology_path, hidden_topology_count + 1)
        await wait_for_request_count("POST", "/api/usage/rollup", hidden_usage_count + 1)
        pre_terminal_usage_count = request_count("POST", "/api/usage/rollup")

        await session_store.append_events(
            WORKFLOW_ACTIVE_SESSION_ID,
            [
                Event(
                    id=f"{WORKFLOW_ACTIVE_SESSION_ID}-completed",
                    type=EventType.SESSION_COMPLETED,
                    session_id=WORKFLOW_ACTIVE_SESSION_ID,
                    agent_name=AGENT_NAME,
                    environment_name=WORKFLOW_ENVIRONMENT,
                )
            ],
        )
        await session_store.update_status(WORKFLOW_ACTIVE_SESSION_ID, SessionStatus.COMPLETED)
        await task_store.complete_task(WORKFLOW_PARENT_TASK_ID, {"verified": True})
        await task_store.resume_task(WORKFLOW_BLOCKED_TASK_ID)
        await task_store.complete_task(WORKFLOW_BLOCKED_TASK_ID, {"verified": True})
        await expect(
            page.get_by_text(
                "Every loaded node is terminal, so routine refresh is stopped.",
            )
        ).to_be_visible(timeout=12_000)
        terminal_topology_count = request_count("POST", topology_path)
        terminal_usage_count = await wait_for_request_quiescence(
            "POST",
            "/api/usage/rollup",
        )
        require(
            terminal_usage_count > pre_terminal_usage_count,
            "an active-to-terminal Workflow transition must reconcile usage once",
        )
        await page.wait_for_timeout(5_500)
        require_equal(
            request_count("POST", topology_path),
            terminal_topology_count,
            "a terminal loaded Workflow must stop routine topology polling",
        )
        require_equal(
            request_count("POST", "/api/usage/rollup"),
            terminal_usage_count,
            "a terminal loaded Workflow must stop routine usage polling after final reconciliation",
        )

        range_usage_count = request_count("POST", "/api/usage/rollup")
        await page.locator("#workflow-range").select_option("7d")
        await page.get_by_role("button", name="Apply view", exact=True).click()
        await expect(page).to_have_url(re.compile(r"[?&]range=7d(?:&|$)"))
        await wait_for_request_count("POST", "/api/usage/rollup", range_usage_count + 1)
        range_usage_body = next(
            body
            for method, path, body in reversed(observed_requests)
            if method == "POST" and path == "/api/usage/rollup"
        )
        if range_usage_body is None:
            raise AssertionError("the changed Workflow usage request body must be a JSON object")
        range_usage_end = range_usage_body.get("end_at")
        if not isinstance(range_usage_end, str):
            raise AssertionError("the changed Workflow usage request must have a string end_at")
        require(
            range_usage_end > initial_usage_end,
            "changing a relative Workflow range must use a fresh stable request boundary",
        )
        await page.wait_for_timeout(10)
        restored_range_usage_count = request_count("POST", "/api/usage/rollup")
        await page.locator("#workflow-range").select_option("30d")
        await page.get_by_role("button", name="Apply view", exact=True).click()
        await wait_for_request_count("POST", "/api/usage/rollup", restored_range_usage_count + 1)
        restored_range_usage_body = next(
            body
            for method, path, body in reversed(observed_requests)
            if method == "POST" and path == "/api/usage/rollup"
        )
        if restored_range_usage_body is None:
            raise AssertionError("the restored Workflow usage request body must be a JSON object")
        restored_range_usage_end = restored_range_usage_body.get("end_at")
        require(
            isinstance(restored_range_usage_end, str)
            and restored_range_usage_end > range_usage_end,
            "returning to a previous relative range must not restore its stale request boundary",
        )

        retained_refresh_bodies: list[dict[str, object]] = []
        for method, path, body in observed_requests[retained_refresh_start:]:
            if method != "POST" or path != topology_path or body is None:
                continue
            child_cursors = body.get("child_cursors")
            if not isinstance(child_cursors, dict):
                continue
            child_cursors = cast("dict[str, object]", child_cursors)
            focus_cursor = child_cursors.get(WORKFLOW_FOCUS_SESSION_ID)
            if isinstance(focus_cursor, str) and focus_cursor:
                retained_refresh_bodies.append(body)
        require(
            bool(retained_refresh_bodies),
            "routine Workflow refresh must advance the retained child-session cursor chain",
        )

        allowed_workflow_requests = {
            ("GET", "/api/contract"),
            ("POST", topology_path),
            ("POST", "/api/usage/rollup"),
        }
        unexpected_workflow_requests = [
            f"{method} {path}"
            for method, path, _body in observed_requests
            if (method, path) not in allowed_workflow_requests
        ]
        require_equal(
            unexpected_workflow_requests,
            [],
            "the Workflow route must use only its bounded topology, usage, and contract APIs",
        )

        exact_task_path = f"/api/tasks/{WORKFLOW_PARENT_TASK_ID}"
        await task_detail_link.click()
        await expect(page).to_have_url(
            re.compile(
                rf"/cayu/tasks\?q={re.escape(WORKFLOW_PARENT_TASK_ID)}&task_id={re.escape(WORKFLOW_PARENT_TASK_ID)}$"
            )
        )
        await expect(page.get_by_role("heading", name="Tasks", exact=True)).to_be_visible()
        task_details_header = page.get_by_text("Task Details", exact=True).locator("..")
        await expect(
            task_details_header.get_by_text(WORKFLOW_PARENT_TASK_ID, exact=True)
        ).to_be_visible()
        require(
            any(
                method == "GET" and path == exact_task_path
                for method, path, _body in observed_requests
            ),
            "a Workflow task deep link must load the exact task detail resource",
        )

        await page.get_by_role("button", name="Clear", exact=True).click()
        fallback_source_row = page.get_by_role("row").filter(has_text=WORKFLOW_BLOCKED_TASK_ID)
        await expect(fallback_source_row).to_have_count(1)
        await fallback_source_row.click()
        await expect(page).to_have_url(
            re.compile(rf"[?&]task_id={re.escape(WORKFLOW_BLOCKED_TASK_ID)}(?:&|$)")
        )
        await page.get_by_label("Filter by task status", exact=True).select_option("failed")
        await expect(page).not_to_have_url(
            re.compile(rf"[?&]task_id={re.escape(WORKFLOW_BLOCKED_TASK_ID)}(?:&|$)")
        )
        fallback_task_ids = parse_qs(urlsplit(page.url).query).get("task_id", [])
        require_equal(
            len(fallback_task_ids),
            1,
            "an automatic task fallback must keep one selected task in the URL",
        )
        await expect(
            task_details_header.get_by_text(fallback_task_ids[0], exact=True)
        ).to_be_visible()

        usage_without_pricing = False

        async def observe_workflow_usage_without_pricing(route, request) -> None:
            nonlocal usage_without_pricing
            body = request.post_data_json
            if (
                isinstance(body, dict)
                and body.get("session_filter", {}).get("causal_budget_id") == WORKFLOW_BUDGET_ID
            ):
                require(
                    body.get("pricing") is None,
                    "Workflow usage without a dashboard price book must omit pricing inputs",
                )
                usage_without_pricing = True
            await route.continue_()

        await page.route(f"{workflow_url}*", _serve_dashboard_without_pricebook)
        await page.route("**/api/usage/rollup", observe_workflow_usage_without_pricing)
        try:
            await page.goto(workflow_url, wait_until="networkidle")
            await expect(
                page.get_by_text(
                    "No dashboard price book is configured. Usage remains available; cost is unavailable rather than zero.",
                    exact=False,
                )
            ).to_be_visible()
            await expect(
                page.get_by_text(
                    'dashboard_config={"priceBook": default_price_book()}',
                    exact=True,
                )
            ).to_be_visible()
            await expect(page.get_by_text("mount_cayu", exact=True)).to_be_visible()
            require(
                usage_without_pricing,
                "a no-pricing Workflow must still issue its bounded usage request",
            )
        finally:
            await page.unroute("**/api/usage/rollup", observe_workflow_usage_without_pricing)
            await page.unroute(f"{workflow_url}*", _serve_dashboard_without_pricebook)

        workflow_topology_requested = False

        def observe_unavailable_workflow_request(request: Request) -> None:
            nonlocal workflow_topology_requested
            if request.method == "POST" and urlsplit(request.url).path == topology_path:
                workflow_topology_requested = True

        async def serve_workflow_unavailable_contract(route) -> None:
            response = await route.fetch()
            body = await response.json()
            body["capabilities"]["surfaces"]["workflow"] = {
                "configured": False,
                "read": {
                    "enabled": False,
                    "unavailable_reason": "unsupported",
                },
                "mutate": {
                    "enabled": False,
                    "unavailable_reason": "unsupported",
                },
            }
            await route.fulfill(response=response, json=body)

        page.on("request", observe_unavailable_workflow_request)
        await page.route("**/api/contract", serve_workflow_unavailable_contract)
        try:
            await page.goto(workflow_url, wait_until="networkidle")
            await expect(page.get_by_test_id("dashboard-capability-unavailable")).to_contain_text(
                "Workflow is unavailable"
            )
            require(
                not workflow_topology_requested,
                "an unavailable Workflow route must not probe the topology endpoint",
            )
        finally:
            await page.unroute("**/api/contract", serve_workflow_unavailable_contract)
            page.remove_listener("request", observe_unavailable_workflow_request)
    finally:
        page.remove_listener("request", record_workflow_request)


async def _exercise_contract_version_gate(page: Page, base_url: str) -> None:
    api_requests_beyond_contract: list[str] = []
    previous_contract_version = str(int(SERVER_CONTRACT_VERSION) - 1)

    async def serve_incompatible_contract(route, request) -> None:
        path = urlsplit(request.url).path
        if path == "/api/contract":
            response = await route.fetch()
            body = await response.json()
            # Reconstruct the immediately preceding valid response. It must be
            # rejected before navigation evaluates any capability requirement.
            body["contract_version"] = previous_contract_version
            body["versioning"]["contract_version"] = previous_contract_version
            await route.fulfill(
                response=response,
                json=body,
            )
            return
        api_requests_beyond_contract.append(f"{request.method} {path}")
        await route.fulfill(
            status=500,
            headers={"content-type": "application/json"},
            body=json.dumps({"detail": "API route rendered before contract compatibility."}),
        )

    await page.route("**/api/**", serve_incompatible_contract)
    try:
        await page.goto(f"{base_url}/cayu/usage", wait_until="networkidle")
        await expect(page.get_by_test_id("dashboard-contract-gate")).to_contain_text(
            f"Dashboard expects CAYU server contract v{SERVER_CONTRACT_VERSION}, "
            f"but the server reports v{previous_contract_version}."
        )
        await expect(page.get_by_role("heading", name="Usage", exact=True)).to_have_count(0)
        require_equal(
            api_requests_beyond_contract,
            [],
            "a previous valid contract must not start route-specific API requests",
        )
    finally:
        await page.unroute("**/api/**", serve_incompatible_contract)


async def _serve_dashboard_without_pricebook(route) -> None:
    response = await route.fetch()
    html = await response.text()
    marker = "window.__CAYU_DASHBOARD_CONFIG__="
    config_start = html.index(marker) + len(marker)
    config_end = html.index(";</script>", config_start)
    config = json.loads(html[config_start:config_end])
    require(
        isinstance(config, dict) and "priceBook" in config,
        "the no-pricing browser scenario requires a configured price book to remove",
    )
    config.pop("priceBook", None)
    config_json = json.dumps(config, separators=(",", ":")).replace("<", "\\u003c")
    await route.fulfill(
        response=response,
        body=f"{html[:config_start]}{config_json}{html[config_end:]}",
    )


async def _exercise_capability_contract(page: Page, base_url: str) -> None:
    observed_requests: list[str] = []

    async def serve_without_task_or_artifact_surface(route) -> None:
        response = await route.fetch()
        body = await response.json()
        body["capabilities"]["surfaces"]["tasks"] = {
            "configured": False,
            "read": {
                "enabled": False,
                "unavailable_reason": "not_configured",
            },
            "mutate": {
                "enabled": False,
                "unavailable_reason": "not_configured",
            },
        }
        body["capabilities"]["surfaces"]["artifacts"] = {
            "configured": False,
            "read": {
                "enabled": False,
                "unavailable_reason": "not_configured",
            },
            "mutate": {
                "enabled": False,
                "unavailable_reason": "not_configured",
            },
        }
        await route.fulfill(response=response, json=body)

    def record_api_request(request) -> None:
        path = urlsplit(request.url).path
        if path.startswith("/api/"):
            observed_requests.append(f"{request.method} {path}")

    await page.route("**/api/contract", serve_without_task_or_artifact_surface)
    page.on("request", record_api_request)
    try:
        await page.goto(f"{base_url}/cayu/tasks", wait_until="networkidle")
        await expect(page.get_by_test_id("dashboard-capability-unavailable")).to_contain_text(
            "Tasks is unavailable"
        )
        await expect(page.get_by_role("link", name="Tasks", exact=True)).to_have_count(0)
        await expect(page.get_by_role("link", name="Knowledge", exact=True)).to_have_count(0)
        await expect(page.get_by_role("link", name="Artifacts", exact=True)).to_have_count(0)
        await expect(page.get_by_role("link", name="Usage", exact=True)).to_be_visible()
        await expect(page.get_by_role("link", name="System", exact=True)).to_be_visible()
        require(
            not any(request.startswith("GET /api/tasks") for request in observed_requests),
            f"an unavailable direct task route issued a task request: {observed_requests!r}",
        )

        observed_requests.clear()
        await page.goto(f"{base_url}/cayu/agents", wait_until="networkidle")
        await expect(page.get_by_role("heading", name="Agents", exact=True)).to_be_visible()
        await expect(page.get_by_text("Tasks are unavailable.", exact=False)).to_be_visible()
        require(
            not any(request.startswith("GET /api/tasks") for request in observed_requests),
            f"the agent page probed unavailable tasks: {observed_requests!r}",
        )

        observed_requests.clear()
        await page.goto(f"{base_url}/cayu/sessions/{SESSION_ID}", wait_until="networkidle")
        await expect(page.get_by_role("heading", name=SESSION_ID)).to_be_visible()
        require(
            not any(request.startswith("GET /api/artifacts") for request in observed_requests),
            f"the session page probed unavailable artifacts: {observed_requests!r}",
        )
    finally:
        page.remove_listener("request", record_api_request)
        await page.unroute("**/api/contract", serve_without_task_or_artifact_surface)

    async def serve_without_evals_configuration(route) -> None:
        response = await route.fetch()
        body = await response.json()
        body["capabilities"]["surfaces"]["evals"] = {
            "configured": False,
            "read": {
                "enabled": False,
                "unavailable_reason": "not_configured",
            },
            "mutate": {
                "enabled": False,
                "unavailable_reason": "not_configured",
            },
        }
        body["capabilities"]["evals_readiness"] = {
            "captured_evaluation": {
                "state": "gated",
                "reason_code": "evaluation_promotion_not_configured",
            },
            "catalog_read": {
                "state": "gated",
                "reason_code": "eval_store_not_configured",
            },
            "catalog_write": {
                "state": "gated",
                "reason_code": "eval_store_not_configured",
            },
            "captured_result_persistence": {
                "state": "gated",
                "reason_code": "eval_store_not_configured",
            },
            "scenario_conversion": {
                "state": "unsupported",
                "reason_code": "scenario_v2_not_available",
            },
            "fresh_launch": {
                "state": "gated",
                "reason_code": "eval_target_not_configured",
            },
            "cancellation": {
                "state": "gated",
                "reason_code": "eval_store_not_configured",
            },
            "comparison": {
                "state": "gated",
                "reason_code": "eval_store_not_configured",
            },
            "reports": {
                "state": "gated",
                "reason_code": "eval_store_not_configured",
            },
        }
        await route.fulfill(response=response, json=body)

    observed_evals_requests: list[str] = []

    def record_evals_request(request) -> None:
        path = urlsplit(request.url).path
        if path.startswith("/api/evals"):
            observed_evals_requests.append(f"{request.method} {path}")

    await page.route("**/api/contract", serve_without_evals_configuration)
    page.on("request", record_evals_request)
    try:
        await page.goto(f"{base_url}/cayu/evals", wait_until="networkidle")
        await expect(page.get_by_role("heading", name="Evals", exact=True)).to_be_visible()
        await expect(page.get_by_role("link", name="Evals", exact=True)).to_be_visible()
        await expect(
            page.get_by_text("The Evals catalog is not ready yet", exact=True)
        ).to_be_visible()
        await expect(
            page.get_by_text(
                "Scenario-v2 conversion is unavailable in this deployment.",
                exact=True,
            )
        ).to_be_visible()
        require(
            not observed_evals_requests,
            f"the unconfigured Evals shell probed absent endpoints: {observed_evals_requests!r}",
        )
    finally:
        page.remove_listener("request", record_evals_request)
        await page.unroute("**/api/contract", serve_without_evals_configuration)

    async def serve_usage_without_pricing_contract(route) -> None:
        response = await route.fetch()
        body = await response.json()
        body["capabilities"]["surfaces"]["pricing"] = {
            "configured": False,
            "read": {
                "enabled": False,
                "unavailable_reason": "not_configured",
            },
            "mutate": {
                "enabled": False,
                "unavailable_reason": "unsupported",
            },
        }
        await route.fulfill(response=response, json=body)

    usage_request_without_pricing = False

    async def observe_usage_without_pricing(route, request) -> None:
        nonlocal usage_request_without_pricing
        body = request.post_data_json
        require(
            isinstance(body, dict) and body.get("pricing") is None,
            "usage without a dashboard price book must not submit pricing inputs",
        )
        usage_request_without_pricing = True
        await route.continue_()

    await page.route("**/api/contract", serve_usage_without_pricing_contract)
    await page.route(f"{base_url}/cayu/usage", _serve_dashboard_without_pricebook)
    await page.route("**/api/usage/rollup", observe_usage_without_pricing)
    try:
        await page.goto(f"{base_url}/cayu/usage", wait_until="networkidle")
        await expect(page.get_by_role("heading", name="Usage", exact=True)).to_be_visible()
        await expect(page.get_by_role("link", name="Usage", exact=True)).to_be_visible()
        await expect(
            page.get_by_text(
                "Usage remains available, but cost is unavailable rather than displayed as zero.",
                exact=False,
            )
        ).to_be_visible()
        await expect(
            page.get_by_text(
                'dashboard_config={"priceBook": default_price_book()}',
                exact=True,
            )
        ).to_be_visible()
        await expect(
            page.get_by_text("complete application-owned PriceBook", exact=False)
        ).to_be_visible()
        require(
            usage_request_without_pricing,
            "usage without a dashboard price book must issue one aggregate request",
        )
    finally:
        await page.unroute("**/api/usage/rollup", observe_usage_without_pricing)
        await page.unroute(f"{base_url}/cayu/usage", _serve_dashboard_without_pricebook)
        await page.unroute("**/api/contract", serve_usage_without_pricing_contract)

    async def serve_read_only_contract(route) -> None:
        response = await route.fetch()
        body = await response.json()
        mutations = body["capabilities"]["mutations"]
        for mutation_name in mutations:
            mutations[mutation_name] = {
                "enabled": False,
                "unavailable_reason": "unsupported",
            }
        await route.fulfill(response=response, json=body)

    await page.route("**/api/contract", serve_read_only_contract)
    try:
        await page.goto(f"{base_url}/cayu/", wait_until="networkidle")
        await expect(page.get_by_role("button", name="New Run", exact=True)).to_be_disabled()
        await expect(page.get_by_test_id("overview-session-execution-unavailable")).to_be_visible()

        await page.goto(f"{base_url}/cayu/sessions", wait_until="networkidle")
        await expect(page.get_by_role("button", name="New Run", exact=True)).to_be_disabled()
        await expect(page.get_by_test_id("session-execution-unavailable")).to_be_visible()

        await page.goto(
            f"{base_url}/cayu/sessions/{INTERRUPT_SESSION_ID}", wait_until="networkidle"
        )
        interrupt = page.get_by_role("button", name="Interrupt session", exact=True)
        await expect(interrupt).to_be_disabled()
        await expect(page.get_by_test_id("session-interruption-unavailable")).to_be_visible()
        await expect(page.get_by_role("button", name="Edit labels", exact=True)).to_be_disabled()
        await expect(page.get_by_test_id("annotations-unavailable")).to_be_visible()
        await expect(page.get_by_role("link", name="New Run", exact=True)).to_have_count(0)

        await page.goto(f"{base_url}/cayu/sessions/{APPROVAL_SESSION_ID}", wait_until="networkidle")
        await expect(page.get_by_role("button", name="Approve", exact=True)).to_be_disabled()
        await expect(page.get_by_role("button", name="Deny", exact=True)).to_be_disabled()
        await expect(page.get_by_test_id("pending-action-unavailable")).to_be_visible()

        await page.goto(f"{base_url}/cayu/run", wait_until="networkidle")
        await expect(page.get_by_test_id("dashboard-capability-unavailable")).to_contain_text(
            "New Run is unavailable"
        )
        await expect(page.get_by_role("textbox")).to_have_count(0)
    finally:
        await page.unroute("**/api/contract", serve_read_only_contract)


async def _exercise_system_page(page: Page, base_url: str) -> None:
    diagnostics_requests = 0

    def count_diagnostics_requests(request) -> None:
        nonlocal diagnostics_requests
        if urlsplit(request.url).path == "/api/system/diagnostics":
            diagnostics_requests += 1

    page.on("request", count_diagnostics_requests)
    try:
        await page.goto(f"{base_url}/cayu/system", wait_until="networkidle")
        await expect(page.get_by_role("heading", name="System", exact=True)).to_be_visible()
        await expect(page.get_by_test_id("system-snapshot-scope")).to_contain_text(
            "does not probe databases, workers, networks, or external services"
        )
        await expect(page.get_by_text("Server contract", exact=True)).to_be_visible()
        await expect(page.get_by_text(f"v{SERVER_CONTRACT_VERSION}", exact=True)).to_be_visible()
        require_equal(diagnostics_requests, 1, "the System page must load one initial snapshot")

        await page.wait_for_timeout(5_500)
        require_equal(
            diagnostics_requests,
            1,
            "the System page must not refresh diagnostics in the background",
        )
        await page.get_by_role("button", name="Refresh snapshot", exact=True).click()
        await expect(
            page.get_by_role("button", name="Refresh snapshot", exact=True)
        ).to_be_enabled()
        require_equal(
            diagnostics_requests,
            2,
            "manual System refresh must issue exactly one additional snapshot request",
        )
    finally:
        page.remove_listener("request", count_diagnostics_requests)


async def _exercise_operational_scope(page: Page, base_url: str) -> None:
    snapshot_path = "**/api/operations/snapshot"
    snapshot_started = asyncio.Event()
    release_snapshot = asyncio.Event()
    snapshot_continued = asyncio.Event()

    async def delay_operational_snapshot(route) -> None:
        snapshot_started.set()
        await release_snapshot.wait()
        try:
            await route.continue_()
        finally:
            snapshot_continued.set()

    await page.route(snapshot_path, delay_operational_snapshot)
    try:
        await page.goto(f"{base_url}/cayu/", wait_until="domcontentloaded")
        await asyncio.wait_for(snapshot_started.wait(), timeout=5)
        active_metric = page.get_by_text("Active Sessions", exact=True).locator("..").locator("..")
        await expect(active_metric.get_by_text("—", exact=True)).to_be_visible()
        await expect(
            active_metric.get_by_text(re.compile(r"Loading the operational session snapshot\."))
        ).to_be_visible()
        await expect(
            page.get_by_text("Loading configured-store metrics...", exact=True)
        ).to_be_visible()
    finally:
        release_snapshot.set()
        if snapshot_started.is_set():
            await asyncio.wait_for(snapshot_continued.wait(), timeout=5)
        await page.unroute(snapshot_path, delay_operational_snapshot)

    await page.wait_for_load_state("networkidle")
    await expect(page.get_by_role("heading", name="Dashboard", exact=True)).to_be_visible()
    operational_scope = page.get_by_test_id("overview-operational-scope")
    await expect(operational_scope).to_be_visible()
    await expect(
        operational_scope.get_by_text("Configured-store operational snapshot", exact=True)
    ).to_be_visible()
    await expect(
        operational_scope.get_by_text(
            re.compile(
                r"Exact session counts as of .* independent, not one cross-store atomic read\."
            )
        )
    ).to_be_visible()
    recent_scope = page.get_by_test_id("overview-sample-scope")
    await expect(recent_scope).to_be_visible()
    await expect(
        recent_scope.get_by_text("Recent lists — bounded drill-down", exact=True)
    ).to_be_visible()
    await expect(
        recent_scope.get_by_text(
            re.compile(r"latest 25 of \d+ sessions by updated time \(25-session limit\)")
        )
    ).to_be_visible()
    for label in (
        "Active Sessions",
        "Completed Sessions",
        "Failed Sessions",
        "Tasks Needing Attention",
    ):
        await expect(page.get_by_text(label, exact=True)).to_be_visible()
    await expect(page.get_by_text("Session Tokens", exact=True)).to_have_count(0)
    await expect(page.get_by_text("Active Work", exact=True)).to_have_count(0)
    await expect(page.get_by_text("Recent Needs Attention", exact=True)).to_be_visible()

    await page.goto(f"{base_url}/cayu/usage", wait_until="networkidle")
    await expect(page.get_by_role("heading", name="Usage", exact=True)).to_be_visible()
    await expect(
        page.get_by_text(
            "Store-native activity, token, and cost totals over an explicit event-time window.",
            exact=True,
        )
    ).to_be_visible()
    usage_scope = page.get_by_test_id("usage-aggregate-scope")
    await expect(usage_scope).to_be_visible()
    await expect(usage_scope.get_by_text("Authoritative store rollup", exact=True)).to_be_visible()
    await expect(
        usage_scope.get_by_text(
            re.compile(
                r"Events from .* inclusive to .* exclusive, as of .* event time is the usage basis\."
            )
        )
    ).to_be_visible()
    for label in (
        "Matching Sessions",
        "Sessions with Activity",
        "Tokens",
        "Model Steps",
        "Tool Calls",
        "Estimated Cost",
    ):
        await expect(page.get_by_text(label, exact=True)).to_be_visible()
    await expect(
        page.get_by_text(
            "Price book dashboard-browser-contract, generated 2026-07-21; "
            "currencies are never combined.",
            exact=True,
        )
    ).to_be_visible()
    bedrock_breakdown = page.get_by_test_id("usage-billing-breakdown")
    await expect(
        bedrock_breakdown.get_by_text("Billing Identity Breakdown", exact=True)
    ).to_be_visible()
    identity_counts = bedrock_breakdown.get_by_test_id("usage-billing-identity-counts")
    await expect(
        identity_counts.get_by_text("Steps with commercial identity", exact=True)
    ).to_be_visible()
    await expect(identity_counts.get_by_text("Evaluated steps", exact=True)).to_be_visible()
    await expect(identity_counts.get_by_text("1", exact=True)).to_be_visible()
    await expect(identity_counts.get_by_text("6", exact=True)).to_be_visible()
    await expect(identity_counts.get_by_text("Without identity", exact=True)).to_have_count(0)
    await expect(bedrock_breakdown.get_by_test_id("usage-billing-identity-gap")).to_have_count(0)
    bedrock_row = bedrock_breakdown.get_by_role("row").filter(
        has_text="global.anthropic.claude-sonnet-4-6"
    )
    await expect(bedrock_row).to_contain_text("us-east-1")
    await expect(bedrock_row).to_contain_text("global")
    await expect(bedrock_row).to_contain_text("default / default")
    await expect(bedrock_row).to_contain_text("bedrock/global.anthropic.claude-sonnet-4-6")

    no_identity_query = urlencode({"provider_name": PROVIDER_NAME})
    await page.goto(f"{base_url}/cayu/usage?{no_identity_query}", wait_until="networkidle")
    no_identity_breakdown = page.get_by_test_id("usage-billing-breakdown")
    await expect(no_identity_breakdown).to_have_count(0)
    await expect(page.get_by_text(re.compile(r"Billing identity is missing for"))).to_have_count(0)

    await page.goto(f"{base_url}/cayu/usage", wait_until="networkidle")

    await page.get_by_text("Session filters", exact=True).click()
    await page.get_by_label("Agent", exact=True).fill(AGENT_NAME)
    await page.locator("#usage-label-filter").fill("stage=initial")
    await page.get_by_role("button", name="Apply", exact=True).click()
    await expect(page).to_have_url(re.compile(r"[?&]agent_name=dashboard-contract-agent(?:&|$)"))
    filtered_query = parse_qs(urlsplit(page.url).query)
    require_equal(
        filtered_query.get("label"),
        ["stage=initial"],
        "usage URL state must preserve repeated exact label parameters",
    )
    await expect(page.get_by_test_id("usage-aggregate-scope")).to_be_visible()
    matching_metric = page.get_by_text("Matching Sessions", exact=True).locator("..").locator("..")
    await expect(matching_metric.get_by_text("1", exact=True)).to_be_visible()
    await page.reload(wait_until="networkidle")
    await expect(page.get_by_label("Agent", exact=True)).to_have_value(AGENT_NAME)
    await expect(page.locator("#usage-label-filter")).to_have_value("stage=initial")

    await page.get_by_role("button", name="Clear filters", exact=True).click()
    await expect(page).not_to_have_url(re.compile(r"[?&](?:agent_name|label)="))
    await page.locator("#usage-range").select_option("7d")
    await page.get_by_role("button", name="Apply", exact=True).click()
    await expect(page).to_have_url(re.compile(r"[?&]range=7d(?:&|$)"))
    await expect(page.get_by_test_id("usage-aggregate-scope")).to_be_visible()

    precise_start = "2026-07-01T00:00:59.999123Z"
    offset_end = "2026-07-02T08:30:00.123456+06:00"
    custom_query = urlencode(
        {
            "range": "custom",
            "start_at": precise_start,
            "end_at": offset_end,
        }
    )
    await page.goto(f"{base_url}/cayu/usage?{custom_query}", wait_until="networkidle")
    await expect(page.get_by_label("Start (UTC)", exact=True)).to_have_value(
        "2026-07-01T00:00:59.999"
    )
    await expect(page.get_by_label("End (UTC)", exact=True)).to_have_value(
        "2026-07-02T02:30:00.123"
    )
    await page.get_by_text("Session filters", exact=True).click()
    await page.get_by_label("Agent", exact=True).fill(AGENT_NAME)
    async with page.expect_request(
        lambda request: (
            request.method == "POST" and urlsplit(request.url).path == "/api/usage/rollup"
        )
    ) as custom_request_info:
        await page.get_by_role("button", name="Apply", exact=True).click()
    custom_request = await custom_request_info.value
    custom_request_body = custom_request.post_data_json
    if not isinstance(custom_request_body, dict):
        raise AssertionError("usage rollup POST body must be an object")
    require_equal(
        custom_request_body.get("start_at"),
        precise_start,
        "the usage POST must preserve the exact custom start boundary",
    )
    require_equal(
        custom_request_body.get("end_at"),
        offset_end,
        "the usage POST must preserve the exact custom end boundary",
    )
    custom_filtered_query = parse_qs(urlsplit(page.url).query)
    require_equal(
        custom_filtered_query.get("start_at"),
        [precise_start],
        "an unrelated usage filter must preserve the exact custom start boundary",
    )
    require_equal(
        custom_filtered_query.get("end_at"),
        [offset_end],
        "an unrelated usage filter must preserve the exact custom end boundary",
    )
    await expect(page.get_by_test_id("usage-aggregate-scope")).to_be_visible()

    await page.goto(f"{base_url}/cayu/usage", wait_until="networkidle")
    await expect(page.get_by_role("heading", name="Usage", exact=True)).to_be_visible()

    usage_path = "**/api/usage/rollup"
    slow_usage_started = asyncio.Event()
    release_slow_usage = asyncio.Event()
    slow_usage_continued = asyncio.Event()

    async def delay_superseded_usage(route, request) -> None:
        body = request.post_data_json
        agent_name = body.get("session_filter", {}).get("agent_name") if body else None
        if agent_name != SLOW_USAGE_AGENT:
            await route.continue_()
            return
        slow_usage_started.set()
        await release_slow_usage.wait()
        try:
            await route.continue_()
        except Exception:
            # The AbortSignal deliberately owns the superseded request. Chromium may
            # dispose its intercepted route before this test releases the server path.
            pass
        finally:
            slow_usage_continued.set()

    await page.route(usage_path, delay_superseded_usage)
    try:
        await page.get_by_text("Session filters", exact=True).click()
        await page.get_by_label("Agent", exact=True).fill(SLOW_USAGE_AGENT)
        await page.get_by_role("button", name="Apply", exact=True).click()
        await asyncio.wait_for(slow_usage_started.wait(), timeout=5)
        await page.get_by_label("Agent", exact=True).fill(AGENT_NAME)
        await page.get_by_role("button", name="Apply", exact=True).click()
        await expect(page).to_have_url(
            re.compile(r"[?&]agent_name=dashboard-contract-agent(?:&|$)")
        )
        await expect(page.get_by_test_id("usage-aggregate-scope")).to_be_visible()
        await expect(
            page.get_by_text("Loading the bounded usage rollup...", exact=True)
        ).to_have_count(0)
    finally:
        release_slow_usage.set()
        if slow_usage_started.is_set():
            await asyncio.wait_for(slow_usage_continued.wait(), timeout=5)
        await page.unroute(usage_path, delay_superseded_usage)

    async def reject_usage_refresh(route) -> None:
        await route.fulfill(
            status=501,
            headers={"content-type": "application/json"},
            body=json.dumps({"detail": "Injected aggregate refresh failure."}),
        )

    await page.route(usage_path, reject_usage_refresh)
    try:
        await page.get_by_role("button", name="Refresh window", exact=True).click()
        await expect(page.get_by_test_id("usage-retained-error")).to_be_visible()
        await expect(page.get_by_test_id("usage-aggregate-scope")).to_be_visible()
    finally:
        await page.unroute(usage_path, reject_usage_refresh)

    await page.get_by_role("button", name="Retry", exact=True).click()
    await expect(page.get_by_test_id("usage-retained-error")).to_have_count(0)
    await expect(page.get_by_test_id("usage-aggregate-scope")).to_be_visible()


async def _exercise_session_annotations(page: Page) -> None:
    labels_path = f"**/api/sessions/{SESSION_ID}/labels"
    metadata_path = f"**/api/sessions/{SESSION_ID}/metadata"
    label_saves = 0
    metadata_saves = 0

    async def reject_first_label_save(route, request) -> None:
        nonlocal label_saves
        if request.method == "PATCH":
            label_saves += 1
            if label_saves == 1:
                await route.fulfill(
                    status=422,
                    headers={"content-type": "application/json"},
                    body=json.dumps({"detail": "Injected session label validation failure."}),
                )
                return
        await route.continue_()

    async def count_metadata_saves(route, request) -> None:
        nonlocal metadata_saves
        if request.method == "PATCH":
            metadata_saves += 1
        await route.continue_()

    await page.route(labels_path, reject_first_label_save)
    await page.route(metadata_path, count_metadata_saves)
    try:
        await expect(page.get_by_text("stage=initial", exact=True)).to_be_visible()
        await expect(page.locator("pre").filter(has_text="dashboard-customer")).to_be_visible()
        await expect(
            page.locator("pre").filter(has_text="runtime-dashboard-marker")
        ).to_be_visible()

        await page.get_by_role("button", name="Edit labels", exact=True).click()
        labels_editor = page.get_by_label("Session labels JSON")
        pending_labels = '{"stage":"draft-preserved-across-metadata-save"}'
        await labels_editor.fill(pending_labels)

        # A confirmed metadata response replaces the detail cache. The independently
        # active labels editor must retain its draft across that parent re-render.
        await page.get_by_role("button", name="Edit metadata", exact=True).click()
        metadata_editor = page.get_by_label("Session metadata JSON")
        await metadata_editor.fill(
            '{"customer":{"id":"dashboard-customer"},"marker":"metadata-browser-marker"}'
        )
        await page.get_by_role("button", name="Save metadata", exact=True).click()
        await expect(page.locator("pre").filter(has_text="metadata-browser-marker")).to_be_visible()
        await expect(labels_editor).to_have_value(pending_labels)
        require_equal(metadata_saves, 1, "saving metadata must send exactly one PATCH")

        await page.get_by_role("button", name="Cancel label editing", exact=True).click()
        await expect(page.get_by_role("button", name="Edit labels", exact=True)).to_be_focused()
        await page.get_by_role("button", name="Edit labels", exact=True).click()
        labels_editor = page.get_by_label("Session labels JSON")
        await expect(labels_editor).to_have_value(re.compile(r'"stage": "initial"'))

        await labels_editor.fill('{"stage":1}')
        await page.get_by_role("button", name="Save labels", exact=True).click()
        await expect(page.get_by_text('Label "stage" must have a string value.')).to_be_visible()
        require_equal(label_saves, 0, "client-invalid labels must not reach the server")

        valid_labels = '{"stage":"review","tenant":"acme"}'
        await labels_editor.fill(valid_labels)
        await page.get_by_role("button", name="Save labels", exact=True).click()
        await expect(
            page.get_by_text("Injected session label validation failure.", exact=True)
        ).to_be_visible()
        await expect(labels_editor).to_have_value(valid_labels)
        await page.get_by_role("button", name="Save labels", exact=True).evaluate(
            "button => { button.click(); button.click(); }"
        )
        await expect(page.get_by_text("stage=review", exact=True)).to_be_visible()
        await expect(page.get_by_text("tenant=acme", exact=True)).to_be_visible()
        require_equal(label_saves, 2, "duplicate save activation must send one retry PATCH")

        await page.get_by_role("button", name="Edit metadata", exact=True).click()
        metadata_editor = page.get_by_label("Session metadata JSON")
        await metadata_editor.fill('{"cancelled":true}')
        cancel_metadata = page.get_by_role("button", name="Cancel metadata editing", exact=True)
        await cancel_metadata.click()
        await expect(page.get_by_role("button", name="Edit metadata", exact=True)).to_be_focused()
        require_equal(metadata_saves, 1, "cancelling metadata editing must not send a PATCH")

        await page.get_by_role("button", name="Edit labels", exact=True).click()
        await page.get_by_label("Session labels JSON").fill("{}")
        await page.get_by_role("button", name="Save labels", exact=True).click()
        await expect(page.get_by_text("No labels.", exact=True)).to_be_visible()

        await page.reload(wait_until="networkidle")
        await expect(page.get_by_text("No labels.", exact=True)).to_be_visible()
        await expect(page.locator("pre").filter(has_text="metadata-browser-marker")).to_be_visible()
        await expect(
            page.locator("pre").filter(has_text="runtime-dashboard-marker")
        ).to_be_visible()
    finally:
        await page.unroute(labels_path, reject_first_label_save)
        await page.unroute(metadata_path, count_metadata_saves)


async def _exercise_session_discovery(
    page: Page,
    faults: MutationDisconnectFaults,
) -> None:
    await page.get_by_text("Advanced filters", exact=True).click()
    await page.get_by_label("Agent", exact=True).fill(AGENT_NAME)
    await page.get_by_label("Environment", exact=True).fill(DISCOVERY_ENVIRONMENT)
    await page.get_by_label("Provider", exact=True).fill(PROVIDER_NAME)
    await page.get_by_label("Model", exact=True).fill(MODEL_NAME)
    await page.get_by_label("Parent session", exact=True).fill(SESSION_ID)
    await page.get_by_label("Causal budget", exact=True).fill(DISCOVERY_BUDGET_ID)
    await page.locator("#session-label-filter").fill("tenant=acme\ntier=critical")
    await page.locator("#session-selector-filter").fill("region in (us,eu)\n!archived")
    await page.get_by_role("button", name="Apply filters").click()

    advanced_query = parse_qs(urlsplit(page.url).query)
    for field in (
        "agent_name",
        "environment_name",
        "provider_name",
        "model",
        "parent_session_id",
        "causal_budget_id",
        "label",
        "label_selector",
    ):
        require(field in advanced_query, f"session discovery URL must preserve {field}")
    require_equal(
        advanced_query["label"],
        ["tenant=acme", "tier=critical"],
        "session discovery URL must preserve repeated exact labels",
    )
    require_equal(
        advanced_query["label_selector"],
        ["region in (us,eu)", "!archived"],
        "session discovery URL must preserve repeated label selectors",
    )
    require("cursor" not in advanced_query, "changing session filters must reset the cursor")
    await expect(page.get_by_role("link", name=FILTERED_SESSION_ID)).to_be_visible()
    require_equal(
        await page.locator("tbody tr").count(),
        100,
        "combined server session filters must retain one bounded first page",
    )
    await expect(page.get_by_role("link", name=f"{PAGINATED_SESSION_PREFIX}-000")).to_have_count(0)

    next_page = page.get_by_role("button", name="Next page")
    await expect(next_page).to_be_visible()
    await next_page.click()
    await expect(page).to_have_url(re.compile(r"[?&]cursor="))
    await expect(page.get_by_role("link", name=f"{PAGINATED_SESSION_PREFIX}-000")).to_be_visible()

    filtered_later_page_url = page.url
    await page.reload(wait_until="networkidle")
    await expect(page).to_have_url(filtered_later_page_url)
    await expect(page.get_by_role("link", name=f"{PAGINATED_SESSION_PREFIX}-000")).to_be_visible()

    await page.get_by_role("button", name="First page").click()
    await expect(page).not_to_have_url(re.compile(r"[?&]cursor="))
    await expect(page.get_by_role("link", name=FILTERED_SESSION_ID)).to_be_visible()
    filtered_first_page_url = page.url
    await page.go_back()
    await expect(page).to_have_url(filtered_later_page_url)
    await expect(page.get_by_role("link", name=f"{PAGINATED_SESSION_PREFIX}-000")).to_be_visible()
    await page.go_forward()
    await expect(page).to_have_url(filtered_first_page_url)
    await expect(page.get_by_role("link", name=FILTERED_SESSION_ID)).to_be_visible()

    await page.get_by_role("button", name="Clear filters").click()
    await expect(page).to_have_url(re.compile(r"/cayu/sessions$"))

    search = page.get_by_label("Search sessions")
    await search.fill(SLOW_SESSION_QUERY)
    await asyncio.wait_for(faults.slow_summary_started.wait(), timeout=5)
    await search.fill(FILTERED_SESSION_ID)
    await expect(page.get_by_role("link", name=FILTERED_SESSION_ID)).to_be_visible()
    await asyncio.sleep(1.1)
    await expect(page.get_by_role("link", name=FILTERED_SESSION_ID)).to_be_visible()
    require_equal(
        await page.locator("tbody tr").count(),
        1,
        "a superseded session response must not replace the active query",
    )
    await search.fill("dashboard-contract-pending-clear")
    await page.get_by_role("button", name="Clear filters").click()
    await asyncio.sleep(0.4)
    await expect(page).to_have_url(re.compile(r"/cayu/sessions$"))

    async def select_session_filter(
        label: str,
        value: str,
        expected_query: dict[str, str],
    ) -> None:
        async with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and urlsplit(response.url).path == "/api/sessions/summary"
                and all(
                    parse_qs(urlsplit(response.url).query).get(field) == [expected]
                    for field, expected in expected_query.items()
                )
            )
        ) as response_info:
            await page.get_by_label(label).select_option(value)
        response = await response_info.value
        require_equal(response.status, 200, f"{label} session summary response")

    await select_session_filter(
        "Filter by status",
        "completed",
        {"status": "completed", "order_by": "updated_at_desc"},
    )
    await select_session_filter(
        "Filter by debug state",
        "tool_issue",
        {
            "status": "completed",
            "debug_state": "tool_issue",
            "order_by": "updated_at_desc",
        },
    )
    await select_session_filter(
        "Sort sessions",
        "created_at_asc",
        {
            "status": "completed",
            "debug_state": "tool_issue",
            "order_by": "created_at_asc",
        },
    )
    final_query = parse_qs(urlsplit(page.url).query)
    require_equal(final_query.get("status"), ["completed"], "status filter URL state")
    require_equal(final_query.get("debug_state"), ["tool_issue"], "debug filter URL state")
    require_equal(final_query.get("order_by"), ["created_at_asc"], "session order URL state")
    await expect(page.get_by_role("link", name=SESSION_ID)).to_be_visible()

    await page.get_by_role("button", name="Clear filters").click()
    cleared_query = parse_qs(urlsplit(page.url).query)
    require("status" not in cleared_query, "clearing filters must remove status")
    require("debug_state" not in cleared_query, "clearing filters must remove debug state")
    require("cursor" not in cleared_query, "clearing filters must remove the cursor")
    require_equal(
        cleared_query.get("order_by"),
        ["created_at_asc"],
        "clearing filters must preserve ordering",
    )
    await expect(page.get_by_label("Sort sessions")).to_have_value("created_at_asc")
    await expect(page.get_by_role("link", name=SESSION_ID)).to_be_visible()


async def _exercise_run_agent_inventory(page: Page, base_url: str) -> None:
    inventory_mode = "empty"

    async def serve_agent_inventory(route) -> None:
        response = await route.fetch()
        body = await response.json()
        source_agents = body.get("agents", []) if isinstance(body, dict) else []
        if inventory_mode == "empty":
            projected_agents = []
        else:
            projected_agents = [
                agent
                for agent in source_agents
                if isinstance(agent, dict) and agent.get("name") == AGENT_NAME
            ]
        await route.fulfill(
            response=response,
            json={"agents": projected_agents, "total_count": len(projected_agents)},
        )

    await page.route("**/api/agents", serve_agent_inventory)
    try:
        await page.goto(f"{base_url}/cayu/run", wait_until="networkidle")
        agent_select = page.get_by_label("Agent", exact=True)
        await expect(agent_select).to_have_value("")
        await expect(page.get_by_text("No agents are registered.", exact=False)).to_be_visible()
        await page.get_by_label("Prompt", exact=True).fill("This run must remain disabled.")
        await expect(page.get_by_role("button", name="Run", exact=True)).to_be_disabled()

        inventory_mode = "single"
        await page.reload(wait_until="networkidle")
        agent_select = page.get_by_label("Agent", exact=True)
        await expect(agent_select).to_have_value(AGENT_NAME)
        await page.get_by_label("Prompt", exact=True).fill("This run has one unambiguous agent.")
        await expect(page.get_by_role("button", name="Run", exact=True)).to_be_enabled()
    finally:
        await page.unroute("**/api/agents", serve_agent_inventory)


async def _exercise_mutation_recovery(page: Page, base_url: str) -> None:
    session_urls: list[str] = []
    submitted_run_agents: list[str | None] = []

    def record_run_request(request) -> None:
        if urlsplit(request.url).path != "/api/run" or "last-event-id" in request.headers:
            return
        body = request.post_data_json
        submitted_run_agents.append(body.get("agent") if isinstance(body, dict) else None)

    page.on("request", record_run_request)

    try:
        for prompt in (
            "recover after HTTP acceptance before the first event",
            "recover after the first durable event",
        ):
            await page.goto(f"{base_url}/cayu/run", wait_until="networkidle")
            await expect(page.get_by_role("heading", name="New Run")).to_be_visible()
            agent_select = page.get_by_label("Agent", exact=True)
            await expect(agent_select).to_have_value("")
            await page.get_by_label("Prompt", exact=True).fill(prompt)
            run_button = page.get_by_role("button", name="Run", exact=True)
            await expect(run_button).to_be_disabled()
            await agent_select.select_option(AGENT_NAME)
            await expect(run_button).to_be_enabled()
            await run_button.click()
            await expect(page.locator('[data-mutation-transport-phase="terminal"]')).to_be_visible(
                timeout=15_000
            )
            await expect(
                page.get_by_text("dashboard mutation recovery completed", exact=True)
            ).to_be_visible()
            session_reference = page.get_by_test_id("run-session-reference")
            await expect(session_reference).to_be_visible()
            session_id = await session_reference.inner_text()
            copy_button = page.get_by_role("button", name="Copy", exact=True)
            await expect(copy_button).to_be_visible()
            await copy_button.click()
            await expect(page.get_by_role("button", name="Copied", exact=True)).to_be_visible()
            session_button = page.get_by_role("button", name="View Session →")
            await expect(session_button).to_be_visible()
            await session_button.click()
            await expect(page).to_have_url(f"{base_url}/cayu/sessions/{session_id}")
            await expect(page.get_by_role("heading", name=session_id)).to_be_visible()
            session_urls.append(page.url)
    finally:
        page.remove_listener("request", record_run_request)

    require_equal(
        len(set(session_urls)), 2, "each recovered run must keep a distinct session identity"
    )
    require_equal(
        submitted_run_agents,
        [AGENT_NAME, AGENT_NAME],
        "each New Run submission must carry the exact operator-selected agent",
    )


async def _exercise_existing_session_mutations(
    page: Page, base_url: str, provider: DashboardContractProvider
) -> None:
    await page.goto(f"{base_url}/cayu/sessions/{SESSION_ID}", wait_until="networkidle")
    resume_input = page.get_by_placeholder("Continue with a new prompt...")
    await expect(resume_input).to_be_visible()
    await resume_input.fill("Resume through the dashboard browser contract.")
    resume_button = page.get_by_role("button", name="Resume", exact=True)
    await resume_button.click()
    await provider.wait_for_direct_requests(1)
    await expect(page.get_by_role("button", name="Interrupt session", exact=True)).to_be_visible(
        timeout=10_000
    )
    await expect(resume_button).to_have_count(0)
    provider.release_direct()
    await expect(page.locator('[data-mutation-transport-phase="terminal"]')).to_be_visible(
        timeout=15_000
    )
    await expect(page.get_by_role("button", name=re.compile(r"session\.resumed"))).to_be_visible()

    await page.goto(f"{base_url}/cayu/sessions/{APPROVAL_SESSION_ID}", wait_until="networkidle")
    await expect(page.get_by_text("Awaiting approval", exact=True)).to_be_visible()
    deny_button = page.get_by_role("button", name="Deny", exact=True)
    await deny_button.click()
    await provider.wait_for_direct_requests(2)
    await expect(page.get_by_role("button", name="Interrupt session", exact=True)).to_be_visible(
        timeout=10_000
    )
    await expect(deny_button).to_have_count(0)
    provider.release_direct()
    await expect(page.locator('[data-mutation-transport-phase="terminal"]')).to_be_visible(
        timeout=15_000
    )
    await expect(page.get_by_text("Awaiting approval", exact=True)).to_have_count(0)

    await page.goto(f"{base_url}/cayu/sessions/{INTERRUPT_SESSION_ID}", wait_until="networkidle")
    await page.get_by_role("button", name="Interrupt session", exact=True).click()
    interrupt_sheet = page.get_by_role("dialog")
    await expect(interrupt_sheet.get_by_text("Interrupt session?", exact=True)).to_be_visible()
    await expect(interrupt_sheet.get_by_role("button", name="Keep running")).to_be_visible()
    await interrupt_sheet.get_by_label("Reason (optional)").fill("browser contract interruption")
    await interrupt_sheet.get_by_role("button", name="Interrupt session", exact=True).click()
    await expect(page.locator('[data-mutation-transport-phase="terminal"]')).to_be_visible(
        timeout=15_000
    )
    await expect(interrupt_sheet).to_be_hidden()
    await expect(page.get_by_text("interrupted", exact=True).first).to_be_visible()

    await page.goto(
        f"{base_url}/cayu/sessions/{RESUME_INTERRUPT_SESSION_ID}",
        wait_until="networkidle",
    )
    interrupted_resume_input = page.get_by_placeholder("Continue with a new prompt...")
    await interrupted_resume_input.fill("Interrupt this resume while its provider is active.")
    interrupted_resume_button = page.get_by_role("button", name="Resume", exact=True)
    await interrupted_resume_button.click()
    await provider.wait_for_direct_requests(3)
    active_interrupt_button = page.get_by_role("button", name="Interrupt session", exact=True)
    await expect(active_interrupt_button).to_be_visible(timeout=10_000)
    await expect(interrupted_resume_button).to_have_count(0)
    await active_interrupt_button.click()
    active_interrupt_sheet = page.get_by_role("dialog")
    await active_interrupt_sheet.get_by_label("Reason (optional)").fill(
        "interrupt an active resumed run"
    )
    await active_interrupt_sheet.get_by_role("button", name="Interrupt session", exact=True).click()
    await expect(page.locator('[data-mutation-transport-phase="terminal"]')).to_be_visible(
        timeout=15_000
    )
    await expect(active_interrupt_sheet).to_be_hidden()
    await expect(page.get_by_text("interrupted", exact=True).first).to_be_visible()

    failure_path = f"**/api/sessions/{INTERRUPT_FAILURE_SESSION_ID}/interrupt"

    async def inject_runtime_failure(route) -> None:
        error = {
            "type": "stream.error",
            "kind": "runtime",
            "code": "runtime_failed",
            "error": "Injected dashboard interrupt failure.",
            "error_type": "DashboardContractFailure",
            "retryable": False,
            "session_id": INTERRUPT_FAILURE_SESSION_ID,
        }
        await route.fulfill(
            status=200,
            headers={"content-type": "text/event-stream"},
            body=f"event: error\ndata: {json.dumps(error, separators=(',', ':'))}\n\n",
        )

    await page.route(failure_path, inject_runtime_failure)
    try:
        await page.goto(
            f"{base_url}/cayu/sessions/{INTERRUPT_FAILURE_SESSION_ID}",
            wait_until="networkidle",
        )
        await page.get_by_role("button", name="Interrupt session", exact=True).click()
        failure_sheet = page.get_by_role("dialog")
        await failure_sheet.get_by_role("button", name="Interrupt session", exact=True).click()
        await expect(
            page.locator('[data-mutation-transport-phase="runtime_failed"]')
        ).to_be_visible(timeout=15_000)
        close_action = failure_sheet.get_by_role("button", name="Close", exact=True).first
        await expect(close_action).to_be_visible()
        await expect(failure_sheet.get_by_role("button", name="Keep running")).to_have_count(0)
        await close_action.click()
        await expect(failure_sheet).to_be_hidden()
    finally:
        await page.unroute(failure_path, inject_runtime_failure)

    await _exercise_manual_mutation_reobservation(page, base_url)


async def _exercise_manual_mutation_reobservation(page: Page, base_url: str) -> None:
    resume_path = "**/api/resume"
    events_path = f"**/api/sessions/{REOBSERVE_SESSION_ID}/events*"
    baseline_response = await page.request.get(
        f"{base_url}/api/sessions/{REOBSERVE_SESSION_ID}/events",
        params={"order_by": "sequence_desc", "limit": 1},
    )
    require_equal(baseline_response.status, 200, "manual recovery baseline must be readable")
    baseline_payload = await baseline_response.json()
    baseline_events = baseline_payload.get("events", [])
    require_equal(len(baseline_events), 1, "manual recovery requires one durable baseline event")
    baseline_sequence = baseline_events[0]["sequence"]
    baseline_event_id = baseline_events[0]["id"]
    require(
        isinstance(baseline_sequence, int) and baseline_sequence > 0,
        "manual recovery baseline must have a positive durable sequence",
    )
    require_equal(
        baseline_event_id,
        f"cayu_event_{baseline_sequence}",
        "manual recovery must begin from the server's public event identity",
    )
    mutation_id: str | None = None
    recovered = False
    timestamp = "2026-01-01T00:00:00Z"
    terminal_event = {
        "id": f"cayu_event_{baseline_sequence + 1}",
        "type": "session.completed",
        "session_id": REOBSERVE_SESSION_ID,
        "interaction_id": None,
        "timestamp": timestamp,
        "agent_name": AGENT_NAME,
        "tool_name": None,
        "environment_name": None,
        "workflow_name": None,
        "payload": {},
    }

    async def inject_observer_failure_then_recovery(route) -> None:
        nonlocal mutation_id, recovered
        request_headers = route.request.headers
        request_mutation_id = request_headers.get("cayu-mutation-id")
        require(bool(request_mutation_id), "dashboard resume must carry a mutation identity")
        last_event_id = request_headers.get("last-event-id")
        if last_event_id is None:
            require(mutation_id is None, "manual recovery may submit the mutation only once")
            mutation_id = request_mutation_id
            error = {
                "type": "stream.error",
                "kind": "observer",
                "code": "event_frame_too_large",
                "error": "Injected non-retryable dashboard observer failure.",
                "error_type": "DashboardContractObserverFailure",
                "retryable": False,
                "session_id": REOBSERVE_SESSION_ID,
            }
            await route.fulfill(
                status=200,
                headers={"content-type": "text/event-stream"},
                body=f"event: error\ndata: {json.dumps(error, separators=(',', ':'))}\n\n",
            )
            return

        require_equal(
            request_mutation_id,
            mutation_id,
            "manual recovery must preserve the original mutation identity",
        )
        require_equal(
            last_event_id,
            f"{REOBSERVE_SESSION_ID}:{baseline_event_id}",
            "manual recovery must replay from the original durable baseline",
        )
        acceptance_event = {
            "id": f"cayu_event_{baseline_sequence + 2}",
            "type": "server.mutation.accepted",
            "session_id": REOBSERVE_SESSION_ID,
            "interaction_id": None,
            "timestamp": timestamp,
            "agent_name": AGENT_NAME,
            "tool_name": None,
            "environment_name": None,
            "workflow_name": None,
            "payload": {
                "mutation_id": mutation_id,
                "mutation_kind": "resume",
                "accepted_event_id": terminal_event["id"],
                "accepted_event_type": terminal_event["type"],
            },
        }
        recovered = True
        frames = []
        for event in (terminal_event, acceptance_event):
            frames.append(
                f"id: {REOBSERVE_SESSION_ID}:{event['id']}\n"
                f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
            )
        await route.fulfill(
            status=200,
            headers={"content-type": "text/event-stream"},
            body="".join(frames),
        )

    async def expose_recovered_events(route) -> None:
        if not recovered:
            await route.continue_()
            return

        query = parse_qs(urlsplit(route.request.url).query)
        order_by = query.get("order_by", ["sequence_asc"])[0]
        limit = int(query.get("limit", ["100"])[0])
        after_sequence = int(query.get("after_sequence", ["0"])[0])
        before_value = query.get("before_sequence", [None])[0]
        before_sequence = int(before_value) if before_value is not None else None
        records = [
            {**terminal_event, "sequence": baseline_sequence + 1},
            {
                "id": f"cayu_event_{baseline_sequence + 2}",
                "type": "server.mutation.accepted",
                "session_id": REOBSERVE_SESSION_ID,
                "interaction_id": None,
                "timestamp": timestamp,
                "agent_name": AGENT_NAME,
                "tool_name": None,
                "environment_name": None,
                "workflow_name": None,
                "payload": {
                    "mutation_id": mutation_id,
                    "mutation_kind": "resume",
                    "accepted_event_id": terminal_event["id"],
                    "accepted_event_type": terminal_event["type"],
                },
                "sequence": baseline_sequence + 2,
            },
        ]
        matching = [record for record in records if record["sequence"] > after_sequence]
        if before_sequence is not None:
            matching = [record for record in matching if record["sequence"] < before_sequence]
        if order_by == "sequence_desc":
            matching.reverse()
        page_records = matching[:limit]
        payload = {
            "session_id": REOBSERVE_SESSION_ID,
            "events": page_records,
            "order_by": order_by,
            "next_sequence": page_records[-1]["sequence"] if page_records else None,
            "scan_through_sequence": baseline_sequence + 2 if before_sequence is None else None,
            "has_more": len(matching) > limit,
        }
        await route.fulfill(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(payload, separators=(",", ":")),
        )

    await page.route(resume_path, inject_observer_failure_then_recovery)
    await page.route(events_path, expose_recovered_events)
    try:
        await page.goto(
            f"{base_url}/cayu/sessions/{REOBSERVE_SESSION_ID}", wait_until="networkidle"
        )
        resume_input = page.get_by_placeholder("Continue with a new prompt...")
        await resume_input.fill("Recover this dashboard observation without resubmitting.")
        await page.get_by_role("button", name="Resume", exact=True).click()
        await expect(
            page.locator('[data-mutation-transport-phase="transport_failed"]')
        ).to_be_visible(timeout=25_000)
        await expect(resume_input).to_have_count(0)
        retry_observation = page.get_by_role("button", name="Retry observation", exact=True)
        await expect(retry_observation).to_be_visible()
        await retry_observation.click()
        await expect(page.locator('[data-mutation-transport-phase="terminal"]')).to_be_visible(
            timeout=15_000
        )
        await expect(retry_observation).to_have_count(0)
        await expect(resume_input).to_be_enabled()
        require(recovered, "manual dashboard recovery must issue a replay request")
    finally:
        await page.unroute(events_path, expose_recovered_events)
        await page.unroute(resume_path, inject_observer_failure_then_recovery)


def _record_browser_failures(
    page: Page,
    failures: dict[str, list[str]],
    expected_observer_aborts: list[ObserverAbort],
    expected_observer_abort_paths: set[str],
    expected_query_aborts: list[str],
    superseded_read_aborts: list[str],
    expected_edit_rejections: list[str],
    expected_edit_console_errors: list[str],
    expected_usage_rejections: list[str],
    expected_usage_console_errors: list[str],
) -> None:
    def record_request_failure(request: Request) -> None:
        path = urlsplit(request.url).path
        headers = request.headers
        last_event_id = headers.get("last-event-id")
        detail = (
            f"{request.method} {request.url}: {request.failure or 'unknown failure'} "
            f"[mutation={headers.get('cayu-mutation-id', '-')}, "
            f"last_event_id={last_event_id or '-'}]"
        )
        # The client deliberately aborts mutation observers after a durable
        # terminal boundary or when an explicit interrupt supersedes active
        # observation. Chromium can also classify an injected initial disconnect
        # as ERR_ABORTED; Last-Event-ID distinguishes replay observers below.
        if (
            request.method == "POST"
            and path in expected_observer_abort_paths
            and request.failure == "net::ERR_ABORTED"
        ):
            expected_observer_aborts.append((path, last_event_id, detail))
            return
        if (
            request.method == "POST"
            and urlsplit(request.url).path == "/api/sessions/summary"
            and parse_qs(urlsplit(request.url).query).get("q") == [SLOW_SESSION_QUERY]
            and request.failure == "net::ERR_ABORTED"
        ):
            expected_query_aborts.append(detail)
            return
        if request.method == "POST" and path == "/api/usage/rollup":
            body = request.post_data_json
            agent_name = body.get("session_filter", {}).get("agent_name") if body else None
            if agent_name == SLOW_USAGE_AGENT and request.failure == "net::ERR_ABORTED":
                expected_query_aborts.append(detail)
                return
        if (
            request.method == "GET"
            and path == "/api/evals/runs"
            and request.failure == "net::ERR_ABORTED"
        ):
            expected_query_aborts.append(detail)
            return
        if (
            request.method == "GET"
            and (path == "/api/evals/corpora" or path.startswith("/api/evals/corpora/"))
            and request.failure == "net::ERR_ABORTED"
        ):
            expected_query_aborts.append(detail)
            return
        if (
            request.method == "POST"
            and path == f"/api/sessions/{WORKFLOW_FOCUS_SESSION_ID}/topology"
            and request.failure == "net::ERR_ABORTED"
        ):
            body = request.post_data_json
            expanded_ids = body.get("expanded_parent_ids", []) if isinstance(body, dict) else []
            if (
                WORKFLOW_ACTIVE_SESSION_ID in expanded_ids
                and WORKFLOW_FAILED_SESSION_ID not in expanded_ids
            ):
                expected_query_aborts.append(detail)
                return
        if is_superseded_browser_read_abort(
            method=request.method,
            path=path,
            failure=request.failure,
            mutation_id=headers.get("cayu-mutation-id"),
        ):
            superseded_read_aborts.append(detail)
            return
        failures["request_failures"].append(detail)

    def record_response(response) -> None:
        request = response.request
        path = urlsplit(response.url).path
        detail = f"{response.status} {request.method} {response.url}"
        if (
            response.status == 422
            and request.method == "PATCH"
            and path == f"/api/sessions/{SESSION_ID}/labels"
        ):
            expected_edit_rejections.append(detail)
            return
        if response.status == 501 and request.method == "POST" and path == "/api/usage/rollup":
            expected_usage_rejections.append(detail)
            return
        if "/api/" in response.url and response.status >= 400:
            failures["api_errors"].append(detail)

    def record_console(message) -> None:
        if message.type != "error":
            return
        if (
            message.text
            == "Failed to load resource: the server responded with a status of 422 (Unprocessable Entity)"
        ):
            expected_edit_console_errors.append(message.text)
            return
        if (
            message.text
            == "Failed to load resource: the server responded with a status of 501 (Not Implemented)"
        ):
            expected_usage_console_errors.append(message.text)
            return
        failures["console_errors"].append(message.text)

    page.on("console", record_console)
    page.on("pageerror", lambda error: failures["page_errors"].append(str(error)))
    page.on("requestfailed", record_request_failure)
    page.on("response", record_response)


def _require_no_browser_failures(failures: dict[str, list[str]]) -> None:
    for kind, messages in failures.items():
        require(not messages, f"dashboard recorded {kind}: {messages}")


async def _capture_diagnostics(
    context: BrowserContext,
    page: Page,
    diagnostics_dir: Path,
    failures: dict[str, list[str]],
) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    screenshot = diagnostics_dir / "dashboard-behavior.png"
    trace = diagnostics_dir / "dashboard-behavior-trace.zip"
    html = diagnostics_dir / "dashboard-behavior.html"
    capture_errors: list[str] = []
    try:
        await page.screenshot(path=str(screenshot), full_page=True)
    except Exception as exc:
        capture_errors.append(f"screenshot: {exc}")
    try:
        html.write_text(await page.content(), encoding="utf-8")
    except Exception as exc:
        capture_errors.append(f"html: {exc}")
    try:
        await context.tracing.stop(path=str(trace))
    except Exception as exc:
        capture_errors.append(f"trace: {exc}")
    print(
        "CAYU_DASHBOARD_DIAGNOSTICS="
        + json.dumps(
            {
                "directory": str(diagnostics_dir),
                "browser_failures": failures,
                "capture_errors": capture_errors,
            },
            sort_keys=True,
        )
    )


def _loopback_listener() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    return listener


async def _wait_for_server(server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    async def wait_until_started() -> None:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("dashboard server stopped before startup")
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait_until_started(), timeout=10)


if __name__ == "__main__":
    asyncio.run(main())
