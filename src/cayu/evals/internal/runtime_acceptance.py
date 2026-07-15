from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from cayu import (
    AgentSpec,
    BudgetLimit,
    BudgetWindow,
    CayuApp,
    ChildSessionCompleted,
    ContextCountingConfig,
    ContextCountingMode,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    EvalCase,
    EvalPlan,
    EvalSuite,
    EventPayloadContains,
    EventType,
    FinalOutputContains,
    InMemoryKnowledgeStore,
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    KnowledgeEntry,
    KnowledgeReviewWorkflow,
    KnowledgeStatus,
    LocalWorkspace,
    MaxTotalTokens,
    Message,
    ModelPrice,
    PriceBook,
    ReadFileTool,
    ReadKnowledgeTool,
    RunRequest,
    ScriptedModelProvider,
    SearchKnowledgeTool,
    SessionCompleted,
    SessionInterrupted,
    SubagentSpec,
    SubagentTool,
    Tool,
    ToolArgsContain,
    ToolCalled,
    ToolContext,
    ToolNotCalled,
    ToolResult,
    ToolResultContains,
    ToolSpec,
    UsageRecorded,
    WriteFileTool,
)
from cayu.providers import ModelRequest, ModelStreamEvent

SUITE_ID = "cayu-internal-runtime-acceptance-v1"
ENVIRONMENT_NAME = "runtime-acceptance-local"
MODEL = "runtime-acceptance-model"

TOOL_PROVIDER = "runtime-acceptance-tool-provider"
WORKSPACE_PROVIDER = "runtime-acceptance-workspace-provider"
CONTEXT_PROVIDER = "runtime-acceptance-context-provider"
KNOWLEDGE_PROVIDER = "runtime-acceptance-knowledge-provider"
SUBAGENT_PARENT_PROVIDER = "runtime-acceptance-subagent-parent-provider"
SUBAGENT_CHILD_PROVIDER = "runtime-acceptance-subagent-child-provider"
USAGE_PROVIDER = "runtime-acceptance-usage-provider"
BUDGET_PROVIDER = "runtime-acceptance-budget-provider"

TOOL_AGENT = "runtime_acceptance_tool"
WORKSPACE_AGENT = "runtime_acceptance_workspace"
CONTEXT_AGENT = "runtime_acceptance_context"
KNOWLEDGE_AGENT = "runtime_acceptance_knowledge"
SUBAGENT_PARENT_AGENT = "runtime_acceptance_subagent_parent"
SUBAGENT_CHILD_AGENT = "runtime_acceptance_subagent_child"
USAGE_AGENT = "runtime_acceptance_usage"
BUDGET_AGENT = "runtime_acceptance_budget"

WORKSPACE_FILE = "runtime-acceptance/workspace-roundtrip.txt"
WORKSPACE_CONTENT = "isolated workspace content"
KNOWLEDGE_ENTRY_ID = "runtime_acceptance_reviewed_fact"
KNOWLEDGE_FACT = "Cayu's deterministic runtime codename is Saffron."


class _RuntimeEchoTool(Tool):
    spec = ToolSpec(
        name="runtime_echo",
        description="Return deterministic text for the runtime acceptance suite.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=f"runtime echo: {args['text']}",
            structured={"text": args["text"]},
        )


class _QueuedSideEffectTool(Tool):
    spec = ToolSpec(
        name="queued_side_effect",
        description="A side effect that the budget-interrupt case must prevent.",
        input_schema={"type": "object", "additionalProperties": False, "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="unexpected side effect executed")


class _CountingScriptedModelProvider(ScriptedModelProvider):
    def __init__(
        self,
        events,
        *,
        name: str,
        count_result: InputTokenCountResult,
    ) -> None:
        super().__init__(events, name=name)
        self._count_result = count_result

    async def count_input_tokens(
        self,
        request: ModelRequest,
    ) -> InputTokenCountResult:
        return self._count_result.model_copy(deep=True)


class _IsolatedLocalEnvironmentFactory(EnvironmentFactory):
    def __init__(self, knowledge_store: InMemoryKnowledgeStore) -> None:
        self._temporary_directory = TemporaryDirectory(prefix="cayu-runtime-acceptance-")
        self._root = Path(self._temporary_directory.name)
        self._knowledge_store = knowledge_store
        self._workspaces: dict[str, LocalWorkspace] = {}

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        workspace = self._workspaces.get(request.session_id)
        if workspace is None:
            directory_name = hashlib.sha256(request.session_id.encode("utf-8")).hexdigest()[:24]
            workspace_root = self._root / directory_name
            workspace_root.mkdir()
            workspace = LocalWorkspace(
                workspace_root,
                workspace_id=f"runtime-acceptance:{request.session_id}",
            )
            self._workspaces[request.session_id] = workspace
        environment = Environment(
            EnvironmentSpec(name=ENVIRONMENT_NAME),
            workspace=workspace,
            knowledge_store=(
                self._knowledge_store if request.agent_name == KNOWLEDGE_AGENT else None
            ),
        )
        return EnvironmentFactoryResult(
            environment=environment,
            metadata={"workspace_id": workspace.id},
            reconnect_metadata={"workspace_id": workspace.id},
        )


async def build() -> EvalPlan:
    """Build Cayu's credential-free, network-free runtime acceptance plan."""

    knowledge_store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id=KNOWLEDGE_ENTRY_ID,
                text=KNOWLEDGE_FACT,
                namespace="cayu:runtime-acceptance",
                kind="fact",
                status=KnowledgeStatus.PENDING,
                title="Reviewed deterministic runtime fact",
            )
        ]
    )
    review = KnowledgeReviewWorkflow(
        knowledge_store,
        namespace="cayu:runtime-acceptance",
    )
    await review.approve(KNOWLEDGE_ENTRY_ID)

    app = CayuApp(
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
        enable_logging=False,
    )
    app.register_environment_factory(
        EnvironmentSpec(name=ENVIRONMENT_NAME),
        _IsolatedLocalEnvironmentFactory(knowledge_store),
        default=True,
    )

    _register_providers(app)
    _register_agents(app)

    return EvalPlan(
        app=app,
        suite=EvalSuite(
            id=SUITE_ID,
            metadata={
                "hermetic": True,
                "network": False,
                "provider_credentials": False,
                "llm_judge": False,
            },
            cases=_cases(),
        ),
    )


def _register_providers(app: CayuApp) -> None:
    providers = (
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="tool-roundtrip-call",
                        name="runtime_echo",
                        arguments={"text": "roundtrip-marker"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                _final_batch("tool roundtrip used runtime echo: roundtrip-marker"),
            ],
            name=TOOL_PROVIDER,
        ),
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="workspace-write-call",
                        name="write_file",
                        arguments={"path": WORKSPACE_FILE, "content": WORKSPACE_CONTENT},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="workspace-read-call",
                        name="read_file",
                        arguments={"path": WORKSPACE_FILE},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                _final_batch(f"workspace roundtrip read {WORKSPACE_CONTENT}"),
            ],
            name=WORKSPACE_PROVIDER,
        ),
        _CountingScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("context observability complete"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "model": MODEL,
                        "usage": {
                            "input_tokens": 15,
                            "output_tokens": 2,
                            "total_tokens": 17,
                        },
                    }
                ),
            ],
            name=CONTEXT_PROVIDER,
            count_result=InputTokenCountResult(
                input_tokens=12,
                method=InputTokenCountMethod.OFFICIAL,
                confidence=InputTokenCountConfidence.HIGH,
                components={"messages": 10, "tools": 2},
                metadata={"source": "deterministic-script"},
            ),
        ),
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="knowledge-search-call",
                        name="search_knowledge",
                        arguments={
                            "query": "Saffron runtime codename",
                            "namespace": "cayu:runtime-acceptance",
                        },
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="knowledge-read-call",
                        name="read_knowledge",
                        arguments={"entry_id": KNOWLEDGE_ENTRY_ID},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                _final_batch("reviewed knowledge says the runtime codename is Saffron"),
            ],
            name=KNOWLEDGE_PROVIDER,
        ),
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="subagent-parent-call",
                        name="subagent",
                        arguments={
                            "agent": "reviewer",
                            "task": "Return the deterministic runtime verdict.",
                        },
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                _final_batch("parent accepted child verdict: runtime stable"),
            ],
            name=SUBAGENT_PARENT_PROVIDER,
        ),
        ScriptedModelProvider(
            _final_batch("child verdict: runtime stable"),
            name=SUBAGENT_CHILD_PROVIDER,
        ),
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("usage accounting complete"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "model": MODEL,
                        "usage": {
                            "input_tokens": 9,
                            "output_tokens": 4,
                            "total_tokens": 13,
                        },
                    }
                ),
            ],
            name=USAGE_PROVIDER,
        ),
        ScriptedModelProvider(
            [
                ModelStreamEvent.tool_call(
                    id="budget-side-effect-call",
                    name="queued_side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "model": MODEL,
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 100,
                            "total_tokens": 1100,
                        },
                    }
                ),
            ],
            name=BUDGET_PROVIDER,
        ),
    )
    for provider in providers:
        app.register_provider(provider, default=provider.name == TOOL_PROVIDER)


def _register_agents(app: CayuApp) -> None:
    app.register_agent(
        AgentSpec(name=TOOL_AGENT, model=MODEL, provider_name=TOOL_PROVIDER),
        tools=[_RuntimeEchoTool()],
    )
    app.register_agent(
        AgentSpec(name=WORKSPACE_AGENT, model=MODEL, provider_name=WORKSPACE_PROVIDER),
        tools=[WriteFileTool(), ReadFileTool()],
    )
    app.register_agent(AgentSpec(name=CONTEXT_AGENT, model=MODEL, provider_name=CONTEXT_PROVIDER))
    app.register_agent(
        AgentSpec(name=KNOWLEDGE_AGENT, model=MODEL, provider_name=KNOWLEDGE_PROVIDER),
        tools=[SearchKnowledgeTool(), ReadKnowledgeTool()],
    )
    app.register_agent(
        AgentSpec(
            name=SUBAGENT_CHILD_AGENT,
            model=MODEL,
            provider_name=SUBAGENT_CHILD_PROVIDER,
        )
    )
    app.register_agent(
        AgentSpec(
            name=SUBAGENT_PARENT_AGENT,
            model=MODEL,
            provider_name=SUBAGENT_PARENT_PROVIDER,
        ),
        tools=[
            SubagentTool(
                app,
                agents={
                    "reviewer": SubagentSpec(
                        agent_name=SUBAGENT_CHILD_AGENT,
                        description="Return a deterministic runtime verdict.",
                        max_steps=1,
                    )
                },
            )
        ],
    )
    app.register_agent(AgentSpec(name=USAGE_AGENT, model=MODEL, provider_name=USAGE_PROVIDER))
    app.register_agent(
        AgentSpec(name=BUDGET_AGENT, model=MODEL, provider_name=BUDGET_PROVIDER),
        tools=[_QueuedSideEffectTool()],
    )


def _cases() -> list[EvalCase]:
    return [
        EvalCase(
            id="tool_roundtrip",
            request=_request(
                TOOL_AGENT,
                "tool-roundtrip",
                "Call runtime_echo and use its result.",
                max_steps=2,
            ),
            assertions=[
                SessionCompleted(),
                ToolCalled("runtime_echo"),
                ToolArgsContain("runtime_echo", {"text": "roundtrip-marker"}),
                ToolResultContains("runtime_echo", "runtime echo: roundtrip-marker"),
                FinalOutputContains("runtime echo: roundtrip-marker"),
            ],
        ),
        EvalCase(
            id="workspace_roundtrip",
            request=_request(
                WORKSPACE_AGENT,
                "workspace-roundtrip",
                "Write, read, and report the isolated workspace marker.",
                max_steps=3,
            ),
            assertions=[
                SessionCompleted(),
                ToolCalled("write_file"),
                ToolArgsContain(
                    "write_file",
                    {"path": WORKSPACE_FILE, "content": WORKSPACE_CONTENT},
                ),
                ToolResultContains("write_file", f"Wrote {len(WORKSPACE_CONTENT)} bytes"),
                ToolCalled("read_file"),
                ToolArgsContain("read_file", {"path": WORKSPACE_FILE}),
                ToolResultContains("read_file", WORKSPACE_CONTENT),
                FinalOutputContains(WORKSPACE_CONTENT),
            ],
        ),
        EvalCase(
            id="context_observability",
            request=_request(
                CONTEXT_AGENT,
                "context-observability",
                "Produce deterministic context observability facts.",
                max_steps=1,
            ),
            assertions=[
                SessionCompleted(),
                EventPayloadContains(
                    EventType.CONTEXT_PRESSURE_ESTIMATED,
                    {
                        "provider": CONTEXT_PROVIDER,
                        "model": MODEL,
                        "messages": {"count": 1},
                        "tools": {"count": 0},
                        "estimate": {"method": "local_full_request_estimate"},
                    },
                ),
                EventPayloadContains(
                    EventType.CONTEXT_COUNTED,
                    {
                        "provider": CONTEXT_PROVIDER,
                        "model": MODEL,
                        "count": {
                            "input_tokens": 12,
                            "method": "official",
                            "confidence": "high",
                            "components": {"messages": 10, "tools": 2},
                        },
                    },
                ),
                EventPayloadContains(
                    EventType.CONTEXT_PRESSURE_RECONCILED,
                    {"actual_input_tokens": 15, "reconciled": True},
                ),
                EventPayloadContains(
                    EventType.CONTEXT_COUNT_RECONCILED,
                    {
                        "actual_input_tokens": 15,
                        "delta_tokens": 3,
                        "reconciled": True,
                    },
                ),
            ],
        ),
        EvalCase(
            id="knowledge_tool_roundtrip",
            request=_request(
                KNOWLEDGE_AGENT,
                "knowledge-tool-roundtrip",
                "Search and read the reviewed runtime codename fact.",
                max_steps=3,
            ),
            assertions=[
                SessionCompleted(),
                ToolCalled("search_knowledge"),
                ToolResultContains("search_knowledge", "Saffron"),
                ToolCalled("read_knowledge"),
                ToolArgsContain("read_knowledge", {"entry_id": KNOWLEDGE_ENTRY_ID}),
                ToolResultContains("read_knowledge", KNOWLEDGE_FACT),
                FinalOutputContains("Saffron"),
            ],
        ),
        EvalCase(
            id="subagent_roundtrip",
            request=_request(
                SUBAGENT_PARENT_AGENT,
                "subagent-roundtrip",
                "Delegate the runtime verdict and use the child result.",
                max_steps=2,
            ),
            assertions=[
                SessionCompleted(),
                ToolCalled("subagent"),
                ToolResultContains("subagent", "child verdict: runtime stable"),
                ChildSessionCompleted(agent_name=SUBAGENT_CHILD_AGENT),
                FinalOutputContains("parent accepted child verdict: runtime stable"),
            ],
        ),
        EvalCase(
            id="usage_accounting",
            request=_request(
                USAGE_AGENT,
                "usage-accounting",
                "Record deterministic model usage.",
                max_steps=1,
            ),
            assertions=[
                SessionCompleted(),
                UsageRecorded(min_total_tokens=1),
                MaxTotalTokens(32),
                EventPayloadContains(
                    EventType.MODEL_COMPLETED,
                    {"usage": {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13}},
                ),
            ],
        ),
        EvalCase(
            id="budget_interrupt",
            request=_request(
                BUDGET_AGENT,
                "budget-interrupt",
                "Queue the side effect, but obey the caller's pricing budget.",
                max_steps=1,
                budget_limits=(_budget_limit(),),
            ),
            assertions=[
                SessionInterrupted(),
                EventPayloadContains(
                    EventType.SESSION_LIMIT_REACHED,
                    {
                        "limit": "estimated_cost",
                        "maximum": "0.001",
                        "actual": "0.002",
                    },
                ),
                EventPayloadContains(
                    EventType.TOOL_CALL_FAILED,
                    {
                        "reason": "limit_reached",
                        "result": {"structured": {"tool_name": "queued_side_effect"}},
                    },
                ),
                EventPayloadContains(
                    EventType.SESSION_INTERRUPTED,
                    {"interruption_type": "limit_reached"},
                ),
                ToolNotCalled("queued_side_effect"),
            ],
        ),
    ]


def _request(
    agent_name: str,
    session_slug: str,
    prompt: str,
    *,
    max_steps: int,
    budget_limits: tuple[BudgetLimit, ...] = (),
) -> RunRequest:
    return RunRequest(
        agent_name=agent_name,
        session_id=f"cayu-eval-runtime-{session_slug}-v1",
        environment_name=ENVIRONMENT_NAME,
        messages=[Message.text("user", prompt)],
        max_steps=max_steps,
        budget_limits=budget_limits,
    )


def _budget_limit() -> BudgetLimit:
    return BudgetLimit(
        max_estimated_cost=Decimal("0.001"),
        window=BudgetWindow.all_time(),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name=BUDGET_PROVIDER,
                    model=MODEL,
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("10"),
                ),
            )
        ),
    )


def _final_batch(text: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]
