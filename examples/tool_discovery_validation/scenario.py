from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from time import perf_counter_ns
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictBool, StrictInt

from cayu import (
    AgentSpec,
    CacheBreakpoint,
    CachePolicy,
    CayuApp,
    Event,
    EventType,
    ExecutionProfileBehaviorIdentity,
    ForkSessionRequest,
    Message,
    ModelPrice,
    ModelRequest,
    ModelStreamEvent,
    PriceBook,
    RequestFingerprintAvailability,
    RequestFootprint,
    RequestFootprintConfig,
    ResumeRequest,
    RunRequest,
    Tool,
    ToolCapabilityCeiling,
    ToolContext,
    ToolDescriptor,
    ToolEffect,
    ToolResult,
    ToolResultPart,
    ToolSpec,
)
from cayu.evals.runner import final_output_text
from cayu.providers import ModelProvider
from cayu.runtime.tool_catalogue import build_tool_catalog_snapshot, build_tool_descriptor
from cayu.runtime.tool_discovery import search_tool_descriptors

_PROVIDER_NAME = "tool-discovery-validation"
_MODEL = "fixture-model"
_EXPECTED_OUTPUT = "quality-ok"
_PARENT_SESSION_ID = "tool-discovery-parent"
_CHILD_SESSION_ID = "tool-discovery-child"
_DIRECT_SESSION_ID = "tool-discovery-eval-direct"
_DISCOVERY_SESSION_ID = "tool-discovery-eval-search-tools"
_TARGET_NAME = "remember_knowledge"
_TARGET_TOOL_ID = f"cayu:{_TARGET_NAME}"
_CORE_TOOL_NAMES = ("search_tools", "call_tool")
_NOISE_TOOL_COUNT = 32


class ToolDiscoveryLifecycleEvidence(BaseModel):
    """Content-minimized evidence for one resume-and-fork lifecycle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_requests: StrictInt = Field(ge=0)
    model_steps: StrictInt = Field(ge=0)
    stable_core_tool_names: tuple[str, ...]
    distinct_tool_manifest_fingerprints: StrictInt = Field(ge=0)
    view_grant_counts_by_request: tuple[int, ...]
    parent_view_revision_before_resume: StrictInt = Field(ge=0)
    parent_view_revision_after_resume: StrictInt = Field(ge=0)
    child_view_revision_before_search: StrictInt = Field(ge=0)
    child_view_revision_after_search: StrictInt = Field(ge=0)
    parent_grant_count: StrictInt = Field(ge=0)
    child_grant_count_before_search: StrictInt = Field(ge=0)
    child_grant_count_after_search: StrictInt = Field(ge=0)
    generation_changed_on_fork: StrictBool
    parent_reference_survived_resume: StrictBool
    copied_parent_reference_rejections: StrictInt = Field(ge=0)
    child_reference_was_fresh: StrictBool
    references_omitted_from_public_evidence: StrictBool
    target_effects: StrictInt = Field(ge=0)
    typed_event_counts: dict[str, int]


class ToolDiscoveryRankingCase(BaseModel):
    """One deterministic search-ranking expectation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    expected_tool_name: str
    observed_rank: StrictInt | None = Field(default=None, ge=1)
    returned_tool_names: tuple[str, ...]


class ToolDiscoveryRankingEvidence(BaseModel):
    """Bounded ranking corpus and direct-exposure exclusion evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_count: StrictInt = Field(ge=0)
    top_1_hits: StrictInt = Field(ge=0)
    mean_reciprocal_rank: Decimal = Field(ge=0, le=1)
    cases: tuple[ToolDiscoveryRankingCase, ...]
    directly_exposed_tool_excluded: StrictBool


class ToolDiscoveryEvaluationSide(BaseModel):
    """Bounded runtime, quality, usage, and cost evidence for one strategy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: Literal["direct_catalogue", "search_tools"]
    provider_requests: StrictInt = Field(ge=0)
    model_steps: StrictInt = Field(ge=0)
    search_calls: StrictInt = Field(ge=0)
    unnecessary_searches: StrictInt = Field(ge=0)
    invalid_argument_attempts: StrictInt = Field(ge=0)
    invalid_argument_rejections: StrictInt = Field(ge=0)
    target_invocations_started: StrictInt = Field(ge=0)
    target_effects: StrictInt = Field(ge=0)
    approval_requests: StrictInt = Field(ge=0)
    provider_tool_counts: tuple[int, ...]
    distinct_tool_manifest_fingerprints: StrictInt = Field(ge=0)
    input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)
    cache_read_input_tokens: StrictInt = Field(ge=0)
    cache_write_input_tokens: StrictInt = Field(ge=0)
    uncached_input_tokens: StrictInt = Field(ge=0)
    observed_latency_ms: Decimal = Field(ge=0)
    quality_passed: StrictBool
    estimated_cost: Decimal = Field(ge=0)
    currency: str


class ToolDiscoveryValidationReport(BaseModel):
    """Versioned deterministic evidence, not a provider benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    evidence_scope: Literal["deterministic_fixture"] = "deterministic_fixture"
    universal_savings_claimed: Literal[False] = False
    lifecycle: ToolDiscoveryLifecycleEvidence
    ranking: ToolDiscoveryRankingEvidence
    direct_catalogue: ToolDiscoveryEvaluationSide
    search_tools: ToolDiscoveryEvaluationSide


class _RememberKnowledgeTool(Tool):
    spec = ToolSpec(
        name=_TARGET_NAME,
        description="Save an important reusable lesson in durable knowledge.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"fact": {"type": "string", "minLength": 1}},
            "required": ["fact"],
        },
        effect=ToolEffect.EXTERNAL,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="examples:tool-discovery-validation:remember-knowledge",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.effects: list[str] = []

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx
        fact = args.get("fact")
        if type(fact) is not str or not fact.strip():
            return ToolResult(content="fact must be a nonblank string", is_error=True)
        self.effects.append(fact)
        return ToolResult(content=f"remembered: {fact}")


class _FixtureTool(Tool):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        property_name: str,
    ) -> None:
        super().__init__(
            ToolSpec(
                name=name,
                description=description,
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {property_name: {"type": "string"}},
                },
                effect=ToolEffect.NONE,
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name=f"examples:tool-discovery-validation:{name}",
                    behavior_version="1",
                    implementation_version="1",
                ),
            )
        )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx, args
        return ToolResult(content=f"{self.name}-ok")


def _catalogue_tools(target: _RememberKnowledgeTool) -> tuple[Tool, ...]:
    semantic = (
        _FixtureTool(
            name="search_saved_notes",
            description="Search previously saved notes by topic.",
            property_name="topic_query",
        ),
        _FixtureTool(
            name="publish_incident_report",
            description="Publish a reviewed incident report.",
            property_name="report_body",
        ),
        _FixtureTool(
            name="read_service_status",
            description="Read the current service health status.",
            property_name="service_name",
        ),
    )
    noise = tuple(
        _FixtureTool(
            name=f"catalogue_capability_{index:03d}",
            description=f"Operate bounded fixture capability number {index}.",
            property_name=f"fixture_value_{index:03d}",
        )
        for index in range(_NOISE_TOOL_COUNT)
    )
    return (target, *semantic, *noise)


def _completed(
    *,
    finish_reason: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> ModelStreamEvent:
    return ModelStreamEvent.completed(
        {
            "finish_reason": finish_reason,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read_tokens,
                "cache_creation_input_tokens": cache_write_tokens,
            },
        }
    )


def _tool_results(request: ModelRequest, tool_name: str) -> list[ToolResultPart]:
    return [
        part
        for message in request.messages
        if message.role == "tool"
        for part in message.content
        if isinstance(part, ToolResultPart) and part.tool_name == tool_name
    ]


def _discovered_match(request: ModelRequest, tool_name: str) -> dict[str, Any]:
    results = _tool_results(request, "search_tools")
    if not results or results[-1].structured is None:
        raise RuntimeError("Expected a private structured search_tools result.")
    matches = results[-1].structured.get("matches")
    if not isinstance(matches, list):
        raise RuntimeError("search_tools result did not contain a match list.")
    for match in matches:
        if isinstance(match, dict) and match.get("name") == tool_name:
            if match.get("tool_id") != f"cayu:{tool_name}":
                raise RuntimeError("search_tools returned a non-canonical fixture tool id.")
            if match.get("readiness") != "registered":
                raise RuntimeError("search_tools returned an unavailable fixture tool.")
            if not str(match.get("descriptor_version", "")).startswith("sha256:"):
                raise RuntimeError("search_tools omitted the descriptor fingerprint.")
            if not str(match.get("schema_fingerprint", "")).startswith("sha256:"):
                raise RuntimeError("search_tools omitted the schema fingerprint.")
            if (
                tool_name == _TARGET_NAME
                and match.get("input_schema") != _RememberKnowledgeTool.spec.input_schema
            ):
                raise RuntimeError("search_tools changed the admitted target schema.")
            return match
    raise RuntimeError(f"search_tools did not return {tool_name!r}.")


class _FixtureProvider(ModelProvider):
    name = _PROVIDER_NAME

    @property
    def identity_variant(self) -> str:
        raise NotImplementedError

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name=f"examples:tool-discovery-validation:provider:{self.identity_variant}",
            behavior_version="1",
            implementation_version="1",
        )

    def request_cache_policy(self, request: ModelRequest) -> CachePolicy:
        del request
        return CachePolicy(
            breakpoints=(
                CacheBreakpoint.SYSTEM_PROMPT,
                CacheBreakpoint.TOOL_DEFINITIONS,
                CacheBreakpoint.CONVERSATION_PREFIX,
            ),
            conversation_prefix_strategy="all_but_last",
        )


class _LifecycleProvider(_FixtureProvider):
    identity_variant = "lifecycle"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.parent_tool_ref: str | None = None
        self.child_tool_ref: str | None = None
        self.parent_resume_used_same_reference = False

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        request_number = len(self.requests)
        usage = {
            "input_tokens": 180 + request_number * 5,
            "output_tokens": 8,
            "cache_read_tokens": 240 if request_number > 1 else 0,
            "cache_write_tokens": 240 if request_number == 1 else 0,
        }

        if request_number == 1:
            yield ModelStreamEvent.tool_call(
                id="parent-search",
                name="search_tools",
                arguments={"query": "remember durable knowledge", "limit": 3},
            )
            yield _completed(finish_reason="tool_calls", **usage)
            return
        if request_number == 2:
            self.parent_tool_ref = str(_discovered_match(request, _TARGET_NAME)["tool_ref"])
            yield ModelStreamEvent.tool_call(
                id="parent-invoke",
                name="call_tool",
                arguments={
                    "tool_ref": self.parent_tool_ref,
                    "arguments": {"fact": "Discovery references survive ordinary resume."},
                },
            )
            yield _completed(finish_reason="tool_calls", **usage)
            return
        if request_number == 3:
            yield ModelStreamEvent.text_delta("parent initial complete")
            yield _completed(finish_reason="stop", **usage)
            return
        if request_number == 4:
            if self.parent_tool_ref is None:
                raise RuntimeError("Parent discovery reference was not captured.")
            self.parent_resume_used_same_reference = True
            yield ModelStreamEvent.tool_call(
                id="parent-resume-invoke",
                name="call_tool",
                arguments={
                    "tool_ref": self.parent_tool_ref,
                    "arguments": {"fact": "Ordinary resume preserves branch-local addressability."},
                },
            )
            yield _completed(finish_reason="tool_calls", **usage)
            return
        if request_number == 5:
            yield ModelStreamEvent.text_delta("parent resume complete")
            yield _completed(finish_reason="stop", **usage)
            return
        if request_number == 6:
            if self.parent_tool_ref is None:
                raise RuntimeError("Parent discovery reference was not captured.")
            yield ModelStreamEvent.tool_call(
                id="child-copied-parent-invoke",
                name="call_tool",
                arguments={
                    "tool_ref": self.parent_tool_ref,
                    "arguments": {"fact": "A copied parent reference must execute no work."},
                },
            )
            yield _completed(finish_reason="tool_calls", **usage)
            return
        if request_number == 7:
            rejected = _tool_results(request, "call_tool")
            if not rejected or rejected[-1].is_error is not True:
                raise RuntimeError("Child did not receive the copied-reference rejection.")
            yield ModelStreamEvent.tool_call(
                id="child-search",
                name="search_tools",
                arguments={"query": "remember durable knowledge", "limit": 3},
            )
            yield _completed(finish_reason="tool_calls", **usage)
            return
        if request_number == 8:
            self.child_tool_ref = str(_discovered_match(request, _TARGET_NAME)["tool_ref"])
            if self.child_tool_ref == self.parent_tool_ref:
                raise RuntimeError("Forked discovery reused a parent reference.")
            yield ModelStreamEvent.tool_call(
                id="child-invoke",
                name="call_tool",
                arguments={
                    "tool_ref": self.child_tool_ref,
                    "arguments": {"fact": "A fork must discover its own addressability."},
                },
            )
            yield _completed(finish_reason="tool_calls", **usage)
            return
        if request_number != 9:
            raise RuntimeError(f"Unexpected lifecycle model request {request_number}.")
        target_results = _tool_results(request, "call_tool")
        if not target_results or target_results[-1].is_error:
            raise RuntimeError("Child discovery invocation did not complete successfully.")
        yield ModelStreamEvent.text_delta("child complete")
        yield _completed(finish_reason="stop", **usage)


class _EvaluationProvider(_FixtureProvider):
    def __init__(self, strategy: Literal["direct_catalogue", "search_tools"]) -> None:
        self.strategy = strategy
        self.requests: list[ModelRequest] = []
        self.tool_ref: str | None = None

    @property
    def identity_variant(self) -> str:
        return self.strategy

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        request_number = len(self.requests)
        if self.strategy == "direct_catalogue":
            async for event in self._stream_direct(request, request_number):
                yield event
            return
        async for event in self._stream_discovery(request, request_number):
            yield event

    async def _stream_direct(
        self,
        request: ModelRequest,
        request_number: int,
    ) -> AsyncIterator[ModelStreamEvent]:
        if request_number == 1:
            yield ModelStreamEvent.tool_call(
                id="direct-invalid",
                name=_TARGET_NAME,
                arguments={},
            )
            yield _completed(
                finish_reason="tool_calls",
                input_tokens=480,
                output_tokens=10,
                cache_write_tokens=1_200,
            )
            return
        if request_number == 2:
            invalid_results = _tool_results(request, _TARGET_NAME)
            if not invalid_results or invalid_results[-1].is_error is not True:
                raise RuntimeError("Direct strategy did not observe its invalid argument result.")
            yield ModelStreamEvent.tool_call(
                id="direct-valid",
                name=_TARGET_NAME,
                arguments={"fact": "Measure the complete bounded workload."},
            )
            yield _completed(
                finish_reason="tool_calls",
                input_tokens=520,
                output_tokens=12,
                cache_read_tokens=1_200,
            )
            return
        if request_number != 3:
            raise RuntimeError(f"Unexpected direct evaluation request {request_number}.")
        valid_results = _tool_results(request, _TARGET_NAME)
        if not valid_results or valid_results[-1].is_error:
            raise RuntimeError("Direct strategy did not complete its valid target call.")
        yield ModelStreamEvent.text_delta(_EXPECTED_OUTPUT)
        yield _completed(
            finish_reason="stop",
            input_tokens=560,
            output_tokens=20,
            cache_read_tokens=1_200,
        )

    async def _stream_discovery(
        self,
        request: ModelRequest,
        request_number: int,
    ) -> AsyncIterator[ModelStreamEvent]:
        if request_number == 1:
            yield ModelStreamEvent.tool_call(
                id="evaluation-search",
                name="search_tools",
                arguments={"query": "remember durable knowledge", "limit": 3},
            )
            yield _completed(
                finish_reason="tool_calls",
                input_tokens=220,
                output_tokens=8,
                cache_write_tokens=300,
            )
            return
        if request_number == 2:
            self.tool_ref = str(_discovered_match(request, _TARGET_NAME)["tool_ref"])
            yield ModelStreamEvent.tool_call(
                id="discovery-invalid",
                name="call_tool",
                arguments={"tool_ref": self.tool_ref, "arguments": {}},
            )
            yield _completed(
                finish_reason="tool_calls",
                input_tokens=250,
                output_tokens=10,
                cache_read_tokens=300,
            )
            return
        if request_number == 3:
            invalid_results = _tool_results(request, "call_tool")
            if not invalid_results or invalid_results[-1].is_error is not True:
                raise RuntimeError("Discovery strategy did not reject invalid inner arguments.")
            yield ModelStreamEvent.tool_call(
                id="discovery-valid",
                name="call_tool",
                arguments={
                    "tool_ref": self.tool_ref,
                    "arguments": {"fact": "Measure the complete bounded workload."},
                },
            )
            yield _completed(
                finish_reason="tool_calls",
                input_tokens=280,
                output_tokens=12,
                cache_read_tokens=300,
            )
            return
        if request_number != 4:
            raise RuntimeError(f"Unexpected discovery evaluation request {request_number}.")
        valid_results = _tool_results(request, "call_tool")
        if not valid_results or valid_results[-1].is_error:
            raise RuntimeError("Discovery strategy did not complete its valid target call.")
        yield ModelStreamEvent.text_delta(_EXPECTED_OUTPUT)
        yield _completed(
            finish_reason="stop",
            input_tokens=310,
            output_tokens=20,
            cache_read_tokens=300,
        )


def _price_book() -> PriceBook:
    return PriceBook(
        price_book_version="tool-discovery-validation-fixture-v1",
        generated_at="2026-08-27T00:00:00Z",
        prices=(
            ModelPrice.fixed(
                provider_name=_PROVIDER_NAME,
                model=_MODEL,
                match="exact",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
                cache_read_input_per_million=Decimal("0.1"),
                cache_write_input_per_million=Decimal("1.25"),
            ),
        ),
    )


def _app(provider: ModelProvider) -> CayuApp:
    app = CayuApp(
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="tool-discovery-validation-fixture",
            fingerprint_key=SecretStr("fixture-tool-discovery-key-material-0001"),
        ),
        retry_policy=None,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    return app


async def _collect(events: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in events]


def _request_footprints(events: Sequence[Event]) -> tuple[RequestFootprint, ...]:
    return tuple(
        RequestFootprint.model_validate(event.payload)
        for event in events
        if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
    )


def _manifest_fingerprints(footprints: Sequence[RequestFootprint]) -> tuple[str, ...]:
    values: list[str] = []
    for footprint in footprints:
        manifest = footprint.fingerprints.tool_manifest
        if manifest.availability is not RequestFingerprintAvailability.AVAILABLE:
            raise RuntimeError("Expected an available keyed tool-manifest fingerprint.")
        if manifest.value is None:
            raise RuntimeError("Available tool-manifest fingerprint did not carry a value.")
        values.append(manifest.value)
    return tuple(values)


def _event_count(events: Sequence[Event], event_type: EventType) -> int:
    return sum(event.type is event_type for event in events)


async def _run_lifecycle() -> ToolDiscoveryLifecycleEvidence:
    provider = _LifecycleProvider()
    target = _RememberKnowledgeTool()
    app = _app(provider)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model=_MODEL,
            system_prompt="Use provider-neutral discovery for hidden capabilities.",
        ),
        tools=_catalogue_tools(target),
        tool_discovery_mode="search_tools",
    )

    initial_events = await _collect(
        app.run(
            RunRequest(
                agent_name="assistant",
                session_id=_PARENT_SESSION_ID,
                messages=[Message.text("user", "Find a tool and save the first lesson.")],
                max_steps=4,
            )
        )
    )
    parent_before_resume = await app.inspect_tool_discovery_view(_PARENT_SESSION_ID)
    resume_events = await _collect(
        app.resume(
            ResumeRequest(
                session_id=_PARENT_SESSION_ID,
                messages=[Message.text("user", "Reuse the discovered capability.")],
                max_steps=4,
            )
        )
    )
    parent_after_resume = await app.inspect_tool_discovery_view(_PARENT_SESSION_ID)
    fork_events = await _collect(
        app.fork_session(
            ForkSessionRequest(
                source_session_id=_PARENT_SESSION_ID,
                session_id=_CHILD_SESSION_ID,
            )
        )
    )
    child_before_search = await app.inspect_tool_discovery_view(_CHILD_SESSION_ID)
    child_events = await _collect(
        app.resume(
            ResumeRequest(
                session_id=_CHILD_SESSION_ID,
                messages=[
                    Message.text(
                        "user",
                        "Try the copied reference, then discover and use your own capability.",
                    )
                ],
                max_steps=4,
            )
        )
    )
    child_after_search = await app.inspect_tool_discovery_view(_CHILD_SESSION_ID)
    parent_final = await app.inspect_tool_discovery_view(_PARENT_SESSION_ID)

    all_events = (*initial_events, *resume_events, *fork_events, *child_events)
    model_events = (*initial_events, *resume_events, *child_events)
    footprints = _request_footprints(model_events)
    manifests = _manifest_fingerprints(footprints)
    grant_counts = tuple(
        0 if footprint.tool_discovery_view is None else footprint.tool_discovery_view.grant_count
        for footprint in footprints
    )
    rejection_events = tuple(
        event
        for event in child_events
        if event.type is EventType.TARGETED_TOOL_REFERENCE_REJECTED
        and event.payload.get("authority_kind") == "tool_discovery"
        and event.payload.get("rejection_reason") == "unknown"
    )
    core_shapes = {tuple(tool["name"] for tool in request.tools) for request in provider.requests}
    if core_shapes != {_CORE_TOOL_NAMES}:
        raise RuntimeError("Discovery lifecycle did not preserve the stable provider tool core.")
    if parent_final != parent_after_resume:
        raise RuntimeError("Child discovery mutated the parent tool view.")
    public_evidence = json.dumps(
        {
            "events": [event.model_dump(mode="json") for event in all_events],
            "parent_view": parent_final.model_dump(mode="json"),
            "child_view": child_after_search.model_dump(mode="json"),
        },
        sort_keys=True,
    )
    captured_references = tuple(
        reference
        for reference in (provider.parent_tool_ref, provider.child_tool_ref)
        if reference is not None
    )

    selected_event_types = (
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FORKED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TARGETED_TOOL_REFERENCE_REJECTED,
        EventType.REQUEST_FOOTPRINT_RECORDED,
    )
    return ToolDiscoveryLifecycleEvidence(
        provider_requests=len(provider.requests),
        model_steps=_event_count(model_events, EventType.MODEL_COMPLETED),
        stable_core_tool_names=_CORE_TOOL_NAMES,
        distinct_tool_manifest_fingerprints=len(set(manifests)),
        view_grant_counts_by_request=grant_counts,
        parent_view_revision_before_resume=parent_before_resume.revision,
        parent_view_revision_after_resume=parent_after_resume.revision,
        child_view_revision_before_search=child_before_search.revision,
        child_view_revision_after_search=child_after_search.revision,
        parent_grant_count=parent_final.grant_count,
        child_grant_count_before_search=child_before_search.grant_count,
        child_grant_count_after_search=child_after_search.grant_count,
        generation_changed_on_fork=(
            parent_after_resume.generation_id != child_before_search.generation_id
        ),
        parent_reference_survived_resume=provider.parent_resume_used_same_reference,
        copied_parent_reference_rejections=len(rejection_events),
        child_reference_was_fresh=(
            provider.parent_tool_ref is not None
            and provider.child_tool_ref is not None
            and provider.parent_tool_ref != provider.child_tool_ref
        ),
        references_omitted_from_public_evidence=(
            len(captured_references) == 2
            and all(reference not in public_evidence for reference in captured_references)
        ),
        target_effects=len(target.effects),
        typed_event_counts={
            event_type.value: _event_count(all_events, event_type)
            for event_type in selected_event_types
        },
    )


def _descriptor_for_tool(tool: Tool) -> ToolDescriptor:
    return build_tool_descriptor(
        name=tool.name,
        description=tool.description,
        input_schema=tool.schema,
        parallel_safe=tool.spec.parallel_safe,
        effect=tool.spec.effect,
        publishes_arguments=True,
        workspace_mutation=tool.spec.workspace_mutation,
    )


def _ranking_evidence() -> ToolDiscoveryRankingEvidence:
    tools = _catalogue_tools(_RememberKnowledgeTool())
    catalogue = build_tool_catalog_snapshot(tuple(_descriptor_for_tool(tool) for tool in tools))
    ceiling = ToolCapabilityCeiling(tool_names=tuple(tool.name for tool in tools))
    cases = (
        ("remember_knowledge", _TARGET_NAME),
        ("durable knowledge", _TARGET_NAME),
        ("fact", _TARGET_NAME),
        (_TARGET_TOOL_ID, _TARGET_NAME),
        ("search saved notes", "search_saved_notes"),
        ("report body", "publish_incident_report"),
    )
    results: list[ToolDiscoveryRankingCase] = []
    reciprocal_rank = Decimal("0")
    top_1_hits = 0
    for query, expected_name in cases:
        names = tuple(
            descriptor.name
            for descriptor in search_tool_descriptors(
                query,
                catalogue=catalogue,
                ceiling=ceiling,
            )[:8]
        )
        rank = names.index(expected_name) + 1 if expected_name in names else None
        if rank is not None:
            reciprocal_rank += Decimal(1) / Decimal(rank)
        if rank == 1:
            top_1_hits += 1
        results.append(
            ToolDiscoveryRankingCase(
                query=query,
                expected_tool_name=expected_name,
                observed_rank=rank,
                returned_tool_names=names,
            )
        )

    excluded = search_tool_descriptors(
        "service health status",
        catalogue=catalogue,
        ceiling=ceiling,
        excluded_names=("read_service_status",),
    )
    return ToolDiscoveryRankingEvidence(
        case_count=len(results),
        top_1_hits=top_1_hits,
        mean_reciprocal_rank=reciprocal_rank / Decimal(len(results)),
        cases=tuple(results),
        directly_exposed_tool_excluded=all(
            descriptor.name != "read_service_status" for descriptor in excluded
        ),
    )


def _target_error_result(event: Event) -> bool:
    if event.tool_name != _TARGET_NAME or event.type not in {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
    }:
        return False
    result = event.payload.get("result")
    return isinstance(result, dict) and result.get("is_error") is True


async def _run_evaluation_side(
    strategy: Literal["direct_catalogue", "search_tools"],
) -> ToolDiscoveryEvaluationSide:
    provider = _EvaluationProvider(strategy)
    target = _RememberKnowledgeTool()
    app = _app(provider)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model=_MODEL,
            system_prompt="Complete the fixed tool-use workload and report quality-ok.",
        ),
        tools=_catalogue_tools(target),
        tool_discovery_mode="search_tools" if strategy == "search_tools" else None,
    )
    session_id = _DIRECT_SESSION_ID if strategy == "direct_catalogue" else _DISCOVERY_SESSION_ID
    started_at = perf_counter_ns()
    events = await _collect(
        app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "Save the fixture lesson and finish the task.")],
                max_steps=4,
            )
        )
    )
    elapsed_ms = Decimal(perf_counter_ns() - started_at) / Decimal(1_000_000)
    transcript = await app.session_store.load_transcript(session_id)
    footprints = _request_footprints(events)
    manifests = _manifest_fingerprints(footprints)
    usage = await app.get_session_usage(session_id)
    cost = await app.get_session_cost(session_id, _price_book())
    search_calls = sum(
        event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        and event.tool_name == "search_tools"
        for event in events
    )
    discovery_invalid = sum(
        event.type is EventType.TARGETED_TOOL_REFERENCE_REJECTED
        and event.payload.get("authority_kind") == "tool_discovery"
        and event.payload.get("rejection_reason") == "invalid_arguments"
        for event in events
    )
    direct_invalid = sum(_target_error_result(event) for event in events)
    expected_searches = 1 if strategy == "search_tools" else 0
    return ToolDiscoveryEvaluationSide(
        strategy=strategy,
        provider_requests=len(provider.requests),
        model_steps=_event_count(events, EventType.MODEL_COMPLETED),
        search_calls=search_calls,
        unnecessary_searches=max(0, search_calls - expected_searches),
        invalid_argument_attempts=1,
        invalid_argument_rejections=discovery_invalid + direct_invalid,
        target_invocations_started=sum(
            event.type is EventType.TOOL_CALL_STARTED and event.tool_name == _TARGET_NAME
            for event in events
        ),
        target_effects=len(target.effects),
        approval_requests=_event_count(events, EventType.TOOL_CALL_APPROVAL_REQUESTED),
        provider_tool_counts=tuple(len(request.tools) for request in provider.requests),
        distinct_tool_manifest_fingerprints=len(set(manifests)),
        input_tokens=usage.usage.input_tokens,
        output_tokens=usage.usage.output_tokens,
        cache_read_input_tokens=usage.usage.cache.read_tokens,
        cache_write_input_tokens=usage.usage.cache.write_tokens,
        uncached_input_tokens=usage.usage.cache.uncached_input_tokens,
        observed_latency_ms=elapsed_ms,
        quality_passed=final_output_text(transcript) == _EXPECTED_OUTPUT,
        estimated_cost=cost.total_cost,
        currency=cost.currency,
    )


async def run_scenario() -> ToolDiscoveryValidationReport:
    """Run the lifecycle proof and bounded deterministic evaluation."""

    lifecycle = await _run_lifecycle()
    direct_catalogue = await _run_evaluation_side("direct_catalogue")
    search_tools = await _run_evaluation_side("search_tools")
    return ToolDiscoveryValidationReport(
        lifecycle=lifecycle,
        ranking=_ranking_evidence(),
        direct_catalogue=direct_catalogue,
        search_tools=search_tools,
    )


__all__ = [
    "ToolDiscoveryEvaluationSide",
    "ToolDiscoveryLifecycleEvidence",
    "ToolDiscoveryRankingCase",
    "ToolDiscoveryRankingEvidence",
    "ToolDiscoveryValidationReport",
    "run_scenario",
]
