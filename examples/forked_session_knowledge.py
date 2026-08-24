"""Run one forked memory session concurrently with its continuing parent.

The credential-free provider makes the dynamic-tool and cache-prefix shape
inspectable. Cayu's session store owns both branch transcripts; no provider-side
conversation state is required. Real applications use their normal model
provider and own the memory prompt, scheduling, finding criteria, and grant
policy.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    Event,
    ForkSessionRequest,
    InMemoryKnowledgeStore,
    InMemorySessionStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeReviewWorkflow,
    KnowledgeStatus,
    Message,
    RememberKnowledgeTool,
    ResumeRequest,
    RunRequest,
    StaticToolExposurePolicy,
    TargetedToolGrant,
    TargetedToolGrantInspection,
    TextPart,
    ToolPolicy,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent

PARENT_SESSION_ID = "session-knowledge-source"
CHILD_SESSION_ID = "session-knowledge-child"
KNOWLEDGE_TOOL_ID = "cayu:remember_knowledge"
KNOWLEDGE_GRANT_REQUEST_ID = "save-stable-gotchas"
DEFAULT_FINDINGS = (
    "Retrying a targeted tool call must retain its original outer call identity.",
    "A fork must receive a fresh targeted grant instead of inheriting parent references.",
)

_TARGETED_CONTEXT_SCHEMA = "cayu.targeted-tool-context.v1"


@dataclass(frozen=True)
class ForkedKnowledgeDemo:
    """Application components retained for inspection by the example and tests."""

    app: CayuApp
    provider: DeterministicKnowledgeProvider
    session_store: InMemorySessionStore
    knowledge_store: InMemoryKnowledgeStore


@dataclass(frozen=True)
class ForkedKnowledgeResult:
    """Observable result of one parent/fork/background-memory workflow."""

    parent_initial_events: tuple[Event, ...]
    fork_events: tuple[Event, ...]
    parent_continuation_events: tuple[Event, ...]
    memory_events: tuple[Event, ...]
    pending_entries: tuple[KnowledgeEntry, ...]
    child_grants: tuple[TargetedToolGrantInspection, ...]


class DeterministicKnowledgeProvider(ModelProvider):
    """Simulate one parent and one memory child without provider credentials."""

    name = "deterministic-forked-knowledge"

    def __init__(self, findings: Iterable[str]) -> None:
        if isinstance(findings, str | bytes):
            raise TypeError("findings must be an iterable of strings.")
        copied: list[str] = []
        for finding in findings:
            if type(finding) is not str or not finding.strip():
                raise ValueError("findings must contain nonblank strings.")
            copied.append(finding)
        self.findings = tuple(copied)
        self.requests: list[ModelRequest] = []
        self.max_concurrent_requests = 0
        self._active_requests = 0
        self._initial_parent_completed = False
        self._parallel_request_count = 0
        self._parallel_gate = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        self._active_requests += 1
        self.max_concurrent_requests = max(
            self.max_concurrent_requests,
            self._active_requests,
        )
        try:
            if not self._initial_parent_completed:
                self._initial_parent_completed = True
                yield ModelStreamEvent.text_delta("Implementation complete.")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return

            descriptor = _targeted_descriptor(request, tool_id=KNOWLEDGE_TOOL_ID)
            contains_tool_result = _contains_tool_result(request)
            if not contains_tool_result:
                await self._join_parallel_pair()

            if contains_tool_result:
                yield ModelStreamEvent.text_delta("Knowledge pass complete.")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            if descriptor is None:
                yield ModelStreamEvent.text_delta("Parent continued independently.")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return

            for index, finding in enumerate(self.findings, start=1):
                yield ModelStreamEvent.tool_call(
                    id=f"knowledge-finding-{index}",
                    name="call_tool",
                    arguments={
                        "tool_ref": descriptor["tool_ref"],
                        "arguments": {
                            "text": finding,
                            "title": f"Forked knowledge finding {index}",
                            "kind": "warning",
                            "aspects": ["forked-knowledge"],
                        },
                    },
                )
            if self.findings:
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            yield ModelStreamEvent.text_delta("No durable knowledge found.")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
        finally:
            self._active_requests -= 1

    async def _join_parallel_pair(self) -> None:
        """Hold each branch until both first post-fork requests are in flight."""

        self._parallel_request_count += 1
        if self._parallel_request_count == 2:
            self._parallel_gate.set()
        await self._parallel_gate.wait()


def build_demo(
    *,
    findings: Iterable[str] = DEFAULT_FINDINGS,
    tool_policy: ToolPolicy | None = None,
) -> ForkedKnowledgeDemo:
    """Build the provider-independent forked-memory composition."""

    access_scope = KnowledgeAccessScope.for_namespace(
        "default",
        allowed_statuses=[KnowledgeStatus.PENDING, KnowledgeStatus.ACTIVE],
    )
    knowledge_store = InMemoryKnowledgeStore(access_scope=access_scope)
    session_store = InMemorySessionStore()
    provider = DeterministicKnowledgeProvider(findings)
    app = CayuApp(
        session_store=session_store,
        knowledge_store=knowledge_store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="knowledge-local"),
            knowledge_store=knowledge_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="deterministic-model",
            system_prompt="Complete application work carefully and preserve durable context.",
        ),
        tools=(RememberKnowledgeTool(),),
        tool_exposure_policy=StaticToolExposurePolicy(
            profile_id="targeted-only",
            tools=(),
        ),
        enable_tool_gateway=True,
        tool_policy=tool_policy,
    )
    return ForkedKnowledgeDemo(
        app=app,
        provider=provider,
        session_store=session_store,
        knowledge_store=knowledge_store,
    )


async def run_demo(
    demo: ForkedKnowledgeDemo,
    *,
    max_calls: int = 3,
    parent_session_id: str = PARENT_SESSION_ID,
    child_session_id: str = CHILD_SESSION_ID,
) -> ForkedKnowledgeResult:
    """Fork one memory child, then run it concurrently with its parent."""

    parent_initial_events = await _collect_events(
        demo.app.run(
            RunRequest(
                agent_name="assistant",
                session_id=parent_session_id,
                messages=[Message.text("user", "Implement the requested change.")],
            )
        )
    )
    fork_events = await _collect_events(
        demo.app.fork_session(
            ForkSessionRequest(
                source_session_id=parent_session_id,
                session_id=child_session_id,
            )
        )
    )

    parent_continuation, memory_pass = await asyncio.gather(
        _collect_events(
            demo.app.resume(
                ResumeRequest(
                    session_id=parent_session_id,
                    messages=[Message.text("user", "Continue with the next task.")],
                )
            )
        ),
        _collect_events(
            demo.app.resume(
                ResumeRequest(
                    session_id=child_session_id,
                    messages=[
                        Message.text(
                            "user",
                            "Review the inherited work for reusable gotchas. Save each "
                            "durable finding with the provided knowledge capability.",
                        )
                    ],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id=KNOWLEDGE_GRANT_REQUEST_ID,
                            tool_id=KNOWLEDGE_TOOL_ID,
                            max_calls=max_calls,
                            lifetime_seconds=300,
                            origin="forked-session-knowledge-example",
                        ),
                    ),
                )
            )
        ),
    )

    reviewer = KnowledgeReviewWorkflow(demo.knowledge_store, namespace="default")
    pending = await reviewer.list_pending(
        source_type="tool",
        source_id=child_session_id,
        limit=max_calls,
    )
    return ForkedKnowledgeResult(
        parent_initial_events=parent_initial_events,
        fork_events=fork_events,
        parent_continuation_events=parent_continuation,
        memory_events=memory_pass,
        pending_entries=tuple(item.entry for item in pending.entries),
        child_grants=await demo.app.inspect_targeted_tool_grants(child_session_id),
    )


async def _collect_events(stream: AsyncIterator[Event]) -> tuple[Event, ...]:
    return tuple([event async for event in stream])


def _contains_tool_result(request: ModelRequest) -> bool:
    return any(
        part.type == "tool_result" for message in request.messages for part in message.content
    )


def _targeted_descriptor(request: ModelRequest, *, tool_id: str) -> dict | None:
    """Read Cayu's model-facing suffix inside the deterministic provider fixture."""

    for message in reversed(request.messages):
        for part in message.content:
            if not isinstance(part, TextPart):
                continue
            text = part.text
            if _TARGETED_CONTEXT_SCHEMA not in text:
                continue
            _instruction, separator, payload_text = text.rpartition("\n")
            if not separator:
                break
            payload = json.loads(payload_text)
            for descriptor in payload.get("tools", []):
                if descriptor.get("tool_id") == tool_id:
                    return descriptor
    return None


async def main() -> None:
    demo = build_demo()
    result = await run_demo(demo)
    summary = {
        "parent_session_id": PARENT_SESSION_ID,
        "child_session_id": CHILD_SESSION_ID,
        "max_concurrent_provider_requests": demo.provider.max_concurrent_requests,
        "provider_tool_sets": [
            [tool["name"] for tool in request.tools] for request in demo.provider.requests
        ],
        "grant_usage": [
            {
                "tool_id": grant.tool_id,
                "used_calls": grant.used_calls,
                "remaining_calls": grant.remaining_calls,
            }
            for grant in result.child_grants
        ],
        "pending_knowledge": [
            {
                "text": entry.text,
                "status": entry.status.value,
                "source_session_id": entry.source_id,
            }
            for entry in result.pending_entries
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
