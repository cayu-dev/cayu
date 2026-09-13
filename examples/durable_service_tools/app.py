"""Credential-free service reconstruction over one durable Cayu session."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    ExecutionProfileBehaviorIdentity,
    IncompleteSessionRecoveryRequest,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeQuery,
    KnowledgeSearchResult,
    KnowledgeStore,
    Message,
    ModelStreamEvent,
    PendingActionQuery,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolContext,
    ToolEffect,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    ToolResult,
    ToolSpec,
)


def identity(name: str, version: str = "1") -> ExecutionProfileBehaviorIdentity:
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version=version, implementation_version="1"
    )


class HandbookReader(Tool):
    def __init__(
        self,
        store: KnowledgeStore | None,
        *,
        environment: bool = False,
        version: str = "1",
        opaque: bool = False,
    ) -> None:
        if not isinstance(store, KnowledgeStore) or store.bound_access_scope() is None:
            raise ValueError("HandbookReader requires the supplied, bound knowledge store")
        self.store = store
        self.environment = environment
        self.spec = ToolSpec(
            name="lookup_handbook",
            description="Search the authorized handbook.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            effect=ToolEffect.NONE,
            execution_profile_identity=None if opaque else identity("handbook-reader", version),
        )
        super().__init__()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        if self.environment:
            # The selected environment supplies these handles, not CayuApp's
            # application-level knowledge configuration.
            assert ctx.knowledge_store is self.store
            assert ctx.knowledge_access_scope == self.store.bound_access_scope()
            found = await ctx.knowledge_store.search(
                KnowledgeQuery(text="shipping", namespace="handbook"),
                access_scope=ctx.knowledge_access_scope,
            )
        else:
            assert ctx.knowledge_store is None
            found = await self.store.search(KnowledgeQuery(text="shipping", namespace="handbook"))
        payload = found.model_dump(mode="json")
        return ToolResult(content=json.dumps(payload), structured=payload)


class TrackedKnowledge(InMemoryKnowledgeStore):
    """A supplied scoped service with current evidence and forbidden decoys."""

    def __init__(self, evidence: str) -> None:
        super().__init__(
            [
                KnowledgeEntry(
                    id="shipping",
                    namespace="handbook",
                    text=f"shipping {evidence}",
                    labels={"organization": "demo"},
                ),
                KnowledgeEntry(
                    id="foreign",
                    namespace="handbook",
                    text="shipping foreign-secret",
                    labels={"organization": "foreign"},
                ),
                KnowledgeEntry(
                    id="unrelated",
                    namespace="unrelated",
                    text="shipping unrelated-secret",
                    labels={"organization": "demo"},
                ),
            ],
            # Trusted application configuration, never a model argument.
            access_scope=KnowledgeAccessScope(
                allowed_namespaces=["handbook"], required_labels={"organization": "demo"}
            ),
        )
        self.calls = 0

    async def search(
        self,
        query: KnowledgeQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        self.calls += 1
        return await super().search(query, access_scope=access_scope)


class HandbookPolicy(ToolPolicy):
    def __init__(self, version: str = "1") -> None:
        self.version = version

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return identity("handbook-policy", self.version)

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        decision = {
            "lookup_handbook": ToolPolicyDecision.ALLOW,
            "record_receipt": ToolPolicyDecision.REQUIRE_APPROVAL,
        }.get(request.tool_name, ToolPolicyDecision.DENY)
        return ToolPolicyResult(decision=decision)


class RecordReceipt(Tool):
    """Bounded local effect: one atomic SQLite insert keyed by a fixed action."""

    spec = ToolSpec(
        name="record_receipt",
        description="Record the demo receipt after approval.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        effect=ToolEffect.IDEMPOTENT,
        parallel_safe=False,
        execution_profile_identity=identity("demo-receipt"),
    )

    def __init__(self, database: Path) -> None:
        super().__init__()
        self.database = database

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        connection = sqlite3.connect(self.database)
        try:
            with connection:
                connection.execute("CREATE TABLE IF NOT EXISTS receipts (action TEXT PRIMARY KEY)")
                connection.execute("INSERT OR IGNORE INTO receipts VALUES ('demo-action')")
        finally:
            connection.close()
        return ToolResult(content="demo-action recorded")


async def exercise(
    phase: str,
    state: Path,
    *,
    wiring: str = "injected",
    version: str = "1",
    policy_version: str = "1",
    environment_version: str = "1",
    opaque: bool = False,
) -> None:
    state.mkdir(parents=True, exist_ok=True)
    knowledge = TrackedKnowledge(f"evidence-{phase}")
    reader = HandbookReader(
        knowledge, environment=wiring == "environment", version=version, opaque=opaque
    )
    resolving = phase in {"approve", "deny", "repeat"}
    batches = [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]]
    if not resolving:
        batches.insert(
            0,
            [
                ModelStreamEvent.tool_call(
                    id=f"call-{phase}",
                    name="record_receipt" if phase == "pause" else "lookup_handbook",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        )
    provider = ScriptedModelProvider(batches)
    session_store = SQLiteSessionStore(state / "sessions.sqlite")
    app = CayuApp(
        session_store=session_store,
        # Deliberately also configured here: injection still must be explicit.
        knowledge_store=knowledge,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    if wiring == "environment":
        app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="handbook",
                    execution_profile_identity=identity(
                        "handbook-environment", environment_version
                    ),
                ),
                knowledge_store=knowledge,
                knowledge_access_scope=knowledge.bound_access_scope(),
            )
        )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted"),
        tools=[reader, RecordReceipt(state / "effects.sqlite")],
        tool_policy=HandbookPolicy(policy_version),
    )
    try:
        if resolving:
            if phase == "repeat":
                approval = ToolApprovalRequest.model_validate_json(
                    (state / "approval.json").read_text()
                )
            else:
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id="service-example")
                )
                pending = await app.session_store.query_pending_actions(
                    PendingActionQuery(session_id="service-example")
                )
                assert len(pending.actions) == 1
                action = pending.actions[0]
                assert action.approval_id is not None
                assert action.round_id is not None
                assert action.tool_call_id is not None
                approval = ToolApprovalRequest(
                    session_id="service-example",
                    approval_id=action.approval_id,
                    tool_round_id=action.round_id,
                    tool_call_id=action.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE
                    if phase == "approve"
                    else ToolApprovalDecision.DENY,
                    reason="trusted local demo operator",
                )
                (state / "approval.json").write_text(approval.model_dump_json())
            stream = app.resolve_tool_approval(approval)
        elif phase in {"start", "pause"}:
            stream = app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="service-example",
                    environment_name="handbook" if wiring == "environment" else None,
                    messages=[Message.text("user", "remember-history")],
                )
            )
        else:
            stream = app.resume(
                ResumeRequest(
                    session_id="service-example", messages=[Message.text("user", "lookup again")]
                )
            )
        events = [event async for event in stream]
        if phase in {"start", "resume"}:
            rendered = json.dumps(
                [m.model_dump(mode="json") for m in provider.requests[-1].messages]
            )
            assert knowledge.calls == 1
            assert f"evidence-{phase}" in rendered
            assert "foreign-secret" not in rendered and "unrelated-secret" not in rendered
            assert "remember-history" in rendered
        print(json.dumps({"phase": phase, "events": len(events), "searches": knowledge.calls}))
    finally:
        # Persist observations even when admission fails, for the subprocess tests.
        (state / f"{phase}-observations.json").write_text(
            json.dumps(
                {
                    "searches": knowledge.calls,
                    "provider_requests": len(provider.requests),
                }
            )
        )
        await session_store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["start", "resume", "pause", "approve", "deny", "repeat"])
    parser.add_argument("state", type=Path)
    parser.add_argument("--wiring", choices=["injected", "environment"], default="injected")
    parser.add_argument("--version", default="1")
    parser.add_argument("--policy-version", default="1")
    parser.add_argument("--environment-version", default="1")
    parser.add_argument("--opaque", action="store_true")
    asyncio.run(exercise(**vars(parser.parse_args())))
