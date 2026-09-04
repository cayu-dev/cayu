from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

from cayu import (
    AgentSpec,
    ArtifactExternalizingToolResultPolicy,
    ArtifactStore,
    ArtifactStoreUnavailableError,
    ArtifactWriteSettlementEvidence,
    ArtifactWriteSettlementFailureCode,
    ArtifactWriteSettlementObservation,
    ArtifactWriteSettlementPhase,
    ArtifactWriteSettlementStatus,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    Message,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ResumeRequest,
    RunRequest,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    artifact_store_identity_sha256,
    record_artifact_write_settlement,
)
from cayu.runtime import (
    RuntimePublicationRequest,
    RuntimePublicationResult,
    SessionStatus,
    SessionStore,
)


class _ConformanceProvider(ModelProvider):
    name = "projection-conformance"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(
                id="call_projection_conformance",
                name="projection_conformance_tool",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _ConformanceTool(Tool):
    spec = ToolSpec(
        name="projection_conformance_tool",
        input_schema={"type": "object"},
        effect=ToolEffect.NONE,
    )

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx, args
        self.calls += 1
        return ToolResult(content=self.content)


class _OrphaningArtifactStore(ArtifactStore):
    def __init__(self, delegate: ArtifactStore) -> None:
        self.delegate = delegate
        self.id = delegate.id
        self.artifact_id: str | None = None

    async def put_bytes(self, content: bytes, *, artifact_id: str | None = None, **kwargs: Any):
        del content, kwargs
        assert artifact_id is not None
        self.artifact_id = artifact_id
        observed_at = datetime.now(UTC)
        evidence = ArtifactWriteSettlementEvidence(
            operation_id="artifact_write_11111111111111111111111111111111",
            artifact_id=artifact_id,
            store_identity_sha256=artifact_store_identity_sha256(self.id),
            status=ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED,
            phase=ArtifactWriteSettlementPhase.COMMIT,
            observation=ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
            started_at=observed_at,
            observed_at=observed_at,
            elapsed_ms=0,
            failure_codes=(ArtifactWriteSettlementFailureCode.COMMIT_FAILED,),
        )
        error = ArtifactStoreUnavailableError("Artifact commit could not be reconciled.")
        record_artifact_write_settlement(evidence, error=error)
        raise error

    async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
        return await self.delegate.read_bytes(artifact_id, max_bytes=max_bytes)

    async def list(self, **kwargs: Any):
        return await self.delegate.list(**kwargs)

    async def delete(self, artifact_id: str) -> None:
        await self.delegate.delete(artifact_id)


async def assert_tool_result_projection_session_store_conformance(
    session_store: SessionStore,
    artifact_store: ArtifactStore,
    *,
    session_id: str,
) -> None:
    original = "session-store-conformance-" + ("s" * 10_000)
    provider = _ConformanceProvider()
    app = CayuApp(
        enable_logging=False,
        session_store=session_store,
        tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
            max_inline_bytes=1024,
            max_inline_token_estimate=None,
            preview_bytes=32,
        ),
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=artifact_store),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[_ConformanceTool(original)],
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                session_id=session_id,
                agent_name="assistant",
                messages=[Message.text("user", "run")],
            )
        )
    ]

    assert events[-1].type == EventType.SESSION_COMPLETED
    transcript = await session_store.load_transcript(session_id)
    tool_result = next(
        part
        for message in transcript
        if message.role == "tool"
        for part in message.content
        if part.tool_call_id == "call_projection_conformance"
    )
    assert original not in tool_result.content
    assert original not in provider.requests[1].model_dump_json()
    reference = next(
        item for item in tool_result.artifacts if item.get("type") == "cayu.tool_result_artifact.v1"
    )
    persisted = await artifact_store.read_bytes(reference["artifact_id"])
    assert persisted.content.decode() == original


async def assert_tool_result_projection_orphan_evidence_conformance(
    session_store: SessionStore,
    artifact_store: ArtifactStore,
    *,
    session_id: str,
) -> None:
    original = "session-store-orphan-" + ("o" * 10_000)
    provider = _ConformanceProvider()
    orphaning_store = _OrphaningArtifactStore(artifact_store)
    app = CayuApp(
        enable_logging=False,
        session_store=session_store,
        tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
            max_inline_bytes=1024,
            max_inline_token_estimate=None,
            preview_bytes=32,
        ),
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=orphaning_store),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[_ConformanceTool(original)],
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                session_id=session_id,
                agent_name="assistant",
                messages=[Message.text("user", "run")],
            )
        )
    ]

    assert events[-1].type is EventType.SESSION_COMPLETED
    durable_events = await session_store.load_events(session_id)
    public_terminal = next(event for event in events if event.type is EventType.TOOL_CALL_COMPLETED)
    durable_terminal = next(
        event for event in durable_events if event.type is EventType.TOOL_CALL_COMPLETED
    )
    for terminal in (public_terminal, durable_terminal):
        projection = terminal.payload["tool_result_projection"]
        settlement = projection["artifact_write_settlement"]
        assert projection["status"] == "failed"
        assert "artifact_id" not in projection
        assert settlement["status"] == "reconciliation_required"
        assert settlement["artifact_id"] == orphaning_store.artifact_id
        assert terminal.payload["result"]["artifacts"] == []
    assert orphaning_store.artifact_id is not None
    assert orphaning_store.artifact_id not in provider.requests[1].model_dump_json()
    transcript = await session_store.load_transcript(session_id)
    tool_result = next(
        part
        for message in transcript
        if message.role == "tool"
        for part in message.content
        if part.tool_call_id == "call_projection_conformance"
    )
    assert tool_result.artifacts == []


async def assert_tool_result_projection_recovery_conformance(
    session_store: SessionStore,
    artifact_store: ArtifactStore,
    *,
    session_id: str,
) -> None:
    original = "session-store-recovery-" + ("r" * 10_000)
    provider = _ConformanceProvider()
    tool = _ConformanceTool(original)
    app = CayuApp(
        enable_logging=False,
        session_store=session_store,
        tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
            max_inline_bytes=1024,
            max_inline_token_estimate=None,
            preview_bytes=32,
        ),
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=artifact_store),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )
    original_publish = session_store.publish_runtime_publication
    publication_attempts = 0

    async def reject_first_tool_round(
        session_id_value: str,
        *,
        request: RuntimePublicationRequest,
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> RuntimePublicationResult:
        nonlocal publication_attempts
        if request.kind == "tool-round":
            publication_attempts += 1
            if publication_attempts == 1:
                raise RuntimeError("tool-round publication rejected before commit")
        return await original_publish(
            session_id_value,
            request=request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    with patch.object(
        session_store,
        "publish_runtime_publication",
        new=reject_first_tool_round,
    ):
        first_events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        assert first_events[-1].type == EventType.SESSION_FAILED
        first_artifacts = await artifact_store.list(session_id=session_id)
        assert len(first_artifacts.artifacts) == 1

        resumed_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                )
            )
        ]

    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert publication_attempts == 2
    assert tool.calls == 1
    replayed_artifacts = await artifact_store.list(session_id=session_id)
    assert replayed_artifacts.artifacts == first_artifacts.artifacts
    transcript = await session_store.load_transcript(session_id)
    tool_result = next(
        part
        for message in transcript
        if message.role == "tool"
        for part in message.content
        if part.tool_call_id == "call_projection_conformance"
    )
    assert original not in tool_result.content
    assert original not in provider.requests[1].model_dump_json()
