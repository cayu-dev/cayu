from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import warnings
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import SecretStr
from tests.core.tool_result_projection_conformance import (
    assert_tool_result_projection_orphan_evidence_conformance,
    assert_tool_result_projection_recovery_conformance,
    assert_tool_result_projection_session_store_conformance,
)

import cayu.artifacts.local as local_artifacts
from cayu import (
    MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES,
    MAX_TOOL_RESULT_ARTIFACT_REFERENCE_BYTES,
    MAX_TOOL_RESULT_PREVIEW_BYTES,
    AgentSpec,
    ArtifactExternalizingToolResultPolicy,
    ArtifactStoreUnavailableError,
    ArtifactWriteSettlementEvidence,
    ArtifactWriteSettlementFailureCode,
    ArtifactWriteSettlementObservation,
    ArtifactWriteSettlementPhase,
    ArtifactWriteSettlementStatus,
    CayuApp,
    CayuConfig,
    Environment,
    EnvironmentSpec,
    Event,
    EventRecord,
    EventType,
    LocalArtifactStore,
    McpInitializeResult,
    McpResourceResult,
    McpServerSpec,
    McpSession,
    McpToolDefinition,
    McpToolResult,
    McpToolset,
    Message,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ReadFileTool,
    ResumeRequest,
    RunRequest,
    SQLiteSessionStore,
    Tool,
    ToolContext,
    ToolEffect,
    ToolExecutionConfig,
    ToolResult,
    ToolResultProjection,
    ToolResultProjectionPolicy,
    ToolResultProjectionRecord,
    ToolResultProjectionRequest,
    ToolResultProjectionStatus,
    ToolSpec,
    artifact_store_identity_sha256,
    record_artifact_write_settlement,
    register_artifact_write_operation,
)
from cayu.core.events import event_with_runtime_nested_payload_authority
from cayu.runtime import (
    InMemorySessionStore,
    InterruptSessionRequest,
    RuntimePublicationRequest,
    RuntimePublicationResult,
    SessionIdentity,
    SessionStatus,
)
from cayu.runtime.public_authority import (
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
)
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class _FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self, events: list[list[ModelStreamEvent]]) -> None:
        self.events = events
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self.events[len(self.requests) - 1]:
            yield event


class _ResultTool(Tool):
    spec = ToolSpec(
        name="result_tool",
        description="Return configured text.",
        input_schema={"type": "object"},
        effect=ToolEffect.NONE,
    )

    def __init__(self, result: ToolResult) -> None:
        self.result = result
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx, args
        self.calls += 1
        return self.result


class _InvalidConstructedResultTool(Tool):
    spec = ToolSpec(
        name="invalid_constructed_result",
        description="Return a result that fails post-execution validation.",
        input_schema={"type": "object"},
        effect=ToolEffect.EXTERNAL,
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx, args

        class _InvalidStructured(dict):
            def items(self):
                raise RuntimeError("tool result traversal should not run")

        return ToolResult.model_construct(
            content="ok",
            structured=_InvalidStructured({"bad": "value"}),
            artifacts=[],
            is_error=False,
        )


class _FailingArtifactStore(LocalArtifactStore):
    async def put_bytes(self, content: bytes, *, filename: str, **kwargs: Any):
        del filename, kwargs
        raise ArtifactStoreUnavailableError(f"failed to store {content.decode()}")


class _BlockingArtifactStore(LocalArtifactStore):
    def __init__(self, root, *, store_id: str) -> None:
        super().__init__(root, store_id=store_id)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.writes = 0

    async def put_bytes(self, content: bytes, *, filename: str, **kwargs: Any):
        self.writes += 1
        self.started.set()
        await self.release.wait()
        return await super().put_bytes(content, filename=filename, **kwargs)


class _LateCompletingArtifactStore(_BlockingArtifactStore):
    def __init__(self, root, *, store_id: str) -> None:
        super().__init__(root, store_id=store_id)
        self.cancellation_observed = asyncio.Event()

    async def put_bytes(self, content: bytes, *, filename: str, **kwargs: Any):
        artifact_id = kwargs.get("artifact_id")
        assert type(artifact_id) is str
        registration = register_artifact_write_operation(
            artifact_id=artifact_id,
            store_id=self.id,
        )
        registration.set_phase(ArtifactWriteSettlementPhase.CONTENT)
        self.writes += 1
        self.started.set()
        try:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancellation_observed.set()
                task = asyncio.current_task()
                if task is not None:
                    while task.cancelling():
                        task.uncancel()
                await self.release.wait()
            artifact = await LocalArtifactStore.put_bytes(
                self,
                content,
                filename=filename,
                **kwargs,
            )
            registration.record(
                status=ArtifactWriteSettlementStatus.COMMITTED,
                phase=ArtifactWriteSettlementPhase.SETTLED,
            )
            return artifact
        except BaseException:
            registration.close()
            raise


class _SelfCancellingProjectionPolicy(ToolResultProjectionPolicy):
    @property
    def identity(self) -> str:
        return "tests.self_cancelling_projection.v1"

    async def project(self, request: ToolResultProjectionRequest):
        del request
        raise asyncio.CancelledError("projection policy cancelled itself")


class _RegisteredFailingArtifactStore(LocalArtifactStore):
    def __init__(
        self,
        root,
        *,
        store_id: str,
        backend_locator: str | None = None,
        backend_version: str | None = None,
    ) -> None:
        super().__init__(root, store_id=store_id)
        self.backend_locator = backend_locator
        self.backend_version = backend_version

    async def put_bytes(self, content: bytes, *, filename: str, **kwargs: Any):
        del content, filename
        artifact_id = kwargs.get("artifact_id")
        assert type(artifact_id) is str
        registration = register_artifact_write_operation(
            artifact_id=artifact_id,
            store_id=self.id,
        )
        registration.set_phase(ArtifactWriteSettlementPhase.COMMIT)
        error = ArtifactStoreUnavailableError("third-party settlement canary")
        registration.record(
            status=ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED,
            phase=ArtifactWriteSettlementPhase.RECONCILIATION,
            error=error,
            failure_codes=(ArtifactWriteSettlementFailureCode.COMMIT_FAILED,),
            backend_locator=self.backend_locator,
            backend_version=self.backend_version,
        )
        raise error


class _ObservedArtifactFailureProjectionPolicy(ToolResultProjectionPolicy):
    def __init__(self, fallback: str) -> None:
        self.fallback = fallback

    @property
    def identity(self) -> str:
        return "tests.observed_artifact_failure_projection.v1"

    async def project(self, request: ToolResultProjectionRequest):
        assert request.artifact_store is not None
        try:
            await request.artifact_store.put_bytes(
                request.result.content.encode(),
                artifact_id=f"art_{'d' * 32}",
                filename="projection.txt",
                session_id=request.session_id,
            )
        except ArtifactStoreUnavailableError:
            if self.fallback == "missing":
                return None
            if self.fallback == "invalid":
                return object()
            raise
        raise AssertionError("failing store unexpectedly returned")


class _HistoricalSettlementProjectionPolicy(ToolResultProjectionPolicy):
    def __init__(self, behavior: str) -> None:
        self.behavior = behavior
        now = datetime.now(UTC)
        self.historical_settlement = ArtifactWriteSettlementEvidence(
            operation_id=f"artifact_write_{'1' * 32}",
            artifact_id=f"art_{'2' * 32}",
            store_identity_sha256=artifact_store_identity_sha256("historical-store"),
            status=ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED,
            phase=ArtifactWriteSettlementPhase.RECONCILIATION,
            observation=ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
            started_at=now,
            observed_at=now,
            elapsed_ms=0,
            failure_codes=(ArtifactWriteSettlementFailureCode.COMMIT_FAILED,),
        )
        self.historical_error = ArtifactStoreUnavailableError("historical write failed")
        record_artifact_write_settlement(
            self.historical_settlement,
            error=self.historical_error,
        )

    @property
    def identity(self) -> str:
        return "tests.historical_settlement_projection.v1"

    async def project(self, request: ToolResultProjectionRequest):
        if self.behavior == "raise":
            raise RuntimeError("current projection failed") from self.historical_error
        projected_result = ToolResult(
            content="[projection rejected]",
            structured=request.result.structured,
            artifacts=request.result.artifacts,
            is_error=request.result.is_error,
        )
        return ToolResultProjection(
            result=projected_result,
            record=ToolResultProjectionRecord(
                status=ToolResultProjectionStatus.FAILED,
                policy_id=self.identity,
                original_bytes=len(request.result.content.encode("utf-8")),
                projected_bytes=len(projected_result.content.encode("utf-8")),
                original_token_estimate=0,
                projected_token_estimate=0,
                token_estimation_method="tests_exact_v1",
                failure_type="historical_failure",
                artifact_write_settlement=self.historical_settlement,
            ),
        )


class _RejectFirstToolRoundPublicationStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.tool_round_publications = 0

    async def publish_runtime_publication(
        self,
        session_id: str,
        *,
        request: RuntimePublicationRequest,
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> RuntimePublicationResult:
        if request.kind == "tool-round":
            self.tool_round_publications += 1
            if self.tool_round_publications == 1:
                raise RuntimeError("tool-round publication rejected before commit")
        return await super().publish_runtime_publication(
            session_id,
            request=request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )


class _ReadbackProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.externalized_artifact_id: str | None = None
        self.readback_arguments: dict[str, Any] | None = None

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(
                id="call_result",
                name="result_tool",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if len(self.requests) == 2:
            tool_result = next(
                message for message in request.messages if message.role == "tool"
            ).content[0]
            reference = next(
                artifact
                for artifact in tool_result.artifacts
                if artifact.get("type") == "cayu.tool_result_artifact.v1"
            )
            self.externalized_artifact_id = reference["artifact_id"]
            read_file_arguments = json.loads(
                tool_result.content.split("Use read_file with ", maxsplit=1)[1].split(
                    " to inspect", maxsplit=1
                )[0]
            )
            self.readback_arguments = read_file_arguments
            yield ModelStreamEvent.tool_call(
                id="call_readback",
                name="read_file",
                arguments=read_file_arguments,
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _LargeMcpSession(McpSession):
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0
        self.definition = McpToolDefinition(
            name="large_result",
            description="Return a large MCP result.",
            input_schema={"type": "object"},
        )

    @property
    def initialize_result(self) -> McpInitializeResult:
        return McpInitializeResult(protocol_version="2025-06-18")

    async def list_tools(self) -> tuple[McpToolDefinition, ...]:
        return (self.definition,)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        del name, arguments
        self.calls += 1
        return McpToolResult(
            content=[{"type": "text", "text": self.content}],
            structured_content={"receipt_id": "mcp-receipt"},
        )

    async def list_resources(self):
        return ()

    async def read_resource(self, uri: str) -> McpResourceResult:
        del uri
        raise NotImplementedError

    async def close(self) -> None:
        return None


async def _collect(events: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in events]


def _run_tool_result(
    *,
    tmp_path,
    content: str,
    policy: ToolResultProjectionPolicy | None,
    store: LocalArtifactStore | None = None,
    secret_redactor: SecretRedactor | None = None,
    structured: dict[str, Any] | None = None,
) -> tuple[
    CayuApp,
    LocalArtifactStore,
    _FakeProvider,
    _ResultTool,
    list[Event],
]:
    artifact_store = store or LocalArtifactStore(
        tmp_path / "runtime-artifacts",
        store_id="runtime-artifacts",
    )
    provider = _FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_result",
                    name="result_tool",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    tool = _ResultTool(
        ToolResult(
            content=content,
            structured=structured,
            artifacts=[{"type": "existing", "id": "existing"}],
        )
    )
    app = CayuApp(
        enable_logging=False,
        secret_redactor=secret_redactor,
        tool_result_projection_policy=policy,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            artifact_store=artifact_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )
    events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    session_id="sess_runtime_projection",
                    agent_name="assistant",
                    messages=[Message.text("user", "run the tool")],
                )
            )
        )
    )
    return app, artifact_store, provider, tool, events


def _request(
    *,
    result: ToolResult,
    artifact_store: LocalArtifactStore | None,
) -> ToolResultProjectionRequest:
    return ToolResultProjectionRequest(
        result=result,
        session_id="sess_projection",
        agent_name="assistant",
        environment_name="local",
        tool_call_id="call_projection",
        artifact_store=artifact_store,
    )


def _attest_runtime_projection(event: Event) -> Event:
    return event_with_runtime_nested_payload_authority(
        event,
        ("tool_result_projection", "policy_id"),
    )


def test_artifact_externalizing_policy_keeps_exact_byte_threshold_unchanged(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    policy = ArtifactExternalizingToolResultPolicy(
        max_inline_bytes=32,
        max_inline_token_estimate=None,
        preview_bytes=3,
    )
    result = ToolResult(
        content="é" * 16,
        structured={"rows": 2},
        artifacts=[{"type": "existing", "artifact_id": "existing"}],
        is_error=True,
    )

    projection = asyncio.run(policy.project(_request(result=result, artifact_store=store)))

    assert projection.result == result
    assert projection.record.model_dump(exclude_none=True) == {
        "schema_version": 1,
        "status": "unchanged",
        "policy_id": "cayu.artifact_externalizing_tool_result.v1",
        "original_bytes": 32,
        "projected_bytes": 32,
        "original_token_estimate": 4,
        "projected_token_estimate": 4,
        "token_estimation_method": "unicode_codepoints_divided_by_4_ceiling_v1",
    }
    assert asyncio.run(store.list(session_id="sess_projection")).artifacts == ()


def test_artifact_externalizing_policy_externalizes_unicode_and_reuses_identity(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    policy = ArtifactExternalizingToolResultPolicy(
        max_inline_bytes=32,
        max_inline_token_estimate=None,
        preview_bytes=3,
    )
    result = ToolResult(
        content=("é" * 16) + "a",
        structured={"rows": 3},
        artifacts=[
            {"type": "existing", "artifact_id": "existing"},
            {
                "type": "cayu.file_attachment.v1",
                "artifact_id": f"art_{'f' * 32}",
                "kind": "image",
                "filename": "existing.png",
                "content_type": "image/png",
                "size_bytes": 8,
            },
        ],
        is_error=True,
    )
    request = _request(result=result, artifact_store=store)

    first = asyncio.run(policy.project(request))
    second = asyncio.run(policy.project(request))

    assert first == second
    assert first.record.status == "externalized"
    assert first.record.original_bytes == 33
    assert first.record.artifact_id is not None
    assert first.record.artifact_sha256 is not None
    assert first.result.is_error is True
    assert first.result.structured == {"rows": 3}
    assert first.result.artifacts[:-1] == result.artifacts
    assert first.result.artifacts[1]["type"] == "cayu.file_attachment.v1"
    reference = first.result.artifacts[-1]
    assert reference == {
        "type": "cayu.tool_result_artifact.v1",
        "artifact_id": first.record.artifact_id,
        "store_id": "artifacts",
        "filename": f"tool-result-{first.record.artifact_id}.txt",
        "content_type": "text/plain; charset=utf-8",
        "size_bytes": 33,
        "sha256": first.record.artifact_sha256,
        "scope": "session",
        "session_id_sha256": hashlib.sha256(b"sess_projection").hexdigest(),
        "projection_authority": "cayu.tool_result_projection.v1",
        "readback_max_bytes": 14,
    }
    assert "\né\n" in first.result.content
    assert ("é" * 16) + "a" not in first.result.content
    assert first.record.projected_bytes == len(first.result.content.encode("utf-8"))
    assert (
        first.record.projected_token_estimate
        == (len(first.result.content) + policy.chars_per_token - 1) // policy.chars_per_token
    )

    listed = asyncio.run(store.list(session_id="sess_projection"))
    assert len(listed.artifacts) == 1
    stored = asyncio.run(store.read_bytes(first.record.artifact_id))
    assert stored.content == (("é" * 16) + "a").encode()
    assert stored.metadata.metadata == {
        "type": "cayu.tool_result_artifact.v1",
        "logical_identity_sha256": first.record.logical_identity_sha256,
        "sha256": first.record.artifact_sha256,
        "policy_id": policy.identity,
        "tool_call_id_sha256": first.record.tool_call_id_sha256,
    }


def test_artifact_externalizing_policy_uses_declared_token_estimate_threshold(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    policy = ArtifactExternalizingToolResultPolicy(
        max_inline_bytes=None,
        max_inline_token_estimate=5,
        preview_bytes=0,
    )

    exact = asyncio.run(
        policy.project(
            _request(
                result=ToolResult(content="a" * 20),
                artifact_store=store,
            )
        )
    )
    over = asyncio.run(
        policy.project(
            _request(
                result=ToolResult(content="a" * 21),
                artifact_store=store,
            )
        )
    )
    empty = asyncio.run(
        policy.project(
            _request(
                result=ToolResult(content=""),
                artifact_store=store,
            )
        )
    )

    assert exact.record.status == "unchanged"
    assert exact.record.original_token_estimate == 5
    assert over.record.status == "externalized"
    assert over.record.original_token_estimate == 6
    assert empty.record.status == "unchanged"
    custom_estimator = ArtifactExternalizingToolResultPolicy(
        max_inline_bytes=None,
        max_inline_token_estimate=10,
        preview_bytes=0,
        chars_per_token=2,
    )
    custom = asyncio.run(
        custom_estimator.project(
            _request(
                result=ToolResult(content="abc"),
                artifact_store=store,
            )
        )
    )
    assert custom.record.original_token_estimate == 2
    assert custom.record.token_estimation_method == ("unicode_codepoints_divided_by_2_ceiling_v1")
    with pytest.raises(AttributeError):
        custom_estimator.chars_per_token = 4
    with pytest.raises(ValueError, match="bounded read_file result"):
        ArtifactExternalizingToolResultPolicy(
            max_inline_bytes=18,
            max_inline_token_estimate=None,
            preview_bytes=0,
        )
    with pytest.raises(ValueError, match="bounded read_file result"):
        ArtifactExternalizingToolResultPolicy(
            max_inline_bytes=None,
            max_inline_token_estimate=4,
            preview_bytes=0,
        )
    with pytest.raises(ValueError, match="preview_bytes"):
        ArtifactExternalizingToolResultPolicy(
            max_inline_bytes=1,
            max_inline_token_estimate=None,
            preview_bytes=MAX_TOOL_RESULT_PREVIEW_BYTES + 1,
        )


def test_artifact_externalizing_policy_fails_bounded_without_artifact_store() -> None:
    policy = ArtifactExternalizingToolResultPolicy(
        max_inline_bytes=64,
        max_inline_token_estimate=None,
        preview_bytes=0,
    )
    result = ToolResult(
        content="x" * 1_000_000,
        structured={"receipt_id": "receipt-1"},
        artifacts=[{"type": "effect_receipt", "id": "receipt-1"}],
    )

    projection = asyncio.run(policy.project(_request(result=result, artifact_store=None)))

    assert projection.record.status == "failed"
    assert projection.record.failure_type == "artifact_store_missing"
    assert projection.record.original_bytes == 1_000_000
    assert projection.record.projected_bytes == len(projection.result.content.encode())
    assert projection.record.projected_bytes < 1024
    assert "x" * 100 not in projection.result.content
    assert projection.result.structured == result.structured
    assert projection.result.artifacts == result.artifacts
    assert projection.result.is_error is result.is_error


def test_artifact_externalizing_policy_bounds_the_store_identity_before_persistence(
    tmp_path,
) -> None:
    store = LocalArtifactStore(
        tmp_path / "long-store-id",
        store_id="store-" + ("s" * 1_000),
    )
    policy = ArtifactExternalizingToolResultPolicy(
        max_inline_bytes=64,
        max_inline_token_estimate=None,
    )

    projection = asyncio.run(
        policy.project(
            _request(
                result=ToolResult(content="oversized" * 10),
                artifact_store=store,
            )
        )
    )

    assert projection.record.status == "failed"
    assert projection.record.failure_type == "ValueError"
    assert projection.record.projected_bytes < 1024
    assert asyncio.run(store.list(session_id="sess_projection")).artifacts == ()


def test_artifact_reference_is_bounded_for_an_extreme_session_identity(tmp_path) -> None:
    store = LocalArtifactStore(
        tmp_path / "long-session-id",
        store_id="bounded-store",
    )
    policy = ArtifactExternalizingToolResultPolicy(
        max_inline_bytes=64,
        max_inline_token_estimate=None,
        preview_bytes=0,
    )
    session_id = "session-" + ("s" * 100_000)

    projection = asyncio.run(
        policy.project(
            ToolResultProjectionRequest(
                result=ToolResult(content="oversized" * 10),
                session_id=session_id,
                agent_name="assistant",
                environment_name="local",
                tool_call_id="call_projection",
                artifact_store=store,
            )
        )
    )

    reference = projection.result.artifacts[-1]
    serialized_reference = projection.result.model_dump(mode="json")["artifacts"][-1]
    assert projection.record.status == "externalized"
    assert "session_id" not in reference
    assert reference["session_id_sha256"] == hashlib.sha256(session_id.encode()).hexdigest()
    assert serialized_reference == reference
    assert (
        len(
            json.dumps(
                serialized_reference,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        <= MAX_TOOL_RESULT_ARTIFACT_REFERENCE_BYTES
    )
    assert len(projection.result.model_dump_json().encode()) < 2_048


def test_cayu_app_keeps_large_tool_results_unchanged_when_policy_is_absent(tmp_path) -> None:
    original = "large-default-off-" + ("x" * 10_000)

    app, store, provider, tool, events = _run_tool_result(
        tmp_path=tmp_path,
        content=original,
        policy=None,
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == 1
    transcript = asyncio.run(app.session_store.load_transcript("sess_runtime_projection"))
    tool_result = next(message for message in transcript if message.role == "tool").content[0]
    provider_result = next(
        message for message in provider.requests[1].messages if message.role == "tool"
    ).content[0]
    assert tool_result.content == provider_result.content == original
    assert asyncio.run(store.list(session_id="sess_runtime_projection")).artifacts == ()


def test_cayu_app_keeps_below_threshold_result_durable_and_model_visible(tmp_path) -> None:
    original = "small"
    policy = ArtifactExternalizingToolResultPolicy(
        max_inline_bytes=64,
        max_inline_token_estimate=None,
        preview_bytes=0,
    )

    app, store, provider, _, events = _run_tool_result(
        tmp_path=tmp_path,
        content=original,
        policy=policy,
    )

    terminal = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert terminal.payload["tool_result_projection"]["status"] == "unchanged"
    transcript = asyncio.run(app.session_store.load_transcript("sess_runtime_projection"))
    transcript_result = next(message for message in transcript if message.role == "tool").content[0]
    provider_result = next(
        message for message in provider.requests[1].messages if message.role == "tool"
    ).content[0]
    assert transcript_result.content == provider_result.content == original
    assert asyncio.run(store.list(session_id="sess_runtime_projection")).artifacts == ()


def test_cayu_app_externalizes_after_redaction_before_terminal_publication(tmp_path) -> None:
    secret = "projection-secret-canary"
    original = f"public:{secret}:" + ("z" * 10_000)
    policy = ArtifactExternalizingToolResultPolicy(
        max_inline_bytes=64,
        max_inline_token_estimate=None,
        preview_bytes=32,
    )

    app, store, provider, tool, events = _run_tool_result(
        tmp_path=tmp_path,
        content=original,
        policy=policy,
        secret_redactor=SecretRedactor(secret),
        structured={"receipt_id": "receipt-1"},
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == 1
    terminal = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    projection = terminal.payload["tool_result_projection"]
    assert projection["status"] == "externalized"
    assert projection["original_bytes"] == len(original.replace(secret, REDACTED_SECRET).encode())
    assert projection["policy_id"] == policy.identity
    assert projection["artifact_id"].startswith("art_")
    assert projection["artifact_sha256"]
    assert projection["token_estimation_method"] == ("unicode_codepoints_divided_by_4_ceiling_v1")

    transcript = asyncio.run(app.session_store.load_transcript("sess_runtime_projection"))
    transcript_result = next(message for message in transcript if message.role == "tool").content[0]
    provider_result = next(
        message for message in provider.requests[1].messages if message.role == "tool"
    ).content[0]
    assert transcript_result == provider_result
    assert transcript_result.content == terminal.payload["result"]["content"]
    assert transcript_result.structured == {"receipt_id": "receipt-1"}
    assert transcript_result.artifacts[0] == {"type": "existing", "id": "existing"}
    assert transcript_result.artifacts[-1]["artifact_id"] == projection["artifact_id"]
    assert original not in transcript_result.content

    stored = asyncio.run(store.read_bytes(projection["artifact_id"]))
    assert stored.content.decode() == original.replace(secret, REDACTED_SECRET)
    serialized = json.dumps(
        {
            "events": [event.model_dump(mode="json") for event in events],
            "transcript": [message.model_dump(mode="json") for message in transcript],
            "provider_request": provider.requests[1].model_dump(mode="json"),
        }
    )
    assert secret not in serialized
    assert original not in serialized


def test_cayu_app_preserves_runtime_owned_projection_identity_after_redaction(tmp_path) -> None:
    original = "runtime-owned-identity-" + ("b" * 10_000)
    policy = ArtifactExternalizingToolResultPolicy(
        max_inline_bytes=64,
        max_inline_token_estimate=None,
        preview_bytes=0,
    )

    app, store, provider, _, events = _run_tool_result(
        tmp_path=tmp_path,
        content=original,
        policy=policy,
        secret_redactor=SecretRedactor(
            [
                "art_",
                "cayu",
                "externalized",
                "session",
            ]
        ),
    )

    terminal = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    projection = terminal.payload["tool_result_projection"]
    reference = terminal.payload["result"]["artifacts"][-1]
    stored = asyncio.run(store.list(session_id="sess_runtime_projection")).artifacts

    assert len(stored) == 1
    assert projection["status"] == "externalized"
    assert projection["policy_id"] == policy.identity
    assert projection["artifact_id"] == stored[0].id
    assert reference["type"] == "cayu.tool_result_artifact.v1"
    assert reference["artifact_id"] == stored[0].id
    assert reference["scope"] == "session"
    assert reference["sha256"] == projection["artifact_sha256"]

    transcript = asyncio.run(app.session_store.load_transcript("sess_runtime_projection"))
    transcript_result = next(message for message in transcript if message.role == "tool").content[0]
    provider_result = next(
        message for message in provider.requests[1].messages if message.role == "tool"
    ).content[0]
    assert transcript_result.artifacts[-1] == reference
    assert provider_result.artifacts[-1] == reference
    assert asyncio.run(store.read_bytes(reference["artifact_id"])).content.decode() == original


def test_malformed_projection_reference_stays_on_event_redaction_path() -> None:
    from cayu.runtime._event_writer import prepare_runtime_event

    secret = "projection-event-secret"
    artifact_id = f"art_{'a' * 32}"
    artifact_sha256 = "b" * 64
    content = f"projected content with {secret}"
    reference = {
        "type": "cayu.tool_result_artifact.v1",
        "artifact_id": artifact_id,
        "store_id": "artifacts",
        "filename": f"tool-result-{artifact_id}.txt",
        "content_type": "text/plain; charset=utf-8",
        "size_bytes": 1_000,
        "sha256": artifact_sha256,
        "scope": "session",
        "session_id_sha256": "c" * 64,
        "projection_authority": "cayu.tool_result_projection.v1",
        "readback_max_bytes": 64,
        "untrusted_detail": secret,
    }
    event = Event(
        type=EventType.TOOL_CALL_COMPLETED,
        session_id="sess_projection_event_boundary",
        tool_name="result_tool",
        payload={
            "tool_call_id": "call_projection_event_boundary",
            "tool_name": "result_tool",
            "result": ToolResult(content=content, artifacts=[reference]).model_dump(mode="json"),
            "tool_result_projection": {
                "schema_version": 1,
                "status": "externalized",
                "policy_id": "cayu.artifact_externalizing_tool_result.v1",
                "original_bytes": 1_000,
                "projected_bytes": len(content.encode("utf-8")),
                "original_token_estimate": 250,
                "projected_token_estimate": 8,
                "token_estimation_method": "unicode_codepoints_divided_by_4_ceiling_v1",
                "artifact_id": artifact_id,
                "artifact_sha256": artifact_sha256,
                "logical_identity_sha256": "d" * 64,
                "tool_call_id_sha256": "e" * 64,
            },
        },
    )

    initially_prepared = prepare_runtime_event(
        event,
        redactor=SecretRedactor([]),
    )
    prepared = prepare_runtime_event(
        initially_prepared,
        redactor=SecretRedactor(secret),
    )

    serialized = json.dumps(prepared.model_dump(mode="json"))
    assert secret not in serialized
    assert prepared.payload["result"]["content"] == (f"projected content with {REDACTED_SECRET}")
    assert prepared.payload["result"]["artifacts"][0]["untrusted_detail"] == REDACTED_SECRET


@pytest.mark.parametrize(
    "event_type",
    [EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_COMPLETED],
)
def test_unattested_projection_lookalike_never_bypasses_event_redaction(
    event_type: EventType,
) -> None:
    from cayu.runtime._event_projection import project_runtime_event
    from cayu.runtime._event_writer import prepare_runtime_event

    secret = "fabricated-projection-store-secret"
    artifact_id = f"art_{'a' * 32}"
    content = "fabricated projection preview"
    reference = {
        "type": "cayu.tool_result_artifact.v1",
        "artifact_id": artifact_id,
        "store_id": secret,
        "filename": f"tool-result-{artifact_id}.txt",
        "content_type": "text/plain; charset=utf-8",
        "size_bytes": 1_000,
        "sha256": "b" * 64,
        "scope": "session",
        "session_id_sha256": "c" * 64,
        "projection_authority": "cayu.tool_result_projection.v1",
        "readback_max_bytes": 64,
    }
    event = Event(
        type=event_type,
        session_id="sess_projection_lookalike",
        tool_name="result_tool",
        payload={
            "tool_call_id": "call_projection_lookalike",
            "tool_name": "result_tool",
            "result": ToolResult(content=content, artifacts=[reference]).model_dump(mode="json"),
            "tool_result_projection": {
                "schema_version": 1,
                "status": "externalized",
                "policy_id": "cayu.artifact_externalizing_tool_result.v1",
                "original_bytes": 1_000,
                "projected_bytes": len(content.encode("utf-8")),
                "original_token_estimate": 250,
                "projected_token_estimate": (len(content) + 3) // 4,
                "token_estimation_method": "unicode_codepoints_divided_by_4_ceiling_v1",
                "artifact_id": artifact_id,
                "artifact_sha256": "b" * 64,
                "logical_identity_sha256": "d" * 64,
                "tool_call_id_sha256": "e" * 64,
            },
        },
    )

    prepared = prepare_runtime_event(event, redactor=SecretRedactor(secret))
    public = project_runtime_event(
        event,
        sequence=1,
        redactor=SecretRedactor(secret),
    )

    for boundary_event in (prepared, public):
        serialized = json.dumps(boundary_event.model_dump(mode="json"))
        assert secret not in serialized
        assert boundary_event.payload["result"]["artifacts"][0]["store_id"] == REDACTED_SECRET


def test_valid_projection_re_redacts_content_under_rotated_event_registry() -> None:
    from cayu.runtime._event_writer import prepare_runtime_event

    secret = "rotated-event-secret"
    artifact_id = f"art_{'a' * 32}"
    artifact_sha256 = "b" * 64
    content = f"projection preview with {secret}"
    reference = {
        "type": "cayu.tool_result_artifact.v1",
        "artifact_id": artifact_id,
        "store_id": "artifacts",
        "filename": f"tool-result-{artifact_id}.txt",
        "content_type": "text/plain; charset=utf-8",
        "size_bytes": 1_000,
        "sha256": artifact_sha256,
        "scope": "session",
        "session_id_sha256": "c" * 64,
        "projection_authority": "cayu.tool_result_projection.v1",
        "readback_max_bytes": 64,
    }
    record = {
        "schema_version": 1,
        "status": "externalized",
        "policy_id": "cayu.artifact_externalizing_tool_result.v1",
        "original_bytes": 1_000,
        "projected_bytes": len(content.encode("utf-8")),
        "original_token_estimate": 250,
        "projected_token_estimate": 8,
        "token_estimation_method": "unicode_codepoints_divided_by_4_ceiling_v1",
        "artifact_id": artifact_id,
        "artifact_sha256": artifact_sha256,
        "logical_identity_sha256": "d" * 64,
        "tool_call_id_sha256": "e" * 64,
    }
    event = Event(
        type=EventType.TOOL_CALL_COMPLETED,
        session_id="sess_projection_rotated_event",
        tool_name="result_tool",
        payload={
            "tool_call_id": "call_projection_rotated_event",
            "tool_name": "result_tool",
            "result": ToolResult(content=content, artifacts=[reference]).model_dump(mode="json"),
            "tool_result_projection": record,
        },
    )
    event = _attest_runtime_projection(event)

    initially_prepared = prepare_runtime_event(
        event,
        redactor=SecretRedactor([]),
    )
    prepared = prepare_runtime_event(
        initially_prepared,
        redactor=SecretRedactor(secret),
    )

    serialized = json.dumps(prepared.model_dump(mode="json"))
    assert secret not in serialized
    assert prepared.payload["result"]["artifacts"][0] == reference
    assert artifact_id in prepared.payload["result"]["content"]
    assert prepared.payload["tool_result_projection"]["artifact_id"] == artifact_id
    assert prepared.payload["tool_result_projection"]["projected_bytes"] == len(
        prepared.payload["result"]["content"].encode("utf-8")
    )
    assert (
        prepared.payload["tool_result_projection"]["projected_token_estimate"]
        == (len(prepared.payload["result"]["content"]) + 3) // 4
    )

    reloaded = Event.model_validate_json(prepared.model_dump_json())
    alias_codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={"test": SecretStr("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")},
        )
    )
    store = InMemorySessionStore(public_authority_alias_codec=alias_codec)
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        secret_redactor=SecretRedactor("art_"),
    )
    caller_supplied = app.project_event_record_for_exposure(EventRecord(sequence=1, event=reloaded))
    assert artifact_id not in caller_supplied.model_dump_json()

    async def expose_reloaded_event() -> tuple[Event, Event]:
        await store.create(
            RunRequest(
                session_id=reloaded.session_id,
                agent_name="assistant",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await store.append_event(reloaded.session_id, reloaded)
        persisted = (await store.query_events())[0]
        exposed_record = app._project_persisted_event_record_for_exposure(persisted)
        exposed_emitted = await app._project_emitted_event_for_public_api(reloaded)
        return exposed_record.event, exposed_emitted

    exposed_record, exposed_emitted = asyncio.run(expose_reloaded_event())
    for exposed in (exposed_record, exposed_emitted):
        assert artifact_id in exposed.payload["result"]["content"]
        assert exposed.payload["result"]["artifacts"][0] == reference
        assert exposed.payload["tool_result_projection"]["artifact_id"] == artifact_id


def test_blocked_projection_restores_reference_after_denial_postprocessing() -> None:
    from cayu.runtime._event_writer import prepare_runtime_event
    from cayu.runtime.tool_result_projection import (
        redact_tool_result_projection_content,
    )

    artifact_id = f"art_{'a' * 32}"
    artifact_sha256 = "b" * 64
    reference = {
        "type": "cayu.tool_result_artifact.v1",
        "artifact_id": artifact_id,
        "store_id": "artifacts",
        "filename": f"tool-result-{artifact_id}.txt",
        "content_type": "text/plain; charset=utf-8",
        "size_bytes": 1_000,
        "sha256": artifact_sha256,
        "scope": "session",
        "session_id_sha256": "c" * 64,
        "projection_authority": "cayu.tool_result_projection.v1",
        "readback_max_bytes": 64,
    }
    content = redact_tool_result_projection_content(
        "blocked projection",
        artifact_id=artifact_id,
        readback_max_bytes=64,
        redact_text=SecretRedactor([]).redact_text,
    )
    event = Event(
        type=EventType.TOOL_CALL_BLOCKED,
        session_id="sess_blocked_projection",
        tool_name="result_tool",
        payload={
            "denied_by": "tool_policy",
            "decision": "deny",
            "reason": "blocked projection",
            "tool_call_id": "call_blocked_projection",
            "tool_name": "result_tool",
            "result": ToolResult(
                content=content,
                artifacts=[reference],
                is_error=True,
            ).model_dump(mode="json"),
            "tool_result_projection": {
                "schema_version": 1,
                "status": "externalized",
                "policy_id": "cayu.artifact_externalizing_tool_result.v1",
                "original_bytes": 1_000,
                "projected_bytes": len(content.encode("utf-8")),
                "original_token_estimate": 250,
                "projected_token_estimate": (len(content) + 3) // 4,
                "token_estimation_method": "unicode_codepoints_divided_by_4_ceiling_v1",
                "artifact_id": artifact_id,
                "artifact_sha256": artifact_sha256,
                "logical_identity_sha256": "d" * 64,
                "tool_call_id_sha256": "e" * 64,
            },
        },
    )
    event = _attest_runtime_projection(event)

    prepared = prepare_runtime_event(
        event,
        redactor=SecretRedactor("art_"),
    )
    assert artifact_id in prepared.payload["result"]["content"]
    assert prepared.payload["result"]["artifacts"][0] == reference
    replayed = prepare_runtime_event(
        prepared,
        redactor=SecretRedactor("art_"),
    )

    assert artifact_id in replayed.payload["result"]["content"]
    assert replayed.payload["result"]["artifacts"][0] == reference
    assert replayed.payload["tool_result_projection"]["projected_bytes"] == len(
        replayed.payload["result"]["content"].encode("utf-8")
    )


@pytest.mark.parametrize("status", ["unchanged", "failed"])
def test_non_externalized_projection_resynchronizes_rotated_event_evidence(
    status: str,
) -> None:
    from cayu.runtime._event_writer import prepare_runtime_event

    secret = "rotated-inline-secret"
    content = f"inline result with {secret}"
    record = {
        "schema_version": 1,
        "status": status,
        "policy_id": "custom.projection.v1",
        "original_bytes": len(content.encode("utf-8")),
        "projected_bytes": len(content.encode("utf-8")),
        "original_token_estimate": (len(content) + 3) // 4,
        "projected_token_estimate": (len(content) + 3) // 4,
        "token_estimation_method": "unicode_codepoints_divided_by_4_ceiling_v1",
    }
    if status == "failed":
        record["failure_type"] = "custom_projection_failure"
    event = Event(
        type=EventType.TOOL_CALL_COMPLETED,
        session_id=f"sess_projection_rotated_{status}",
        tool_name="result_tool",
        payload={
            "tool_call_id": f"call_projection_rotated_{status}",
            "tool_name": "result_tool",
            "result": ToolResult(content=content).model_dump(mode="json"),
            "tool_result_projection": record,
        },
    )
    event = _attest_runtime_projection(event)

    initially_prepared = prepare_runtime_event(
        event,
        redactor=SecretRedactor([]),
    )
    prepared = prepare_runtime_event(
        initially_prepared,
        redactor=SecretRedactor(secret),
    )

    projected_content = prepared.payload["result"]["content"]
    projected_record = prepared.payload["tool_result_projection"]
    assert secret not in projected_content
    assert projected_record["projected_bytes"] == len(projected_content.encode("utf-8"))
    assert projected_record["projected_token_estimate"] == (len(projected_content) + 3) // 4


def test_application_artifact_cannot_claim_runtime_projection_ownership() -> None:
    from cayu.core import MessageRole, ToolResultPart
    from cayu.runtime._message_redaction import (
        redact_runtime_message_for_boundary,
        redact_untrusted_message_for_boundary,
    )

    secret = "application-artifact-secret"
    artifact_id = f"art_{'a' * 32}"
    lookalike = Message(
        role=MessageRole.TOOL,
        content=(
            ToolResultPart(
                tool_call_id="call_application_artifact",
                tool_name="application_tool",
                content=f"application content with {secret}",
                artifacts=[
                    {
                        "type": "cayu.tool_result_artifact.v1",
                        "artifact_id": artifact_id,
                        "store_id": secret,
                        "filename": f"tool-result-{artifact_id}.txt",
                        "content_type": "text/plain; charset=utf-8",
                        "size_bytes": 1_000,
                        "sha256": "b" * 64,
                        "scope": "session",
                        "session_id_sha256": "c" * 64,
                        "projection_authority": "cayu.tool_result_projection.v1",
                        "readback_max_bytes": 64,
                    }
                ],
            ),
        ),
    )

    untrusted = redact_untrusted_message_for_boundary(
        lookalike,
        redactor=SecretRedactor([]),
        field_name="message",
    )
    redacted = redact_runtime_message_for_boundary(
        untrusted,
        redactor=SecretRedactor(secret),
        field_name="message",
    )

    serialized = json.dumps(redacted.model_dump(mode="json"))
    assert secret not in serialized
    assert redacted.content[0].content == f"application content with {REDACTED_SECRET}"
    assert redacted.content[0].artifacts[0]["store_id"] == REDACTED_SECRET


def test_application_result_cannot_claim_runtime_execution_control_ownership() -> None:
    from cayu.core import MessageRole, ToolResultPart
    from cayu.runtime._message_redaction import (
        redact_runtime_message_for_boundary,
        redact_untrusted_message_for_boundary,
    )

    secret = "application_control_secret"
    lookalike = Message(
        role=MessageRole.TOOL,
        content=(
            ToolResultPart(
                tool_call_id="call_application_control",
                tool_name="application_tool",
                content="application result",
                structured={
                    "isolated_tool_failure_code": secret,
                    "tool_execution_boundary": "posix_process",
                    "tool_timeout_strength": "hard_process_deadline",
                    "detail": "safe",
                },
                is_error=True,
            ),
        ),
    )

    untrusted = redact_untrusted_message_for_boundary(
        lookalike,
        redactor=SecretRedactor([]),
        field_name="message",
    )
    redacted = redact_runtime_message_for_boundary(
        untrusted,
        redactor=SecretRedactor(secret),
        field_name="message",
    )

    assert secret not in json.dumps(redacted.model_dump(mode="json"))
    assert redacted.content[0].structured == {"detail": "safe"}


@pytest.mark.parametrize("result_source", ["tool", "before_hook", "after_hook"])
def test_public_tool_result_cannot_claim_execution_controls_after_registry_rotation(
    result_source: str,
) -> None:
    from cayu import (
        AfterToolCallDecision,
        BeforeToolCallDecision,
        ExecutionProfileAdoptionIntent,
        ExecutionProfileAuthorityDecision,
        ExecutionProfilePolicy,
        ExecutionProfilePolicyAction,
        ExecutionProfilePolicyRequest,
        ExecutionProfilePolicyResult,
        ResolutionActor,
        ResolutionActorSource,
        RuntimeHook,
    )

    class AdoptCurrentProfile(ExecutionProfilePolicy):
        @property
        def identity(self) -> str:
            return "tests:adopt-tool-result-redaction-profile:v1"

        async def decide(
            self,
            request: ExecutionProfilePolicyRequest,
        ) -> ExecutionProfilePolicyResult:
            del request
            return ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="The test operator authorizes the reconstructed application profile.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )

    secret = "application_control_secret"
    session_id = "sess_application_control_rotation"
    session_store = InMemorySessionStore()
    policy = AdoptCurrentProfile()
    claimed_result = ToolResult(
        content="application result",
        structured={
            "isolated_tool_failure_code": secret,
            "tool_execution_boundary": "posix_process",
            "tool_timeout_strength": "hard_process_deadline",
            "detail": "safe",
        },
        is_error=True,
    )

    class ClaimingHook(RuntimeHook):
        async def before_tool_call(
            self,
            context: Any,
        ) -> BeforeToolCallDecision | None:
            del context
            if result_source == "before_hook":
                return BeforeToolCallDecision(
                    action="short_circuit",
                    synthetic_result=claimed_result,
                )
            return None

        async def after_tool_call(
            self,
            context: Any,
        ) -> AfterToolCallDecision | None:
            del context
            if result_source == "after_hook":
                return AfterToolCallDecision(
                    action="modify",
                    modified_result=claimed_result,
                )
            return None

    hook = ClaimingHook()
    tool = _ResultTool(
        claimed_result
        if result_source == "tool"
        else ToolResult(
            content="ordinary application result",
            is_error=result_source == "after_hook",
        )
    )
    first_provider = _FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_application_control",
                    name="result_tool",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    first_app = CayuApp(
        enable_logging=False,
        session_store=session_store,
        execution_profile_policy=policy,
    )
    first_app.register_provider(first_provider, default=True)
    first_app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    first_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[hook],
    )

    first_events = asyncio.run(
        _collect(
            first_app.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "run")],
                )
            )
        )
    )

    assert first_events[-1].type is EventType.SESSION_FAILED
    first_transcript = asyncio.run(session_store.load_transcript(session_id))
    first_result = next(message for message in first_transcript if message.role == "tool").content[
        0
    ]
    assert first_result.structured == {"detail": "safe"}

    second_provider = _FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    second_app = CayuApp(
        enable_logging=False,
        session_store=session_store,
        secret_redactor=SecretRedactor(secret),
        execution_profile_policy=policy,
    )
    second_app.register_provider(second_provider, default=True)
    second_app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    second_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[hook],
    )
    resumed_events = asyncio.run(
        _collect(
            second_app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key="adopt-rotated-tool-result-registry",
                        reason="The operator authorizes the updated secret registry.",
                        requested_by=ResolutionActor(
                            subject="test-operator",
                            source=ResolutionActorSource.REQUEST,
                        ),
                    ),
                )
            )
        )
    )

    assert resumed_events[-1].type is EventType.SESSION_COMPLETED
    provider_request = second_provider.requests[0]
    provider_result = next(
        message for message in provider_request.messages if message.role == "tool"
    ).content[0]
    assert provider_result.structured == {"detail": "safe"}
    serialized = json.dumps(
        {
            "events": [event.model_dump(mode="json") for event in resumed_events],
            "provider_request": provider_request.model_dump(mode="json"),
            "transcript": [
                message.model_dump(mode="json")
                for message in asyncio.run(session_store.load_transcript(session_id))
            ],
        }
    )
    assert secret not in serialized


def test_partial_hook_control_is_not_promoted_during_public_event_projection() -> None:
    from cayu import AfterToolCallDecision, RuntimeHook

    secret = "application_partial_terminal_outcome"
    session_id = "sess_partial_hook_control"
    session_store = InMemorySessionStore()

    class FailingTool(Tool):
        spec = ToolSpec(
            name="failing_tool",
            input_schema={"type": "object"},
            effect=ToolEffect.NONE,
        )

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx, args
            raise RuntimeError("deterministic tool failure")

    class PartialControlHook(RuntimeHook):
        async def after_tool_call(self, context: Any) -> AfterToolCallDecision | None:
            del context
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(
                    content="application replacement",
                    structured={
                        "terminal_outcome": secret,
                        "detail": "safe",
                    },
                    is_error=True,
                ),
            )

    provider = _FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_partial_hook_control",
                    name="failing_tool",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        enable_logging=False,
        session_store=session_store,
        secret_redactor=SecretRedactor(secret),
    )
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[FailingTool()],
        runtime_hooks=[PartialControlHook()],
    )

    events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "run")],
                )
            )
        )
    )
    durable_events = asyncio.run(session_store.load_events(session_id))
    transcript = asyncio.run(session_store.load_transcript(session_id))

    public_terminal = next(event for event in events if event.type is EventType.TOOL_CALL_FAILED)
    durable_terminal = next(
        event for event in durable_events if event.type is EventType.TOOL_CALL_FAILED
    )
    transcript_result = next(message for message in transcript if message.role == "tool").content[0]
    provider_result = next(
        message for message in provider.requests[1].messages if message.role == "tool"
    ).content[0]
    for structured in (
        public_terminal.payload["result"]["structured"],
        durable_terminal.payload["result"]["structured"],
        transcript_result.structured,
        provider_result.structured,
    ):
        assert structured["terminal_outcome"] == REDACTED_SECRET
        assert structured["detail"] == "safe"


def test_partial_recovery_control_survives_without_runtime_authority_overlay() -> None:
    from cayu.runtime import _tool_results as tool_results

    structured = {
        "recovered": True,
        "outcome_unknown": True,
        "recovery_reason": "pending_tool_round_missing_terminal_event",
    }

    assert tool_results.restore_runtime_tool_result_control_authority(structured, {}) == structured


def test_partial_hook_control_cannot_poison_runtime_timeout_boundary() -> None:
    from cayu import AfterToolCallDecision, RuntimeHook

    session_id = "sess_partial_hook_timeout_boundary"
    session_store = InMemorySessionStore()

    class SlowTool(Tool):
        spec = ToolSpec(
            name="slow_tool",
            input_schema={"type": "object"},
            effect=ToolEffect.NONE,
        )

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx, args
            await asyncio.sleep(30)
            return ToolResult(content="unexpected")

    class PartialControlHook(RuntimeHook):
        async def after_tool_call(self, context: Any) -> AfterToolCallDecision | None:
            del context
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(
                    content="application replacement",
                    structured={
                        "isolated_tool_failure_code": "application_value",
                        "detail": "safe",
                    },
                    is_error=True,
                ),
            )

    provider = _FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_partial_hook_timeout_boundary",
                    name="slow_tool",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        enable_logging=False,
        session_store=session_store,
        secret_redactor=SecretRedactor("in_process"),
        config=CayuConfig(tool_execution=ToolExecutionConfig(tool_timeout_seconds=0.01)),
    )
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SlowTool()],
        runtime_hooks=[PartialControlHook()],
    )

    public_events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "run")],
                )
            )
        )
    )
    durable_events = asyncio.run(session_store.load_events(session_id))
    transcript = asyncio.run(session_store.load_transcript(session_id))

    public_terminal = next(
        event for event in public_events if event.type is EventType.TOOL_CALL_FAILED
    )
    durable_terminal = next(
        event for event in durable_events if event.type is EventType.TOOL_CALL_FAILED
    )
    transcript_result = next(message for message in transcript if message.role == "tool").content[0]
    provider_result = next(
        message for message in provider.requests[1].messages if message.role == "tool"
    ).content[0]
    for event in (public_terminal, durable_terminal):
        assert event.payload["terminal_outcome"] == "tool_execution_timeout"
        assert event.payload["tool_execution_boundary"] == "in_process"
        assert event.payload["tool_timeout_strength"] == "cooperative_in_process"
    for structured in (
        public_terminal.payload["result"]["structured"],
        durable_terminal.payload["result"]["structured"],
        transcript_result.structured,
        provider_result.structured,
    ):
        assert structured["tool_execution_boundary"] == "in_process"
        assert structured["tool_timeout_strength"] == "cooperative_in_process"
        assert structured["detail"] == "safe"
        assert "isolated_tool_failure_code" not in structured


@pytest.mark.parametrize("result_source", ["before_hook", "after_hook"])
@pytest.mark.parametrize("result_validity", ["valid", "invalid"])
def test_hook_result_is_sanitized_before_secondary_publication_failure(
    result_source: str,
    result_validity: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cayu import AfterToolCallDecision, BeforeToolCallDecision, RuntimeHook

    class FailingHookPublicationStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def append_events(self, session_id: str, events: list[Event]) -> None:
            failure_event_type = (
                EventType.HOOK_COMPLETED if result_validity == "valid" else EventType.HOOK_FAILED
            )
            if not self.failed and any(event.type is failure_event_type for event in events):
                self.failed = True
                raise SystemExit("hook result publication failed")
            await super().append_events(session_id, events)

    class SecretResultHook(RuntimeHook):
        @staticmethod
        def _claimed_result() -> ToolResult:
            if result_validity == "invalid":
                return ToolResult.model_construct(
                    content="invalid hook result",
                    structured={"HOOK_PUBLICATION_SECRET_CANARY": object()},
                    artifacts=[],
                    is_error=True,
                )
            return ToolResult(
                content="HOOK_PUBLICATION_SECRET_CANARY",
                structured={
                    "isolated_tool_failure_code": "HOOK_PUBLICATION_SECRET_CANARY",
                    "tool_execution_boundary": "posix_process",
                    "tool_timeout_strength": "hard_process_deadline",
                    "detail": "safe",
                },
                is_error=True,
            )

        async def before_tool_call(self, context: Any) -> BeforeToolCallDecision | None:
            del context
            if result_source != "before_hook":
                return None
            if result_validity == "invalid":
                return BeforeToolCallDecision.model_construct(
                    action="short_circuit",
                    synthetic_result=self._claimed_result(),
                )
            return BeforeToolCallDecision(
                action="short_circuit",
                synthetic_result=self._claimed_result(),
            )

        async def after_tool_call(self, context: Any) -> AfterToolCallDecision | None:
            del context
            if result_source != "after_hook":
                return None
            if result_validity == "invalid":
                return AfterToolCallDecision.model_construct(
                    action="modify",
                    modified_result=self._claimed_result(),
                )
            return AfterToolCallDecision(
                action="modify",
                modified_result=self._claimed_result(),
            )

    async def scenario() -> None:
        provider = _FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_hook_publication_failure",
                        name="result_tool",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        app = CayuApp(
            enable_logging=False,
            session_store=FailingHookPublicationStore(),
            secret_redactor=SecretRedactor("HOOK_PUBLICATION_SECRET_CANARY"),
        )
        app.register_provider(provider, default=True)
        app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_ResultTool(ToolResult(content="ordinary application result"))],
            runtime_hooks=[SecretResultHook()],
        )
        await _collect(
            app.run(
                RunRequest(
                    session_id=(f"sess_hook_publication_failure_{result_source}_{result_validity}"),
                    agent_name="assistant",
                    messages=[Message.text("user", "run")],
                )
            )
        )

    with (
        warnings.catch_warnings(record=True) as captured_warnings,
        pytest.raises(SystemExit) as caught,
    ):
        asyncio.run(scenario())

    captured_output = capsys.readouterr()
    diagnostics = [
        str(caught.value),
        repr(caught.value),
        caplog.text,
        captured_output.out,
        captured_output.err,
        *(str(item.message) for item in captured_warnings),
    ]
    current: BaseException | None = caught.value
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        diagnostics.extend((str(current), repr(current), repr(current.args)))
        traceback = current.__traceback__
        while traceback is not None:
            if "/src/cayu/" in traceback.tb_frame.f_code.co_filename:
                diagnostics.extend(repr(value) for value in traceback.tb_frame.f_locals.values())
            traceback = traceback.tb_next
        current = current.__cause__ or current.__context__

    assert "HOOK_PUBLICATION_SECRET_CANARY" not in "\n".join(diagnostics)


def test_application_result_cannot_claim_web_access_control_ownership() -> None:
    from cayu.core import MessageRole, ToolResultPart
    from cayu.runtime._message_redaction import redact_runtime_message_for_boundary

    secret = "application-web-control-secret"
    lookalike = Message(
        role=MessageRole.TOOL,
        content=(
            ToolResultPart(
                tool_call_id="call_application_web_control",
                tool_name="application_tool",
                content="application result",
                structured={
                    "access": {
                        "schema_version": 1,
                        "outcome": "bot_challenge",
                        "source": "hosted_provider",
                        "signal": "provider_status",
                        "destination_fingerprint": secret,
                        "status_code": secret,
                    }
                },
            ),
        ),
    )

    redacted = redact_runtime_message_for_boundary(
        lookalike,
        redactor=SecretRedactor(secret),
        field_name="message",
    )

    structured = redacted.content[0].structured
    assert structured["access"]["outcome"] == "bot_challenge"
    assert structured["access"]["destination_fingerprint"] == REDACTED_SECRET
    assert structured["access"]["status_code"] == REDACTED_SECRET


def test_valid_projection_re_redacts_content_under_rotated_message_registry() -> None:
    from cayu.core import MessageRole, ToolResultPart
    from cayu.runtime._message_redaction import redact_runtime_message_for_boundary

    secret = "rotated-message-secret"
    artifact_id = f"art_{'a' * 32}"
    reference = {
        "type": "cayu.tool_result_artifact.v1",
        "artifact_id": artifact_id,
        "store_id": "artifacts",
        "filename": f"tool-result-{artifact_id}.txt",
        "content_type": "text/plain; charset=utf-8",
        "size_bytes": 1_000,
        "sha256": "b" * 64,
        "scope": "session",
        "session_id_sha256": "c" * 64,
        "projection_authority": "cayu.tool_result_projection.v1",
        "readback_max_bytes": 64,
    }
    message = Message(
        role=MessageRole.TOOL,
        content=(
            ToolResultPart(
                tool_call_id="call_rotated_message",
                tool_name="result_tool",
                content=f"projection preview with {secret}",
                artifacts=[reference],
            ),
        ),
    )

    redacted = redact_runtime_message_for_boundary(
        message,
        redactor=SecretRedactor(secret),
        field_name="message",
    )

    serialized = json.dumps(redacted.model_dump(mode="json"))
    assert secret not in serialized
    assert redacted.content[0].artifacts[0] == reference
    assert artifact_id in redacted.content[0].content
    assert "read_file" in redacted.content[0].content


def test_rotated_secret_expansion_keeps_projected_content_bounded() -> None:
    from cayu.runtime.tool_result_projection import (
        redact_tool_result_projection_content,
    )

    artifact_id = f"art_{'a' * 32}"
    projected = redact_tool_result_projection_content(
        "z" * 60_000,
        artifact_id=artifact_id,
        readback_max_bytes=64,
        redact_text=SecretRedactor("z").redact_text,
    )

    assert len(projected.encode("utf-8")) <= MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES
    assert "z" not in projected
    assert artifact_id in projected


def test_cayu_app_publishes_bounded_failure_without_oversized_fallback(tmp_path) -> None:
    original = "never-publish-" + ("q" * 10_000)
    store = _FailingArtifactStore(
        tmp_path / "failing-artifacts",
        store_id="failing-artifacts",
    )
    policy = ArtifactExternalizingToolResultPolicy(
        max_inline_bytes=64,
        max_inline_token_estimate=None,
        preview_bytes=16,
    )

    app, _, provider, _, events = _run_tool_result(
        tmp_path=tmp_path,
        content=original,
        policy=policy,
        store=store,
        structured={"receipt_id": "receipt-1"},
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    terminal = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert terminal.payload["tool_result_projection"]["status"] == "failed"
    assert terminal.payload["tool_result_projection"]["failure_type"] == (
        "ArtifactStoreUnavailableError"
    )
    assert terminal.payload["result"]["structured"] == {"receipt_id": "receipt-1"}
    assert terminal.payload["result"]["is_error"] is False
    assert original not in terminal.payload["result"]["content"]
    assert len(terminal.payload["result"]["content"].encode()) < 1024

    transcript = asyncio.run(app.session_store.load_transcript("sess_runtime_projection"))
    serialized = json.dumps(
        {
            "events": [event.model_dump(mode="json") for event in events],
            "transcript": [message.model_dump(mode="json") for message in transcript],
            "provider_request": provider.requests[1].model_dump(mode="json"),
        }
    )
    assert original not in serialized
    assert "q" * 100 not in serialized


def test_cayu_app_projects_local_orphan_candidate_without_artifact_authority(
    tmp_path,
    monkeypatch,
) -> None:
    primary_canary = "provider-primary-secret-canary"
    cleanup_canary = "provider-cleanup-secret-canary"

    def fail_publication(*_args, **_kwargs) -> None:
        raise OSError(primary_canary)

    def fail_cleanup(*_args, **_kwargs) -> None:
        raise OSError(cleanup_canary)

    monkeypatch.setattr(local_artifacts, "_rename_directory_no_replace", fail_publication)
    monkeypatch.setattr(
        local_artifacts,
        "_remove_artifact_directory_if_unchanged",
        fail_cleanup,
    )
    policy = ArtifactExternalizingToolResultPolicy(
        max_inline_bytes=64,
        max_inline_token_estimate=None,
    )

    _app, _store, _provider, tool, events = _run_tool_result(
        tmp_path=tmp_path,
        content="orphan-" + ("x" * 10_000),
        policy=policy,
    )

    terminal = next(event for event in events if event.type is EventType.TOOL_CALL_COMPLETED)
    projection = terminal.payload["tool_result_projection"]
    settlement = projection["artifact_write_settlement"]
    assert projection["status"] == "failed"
    assert settlement["status"] == "reconciliation_required"
    assert settlement["phase"] == "cleanup"
    assert settlement["artifact_id"].startswith("art_")
    assert settlement["failure_codes"] == ["mutation_failed", "cleanup_failed"]
    assert "artifact_id" not in projection
    assert terminal.payload["result"]["artifacts"] == tool.result.artifacts
    serialized = json.dumps(terminal.model_dump(mode="json"))
    assert primary_canary not in serialized
    assert cleanup_canary not in serialized


@pytest.mark.parametrize("mutation", ("hostile_value", "oversized_mapping"))
def test_runtime_rejects_mutated_settlement_before_diagnostic_serialization(
    tmp_path,
    caplog,
    capsys,
    mutation,
) -> None:
    secret = "rejected-projection-settlement-secret-canary"

    class SecretValue:
        def __str__(self) -> str:
            return secret

        def __repr__(self) -> str:
            return secret

    class MutatedSettlementPolicy(ToolResultProjectionPolicy):
        @property
        def identity(self) -> str:
            return "tests.mutated_settlement_projection.v1"

        async def project(self, request: ToolResultProjectionRequest):
            projected_result = ToolResult(
                content="[projection rejected]",
                structured=request.result.structured,
                artifacts=request.result.artifacts,
                is_error=request.result.is_error,
            )
            now = datetime.now(UTC)
            settlement = ArtifactWriteSettlementEvidence(
                operation_id=f"artifact_write_{'e' * 32}",
                artifact_id=f"art_{'f' * 32}",
                store_identity_sha256=artifact_store_identity_sha256("test-store"),
                status=ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED,
                phase=ArtifactWriteSettlementPhase.COMMIT,
                observation=ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
                started_at=now,
                observed_at=now,
                elapsed_ms=0,
                failure_codes=(ArtifactWriteSettlementFailureCode.COMMIT_FAILED,),
            )
            projection = ToolResultProjection(
                result=projected_result,
                record=ToolResultProjectionRecord(
                    status=ToolResultProjectionStatus.FAILED,
                    policy_id=self.identity,
                    original_bytes=len(request.result.content.encode("utf-8")),
                    projected_bytes=len(projected_result.content.encode("utf-8")),
                    original_token_estimate=0,
                    projected_token_estimate=0,
                    token_estimation_method="tests_exact_v1",
                    failure_type="test_failure",
                    artifact_write_settlement=settlement,
                ),
            )
            if mutation == "hostile_value":
                copied_settlement = projection.record.artifact_write_settlement
                assert copied_settlement is not None
                object.__setattr__(copied_settlement, "artifact_id", SecretValue())
            else:
                object.__setattr__(
                    projection.record,
                    "artifact_write_settlement",
                    {"unexpected": secret * 1024},
                )
            return projection

    with warnings.catch_warnings(record=True) as captured_warnings:
        _app, _store, _provider, _tool, events = _run_tool_result(
            tmp_path=tmp_path,
            content="ordinary application result",
            policy=MutatedSettlementPolicy(),
        )

    terminal = next(event for event in events if event.type is EventType.TOOL_CALL_COMPLETED)
    projection = terminal.payload["tool_result_projection"]
    output = capsys.readouterr()
    diagnostics = (
        caplog.text,
        output.out,
        output.err,
        *(str(item.message) for item in captured_warnings),
    )
    assert projection["status"] == "failed"
    assert "artifact_write_settlement" not in projection
    assert all(secret not in diagnostic for diagnostic in diagnostics)


@pytest.mark.parametrize(
    ("fallback", "failure_type"),
    (
        ("raise", "ArtifactStoreUnavailableError"),
        ("missing", "missing_projection_result"),
        ("invalid", "TypeError"),
    ),
)
def test_custom_projection_failure_preserves_observed_store_settlement(
    tmp_path,
    fallback,
    failure_type,
) -> None:
    store = _RegisteredFailingArtifactStore(
        tmp_path / "registered-failing-artifacts",
        store_id="registered-failing-artifacts",
    )

    _app, _store, _provider, tool, events = _run_tool_result(
        tmp_path=tmp_path,
        content="custom policy result",
        policy=_ObservedArtifactFailureProjectionPolicy(fallback),
        store=store,
    )

    terminal = next(event for event in events if event.type is EventType.TOOL_CALL_COMPLETED)
    projection = terminal.payload["tool_result_projection"]
    settlement = projection["artifact_write_settlement"]
    assert projection["status"] == "failed"
    assert projection["failure_type"] == failure_type
    assert settlement["status"] == "reconciliation_required"
    assert settlement["phase"] == "reconciliation"
    assert settlement["failure_codes"] == ["commit_failed"]
    assert "artifact_id" not in projection
    assert terminal.payload["result"]["artifacts"] == tool.result.artifacts
    assert "third-party settlement canary" not in json.dumps(terminal.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("behavior", "failure_type"),
    (("raise", "RuntimeError"), ("return", "ValueError")),
)
def test_runtime_rejects_historical_settlement_not_observed_for_current_projection(
    tmp_path,
    behavior,
    failure_type,
) -> None:
    _app, _store, _provider, _tool, events = _run_tool_result(
        tmp_path=tmp_path,
        content="ordinary current result",
        policy=_HistoricalSettlementProjectionPolicy(behavior),
    )

    terminal = next(event for event in events if event.type is EventType.TOOL_CALL_COMPLETED)
    projection = terminal.payload["tool_result_projection"]
    assert projection["status"] == "failed"
    assert projection["failure_type"] == failure_type
    assert "artifact_write_settlement" not in projection


@pytest.mark.parametrize(
    ("backend_locator", "backend_version", "omitted_field", "retained_field"),
    (
        (
            "s3://private-bucket/{secret}",
            "safe-driver-version",
            "backend_locator",
            "backend_version",
        ),
        (
            "safe-backend-locator",
            "driver-{secret}",
            "backend_version",
            "backend_locator",
        ),
    ),
)
def test_runtime_omits_only_secret_bearing_settlement_extension_metadata(
    tmp_path,
    backend_locator,
    backend_version,
    omitted_field,
    retained_field,
) -> None:
    secret = "registered-backend-locator-secret"
    backend_locator = backend_locator.format(secret=secret)
    backend_version = backend_version.format(secret=secret)
    store = _RegisteredFailingArtifactStore(
        tmp_path / "secret-locator-artifacts",
        store_id="secret-locator-artifacts",
        backend_locator=backend_locator,
        backend_version=backend_version,
    )

    app, _store, _provider, _tool, events = _run_tool_result(
        tmp_path=tmp_path,
        content="custom policy result",
        policy=_ObservedArtifactFailureProjectionPolicy("raise"),
        store=store,
        secret_redactor=SecretRedactor(secret),
    )
    durable_events = asyncio.run(app.session_store.load_events("sess_runtime_projection"))

    for event_set in (events, durable_events):
        terminal = next(event for event in event_set if event.type is EventType.TOOL_CALL_COMPLETED)
        settlement = terminal.payload["tool_result_projection"]["artifact_write_settlement"]
        assert settlement["status"] == "reconciliation_required"
        assert omitted_field not in settlement
        expected_retained = (
            backend_locator if retained_field == "backend_locator" else backend_version
        )
        assert settlement[retained_field] == expected_retained
        assert secret not in json.dumps(terminal.model_dump(mode="json"))


def test_self_cancelled_projection_policy_publishes_bounded_failure() -> None:
    provider = _FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_self_cancelled_projection",
                    name="result_tool",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        enable_logging=False,
        tool_result_projection_policy=_SelfCancellingProjectionPolicy(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[_ResultTool(ToolResult(content="completed tool result"))],
    )

    events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    session_id="sess_self_cancelled_projection",
                    agent_name="assistant",
                    messages=[Message.text("user", "run")],
                )
            )
        )
    )

    terminal = next(event for event in events if event.type is EventType.TOOL_CALL_COMPLETED)
    assert terminal.payload["tool_result_projection"]["status"] == "failed"
    assert terminal.payload["tool_result_projection"]["failure_type"] == "RuntimeError"
    assert terminal.payload["result"]["content"].startswith(
        "Cayu could not externalize this oversized tool result"
    )
    assert events[-1].type is EventType.SESSION_COMPLETED


def test_externalized_tool_result_can_be_read_through_bounded_read_file(tmp_path) -> None:
    original = "readback-" + ("r" * 10_000)
    store = LocalArtifactStore(
        tmp_path / "readback-artifacts",
        store_id="readback-artifacts",
    )
    provider = _ReadbackProvider()
    tool = _ResultTool(ToolResult(content=original))
    app = CayuApp(
        enable_logging=False,
        tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
            max_inline_bytes=2048,
            max_inline_token_estimate=None,
            preview_bytes=32,
        ),
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=store),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool, ReadFileTool()],
    )

    events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    session_id="sess_projection_readback",
                    agent_name="assistant",
                    messages=[Message.text("user", "run and inspect")],
                )
            )
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert provider.externalized_artifact_id is not None
    assert provider.readback_arguments == {
        "artifact_id": provider.externalized_artifact_id,
        "max_bytes": 2030,
    }
    assert len(provider.requests) == 3
    readback_result = next(
        part
        for message in provider.requests[2].messages
        if message.role == "tool"
        for part in message.content
        if part.tool_call_id == "call_readback"
    )
    assert readback_result.structured["source"] == "artifact"
    assert readback_result.structured["artifact_id"] == provider.externalized_artifact_id
    assert readback_result.structured["truncated"] is True
    assert readback_result.content.startswith(original[:64])
    assert original not in readback_result.content
    readback_terminal = next(
        event
        for event in events
        if event.type is EventType.TOOL_CALL_COMPLETED and event.tool_name == "read_file"
    )
    assert readback_terminal.payload["tool_result_projection"]["status"] == "unchanged"


def test_mcp_tool_results_cross_the_same_projection_boundary(tmp_path) -> None:
    original = "mcp-large-" + ("m" * 10_000)
    session = _LargeMcpSession(original)
    toolset = McpToolset(
        server=McpServerSpec(
            name="large-mcp",
            connection_id="large-mcp-v1",
            command=["unused"],
        ),
        session=session,
        definitions=(session.definition,),
    )
    store = LocalArtifactStore(
        tmp_path / "mcp-artifacts",
        store_id="mcp-artifacts",
    )
    provider = _FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_mcp",
                    name=toolset.tools[0].name,
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        enable_logging=False,
        tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
            max_inline_bytes=64,
            max_inline_token_estimate=None,
            preview_bytes=16,
        ),
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=store),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=toolset.tools,
    )

    events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    session_id="sess_mcp_projection",
                    agent_name="assistant",
                    messages=[Message.text("user", "run MCP")],
                )
            )
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert session.calls == 1
    terminal = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert terminal.payload["tool_result_projection"]["status"] == "externalized"
    artifact_id = terminal.payload["tool_result_projection"]["artifact_id"]
    stored = asyncio.run(store.read_bytes(artifact_id))
    assert original in stored.content.decode()
    provider_result = next(
        message for message in provider.requests[1].messages if message.role == "tool"
    ).content[0]
    assert original not in provider_result.content
    assert provider_result.structured["mcp_structured_content"] == {"receipt_id": "mcp-receipt"}


def test_effectful_terminal_failure_crosses_projection_before_observational_hooks(
    tmp_path,
) -> None:
    store = LocalArtifactStore(tmp_path / "effectful-artifacts", store_id="effectful-artifacts")
    provider = _FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_effectful_failure",
                    name="invalid_constructed_result",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        enable_logging=False,
        tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
            max_inline_bytes=20,
            max_inline_token_estimate=None,
            preview_bytes=0,
        ),
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=store),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[_InvalidConstructedResultTool()],
    )

    events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    session_id="sess_effectful_failure_projection",
                    agent_name="assistant",
                    messages=[Message.text("user", "run")],
                )
            )
        )
    )

    terminal = next(event for event in events if event.type is EventType.TOOL_CALL_FAILED)
    assert terminal.payload["tool_effect"] == "external"
    assert terminal.payload["outcome_unknown"] is True
    assert terminal.payload["manual_reconciliation_required"] is True
    assert terminal.payload["tool_result_projection"]["status"] == "externalized"
    artifact = asyncio.run(
        store.read_bytes(terminal.payload["tool_result_projection"]["artifact_id"])
    )
    assert artifact.content == b"Tool returned a non-portable result after execution."


def test_interruption_during_artifact_persistence_does_not_repeat_tool_or_store(
    tmp_path,
) -> None:
    async def scenario() -> tuple[
        _ResultTool,
        _BlockingArtifactStore,
        list[Event],
        list[Event],
        list[Event],
    ]:
        original = "interruptible-" + ("i" * 10_000)
        artifact_store = _BlockingArtifactStore(
            tmp_path / "interrupt-artifacts",
            store_id="interrupt-artifacts",
        )
        session_store = InMemorySessionStore()
        provider = _FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_interrupt",
                        name="result_tool",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        tool = _ResultTool(ToolResult(content=original))
        app = CayuApp(
            enable_logging=False,
            session_store=session_store,
            tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
                max_inline_bytes=1024,
                max_inline_token_estimate=None,
            ),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                artifact_store=artifact_store,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )
        session_id = "sess_projection_interrupt"
        run_task = asyncio.create_task(
            _collect(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run")],
                    )
                )
            )
        )
        await asyncio.wait_for(artifact_store.started.wait(), timeout=5)
        interrupt_task = asyncio.create_task(
            _collect(
                app.interrupt_session(
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="test projection interruption",
                    )
                )
            )
        )

        async def wait_until_interrupting() -> None:
            while True:
                session = await session_store.load(session_id)
                if session is not None and session.status is SessionStatus.INTERRUPTING:
                    return
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_until_interrupting(), timeout=5)
        artifact_store.release.set()
        interrupt_events = await asyncio.wait_for(interrupt_task, timeout=5)
        run_events = await asyncio.wait_for(run_task, timeout=5)
        resumed_events = await asyncio.wait_for(
            _collect(
                app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "continue")],
                    )
                )
            ),
            timeout=5,
        )
        return tool, artifact_store, interrupt_events, run_events, resumed_events

    tool, artifact_store, interrupt_events, run_events, resumed_events = asyncio.run(scenario())

    assert interrupt_events[-1].type == EventType.SESSION_INTERRUPTED
    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == 1
    assert artifact_store.writes == 1
    terminal = next(event for event in run_events if event.type is EventType.TOOL_CALL_COMPLETED)
    assert terminal.payload["tool_result_projection"]["status"] == "externalized"
    listed = asyncio.run(artifact_store.list(session_id="sess_projection_interrupt"))
    assert len(listed.artifacts) == 1


def test_projection_timeout_allows_interrupt_to_finish_without_store_release(
    tmp_path,
    monkeypatch,
) -> None:
    from cayu.runtime import _tool_round_executor

    monkeypatch.setattr(
        _tool_round_executor,
        "_TOOL_RESULT_PROJECTION_TIMEOUT_SECONDS",
        0.01,
    )

    async def scenario() -> tuple[list[Event], list[Event], _BlockingArtifactStore]:
        artifact_store = _BlockingArtifactStore(
            tmp_path / "timeout-artifacts",
            store_id="timeout-artifacts",
        )
        provider = _FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_projection_timeout",
                        name="result_tool",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        app = CayuApp(
            enable_logging=False,
            tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
                max_inline_bytes=64,
                max_inline_token_estimate=None,
            ),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), artifact_store=artifact_store),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_ResultTool(ToolResult(content="x" * 10_000))],
        )
        session_id = "sess_projection_timeout"
        run_task = asyncio.create_task(
            _collect(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run")],
                    )
                )
            )
        )
        await asyncio.wait_for(artifact_store.started.wait(), timeout=5)
        interrupt_events = await asyncio.wait_for(
            _collect(
                app.interrupt_session(
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="projection store did not settle",
                    )
                )
            ),
            timeout=1,
        )
        run_events = await asyncio.wait_for(run_task, timeout=1)
        return interrupt_events, run_events, artifact_store

    interrupt_events, run_events, artifact_store = asyncio.run(scenario())

    assert interrupt_events[-1].type is EventType.SESSION_INTERRUPTED
    assert run_events[-1].type is EventType.SESSION_INTERRUPTED
    terminal = next(event for event in run_events if event.type is EventType.TOOL_CALL_COMPLETED)
    assert terminal.payload["tool_result_projection"]["status"] == "failed"
    assert terminal.payload["tool_result_projection"]["failure_type"] == "projection_timeout"
    assert artifact_store.writes == 1


def test_projection_timeout_records_active_local_write_without_artifact_authority(
    tmp_path,
    monkeypatch,
) -> None:
    from cayu.runtime import _tool_round_executor

    monkeypatch.setattr(
        _tool_round_executor,
        "_TOOL_RESULT_PROJECTION_TIMEOUT_SECONDS",
        0.01,
    )
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    rename = local_artifacts._rename_directory_no_replace

    def blocked_rename(*args, **kwargs) -> None:
        started.set()
        try:
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release local artifact publication")
            rename(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(local_artifacts, "_rename_directory_no_replace", blocked_rename)

    async def scenario() -> tuple[list[Event], list[Event]]:
        artifact_store = LocalArtifactStore(
            tmp_path / "active-timeout-artifacts",
            store_id="active-timeout-artifacts",
        )
        provider = _FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_active_projection_timeout",
                        name="result_tool",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        app = CayuApp(
            enable_logging=False,
            tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
                max_inline_bytes=64,
                max_inline_token_estimate=None,
            ),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), artifact_store=artifact_store),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_ResultTool(ToolResult(content="x" * 10_000))],
        )
        session_id = "sess_active_projection_timeout"
        run_task = asyncio.create_task(
            _collect(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run")],
                    )
                )
            )
        )
        try:
            while not started.is_set():
                await asyncio.sleep(0)
            interrupt_events = await asyncio.wait_for(
                _collect(
                    app.interrupt_session(
                        InterruptSessionRequest(
                            session_id=session_id,
                            reason="active local publication did not settle",
                        )
                    )
                ),
                timeout=1,
            )
            run_events = await asyncio.wait_for(run_task, timeout=1)
            return interrupt_events, run_events
        finally:
            release.set()
            while not finished.is_set():
                await asyncio.sleep(0)
            await asyncio.sleep(0)

    interrupt_events, run_events = asyncio.run(scenario())

    assert interrupt_events[-1].type is EventType.SESSION_INTERRUPTED
    assert run_events[-1].type is EventType.SESSION_INTERRUPTED
    terminal = next(event for event in run_events if event.type is EventType.TOOL_CALL_COMPLETED)
    projection = terminal.payload["tool_result_projection"]
    settlement = projection["artifact_write_settlement"]
    assert projection["status"] == "failed"
    assert projection["failure_type"] == "projection_timeout"
    assert "artifact_id" not in projection
    assert settlement["status"] == "reconciliation_required"
    # The deadline may race content fsync with the immediately following
    # commit-phase notification. Both are truthful last-observed phases.
    assert settlement["phase"] in {"content", "commit"}
    assert settlement["failure_codes"] == ["settlement_deadline_expired"]
    assert settlement["artifact_id"].startswith("art_")
    assert terminal.payload["result"]["artifacts"] == []


def test_late_projection_completion_is_an_identifiable_publication_orphan(
    tmp_path,
    monkeypatch,
) -> None:
    from cayu.runtime import _tool_round_executor

    monkeypatch.setattr(
        _tool_round_executor,
        "_TOOL_RESULT_PROJECTION_TIMEOUT_SECONDS",
        0.01,
    )

    async def scenario() -> tuple[list[Event], dict[str, Any]]:
        artifact_store = _LateCompletingArtifactStore(
            tmp_path / "late-artifacts",
            store_id="late-artifacts",
        )
        provider = _FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_late_projection",
                        name="result_tool",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            enable_logging=False,
            tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
                max_inline_bytes=64,
                max_inline_token_estimate=None,
            ),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), artifact_store=artifact_store),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_ResultTool(ToolResult(content="late-" + ("x" * 10_000)))],
        )

        events: list[Event] = []

        async def collect_run() -> None:
            async for event in app.run(
                RunRequest(
                    session_id="sess_late_projection",
                    agent_name="assistant",
                    messages=[Message.text("user", "run")],
                )
            ):
                events.append(event)

        run_task = asyncio.create_task(collect_run())
        await asyncio.wait_for(artifact_store.cancellation_observed.wait(), timeout=1)

        async def wait_for_terminal_projection() -> None:
            while not any(event.type is EventType.TOOL_CALL_COMPLETED for event in events):
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_terminal_projection(), timeout=1)
        terminal = next(event for event in events if event.type is EventType.TOOL_CALL_COMPLETED)
        settlement = terminal.payload["tool_result_projection"]["artifact_write_settlement"]
        assert settlement["status"] == "reconciliation_required"
        assert settlement["phase"] == "content"
        assert not run_task.done()
        artifact_store.release.set()
        await asyncio.wait_for(run_task, timeout=1)

        async def wait_for_orphan() -> dict[str, Any]:
            while True:
                listed = await artifact_store.list(session_id="sess_late_projection")
                if listed.artifacts:
                    return dict(listed.artifacts[0].metadata)
                await asyncio.sleep(0)

        metadata = await asyncio.wait_for(wait_for_orphan(), timeout=1)
        return events, metadata

    events, metadata = asyncio.run(scenario())

    terminal = next(event for event in events if event.type is EventType.TOOL_CALL_COMPLETED)
    projection = terminal.payload["tool_result_projection"]
    assert projection["failure_type"] == "projection_timeout"
    assert "artifact_id" not in projection
    settlement = projection["artifact_write_settlement"]
    assert settlement["status"] == "reconciliation_required"
    assert settlement["phase"] == "content"
    assert settlement["artifact_id"].startswith("art_")
    assert terminal.payload["result"]["artifacts"] == []
    assert metadata["type"] == "cayu.tool_result_artifact.v1"
    assert metadata["logical_identity_sha256"]
    assert metadata["tool_call_id_sha256"]


def test_recovery_reuses_the_persisted_projection_without_reexecuting_the_tool(
    tmp_path,
) -> None:
    original = "recoverable-" + ("z" * 10_000)
    artifact_store = LocalArtifactStore(
        tmp_path / "recovery-artifacts",
        store_id="recovery-artifacts",
    )
    session_store = _RejectFirstToolRoundPublicationStore()
    provider = _FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_recoverable",
                    name="result_tool",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    tool = _ResultTool(ToolResult(content=original))
    app = CayuApp(
        enable_logging=False,
        session_store=session_store,
        tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
            max_inline_bytes=1024,
            max_inline_token_estimate=None,
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
    session_id = "sess_projection_recovery"

    first_events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "run the tool")],
                )
            )
        )
    )

    assert first_events[-1].type == EventType.SESSION_FAILED
    assert tool.calls == 1
    assert session_store.tool_round_publications == 1
    first_terminals = [
        event
        for event in asyncio.run(session_store.load_events(session_id))
        if event.type == EventType.TOOL_CALL_COMPLETED
    ]
    assert len(first_terminals) == 1
    artifact_id = first_terminals[0].payload["tool_result_projection"]["artifact_id"]
    assert [
        item.id for item in asyncio.run(artifact_store.list(session_id=session_id)).artifacts
    ] == [artifact_id]

    resumed_events = asyncio.run(
        _collect(
            app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                )
            )
        )
    )

    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == 1
    assert session_store.tool_round_publications == 2
    assert [
        item.id for item in asyncio.run(artifact_store.list(session_id=session_id)).artifacts
    ] == [artifact_id]
    transcript = asyncio.run(session_store.load_transcript(session_id))
    tool_result = next(
        part
        for message in transcript
        if message.role == "tool"
        for part in message.content
        if part.tool_call_id == "call_recoverable"
    )
    assert tool_result.content == first_terminals[0].payload["result"]["content"]
    assert original not in tool_result.content
    assert original not in provider.requests[1].model_dump_json()


def test_in_memory_session_store_preserves_projected_tool_results(tmp_path) -> None:
    async def scenario() -> None:
        session_store = InMemorySessionStore()
        artifact_store = LocalArtifactStore(tmp_path / "in-memory-artifacts")
        await assert_tool_result_projection_session_store_conformance(
            session_store,
            artifact_store,
            session_id="sess_projection_in_memory",
        )
        await assert_tool_result_projection_recovery_conformance(
            session_store,
            artifact_store,
            session_id="sess_projection_recovery_in_memory",
        )
        await assert_tool_result_projection_orphan_evidence_conformance(
            session_store,
            LocalArtifactStore(tmp_path / "in-memory-orphan-artifacts"),
            session_id="sess_projection_orphan_in_memory",
        )

    asyncio.run(scenario())


def test_sqlite_session_store_preserves_projected_tool_results(tmp_path) -> None:
    session_store = SQLiteSessionStore(tmp_path / "projection.sqlite")

    async def scenario() -> None:
        try:
            await assert_tool_result_projection_session_store_conformance(
                session_store,
                LocalArtifactStore(tmp_path / "sqlite-artifacts"),
                session_id="sess_projection_sqlite",
            )
            await assert_tool_result_projection_recovery_conformance(
                session_store,
                LocalArtifactStore(tmp_path / "sqlite-recovery-artifacts"),
                session_id="sess_projection_recovery_sqlite",
            )
            await assert_tool_result_projection_orphan_evidence_conformance(
                session_store,
                LocalArtifactStore(tmp_path / "sqlite-orphan-artifacts"),
                session_id="sess_projection_orphan_sqlite",
            )
        finally:
            await session_store.close()

    asyncio.run(scenario())
