from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import SecretStr
from tests.core.tool_result_projection_conformance import (
    assert_tool_result_projection_recovery_conformance,
    assert_tool_result_projection_session_store_conformance,
)

from cayu import (
    MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES,
    MAX_TOOL_RESULT_ARTIFACT_REFERENCE_BYTES,
    MAX_TOOL_RESULT_PREVIEW_BYTES,
    AgentSpec,
    ArtifactExternalizingToolResultPolicy,
    ArtifactStoreUnavailableError,
    CayuApp,
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
    ToolResult,
    ToolResultProjectionPolicy,
    ToolResultProjectionRequest,
    ToolSpec,
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
        self.writes += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancellation_observed.set()
            task = asyncio.current_task()
            if task is not None:
                while task.cancelling():
                    task.uncancel()
            await self.release.wait()
        return await LocalArtifactStore.put_bytes(
            self,
            content,
            filename=filename,
            **kwargs,
        )


class _SelfCancellingProjectionPolicy(ToolResultProjectionPolicy):
    @property
    def identity(self) -> str:
        return "tests.self_cancelling_projection.v1"

    async def project(self, request: ToolResultProjectionRequest):
        del request
        raise asyncio.CancelledError("projection policy cancelled itself")


class _RejectFirstToolRoundPublicationStore(InMemorySessionStore):
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
    policy: ArtifactExternalizingToolResultPolicy | None,
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
    assert projection.record.status == "externalized"
    assert "session_id" not in reference
    assert reference["session_id_sha256"] == hashlib.sha256(session_id.encode()).hexdigest()
    assert (
        len(
            json.dumps(
                reference,
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

        events = await asyncio.wait_for(
            _collect(
                app.run(
                    RunRequest(
                        session_id="sess_late_projection",
                        agent_name="assistant",
                        messages=[Message.text("user", "run")],
                    )
                )
            ),
            timeout=1,
        )
        await asyncio.wait_for(artifact_store.cancellation_observed.wait(), timeout=1)
        artifact_store.release.set()

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
    assert terminal.payload["tool_result_projection"]["failure_type"] == "projection_timeout"
    assert "artifact_id" not in terminal.payload["tool_result_projection"]
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
        finally:
            await session_store.close()

    asyncio.run(scenario())
