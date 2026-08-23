from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator

import pytest
from tests.core.postgres_contention_support import drop_cayu_tables

from cayu import (
    EVAL_SCENARIO_MAX_TOTAL_ARTIFACT_BYTES,
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    ArtifactReadResult,
    ArtifactScope,
    CayuApp,
    EnqueueSessionMessageRequest,
    Environment,
    EnvironmentSpec,
    EvaluationSourceIdentityV1,
    EventType,
    FileAttachment,
    FilePart,
    InMemorySessionStore,
    LocalArtifactStore,
    Message,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    PostgresSessionStore,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RunRequest,
    ScenarioCaptureDiagnosticCode,
    SessionMessageDeliveryMode,
    SQLiteSessionStore,
    TextPart,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolContext,
    ToolResult,
    ToolSpec,
    UserInputResponse,
    capture_eval_scenario_from_session,
    file_attachment,
    trajectory_from_session,
)
from cayu.artifacts.attachments import MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY
from cayu.core.events import event_payload_authority_is_runtime_generated
from cayu.evals.scenario_capture import _resolve_artifact_requirements
from cayu.runtime.sessions import (
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    parse_session_input_contract_evidence,
)
from cayu.storage.migrations import SchemaMode
from cayu.tools.user_input import UserInputTool
from cayu.vaults import REDACTED_SECRET, SecretRedactor


def _source() -> EvaluationSourceIdentityV1:
    return EvaluationSourceIdentityV1(
        application_release_id="release-current",
        app_manifest_schema_version="1",
        app_manifest_fingerprint="a" * 64,
        evidence_revision="sha256:" + "b" * 64,
    )


@pytest.fixture(params=("memory", "sqlite", pytest.param("postgres", marks=pytest.mark.postgres)))
def scenario_capture_store_case(request, tmp_path):
    if request.param == "postgres":
        return request.param, tmp_path, request.getfixturevalue("postgres_dsn")
    return request.param, tmp_path, None


async def _open_store(case, filename: str):
    kind, tmp_path, postgres_dsn = case
    if kind == "memory":
        return InMemorySessionStore()
    if kind == "sqlite":
        return SQLiteSessionStore(tmp_path / filename)
    await drop_cayu_tables(postgres_dsn)
    return PostgresSessionStore(
        postgres_dsn,
        min_size=1,
        max_size=4,
        schema_mode=SchemaMode.CREATE,
    )


class _RepeatableProvider(ModelProvider):
    name = "capture-provider"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.text_delta("captured answer")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _BlockingTwoTurnProvider(ModelProvider):
    name = "capture-queue-provider"

    def __init__(self) -> None:
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.request_count = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.request_count += 1
        if self.request_count == 1:
            self.first_started.set()
            await self.release_first.wait()
            output = "first answer"
        else:
            output = "queued answer"
        yield ModelStreamEvent.text_delta(output)
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _ApprovalProvider(ModelProvider):
    name = "capture-approval-provider"

    def __init__(self) -> None:
        self.request_count = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.request_count += 1
        if self.request_count == 1:
            yield ModelStreamEvent.tool_call(
                id="call-approval",
                name="review_action",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("approved answer")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _UserInputProvider(ModelProvider):
    name = "capture-user-input-provider"

    def __init__(self) -> None:
        self.request_count = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.request_count += 1
        if self.request_count == 1:
            yield ModelStreamEvent.tool_call(
                id="call-user-input",
                name="ask_user",
                arguments={"question": "Which environment?"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("Deploying to production.")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _ReviewTool(Tool):
    spec = ToolSpec(
        name="review_action",
        description="Perform one reviewed action.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(content="reviewed")


def _app(store, provider: ModelProvider, *, secret_redactor=None) -> CayuApp:
    app = CayuApp(
        session_store=store,
        secret_redactor=secret_redactor,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="capture-model"))
    return app


async def _capture(app: CayuApp, session_id: str):
    return await capture_eval_scenario_from_session(
        app,
        session_id,
        target_key="assistant.default",
        source_agent_name="assistant",
        source=_source(),
        name="Retained production behavior",
    )


def test_capture_reconstructs_initial_and_resumed_input_with_durable_proof(
    scenario_capture_store_case,
) -> None:
    async def scenario():
        store = await _open_store(
            scenario_capture_store_case,
            "scenario-capture.sqlite",
        )
        app = _app(store, _RepeatableProvider())
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="capture-resume",
                messages=[Message.text("user", "initial request")],
            )
        ):
            pass
        async for _ in app.resume(
            ResumeRequest(
                session_id="capture-resume",
                messages=[Message.text("user", "follow-up request")],
            )
        ):
            pass
        evidence = await store.load_terminal_session_evidence("capture-resume")
        captured = await _capture(app, "capture-resume")
        trajectory = await trajectory_from_session(app, "capture-resume")
        close = getattr(store, "close", None)
        if close is not None:
            await close()
        return evidence, captured, trajectory

    evidence, captured, trajectory = asyncio.run(scenario())
    resumed = next(
        record.event for record in evidence.events if record.event.type == EventType.SESSION_RESUMED
    )
    marker = resumed.payload[SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY]
    assert event_payload_authority_is_runtime_generated(
        resumed,
        field_name=SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
        value=marker,
    )
    assert captured.available is True
    assert captured.diagnostics == ()
    assert captured.scenario is not None
    assert [event.kind for event in captured.scenario.events] == ["initial", "resumed"]
    assert [
        event.input.messages[0].content[0].text  # type: ignore[union-attr]
        for event in captured.scenario.events
    ] == ["initial request", "follow-up request"]
    assert all(
        SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY not in event.payload
        for event in trajectory.events
    )


def test_capture_reconstructs_queued_input_without_copying_queue_authority(
    scenario_capture_store_case,
) -> None:
    async def scenario():
        store = await _open_store(
            scenario_capture_store_case,
            "scenario-queue.sqlite",
        )
        provider = _BlockingTwoTurnProvider()
        app = _app(store, provider)

        async def execute() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="capture-queue",
                    messages=[Message.text("user", "initial request")],
                )
            ):
                pass

        task = asyncio.create_task(execute())
        await provider.first_started.wait()
        accepted = await app.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="capture-queue",
                idempotency_key="queued-follow-up",
                content="queued request",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
        )
        provider.release_first.set()
        await task
        evidence = await store.load_terminal_session_evidence("capture-queue")
        captured = await _capture(app, "capture-queue")
        close = getattr(store, "close", None)
        if close is not None:
            await close()
        return accepted.message.queue_id, evidence, captured

    queue_id, evidence, captured = asyncio.run(scenario())
    delivered = next(
        record.event
        for record in evidence.events
        if record.event.type == EventType.SESSION_MESSAGE_DELIVERED
    )
    marker = delivered.payload[SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY]
    assert event_payload_authority_is_runtime_generated(
        delivered,
        field_name=SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
        value=marker,
    )
    assert captured.available is True
    assert captured.scenario is not None
    assert [event.kind for event in captured.scenario.events] == ["initial", "queued"]
    queued = captured.scenario.events[1]
    assert queued.delivery_mode == "next_turn"  # type: ignore[union-attr]
    assert queued.input.messages[0].content[0].text == "queued request"  # type: ignore[union-attr]
    encoded = captured.scenario.model_dump_json()
    assert queue_id not in encoded
    assert SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY not in encoded


def test_capture_rejects_redacted_queued_input_without_exposing_private_proof() -> None:
    async def scenario():
        provider = _BlockingTwoTurnProvider()
        app = _app(
            InMemorySessionStore(),
            provider,
            secret_redactor=SecretRedactor("production-secret-value"),
        )

        async def execute() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="capture-redacted-queue",
                    messages=[Message.text("user", "initial request")],
                )
            ):
                pass

        task = asyncio.create_task(execute())
        await provider.first_started.wait()
        accepted = await app.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="capture-redacted-queue",
                idempotency_key="redacted-follow-up",
                content="use production-secret-value",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
        )
        provider.release_first.set()
        await task
        evidence = await app.session_store.load_terminal_session_evidence("capture-redacted-queue")
        return accepted, evidence, await _capture(app, "capture-redacted-queue")

    accepted, evidence, captured = asyncio.run(scenario())
    assert SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY not in accepted.event.payload
    queued = next(
        record.event
        for record in evidence.events
        if record.event.type == EventType.SESSION_MESSAGE_QUEUED
    )
    contract = parse_session_input_contract_evidence(
        queued.payload[SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY]
    )
    assert contract.redactions_applied is True
    assert captured.available is False
    assert captured.scenario is None
    assert ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_REDACTED in {
        diagnostic.code for diagnostic in captured.diagnostics
    }


def test_capture_projects_approval_as_a_fresh_checkpoint() -> None:
    async def scenario():
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(_ApprovalProvider(), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="capture-model"),
            tools=[_ReviewTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )
        initial = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="capture-approval",
                    messages=[Message.text("user", "review this")],
                )
            )
        ]
        requested = next(
            event for event in initial if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval = requested.payload["approval"]
        async for _ in app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id="capture-approval",
                approval_id=approval["approval_id"],
                tool_round_id=approval["tool_round_id"],
                tool_call_id=approval["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
                resolved_by=ResolutionActor(
                    subject="reviewer",
                    source=ResolutionActorSource.REQUEST,
                ),
            )
        ):
            pass
        return approval["approval_id"], await _capture(app, "capture-approval")

    approval_id, captured = asyncio.run(scenario())
    assert captured.available is True
    assert captured.scenario is not None
    assert [event.kind for event in captured.scenario.events] == [
        "initial",
        "approval_checkpoint",
    ]
    checkpoint = captured.scenario.events[1]
    assert checkpoint.tool_name == "review_action"  # type: ignore[union-attr]
    assert checkpoint.occurrence == 1  # type: ignore[union-attr]
    assert checkpoint.resolution == "fresh_decision"  # type: ignore[union-attr]
    assert approval_id not in captured.scenario.model_dump_json()


def test_capture_fails_closed_for_unrepresentable_user_input_continuation() -> None:
    async def scenario():
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(_UserInputProvider(), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="capture-model"),
            tools=[UserInputTool()],
        )
        paused = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="capture-user-input",
                    messages=[Message.text("user", "deploy")],
                )
            )
        ]
        awaiting = next(
            event for event in paused if event.type == EventType.SESSION_AWAITING_USER_INPUT
        )
        async for _ in app.resolve_user_input(
            UserInputResponse(
                session_id="capture-user-input",
                input_id=awaiting.payload["input_id"],
                answer="production",
            )
        ):
            pass
        return await _capture(app, "capture-user-input")

    captured = asyncio.run(scenario())
    assert captured.available is False
    assert captured.scenario is None
    assert [diagnostic.code for diagnostic in captured.diagnostics] == [
        ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_UNAVAILABLE
    ]
    assert captured.diagnostics[0].event_sequence is not None


def test_capture_attributes_resume_after_on_idle_queue_to_its_own_interaction() -> None:
    async def scenario():
        provider = _BlockingTwoTurnProvider()
        app = _app(InMemorySessionStore(), provider)

        async def execute() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="capture-queue-then-resume",
                    messages=[Message.text("user", "initial request")],
                )
            ):
                pass

        task = asyncio.create_task(execute())
        await provider.first_started.wait()
        await app.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="capture-queue-then-resume",
                idempotency_key="queued-on-idle",
                content="queued request",
                delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
            )
        )
        provider.release_first.set()
        await task
        async for _ in app.resume(
            ResumeRequest(
                session_id="capture-queue-then-resume",
                messages=[Message.text("user", "ordinary resume")],
            )
        ):
            pass
        return await _capture(app, "capture-queue-then-resume")

    captured = asyncio.run(scenario())
    assert captured.available is True
    assert captured.scenario is not None
    assert [event.kind for event in captured.scenario.events] == [
        "initial",
        "queued",
        "resumed",
    ]
    resumed = captured.scenario.events[2]
    assert resumed.input.messages[0].content[0].text == "ordinary resume"  # type: ignore[union-attr]


def test_capture_binds_file_input_to_retained_content_digest(
    tmp_path,
    scenario_capture_store_case,
) -> None:
    async def scenario():
        content = b"%PDF-1.4 retained invoice"
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="capture-files")
        metadata = await artifact_store.put_bytes(
            content,
            filename="invoice.pdf",
            content_type="application/pdf",
            scope=ArtifactScope.SESSION,
            session_id="capture-file",
            environment_name="files",
        )
        store = await _open_store(
            scenario_capture_store_case,
            "scenario-capture-files.sqlite",
        )
        app = _app(store, _RepeatableProvider())
        app.register_environment(
            Environment(
                EnvironmentSpec(name="files"),
                artifact_store=artifact_store,
            ),
            default=True,
        )
        attachment = file_attachment(
            artifact_id=metadata.id,
            kind="document",
            filename=metadata.filename,
            content_type=metadata.content_type,
            size_bytes=metadata.size_bytes,
        )
        public_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="capture-file",
                    messages=[
                        Message(
                            role="user",
                            content=(
                                TextPart(text="Read this invoice."),
                                FilePart(attachment=attachment),
                            ),
                        )
                    ],
                )
            )
        ]
        evidence = await store.load_terminal_session_evidence("capture-file")
        captured = await _capture(app, "capture-file")
        await artifact_store.delete(metadata.id)
        missing = await _capture(app, "capture-file")
        close = getattr(store, "close", None)
        if close is not None:
            await close()
        return content, metadata.id, public_events, evidence, captured, missing

    content, artifact_id, public_events, evidence, captured, missing = asyncio.run(scenario())
    private_started = next(
        record.event for record in evidence.events if record.event.type == EventType.MODEL_STARTED
    )
    marker = private_started.payload[MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY]
    assert event_payload_authority_is_runtime_generated(
        private_started,
        field_name=MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY,
        value=marker,
    )
    assert all(
        MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY not in event.payload
        for event in public_events
    )
    assert captured.available is True
    assert captured.scenario is not None
    assert len(captured.scenario.artifact_requirements) == 1
    requirement = captured.scenario.artifact_requirements[0]
    assert requirement.reference == artifact_id
    assert requirement.content_sha256 == hashlib.sha256(content).hexdigest()
    assert requirement.size_bytes == len(content)
    assert content.decode() not in captured.scenario.model_dump_json()
    assert missing.available is False
    assert missing.scenario is None
    assert [diagnostic.code for diagnostic in missing.diagnostics] == [
        ScenarioCaptureDiagnosticCode.ARTIFACT_NOT_RETAINED
    ]


def test_capture_rejects_reused_artifact_id_with_replacement_bytes(tmp_path) -> None:
    async def scenario():
        original = b"%PDF-1.4 original!"
        replacement = b"%PDF-1.4 replaced!"
        artifact_store = LocalArtifactStore(
            tmp_path / "reused-artifacts",
            store_id="capture-reused-files",
        )
        metadata = await artifact_store.put_bytes(
            original,
            artifact_id="art_11111111111111111111111111111111",
            filename="invoice.pdf",
            content_type="application/pdf",
            scope=ArtifactScope.SESSION,
            session_id="capture-reused-file",
            environment_name="files",
        )
        app = _app(InMemorySessionStore(), _RepeatableProvider())
        app.register_environment(
            Environment(EnvironmentSpec(name="files"), artifact_store=artifact_store),
            default=True,
        )
        attachment = file_attachment(
            artifact_id=metadata.id,
            kind="document",
            filename=metadata.filename,
            content_type=metadata.content_type,
            size_bytes=metadata.size_bytes,
        )
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="capture-reused-file",
                messages=[Message(role="user", content=(FilePart(attachment=attachment),))],
            )
        ):
            pass
        await artifact_store.delete(metadata.id)
        await artifact_store.put_bytes(
            replacement,
            artifact_id=metadata.id,
            filename=metadata.filename,
            content_type=metadata.content_type,
            scope=metadata.scope,
            session_id=metadata.session_id,
            environment_name=metadata.environment_name,
        )
        return await _capture(app, "capture-reused-file")

    captured = asyncio.run(scenario())
    assert captured.available is False
    assert captured.scenario is None
    assert [diagnostic.code for diagnostic in captured.diagnostics] == [
        ScenarioCaptureDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT
    ]


def test_capture_contains_malformed_artifact_store_result_as_a_diagnostic(tmp_path) -> None:
    class MalformedAfterRunStore(LocalArtifactStore):
        malformed = False

        async def read_bytes(
            self,
            artifact_id: str,
            *,
            max_bytes: int | None = None,
        ) -> ArtifactReadResult:
            if self.malformed:
                return object()  # type: ignore[return-value]
            return await super().read_bytes(artifact_id, max_bytes=max_bytes)

    async def scenario():
        artifact_store = MalformedAfterRunStore(
            tmp_path / "malformed-artifacts",
            store_id="capture-malformed-files",
        )
        metadata = await artifact_store.put_bytes(
            b"%PDF-1.4 retained invoice",
            filename="invoice.pdf",
            content_type="application/pdf",
            scope=ArtifactScope.SESSION,
            session_id="capture-malformed-file",
            environment_name="files",
        )
        app = _app(InMemorySessionStore(), _RepeatableProvider())
        app.register_environment(
            Environment(EnvironmentSpec(name="files"), artifact_store=artifact_store),
            default=True,
        )
        attachment = file_attachment(
            artifact_id=metadata.id,
            kind="document",
            filename=metadata.filename,
            content_type=metadata.content_type,
            size_bytes=metadata.size_bytes,
        )
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="capture-malformed-file",
                messages=[Message(role="user", content=(FilePart(attachment=attachment),))],
            )
        ):
            pass
        artifact_store.malformed = True
        return await _capture(app, "capture-malformed-file")

    captured = asyncio.run(scenario())
    assert captured.available is False
    assert captured.scenario is None
    assert [diagnostic.code for diagnostic in captured.diagnostics] == [
        ScenarioCaptureDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT
    ]


def test_capture_rejects_file_metadata_that_scenario_v2_cannot_represent(tmp_path) -> None:
    async def scenario():
        artifact_store = LocalArtifactStore(
            tmp_path / "metadata-artifacts",
            store_id="capture-file-metadata",
        )
        metadata = await artifact_store.put_bytes(
            b"%PDF-1.4 retained invoice",
            filename="invoice.pdf",
            content_type="application/pdf",
            scope=ArtifactScope.SESSION,
            session_id="capture-file-metadata",
            environment_name="files",
        )
        app = _app(InMemorySessionStore(), _RepeatableProvider())
        app.register_environment(
            Environment(EnvironmentSpec(name="files"), artifact_store=artifact_store),
            default=True,
        )
        attachment = file_attachment(
            artifact_id=metadata.id,
            kind="document",
            filename=metadata.filename,
            content_type=metadata.content_type,
            size_bytes=metadata.size_bytes,
            metadata={"page": 2},
        )
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="capture-file-metadata",
                messages=[
                    Message(
                        role="user",
                        content=(FilePart(attachment=attachment),),
                    )
                ],
            )
        ):
            pass
        return await _capture(app, "capture-file-metadata")

    captured = asyncio.run(scenario())
    assert captured.available is False
    assert captured.scenario is None
    assert [diagnostic.code for diagnostic in captured.diagnostics] == [
        ScenarioCaptureDiagnosticCode.SOURCE_INPUT_PART_UNSUPPORTED
    ]


def test_capture_rejects_oversized_artifact_set_before_reading_content(tmp_path) -> None:
    class CountingArtifactStore(LocalArtifactStore):
        def __init__(self) -> None:
            super().__init__(tmp_path / "oversized-artifacts", store_id="capture-limits")
            self.read_count = 0

        async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
            self.read_count += 1
            return await super().read_bytes(artifact_id, max_bytes=max_bytes)

    async def scenario():
        store = CountingArtifactStore()
        app = _app(InMemorySessionStore(), _RepeatableProvider())
        app.register_environment(
            Environment(EnvironmentSpec(name="files"), artifact_store=store),
            default=True,
        )
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="capture-artifact-limit",
                messages=[Message.text("user", "initial request")],
            )
        ):
            pass
        evidence = await app.session_store.load_terminal_session_evidence("capture-artifact-limit")
        attachment = FileAttachment(
            artifact_id="artifact-too-large",
            kind="document",
            filename="oversized.pdf",
            content_type="application/pdf",
            size_bytes=EVAL_SCENARIO_MAX_TOTAL_ARTIFACT_BYTES + 1,
        )
        _, diagnostics = await _resolve_artifact_requirements(
            app,
            evidence,
            {attachment.artifact_id: attachment},
        )
        return store.read_count, diagnostics

    read_count, diagnostics = asyncio.run(scenario())
    assert read_count == 0
    assert [diagnostic.code for diagnostic in diagnostics] == [
        ScenarioCaptureDiagnosticCode.SCENARIO_LIMIT_EXCEEDED
    ]


def test_capture_reports_redacted_source_payload_instead_of_replaying_placeholder() -> None:
    async def scenario():
        app = _app(
            InMemorySessionStore(),
            _RepeatableProvider(),
            secret_redactor=SecretRedactor("production-secret-value"),
        )
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="capture-redacted",
                messages=[Message.text("user", "use production-secret-value")],
            )
        ):
            pass
        transcript = await app.session_store.load_transcript("capture-redacted")
        return transcript, await _capture(app, "capture-redacted")

    transcript, captured = asyncio.run(scenario())
    assert REDACTED_SECRET in transcript[0].content[0].text  # type: ignore[union-attr]
    assert captured.available is False
    assert captured.scenario is None
    assert ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_REDACTED in {
        diagnostic.code for diagnostic in captured.diagnostics
    }


def test_capture_tracks_redaction_at_a_resumed_input_boundary() -> None:
    async def scenario():
        app = _app(
            InMemorySessionStore(),
            _RepeatableProvider(),
            secret_redactor=SecretRedactor("production-secret-value"),
        )
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="capture-redacted-resume",
                messages=[Message.text("user", "initial request")],
            )
        ):
            pass
        async for _ in app.resume(
            ResumeRequest(
                session_id="capture-redacted-resume",
                messages=[Message.text("user", "use production-secret-value")],
            )
        ):
            pass
        evidence = await app.session_store.load_terminal_session_evidence("capture-redacted-resume")
        return evidence, await _capture(app, "capture-redacted-resume")

    evidence, captured = asyncio.run(scenario())
    resumed = next(
        record.event for record in evidence.events if record.event.type == EventType.SESSION_RESUMED
    )
    contract = parse_session_input_contract_evidence(
        resumed.payload[SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY]
    )
    assert contract.redactions_applied is True
    assert captured.available is False
    assert ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_REDACTED in {
        diagnostic.code for diagnostic in captured.diagnostics
    }


def test_capture_identity_is_stable_and_distinguishes_external_stimuli() -> None:
    async def scenario():
        app = _app(InMemorySessionStore(), _RepeatableProvider())
        for session_id, text in (
            ("capture-identity-a", "first stimulus"),
            ("capture-identity-b", "second stimulus"),
        ):
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", text)],
                )
            ):
                pass
        first = await _capture(app, "capture-identity-a")
        repeated = await _capture(app, "capture-identity-a")
        second = await _capture(app, "capture-identity-b")
        return first, repeated, second

    first, repeated, second = asyncio.run(scenario())
    assert first.available is True
    assert repeated.available is True
    assert second.available is True
    assert first.scenario is not None
    assert repeated.scenario is not None
    assert second.scenario is not None
    assert first.scenario.id == repeated.scenario.id
    assert first.scenario.id != second.scenario.id
