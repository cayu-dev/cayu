from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu.artifacts import file_attachment
from cayu.core import (
    AgentSpec,
    EventType,
    ExecutionProfileBehaviorIdentity,
    FilePart,
    Message,
    MessageRole,
    ProviderStatePart,
    ToolCallPart,
    ToolResultPart,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    CompactSessionRequest,
    ContextCompactor,
    ContextRequest,
    EventQuery,
    InMemoryEventSink,
    InMemorySessionStore,
    ModelCompactor,
    RunRequest,
    SessionIdentity,
    SessionStatus,
    TranscriptSnapshot,
)
from cayu.runtime.context import ContextBuildResult
from cayu.vaults import REDACTED_SECRET, SecretRedactor


def _serialized(value: Any) -> str:
    def jsonable(candidate: Any) -> Any:
        if hasattr(candidate, "model_dump"):
            return jsonable(candidate.model_dump(mode="json"))
        if type(candidate) is dict:
            return {key: jsonable(item) for key, item in candidate.items()}
        if isinstance(candidate, (list, tuple)):
            return [jsonable(item) for item in candidate]
        return candidate

    return json.dumps(jsonable(value), sort_keys=True)


def _serialized_messages(messages: list[Message]) -> str:
    return _serialized([message.model_dump(mode="json") for message in messages])


def _assert_cayu_traceback_has_no_secret(error: BaseException, secret: str) -> None:
    traceback = error.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            retained = {
                name: type(value).__name__
                for name, value in traceback.tb_frame.f_locals.items()
                if secret in repr(value)
            }
            assert retained == {}, (
                traceback.tb_frame.f_code.co_filename,
                traceback.tb_frame.f_code.co_name,
                retained,
            )
        traceback = traceback.tb_next


class _CapturingCompactor(ContextCompactor):
    def __init__(self, *, fail: bool = False) -> None:
        self.requests: list[CompactionRequest] = []
        self.fail = fail

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:explicit-compaction-capturing-compactor",
            behavior_version="1",
            implementation_version="1",
        )

    def provider_budget_identity(self, _session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.requests.append(request.model_copy(deep=True))
        if self.fail:
            raise RuntimeError("safe compactor failure")
        return CompactionResult(
            summary="safe compacted summary",
            covered_message_count=len(request.messages),
            represented_existing_summary_sha256=(
                hashlib.sha256(request.existing_summary.encode("utf-8")).hexdigest()
                if request.existing_summary is not None
                else None
            ),
        )


class _CapturingPolicy(CheckpointCompactionContextPolicy):
    def __init__(self, *, compactor: ContextCompactor) -> None:
        super().__init__(compactor=compactor, max_user_turns=1)
        self.requests: list[ContextRequest] = []
        self.checkpoints: list[dict[str, Any] | None] = []

    async def build_with_checkpoint(
        self,
        request: ContextRequest,
        *,
        checkpoint: dict[str, Any] | None,
    ) -> ContextBuildResult:
        self.requests.append(request.model_copy(deep=True))
        self.checkpoints.append(None if checkpoint is None else json.loads(json.dumps(checkpoint)))
        return await super().build_with_checkpoint(request, checkpoint=checkpoint)


class _CapturingProvider(ModelProvider):
    name = "explicit-compaction-capture"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request.model_copy(deep=True))
        yield ModelStreamEvent.text_delta("safe provider summary")
        yield ModelStreamEvent.completed({})


class _IdentityProbeCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.identity_calls = 0
        self.compact_calls = 0

    def _uses_runtime_provider_dispatch_runner_for_forced_compaction(self) -> bool:
        return True

    def provider_budget_identity(self, _session) -> tuple[str, str]:
        self.identity_calls += 1
        return "probe-provider", "probe-model"

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.compact_calls += 1
        return CompactionResult(
            summary="unreachable",
            covered_message_count=len(request.messages),
        )


class _BlockingCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    def provider_budget_identity(self, _session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        del request
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def _create_completed_session(
    store: InMemorySessionStore,
    *,
    session_id: str,
    transcript: list[Message],
):
    session = await store.create(
        RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    await store.append_transcript_messages(session.id, transcript)
    return await store.update_status(session.id, SessionStatus.COMPLETED)


def _legacy_compaction_checkpoint(summary: str) -> dict[str, Any]:
    return {
        "context_compaction": {
            "version": 2,
            "summary": summary,
            "compacted_transcript_cursor": 2,
            "metadata": {"compactor": "legacy", "mode": "deterministic"},
        }
    }


def _runtime_tool_result_artifact_reference(
    *,
    artifact_id: str = f"art_{'a' * 32}",
    store_id: str = "artifacts",
) -> dict[str, Any]:
    return {
        "type": "cayu.tool_result_artifact.v1",
        "artifact_id": artifact_id,
        "store_id": store_id,
        "filename": f"tool-result-{artifact_id}.txt",
        "content_type": "text/plain; charset=utf-8",
        "size_bytes": 1,
        "sha256": "b" * 64,
        "scope": "session",
        "session_id_sha256": "c" * 64,
        "readback_max_bytes": 64,
        "projection_authority": "cayu.tool_result_projection.v1",
    }


def test_explicit_model_compaction_projects_legacy_summary_before_provider() -> None:
    async def scenario() -> None:
        secret = "explicit-legacy-summary-provider-secret"
        legacy_summary = f"legacy summary containing {secret}"
        transcript = [
            Message.text("user", f"old request containing {secret}"),
            Message.text("assistant", "old answer"),
            Message.text("user", "newer request"),
            Message.text("assistant", "newer answer"),
            Message.text("user", "current request"),
        ]
        store = InMemorySessionStore()
        provider = _CapturingProvider()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        completed = await _create_completed_session(
            store,
            session_id="sess_explicit_legacy_summary_provider",
            transcript=transcript,
        )
        await store.checkpoint(completed.id, _legacy_compaction_checkpoint(legacy_summary))

        async for _event in app.compact_session(
            CompactSessionRequest(
                session_id=completed.id,
                idempotency_key="project-legacy-summary-provider",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
        ):
            pass

        assert len(provider.requests) == 1
        provider_payload = _serialized(provider.requests)
        assert secret not in provider_payload
        assert f"legacy summary containing {REDACTED_SECRET}" in provider_payload
        checkpoint = await store.load_checkpoint(completed.id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"]["summary"] == "safe provider summary"
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 4

    asyncio.run(scenario())


def test_explicit_custom_compactor_binds_projected_legacy_summary() -> None:
    async def scenario() -> None:
        secret = "explicit-legacy-summary-custom-secret"
        legacy_summary = f"legacy summary containing {secret}"
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "newer request"),
            Message.text("assistant", "newer answer"),
            Message.text("user", "current request"),
        ]
        store = InMemorySessionStore()
        compactor = _CapturingCompactor()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        completed = await _create_completed_session(
            store,
            session_id="sess_explicit_legacy_summary_custom",
            transcript=transcript,
        )
        await store.checkpoint(completed.id, _legacy_compaction_checkpoint(legacy_summary))

        async for _event in app.compact_session(
            CompactSessionRequest(
                session_id=completed.id,
                idempotency_key="project-legacy-summary-custom",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
        ):
            pass

        assert len(compactor.requests) == 1
        assert compactor.requests[0].existing_summary == (
            f"legacy summary containing {REDACTED_SECRET}"
        )
        assert secret not in _serialized(compactor.requests)
        checkpoint = await store.load_checkpoint(completed.id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"]["summary"] == "safe compacted summary"
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 4

    asyncio.run(scenario())


def test_explicit_compaction_rejects_declared_opaque_provider_before_dispatch() -> None:
    class OpaqueProviderCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider
            self.compact_calls = 0

        def provider_budget_identity(self, _session) -> tuple[str, str]:
            return "explicit-compaction-capture", "summary-model"

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.compact_calls += 1
            async for _event in self.provider.stream(
                ModelRequest(model="summary-model", messages=request.messages)
            ):
                pass
            return CompactionResult(
                summary="opaque summary",
                covered_message_count=len(request.messages),
            )

    async def scenario() -> None:
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        store = InMemorySessionStore()
        provider = _CapturingProvider()
        compactor = OpaqueProviderCompactor(provider)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        completed = await _create_completed_session(
            store,
            session_id="sess_explicit_opaque_compactor_footprint",
            transcript=transcript,
        )
        before_events = await store.query_events(EventQuery(session_id=completed.id, limit=100))

        with pytest.raises(
            RuntimeError,
            match="cannot observe each provider dispatch independently",
        ):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=completed.id,
                    idempotency_key="reject-opaque-provider",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert compactor.compact_calls == 0
        assert provider.requests == []
        assert (
            await store.query_events(EventQuery(session_id=completed.id, limit=100))
            == before_events
        )
        assert await store.load_session_operation(completed.id, "reject-opaque-provider") is None

    asyncio.run(scenario())


def test_explicit_compaction_rejects_missing_provider_identity_before_dispatch() -> None:
    class UndeclaredProviderCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider
            self.compact_calls = 0

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.compact_calls += 1
            async for _event in self.provider.stream(
                ModelRequest(model="summary-model", messages=request.messages)
            ):
                pass
            return CompactionResult(
                summary="opaque summary",
                covered_message_count=len(request.messages),
            )

    async def scenario() -> None:
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        store = InMemorySessionStore()
        provider = _CapturingProvider()
        compactor = UndeclaredProviderCompactor(provider)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        completed = await _create_completed_session(
            store,
            session_id="sess_explicit_undeclared_compactor_footprint",
            transcript=transcript,
        )
        before_events = await store.query_events(EventQuery(session_id=completed.id, limit=100))

        with pytest.raises(
            RuntimeError,
            match="explicitly declare provider_budget_identity",
        ):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=completed.id,
                    idempotency_key="reject-undeclared-provider",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert compactor.compact_calls == 0
        assert provider.requests == []
        assert (
            await store.query_events(EventQuery(session_id=completed.id, limit=100))
            == before_events
        )
        assert (
            await store.load_session_operation(completed.id, "reject-undeclared-provider") is None
        )

    asyncio.run(scenario())


def test_explicit_compaction_failure_does_not_rewrite_legacy_summary() -> None:
    async def scenario() -> None:
        secret = "explicit-legacy-summary-failure-secret"
        legacy_summary = f"legacy summary containing {secret}"
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "newer request"),
            Message.text("assistant", "newer answer"),
            Message.text("user", "current request"),
        ]
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        compactor = _CapturingCompactor(fail=True)
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        completed = await _create_completed_session(
            store,
            session_id="sess_explicit_legacy_summary_failure",
            transcript=transcript,
        )
        legacy_checkpoint = _legacy_compaction_checkpoint(legacy_summary)
        await store.checkpoint(completed.id, legacy_checkpoint)

        with pytest.raises(RuntimeError, match="safe compactor failure") as exc_info:
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=completed.id,
                    idempotency_key="project-legacy-summary-failure",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        _assert_cayu_traceback_has_no_secret(exc_info.value, secret)
        assert len(compactor.requests) == 1
        assert secret not in _serialized(compactor.requests)
        checkpoint = await store.load_checkpoint(completed.id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"] == legacy_checkpoint["context_compaction"]
        events = await store.query_events(EventQuery(session_id=completed.id, limit=100))
        assert secret not in _serialized(
            {
                "durable": [record.event.model_dump(mode="json") for record in events],
                "sink": [event.model_dump(mode="json") for event in sink.events],
            }
        )

    asyncio.run(scenario())


def test_explicit_compaction_cancellation_releases_raw_legacy_summary() -> None:
    async def scenario() -> None:
        secret = "explicit-legacy-summary-cancellation-secret"
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "newer request"),
            Message.text("assistant", "newer answer"),
            Message.text("user", "current request"),
        ]
        store = InMemorySessionStore()
        compactor = _BlockingCompactor()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        completed = await _create_completed_session(
            store,
            session_id="sess_explicit_legacy_summary_cancellation",
            transcript=transcript,
        )
        await store.checkpoint(
            completed.id,
            _legacy_compaction_checkpoint(f"legacy summary containing {secret}"),
        )

        async def collect() -> list:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=completed.id,
                        idempotency_key="project-legacy-summary-cancellation",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await compactor.started.wait()
        task.cancel("cancel explicit compaction")
        with pytest.raises(asyncio.CancelledError, match="cancel explicit compaction") as exc_info:
            await task

        _assert_cayu_traceback_has_no_secret(exc_info.value, secret)
        assert task.cancelled()

    asyncio.run(scenario())


def test_explicit_custom_policy_receives_projected_version_one_summary() -> None:
    async def scenario() -> None:
        secret = "explicit-version-one-summary-secret"
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        store = InMemorySessionStore()
        compactor = _CapturingCompactor()
        policy = _CapturingPolicy(compactor=compactor)
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=policy,
        )
        completed = await _create_completed_session(
            store,
            session_id="sess_explicit_version_one_summary",
            transcript=transcript,
        )
        await store.checkpoint(
            completed.id,
            {
                "context_compaction": {
                    "version": 1,
                    "summary": f"legacy summary containing {secret}",
                    "compacted_transcript_cursor": 2,
                    "metadata": {"compactor": "legacy"},
                }
            },
        )

        async for _event in app.compact_session(
            CompactSessionRequest(
                session_id=completed.id,
                idempotency_key="project-version-one-summary",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
        ):
            pass

        assert len(policy.checkpoints) == 1
        assert policy.checkpoints[0] is not None
        assert policy.checkpoints[0]["context_compaction"]["summary"] == (
            f"legacy summary containing {REDACTED_SECRET}"
        )
        assert secret not in _serialized(policy.checkpoints)
        assert len(compactor.requests) == 1
        assert compactor.requests[0].existing_summary is None

    asyncio.run(scenario())


def test_explicit_model_compaction_projects_legacy_transcript_before_provider() -> None:
    async def scenario() -> None:
        secret = "explicit-model-compaction-secret-canary"
        transcript = [
            Message.text("user", f"old request containing {secret}"),
            Message.text("assistant", secret),
            Message.text("user", "current request"),
        ]
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        provider = _CapturingProvider()
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        completed = await _create_completed_session(
            store,
            session_id="sess_explicit_model_secret_projection",
            transcript=transcript,
        )

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=completed.id,
                    idempotency_key="project-model-transcript",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]

        assert len(provider.requests) == 1
        provider_payload = _serialized(provider.requests)
        assert secret not in provider_payload
        assert REDACTED_SECRET in provider_payload
        assert await store.load_transcript(completed.id) == transcript
        checkpoint = await store.load_checkpoint(completed.id)
        assert checkpoint is not None
        durable_records = await store.query_events(EventQuery(session_id=completed.id, limit=100))
        public_payload = _serialized(
            {
                "returned": [event.model_dump(mode="json") for event in events],
                "durable": [record.event.model_dump(mode="json") for record in durable_records],
                "sink": [event.model_dump(mode="json") for event in sink.events],
                "checkpoint": checkpoint,
            }
        )
        assert secret not in public_payload

    asyncio.run(scenario())


def test_explicit_custom_policy_and_compactor_receive_only_projected_messages() -> None:
    async def scenario() -> None:
        secrets = [
            "explicit-first-position-secret",
            "explicit-middle-position-secret",
            "explicit-final-position-secret",
        ]
        transcript = [
            Message.text("user", f"first {secrets[0]}"),
            Message.text("assistant", f"middle {secrets[1]}"),
            Message.text("user", "current request"),
            Message.text("assistant", f"final {secrets[2]}"),
        ]
        store = InMemorySessionStore()
        compactor = _CapturingCompactor()
        policy = _CapturingPolicy(compactor=compactor)
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secrets),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=policy,
        )
        completed = await _create_completed_session(
            store,
            session_id="sess_explicit_custom_secret_projection",
            transcript=transcript,
        )

        async for _event in app.compact_session(
            CompactSessionRequest(
                session_id=completed.id,
                idempotency_key="project-custom-transcript",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
        ):
            pass

        assert len(policy.requests) == 1
        policy_messages = _serialized_messages(policy.requests[0].messages)
        assert all(secret not in policy_messages for secret in secrets)
        assert policy_messages.count(REDACTED_SECRET) == 3
        assert len(compactor.requests) == 1
        compactor_messages = _serialized_messages(compactor.requests[0].messages)
        assert secrets[0] not in compactor_messages
        assert secrets[1] not in compactor_messages
        assert REDACTED_SECRET in compactor_messages
        assert await store.load_transcript(completed.id) == transcript

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "authority_field",
    [
        "tool_call_id",
        "provider",
        "attachment_artifact_id",
        "tool_result_attachment_artifact_id",
        "runtime_projection_artifact_id",
        "runtime_projection_store_id",
    ],
)
def test_explicit_compaction_rejects_secret_bearing_legacy_authority_before_claim(
    authority_field: str,
) -> None:
    async def scenario() -> None:
        secret = f"explicit-{authority_field}-authority-secret"
        if authority_field == "tool_call_id":
            part = ToolCallPart(
                tool_call_id=secret,
                tool_name="safe_tool",
                arguments={},
            )
        elif authority_field == "provider":
            part = ProviderStatePart(provider=secret, state={"safe": True})
        elif authority_field == "attachment_artifact_id":
            part = FilePart(
                attachment=file_attachment(
                    artifact_id=secret,
                    kind="image",
                    filename="safe.png",
                    content_type="image/png",
                    size_bytes=1,
                )
            )
        elif authority_field == "tool_result_attachment_artifact_id":
            part = ToolResultPart(
                tool_call_id="safe-call",
                tool_name="safe_tool",
                artifacts=[
                    file_attachment(
                        artifact_id=f"artifact-{secret}-suffix",
                        kind="image",
                        filename="safe.png",
                        content_type="image/png",
                        size_bytes=1,
                    )
                ],
            )
        elif authority_field == "runtime_projection_artifact_id":
            secret = "deadbeef"
            part = ToolResultPart(
                tool_call_id="safe-call",
                tool_name="safe_tool",
                artifacts=[
                    _runtime_tool_result_artifact_reference(
                        artifact_id=f"art_{secret}{'a' * 24}",
                    )
                ],
            )
        else:
            part = ToolResultPart(
                tool_call_id="safe-call",
                tool_name="safe_tool",
                artifacts=[
                    _runtime_tool_result_artifact_reference(
                        store_id=f"store-{secret}-suffix",
                    )
                ],
            )
        transcript = [
            Message(
                role=(
                    MessageRole.USER
                    if authority_field == "attachment_artifact_id"
                    else (
                        MessageRole.TOOL
                        if authority_field
                        in {
                            "tool_result_attachment_artifact_id",
                            "runtime_projection_artifact_id",
                            "runtime_projection_store_id",
                        }
                        else MessageRole.ASSISTANT
                    )
                ),
                content=(part,),
            ),
            Message.text("user", "current request"),
        ]
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        compactor = _IdentityProbeCompactor()
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        completed = await _create_completed_session(
            store,
            session_id=f"sess_explicit_{authority_field}_rejection",
            transcript=transcript,
        )
        before_events = await store.query_events(EventQuery(session_id=completed.id, limit=100))
        before_checkpoint = await store.load_checkpoint(completed.id)
        request = CompactSessionRequest(
            session_id=completed.id,
            idempotency_key=f"reject-{authority_field}",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        with pytest.raises(
            ValueError,
            match="workload secret in execution authority",
        ) as exc_info:
            async for _event in app.compact_session(request):
                pass

        _assert_cayu_traceback_has_no_secret(exc_info.value, secret)
        current = await store.load(completed.id)
        assert current is not None
        assert current.status == completed.status
        assert current.run_epoch == completed.run_epoch
        assert await store.load_transcript(completed.id) == transcript
        assert await store.load_checkpoint(completed.id) == before_checkpoint
        assert (
            await store.query_events(EventQuery(session_id=completed.id, limit=100))
            == before_events
        )
        assert await store.load_session_operation(completed.id, request.idempotency_key) is None
        assert sink.events == []
        assert compactor.identity_calls == 0
        assert compactor.compact_calls == 0

    asyncio.run(scenario())


def test_explicit_compaction_failure_does_not_retain_raw_transcript_secret() -> None:
    async def scenario() -> None:
        secret = "explicit-compactor-failure-secret-canary"
        transcript = [
            Message.text("user", f"old {secret}"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        compactor = _CapturingCompactor(fail=True)
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        completed = await _create_completed_session(
            store,
            session_id="sess_explicit_failure_projection",
            transcript=transcript,
        )

        with pytest.raises(RuntimeError, match="safe compactor failure") as exc_info:
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=completed.id,
                    idempotency_key="safe-projected-failure",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        _assert_cayu_traceback_has_no_secret(exc_info.value, secret)
        assert len(compactor.requests) == 1
        assert secret not in _serialized(compactor.requests)
        durable_records = await store.query_events(EventQuery(session_id=completed.id, limit=100))
        assert secret not in _serialized(
            {
                "durable": [record.event.model_dump(mode="json") for record in durable_records],
                "sink": [event.model_dump(mode="json") for event in sink.events],
                "checkpoint": await store.load_checkpoint(completed.id),
            }
        )
        assert await store.load_transcript(completed.id) == transcript

    asyncio.run(scenario())


def test_explicit_compaction_rechecks_cursor_after_projecting_snapshot() -> None:
    class SnapshotBarrierStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.snapshot_loaded = asyncio.Event()
            self.release_snapshot = asyncio.Event()

        async def load_transcript_snapshot(self, session_id: str) -> TranscriptSnapshot:
            snapshot = await super().load_transcript_snapshot(session_id)
            self.snapshot_loaded.set()
            await self.release_snapshot.wait()
            return snapshot

    async def scenario() -> None:
        secret = "explicit-snapshot-secret-canary"
        transcript = [
            Message.text("user", f"old {secret}"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        store = SnapshotBarrierStore()
        compactor = _CapturingCompactor()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        completed = await _create_completed_session(
            store,
            session_id="sess_explicit_snapshot_race",
            transcript=transcript,
        )
        request = CompactSessionRequest(
            session_id=completed.id,
            idempotency_key="snapshot-race",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        async def collect() -> list:
            return [event async for event in app.compact_session(request)]

        task = asyncio.create_task(collect())
        await store.snapshot_loaded.wait()
        tail = [Message.text("assistant", "concurrent append")]
        await store.append_transcript_messages(completed.id, tail)
        store.release_snapshot.set()
        with pytest.raises(ValueError, match="transcript cursor is stale"):
            await task

        assert compactor.requests == []
        assert await store.load_session_operation(completed.id, request.idempotency_key) is None
        assert await store.load_transcript(completed.id) == [*transcript, *tail]

    asyncio.run(scenario())


def test_explicit_compaction_replay_does_not_repeat_sanitized_work_after_lost_ack() -> None:
    class LostTerminalAcknowledgementStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def publish_session_operation_guarded(self, session_id: str, **kwargs):
            result = await super().publish_session_operation_guarded(session_id, **kwargs)
            if not self.failed and any(
                event.type == EventType.SESSION_CHECKPOINTED for event in kwargs.get("events", [])
            ):
                self.failed = True
                raise ConnectionError("terminal acknowledgement lost after commit")
            return result

    async def scenario() -> None:
        secret = "explicit-replay-secret-canary"
        transcript = [
            Message.text("user", f"old {secret}"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        store = LostTerminalAcknowledgementStore()
        compactor = _CapturingCompactor()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        completed = await _create_completed_session(
            store,
            session_id="sess_explicit_projected_replay",
            transcript=transcript,
        )
        request = CompactSessionRequest(
            session_id=completed.id,
            idempotency_key="projected-lost-ack",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        with pytest.raises(ConnectionError, match="terminal acknowledgement lost"):
            async for _event in app.compact_session(request):
                pass
        replay = [event async for event in app.compact_session(request)]

        assert replay
        assert len(compactor.requests) == 1
        assert secret not in _serialized(compactor.requests)
        assert REDACTED_SECRET in _serialized(compactor.requests)
        assert await store.load_transcript(completed.id) == transcript
        durable_records = await store.query_events(EventQuery(session_id=completed.id, limit=100))
        assert secret not in _serialized(
            [record.event.model_dump(mode="json") for record in durable_records]
        )

    asyncio.run(scenario())


def test_explicit_compaction_reprojects_transcript_when_reclaiming_after_restart() -> None:
    async def scenario() -> None:
        accepted_at = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
        secret = "explicit-restart-secret-canary"
        transcript = [
            Message.text("user", f"old {secret}"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        store = InMemorySessionStore()
        first_compactor = _CapturingCompactor()
        first_app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
            clock=lambda: accepted_at,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=first_compactor,
                max_user_turns=1,
            ),
        )
        completed = await _create_completed_session(
            store,
            session_id="sess_explicit_projected_restart",
            transcript=transcript,
        )
        request = CompactSessionRequest(
            session_id=completed.id,
            idempotency_key="projected-restart",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        abandoned_stream = first_app.compact_session(request)
        await anext(abandoned_stream)
        await abandoned_stream.aclose()
        assert first_compactor.requests == []

        recovering_compactor = _CapturingCompactor()
        recovered_app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
            clock=lambda: accepted_at + timedelta(minutes=6),
        )
        recovered_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=recovering_compactor,
                max_user_turns=1,
            ),
        )

        recovered = [event async for event in recovered_app.compact_session(request)]
        replay = [event async for event in recovered_app.compact_session(request)]

        assert recovered
        assert replay
        assert len(recovering_compactor.requests) == 1
        recovered_request = _serialized(recovering_compactor.requests)
        assert secret not in recovered_request
        assert REDACTED_SECRET in recovered_request
        assert await store.load_transcript(completed.id) == transcript
        durable_records = await store.query_events(EventQuery(session_id=completed.id, limit=100))
        assert secret not in _serialized(
            [record.event.model_dump(mode="json") for record in durable_records]
        )

    asyncio.run(scenario())
