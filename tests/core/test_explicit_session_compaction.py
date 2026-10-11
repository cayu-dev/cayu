"""Explicit compaction requests, execution profiles and public event boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError
from tests.core._execution_profile_fixtures import profiled_session_identity
from tests.core._explicit_compaction_fixtures import (
    CompletingProvider,
    RecordingCompactor,
    _create_profiled_session,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextBuildError,
    ContextCompactor,
    ContextRequest,
    ModelCompactor,
)
from cayu.context.footprints import RequestFootprint, RequestVariant
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelProviderError, ModelRequest, ModelStreamEvent
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileMismatchError,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import CompactSessionRequest, ForkSessionRequest, RunRequest
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor


def test_compact_session_rejects_incomplete_session_authority_before_mutation() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        compactor = RecordingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(compactor=compactor),
        )
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="incomplete-compaction-authority",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_transcript_messages(
            session.id,
            [Message.text("user", "must remain untouched")],
        )
        session = await store.update_status(session.id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="no durable execution-profile identity"):
            _ = [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=session.id,
                        idempotency_key="reject-incomplete-authority",
                        expected_run_epoch=session.run_epoch,
                        expected_transcript_cursor=1,
                    )
                )
            ]

        assert compactor.requests == []
        assert await store.load_events(session.id) == []
        assert await store.load_checkpoint(session.id) is None

    asyncio.run(run())


def test_policy_model_compaction_does_not_acknowledge_omitted_history() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = RecordingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        policy = CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=provider,
                model="summary-model",
                max_input_chars=1000,
            ),
            max_user_turns=1,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="summary-model"),
            context_policy=policy,
        )
        session = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_coverage_policy",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="recording-compaction", model="summary-model"),
        )
        transcript = [
            Message.text("user", "OLDEST_MUST_SURVIVE " + "x" * 5000),
            Message.text("assistant", "LATEST_COMPACTABLE"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(session.id, transcript)
        completed = await store.update_status(session.id, SessionStatus.COMPLETED)
        await store.checkpoint(
            session.id,
            {"future_additive_field": {"preserved": True}},
        )
        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=session.id,
                    idempotency_key="coverage-1",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]
        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is not None
        assert checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
        assert checkpoint["future_additive_field"] == {"preserved": True}
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 1
        first_request_count = len(provider.requests)
        first_footprints = [
            RequestFootprint.model_validate(event.payload)
            for event in events
            if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert len(first_footprints) == first_request_count
        assert all(
            footprint.request_variant == RequestVariant.CONTEXT_COMPACTION
            for footprint in first_footprints
        )
        first_prompts = [request.messages[-1].content[0].text for request in provider.requests]
        assert any("OLDEST_MUST_SURVIVE" in prompt for prompt in first_prompts)
        assert all("LATEST_COMPACTABLE" not in prompt for prompt in first_prompts)
        assert "OLDEST_MUST_SURVIVE" in checkpoint["context_compaction"]["summary"]
        completed_event = next(
            event for event in events if event.type == EventType.CONTEXT_COMPACTION_COMPLETED
        )
        assert completed_event.payload["requested_source_start"] == 0
        assert completed_event.payload["requested_source_end"] == 2
        assert completed_event.payload["represented_source_start"] == 0
        assert completed_event.payload["represented_source_end"] == 1
        assert completed_event.payload["represented_message_count"] == 1
        assert completed_event.payload["coverage_mode"] == "partial_prefix"
        assert completed_event.payload["chunk_count"] > 1
        assert completed_event.payload["chunk_mode"] == "hierarchical_atomic_unit"
        assert completed_event.payload["bounded_input"] is True
        assert completed_event.payload["compaction_failed"] is False
        assert "OLDEST_MUST_SURVIVE" not in json.dumps(completed_event.model_dump(mode="json"))
        checkpointed_event = next(
            event for event in events if event.type == EventType.SESSION_CHECKPOINTED
        )
        assert checkpointed_event.payload["compacted_transcript_cursor"] == 1
        assert checkpointed_event.payload["newly_compacted_message_count"] == 1

        first_projection = await policy.build_with_checkpoint(
            ContextRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="summary-model"),
                messages=transcript,
                step=1,
            ),
            checkpoint=checkpoint,
        )
        first_projection_text = json.dumps(
            [message.model_dump(mode="json") for message in first_projection.messages]
        )
        assert first_projection_text.count("OLDEST_MUST_SURVIVE") == 1
        assert first_projection_text.count("LATEST_COMPACTABLE") == 1

        retry_events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=session.id,
                    idempotency_key="coverage-2",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]
        retry_checkpoint = await store.load_checkpoint(session.id)
        assert retry_checkpoint is not None
        assert retry_checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2
        retry_prompts = [
            request.messages[-1].content[0].text
            for request in provider.requests[first_request_count:]
        ]
        retry_footprints = [
            RequestFootprint.model_validate(event.payload)
            for event in retry_events
            if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert len(retry_footprints) == len(retry_prompts)
        assert any("OLDEST_MUST_SURVIVE" in prompt for prompt in retry_prompts)
        assert any("LATEST_COMPACTABLE" in prompt for prompt in retry_prompts)
        retry_projection = await policy.build_with_checkpoint(
            ContextRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="summary-model"),
                messages=transcript,
                step=1,
            ),
            checkpoint=retry_checkpoint,
        )
        retry_projection_text = json.dumps(
            [message.model_dump(mode="json") for message in retry_projection.messages]
        )
        assert retry_projection_text.count("OLDEST_MUST_SURVIVE") == 1
        assert retry_projection_text.count("LATEST_COMPACTABLE") == 1
        assert any(event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in retry_events)

    asyncio.run(run())


def test_explicit_compaction_retry_rejects_live_provider_drift_before_dispatch(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("must not dispatch"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ],
            name="compaction-provider",
        )
        consume_batch = provider._consume_batch

        def consume_then_mutate(request: ModelRequest) -> tuple[ModelStreamEvent, ...]:
            batch = consume_batch(request)
            if len(provider.requests) == 1:
                monkeypatch.setattr(provider, "_operation_adapter", object())
                raise ModelProviderError(
                    "compactor overloaded",
                    provider=provider.name,
                    status_code=503,
                    retryable=True,
                )
            return batch

        monkeypatch.setattr(provider, "_consume_batch", consume_then_mutate)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
                ),
                max_user_turns=1,
            ),
        )
        session_id = "explicit-compaction-live-provider-drift"
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(session_id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(ContextBuildError) as caught:
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=session_id,
                    idempotency_key="explicit-live-provider-drift",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        profile_mismatch = caught.value.__cause__
        assert isinstance(profile_mismatch, ExecutionProfileMismatchError)
        assert profile_mismatch.changed_component_classes == (
            ExecutionProfileComponentClass.CONTEXT_COMPACTION,
        )
        assert len(provider.requests) == 1
        compaction_completions = [
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(compaction_completions) == 1
        assert compaction_completions[0].payload["compaction_outcome"] == "provider_error"

    asyncio.run(scenario())


class RecordingCompactionProvider(ModelProvider):
    name = "recording-compaction"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        prompt = request.messages[-1].content[0].text
        preserved = [
            marker for marker in ("OLDEST_MUST_SURVIVE", "LATEST_COMPACTABLE") if marker in prompt
        ]
        yield ModelStreamEvent.text_delta(" ".join(preserved) or "provider summary")
        yield ModelStreamEvent.completed({})


def test_compact_session_request_requires_idempotency_and_source_fences() -> None:
    with pytest.raises(ValidationError):
        CompactSessionRequest(  # type: ignore[call-arg]
            session_id="sess_compact_contract",
            expected_run_epoch=0,
            expected_transcript_cursor=3,
        )

        with pytest.raises(ValidationError):
            CompactSessionRequest(
                session_id="sess_compact_contract",
                idempotency_key="compact-1",
                expected_run_epoch=0,
                expected_transcript_cursor=3,
                instructions="x" * 4097,
            )

    with pytest.raises(ValidationError):
        CompactSessionRequest(
            session_id="sess_compact_contract",
            idempotency_key="compact-1",
            expected_run_epoch=0,
            expected_transcript_cursor=3,
            instructions="invalid surrogate: \ud800",
        )

    for field_name, value in (
        ("session_id", "invalid-surrogate-\ud800"),
        ("session_id", "invalid-nul-\x00"),
        ("idempotency_key", "invalid-nul-\x00"),
        ("instructions", "invalid-nul-\x00"),
    ):
        values = {
            "session_id": "sess_compact_contract",
            "idempotency_key": "compact-1",
            "expected_run_epoch": 0,
            "expected_transcript_cursor": 3,
            field_name: value,
        }
        with pytest.raises(ValidationError):
            CompactSessionRequest(**values)

    invalid_price = ModelPrice.fixed(
        provider_name="invalid-provider-\ud800",
        model="summary-model",
        input_per_million=Decimal("1"),
        output_per_million=Decimal("1"),
    )
    with pytest.raises(ValidationError):
        CompactSessionRequest(
            session_id="sess_compact_contract",
            idempotency_key="compact-1",
            expected_run_epoch=0,
            expected_transcript_cursor=3,
            budget_limits=(
                BudgetLimit(
                    max_estimated_cost=Decimal("1"),
                    pricing=PriceBook(prices=(invalid_price,)),
                ),
            ),
        )


@pytest.mark.parametrize(
    "values",
    [
        {"summary": "invalid-\x00summary"},
        {"summary": "invalid-\ud800-summary"},
        {"summary": "summary", "metadata": {"nested": "invalid-\x00metadata"}},
        {
            "summary": "summary",
            "model_completed_payloads": [{"model": "invalid-\ud800-model"}],
        },
    ],
)
def test_compaction_result_rejects_text_that_cannot_be_durably_persisted(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CompactionResult.model_validate(values)


def test_compact_session_instructions_change_the_governing_profile_without_publication() -> None:
    async def run() -> tuple[list[Event], list[Event]]:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )

        async def compact(session_id: str, instructions: str) -> list[Event]:
            created = await _create_profiled_session(
                app,
                store,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            transcript = [
                Message.text("user", "old request"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current request"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key=f"compact-{session_id}",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                        instructions=instructions,
                    )
                )
            ]

        return (
            await compact("sess_compaction_profile_a", "Preserve decisions."),
            await compact("sess_compaction_profile_b", "Preserve file names."),
        )

    first, second = asyncio.run(run())

    first_profiles = {event.payload.get("execution_profile_fingerprint") for event in first}
    second_profiles = {event.payload.get("execution_profile_fingerprint") for event in second}
    assert len(first_profiles) == 1
    assert len(second_profiles) == 1
    assert None not in first_profiles
    assert None not in second_profiles
    assert first_profiles != second_profiles
    assert (
        first[0].payload["instruction_digest"] == hashlib.sha256(b"Preserve decisions.").hexdigest()
    )
    serialized = str([event.model_dump(mode="json") for event in (*first, *second)])
    assert "Preserve decisions." not in serialized
    assert "Preserve file names." not in serialized


def test_compact_session_redacts_instructions_before_provider_and_durable_boundaries() -> None:
    secret = "explicit-compaction-instruction-canary"

    async def run() -> tuple[RecordingCompactor, list[Event], dict[str, object] | None]:
        store = InMemorySessionStore()
        compactor = RecordingCompactor()
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
                compact_after_messages=100,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_boundary",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-boundary-1",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    instructions=f"Preserve decisions but never expose {secret}.",
                )
            )
        ]
        return compactor, events, await store.load_checkpoint(created.id)

    compactor, events, checkpoint = asyncio.run(run())

    assert compactor.requests[0].instructions == (
        f"Preserve decisions but never expose {REDACTED_SECRET}."
    )
    assert secret not in str([event.model_dump(mode="json") for event in events])
    assert secret not in str(checkpoint)


def test_compact_session_rejects_secret_policy_checkpoint_before_publication() -> None:
    from cayu.context.base import ContextBuildResult

    secret = "explicit-policy-checkpoint-secret-canary"

    class SecretCheckpointPolicy(CheckpointCompactionContextPolicy):
        async def build_with_checkpoint(
            self,
            request: ContextRequest,
            *,
            checkpoint: dict[str, Any] | None,
        ) -> ContextBuildResult:
            del checkpoint
            return ContextBuildResult(
                messages=[Message.text("assistant", f"unsafe result {secret}")],
                checkpoint={"context_compaction": {"summary": secret}},
                checkpoint_event_payload={
                    "compacted_transcript_cursor": len(request.messages),
                    "custom_detail": secret,
                },
            )

    async def run() -> tuple[
        ValueError,
        list[Event],
        dict[str, object] | None,
        list[Event],
    ]:
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=SecretCheckpointPolicy(
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_secret_checkpoint",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        observed: list[Event] = []
        with pytest.raises(ValueError, match="contains a workload secret") as exc_info:
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="secret-checkpoint",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                observed.append(event)
        return (
            exc_info.value,
            observed,
            await store.load_checkpoint(created.id),
            await store.load_events(created.id),
        )

    error, observed, checkpoint, durable_events = asyncio.run(run())

    assert all(event.type is not EventType.SESSION_CHECKPOINTED for event in observed)
    assert secret not in repr(observed)
    assert secret not in repr(checkpoint)
    assert secret not in repr(durable_events)
    traceback = error.__traceback__
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("\\", "/")
        if "/src/cayu/" in filename:
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next


@pytest.mark.parametrize("secret_location", ["key", "value", "nested_typed_key"])
def test_compact_session_rejects_secret_checkpoint_event_payload_before_publication(
    secret_location: str,
) -> None:
    from cayu.context.base import ContextBuildResult

    secret = (
        "checkpoint"
        if secret_location == "nested_typed_key"
        else "explicit-policy-event-payload-secret-canary"
    )

    class SecretEventPayloadPolicy(CheckpointCompactionContextPolicy):
        def __init__(self) -> None:
            super().__init__(
                max_user_turns=1,
                compact_after_messages=1,
            )
            self.result: ContextBuildResult | None = None

        async def build_with_checkpoint(
            self,
            request: ContextRequest,
            *,
            checkpoint: dict[str, Any] | None,
        ) -> ContextBuildResult:
            del checkpoint
            if secret_location == "nested_typed_key":
                unsafe_payload = {"custom": {"checkpoint": "safe"}}
            else:
                unsafe_field = secret if secret_location == "key" else "custom_detail"
                unsafe_value = "safe" if secret_location == "key" else secret
                unsafe_payload = {unsafe_field: unsafe_value}
            self.result = ContextBuildResult(
                messages=[Message.text("assistant", f"unsafe result {secret}")],
                checkpoint={"context_compaction": {"summary": "safe summary"}},
                checkpoint_event_payload={
                    "compacted_transcript_cursor": len(request.messages),
                    **unsafe_payload,
                },
            )
            return self.result

    async def run() -> tuple[
        ValueError,
        SecretEventPayloadPolicy,
        dict[str, object] | None,
        list[Event],
    ]:
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        policy = SecretEventPayloadPolicy()
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=policy,
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_explicit_secret_event_{secret_location}",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        with pytest.raises(ValueError, match="contains a workload secret") as exc_info:
            async for _ in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key=f"secret-event-{secret_location}",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass
        return (
            exc_info.value,
            policy,
            await store.load_checkpoint(created.id),
            await store.load_events(created.id),
        )

    error, policy, checkpoint, durable_events = asyncio.run(run())

    assert checkpoint == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
    assert secret not in repr(durable_events)
    assert policy.result is not None
    assert policy.result.messages == []
    assert policy.result.checkpoint == {}
    assert policy.result.checkpoint_event_payload == {}
    if secret_location == "nested_typed_key":
        return
    traceback = error.__traceback__
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("\\", "/")
        if "/src/cayu/" in filename:
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next


@pytest.mark.parametrize(
    "failure_checkpoint",
    [
        None,
        {"context_compaction": {"summary": "safe summary"}},
    ],
)
def test_compact_session_discards_secret_checkpoint_event_payload_from_failure(
    failure_checkpoint: dict[str, Any] | None,
) -> None:
    secret = "explicit-failure-event-payload-secret-canary"

    class FailingCheckpointPolicy(CheckpointCompactionContextPolicy):
        async def build_with_checkpoint(
            self,
            request: ContextRequest,
            *,
            checkpoint: dict[str, Any] | None,
        ):
            del request, checkpoint
            raise ContextBuildError(
                "context build failed",
                compaction_telemetry=[],
                checkpoint=failure_checkpoint,
                checkpoint_event_payload={
                    "checkpoint": "context_compaction",
                    "custom_detail": secret,
                },
                cause=RuntimeError("safe policy failure"),
            )

    async def run() -> tuple[ContextBuildError, dict[str, object] | None, list[Event]]:
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=FailingCheckpointPolicy(
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_secret_failure_event_payload",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(ContextBuildError) as exc_info:
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="secret-failure-event-payload",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass
        return (
            exc_info.value,
            await store.load_checkpoint(created.id),
            await store.load_events(created.id),
        )

    error, checkpoint, durable_events = asyncio.run(run())

    assert error.checkpoint is None
    assert error.checkpoint_event_payload is None
    assert secret not in repr(error.__dict__)
    traceback = error.__traceback__
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("\\", "/")
        if "/src/cayu/" in filename:
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next
    assert secret not in repr(checkpoint)
    assert secret not in repr(durable_events)


def test_fork_inherits_compacted_context_without_source_operation_records() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider())
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_fork_source",
                messages=[Message.text("user", "create only")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        source = await store.update_status(created.id, SessionStatus.COMPLETED)
        idempotency_key = "compact-fork-independent-key"
        source_events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=source.id,
                    idempotency_key=idempotency_key,
                    expected_run_epoch=source.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]
        assert source_events[-1].type == EventType.SESSION_CHECKPOINTED

        fork_events = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id="sess_compact_fork_child",
                )
            )
        ]
        assert fork_events[-1].type == EventType.SESSION_FORKED
        child = await store.load("sess_compact_fork_child")
        assert child is not None
        child_checkpoint = await store.load_checkpoint(child.id)
        assert child_checkpoint is not None
        assert child_checkpoint["context_compaction"]["summary"] == "durable compact summary"
        assert "session_operations" not in child_checkpoint

        child_messages = [
            Message.text("user", "child request"),
            Message.text("assistant", "child answer"),
        ]
        await store.append_transcript_messages(child.id, child_messages)
        child_events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=child.id,
                    idempotency_key=idempotency_key,
                    expected_run_epoch=child.run_epoch,
                    expected_transcript_cursor=len(transcript) + len(child_messages),
                )
            )
        ]

        assert child_events[-1].type == EventType.SESSION_CHECKPOINTED
        child_checkpoint = await store.load_checkpoint(child.id)
        assert child_checkpoint is not None
        assert "session_operations" not in child_checkpoint
        child_operation = await store.load_session_operation(child.id, idempotency_key)
        assert child_operation is not None
        assert child_operation["status"] == "completed"

    asyncio.run(run())


def test_compact_session_sanitizes_failure_type_in_event_and_operation_record() -> None:
    unsafe_error_type = type("Unsafe\n" * 100, (BaseException,), {})

    class UnsafeFailureCompactor(ContextCompactor):
        def provider_budget_identity(self, session) -> None:
            del session
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            del request
            raise unsafe_error_type("compaction failed")

    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=UnsafeFailureCompactor(),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_unsafe_error_type",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-unsafe-error-type",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        events: list[Event] = []
        with pytest.raises(unsafe_error_type):
            async for event in app.compact_session(request):
                events.append(event)

        assert [event.type for event in events] == [EventType.CONTEXT_COMPACTION_STARTED]
        events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        failed = next(
            event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED
        )
        assert failed.payload["error_type"] == "BaseException"
        operation = await store.load_session_operation(created.id, request.idempotency_key)
        assert operation is not None
        assert operation["error_type"] == "BaseException"
        assert "\n" not in json.dumps(failed.model_dump(mode="json"))

    asyncio.run(run())
