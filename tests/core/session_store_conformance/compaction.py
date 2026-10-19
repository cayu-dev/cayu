"""Shared session-store compaction contract cases."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from tests.core._bounded_waits import FAKE_STEP_TIMEOUT_SECONDS
from tests.core._execution_profile_fixtures import (
    profiled_session_identity,
)
from tests.core._session_operation_fault_harness import (
    MatchPolicy,
    PauseBeforeTransform,
    PublicationBarrier,
    PublicationBoundary,
    PublicationFaultActionKind,
    PublicationFaultOutcome,
    SessionOperationFaultHarness,
    SessionOperationFaultRule,
    SessionOperationSelector,
)
from tests.core.session_store_conformance.support import (
    _close_store,
    _collect_events,
    _ConformanceBlockingCompactor,
    _ConformanceCompactor,
    _ConformanceOverlappingCompactor,
    _ConformancePartialCancellationCompactor,
    _ConformancePartialCompactor,
    _expire_compaction_claim_for_test,
    _open_store,
    _reopen_store,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    ResolutionActor,
)
from cayu.artifacts.attachments import file_attachment
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    ModelCompactor,
)
from cayu.context.footprints import RequestFootprintConfig
from cayu.events import (
    Event,
    EventType,
)
from cayu.messages import (
    Message,
    MessageRole,
    ToolResultPart,
)
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime._compaction import automatic as _automatic_compaction
from cayu.runtime._compaction import explicit as _session_compaction
from cayu.runtime._event_projection import (
    public_event_sequence,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import (
    CompactSessionRequest,
    RunRequest,
)
from cayu.tools.exposure import (
    ToolCapabilityCeiling,
)
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

pytestmark = pytest.mark.postgres


def test_automatic_compaction_publication_contention_conformance(
    session_store_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ConcurrentProvider(ModelProvider):
        name = "concurrent-store-compactor"

        def __init__(self, *, summary: bool) -> None:
            self.summary = summary
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            if self.summary:
                yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed({"model": request.model})

    async def run() -> None:
        store = await _open_store(session_store_case)
        compaction_provider = ConcurrentProvider(summary=True)
        runtime_provider = ConcurrentProvider(summary=False)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(runtime_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="runtime-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        barriers = [PublicationBarrier() for _index in range(6)]
        rules = tuple(
            SessionOperationFaultRule(
                rule_id=f"compaction-model-start-{index}",
                selector=SessionOperationSelector(
                    session_id=(f"sess_compaction_contention_{session_store_case[0]}_{index}"),
                    event_types=frozenset({EventType.MODEL_STARTED}),
                ),
                actions=(PauseBeforeTransform(barrier),),
                on_exhausted=MatchPolicy.DELEGATE,
            )
            for index, barrier in enumerate(barriers)
        )

        # Each session's first compaction start pauses at its barrier, and the
        # runtime must give up on it, cancel it, and publish it again. Time out
        # only that paused publication, once it is at its barrier, and leave
        # every other publication its production deadline. A short global
        # deadline also failed publications that were merely slow on a busy
        # runner, before their session reached its barrier.
        barrier_by_session = {
            f"sess_compaction_contention_{session_store_case[0]}_{index}": barrier
            for index, barrier in enumerate(barriers)
        }
        paused_start_barriers: dict[object, PublicationBarrier] = {}
        original_persist_exact_replay = app._event_writer.persist_exact_replay
        original_await_shielded_task_outcome = _automatic_compaction.await_shielded_task_outcome

        def persist_exact_replay(event: Event):
            publication = original_persist_exact_replay(event)
            if (
                event.type is EventType.MODEL_STARTED
                and event.payload.get("purpose") == "context_compaction"
                and event.session_id in barrier_by_session
            ):
                paused_start_barriers[publication] = barrier_by_session.pop(event.session_id)
            return publication

        async def await_shielded_task_outcome(task, **kwargs):
            barrier = paused_start_barriers.pop(task.get_coro(), None)
            if barrier is not None:
                await barrier.wait_until_entered(timeout=FAKE_STEP_TIMEOUT_SECONDS)
                kwargs["timeout_s"] = 0.0
            return await original_await_shielded_task_outcome(task, **kwargs)

        monkeypatch.setattr(app._event_writer, "persist_exact_replay", persist_exact_replay)
        monkeypatch.setattr(
            _automatic_compaction,
            "await_shielded_task_outcome",
            await_shielded_task_outcome,
        )
        try:
            async with SessionOperationFaultHarness(
                store,
                rules=rules,
                boundary=PublicationBoundary.EVENT_APPEND,
            ) as faults:
                results = await asyncio.gather(
                    *(
                        _collect_events(
                            app.run(
                                RunRequest(
                                    agent_name="assistant",
                                    session_id=(
                                        "sess_compaction_contention_"
                                        f"{session_store_case[0]}_{index}"
                                    ),
                                    messages=[
                                        Message.text("user", "old"),
                                        Message.text("assistant", "old answer"),
                                        Message.text("user", "current"),
                                    ],
                                )
                            )
                        )
                        for index in range(6)
                    )
                )
            terminal_summaries = [(events[-1].type, events[-1].payload) for events in results]
            assert all(
                event_type == EventType.SESSION_COMPLETED
                for event_type, _payload in terminal_summaries
            ), terminal_summaries
            assert compaction_provider.calls == 6
            assert runtime_provider.calls == 6
            scheduled = [trace for trace in faults.trace if trace.matched_rule_id is not None]
            paused = [
                trace
                for trace in scheduled
                if trace.action is PublicationFaultActionKind.PAUSE_BEFORE_TRANSFORM
            ]
            assert len(paused) == 6
            assert all(trace.outcome is PublicationFaultOutcome.CANCELLED for trace in paused)
            assert faults.dropped_trace_entries == 0
            for index in range(6):
                durable = await store.load_events(
                    f"sess_compaction_contention_{session_store_case[0]}_{index}"
                )
                assert (
                    sum(
                        event.type == EventType.MODEL_STARTED
                        and event.payload.get("purpose") == "context_compaction"
                        for event in durable
                    )
                    == 1
                )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_explicit_compaction_operation(session_store_case) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            secret = "explicit-conformance-transcript-secret"
            compactor = _ConformanceCompactor()
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
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_compaction_conformance",
                    messages=[Message.text("user", "create only")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                    app=app,
                ),
            )
            transcript = [
                Message.text("user", f"old request containing {secret}"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current request"),
                Message.text("assistant", "current answer"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            await store.append_event(
                created.id,
                Event(
                    type=EventType.INTERACTION_COMPLETED,
                    session_id=created.id,
                    interaction_id="interaction-deleted-with-session",
                ),
            )
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-conformance-1",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )

            first = [event async for event in app.compact_session(request)]
            store = await _reopen_store(session_store_case, store)
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
            replay = [event async for event in app.compact_session(request)]
            assert [event.id for event in replay] == [event.id for event in first]
            assert compactor.calls == 1
            assert secret not in json.dumps(
                [message.model_dump(mode="json") for message in compactor.requests[0].messages]
            )
            assert REDACTED_SECRET in json.dumps(
                [message.model_dump(mode="json") for message in compactor.requests[0].messages]
            )
            assert await store.load_transcript(created.id) == transcript
            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            assert checkpoint["context_compaction"]["summary"] == "summary-1"
            assert "session_operations" not in checkpoint
            completed_operation = await store.load_session_operation(
                created.id,
                request.idempotency_key,
            )
            assert completed_operation is not None
            assert completed_operation["status"] == "completed"

            with pytest.raises(ValueError, match="transcript cursor is stale"):
                async for _event in app.compact_session(
                    request.model_copy(
                        update={
                            "idempotency_key": "compact-stale",
                            "expected_transcript_cursor": len(transcript) - 1,
                        }
                    )
                ):
                    pass

            tail = [
                Message.text("user", "later request"),
                Message.text("assistant", "later answer"),
            ]
            await store.append_transcript_messages(created.id, tail)
            failed_request = request.model_copy(
                update={
                    "idempotency_key": "compact-failure",
                    "expected_transcript_cursor": len(transcript) + len(tail),
                }
            )
            compactor.fail_next = True
            with pytest.raises(RuntimeError, match="conformance compactor failed"):
                async for _event in app.compact_session(failed_request):
                    pass
            assert compactor.calls == 2
            failed_operation = await store.load_session_operation(
                created.id,
                failed_request.idempotency_key,
            )
            assert failed_operation is not None
            assert failed_operation["status"] == "failed"

            retry = [
                event
                async for event in app.compact_session(
                    failed_request.model_copy(update={"idempotency_key": "compact-retry"})
                )
            ]
            assert retry[-1].type == EventType.SESSION_CHECKPOINTED
            assert compactor.calls == 3
            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            assert "session_operations" not in checkpoint
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_explicit_compaction_projects_legacy_summary(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            secret = "explicit-conformance-legacy-summary-secret"
            legacy_summary = f"legacy summary containing {secret}"
            compactor = _ConformanceCompactor()
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
            session_id = f"sess_compaction_legacy_summary_{session_store_case[0]}"
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
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
                Message.text("user", "newer request"),
                Message.text("assistant", "newer answer"),
                Message.text("user", "current request"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            await store.checkpoint(
                created.id,
                {
                    "context_compaction": {
                        "version": 2,
                        "summary": legacy_summary,
                        "compacted_transcript_cursor": 2,
                        "metadata": {"compactor": "legacy", "mode": "deterministic"},
                    }
                },
            )
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-legacy-summary",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )

            first = [event async for event in app.compact_session(request)]
            store = await _reopen_store(session_store_case, store)
            replay_app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
            replay_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            replay = [event async for event in replay_app.compact_session(request)]

            assert [event.id for event in replay] == [event.id for event in first]
            assert compactor.calls == 1
            assert compactor.requests[0].existing_summary == (
                f"legacy summary containing {REDACTED_SECRET}"
            )
            assert secret not in json.dumps(compactor.requests[0].model_dump(mode="json"))
            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            assert checkpoint["context_compaction"]["summary"] == (
                f"legacy summary containing {REDACTED_SECRET}|summary-1"
            )
            assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 4
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("authority_position", ("first", "middle", "final"))
@pytest.mark.parametrize(
    "authority_kind",
    ("file_attachment", "runtime_projection_artifact_id", "runtime_projection_store_id"),
)
def test_session_store_conformance_explicit_compaction_rejects_secret_authority(
    session_store_case,
    authority_position: str,
    authority_kind: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            secret = (
                "deadbeef"
                if authority_kind == "runtime_projection_artifact_id"
                else "explicit-conformance-authority-secret"
            )
            compactor = _ConformanceCompactor()
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
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_compaction_authority_conformance",
                    messages=[],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                    app=app,
                ),
            )
            if authority_kind == "file_attachment":
                artifact = file_attachment(
                    artifact_id=f"artifact-{secret}-suffix",
                    kind="image",
                    filename="safe.png",
                    content_type="image/png",
                    size_bytes=1,
                )
            else:
                artifact_id = (
                    f"art_{secret}{'a' * 24}"
                    if authority_kind == "runtime_projection_artifact_id"
                    else f"art_{'a' * 32}"
                )
                artifact = {
                    "type": "cayu.tool_result_artifact.v1",
                    "artifact_id": artifact_id,
                    "store_id": (
                        "artifacts"
                        if authority_kind == "runtime_projection_artifact_id"
                        else f"store-{secret}-suffix"
                    ),
                    "filename": f"tool-result-{artifact_id}.txt",
                    "content_type": "text/plain; charset=utf-8",
                    "size_bytes": 1,
                    "sha256": "b" * 64,
                    "scope": "session",
                    "session_id_sha256": "c" * 64,
                    "readback_max_bytes": 64,
                    "projection_authority": "cayu.tool_result_projection.v1",
                }
            authority_message = Message(
                role=MessageRole.TOOL,
                content=(
                    ToolResultPart(
                        tool_call_id="safe-call",
                        tool_name="safe_tool",
                        artifacts=[artifact],
                    ),
                ),
            )
            safe_messages = [
                Message.text("user", "old request"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current request"),
            ]
            authority_index = {
                "first": 0,
                "middle": 1,
                "final": len(safe_messages),
            }[authority_position]
            transcript = list(safe_messages)
            transcript.insert(authority_index, authority_message)
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            before_events = await store.query_events(EventQuery(session_id=created.id, limit=100))
            before_checkpoint = await store.load_checkpoint(created.id)
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key=f"reject-{authority_kind}-authority-conformance",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )

            with pytest.raises(
                ValueError,
                match="workload secret in execution authority",
            ):
                async for _event in app.compact_session(request):
                    pass

            current = await store.load(created.id)
            assert current is not None
            assert current.status == completed.status
            assert current.run_epoch == completed.run_epoch
            assert await store.load_transcript(created.id) == transcript
            assert await store.load_checkpoint(created.id) == before_checkpoint
            assert (
                await store.query_events(EventQuery(session_id=created.id, limit=100))
                == before_events
            )
            assert (
                await store.load_session_operation(
                    created.id,
                    request.idempotency_key,
                )
                is None
            )
            assert compactor.calls == 0
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_partial_compaction_cursor_survives_reopen(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=_ConformancePartialCompactor(), max_user_turns=1
                ),
            )
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_partial_coverage_conformance",
                    messages=[],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                    app=app,
                ),
            )
            transcript = [
                Message.text("user", "old"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="partial-coverage-1",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
            events = [event async for event in app.compact_session(request)]
            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 1
            assert await store.load_transcript(created.id) == transcript
            reopened = await _reopen_store(session_store_case, store)
            store = reopened
            assert (await store.load_checkpoint(created.id))["context_compaction"][
                "compacted_transcript_cursor"
            ] == 1
            assert any(event.type == EventType.SESSION_CHECKPOINTED for event in events)

            replay_app = CayuApp(session_store=store, enable_logging=False)
            replay_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=_ConformancePartialCompactor(), max_user_turns=1
                ),
            )
            second = request.model_copy(update={"idempotency_key": "partial-coverage-2"})
            second_events = [event async for event in replay_app.compact_session(second)]
            second_checkpoint = await store.load_checkpoint(created.id)
            assert second_checkpoint is not None
            assert second_checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2
            assert await store.load_transcript(created.id) == transcript
            assert any(event.type == EventType.SESSION_CHECKPOINTED for event in second_events)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_cancelled_partial_compaction_publishes_no_cursor(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            compactor = _ConformancePartialCancellationCompactor()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_partial_cancel_conformance",
                    messages=[],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                    app=app,
                ),
            )
            transcript = [
                Message.text("user", "old"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="partial-cancel",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )

            async def collect() -> list[Event]:
                return [event async for event in app.compact_session(request)]

            task = asyncio.create_task(collect())
            await compactor.started.wait()
            task.cancel("cancel partial publication")
            with pytest.raises(asyncio.CancelledError, match="cancel partial publication"):
                await task

            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is None or "context_compaction" not in checkpoint
            assert await store.load_transcript(created.id) == transcript
            durable_events = [
                record.event
                for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
            ]
            assert EventType.CONTEXT_COMPACTION_COMPLETED not in {
                event.type for event in durable_events
            }
            assert EventType.SESSION_CHECKPOINTED not in {event.type for event in durable_events}
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_fences_reclaimed_compaction_attempts(
    session_store_case,
) -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        store = await _open_store(session_store_case)
        try:
            compactor = _ConformanceOverlappingCompactor()

            def configured_app(*, now: datetime) -> CayuApp:
                app = CayuApp(
                    session_store=store,
                    request_footprint=RequestFootprintConfig(enabled=False),
                    enable_logging=False,
                    clock=lambda: now,
                )
                app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    context_policy=CheckpointCompactionContextPolicy(
                        compactor=compactor,
                        max_user_turns=1,
                    ),
                )
                return app

            first_app = configured_app(now=accepted_at)
            recovered_app = configured_app(now=accepted_at + timedelta(minutes=6))
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_compaction_claim_conformance",
                    messages=[Message.text("user", "create only")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                    app=first_app,
                ),
            )
            transcript = [
                Message.text("user", "old request"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current request"),
                Message.text("assistant", "current answer"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            first_request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-claim-conformance",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
                requested_by=ResolutionActor(subject="operator-a"),
            )
            recovered_request = first_request.model_copy(
                update={"requested_by": ResolutionActor(subject="operator-b")}
            )

            async def collect(app: CayuApp, request: CompactSessionRequest) -> list[Event]:
                return [event async for event in app.compact_session(request)]

            first_task = asyncio.create_task(collect(first_app, first_request))
            await asyncio.wait_for(compactor.started[0].wait(), timeout=10)
            await _expire_compaction_claim_for_test(
                store, created.id, first_request.idempotency_key
            )
            recovered_task = asyncio.create_task(collect(recovered_app, recovered_request))
            await asyncio.wait_for(compactor.started[1].wait(), timeout=10)
            compactor.release[1].set()
            recovered_events = await recovered_task
            compactor.release[0].set()
            with pytest.raises(RuntimeError, match="superseded"):
                await first_task

            store = await _reopen_store(session_store_case, store)
            replay_app = CayuApp(session_store=store, enable_logging=False)
            replay_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            durable_records = await store.query_events(EventQuery(session_id=created.id, limit=100))
            durable_events = [record.event for record in durable_records]
            replay = [event async for event in replay_app.compact_session(recovered_request)]

            assert recovered_events[-1].type == EventType.SESSION_CHECKPOINTED
            assert [public_event_sequence(event.id) for event in replay] == [
                record.sequence for record in durable_records
            ]
            assert (
                sum(
                    event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in durable_events
                )
                == 1
            )
            assert (
                sum(event.type == EventType.SESSION_CHECKPOINTED for event in durable_events) == 1
            )
            assert sum(event.type == EventType.MODEL_COMPLETED for event in durable_events) == 2
            operation_events = [
                event
                for event in durable_events
                if event.type != EventType.REQUEST_FOOTPRINT_RECORDED
            ]
            assert len({event.payload["operation_id"] for event in operation_events}) == 1
            assert len({event.payload["attempt_id"] for event in operation_events}) == 2
            delivery_ids = {
                delivery.event_id
                for delivery in await store.list_persisted_event_side_effect_deliveries(limit=1000)
            }
            assert {event.id for event in durable_events} <= delivery_ids
            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            assert checkpoint["context_compaction"]["summary"] == "summary from attempt 2"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_heartbeats_active_compaction_claim(
    session_store_case,
    monkeypatch,
) -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = await _open_store(session_store_case)
        compactor = _ConformanceBlockingCompactor()
        task: asyncio.Task[list[Event]] | None = None
        try:
            app = CayuApp(
                session_store=store,
                enable_logging=False,
                clock=lambda: now["value"],
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_compaction_heartbeat_conformance",
                    messages=[],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
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
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-heartbeat-conformance",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )

            async def collect() -> list[Event]:
                return [event async for event in app.compact_session(request)]

            task = asyncio.create_task(collect())
            await asyncio.wait_for(compactor.started.wait(), timeout=10)
            now["value"] = accepted_at + timedelta(minutes=4)
            first_renewal_expiry = accepted_at + timedelta(minutes=9)
            async with asyncio.timeout(5):
                while True:
                    checkpoint = await store.load_checkpoint(created.id)
                    assert checkpoint is not None
                    record = checkpoint["session_operations"]["records"][request.idempotency_key]
                    if datetime.fromisoformat(record["claim_expires_at"]) >= first_renewal_expiry:
                        break
                    await asyncio.sleep(0)

            now["value"] = accepted_at + timedelta(minutes=6)
            expected_expiry = accepted_at + timedelta(minutes=11)
            async with asyncio.timeout(5):
                while True:
                    checkpoint = await store.load_checkpoint(created.id)
                    assert checkpoint is not None
                    record = checkpoint["session_operations"]["records"][request.idempotency_key]
                    if datetime.fromisoformat(record["claim_expires_at"]) >= expected_expiry:
                        break
                    await asyncio.sleep(0)

            with pytest.raises(RuntimeError, match="already running"):
                async for _event in app.compact_session(request):
                    pass
            assert compactor.calls == 1

            compactor.release.set()
            events = await task
            task = None
            assert events[-1].type == EventType.SESSION_CHECKPOINTED
            assert compactor.calls == 1

            store = await _reopen_store(session_store_case, store)
            operation = await store.load_session_operation(created.id, request.idempotency_key)
            assert operation is not None
            assert operation["status"] == "completed"
        finally:
            compactor.release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_blocks_delete_during_explicit_compaction(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            compactor = _ConformanceOverlappingCompactor()
            app = CayuApp(
                session_store=store,
                request_footprint=RequestFootprintConfig(enabled=False),
                enable_logging=False,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_compaction_delete_conformance",
                    messages=[Message.text("user", "create only")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
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
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            await store.append_event(
                created.id,
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=created.id,
                ),
            )
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-delete-conformance",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )

            async def collect() -> list[Event]:
                return [event async for event in app.compact_session(request)]

            task = asyncio.create_task(collect())
            await compactor.started[0].wait()
            with pytest.raises(ValueError, match="durable operation .* is active"):
                await store.delete_session(created.id)
            assert await store.load(created.id) is not None

            compactor.release[0].set()
            events = await task
            assert events[-1].type == EventType.SESSION_CHECKPOINTED
            await store.delete_session(created.id)
            assert await store.load(created.id) is None
            with pytest.raises(KeyError, match="Session not found"):
                await store.query_latest_interaction_events(created.id, limit=1)
            with pytest.raises(KeyError, match="Session not found"):
                await store.load_session_operation(created.id, request.idempotency_key)
        finally:
            await _close_store(store)

    asyncio.run(run())
