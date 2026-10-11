"""Shared session-store durable inputs contract cases."""

from __future__ import annotations

import asyncio
import io
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from tests.core.session_store_conformance.support import (
    _assert_durable_error,
    _assert_portable_number_probe,
    _close_store,
    _identity,
    _new_postgres_store,
    _open_store,
    _portable_number_probe,
    _reopen_store,
    _reset_postgres_data,
)

import cayu.sessions.base as sessions_module
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    DurableValueError,
    extract_durable_value_error,
)
from cayu.events import (
    Event,
    EventType,
    event_durable_envelope,
)
from cayu.messages import (
    CitationPart,
    CitationProvenance,
    HostedToolCallPart,
    Message,
    MessageRole,
    ToolCallPart,
    WebSearchAction,
    WebSearchAPISource,
    WebSearchSource,
)
from cayu.sessions._model_failover import ModelTarget
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.messaging import (
    EnqueueSessionMessageRequest,
    SessionMessageDeliveryMode,
    SessionMessageQueueStatus,
    SessionQueuedMessage,
)
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import (
    CompactSessionRequest,
    RunRequest,
)
from cayu.storage.jsonl_export import export_sessions, import_sessions
from cayu.storage.sqlite import SQLiteSessionStore

pytestmark = pytest.mark.postgres


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: CompactSessionRequest(
            session_id="sess_1",
            idempotency_key="compact_1",
            expected_run_epoch=0,
            expected_transcript_cursor=0,
            instructions=value,
        ),
        lambda value: EnqueueSessionMessageRequest(
            session_id="sess_1",
            idempotency_key="queue_1",
            content=value,
            delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
        ),
        lambda value: SessionQueuedMessage(
            queue_id="queue_1",
            session_id="sess_1",
            idempotency_key="queue_1",
            content=value,
            delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            status=SessionMessageQueueStatus.QUEUED,
            ordering_key=1,
            accepted_run_epoch=0,
            accepted_transcript_cursor=0,
            accepted_event_id="event_1",
            accepted_at=datetime.now(UTC),
        ),
    ],
)
def test_durable_queue_and_compaction_validation_does_not_echo_rejected_input(factory) -> None:
    secret = "workload-secret-value\x00"

    with pytest.raises(ValidationError) as raised:
        factory(secret)

    durable_error = extract_durable_value_error(raised.value)
    assert durable_error is not None
    assert durable_error.code == "nul_character"
    assert "workload-secret-value" not in str(raised.value)


@pytest.mark.parametrize("lose_acknowledgement", [False, True])
def test_session_store_conformance_uncertain_output_evidence(
    session_store_case, monkeypatch, lose_acknowledgement
) -> None:
    from tests.core.test_tool_effect_state import _intent

    from cayu.runtime._tool_effect_state import ToolEffectConflict, ToolEffectStateOwner

    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            intent = await _intent(store)
            owner = ToolEffectStateOwner(store)
            executing = await owner.begin(intent, run_epoch=0)
            session = await store.load(intent.session_id)
            assert session is not None
            evidence = {"portable_result_evidence": {"receipt_id": "unverified-effect-receipt"}}
            original_publish = store.publish_session_operation
            armed = lose_acknowledgement

            async def publish(*args, **kwargs):
                nonlocal armed
                result = await original_publish(*args, **kwargs)
                if armed:
                    armed = False
                    raise OSError("publication acknowledgement lost")
                return result

            monkeypatch.setattr(store, "publish_session_operation", publish)
            assert await owner.preserve_unresolved(
                session,
                tool_round_id=intent.tool_round_id,
                tool_call_ids=(intent.tool_call_id,),
                unverified_output=evidence,
            )
            evidence["portable_result_evidence"]["receipt_id"] = "caller mutation"
            restored = await ToolEffectStateOwner(store).load(intent)
            assert restored is not None
            assert restored.state == "outcome_unknown"
            assert restored.terminal is None and restored.observation is None
            assert restored.unverified_output == {
                "portable_result_evidence": {"receipt_id": "unverified-effect-receipt"}
            }
            events = await store.load_events(intent.session_id)
            assert len(events) == 1
            assert events[0].payload["unverified_output"] == restored.unverified_output
            assert await owner.preserve_unresolved(
                session,
                tool_round_id=intent.tool_round_id,
                tool_call_ids=(intent.tool_call_id,),
                unverified_output=restored.unverified_output,
            )
            with pytest.raises(ToolEffectConflict):
                await owner.preserve_unresolved(
                    session,
                    tool_round_id=intent.tool_round_id,
                    tool_call_ids=(intent.tool_call_id,),
                    unverified_output=evidence,
                )
            with pytest.raises(ToolEffectConflict):
                await owner.begin(intent, run_epoch=0)
            with pytest.raises(ToolEffectConflict):
                await owner.transition(
                    executing,
                    state="outcome_unknown",
                    run_epoch=0,
                    unverified_output={"receipt_id": "conflicting receipt"},
                )
            assert await owner.load(intent) == restored
            assert len(await store.load_events(intent.session_id)) == 1
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_oversize_checkpoint_preserves_transcript(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "checkpoint-size-contract"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create")],
                ),
                identity=_identity(),
            )
            stable = {"stable": True}
            await store.checkpoint(session_id, stable)
            before = await store.load(session_id)
            oversized = {"value": "x" * (16 * 1024 * 1024)}
            with pytest.raises(DurableValueError) as caught:
                await store.append_transcript_messages_and_transform_checkpoint(
                    session_id,
                    [Message.text("assistant", "must not append")],
                    lambda _session, _checkpoint: oversized,
                )
            assert caught.value.dimension == "bytes"
            assert caught.value.limit == 16 * 1024 * 1024
            assert await store.load_checkpoint(session_id) == stable
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []
            assert await store.load(session_id) == before
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_event_envelope_size_is_atomic(session_store_case) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "event-size-contract"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create")],
                ),
                identity=_identity(),
            )
            event = Event(type="custom.size", session_id=session_id, payload={"value": ""})
            overhead = len(
                json.dumps(
                    event_durable_envelope(event), ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            )
            remaining = 2 * 1024 * 1024 - overhead
            event.payload["value"] = "€" * (remaining // 3) + "x" * (remaining % 3)
            await store.append_event(session_id, event)
            stored = await store.load_events(session_id)
            assert len(stored) == 1
            assert stored[0].payload == event.payload
            from cayu.server.sse import event_to_sse_message

            assert len(event_to_sse_message(stored[0])["data"].encode("utf-8")) <= 2 * 1024 * 1024
            bad = event.model_copy(deep=True)
            bad.payload["value"] += "x"
            good = Event(type="custom.size", session_id=session_id)
            with pytest.raises((DurableValueError, ValidationError)) as caught:
                await store.append_events(session_id, [good, bad])
            error = extract_durable_value_error(caught.value)
            assert error is not None
            assert error.field_name == "event"
            assert error.dimension == "bytes"
            assert error.limit == 2 * 1024 * 1024
            assert error.observed_lower_bound == error.limit + 1
            assert await store.load_events(session_id) == stored
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_annotation_size_limits_are_atomic(session_store_case) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "annotation-size-contract"
            limit = 256 * 1024
            overhead = len(json.dumps({"value": ""}, separators=(",", ":")).encode())
            exact = {"value": "x" * (limit - overhead)}
            labels = {f"label-{index}": "value" for index in range(200)}
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create")],
                    metadata=exact,
                    labels=labels,
                ),
                identity=_identity(),
            )
            before = await store.load(session_id)
            assert before is not None
            assert before.labels == labels
            assert sessions_module.session_user_metadata(before.metadata) == exact
            with pytest.raises(DurableValueError) as metadata_error:
                await store.update_metadata(session_id, {"value": exact["value"] + "x"})
            assert metadata_error.value.limit == limit
            assert metadata_error.value.observed_lower_bound == limit + 1
            with pytest.raises(DurableValueError) as labels_error:
                await store.update_labels(session_id, {**labels, "extra": "value"})
            assert labels_error.value.dimension == "entries"
            assert labels_error.value.limit == 200
            after = await store.load(session_id)
            assert after == before
            assert await store.load_events(session_id) == []

            # Post-construction mutation must not bypass the direct store boundary.
            invalid = RunRequest(
                agent_name="assistant",
                session_id="oversize-create",
                messages=[Message.text("user", "create")],
            )
            invalid.metadata = {"value": exact["value"] + "x"}
            with pytest.raises((DurableValueError, ValidationError)):
                await store.create(invalid, identity=_identity())
            assert await store.load("oversize-create") is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_revalidates_all_mutable_durable_inputs_atomically(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            poisoned_create = RunRequest(
                agent_name="assistant",
                session_id="sess_poisoned_create",
                messages=[
                    Message.tool_call(
                        tool_call_id="call_create",
                        tool_name="echo",
                        arguments={"safe": True},
                    )
                ],
            )
            create_part = poisoned_create.messages[0].content[0]
            assert isinstance(create_part, ToolCallPart)
            create_part.arguments["bad"] = float("nan")
            with pytest.raises((DurableValueError, ValidationError)) as invalid_create:
                await store.create(poisoned_create, identity=_identity())
            _assert_durable_error(invalid_create.value, "non_finite_number")
            assert await store.load("sess_poisoned_create") is None

            session_id = "sess_portable_revalidation"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create")],
                    metadata={"stable": True},
                ),
                identity=_identity(),
            )
            await store.checkpoint(session_id, {"stable": True})

            good_event = Event(
                id="portable-good-event",
                type=EventType.SESSION_STARTED,
                session_id=session_id,
                payload={"safe": True},
            )
            poisoned_event = Event(
                id="portable-bad-event",
                type=EventType.SESSION_STARTED,
                session_id=session_id,
                payload={"safe": True},
            )
            poisoned_event.payload["bad"] = float("inf")
            with pytest.raises((DurableValueError, ValidationError)) as invalid_event:
                await store.append_events(session_id, [good_event, poisoned_event])
            _assert_durable_error(invalid_event.value, "non_finite_number")
            assert await store.load_events(session_id) == []

            good_message = Message.text("assistant", "safe")
            poisoned_message = Message.tool_call(
                tool_call_id="call_transcript",
                tool_name="echo",
                arguments={"safe": True},
            )
            transcript_part = poisoned_message.content[0]
            assert isinstance(transcript_part, ToolCallPart)
            transcript_part.arguments["bad"] = "value\ud800"
            with pytest.raises((DurableValueError, ValidationError)) as invalid_transcript:
                await store.append_transcript_messages(
                    session_id,
                    [good_message, poisoned_message],
                )
            _assert_durable_error(invalid_transcript.value, "unicode_surrogate")
            assert await store.load_transcript(session_id) == []

            with pytest.raises(DurableValueError) as invalid_checkpoint:
                await store.checkpoint(
                    session_id,
                    {"stable": False, "nested": {"bad": MAX_DURABLE_JSON_INTEGER + 1}},
                )
            assert invalid_checkpoint.value.code == "integer_out_of_range"
            assert await store.load_checkpoint(session_id) == {"stable": True}

            with pytest.raises(DurableValueError) as invalid_metadata:
                await store.update_metadata(session_id, {"bad": "value\x00"})
            assert invalid_metadata.value.code == "nul_character"
            loaded = await store.load(session_id)
            assert loaded is not None
            assert sessions_module.session_user_metadata(loaded.metadata) == {"stable": True}

            queue_request = EnqueueSessionMessageRequest(
                session_id=session_id,
                idempotency_key="portable-queue",
                content="safe",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
            queue_request.content = "poisoned\x00content"
            with pytest.raises((DurableValueError, ValidationError)) as invalid_queue:
                await store.enqueue_session_message(queue_request)
            _assert_durable_error(invalid_queue.value, "nul_character")
            assert await store.load_events(session_id) == []
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("invalid_text", "code"),
    [
        ("workload-secret-value\x00", "nul_character"),
        ("workload-secret-value\ud800", "unicode_surrogate"),
    ],
)
def test_session_store_conformance_rejects_nonportable_identifiers_and_query_text(
    session_store_case,
    invalid_text: str,
    code: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_portable_text_boundary"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create")],
                ),
                identity=_identity(),
            )

            with pytest.raises(DurableValueError) as invalid_identifier:
                await store.update_metadata(invalid_text, {"mutated": True})
            assert invalid_identifier.value.code == code
            assert "workload-secret-value" not in str(invalid_identifier.value)

            forged_query = SessionQuery(q="safe")
            forged_query.q = invalid_text
            with pytest.raises(ValidationError) as invalid_query:
                await store.list_sessions(forged_query)
            _assert_durable_error(invalid_query.value, code)
            assert "workload-secret-value" not in str(invalid_query.value)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_forged_out_of_range_query_cursors(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_query = SessionQuery()
            session_query.offset = MAX_DURABLE_JSON_INTEGER + 1
            with pytest.raises(ValidationError):
                await store.list_sessions(session_query)

            event_query = EventQuery(session_id="sess_portable_integer_boundary")
            event_query.after_sequence = MAX_DURABLE_JSON_INTEGER + 1
            with pytest.raises(ValidationError):
                await store.query_events(event_query)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_preserves_exact_portable_number_representation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_portable_numbers"
            request = RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create")],
                metadata={"numbers": {"safe": True}},
            )
            request.metadata["numbers"] = _portable_number_probe()
            await store.create(request, identity=_identity())

            event = Event(
                id="portable-number-event",
                type=EventType.SESSION_STARTED,
                session_id=session_id,
                payload={"numbers": {"safe": True}},
            )
            event.payload["numbers"] = _portable_number_probe()
            await store.append_events(session_id, [event])

            message = Message.tool_call(
                tool_call_id="portable-number-call",
                tool_name="echo",
                arguments={"numbers": {"safe": True}},
            )
            message_part = message.content[0]
            assert isinstance(message_part, ToolCallPart)
            message_part.arguments["numbers"] = _portable_number_probe()
            await store.append_transcript_messages(session_id, [message])
            await store.checkpoint(session_id, {"numbers": _portable_number_probe()})

            store = await _reopen_store(session_store_case, store)
            loaded = await store.load(session_id)
            assert loaded is not None
            _assert_portable_number_probe(loaded.metadata["numbers"])

            events = await store.load_events(session_id)
            assert len(events) == 1
            _assert_portable_number_probe(events[0].payload["numbers"])

            transcript = await store.load_transcript(session_id)
            assert len(transcript) == 1
            loaded_part = transcript[0].content[0]
            assert isinstance(loaded_part, ToolCallPart)
            _assert_portable_number_probe(loaded_part.arguments["numbers"])

            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            _assert_portable_number_probe(checkpoint["numbers"])
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_preserves_hosted_search_evidence(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"hosted-search-evidence-{session_store_case[0]}"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "search")],
                ),
                identity=_identity(),
            )
            message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    HostedToolCallPart(
                        call_id="ws_1",
                        status="completed",
                        action=WebSearchAction(
                            type="search",
                            query="Cayu",
                            sources=(
                                WebSearchAPISource(name="redacted"),
                                WebSearchSource(
                                    url="https://github.com/example/cayu",
                                    title="Cayu",
                                ),
                            ),
                        ),
                        provider_name="openai",
                        model="gpt-test",
                        model_step_id="mstep_00000000000000000000000000000000",
                        model_attempt_id="matt_00000000000000000000000000000000",
                    ),
                    Message.text("assistant", "Cayu").content[0],
                    CitationPart(
                        url="https://github.com/example/cayu",
                        title="Cayu",
                        start_index=0,
                        end_index=4,
                        provenance=CitationProvenance(provider_name="openai"),
                        model_step_id="mstep_00000000000000000000000000000000",
                        model_attempt_id="matt_00000000000000000000000000000000",
                    ),
                ],
            )
            await store.append_transcript_messages(session_id, [message])

            store = await _reopen_store(session_store_case, store)

            assert await store.load_transcript(session_id) == [message]
            stream = io.StringIO()
            assert await export_sessions(store, stream=stream) == 1
            imported = list(import_sessions(io.StringIO(stream.getvalue())))
            assert len(imported) == 1
            assert imported[0].transcript == [message]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_sqlite_jsonl_to_postgres_preserves_exact_portable_number_representation(
    tmp_path,
    conformance_postgres_dsn,
) -> None:
    async def run() -> None:
        session_id = "sess_sqlite_postgres_portable_numbers"
        sqlite_store = SQLiteSessionStore(tmp_path / "portable-export.sqlite")
        try:
            await sqlite_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "export")],
                    metadata={"numbers": _portable_number_probe()},
                ),
                identity=_identity(),
            )
            await sqlite_store.append_events(
                session_id,
                [
                    Event(
                        id="sqlite-postgres-portable-event",
                        type=EventType.SESSION_STARTED,
                        session_id=session_id,
                        payload={"numbers": _portable_number_probe()},
                    )
                ],
            )
            await sqlite_store.append_transcript_messages(
                session_id,
                [
                    Message.tool_call(
                        tool_call_id="sqlite-postgres-portable-call",
                        tool_name="echo",
                        arguments={"numbers": _portable_number_probe()},
                    )
                ],
            )
            await sqlite_store.checkpoint(
                session_id,
                {"numbers": _portable_number_probe()},
            )

            stream = io.StringIO()
            assert await export_sessions(sqlite_store, stream=stream) == 1
            imported_records = list(import_sessions(io.StringIO(stream.getvalue())))
            assert len(imported_records) == 1
            imported = imported_records[0]
        finally:
            await _close_store(sqlite_store)

        _assert_portable_number_probe(imported.session.metadata["numbers"])
        _assert_portable_number_probe(imported.events[0].payload["numbers"])
        imported_part = imported.transcript[0].content[0]
        assert isinstance(imported_part, ToolCallPart)
        _assert_portable_number_probe(imported_part.arguments["numbers"])
        assert imported.checkpoint is not None
        _assert_portable_number_probe(imported.checkpoint["numbers"])

        await _reset_postgres_data(conformance_postgres_dsn)
        postgres_store = _new_postgres_store(conformance_postgres_dsn)
        try:
            source = imported.session
            await postgres_store.create(
                RunRequest(
                    agent_name=source.agent_name,
                    session_id=source.id,
                    parent_session_id=source.parent_session_id,
                    causal_budget_id=source.causal_budget_id,
                    target=ModelTarget(
                        provider_name=source.provider_name,
                        model=source.model,
                    ),
                    environment_name=source.environment_name,
                    messages=[],
                    labels=source.labels,
                    metadata=sessions_module.session_user_metadata(source.metadata),
                ),
                identity=SessionIdentity(
                    provider_name=source.provider_name,
                    model=source.model,
                    runtime_name=source.runtime_name,
                    runtime_version=source.runtime_version,
                    runtime_build_provenance=source.runtime_build_provenance,
                ),
            )
            await postgres_store.append_events(source.id, imported.events)
            await postgres_store.append_transcript_messages(source.id, imported.transcript)
            await postgres_store.checkpoint(source.id, imported.checkpoint)

            postgres_store = await _reopen_store(
                ("postgres", tmp_path, conformance_postgres_dsn),
                postgres_store,
            )
            restored = await postgres_store.load(source.id)
            assert restored is not None
            assert restored.runtime_build_provenance == source.runtime_build_provenance
            _assert_portable_number_probe(restored.metadata["numbers"])
            restored_events = await postgres_store.load_events(source.id)
            _assert_portable_number_probe(restored_events[0].payload["numbers"])
            restored_transcript = await postgres_store.load_transcript(source.id)
            restored_part = restored_transcript[0].content[0]
            assert isinstance(restored_part, ToolCallPart)
            _assert_portable_number_probe(restored_part.arguments["numbers"])
            restored_checkpoint = await postgres_store.load_checkpoint(source.id)
            assert restored_checkpoint is not None
            _assert_portable_number_probe(restored_checkpoint["numbers"])
        finally:
            await _close_store(postgres_store)

    asyncio.run(run())
