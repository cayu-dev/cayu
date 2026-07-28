from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import UTC, datetime
from typing import TypeVar

import pytest

from cayu.core.events import Event, EventType
from cayu.core.messages import ToolResultPart
from cayu.core.tools import ToolResult
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._tool_execution import tool_idempotency_key
from cayu.runtime._tool_round_publication import (
    PreparedToolRoundPublication,
    build_tool_round_publication_request,
    collect_tool_round_publication_evidence,
    prepare_tool_round_publication,
    publish_tool_round_publication,
)
from cayu.runtime._tool_round_recovery import (
    PENDING_TOOL_ROUND_CHECKPOINT_KEY,
    PendingToolRound,
    pending_tool_round_identity,
)
from cayu.runtime.approvals import PendingToolCallApproval
from cayu.runtime.budgets import InMemoryBudgetStore
from cayu.runtime.event_sinks import InMemoryEventSink
from cayu.runtime.sessions import (
    InMemorySessionStore,
    RunRequest,
    RuntimePublicationReceipt,
    RuntimePublicationRequest,
    RuntimePublicationResult,
    Session,
    SessionIdentity,
    SessionRuntimePublicationConflict,
    SessionStatus,
)
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
)


def _pending_round(*, structured: bool = False) -> PendingToolRound:
    first_tool_name = STRUCTURED_OUTPUT_TOOL_NAME if structured else "lookup"
    return PendingToolRound(
        model_step_id=f"mstep_{'1' * 32}",
        model_attempt_id=f"matt_{'2' * 32}",
        tool_round_id=f"tround_{'3' * 32}",
        agent_name="assistant",
        environment_name=None,
        task_id="task-1",
        tool_calls=[
            PendingToolCallApproval(
                tool_call_id="call-a",
                tool_name=first_tool_name,
                arguments={"query": "alpha"},
            ),
            PendingToolCallApproval(
                tool_call_id="call-b",
                tool_name="write",
                arguments={"value": 2},
            ),
        ],
        structured_output=(
            StructuredOutputSpec(json_schema={"type": "object"}) if structured else None
        ),
    )


def _source_checkpoint(pending_round: PendingToolRound) -> dict:
    return {
        "unrelated": {"keep": True},
        PENDING_TOOL_ROUND_CHECKPOINT_KEY: pending_round.model_dump(mode="json"),
    }


def _lifecycle_events(
    pending_round: PendingToolRound,
    *,
    session_id: str = "session-1",
) -> list[Event]:
    first_call, second_call = pending_round.tool_calls
    identity_payload = pending_tool_round_identity(pending_round).payload()
    first_key = tool_idempotency_key(
        session_id=session_id,
        tool_round_id=pending_round.tool_round_id,
        tool_call_id=first_call.tool_call_id,
    )
    second_key = tool_idempotency_key(
        session_id=session_id,
        tool_round_id=pending_round.tool_round_id,
        tool_call_id=second_call.tool_call_id,
    )
    started = Event(
        id="started-a",
        type=EventType.TOOL_CALL_STARTED,
        session_id=session_id,
        agent_name=pending_round.agent_name,
        environment_name=pending_round.environment_name,
        tool_name=first_call.tool_name,
        payload={
            **identity_payload,
            "tool_call_id": first_call.tool_call_id,
            "idempotency_key": first_key,
            "arguments": first_call.arguments,
        },
    )
    completed = Event(
        id="terminal-a",
        type=EventType.TOOL_CALL_COMPLETED,
        session_id=session_id,
        agent_name=pending_round.agent_name,
        environment_name=pending_round.environment_name,
        tool_name=first_call.tool_name,
        payload={
            **identity_payload,
            "tool_call_id": first_call.tool_call_id,
            "idempotency_key": first_key,
            "result": ToolResult(
                content="alpha result",
                structured={"found": True},
            ).model_dump(mode="json"),
        },
    )
    failed = Event(
        id="terminal-b",
        type=EventType.TOOL_CALL_FAILED,
        session_id=session_id,
        agent_name=pending_round.agent_name,
        environment_name=pending_round.environment_name,
        tool_name=second_call.tool_name,
        payload={
            **identity_payload,
            "tool_call_id": second_call.tool_call_id,
            "idempotency_key": second_key,
            "result": ToolResult(
                content="write failed",
                structured={"retryable": False},
                is_error=True,
            ).model_dump(mode="json"),
        },
    )
    # Deliberately scramble discovery order. Publication order must come only
    # from the durable pending-round call order.
    return [failed, completed, started]


def _structured_pending_round(
    *,
    max_retries: int = 2,
    valid: bool = True,
) -> PendingToolRound:
    model_step_id = f"mstep_{'4' * 32}"
    return PendingToolRound(
        model_step_id=model_step_id,
        model_attempt_id=f"matt_{'5' * 32}",
        tool_round_id=f"tround_{'6' * 32}",
        agent_name="assistant",
        environment_name=None,
        task_id="structured-task-1",
        source_model_step_id=model_step_id,
        source_transcript_cursor=0,
        model_step=3,
        structured_output_attempt=1,
        tool_calls=[
            PendingToolCallApproval(
                tool_call_id="call-final",
                tool_name=STRUCTURED_OUTPUT_TOOL_NAME,
                arguments={"output": {"answer": "ok" if valid else 7}},
            )
        ],
        structured_output=StructuredOutputSpec(
            name="answer",
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            max_retries=max_retries,
        ),
    )


def _structured_lifecycle_events(
    pending_round: PendingToolRound,
    *,
    valid: bool,
    session_id: str = "session-1",
) -> list[Event]:
    pending_call = pending_round.tool_calls[0]
    identity_payload = pending_tool_round_identity(pending_round).payload()
    result = ToolResult(
        content=(
            "Structured output accepted."
            if valid
            else "Structured output rejected: invalid answer."
        ),
        structured=(
            {"output": {"answer": "ok"}}
            if valid
            else {
                "structured_output_errors": [
                    {
                        "path": "$.answer",
                        "message": "invalid answer",
                        "schema_path": "$.properties.answer.type",
                    }
                ]
            }
        ),
        is_error=not valid,
    )
    return [
        Event(
            id="structured-terminal",
            type=(EventType.TOOL_CALL_COMPLETED if valid else EventType.TOOL_CALL_FAILED),
            session_id=session_id,
            agent_name=pending_round.agent_name,
            environment_name=pending_round.environment_name,
            tool_name=pending_call.tool_name,
            payload={
                **identity_payload,
                "tool_call_id": pending_call.tool_call_id,
                "idempotency_key": tool_idempotency_key(
                    session_id=session_id,
                    tool_round_id=pending_round.tool_round_id,
                    tool_call_id=pending_call.tool_call_id,
                ),
                "result": result.model_dump(mode="json"),
            },
        )
    ]


def _structured_auxiliary_events(
    pending_round: PendingToolRound,
    *,
    valid: bool,
    retry_scheduled: bool,
    session_id: str = "session-1",
    step: int = 3,
    attempt: int = 1,
) -> list[Event]:
    spec = pending_round.structured_output
    assert spec is not None
    common_payload = {
        **pending_tool_round_identity(pending_round).payload(),
        "name": spec.name,
        "step": step,
        "attempt": attempt,
        "max_retries": spec.max_retries,
    }
    events = [
        Event(
            id="structured-validating",
            type=EventType.STRUCTURED_OUTPUT_VALIDATING,
            session_id=session_id,
            agent_name=pending_round.agent_name,
            environment_name=pending_round.environment_name,
            payload={
                **common_payload,
                "strategy": "tool",
            },
        ),
        Event(
            id=("structured-validated" if valid else "structured-failed"),
            type=(
                EventType.STRUCTURED_OUTPUT_VALIDATED
                if valid
                else EventType.STRUCTURED_OUTPUT_FAILED
            ),
            session_id=session_id,
            agent_name=pending_round.agent_name,
            environment_name=pending_round.environment_name,
            payload={
                **common_payload,
                "valid": valid,
                "errors": (
                    []
                    if valid
                    else [
                        {
                            "path": "$.answer",
                            "message": "invalid answer",
                            "schema_path": "$.properties.answer.type",
                        }
                    ]
                ),
                **({"output": {"answer": "ok"}} if valid else {}),
            },
        ),
    ]
    if retry_scheduled:
        events.append(
            Event(
                id="structured-retry",
                type=EventType.STRUCTURED_OUTPUT_RETRY,
                session_id=session_id,
                agent_name=pending_round.agent_name,
                environment_name=pending_round.environment_name,
                payload={
                    **common_payload,
                    "valid": False,
                    "errors": [
                        {
                            "path": "$.answer",
                            "message": "invalid answer",
                            "schema_path": "$.properties.answer.type",
                        }
                    ],
                },
            )
        )
    return events


class _StructuredOutputPublicationExtension:
    def __init__(
        self,
        *,
        events: list[Event],
        valid: bool,
        retry_scheduled: bool,
        step: int = 3,
        attempt: int = 1,
    ) -> None:
        self._events = tuple(events)
        self._valid = valid
        self._retry_scheduled = retry_scheduled
        self._step = step
        self._attempt = attempt

    def build_request(self, *, ordinary_request, pending_round):
        assert pending_round.structured_output is not None
        return RuntimePublicationRequest(
            publication_id=ordinary_request.publication_id,
            kind=ordinary_request.kind,
            intent={
                **ordinary_request.intent,
                "auxiliary": {
                    "schema_version": 1,
                    "kind": "structured-output-validation",
                    "step": self._step,
                    "attempt": self._attempt,
                    "valid": self._valid,
                    "retry_scheduled": self._retry_scheduled,
                    "event_ids": [event.id for event in self._events],
                },
            },
            mutation=ordinary_request.mutation,
            transcript_messages=ordinary_request.transcript_messages,
            events=self._events,
            referenced_events=ordinary_request.referenced_events,
        )


def test_collect_evidence_orders_by_pending_round_and_requires_exact_identity() -> None:
    pending_round = _pending_round()
    evidence = collect_tool_round_publication_evidence(
        session_id="session-1",
        pending_round=pending_round,
        durable_events=_lifecycle_events(pending_round),
    )

    assert [event.id for event in evidence.lifecycle_events] == [
        "started-a",
        "terminal-a",
        "terminal-b",
    ]
    assert [event.id for event in evidence.terminal_events] == [
        "terminal-a",
        "terminal-b",
    ]
    assert [outcome.call.id for outcome in evidence.outcomes] == ["call-a", "call-b"]
    assert [outcome.result.is_error for outcome in evidence.outcomes] == [False, True]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda events: events[1:],
            "exactly one terminal event",
        ),
        (
            lambda events: [
                *events,
                events[0].model_copy(update={"id": "duplicate-terminal"}),
            ],
            "duplicate durable terminal",
        ),
        (
            lambda events: [
                events[0].model_copy(
                    update={
                        "payload": {
                            **events[0].payload,
                            "tool_round_id": "another-round",
                        }
                    }
                ),
                *events[1:],
            ],
            "conflicting execution identity",
        ),
        (
            lambda events: [
                *events[:-1],
                events[-1].model_copy(
                    update={
                        "payload": {
                            **events[-1].payload,
                            "arguments": {"query": "changed"},
                        }
                    }
                ),
            ],
            "arguments that conflict",
        ),
        (
            lambda events: [
                events[0].model_copy(
                    update={
                        "payload": {
                            **events[0].payload,
                            "idempotency_key": "wrong",
                        }
                    }
                ),
                *events[1:],
            ],
            "conflicting idempotency key",
        ),
    ],
)
def test_collect_evidence_fails_closed_on_incomplete_or_conflicting_material(
    mutate,
    message: str,
) -> None:
    pending_round = _pending_round()

    with pytest.raises(ValueError, match=message):
        collect_tool_round_publication_evidence(
            session_id="session-1",
            pending_round=pending_round,
            durable_events=mutate(_lifecycle_events(pending_round)),
        )


def test_build_request_binds_exact_marker_grouped_results_and_durable_events() -> None:
    pending_round = _pending_round()
    source_checkpoint = _source_checkpoint(pending_round)
    request = build_tool_round_publication_request(
        session_id="session-1",
        pending_round=pending_round,
        source_checkpoint=source_checkpoint,
        durable_events=_lifecycle_events(pending_round),
    )

    assert request.publication_id == f"tool-round:{pending_round.tool_round_id}"
    assert request.kind == "tool-round"
    assert request.intent["round_id"] == pending_round.tool_round_id
    assert request.intent["tool_call_ids"] == ["call-a", "call-b"]
    assert request.events == ()
    assert [reference.event_id for reference in request.referenced_events] == [
        "started-a",
        "terminal-a",
        "terminal-b",
    ]
    assert len(request.mutation.operations) == 1
    marker_delete = request.mutation.operations[0]
    assert marker_delete.key == PENDING_TOOL_ROUND_CHECKPOINT_KEY
    assert marker_delete.action == "delete"
    assert marker_delete.expected_value_digest == request.intent["pending_round_digest"]
    assert len(request.transcript_messages) == 1
    result_parts = [
        part for part in request.transcript_messages[0].content if type(part) is ToolResultPart
    ]
    assert len(result_parts) == len(request.transcript_messages[0].content)
    assert [part.tool_call_id for part in result_parts] == ["call-a", "call-b"]


def test_build_request_rejects_durable_marker_drift() -> None:
    pending_round = _pending_round()
    source_checkpoint = _source_checkpoint(pending_round)
    source_checkpoint[PENDING_TOOL_ROUND_CHECKPOINT_KEY]["tool_calls"][0]["arguments"] = {
        "query": "changed"
    }

    with pytest.raises(ValueError, match="conflicts with the durable"):
        build_tool_round_publication_request(
            session_id="session-1",
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=_lifecycle_events(pending_round),
        )


def test_structured_output_requires_explicit_nonweakening_extension() -> None:
    pending_round = _structured_pending_round()
    events = _structured_lifecycle_events(pending_round, valid=True)

    with pytest.raises(ValueError, match="explicit publication extension"):
        build_tool_round_publication_request(
            session_id="session-1",
            pending_round=pending_round,
            source_checkpoint=_source_checkpoint(pending_round),
            durable_events=events,
        )

    auxiliary_events = _structured_auxiliary_events(
        pending_round,
        valid=True,
        retry_scheduled=False,
    )

    request = build_tool_round_publication_request(
        session_id="session-1",
        pending_round=pending_round,
        source_checkpoint=_source_checkpoint(pending_round),
        durable_events=events,
        extension=_StructuredOutputPublicationExtension(
            events=auxiliary_events,
            valid=True,
            retry_scheduled=False,
        ),
    )
    assert request.events == tuple(auxiliary_events)
    assert request.intent["auxiliary"] == {
        "schema_version": 1,
        "kind": "structured-output-validation",
        "step": 3,
        "attempt": 1,
        "valid": True,
        "retry_scheduled": False,
        "event_ids": ["structured-validating", "structured-validated"],
    }

    class WeakeningExtension:
        def build_request(self, *, ordinary_request, pending_round):
            del pending_round
            return RuntimePublicationRequest(
                publication_id=ordinary_request.publication_id,
                kind=ordinary_request.kind,
                intent=ordinary_request.intent,
                mutation=ordinary_request.mutation,
                transcript_messages=(),
                events=(),
                referenced_events=ordinary_request.referenced_events,
            )

    with pytest.raises(ValueError, match="grouped result message"):
        build_tool_round_publication_request(
            session_id="session-1",
            pending_round=pending_round,
            source_checkpoint=_source_checkpoint(pending_round),
            durable_events=events,
            extension=WeakeningExtension(),
        )


_StoreT = TypeVar("_StoreT", bound=InMemorySessionStore)


async def _created_store(
    store: _StoreT,
    *,
    session_id: str,
) -> tuple[_StoreT, Session]:
    session = await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    return store, session


def test_publish_commits_once_and_exactly_replays() -> None:
    async def scenario() -> None:
        store, session = await _created_store(
            InMemorySessionStore(),
            session_id="session-1",
        )
        pending_round = _pending_round()
        source_checkpoint = _source_checkpoint(pending_round)
        events = _lifecycle_events(pending_round)
        await store.checkpoint(session.id, source_checkpoint)
        await store.append_events(session.id, events)
        prepared = prepare_tool_round_publication(
            session_id=session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=events,
            expected_statuses={SessionStatus.PENDING},
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[],
        )

        published = await publish_tool_round_publication(
            prepared,
            session_store=store,
            event_writer=writer,
        )
        replayed = await publish_tool_round_publication(
            prepared,
            session_store=store,
            event_writer=writer,
        )

        assert published.replayed is False
        assert replayed.replayed is True
        assert replayed.receipt == published.receipt
        assert await store.load_checkpoint(session.id) == {"unrelated": {"keep": True}}
        transcript = await store.load_transcript(session.id)
        assert len(transcript) == 1
        result_parts = [part for part in transcript[0].content if type(part) is ToolResultPart]
        assert len(result_parts) == len(transcript[0].content)
        assert [part.tool_call_id for part in result_parts] == ["call-a", "call-b"]

    asyncio.run(scenario())


def test_store_rejects_a_durable_terminal_omitted_by_the_caller() -> None:
    async def scenario() -> None:
        store, session = await _created_store(
            InMemorySessionStore(),
            session_id="session-1",
        )
        pending_round = _pending_round()
        source_checkpoint = _source_checkpoint(pending_round)
        events = _lifecycle_events(pending_round)
        contradictory_terminal = events[0].model_copy(update={"id": "terminal-b-conflict"})
        await store.checkpoint(session.id, source_checkpoint)
        await store.append_events(session.id, [*events, contradictory_terminal])
        prepared = prepare_tool_round_publication(
            session_id=session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=events,
            expected_statuses={SessionStatus.PENDING},
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[],
        )

        with pytest.raises(
            SessionRuntimePublicationConflict,
            match="every durable lifecycle event",
        ):
            await publish_tool_round_publication(
                prepared,
                session_store=store,
                event_writer=writer,
            )
        assert await store.load_checkpoint(session.id) == source_checkpoint
        assert await store.load_transcript(session.id) == []

    asyncio.run(scenario())


def test_store_scopes_reused_tool_call_ids_to_the_exact_round() -> None:
    async def scenario() -> None:
        store, session = await _created_store(
            InMemorySessionStore(),
            session_id="session-1",
        )
        pending_round = _pending_round()
        source_checkpoint = _source_checkpoint(pending_round)
        current_events = _lifecycle_events(pending_round)
        old_identity = {
            "model_step_id": f"mstep_{'a' * 32}",
            "model_attempt_id": f"matt_{'b' * 32}",
            "tool_round_id": f"tround_{'c' * 32}",
        }
        old_events = [
            Event(
                id=f"old-{event.id}",
                type=event.type,
                session_id=event.session_id,
                agent_name=event.agent_name,
                environment_name=event.environment_name,
                tool_name=event.tool_name,
                payload={
                    **event.payload,
                    **old_identity,
                },
            )
            for event in current_events
        ]
        await store.checkpoint(session.id, source_checkpoint)
        await store.append_events(session.id, [*old_events, *current_events])
        prepared = prepare_tool_round_publication(
            session_id=session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=current_events,
            expected_statuses={SessionStatus.PENDING},
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[],
        )

        result = await publish_tool_round_publication(
            prepared,
            session_store=store,
            event_writer=writer,
        )

        assert result.replayed is False
        assert len(await store.load_transcript(session.id)) == 1

    asyncio.run(scenario())


def test_store_rejects_ambiguous_roundless_lifecycle_evidence() -> None:
    async def scenario() -> None:
        store, session = await _created_store(
            InMemorySessionStore(),
            session_id="session-1",
        )
        pending_round = _pending_round()
        source_checkpoint = _source_checkpoint(pending_round)
        current_events = _lifecycle_events(pending_round)
        ambiguous_event = Event(
            id="roundless-started",
            type=EventType.TOOL_CALL_STARTED,
            session_id=session.id,
            agent_name=pending_round.agent_name,
            tool_name=pending_round.tool_calls[0].tool_name,
            payload={
                "tool_call_id": pending_round.tool_calls[0].tool_call_id,
                "idempotency_key": "legacy-roundless-key",
                "arguments": pending_round.tool_calls[0].arguments,
            },
        )
        await store.checkpoint(session.id, source_checkpoint)
        await store.append_events(
            session.id,
            [ambiguous_event, *current_events],
        )
        prepared = prepare_tool_round_publication(
            session_id=session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=current_events,
            expected_statuses={SessionStatus.PENDING},
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[],
        )

        with pytest.raises(ValueError, match="requires a complete valid execution identity"):
            await publish_tool_round_publication(
                prepared,
                session_store=store,
                event_writer=writer,
            )
        assert await store.load_checkpoint(session.id) == source_checkpoint
        assert await store.load_transcript(session.id) == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("valid", "retry_scheduled", "expected_event_types"),
    [
        (
            True,
            False,
            [
                EventType.STRUCTURED_OUTPUT_VALIDATING,
                EventType.STRUCTURED_OUTPUT_VALIDATED,
            ],
        ),
        (
            False,
            False,
            [
                EventType.STRUCTURED_OUTPUT_VALIDATING,
                EventType.STRUCTURED_OUTPUT_FAILED,
            ],
        ),
        (
            False,
            True,
            [
                EventType.STRUCTURED_OUTPUT_VALIDATING,
                EventType.STRUCTURED_OUTPUT_FAILED,
                EventType.STRUCTURED_OUTPUT_RETRY,
            ],
        ),
    ],
)
def test_store_accepts_exact_structured_output_auxiliary_sequences(
    valid: bool,
    retry_scheduled: bool,
    expected_event_types: list[EventType],
) -> None:
    async def scenario() -> None:
        store, session = await _created_store(
            InMemorySessionStore(),
            session_id="session-1",
        )
        pending_round = _structured_pending_round(valid=valid)
        source_checkpoint = _source_checkpoint(pending_round)
        lifecycle_events = _structured_lifecycle_events(
            pending_round,
            valid=valid,
            session_id=session.id,
        )
        auxiliary_events = _structured_auxiliary_events(
            pending_round,
            valid=valid,
            retry_scheduled=retry_scheduled,
            session_id=session.id,
        )
        await store.checkpoint(session.id, source_checkpoint)
        await store.append_events(session.id, lifecycle_events)
        prepared = prepare_tool_round_publication(
            session_id=session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=lifecycle_events,
            expected_statuses={SessionStatus.PENDING},
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
            extension=_StructuredOutputPublicationExtension(
                events=auxiliary_events,
                valid=valid,
                retry_scheduled=retry_scheduled,
            ),
        )
        sink = InMemoryEventSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[sink],
        )

        first = await publish_tool_round_publication(
            prepared,
            session_store=store,
            event_writer=writer,
        )
        replay = await publish_tool_round_publication(
            prepared,
            session_store=store,
            event_writer=writer,
        )

        assert first.replayed is False
        assert replay.replayed is True
        assert await store.load_checkpoint(session.id) == {"unrelated": {"keep": True}}
        assert [event.type for event in sink.events] == expected_event_types
        persisted = await store.load_events(session.id)
        assert [event.type for event in persisted[-len(auxiliary_events) :]] == (
            expected_event_types
        )
        assert len(await store.load_transcript(session.id)) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("ordinary_events", "without a validated intent.auxiliary"),
        ("auxiliary_fields", "invalid fields"),
        ("event_order", "event ids and order"),
        ("wrong_round", "conflicting round id"),
        ("wrong_step", "conflicting step"),
        ("wrong_attempt", "conflicting attempt"),
        ("wrong_valid", "conflicts with intent.auxiliary.valid"),
        ("retry_sequence", "invalid type sequence"),
        ("marker_no_reserved_tool", "requires the reserved tool"),
        ("marker_no_config", "requires durable structured-output config"),
        ("marker_name", "conflicts with the durable output name"),
        ("marker_agent", "pending-round identity"),
        ("marker_step", "step conflicts with the durable pending marker"),
        ("marker_attempt", "attempt conflicts with the durable pending marker"),
        ("retry_limit", "retry exceeds the durable retry policy"),
    ],
)
def test_store_rejects_malformed_structured_output_auxiliary_publications(
    case: str,
    message: str,
) -> None:
    async def scenario() -> None:
        store, session = await _created_store(
            InMemorySessionStore(),
            session_id="session-1",
        )
        valid = case not in {"retry_sequence", "retry_limit"}
        retry_scheduled = case in {"retry_sequence", "retry_limit"}
        pending_round = _structured_pending_round(valid=valid)
        attempt = 3 if case == "retry_limit" else 1
        source_checkpoint = _source_checkpoint(pending_round)
        checkpoint = deepcopy(source_checkpoint)
        lifecycle_events = _structured_lifecycle_events(
            pending_round,
            valid=valid,
            session_id=session.id,
        )
        auxiliary_events = _structured_auxiliary_events(
            pending_round,
            valid=valid,
            retry_scheduled=retry_scheduled,
            session_id=session.id,
            attempt=attempt,
        )
        request = build_tool_round_publication_request(
            session_id=session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=lifecycle_events,
            extension=_StructuredOutputPublicationExtension(
                events=auxiliary_events,
                valid=valid,
                retry_scheduled=retry_scheduled,
                attempt=attempt,
            ),
        )

        intent = deepcopy(request.intent)
        events = list(request.events)
        if case == "ordinary_events":
            intent.pop("auxiliary")
        elif case == "auxiliary_fields":
            intent["auxiliary"]["unexpected"] = True
        elif case == "event_order":
            events.reverse()
        elif case == "wrong_round":
            events[0] = events[0].model_copy(
                update={
                    "payload": {
                        **events[0].payload,
                        "tool_round_id": "another-round",
                    }
                }
            )
        elif case == "wrong_step":
            events[1] = events[1].model_copy(
                update={
                    "payload": {
                        **events[1].payload,
                        "step": 4,
                    }
                }
            )
        elif case == "wrong_attempt":
            events[1] = events[1].model_copy(
                update={
                    "payload": {
                        **events[1].payload,
                        "attempt": 2,
                    }
                }
            )
        elif case == "wrong_valid":
            events[1] = events[1].model_copy(
                update={
                    "payload": {
                        **events[1].payload,
                        "valid": not valid,
                    }
                }
            )
        elif case == "retry_sequence":
            events.pop()
            intent["auxiliary"]["event_ids"] = [event.id for event in events]
        elif case == "marker_no_reserved_tool":
            checkpoint[PENDING_TOOL_ROUND_CHECKPOINT_KEY]["tool_calls"][0]["tool_name"] = "lookup"
        elif case == "marker_no_config":
            checkpoint[PENDING_TOOL_ROUND_CHECKPOINT_KEY]["structured_output"] = None
        elif case == "marker_name":
            checkpoint[PENDING_TOOL_ROUND_CHECKPOINT_KEY]["structured_output"]["name"] = (
                "another-name"
            )
        elif case == "marker_agent":
            checkpoint[PENDING_TOOL_ROUND_CHECKPOINT_KEY]["agent_name"] = "another-agent"
        elif case == "marker_step":
            checkpoint[PENDING_TOOL_ROUND_CHECKPOINT_KEY]["model_step"] = 4
        elif case == "marker_attempt":
            checkpoint[PENDING_TOOL_ROUND_CHECKPOINT_KEY]["structured_output_attempt"] = 2

        malformed_request = RuntimePublicationRequest(
            publication_id=request.publication_id,
            kind=request.kind,
            intent=intent,
            mutation=request.mutation,
            transcript_messages=request.transcript_messages,
            events=tuple(events),
            referenced_events=request.referenced_events,
        )
        await store.checkpoint(session.id, checkpoint)
        await store.append_events(session.id, lifecycle_events)

        with pytest.raises(ValueError, match=message):
            await store.publish_runtime_publication(
                session.id,
                request=malformed_request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=session.run_epoch,
                expected_transcript_cursor=0,
            )
        assert await store.load_checkpoint(session.id) == checkpoint
        assert await store.load_transcript(session.id) == []
        assert await store.load_events(session.id) == lifecycle_events

    asyncio.run(scenario())


class _LostPublicationAcknowledgementStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.request_identities: list[int] = []
        self._lose_next_acknowledgement = True

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        self.request_identities.append(id(request))
        result = await super().publish_runtime_publication(
            session_id,
            request=request,
            **kwargs,
        )
        if self._lose_next_acknowledgement:
            self._lose_next_acknowledgement = False
            raise RuntimeError("publication acknowledgement lost")
        return result


def test_ambiguous_acknowledgement_reuses_exact_prepared_request() -> None:
    async def scenario() -> None:
        store, session = await _created_store(
            _LostPublicationAcknowledgementStore(),
            session_id="session-1",
        )
        pending_round = _pending_round()
        source_checkpoint = _source_checkpoint(pending_round)
        events = _lifecycle_events(pending_round)
        await store.checkpoint(session.id, source_checkpoint)
        await store.append_events(session.id, events)
        prepared = prepare_tool_round_publication(
            session_id=session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=events,
            expected_statuses={SessionStatus.PENDING},
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[],
        )

        with pytest.raises(RuntimeError, match="acknowledgement lost"):
            await publish_tool_round_publication(
                prepared,
                session_store=store,
                event_writer=writer,
            )
        replayed = await publish_tool_round_publication(
            prepared,
            session_store=store,
            event_writer=writer,
        )

        assert replayed.replayed is True
        assert len(set(store.request_identities)) == 1
        assert len(await store.load_transcript(session.id)) == 1

    asyncio.run(scenario())


class _CommitThenBlockStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.committed = asyncio.Event()
        self.release_acknowledgement = asyncio.Event()
        self._block_next_acknowledgement = True

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        result = await super().publish_runtime_publication(
            session_id,
            request=request,
            **kwargs,
        )
        if self._block_next_acknowledgement:
            self._block_next_acknowledgement = False
            self.committed.set()
            await self.release_acknowledgement.wait()
        return result


def test_cancellation_waits_for_commit_and_retains_replayable_request() -> None:
    async def scenario() -> None:
        store, session = await _created_store(
            _CommitThenBlockStore(),
            session_id="session-1",
        )
        pending_round = _pending_round()
        source_checkpoint = _source_checkpoint(pending_round)
        events = _lifecycle_events(pending_round)
        await store.checkpoint(session.id, source_checkpoint)
        await store.append_events(session.id, events)
        prepared = prepare_tool_round_publication(
            session_id=session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=events,
            expected_statuses={SessionStatus.PENDING},
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[],
        )
        publication = asyncio.create_task(
            publish_tool_round_publication(
                prepared,
                session_store=store,
                event_writer=writer,
            )
        )
        await store.committed.wait()
        publication.cancel("caller stopped")
        store.release_acknowledgement.set()

        with pytest.raises(asyncio.CancelledError, match="caller stopped"):
            await publication
        replayed = await publish_tool_round_publication(
            prepared,
            session_store=store,
            event_writer=writer,
        )
        assert replayed.replayed is True
        assert len(await store.load_transcript(session.id)) == 1

    asyncio.run(scenario())


class _BlockBeforeCommitStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.publication_started = asyncio.Event()
        self.release_commit = asyncio.Event()
        self._block_next_commit = True

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        if self._block_next_commit:
            self._block_next_commit = False
            self.publication_started.set()
            await self.release_commit.wait()
        return await super().publish_runtime_publication(
            session_id,
            request=request,
            **kwargs,
        )


def test_cancel_before_commit_defers_cancellation_through_exactly_one_commit() -> None:
    async def scenario() -> None:
        store, session = await _created_store(
            _BlockBeforeCommitStore(),
            session_id="session-1",
        )
        pending_round = _pending_round()
        source_checkpoint = _source_checkpoint(pending_round)
        events = _lifecycle_events(pending_round)
        await store.checkpoint(session.id, source_checkpoint)
        await store.append_events(session.id, events)
        prepared = prepare_tool_round_publication(
            session_id=session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=events,
            expected_statuses={SessionStatus.PENDING},
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[],
        )
        publication = asyncio.create_task(
            publish_tool_round_publication(
                prepared,
                session_store=store,
                event_writer=writer,
            )
        )
        await store.publication_started.wait()
        publication.cancel("cancel before commit")
        await asyncio.sleep(0)
        assert await store.load_checkpoint(session.id) == source_checkpoint
        store.release_commit.set()

        with pytest.raises(asyncio.CancelledError, match="cancel before commit"):
            await publication
        assert await store.load_checkpoint(session.id) == {"unrelated": {"keep": True}}
        assert len(await store.load_transcript(session.id)) == 1
        replayed = await publish_tool_round_publication(
            prepared,
            session_store=store,
            event_writer=writer,
        )
        assert replayed.replayed is True
        assert len(await store.load_transcript(session.id)) == 1

    asyncio.run(scenario())


class _AuxiliaryPublicationStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self._published_at = datetime.now(UTC)

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        del kwargs
        session = await self.load(session_id)
        assert session is not None
        replayed = self.calls > 0
        self.calls += 1
        if not replayed:
            await self.append_events(session_id, list(request.events))
        receipt = RuntimePublicationReceipt(
            session_id=session_id,
            publication_id=request.publication_id,
            kind=request.kind,
            intent=request.intent,
            request_digest="1" * 64,
            publication_digest="2" * 64,
            checkpoint_digest="3" * 64,
            transcript_digest="4" * 64,
            events_digest="5" * 64,
            source_status=session.status,
            source_run_epoch=session.run_epoch,
            transcript_start_cursor=0,
            transcript_end_cursor=len(request.transcript_messages),
            appended_event_ids=tuple(event.id for event in request.events),
            referenced_events=request.referenced_events,
            published_at=self._published_at,
        )
        return RuntimePublicationResult(
            session=session,
            receipt=receipt,
            replayed=replayed,
        )


def test_auxiliary_event_fan_out_runs_after_commit_and_is_replay_safe() -> None:
    async def scenario() -> None:
        store, session = await _created_store(
            _AuxiliaryPublicationStore(),
            session_id="session-1",
        )
        pending_round = _structured_pending_round()
        source_checkpoint = _source_checkpoint(pending_round)
        events = _structured_lifecycle_events(
            pending_round,
            valid=True,
            session_id=session.id,
        )
        auxiliary_events = _structured_auxiliary_events(
            pending_round,
            valid=True,
            retry_scheduled=False,
            session_id=session.id,
        )

        prepared = prepare_tool_round_publication(
            session_id=session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=events,
            expected_statuses={SessionStatus.PENDING},
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
            extension=_StructuredOutputPublicationExtension(
                events=auxiliary_events,
                valid=True,
                retry_scheduled=False,
            ),
        )
        sink = InMemoryEventSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[sink],
        )

        first = await publish_tool_round_publication(
            prepared,
            session_store=store,
            event_writer=writer,
        )
        replay = await publish_tool_round_publication(
            prepared,
            session_store=store,
            event_writer=writer,
        )

        assert first.replayed is False
        assert replay.replayed is True
        assert [event.id for event in sink.events] == [
            "structured-validating",
            "structured-validated",
        ]

    asyncio.run(scenario())


def test_prepared_request_property_is_detached_from_replay_material() -> None:
    pending_round = _pending_round()
    prepared = prepare_tool_round_publication(
        session_id="session-1",
        pending_round=pending_round,
        source_checkpoint=_source_checkpoint(pending_round),
        durable_events=_lifecycle_events(pending_round),
        expected_statuses={SessionStatus.PENDING},
        expected_run_epoch=0,
        expected_transcript_cursor=0,
    )

    assert type(prepared) is PreparedToolRoundPublication
    inspection_copy = prepared.request
    inspection_copy.intent["round_id"] = "mutated"
    assert prepared.request.intent["round_id"] == pending_round.tool_round_id
