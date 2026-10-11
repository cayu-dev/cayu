"""Shared session-store tool rounds contract cases."""

from __future__ import annotations

import asyncio
import copy
from datetime import UTC, datetime, timedelta

import pytest
from tests.core._execution_profile_fixtures import (
    create_admitted_session,
)
from tests.core.session_store_conformance.support import (
    _close_store,
    _identity,
    _open_store,
    _publication_tool_round_identity,
    _reopen_store,
    _structured_output_coherence_publication,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.context.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import (
    Event,
    EventType,
)
from cayu.execution_units import ToolRoundIdentity
from cayu.messages import (
    Message,
)
from cayu.runtime.authority import SessionRunFenced
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._checkpoint_mutations import (
    RuntimePublicationCheckpointOperation,
    RuntimePublicationMutation,
    SessionRuntimePublicationConflict,
    runtime_publication_checkpoint_mutation,
)
from cayu.sessions._model_stage import (
    ModelCompletionStageRequest,
    SessionModelCompletionStageConflict,
)
from cayu.sessions._runtime_publication import (
    RuntimePublicationRequest,
    runtime_publication_event_reference,
)
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import (
    RunRequest,
)

pytestmark = pytest.mark.postgres


def test_session_store_conformance_tool_round_rejects_omitted_terminal_evidence(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_tool_round_omitted_terminal"
            round_identity = _publication_tool_round_identity("round-with-conflicting-terminal")
            round_id = round_identity["tool_round_id"]
            pending_round = {
                **round_identity,
                "agent_name": "assistant",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "tool_name": "lookup",
                        "arguments": {},
                    }
                ],
            }
            source_checkpoint = {"pending_tool_round": pending_round}
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            await store.checkpoint(session_id, source_checkpoint)
            lifecycle_payload = {
                **round_identity,
                "tool_call_id": "call-1",
                "idempotency_key": "tool-call:call-1",
            }
            started = Event(
                id="omitted-terminal-started",
                type=EventType.TOOL_CALL_STARTED,
                session_id=session_id,
                tool_name="lookup",
                payload=lifecycle_payload,
            )
            authoritative_result = {
                "content": "first terminal",
                "structured": None,
                "artifacts": [],
                "is_error": False,
            }
            first_terminal = Event(
                id="omitted-terminal-first",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                tool_name="lookup",
                payload={**lifecycle_payload, "result": authoritative_result},
            )
            second_terminal = Event(
                id="omitted-terminal-second",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                tool_name="lookup",
                payload={
                    **lifecycle_payload,
                    "result": {
                        **authoritative_result,
                        "content": "contradictory second terminal",
                    },
                },
            )
            await store.append_events(
                session_id,
                [started, first_terminal, second_terminal],
            )
            lifecycle_events = await store.load_tool_round_lifecycle_events(
                session_id,
                ["call-1"],
            )
            assert [event.id for event in lifecycle_events] == [
                started.id,
                first_terminal.id,
                second_terminal.id,
            ]
            request = RuntimePublicationRequest(
                publication_id=f"tool-round:{round_id}",
                kind="tool-round",
                intent={
                    "round_id": round_id,
                    **round_identity,
                    "tool_call_ids": ["call-1"],
                },
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    {},
                ),
                transcript_messages=[
                    Message.tool_result(
                        tool_call_id="call-1",
                        tool_name="lookup",
                        content="first terminal",
                        **round_identity,
                    )
                ],
                events=[],
                referenced_events=[
                    runtime_publication_event_reference(started),
                    runtime_publication_event_reference(first_terminal),
                ],
            )

            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="bind every durable lifecycle event",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=request,
                    expected_transcript_cursor=0,
                )

            assert await store.load_checkpoint(session_id) == source_checkpoint
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == [
                started,
                first_terminal,
                second_terminal,
            ]
            assert (
                await store.load_runtime_publication_receipt(
                    session_id,
                    request.publication_id,
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_structured_output_tool_round_auxiliary_publication(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_structured_output_tool_round_publication"
            round_identity = _publication_tool_round_identity("structured-output-round")
            round_id = round_identity["tool_round_id"]
            tool_call_id = "structured-output-call"
            spec = StructuredOutputSpec(
                name="answer",
                json_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                max_retries=2,
            )
            pending_round = {
                **round_identity,
                "agent_name": "assistant",
                "environment_name": None,
                "task_id": None,
                "source_model_step_id": "model-step:structured-output",
                "source_transcript_cursor": 0,
                "model_step": 2,
                "structured_output_attempt": 1,
                "structured_output_validation": {
                    "valid": True,
                    "output": {"answer": "ok"},
                    "errors": [],
                },
                "tool_calls": [
                    {
                        "tool_call_id": tool_call_id,
                        "tool_name": STRUCTURED_OUTPUT_TOOL_NAME,
                        "arguments": {"output": {"answer": "ok"}},
                    }
                ],
                "structured_output": spec.model_dump(mode="json"),
            }
            source_checkpoint = {"pending_tool_round": pending_round}
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.checkpoint(session_id, source_checkpoint)
            result_payload = {
                "content": "Structured output accepted.",
                "structured": {"output": {"answer": "ok"}},
                "artifacts": [],
                "is_error": False,
            }
            terminal = Event(
                id="structured-output-terminal",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                agent_name="assistant",
                tool_name=STRUCTURED_OUTPUT_TOOL_NAME,
                payload={
                    **round_identity,
                    "tool_call_id": tool_call_id,
                    "idempotency_key": "structured-output-idempotency",
                    "result": result_payload,
                },
            )
            await store.append_event(session_id, terminal)
            validating = Event(
                id="structured-output-validating",
                type=EventType.STRUCTURED_OUTPUT_VALIDATING,
                session_id=session_id,
                agent_name="assistant",
                payload={
                    **round_identity,
                    "name": spec.name,
                    "strategy": "tool",
                    "step": 2,
                    "attempt": 1,
                    "max_retries": spec.max_retries,
                },
            )
            validated = Event(
                id="structured-output-validated",
                type=EventType.STRUCTURED_OUTPUT_VALIDATED,
                session_id=session_id,
                agent_name="assistant",
                payload={
                    **round_identity,
                    "name": spec.name,
                    "step": 2,
                    "attempt": 1,
                    "max_retries": spec.max_retries,
                    "valid": True,
                    "errors": [],
                    "output": {"answer": "ok"},
                },
            )
            request = RuntimePublicationRequest(
                publication_id=f"tool-round:{round_id}",
                kind="tool-round",
                intent={
                    "round_id": round_id,
                    **round_identity,
                    "tool_call_ids": [tool_call_id],
                    "auxiliary": {
                        "schema_version": 1,
                        "kind": "structured-output-validation",
                        "step": 2,
                        "attempt": 1,
                        "valid": True,
                        "retry_scheduled": False,
                        "event_ids": [validating.id, validated.id],
                    },
                },
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    {},
                ),
                transcript_messages=[
                    Message.tool_result(
                        tool_call_id=tool_call_id,
                        tool_name=STRUCTURED_OUTPUT_TOOL_NAME,
                        content=result_payload["content"],
                        structured=result_payload["structured"],
                        **round_identity,
                    )
                ],
                events=[validating, validated],
                referenced_events=[
                    runtime_publication_event_reference(terminal),
                ],
            )

            published = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=created.run_epoch,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False
            assert await store.load_checkpoint(session_id) == {}
            assert await store.load_transcript(session_id) == list(request.transcript_messages)
            assert await store.load_events(session_id) == [
                terminal,
                validating,
                validated,
            ]

            store = await _reopen_store(session_store_case, store)
            replayed = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=created.run_epoch,
                expected_transcript_cursor=0,
            )
            assert replayed.replayed is True
            assert replayed.receipt == published.receipt
            assert await store.load_events(session_id) == [
                terminal,
                validating,
                validated,
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("case", "error_match"),
    [
        (
            "output_mismatch",
            "outcome conflicts with its authoritative validation",
        ),
        (
            "valid_with_errors",
            "valid structured-output event requires output and no errors",
        ),
        (
            "failed_retry_disagreement",
            "outcome and retry events contain conflicting results",
        ),
        (
            "authoritative_validity",
            "outcome conflicts with its authoritative validation",
        ),
    ],
)
def test_session_store_conformance_rejects_structured_output_auxiliary_incoherence(
    session_store_case,
    case: str,
    error_match: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_structured_output_coherence_{case}"
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            source_checkpoint, terminal, request = _structured_output_coherence_publication(
                session_id=session_id,
                case=case,
            )
            await store.checkpoint(session_id, source_checkpoint)
            await store.append_event(session_id, terminal)

            with pytest.raises(ValueError, match=error_match):
                await store.publish_runtime_publication(
                    session_id,
                    request=request,
                    expected_statuses={SessionStatus.PENDING},
                    expected_run_epoch=created.run_epoch,
                    expected_transcript_cursor=0,
                )

            assert await store.load_checkpoint(session_id) == source_checkpoint
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == [terminal]
            assert (
                await store.load_runtime_publication_receipt(
                    session_id,
                    request.publication_id,
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_tool_round_scopes_reused_call_id_by_round(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_tool_round_reused_call_id"
            current_identity = _publication_tool_round_identity("current-round")
            old_identity = _publication_tool_round_identity("old-round")
            round_id = current_identity["tool_round_id"]
            tool_call_id = "reused-call"
            source_checkpoint = {
                "pending_tool_round": {
                    **current_identity,
                    "agent_name": "assistant",
                    "tool_calls": [
                        {
                            "tool_call_id": tool_call_id,
                            "tool_name": "lookup",
                            "arguments": {},
                        }
                    ],
                }
            }
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.checkpoint(session_id, source_checkpoint)
            result_payload = {
                "content": "current result",
                "structured": None,
                "artifacts": [],
                "is_error": False,
            }
            old_terminal = Event(
                id="reused-call-old-terminal",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                tool_name="lookup",
                payload={
                    **old_identity,
                    "tool_call_id": tool_call_id,
                    "idempotency_key": "old-idempotency",
                    "result": {
                        **result_payload,
                        "content": "old result",
                    },
                },
            )
            current_terminal = Event(
                id="reused-call-current-terminal",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                tool_name="lookup",
                payload={
                    **current_identity,
                    "tool_call_id": tool_call_id,
                    "idempotency_key": "current-idempotency",
                    "result": result_payload,
                },
            )
            await store.append_events(
                session_id,
                [old_terminal, current_terminal],
            )
            assert [
                event.id
                for event in await store.load_tool_round_lifecycle_events(
                    session_id,
                    [tool_call_id],
                )
            ] == [old_terminal.id, current_terminal.id]
            assert [
                event.id
                for event in await store.load_tool_round_lifecycle_events_for_round(
                    session_id,
                    [tool_call_id],
                    tool_round_identity=ToolRoundIdentity.model_validate(current_identity),
                )
            ] == [current_terminal.id]
            request = RuntimePublicationRequest(
                publication_id=f"tool-round:{round_id}",
                kind="tool-round",
                intent={
                    "round_id": round_id,
                    **current_identity,
                    "tool_call_ids": [tool_call_id],
                },
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    {},
                ),
                transcript_messages=[
                    Message.tool_result(
                        tool_call_id=tool_call_id,
                        tool_name="lookup",
                        content=result_payload["content"],
                        **current_identity,
                    )
                ],
                events=[],
                referenced_events=[
                    runtime_publication_event_reference(current_terminal),
                ],
            )

            published = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_transcript_cursor=0,
            )

            assert published.replayed is False
            assert await store.load_checkpoint(session_id) == {}
            assert await store.load_transcript(session_id) == list(request.transcript_messages)
            assert await store.load_events(session_id) == [
                old_terminal,
                current_terminal,
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_round_lookup_retains_ambiguous_reused_id_evidence(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_tool_round_ambiguous_reused_call_id"
            tool_call_id = "reused-call"
            old_identity = _publication_tool_round_identity("ambiguous-old-round")
            current_identity = _publication_tool_round_identity("ambiguous-current-round")
            conflicting_identity = _publication_tool_round_identity("ambiguous-conflicting-round")
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.checkpoint(
                session_id,
                {
                    "pending_tool_round": {
                        **current_identity,
                        "agent_name": "assistant",
                        "tool_calls": [
                            {
                                "tool_call_id": tool_call_id,
                                "tool_name": "lookup",
                                "arguments": {},
                            }
                        ],
                    }
                },
            )
            events = [
                Event(
                    id="reused-call-old-valid",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="lookup",
                    payload={
                        **old_identity,
                        "tool_call_id": tool_call_id,
                    },
                ),
                Event(
                    id="reused-call-missing-round",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="lookup",
                    payload={
                        "tool_call_id": tool_call_id,
                        "model_step_id": current_identity["model_step_id"],
                        "model_attempt_id": current_identity["model_attempt_id"],
                    },
                ),
                Event(
                    id="reused-call-padded-round",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="lookup",
                    payload={
                        "tool_call_id": tool_call_id,
                        "model_step_id": current_identity["model_step_id"],
                        "model_attempt_id": current_identity["model_attempt_id"],
                        "tool_round_id": " padded-round ",
                    },
                ),
                Event(
                    id="reused-call-malformed-model-identity",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="lookup",
                    payload={
                        "tool_call_id": tool_call_id,
                        "model_step_id": "malformed-model-step",
                        "model_attempt_id": conflicting_identity["model_attempt_id"],
                        "tool_round_id": conflicting_identity["tool_round_id"],
                    },
                ),
                Event(
                    id="reused-call-conflicting-round",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="lookup",
                    payload={
                        **current_identity,
                        "tool_round_id": conflicting_identity["tool_round_id"],
                        "tool_call_id": tool_call_id,
                    },
                ),
                Event(
                    id="reused-call-current-valid",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="lookup",
                    payload={
                        **current_identity,
                        "tool_call_id": tool_call_id,
                    },
                ),
            ]
            await store.append_events(session_id, events)

            scoped = await store.load_tool_round_lifecycle_events_for_round(
                session_id,
                [tool_call_id],
                tool_round_identity=ToolRoundIdentity.model_validate(current_identity),
            )

            assert [event.id for event in scoped] == [
                "reused-call-missing-round",
                "reused-call-padded-round",
                "reused-call-malformed-model-identity",
                "reused-call-conflicting-round",
                "reused-call-current-valid",
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "interruption", [None, "prepare_error", "prepare_cancel", "dispatch_cancel"]
)
def test_auxiliary_owner_dispatch_fences_survive_interruption(
    session_store_case, monkeypatch, interruption
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_auxiliary_dispatch"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id, from_statuses={SessionStatus.PENDING}, to_status=SessionStatus.RUNNING
            )
            await store.checkpoint(session_id, {"parent": "unchanged"})
            app = CayuApp(session_store=store, enable_logging=False)
            owner = app._tool_round_executor.invocation._auxiliary_inference
            request = ModelCompletionStageRequest(
                stage_id="aux_dispatch_attempt",
                logical_step_id="aux_dispatch_attempt",
                dispatch_ordinal=1,
                purpose="auxiliary-inference",
                intent={
                    "model_step_id": "mstep_" + "1" * 32,
                    "model_attempt_id": "matt_" + "2" * 32,
                    "interaction_id": "interaction_auxiliary",
                    "request_fingerprint": "3" * 64,
                },
            )
            entered = []
            failure = RuntimeError("preparation acknowledgement lost")

            async def dispatch():
                stage = await owner.prepare_dispatch(
                    session_id=session_id,
                    request=request,
                    reservations=(),
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
                entered.append(stage.stage_id)
                return stage

            boundary = (
                "mark_model_completion_stage_dispatched"
                if interruption == "dispatch_cancel"
                else "prepare_model_completion_stage"
            )
            original = getattr(app._runtime_session_store, boundary)

            async def interrupt_after_commit(*args, **kwargs):
                result = await original(*args, **kwargs)
                if interruption == "prepare_error":
                    raise failure
                task = asyncio.current_task()
                assert task is not None
                task.cancel()
                assert task.cancelling() == 1
                await asyncio.sleep(0)
                return result

            with monkeypatch.context() as patch:
                if interruption is not None:
                    patch.setattr(app._runtime_session_store, boundary, interrupt_after_commit)
                task = asyncio.create_task(dispatch())
                if interruption == "prepare_error":
                    with pytest.raises(RuntimeError) as caught:
                        await task
                    assert caught.value is failure
                elif interruption is not None:
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert task.cancelled()
                    assert task.cancelling() == 1
                else:
                    stage = await task
                    assert stage.dispatch_ordinal == 1
                    assert not task.cancelled()
            assert entered == ([] if interruption else [request.stage_id])
            store = await _reopen_store(session_store_case, store)
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage.stage_id == request.stage_id
            assert active.stage.state == "in_flight"
            receipt = await store.load_model_completion_stage_dispatch(session_id, request.stage_id)
            assert (receipt is not None) == (interruption in (None, "dispatch_cancel"))
            app = CayuApp(session_store=store, enable_logging=False)
            owner = app._tool_round_executor.invocation._auxiliary_inference
            # Readback after restart is evidence, never another dispatch grant.
            with pytest.raises(SessionModelCompletionStageConflict, match="replay"):
                await dispatch()
            assert entered == ([] if interruption else [request.stage_id])
            assert await store.load_checkpoint(session_id) == {"parent": "unchanged"}
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []
            if interruption in {"prepare_error", "prepare_cancel"}:
                recovered = (
                    await app._model_completion_recovery.reconcile_model_completion_boundary(
                        running
                    )
                )
                assert recovered.state == "prepared_abandoned"
                assert await store.load_active_model_completion_stage(session_id) is None
                replay = await app._model_completion_recovery.reconcile_model_completion_boundary(
                    running
                )
                assert replay.state == "none"
                assert entered == []
                assert await store.load_checkpoint(session_id) == {"parent": "unchanged"}
                assert await store.load_transcript(session_id) == []
                assert await store.load_events(session_id) == []
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "outcome", ["completed", "failed", "cancelled", "timed_out", "outcome_unknown"]
)
@pytest.mark.parametrize("through_owner", [False, True])
@pytest.mark.parametrize(
    "promotion_status", [None, SessionStatus.COMPLETED, SessionStatus.INTERRUPTED]
)
def test_session_store_conformance_auxiliary_stage_preserves_parent_and_replays(
    session_store_case, outcome, through_owner, monkeypatch, promotion_status
) -> None:
    from cayu.runtime._run_limits import _model_completion_with_budget_settlement_evidence

    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_auxiliary_stage"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id, from_statuses={SessionStatus.PENDING}, to_status=SessionStatus.RUNNING
            )
            parent_checkpoint = {"parent_tool_round_sentinel": {"state": "pending"}}
            await store.checkpoint(session_id, parent_checkpoint)
            intent = {
                "model_step_id": "mstep_" + "4" * 32,
                "model_attempt_id": "matt_" + "5" * 32,
                "interaction_id": "interaction_auxiliary",
                "request_fingerprint": "1" * 64,
                "provider_name": "fake",
                "requested_model": "fake-model",
                "model": "fake-model",
                "execution_profile_fingerprint": "0" * 64,
                "auxiliary_inference": {
                    "operation_id": "aux_operation",
                    "purpose": "tool.summary",
                    "parent": {
                        "model_step_id": "mstep_" + "1" * 32,
                        "model_attempt_id": "matt_" + "2" * 32,
                        "tool_round_id": "tround_" + "3" * 32,
                    },
                    "tool_call_id": "call_parent",
                },
            }
            prepared = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id="aux_attempt_1",
                    logical_step_id="aux_publication_1",
                    dispatch_ordinal=0,
                    purpose="auxiliary-inference",
                    intent=intent,
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert prepared.dispatch_authorized
            await store.mark_model_completion_stage_dispatched(session_id, stage=prepared.stage)
            event = Event(
                type=EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
                session_id=session_id,
                payload={
                    **intent,
                    "auxiliary_outcome": outcome,
                    "attempt": 1,
                    "usage_status": "observed",
                    "usage": {"input_tokens": 4, "output_tokens": 2},
                },
            )
            event = _model_completion_with_budget_settlement_evidence(
                event,
                (),
                prepare_event=lambda value: value.model_copy(deep=True),
                settled_at=event.timestamp,
            )
            assert event.payload["budget_settlements"] == []
            publication = RuntimePublicationRequest(
                publication_id=prepared.stage.logical_step_id,
                kind="auxiliary-inference",
                intent=intent,
                mutation=RuntimePublicationMutation(),
                transcript_messages=(),
                events=(event,),
            )
            # None of these invalid terminals may partially publish or settle.
            invalid_payloads = [
                {**event.payload, "attempt": True},
                {**event.payload, "attempt": 2},
                {key: value for key, value in event.payload.items() if key != "attempt"},
                {**event.payload, "provider_name": "other"},
                {**event.payload, "model_attempt_id": "matt_other"},
                {**event.payload, "auxiliary_outcome": "future"},
                {**event.payload, "usage_status": "future"},
                {**event.payload, "auxiliary_outcome": {}},
                {**event.payload, "usage_status": True},
                {**event.payload, "budget_settlements": ["not-a-settlement"]},
                {**event.payload, "transcript_cursor": 0},
                {
                    **event.payload,
                    "auxiliary_inference": {
                        **intent["auxiliary_inference"],
                        "tool_call_id": "other",
                    },
                },
            ]
            for payload in invalid_payloads:
                invalid = publication.model_copy(
                    update={"events": (event.model_copy(update={"payload": payload}),)}
                )
                with pytest.raises(ValueError):
                    await store.complete_model_completion_stage(
                        session_id, stage_id=prepared.stage.stage_id, publication=invalid
                    )
                assert await store.load_events(session_id) == []
                assert (
                    await store.load_model_completion_stage(session_id, prepared.stage.stage_id)
                ).state == "in_flight"
            for update in (
                {"transcript_messages": (Message.text("assistant", "not a turn"),)},
                {"events": (event, event.model_copy(update={"id": "extra-settlement"}))},
                {
                    "mutation": RuntimePublicationMutation(
                        operations=(
                            RuntimePublicationCheckpointOperation(
                                key="unexpected", action="set", value=1, expected_value_digest=None
                            ),
                        )
                    )
                },
            ):
                with pytest.raises(ValueError):
                    await store.complete_model_completion_stage(
                        session_id,
                        stage_id=prepared.stage.stage_id,
                        publication=publication.model_copy(update=update),
                    )
                assert await store.load_checkpoint(session_id) == parent_checkpoint
                assert await store.load_transcript(session_id) == []
                assert await store.load_events(session_id) == []
            completion_store = store
            if through_owner:
                from cayu.runtime._auxiliary_inference import AuxiliaryInferenceOwner

                app = CayuApp(session_store=store, enable_logging=False)
                completion_store = app._runtime_session_store
                owner = AuxiliaryInferenceOwner(
                    session_store=completion_store,
                    event_writer=app._event_writer,
                    run_limit_controller=app._run_limit_controller,
                    clock=app._clock,
                )
                without_settlements = event.model_copy(deep=True)
                without_settlements.payload.pop("budget_settlements")
                # Production terminal producers attest the profile before the
                # writer prepares the event. Raw payload text is not authority.
                with pytest.raises(ValueError, match="target/profile conflicts"):
                    owner.prepare_terminal(
                        stage=prepared.stage, event=without_settlements, reservations=()
                    )
                from cayu.events import event_with_runtime_payload_authority

                without_settlements = event_with_runtime_payload_authority(
                    without_settlements, "execution_profile_fingerprint"
                )
                publication = owner.prepare_terminal(
                    stage=prepared.stage, event=without_settlements, reservations=()
                )
            await completion_store.complete_model_completion_stage(
                session_id, stage_id=prepared.stage.stage_id, publication=publication
            )
            expected_transcript = []
            if promotion_status is not None:
                expected_transcript = [Message.text("assistant", "Parent advanced after inference")]
                await store.append_transcript_messages(session_id, expected_transcript)
                await store.transition_status(
                    session_id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=promotion_status,
                )
            store = await _reopen_store(session_store_case, store)
            if through_owner:
                from cayu.runtime._auxiliary_inference import AuxiliaryInferenceOwner

                app = CayuApp(session_store=store, enable_logging=False)
                owner = AuxiliaryInferenceOwner(
                    session_store=app._runtime_session_store,
                    event_writer=app._event_writer,
                    run_limit_controller=app._run_limit_controller,
                    clock=app._clock,
                )

                async def fail_settlement(*args, **kwargs):
                    raise RuntimeError("injected settlement failure")

                with monkeypatch.context() as patch:
                    patch.setattr(
                        app._run_limit_controller,
                        "reconcile_model_completion_settlements",
                        fail_settlement,
                    )
                    with pytest.raises(RuntimeError, match="injected settlement failure"):
                        await owner.publish_terminal(
                            stage=prepared.stage,
                            publication=publication,
                            expected_run_epoch=running.run_epoch,
                        )
                active = await store.load_active_model_completion_stage(session_id)
                assert active is not None
                assert active.stage.stage_id == prepared.stage.stage_id
                assert active.stage.state == "completed"
                assert await store.load_events(session_id) == []
                first_events = await owner.publish_terminal(
                    stage=prepared.stage,
                    publication=publication,
                    expected_run_epoch=running.run_epoch,
                )
                replay_events = await owner.publish_terminal(
                    stage=prepared.stage,
                    publication=publication,
                    expected_run_epoch=running.run_epoch,
                )
                assert [item.id for item in first_events] == [event.id]
                assert [item.id for item in replay_events] == [event.id]
            else:
                first = await store.promote_model_completion_stage(
                    session_id,
                    stage_id=prepared.stage.stage_id,
                    expected_run_epoch=running.run_epoch,
                )
                assert not first.replayed
            replay = await store.promote_model_completion_stage(
                session_id, stage_id=prepared.stage.stage_id, expected_run_epoch=running.run_epoch
            )
            assert replay.replayed
            assert await store.load_checkpoint(session_id) == parent_checkpoint
            assert await store.load_transcript(session_id) == expected_transcript
            assert (await store.load(session_id)).status == (
                promotion_status or SessionStatus.RUNNING
            )
            assert [item.id for item in await store.load_events(session_id)] == [event.id]
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("dispatched", [False, True])
@pytest.mark.parametrize("with_plan", [False, True])
def test_auxiliary_recovery_terminal_requires_live_exact_owner(
    session_store_case, monkeypatch, dispatched, with_plan
) -> None:
    from cayu.runtime._durable_operation_ownership import DurableOperationOwnership
    from cayu.sessions._model_stage import ModelCompletionStageRecoveryFence

    async def run() -> None:
        store = await _open_store(session_store_case)
        claim = None
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            provider = ScriptedModelProvider([])
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
            admitted = await create_admitted_session(
                store,
                request=RunRequest(agent_name="assistant", session_id="aux_recovery", messages=[]),
                provider_name=provider.name,
                model="scripted-model",
                provider=provider,
                app=app,
            )
            session = admitted.session
            intent = {
                "model_step_id": "mstep_" + "4" * 32,
                "model_attempt_id": "matt_" + "5" * 32,
                "interaction_id": admitted.active_invocation_profile.interaction_id,
                "request_fingerprint": "1" * 64,
                "provider_name": provider.name,
                "requested_model": "scripted-model",
                "execution_profile_fingerprint": admitted.active_invocation_profile.profile.fingerprint,
                "auxiliary_inference": {
                    "operation_id": "aux_operation",
                    "purpose": "tool.summary",
                    "parent": {
                        "model_step_id": "mstep_" + "1" * 32,
                        "model_attempt_id": "matt_" + "2" * 32,
                        "tool_round_id": "tround_" + "3" * 32,
                    },
                    "tool_call_id": "call_parent",
                },
            }
            prepared = await store.prepare_model_completion_stage(
                session.id,
                request=ModelCompletionStageRequest(
                    stage_id="aux_recovery_stage",
                    logical_step_id="aux_recovery_publication",
                    dispatch_ordinal=0,
                    purpose="auxiliary-inference",
                    intent=intent,
                ),
                expected_statuses={session.status},
                expected_run_epoch=session.run_epoch,
                expected_transcript_cursor=0,
            )
            if dispatched:
                await store.mark_model_completion_stage_dispatched(session.id, stage=prepared.stage)
            claim = await app._session_engine._recovery_ownership.claim(
                session=session,
                inactive_for_seconds=None,
                execution_profile_snapshot=admitted.active_invocation_profile,
            )
            assert claim is not None
            checkpoint = await store.load_checkpoint(session.id)
            profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            assert profile is not None
            plan_owner = None
            if with_plan:
                now = datetime.now(UTC)
                plan_owner = DurableOperationOwnership(
                    operation_id="auxiliary-recovery-plan",
                    claim_id="plan-claim",
                    owner_id="plan-worker",
                    generation=1,
                    acquired_at=now,
                    renewed_at=now,
                    lease_expires_at=now + timedelta(minutes=5),
                )
                checkpoint = copy.deepcopy(checkpoint)
                checkpoint["recovery_plan_execution"] = {
                    "ownership": plan_owner.model_dump(mode="json")
                }
                await store.transform_checkpoint(session.id, lambda _session, _current: checkpoint)
            fence = ModelCompletionStageRecoveryFence(
                expected_session_instance_id=session.instance_id,
                expected_run_epoch=claim.session.run_epoch,
                expected_active_profile=profile,
                recovery_claim_id=claim.claim_id,
                preparation_digest=prepared.stage.preparation_digest,
                recovery_plan_ownership=plan_owner,
            )
            publication = RuntimePublicationRequest(
                publication_id=prepared.stage.logical_step_id,
                kind="auxiliary-inference",
                intent=intent,
                mutation=RuntimePublicationMutation(),
                transcript_messages=(),
                events=(
                    Event(
                        type=EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
                        session_id=session.id,
                        payload={
                            **intent,
                            "auxiliary_outcome": "outcome_unknown",
                            "attempt": 1,
                            "usage_status": "missing",
                            "budget_settlements": [],
                        },
                    ),
                ),
            )
            initial_events = await store.load_events(session.id)
            if not dispatched:
                with pytest.raises(SessionModelCompletionStageConflict, match="dispatch receipt"):
                    await store.complete_recovered_model_completion_stage(
                        session.id,
                        stage_id=prepared.stage.stage_id,
                        publication=publication,
                        recovery_fence=fence,
                    )
                assert (
                    await store.load_model_completion_stage(session.id, prepared.stage.stage_id)
                    == prepared.stage
                )
                assert await store.load_checkpoint(session.id) == checkpoint
                assert await store.load_events(session.id) == initial_events
                return
            if plan_owner is not None:
                wrong_owners = [None] + [
                    plan_owner.model_copy(update={field: value})
                    for field, value in (
                        ("claim_id", "different-claim"),
                        ("owner_id", "different-worker"),
                        ("operation_id", "different-operation"),
                        ("generation", 2),
                    )
                ]
                for wrong_owner in wrong_owners:
                    with pytest.raises(SessionRunFenced):
                        await app._runtime_session_store.complete_recovered_model_completion_stage(
                            session.id,
                            stage_id=prepared.stage.stage_id,
                            publication=publication,
                            recovery_fence=fence.model_copy(
                                update={"recovery_plan_ownership": wrong_owner}
                            ),
                        )
                    assert await store.load_checkpoint(session.id) == checkpoint
                    assert await store.load_events(session.id) == initial_events
                    assert (
                        await store.load_model_completion_stage(session.id, prepared.stage.stage_id)
                        == prepared.stage
                    )
                for marker_present in (False, True):
                    unavailable = copy.deepcopy(checkpoint)
                    if marker_present:
                        expired_owner = plan_owner.model_copy(
                            update={
                                "acquired_at": now - timedelta(minutes=3),
                                "renewed_at": now - timedelta(minutes=2),
                                "lease_expires_at": now - timedelta(minutes=1),
                            }
                        )
                        unavailable["recovery_plan_execution"]["ownership"] = (
                            expired_owner.model_dump(mode="json")
                        )
                    else:
                        unavailable.pop("recovery_plan_execution")
                    await store.transform_checkpoint(
                        session.id, lambda _session, _current, candidate=unavailable: candidate
                    )
                    with pytest.raises(SessionRunFenced):
                        await store.complete_recovered_model_completion_stage(
                            session.id,
                            stage_id=prepared.stage.stage_id,
                            publication=publication,
                            recovery_fence=fence,
                        )
                    assert await store.load_checkpoint(session.id) == unavailable
                    assert await store.load_events(session.id) == initial_events
                    assert (
                        await store.load_model_completion_stage(session.id, prepared.stage.stage_id)
                        == prepared.stage
                    )
                await store.transform_checkpoint(session.id, lambda _session, _current: checkpoint)
            for updates in (
                {"expected_session_instance_id": "wrong-instance"},
                {"recovery_claim_id": "wrong-claim"},
                {"preparation_digest": "f" * 64},
                {
                    "expected_run_epoch": fence.expected_run_epoch + 1,
                    "expected_active_profile": profile.model_copy(
                        update={"run_epoch": fence.expected_run_epoch + 1}
                    ),
                },
                {"expected_active_profile": profile.model_copy(update={"interaction_id": "wrong"})},
            ):
                with pytest.raises((SessionRunFenced, SessionModelCompletionStageConflict)):
                    await app._runtime_session_store.complete_recovered_model_completion_stage(
                        session.id,
                        stage_id=prepared.stage.stage_id,
                        publication=publication,
                        recovery_fence=fence.model_copy(update=updates),
                    )
                assert (
                    await store.load_model_completion_stage(session.id, prepared.stage.stage_id)
                    == prepared.stage
                )
                assert await store.load_events(session.id) == initial_events
                assert await store.load_checkpoint(session.id) == checkpoint

            def expire(_session, current):
                updated = copy.deepcopy(current)
                marker = updated["incomplete_session_recovery_claim"]
                marker["claimed_at"] = (datetime.now(UTC) - timedelta(minutes=2)).isoformat()
                marker["claim_expires_at"] = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
                return updated

            await store.transform_checkpoint(session.id, expire)
            with pytest.raises(SessionRunFenced):
                await store.complete_recovered_model_completion_stage(
                    session.id,
                    stage_id=prepared.stage.stage_id,
                    publication=publication,
                    recovery_fence=fence,
                )
            assert (
                await store.load_model_completion_stage(session.id, prepared.stage.stage_id)
                == prepared.stage
            )
            await store.transform_checkpoint(session.id, lambda _session, _current: checkpoint)

            entered = asyncio.Event()
            proceed = asyncio.Event()
            original = store._complete_model_completion_stage_atomic

            async def delayed(preparation):
                entered.set()
                await proceed.wait()
                return await original(preparation)

            with monkeypatch.context() as patch:
                patch.setattr(store, "_complete_model_completion_stage_atomic", delayed)
                pending = asyncio.create_task(
                    store.complete_recovered_model_completion_stage(
                        session.id,
                        stage_id=prepared.stage.stage_id,
                        publication=publication,
                        recovery_fence=fence,
                    )
                )
                await asyncio.wait_for(entered.wait(), timeout=5)
                try:

                    def replace_claim(_session, current):
                        updated = copy.deepcopy(current)
                        if with_plan:
                            updated["recovery_plan_execution"]["ownership"]["generation"] += 1
                        else:
                            updated["incomplete_session_recovery_claim"]["claim_id"] = "new-owner"
                        return updated

                    await store.transform_checkpoint(session.id, replace_claim)
                finally:
                    proceed.set()
                with pytest.raises(SessionRunFenced):
                    await pending
            assert (
                await store.load_model_completion_stage(session.id, prepared.stage.stage_id)
                == prepared.stage
            )
            assert await store.load_events(session.id) == initial_events
            await store.transform_checkpoint(session.id, lambda _session, _current: checkpoint)
            first = await app._runtime_session_store.complete_recovered_model_completion_stage(
                session.id,
                stage_id=prepared.stage.stage_id,
                publication=publication,
                recovery_fence=fence,
            )
            assert not first.replayed
            await store.transform_checkpoint(session.id, expire)
            store = await _reopen_store(session_store_case, store)
            replay = await store.complete_recovered_model_completion_stage(
                session.id,
                stage_id=prepared.stage.stage_id,
                publication=publication,
                recovery_fence=fence,
            )
            assert replay.replayed
            assert replay.stage == first.stage
            with pytest.raises(SessionModelCompletionStageConflict):
                await store.complete_recovered_model_completion_stage(
                    session.id,
                    stage_id=prepared.stage.stage_id,
                    publication=publication,
                    recovery_fence=fence.model_copy(update={"preparation_digest": "f" * 64}),
                )
            assert await store.load_events(session.id) == initial_events
        finally:
            if claim is not None:
                claim.require_authority().retire()
            await _close_store(store)

    asyncio.run(run())
