"""Shared session-store user input contract cases."""

from __future__ import annotations

import asyncio
import contextvars
import copy
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _collect_events,
    _open_peer_store,
    _open_store,
    _private_event_for_public_event,
    _reopen_store,
    _SimulatedProcessLoss,
    _UserInputRecoveryProvider,
)

import cayu.runtime._session_control as session_control_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.user_input import (
    PENDING_USER_INPUT_CHECKPOINT_KEY,
    USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY,
    PendingUserInput,
    UserInputResponse,
    user_input_answer_request_digest,
    user_input_resolution_intent_for,
    user_input_resolution_request_digest,
)
from cayu.events import (
    Event,
    EventType,
)
from cayu.messages import (
    Message,
)
from cayu.runtime import _recovery_ownership as recovery_ownership_module
from cayu.runtime._event_projection import (
    public_event_linkage_id,
)
from cayu.runtime._interruption_coordinator import suppress_interruption_cascade
from cayu.sessions._checkpoint_mutations import (
    SessionRuntimePublicationConflict,
)
from cayu.sessions._invocation_terminal_decision import (
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryAction, IncompleteSessionRecoveryRequest
from cayu.sessions.requests import (
    InterruptSessionRequest,
    RunRequest,
)
from cayu.sessions.store import SessionStore
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.postgres


def test_session_store_conformance_user_input_open_close_process_loss_and_exact_replay(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"user-input-process-loss-{session_store_case[0]}"
        try:
            opening_publish = store.publish_runtime_publication
            lost_ack_kinds: set[str] = set()
            stale_instance_rejected = False
            open_event_authority_rejected = False

            async def publish_then_lose_ack(session_id: str, **kwargs):
                nonlocal open_event_authority_rejected, stale_instance_rejected
                request = kwargs["request"]
                if request.kind == "user-input-open" and not open_event_authority_rejected:
                    checkpoint_before = await store.load_checkpoint(session_id)
                    events_before = await store.load_events(session_id)
                    mutations = (
                        (0, "checkpoint", "pending_tool_round"),
                        (0, "source_run_epoch", request.intent["source_run_epoch"] + 1),
                        (
                            1,
                            "pause_digest",
                            "0" * 64 if request.intent["pause_digest"] != "0" * 64 else "1" * 64,
                        ),
                        (
                            1,
                            "execution_profile_fingerprint",
                            "0" * 64
                            if request.intent["execution_profile_fingerprint"] != "0" * 64
                            else "1" * 64,
                        ),
                    )
                    for event_index, field_name, value in mutations:
                        malformed_events = list(request.events)
                        source_event = malformed_events[event_index]
                        malformed_events[event_index] = source_event.model_copy(
                            update={"payload": {**source_event.payload, field_name: value}}
                        )
                        malformed_request = request.model_copy(
                            update={"events": tuple(malformed_events)}
                        )
                        with pytest.raises(
                            ValueError,
                            match="opening events conflict with their pause authority",
                        ):
                            await opening_publish(
                                session_id,
                                **{**kwargs, "request": malformed_request},
                            )
                        assert await store.load_checkpoint(session_id) == checkpoint_before
                        assert await store.load_events(session_id) == events_before
                        assert (
                            await store.load_runtime_publication_receipt(
                                session_id,
                                request.publication_id,
                            )
                            is None
                        )
                    open_event_authority_rejected = True
                if request.kind == "user-input-open" and not stale_instance_rejected:
                    stale_instance_rejected = True
                    stale_instance_id = f"stale-{uuid4()}"
                    malformed_intent = {
                        **request.intent,
                        "session_instance_id": stale_instance_id,
                    }
                    malformed_operations = tuple(
                        operation.model_copy(
                            update={
                                "value": {
                                    **cast("dict[str, Any]", operation.value),
                                    "session_instance_id": stale_instance_id,
                                }
                            }
                        )
                        if operation.key == "pending_user_input"
                        else operation
                        for operation in request.mutation.operations
                    )
                    malformed_request = request.model_copy(
                        update={
                            "intent": malformed_intent,
                            "mutation": request.mutation.model_copy(
                                update={"operations": malformed_operations}
                            ),
                        }
                    )
                    with pytest.raises(
                        (ValueError, SessionRuntimePublicationConflict),
                        match="session|invocation authority",
                    ):
                        await opening_publish(
                            session_id,
                            **{**kwargs, "request": malformed_request},
                        )
                result = await opening_publish(session_id, **kwargs)
                kind = request.kind
                if kind == "user-input-open" and kind not in lost_ack_kinds:
                    lost_ack_kinds.add(kind)
                    raise OSError("user-input-open process lost after commit")
                return result

            store.publish_runtime_publication = publish_then_lose_ack  # type: ignore[method-assign]
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "ask a question")],
                    )
                )
            ]
            awaiting = next(
                event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            opened_checkpoint = checkpoint
            private_input_id = checkpoint["pending_user_input"]["input_id"]
            open_receipt = await store.load_runtime_publication_receipt(
                session_id,
                f"user-input-open:{private_input_id}",
            )
            assert open_receipt is not None
            assert lost_ack_kinds == {"user-input-open"}
            assert stale_instance_rejected is True
            assert open_event_authority_rejected is True

            store = await _reopen_store(session_store_case, store)
            closing_publish = store.publish_runtime_publication
            omitted_lifecycle_evidence_rejected = False
            close_event_authority_rejected = False

            async def close_then_lose_ack(session_id: str, **kwargs):
                nonlocal close_event_authority_rejected, omitted_lifecycle_evidence_rejected
                request = kwargs["request"]
                if request.kind == "user-input-close" and not close_event_authority_rejected:
                    checkpoint_before = await store.load_checkpoint(session_id)
                    events_before = await store.load_events(session_id)
                    for field_name, value in (
                        ("checkpoint", "pending_tool_round"),
                        ("source_run_epoch", request.intent["source_run_epoch"] + 1),
                    ):
                        source_event = request.events[0]
                        malformed_request = request.model_copy(
                            update={
                                "events": (
                                    source_event.model_copy(
                                        update={
                                            "payload": {
                                                **source_event.payload,
                                                field_name: value,
                                            }
                                        }
                                    ),
                                )
                            }
                        )
                        with pytest.raises(
                            ValueError,
                            match="closure event conflicts with its intent",
                        ):
                            await closing_publish(
                                session_id,
                                **{**kwargs, "request": malformed_request},
                            )
                        assert await store.load_checkpoint(session_id) == checkpoint_before
                        assert await store.load_events(session_id) == events_before
                        assert (
                            await store.load_runtime_publication_receipt(
                                session_id,
                                request.publication_id,
                            )
                            is None
                        )
                    close_event_authority_rejected = True
                if request.kind == "user-input-close" and not omitted_lifecycle_evidence_rejected:
                    omitted_lifecycle_evidence_rejected = True
                    assert request.referenced_events
                    retained_references = request.referenced_events[:-1]
                    malformed_request = request.model_copy(
                        update={
                            "intent": {
                                **request.intent,
                                "referenced_event_ids": [
                                    reference.event_id for reference in retained_references
                                ],
                            },
                            "referenced_events": retained_references,
                        }
                    )
                    with pytest.raises(
                        SessionRuntimePublicationConflict,
                        match="bind every durable lifecycle event",
                    ):
                        await closing_publish(
                            session_id,
                            **{**kwargs, "request": malformed_request},
                        )
                result = await closing_publish(session_id, **kwargs)
                kind = request.kind
                if kind == "user-input-close" and kind not in lost_ack_kinds:
                    lost_ack_kinds.add(kind)
                    raise OSError("user-input-close process lost after commit")
                return result

            store.publish_runtime_publication = close_then_lose_ack  # type: ignore[method-assign]
            continuation_provider = _UserInputRecoveryProvider(complete_without_tools=True)
            resumed_app = CayuApp(session_store=store, enable_logging=False)
            resumed_app.register_provider(continuation_provider, default=True)
            resumed_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            response = UserInputResponse(
                session_id=session_id,
                input_id=awaiting.payload["input_id"],
                answer="yes",
                metadata={"source": "conformance"},
            )
            resolved = [event async for event in resumed_app.resolve_user_input(response)]
            assert resolved[-1].type is EventType.SESSION_COMPLETED
            assert lost_ack_kinds == {"user-input-open", "user-input-close"}
            assert omitted_lifecycle_evidence_rejected is True
            assert close_event_authority_rejected is True
            assert len(continuation_provider.requests) == 1
            close_receipt = await store.load_runtime_publication_receipt(
                session_id,
                f"user-input-close:{private_input_id}",
            )
            assert close_receipt is not None
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            assert "pending_user_input" not in checkpoint
            assert "user_input_resolution_intent" not in checkpoint

            store = await _reopen_store(session_store_case, store)
            replay_provider = _UserInputRecoveryProvider(complete_without_tools=True)
            replay_app = CayuApp(session_store=store, enable_logging=False)
            replay_app.register_provider(replay_provider, default=True)
            replay_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            replayed = [event async for event in replay_app.resolve_user_input(response)]
            private_replayed = [
                await _private_event_for_public_event(store, event) for event in replayed
            ]
            assert tuple(event.id for event in private_replayed) == (
                close_receipt.appended_event_ids
            )
            assert replay_provider.requests == []

            original_checkpoint_loader = store.load_checkpoint

            async def load_conflicting_open_checkpoint(session_id: str):
                assert session_id == response.session_id
                return opened_checkpoint

            store.load_checkpoint = load_conflicting_open_checkpoint  # type: ignore[method-assign]
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="conflicts with durable lifecycle state",
            ):
                _ = [event async for event in replay_app.resolve_user_input(response)]
            store.load_checkpoint = original_checkpoint_loader  # type: ignore[method-assign]

            original_receipt_loader = store.load_runtime_publication_receipt

            async def load_receipt_with_conflicting_open_authority(
                session_id: str,
                publication_id: str,
            ):
                loaded = await original_receipt_loader(session_id, publication_id)
                if loaded is None or publication_id != f"user-input-close:{private_input_id}":
                    return loaded
                return loaded.model_copy(
                    update={
                        "intent": {
                            **loaded.intent,
                            "execution_profile_fingerprint": "0" * 64,
                        }
                    }
                )

            store.load_runtime_publication_receipt = (  # type: ignore[method-assign]
                load_receipt_with_conflicting_open_authority
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="conflicts with durable lifecycle state",
            ):
                _ = [event async for event in replay_app.resolve_user_input(response)]
            store.load_runtime_publication_receipt = original_receipt_loader  # type: ignore[method-assign]
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="conflicting resolution authority",
            ):
                _ = [
                    event
                    async for event in replay_app.resolve_user_input(
                        response.model_copy(update={"answer": "different"})
                    )
                ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_user_input_open_close_cancellation_reconstructs_exact_receipts(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"user-input-cancellation-{session_store_case[0]}"
        try:
            original_publish = store.publish_runtime_publication
            open_committed = asyncio.Event()
            release_open_acknowledgement = asyncio.Event()
            open_blocked = False

            async def hold_open_acknowledgement(session_id: str, **kwargs):
                nonlocal open_blocked
                result = await original_publish(session_id, **kwargs)
                if kwargs["request"].kind == "user-input-open" and not open_blocked:
                    open_blocked = True
                    open_committed.set()
                    await release_open_acknowledgement.wait()
                return result

            store.publish_runtime_publication = hold_open_acknowledgement  # type: ignore[method-assign]
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            opening = asyncio.create_task(
                _collect_events(
                    app.run(
                        RunRequest(
                            session_id=session_id,
                            agent_name="assistant",
                            messages=[Message.text("user", "ask a question")],
                        )
                    )
                )
            )
            await asyncio.wait_for(open_committed.wait(), timeout=5)
            assert opening.cancelling() == 0
            opening.cancel("cancel user-input open acknowledgement")
            assert opening.cancelling() == 1
            release_open_acknowledgement.set()
            with pytest.raises(asyncio.CancelledError) as open_cancellation:
                await opening
            assert open_cancellation.value.args == ("cancel user-input open acknowledgement",)
            assert opening.cancelled() is True
            # The publication owner consumes the request while it settles the
            # committed child, then raises the preserved cancellation.
            assert opening.cancelling() == 0
            store.publish_runtime_publication = original_publish  # type: ignore[method-assign]

            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            private_input_id = checkpoint["pending_user_input"]["input_id"]
            open_receipt = await store.load_runtime_publication_receipt(
                session_id,
                f"user-input-open:{private_input_id}",
            )
            assert open_receipt is not None
            awaiting_records = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_AWAITING_USER_INPUT,
                )
            )
            assert len(awaiting_records) == 1
            public_input_id = public_event_linkage_id(
                awaiting_records[0].sequence,
                "input_id",
            )

            store = await _reopen_store(session_store_case, store)
            original_publish = store.publish_runtime_publication
            close_committed = asyncio.Event()
            release_close_acknowledgement = asyncio.Event()
            close_blocked = False

            async def hold_close_acknowledgement(session_id: str, **kwargs):
                nonlocal close_blocked
                result = await original_publish(session_id, **kwargs)
                if kwargs["request"].kind == "user-input-close" and not close_blocked:
                    close_blocked = True
                    close_committed.set()
                    await release_close_acknowledgement.wait()
                return result

            store.publish_runtime_publication = hold_close_acknowledgement  # type: ignore[method-assign]
            resumed_provider = _UserInputRecoveryProvider(complete_without_tools=True)
            resumed_app = CayuApp(session_store=store, enable_logging=False)
            resumed_app.register_provider(resumed_provider, default=True)
            resumed_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            response = UserInputResponse(
                session_id=session_id,
                input_id=public_input_id,
                answer="yes",
                metadata={"source": "cancellation-conformance"},
            )
            closing = asyncio.create_task(_collect_events(resumed_app.resolve_user_input(response)))
            await asyncio.wait_for(close_committed.wait(), timeout=5)
            assert closing.cancelling() == 0
            closing.cancel("cancel user-input close acknowledgement")
            assert closing.cancelling() == 1
            release_close_acknowledgement.set()
            with pytest.raises(asyncio.CancelledError) as close_cancellation:
                await closing
            assert close_cancellation.value.args == ("cancel user-input close acknowledgement",)
            assert closing.cancelled() is True
            assert closing.cancelling() == 0
            store.publish_runtime_publication = original_publish  # type: ignore[method-assign]

            close_receipt = await store.load_runtime_publication_receipt(
                session_id,
                f"user-input-close:{private_input_id}",
            )
            assert close_receipt is not None
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            assert "pending_user_input" not in checkpoint
            assert "user_input_resolution_intent" not in checkpoint
            assert resumed_provider.requests == []

            store = await _reopen_store(session_store_case, store)
            replay_provider = _UserInputRecoveryProvider(complete_without_tools=True)
            replay_app = CayuApp(session_store=store, enable_logging=False)
            replay_app.register_provider(replay_provider, default=True)
            replay_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            replayed = [event async for event in replay_app.resolve_user_input(response)]
            private_replayed = [
                await _private_event_for_public_event(store, event) for event in replayed
            ]
            assert tuple(event.id for event in private_replayed) == close_receipt.appended_event_ids
            assert replay_provider.requests == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_supersedes_fully_paused_user_input(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"user-input-steady-supersession-{session_store_case[0]}"
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "ask a question")],
                    )
                )
            ]
            paused_session = await store.load(session_id)
            assert paused_session is not None
            assert paused_session.status is SessionStatus.INTERRUPTED
            awaiting = next(
                event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            private_input_id = checkpoint["pending_user_input"]["input_id"]

            request = InterruptSessionRequest(
                session_id=session_id,
                reason="operator supersedes fully published pause",
            )
            with suppress_interruption_cascade():
                interrupted = await _collect_events(app.interrupt_session(request))
            assert interrupted[-1].type is EventType.SESSION_INTERRUPTED
            assert interrupted[-1].payload["interruption_type"] == "operator_requested"

            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            assert "pending_user_input" not in checkpoint
            assert "user_input_resolution_intent" not in checkpoint
            operator_records = [
                record
                for record in await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_type=EventType.SESSION_INTERRUPTED,
                    )
                )
                if record.event.payload.get("interruption_type") == "operator_requested"
            ]
            assert len(operator_records) == 1
            marker = operator_records[0].event.payload["user_input_supersession_intent"]
            assert marker["state"] == "active"
            assert marker["input_id"] == private_input_id

            store = await _reopen_store(session_store_case, store)
            replay_app = CayuApp(session_store=store, enable_logging=False)
            replay_app.register_provider(_UserInputRecoveryProvider(), default=True)
            replay_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            with suppress_interruption_cascade():
                replayed = await _collect_events(replay_app.interrupt_session(request))
            assert replayed[-1].id == interrupted[-1].id
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="superseded by an external interruption",
            ):
                _ = [
                    event
                    async for event in replay_app.resolve_user_input(
                        UserInputResponse(
                            session_id=session_id,
                            input_id=awaiting.payload["input_id"],
                            answer="stale",
                        )
                    )
                ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_recovers_user_input_supersession_after_process_loss(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"user-input-supersession-process-loss-{session_store_case[0]}"
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            await _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "ask a question")],
                    )
                )
            )

            original_transition = store.transition_status_and_checkpoint
            process_loss_injected = False

            async def commit_interrupt_then_lose_ack(*args, **kwargs):
                nonlocal process_loss_injected
                transitioned = await original_transition(*args, **kwargs)
                if (
                    kwargs.get("to_status") is SessionStatus.INTERRUPTING
                    and not process_loss_injected
                ):
                    process_loss_injected = True
                    raise _SimulatedProcessLoss(
                        "process stopped after user-input supersession committed"
                    )
                return transitioned

            store.transition_status_and_checkpoint = (  # type: ignore[method-assign]
                commit_interrupt_then_lose_ack
            )
            request = InterruptSessionRequest(
                session_id=session_id,
                reason="operator supersedes the paused question",
            )
            with (
                suppress_interruption_cascade(),
                pytest.raises(
                    _SimulatedProcessLoss,
                    match="after user-input supersession committed",
                ),
            ):
                await _collect_events(app.interrupt_session(request))
            assert process_loss_injected is True
            store.transition_status_and_checkpoint = original_transition  # type: ignore[method-assign]

            interrupted = await store.load(session_id)
            checkpoint = await store.load_checkpoint(session_id)
            assert interrupted is not None
            assert interrupted.status is SessionStatus.INTERRUPTING
            assert checkpoint is not None
            assert "pending_user_input" not in checkpoint
            pending_interrupt = checkpoint["pending_session_interrupt"]
            assert pending_interrupt["interruption_type"] == "operator_requested"
            assert pending_interrupt["user_input_supersession_intent"]["state"] == "active"

            store = await _reopen_store(session_store_case, store)
            recovery_app = CayuApp(session_store=store, enable_logging=False)
            recovery_app.register_provider(_UserInputRecoveryProvider(), default=True)
            recovery_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            with suppress_interruption_cascade():
                recovered = await _collect_events(recovery_app.interrupt_session(request))
            assert recovered[-1].type is EventType.SESSION_INTERRUPTED
            assert recovered[-1].payload["interruption_type"] == "operator_requested"

            finalized = await store.load(session_id)
            finalized_checkpoint = await store.load_checkpoint(session_id)
            assert finalized is not None
            assert finalized.status is SessionStatus.INTERRUPTED
            assert finalized_checkpoint is not None
            assert "pending_session_interrupt" not in finalized_checkpoint
            operator_records = [
                record
                for record in await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_type=EventType.SESSION_INTERRUPTED,
                    )
                )
                if record.event.payload.get("interruption_type") == "operator_requested"
            ]
            assert len(operator_records) == 1
            assert (
                operator_records[0].event.id
                == (await _private_event_for_public_event(store, recovered[-1])).id
            )

            with suppress_interruption_cascade():
                replayed = await _collect_events(recovery_app.interrupt_session(request))
            assert replayed[-1].id == recovered[-1].id
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("preparation_failure_type", "expected_failure_type", "expected_message"),
    [
        pytest.param(
            OSError,
            OSError,
            "provider cancellation preparation failed",
            id="ordinary-failure",
        ),
        pytest.param(
            ValueError,
            TimeoutError,
            "Session interruption is still finalizing",
            id="value-error-reconciliation",
        ),
        pytest.param(
            SessionRuntimePublicationConflict,
            SessionRuntimePublicationConflict,
            "provider cancellation preparation failed",
            id="publication-conflict",
        ),
    ],
)
def test_session_store_conformance_releases_offline_supersession_claim_after_preparation_failure(
    session_store_case,
    preparation_failure_type,
    expected_failure_type,
    expected_message: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"offline-supersession-preparation-failure-{session_store_case[0]}"
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            await _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "ask a question")],
                    )
                )
            )
            request = InterruptSessionRequest(
                session_id=session_id,
                reason="operator supersedes the paused question",
            )
            original_cancel_provider = (
                app._model_completion_recovery.cancel_provider_operation_for_interruption
            )

            async def fail_after_terminal_claim(*args, **kwargs):
                checkpoint = await store.load_checkpoint(session_id)
                assert checkpoint is not None
                assert "incomplete_session_recovery_claim" in checkpoint
                raise preparation_failure_type("provider cancellation preparation failed")

            app._model_completion_recovery.cancel_provider_operation_for_interruption = (  # type: ignore[method-assign]
                fail_after_terminal_claim
            )
            with (
                suppress_interruption_cascade(),
                pytest.raises(expected_failure_type, match=expected_message),
            ):
                await _collect_events(app.interrupt_session(request))
            app._model_completion_recovery.cancel_provider_operation_for_interruption = (  # type: ignore[method-assign]
                original_cancel_provider
            )

            retained = await store.load_checkpoint(session_id)
            assert retained is not None
            assert "pending_session_interrupt" in retained
            assert "incomplete_session_recovery_claim" not in retained
            with suppress_interruption_cascade():
                events = await _collect_events(app.interrupt_session(request))
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            final_checkpoint = await store.load_checkpoint(session_id)
            assert final_checkpoint is not None
            assert "pending_session_interrupt" not in final_checkpoint
            assert "incomplete_session_recovery_claim" not in final_checkpoint
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "loss_boundary",
    [
        pytest.param("before-publication", id="before-publication"),
        pytest.param("after-publication", id="after-publication"),
    ],
)
@pytest.mark.parametrize(
    "replay_entrance",
    [
        pytest.param("interrupt", id="interrupt-retry"),
        pytest.param("incomplete-recovery", id="incomplete-recovery"),
    ],
)
def test_session_store_conformance_replays_user_input_supersession_after_terminal_process_loss(
    session_store_case,
    loss_boundary: str,
    replay_entrance: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        peer_store: SessionStore | None = None
        retry_release: asyncio.Event | None = None
        retry_tasks: list[asyncio.Task[list[Event]]] = []
        session_id = (
            "user-input-terminal-supersession-process-loss-"
            f"{loss_boundary}-{replay_entrance}-{session_store_case[0]}"
        )
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            paused = await _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "ask a question")],
                    )
                )
            )
            awaiting = next(
                event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )

            process_loss_injected = False
            if loss_boundary == "before-publication":
                engine = app._session_engine
                original_terminal_preparation = (
                    engine._session_finalization.prepare_terminal_event_for_atomic_transition
                )

                async def prepare_terminal_then_lose_process(*args, **kwargs):
                    nonlocal process_loss_injected
                    transitioned = await original_terminal_preparation(*args, **kwargs)
                    if not process_loss_injected:
                        process_loss_injected = True
                        raise _SimulatedProcessLoss(
                            "process stopped after terminal preparation before publication"
                        )
                    return transitioned

                engine._session_finalization.prepare_terminal_event_for_atomic_transition = (  # type: ignore[method-assign]
                    prepare_terminal_then_lose_process
                )
            else:
                engine = app._session_engine
                original_publish_terminal = (
                    engine._session_finalization.publish_interaction_transition
                )

                async def commit_terminal_event_then_lose_process(*args, **kwargs):
                    nonlocal process_loss_injected
                    published = await original_publish_terminal(*args, **kwargs)
                    if not process_loss_injected:
                        process_loss_injected = True
                        raise _SimulatedProcessLoss(
                            "process stopped after atomic terminal publication"
                        )
                    return published

                engine._session_finalization.publish_interaction_transition = (  # type: ignore[method-assign]
                    commit_terminal_event_then_lose_process
                )

            request = InterruptSessionRequest(
                session_id=session_id,
                reason="operator supersedes the paused question",
            )
            with (
                suppress_interruption_cascade(),
                pytest.raises(_SimulatedProcessLoss),
            ):
                await _collect_events(app.interrupt_session(request))
            assert process_loss_injected is True

            interrupted = await store.load(session_id)
            checkpoint = await store.load_checkpoint(session_id)
            assert interrupted is not None
            assert interrupted.status is (
                SessionStatus.INTERRUPTING
                if loss_boundary == "before-publication"
                else SessionStatus.INTERRUPTED
            )
            assert checkpoint is not None
            assert "pending_user_input" not in checkpoint
            assert "user_input_resolution_intent" not in checkpoint
            before_replay = [
                record
                for record in await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_type=EventType.SESSION_INTERRUPTED,
                    )
                )
                if record.event.payload.get("interruption_type") == "operator_requested"
            ]
            assert len(before_replay) == (0 if loss_boundary == "before-publication" else 1)
            if loss_boundary == "before-publication":
                assert (
                    checkpoint["pending_session_interrupt"]["user_input_supersession_intent"][
                        "state"
                    ]
                    == "active"
                )
            else:
                assert "pending_session_interrupt" in checkpoint
                assert settled_invocation_terminal_decision_from_checkpoint(checkpoint) is not None

            if loss_boundary == "before-publication":
                engine._session_finalization.prepare_terminal_event_for_atomic_transition = (  # type: ignore[method-assign]
                    original_terminal_preparation
                )
            else:
                engine._session_finalization.publish_interaction_transition = (  # type: ignore[method-assign]
                    original_publish_terminal
                )
            store = await _reopen_store(session_store_case, store)
            supersession = (
                checkpoint["pending_session_interrupt"]["user_input_supersession_intent"]
                if loss_boundary == "before-publication"
                else before_replay[0].event.payload["user_input_supersession_intent"]
            )
            replay_app = CayuApp(
                session_store=store,
                enable_logging=False,
                secret_redactor=SecretRedactor(supersession["input_id"][4:12]),
            )
            replay_app.register_provider(_UserInputRecoveryProvider(), default=True)
            replay_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"), tools=[UserInputTool()]
            )

            if replay_entrance == "incomplete-recovery" and loss_boundary == "before-publication":
                original_receipt_loader = store.load_runtime_publication_receipt

                async def hide_open_receipt(*args, **kwargs):
                    publication_id = args[1] if len(args) > 1 else kwargs["publication_id"]
                    if publication_id == f"user-input-open:{supersession['input_id']}":
                        return None
                    return await original_receipt_loader(*args, **kwargs)

                store.load_runtime_publication_receipt = (  # type: ignore[method-assign]
                    hide_open_receipt
                )
                with pytest.raises(
                    SessionRuntimePublicationConflict,
                    match="no exact durable opening receipt",
                ):
                    await replay_app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(session_id=session_id)
                    )
                store.load_runtime_publication_receipt = (  # type: ignore[method-assign]
                    original_receipt_loader
                )
                retained = await store.load_checkpoint(session_id)
                assert retained is not None
                assert (
                    retained["pending_session_interrupt"] == checkpoint["pending_session_interrupt"]
                )
            if replay_entrance == "interrupt" and loss_boundary == "before-publication":
                peer_store = await _open_peer_store(session_store_case, store)
                peer_app = CayuApp(
                    session_store=peer_store,
                    enable_logging=False,
                    secret_redactor=SecretRedactor(supersession["input_id"][4:12]),
                )
                peer_app.register_provider(_UserInputRecoveryProvider(), default=True)
                peer_app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"), tools=[UserInputTool()]
                )
                repair_clear_started = asyncio.Event()
                release_repair_clear = asyncio.Event()
                retry_release = release_repair_clear
                peer_retry_started = asyncio.Event()
                replay_engine = replay_app._session_engine
                original_repair_clear = (
                    replay_engine._session_finalization.clear_pending_session_interrupt
                )

                async def hold_repair_before_marker_clear(*args, **kwargs):
                    repair_clear_started.set()
                    await release_repair_clear.wait()
                    return await original_repair_clear(*args, **kwargs)

                async def run_competing_retry():
                    peer_retry_started.set()
                    return await _collect_events(peer_app.interrupt_session(request))

                replay_engine._session_finalization.clear_pending_session_interrupt = (  # type: ignore[method-assign]
                    hold_repair_before_marker_clear
                )
                with suppress_interruption_cascade():
                    first_retry = asyncio.create_task(
                        _collect_events(replay_app.interrupt_session(request))
                    )
                    retry_tasks.append(first_retry)
                    for _attempt in range(1000):
                        if repair_clear_started.is_set():
                            break
                        if first_retry.done():
                            await first_retry
                        await asyncio.sleep(0.01)
                    assert repair_clear_started.is_set(), (
                        "terminal repair did not reach cleanup",
                        [event.type for event in first_retry.result()]
                        if first_retry.done()
                        else [frame.f_code.co_name for frame in first_retry.get_stack()],
                    )
                    second_retry = asyncio.create_task(run_competing_retry())
                    retry_tasks.append(second_retry)
                    await asyncio.wait_for(peer_retry_started.wait(), timeout=5)
                    # The event is already committed: a peer may acknowledge
                    # that exact winner while the first caller clears metadata.
                    release_repair_clear.set()
                    first_replayed, second_replayed = await asyncio.wait_for(
                        asyncio.gather(first_retry, second_retry),
                        timeout=10,
                    )
                replay_engine._session_finalization.clear_pending_session_interrupt = (  # type: ignore[method-assign]
                    original_repair_clear
                )
                assert first_replayed[-1].id == second_replayed[-1].id
                replayed = first_replayed
            elif replay_entrance == "incomplete-recovery" and loss_boundary == "before-publication":
                recovered = await replay_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                )
                assert recovered.actions == (IncompleteSessionRecoveryAction.FINALIZED_INTERRUPT,)
                replayed = [
                    event
                    for event in recovered.events
                    if event.type is EventType.SESSION_INTERRUPTED
                ]
                assert len(replayed) == 1
            else:
                if replay_entrance == "incomplete-recovery":
                    recovered = await replay_app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(session_id=session_id)
                    )
                    assert recovered.actions == (
                        IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
                    )
                    returned_terminals = [
                        event
                        for event in recovered.events
                        if event.type is EventType.SESSION_INTERRUPTED
                    ]
                    assert len(returned_terminals) == 1
                    assert (
                        await _private_event_for_public_event(store, returned_terminals[0])
                    ).id == before_replay[0].event.id
                with suppress_interruption_cascade():
                    replayed = await _collect_events(replay_app.interrupt_session(request))
            assert replayed[-1].type is EventType.SESSION_INTERRUPTED
            final_checkpoint = await store.load_checkpoint(session_id)
            assert final_checkpoint is not None
            assert "pending_session_interrupt" not in final_checkpoint
            terminal_records = [
                record
                for record in await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_type=EventType.SESSION_INTERRUPTED,
                    )
                )
                if record.event.payload.get("interruption_type") == "operator_requested"
            ]
            assert len(terminal_records) == 1
            assert (
                terminal_records[0].event.payload["user_input_supersession_intent"]["input_id"]
                == supersession["input_id"]
            )
            assert (
                terminal_records[0].event.id
                == (await _private_event_for_public_event(store, replayed[-1])).id
            )

            with suppress_interruption_cascade():
                exact_retry = await _collect_events(replay_app.interrupt_session(request))
            assert exact_retry[-1].id == replayed[-1].id
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="superseded by an external interruption",
            ):
                await _collect_events(
                    replay_app.resolve_user_input(
                        UserInputResponse(
                            session_id=session_id,
                            input_id=awaiting.payload["input_id"],
                            answer="stale",
                        )
                    )
                )
        finally:
            if retry_release is not None:
                retry_release.set()
            for retry_task in retry_tasks:
                if not retry_task.done():
                    retry_task.cancel()
            if retry_tasks:
                await asyncio.gather(*retry_tasks, return_exceptions=True)
            if peer_store is not None and peer_store is not store:
                await _close_store(peer_store)
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "corruption",
    [
        pytest.param("orphan-intent", id="orphan-intent"),
        pytest.param("future-claim", id="future-claim"),
    ],
)
def test_session_store_conformance_rejects_ambiguous_user_input_claim_topology(
    session_store_case,
    corruption: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"user-input-ambiguous-claim-{corruption}-{session_store_case[0]}"
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            await _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "ask a question")],
                    )
                )
            )
            session = await store.load(session_id)
            checkpoint = await store.load_checkpoint(session_id)
            assert session is not None
            assert checkpoint is not None
            pending = PendingUserInput.model_validate(checkpoint[PENDING_USER_INPUT_CHECKPOINT_KEY])
            claim_run_epoch = max(session.run_epoch + 16, pending.source_run_epoch + 1)
            intent = user_input_resolution_intent_for(
                pending,
                answer_request_digest="b" * 64,
                resolution_stage="answer",
                resolution_request_digest="c" * 64,
                claim_run_epoch=claim_run_epoch,
            )

            def corrupt_claim_topology(_session: Session, current):
                updated = dict({} if current is None else current)
                updated[USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY] = intent.model_dump(
                    mode="json"
                )
                if corruption == "orphan-intent":
                    updated.pop(PENDING_USER_INPUT_CHECKPOINT_KEY)
                elif corruption != "future-claim":  # pragma: no cover - exhaustive parameter
                    raise AssertionError(f"Unknown corruption: {corruption}")
                return updated

            await store.transform_checkpoint(session_id, corrupt_claim_topology)
            store = await _reopen_store(session_store_case, store)
            replay_app = CayuApp(session_store=store, enable_logging=False)
            replay_app.register_provider(_UserInputRecoveryProvider(), default=True)
            replay_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )

            with pytest.raises(RuntimeError, match="User-input .*authority|session run epoch"):
                await replay_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                )

            retained = await store.load_checkpoint(session_id)
            assert retained is not None
            assert retained[USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY] == intent.model_dump(
                mode="json"
            )
            if corruption == "orphan-intent":
                assert PENDING_USER_INPUT_CHECKPOINT_KEY not in retained
            else:
                assert retained[PENDING_USER_INPUT_CHECKPOINT_KEY] == pending.model_dump(
                    mode="json"
                )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reconstructs_active_user_input_supersession(
    session_store_case,
    monkeypatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        peer_store: SessionStore | None = None
        release_handler = asyncio.Event()
        session_id = f"user-input-active-supersession-{session_store_case[0]}"
        try:
            monkeypatch.setattr(
                recovery_ownership_module,
                "_INCOMPLETE_RECOVERY_CLAIM_LEASE",
                timedelta(seconds=1),
            )
            monkeypatch.setattr(
                recovery_ownership_module,
                "_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
                0.02,
            )
            monkeypatch.setattr(
                recovery_ownership_module,
                "_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_RETRY_SECONDS",
                0.01,
            )
            awaiting_event_id: str | None = None
            original_publish = store.publish_runtime_publication
            original_claim = store.claim_persisted_event_side_effect
            fan_out_started = asyncio.Event()

            async def remember_awaiting_event(session_id: str, **kwargs):
                nonlocal awaiting_event_id
                result = await original_publish(session_id, **kwargs)
                if kwargs["request"].kind == "user-input-open":
                    awaiting_event_id = next(
                        event.id
                        for event in kwargs["request"].events
                        if event.type is EventType.SESSION_AWAITING_USER_INPUT
                    )
                return result

            async def block_awaiting_fan_out(*, session_id: str, event_id: str):
                if event_id == awaiting_event_id:
                    fan_out_started.set()
                    await asyncio.Event().wait()
                return await original_claim(session_id=session_id, event_id=event_id)

            store.publish_runtime_publication = remember_awaiting_event  # type: ignore[method-assign]
            store.claim_persisted_event_side_effect = block_awaiting_fan_out  # type: ignore[method-assign]
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            handler_entered = asyncio.Event()
            original_handle_interrupted = app._session_finalization.handle_session_interrupted

            async def hold_live_handler(*args, **kwargs):
                handler_entered.set()
                await release_handler.wait()
                async for event in original_handle_interrupted(*args, **kwargs):
                    yield event

            app._session_finalization.handle_session_interrupted = (  # type: ignore[method-assign]
                hold_live_handler
            )
            running = asyncio.create_task(
                _collect_events(
                    app.run(
                        RunRequest(
                            session_id=session_id,
                            agent_name="assistant",
                            messages=[Message.text("user", "ask a question")],
                        )
                    )
                )
            )
            await asyncio.wait_for(fan_out_started.wait(), timeout=10)
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            private_input_id = checkpoint["pending_user_input"]["input_id"]
            awaiting_records = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_AWAITING_USER_INPUT,
                )
            )
            assert len(awaiting_records) == 1
            public_input_id = public_event_linkage_id(
                awaiting_records[0].sequence,
                "input_id",
            )

            request = InterruptSessionRequest(
                session_id=session_id,
                reason="operator supersedes active pause",
            )
            with suppress_interruption_cascade():
                interrupting = asyncio.create_task(_collect_events(app.interrupt_session(request)))
                await asyncio.wait_for(handler_entered.wait(), timeout=10)
                claimed_checkpoint = await store.load_checkpoint(session_id)
                assert claimed_checkpoint is not None
                durable_claim = claimed_checkpoint["incomplete_session_recovery_claim"]

                # The live worker is delayed beyond the original lease before
                # it can enter terminal publication. Its retained handoff must
                # renew the exact claim rather than letting a peer take it.
                await asyncio.sleep(1.2)
                renewed_checkpoint = await store.load_checkpoint(session_id)
                assert renewed_checkpoint is not None
                renewed_claim = renewed_checkpoint["incomplete_session_recovery_claim"]
                assert renewed_claim["claim_id"] == durable_claim["claim_id"]
                assert renewed_claim["claim_expires_at"] > durable_claim["claim_expires_at"]

                peer_store = await _open_peer_store(session_store_case, store)
                peer_app = CayuApp(session_store=peer_store, enable_logging=False)
                peer_app.register_provider(_UserInputRecoveryProvider(), default=True)
                peer_app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    tools=[UserInputTool()],
                )
                claim_entered = asyncio.Event()
                original_transfer = (
                    peer_app._recovery_coordinator.terminal_finalization.claim_pending
                )

                async def transfer_after_live_settlement(**kwargs):
                    # Force the real owner to finish after the peer has resolved
                    # pending authority, but before its atomic claim comparison.
                    claim_entered.set()
                    await asyncio.gather(running, interrupting)
                    settled_session = await peer_store.load(session_id)
                    settled_checkpoint = await peer_store.load_checkpoint(session_id)
                    for changed in (
                        {
                            **kwargs,
                            "session": kwargs["session"].model_copy(
                                update={"instance_id": "different-incarnation"}
                            ),
                        },
                        {
                            **kwargs,
                            "session": kwargs["session"].model_copy(
                                update={"run_epoch": settled_session.run_epoch + 1}
                            ),
                        },
                        {
                            **kwargs,
                            "expected_payload": {
                                **kwargs["expected_payload"],
                                "interruption_request_id": "other-request",
                            },
                        },
                        {
                            **kwargs,
                            "expected_payload": {
                                **kwargs["expected_payload"],
                                "reason": "different-reason",
                            },
                        },
                    ):
                        with pytest.raises((SessionRuntimePublicationConflict, RuntimeError)):
                            await original_transfer(**changed)
                    result = await original_transfer(**kwargs)
                    assert isinstance(result, Event)
                    readback_entered = asyncio.Event()

                    async def hold_readback(*args, **read_kwargs):
                        readback_entered.set()
                        await asyncio.Event().wait()

                    with monkeypatch.context() as cancellation_patch:
                        cancellation_patch.setattr(
                            peer_app._session_control, "latest_interrupted_event", hold_readback
                        )
                        cancelled = asyncio.create_task(original_transfer(**kwargs))
                        await asyncio.wait_for(readback_entered.wait(), timeout=10)
                        cancelled.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await cancelled
                        assert cancelled.cancelled() and cancelled.cancelling() == 1
                    assert await peer_store.load(session_id) == settled_session
                    assert await peer_store.load_checkpoint(session_id) == settled_checkpoint
                    return result

                monkeypatch.setattr(
                    peer_app._recovery_coordinator.terminal_finalization,
                    "claim_pending",
                    transfer_after_live_settlement,
                )
                retrying = asyncio.create_task(_collect_events(peer_app.interrupt_session(request)))
                await asyncio.wait_for(claim_entered.wait(), timeout=10)
                assert not retrying.done()
                release_handler.set()
                run_events, interrupt_events, retry_events = await asyncio.wait_for(
                    asyncio.gather(running, interrupting, retrying),
                    timeout=10,
                )
            assert retry_events[-1].id == interrupt_events[-1].id
            assert run_events[-1].type is EventType.SESSION_INTERRUPTED
            assert interrupt_events[-1].type is EventType.SESSION_INTERRUPTED
            store.publish_runtime_publication = original_publish  # type: ignore[method-assign]
            store.claim_persisted_event_side_effect = original_claim  # type: ignore[method-assign]

            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            assert "pending_user_input" not in checkpoint
            assert "user_input_resolution_intent" not in checkpoint
            operator_records = [
                record
                for record in await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_type=EventType.SESSION_INTERRUPTED,
                    )
                )
                if record.event.payload.get("interruption_type") == "operator_requested"
            ]
            assert len(operator_records) == 1
            marker = operator_records[0].event.payload["user_input_supersession_intent"]
            assert marker["state"] == "active"
            assert marker["input_id"] == private_input_id

            await store.append_events(
                session_id,
                [
                    Event(
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=session_id,
                        agent_name="assistant",
                        payload={
                            "interruption_type": "supersession_lookup_noise",
                            "input_id": f"unrelated-input-{index}",
                        },
                    )
                    for index in range(65)
                ],
            )

            store = await _reopen_store(session_store_case, store)
            replay_app = CayuApp(session_store=store, enable_logging=False)
            replay_app.register_provider(_UserInputRecoveryProvider(), default=True)
            replay_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            original_supersession_loader = store.load_user_input_supersession_events

            async def load_conflicting_supersession(
                session_id: str,
                input_id: str,
            ) -> list[Event]:
                events = await original_supersession_loader(session_id, input_id)
                assert len(events) == 1
                event = events[0]
                supersession = {
                    **event.payload["user_input_supersession_intent"],
                    "execution_profile_fingerprint": "0" * 64,
                }
                return [
                    event.model_copy(
                        update={
                            "payload": {
                                **event.payload,
                                "user_input_supersession_intent": supersession,
                            }
                        }
                    )
                ]

            store.load_user_input_supersession_events = (  # type: ignore[method-assign]
                load_conflicting_supersession
            )
            with pytest.raises(RuntimeError, match="no pending user input"):
                _ = [
                    event
                    async for event in replay_app.resolve_user_input(
                        UserInputResponse(
                            session_id=session_id,
                            input_id=public_input_id,
                            answer="stale",
                        )
                    )
                ]
            store.load_user_input_supersession_events = (  # type: ignore[method-assign]
                original_supersession_loader
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="superseded by an external interruption",
            ):
                _ = [
                    event
                    async for event in replay_app.resolve_user_input(
                        UserInputResponse(
                            session_id=session_id,
                            input_id=public_input_id,
                            answer="stale",
                        )
                    )
                ]
        finally:
            release_handler.set()
            if peer_store is not None and peer_store is not store:
                await _close_store(peer_store)
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "read_boundary",
    [
        pytest.param("session", id="session-read"),
        pytest.param("checkpoint", id="checkpoint-read"),
    ],
)
def test_session_store_conformance_releases_live_supersession_handoff_after_read_failure(
    session_store_case,
    read_boundary: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"live-supersession-read-failure-{read_boundary}-{session_store_case[0]}"
        interrupting: asyncio.Task[list[Event]] | None = None
        running: asyncio.Task[list[Event]] | None = None
        try:
            awaiting_event_id: str | None = None
            original_publish = store.publish_runtime_publication
            original_side_effect_claim = store.claim_persisted_event_side_effect
            fan_out_started = asyncio.Event()

            async def remember_awaiting_event(session_id: str, **kwargs):
                nonlocal awaiting_event_id
                result = await original_publish(session_id, **kwargs)
                if kwargs["request"].kind == "user-input-open":
                    awaiting_event_id = next(
                        event.id
                        for event in kwargs["request"].events
                        if event.type is EventType.SESSION_AWAITING_USER_INPUT
                    )
                return result

            async def block_awaiting_fan_out(*, session_id: str, event_id: str):
                if event_id == awaiting_event_id:
                    fan_out_started.set()
                    await asyncio.Event().wait()
                return await original_side_effect_claim(
                    session_id=session_id,
                    event_id=event_id,
                )

            store.publish_runtime_publication = remember_awaiting_event  # type: ignore[method-assign]
            store.claim_persisted_event_side_effect = block_awaiting_fan_out  # type: ignore[method-assign]
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )

            original_handler = app._session_finalization.handle_session_interrupted
            handler_finished = asyncio.Event()
            read_failure_injected = False
            handoff_read_scope: contextvars.ContextVar[bool] = contextvars.ContextVar(
                "handoff_read_scope",
                default=False,
            )

            async def fail_first_handoff_read(*args, **kwargs):
                nonlocal read_failure_injected
                target_session = kwargs["session"]
                if (
                    target_session.id
                    not in app._session_engine._session_control._terminal_finalization_claim_handoffs
                ):
                    async for event in original_handler(*args, **kwargs):
                        yield event
                    return
                original_load = store.load
                original_load_checkpoint = store.load_checkpoint

                async def fail_session_read(loaded_session_id: str):
                    nonlocal read_failure_injected
                    if handoff_read_scope.get() and not read_failure_injected:
                        read_failure_injected = True
                        raise OSError("live handoff session read failed")
                    return await original_load(loaded_session_id)

                async def fail_checkpoint_read(loaded_session_id: str):
                    nonlocal read_failure_injected
                    if handoff_read_scope.get() and not read_failure_injected:
                        read_failure_injected = True
                        raise OSError("live handoff checkpoint read failed")
                    return await original_load_checkpoint(loaded_session_id)

                if read_boundary == "session":
                    store.load = fail_session_read  # type: ignore[method-assign]
                else:
                    store.load_checkpoint = fail_checkpoint_read  # type: ignore[method-assign]
                scope_token = handoff_read_scope.set(True)
                try:
                    async for event in original_handler(*args, **kwargs):
                        yield event
                finally:
                    handoff_read_scope.reset(scope_token)
                    store.load = original_load  # type: ignore[method-assign]
                    store.load_checkpoint = original_load_checkpoint  # type: ignore[method-assign]
                    handler_finished.set()

            app._session_finalization.handle_session_interrupted = (  # type: ignore[method-assign]
                fail_first_handoff_read
            )
            running = asyncio.create_task(
                _collect_events(
                    app.run(
                        RunRequest(
                            session_id=session_id,
                            agent_name="assistant",
                            messages=[Message.text("user", "ask a question")],
                        )
                    )
                )
            )
            await asyncio.wait_for(fan_out_started.wait(), timeout=10)
            request = InterruptSessionRequest(
                session_id=session_id,
                reason="operator supersedes active pause",
            )
            with suppress_interruption_cascade():
                interrupting = asyncio.create_task(_collect_events(app.interrupt_session(request)))
                await asyncio.wait_for(handler_finished.wait(), timeout=5)
            assert read_failure_injected is True
            await asyncio.wait_for(
                asyncio.gather(running, return_exceptions=True),
                timeout=5,
            )
            if not interrupting.done():
                interrupting.cancel("stop failed interruption observer")
            await asyncio.gather(interrupting, return_exceptions=True)

            retained = await store.load_checkpoint(session_id)
            assert retained is not None
            assert "pending_session_interrupt" in retained
            assert "incomplete_session_recovery_claim" not in retained
            assert (
                session_id
                not in app._session_engine._session_control._terminal_finalization_claim_handoffs
            )

            store.publish_runtime_publication = original_publish  # type: ignore[method-assign]
            store.claim_persisted_event_side_effect = (  # type: ignore[method-assign]
                original_side_effect_claim
            )
            app._session_finalization.handle_session_interrupted = (  # type: ignore[method-assign]
                original_handler
            )
            with suppress_interruption_cascade():
                retried = await _collect_events(app.interrupt_session(request))
            assert retried[-1].type is EventType.SESSION_INTERRUPTED
            final_checkpoint = await store.load_checkpoint(session_id)
            assert final_checkpoint is not None
            assert "pending_session_interrupt" not in final_checkpoint
            terminal_records = [
                record
                for record in await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_type=EventType.SESSION_INTERRUPTED,
                    )
                )
                if record.event.payload.get("interruption_type") == "operator_requested"
            ]
            assert len(terminal_records) == 1
        finally:
            for task in (interrupting, running):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (interrupting, running) if task is not None),
                return_exceptions=True,
            )
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reclaims_unaccepted_live_supersession_handoff(
    session_store_case,
    monkeypatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"unaccepted-live-supersession-handoff-{session_store_case[0]}"
        running: asyncio.Task[list[Event]] | None = None
        interrupting: asyncio.Task[list[Event]] | None = None
        release_fan_out = asyncio.Event()
        try:
            monkeypatch.setattr(
                session_control_module,
                "ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS",
                2,
            )
            monkeypatch.setattr(
                session_control_module,
                "ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S",
                0.001,
            )
            awaiting_event_id: str | None = None
            original_publish = store.publish_runtime_publication
            original_side_effect_claim = store.claim_persisted_event_side_effect
            fan_out_started = asyncio.Event()

            async def remember_awaiting_event(session_id: str, **kwargs):
                nonlocal awaiting_event_id
                result = await original_publish(session_id, **kwargs)
                if kwargs["request"].kind == "user-input-open":
                    awaiting_event_id = next(
                        event.id
                        for event in kwargs["request"].events
                        if event.type is EventType.SESSION_AWAITING_USER_INPUT
                    )
                return result

            async def block_awaiting_fan_out(*, session_id: str, event_id: str):
                if event_id == awaiting_event_id:
                    fan_out_started.set()
                    await release_fan_out.wait()
                return await original_side_effect_claim(
                    session_id=session_id,
                    event_id=event_id,
                )

            store.publish_runtime_publication = remember_awaiting_event  # type: ignore[method-assign]
            store.claim_persisted_event_side_effect = block_awaiting_fan_out  # type: ignore[method-assign]
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            original_handler = app._session_finalization.handle_session_interrupted
            rejected_before_take = asyncio.Event()

            async def reject_before_handoff_acceptance(*args, **kwargs):
                rejected_before_take.set()
                if False:  # pragma: no cover - preserve async-generator shape
                    yield args, kwargs
                raise OSError("live handler failed before accepting its handoff")

            app._session_finalization.handle_session_interrupted = (  # type: ignore[method-assign]
                reject_before_handoff_acceptance
            )
            running = asyncio.create_task(
                _collect_events(
                    app.run(
                        RunRequest(
                            session_id=session_id,
                            agent_name="assistant",
                            messages=[Message.text("user", "ask a question")],
                        )
                    )
                )
            )
            await asyncio.wait_for(fan_out_started.wait(), timeout=10)
            request = InterruptSessionRequest(
                session_id=session_id,
                reason="operator supersedes active pause",
            )
            with suppress_interruption_cascade():
                interrupting = asyncio.create_task(_collect_events(app.interrupt_session(request)))
                await asyncio.wait_for(rejected_before_take.wait(), timeout=5)
                await asyncio.wait_for(
                    asyncio.gather(running, return_exceptions=True),
                    timeout=5,
                )
                await asyncio.wait_for(
                    asyncio.gather(interrupting, return_exceptions=True),
                    timeout=10,
                )
            assert (
                session_id
                not in app._session_engine._session_control._terminal_finalization_claim_handoffs
            )
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            assert "pending_session_interrupt" in checkpoint
            assert "incomplete_session_recovery_claim" in checkpoint

            def expire_abandoned_claim(_session, current_checkpoint):
                assert current_checkpoint is not None
                updated = copy.deepcopy(current_checkpoint)
                expired_at = datetime.now(UTC) - timedelta(minutes=5)
                marker = updated["incomplete_session_recovery_claim"]
                marker["claimed_at"] = (expired_at - timedelta(minutes=5)).isoformat()
                marker["claim_expires_at"] = expired_at.isoformat()
                return updated

            await store.transform_checkpoint(session_id, expire_abandoned_claim)

            app._session_finalization.handle_session_interrupted = (  # type: ignore[method-assign]
                original_handler
            )
            store.publish_runtime_publication = original_publish  # type: ignore[method-assign]
            store.claim_persisted_event_side_effect = (  # type: ignore[method-assign]
                original_side_effect_claim
            )
            release_fan_out.set()
            with suppress_interruption_cascade():
                retried = await _collect_events(app.interrupt_session(request))
            assert retried[-1].type is EventType.SESSION_INTERRUPTED
        finally:
            release_fan_out.set()
            for task in (running, interrupting):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (running, interrupting) if task is not None),
                return_exceptions=True,
            )
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reconstructs_answering_user_input_supersession(
    session_store_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"user-input-answering-supersession-{session_store_case[0]}"
        try:
            provider = _UserInputRecoveryProvider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "ask a question")],
                    )
                )
            ]
            awaiting = next(
                event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            public_input_id = awaiting.payload["input_id"]
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            private_input_id = checkpoint["pending_user_input"]["input_id"]

            claim_committed = asyncio.Event()
            release_claim_acknowledgement = asyncio.Event()
            response = UserInputResponse(
                session_id=session_id,
                input_id=public_input_id,
                answer="yes",
            )

            original_admission = app._recovery_coordinator._admit_user_input_resolution_execution

            async def block_before_execution_admission(*args: Any, **kwargs: Any):
                current = await store.load_checkpoint(session_id)
                intent = (
                    None
                    if current is None
                    else current.get(USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY)
                )
                if type(intent) is dict and intent.get("execution_state") == "claimed":
                    claim_committed.set()
                    await release_claim_acknowledgement.wait()
                return await original_admission(*args, **kwargs)

            monkeypatch.setattr(
                app._recovery_coordinator,
                "_admit_user_input_resolution_execution",
                block_before_execution_admission,
            )

            async def collect_after_answer_claim() -> list[Event]:
                return [event async for event in app.resolve_user_input(response)]

            resolving = asyncio.create_task(collect_after_answer_claim())
            await asyncio.wait_for(claim_committed.wait(), timeout=5)
            claimed_checkpoint = await store.load_checkpoint(session_id)
            assert claimed_checkpoint is not None
            resolution_intent = claimed_checkpoint["user_input_resolution_intent"]
            assert resolution_intent["execution_state"] == "claimed"
            private_response = response.model_copy(update={"input_id": private_input_id})
            assert resolution_intent["answer_request_digest"] == user_input_answer_request_digest(
                private_response
            )
            assert resolution_intent[
                "resolution_request_digest"
            ] == user_input_resolution_request_digest(private_response)

            interrupt_app = CayuApp(session_store=store, enable_logging=False)
            interrupt_app.register_provider(_UserInputRecoveryProvider(), default=True)
            interrupt_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            with pytest.raises(TimeoutError, match="still finalizing"):
                _ = [
                    event
                    async for event in interrupt_app.interrupt_session(
                        InterruptSessionRequest(
                            session_id=session_id,
                            reason="operator supersedes answer claim",
                        )
                    )
                ]
            release_claim_acknowledgement.set()
            resolved = await asyncio.wait_for(resolving, timeout=10)
            assert resolved[-1].type is EventType.SESSION_INTERRUPTED
            assert len(provider.requests) == 1
            assert not any(
                event.type is EventType.SESSION_RESUMED
                for event in await store.load_events(session_id)
            )

            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            assert "pending_user_input" not in checkpoint
            assert "user_input_resolution_intent" not in checkpoint
            operator_records = [
                record
                for record in await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_type=EventType.SESSION_INTERRUPTED,
                    )
                )
                if record.event.payload.get("interruption_type") == "operator_requested"
            ]
            assert operator_records
            marker = operator_records[-1].event.payload["user_input_supersession_intent"]
            assert marker["state"] == "answering"
            assert marker["input_id"] == private_input_id
            assert marker["claim_run_epoch"] == resolution_intent["claim_run_epoch"]
            assert (
                marker["resolution_request_digest"]
                == (resolution_intent["resolution_request_digest"])
            )

            store = await _reopen_store(session_store_case, store)
            replay_app = CayuApp(session_store=store, enable_logging=False)
            replay_app.register_provider(_UserInputRecoveryProvider(), default=True)
            replay_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="superseded by an external interruption",
            ):
                _ = [event async for event in replay_app.resolve_user_input(response)]
        finally:
            await _close_store(store)

    asyncio.run(run())
