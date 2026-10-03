"""Public root execution parks at the real engine boundary with authenticated targets."""

import asyncio

import pytest
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_collaboration_waits import wait_for
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import stores as stores

from cayu.agents import AgentSpec
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._session_continuation import continuation_digest
from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionStatus
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("cleanup_scenario", ("revoked", "expired", "pruned", "pruned_after_ack"))
async def test_released_wait_cleanup_without_disclosure_or_foreign_history(
    stores, tmp_path, request, monkeypatch, cleanup_scenario
):
    await test_public_execution_wait_authenticates_and_binds_the_complete_wait(
        stores,
        tmp_path,
        request,
        monkeypatch,
        False,
        "cancelled",
        cleanup_scenario=cleanup_scenario,
    )


@pytest.mark.parametrize("cleanup_scenario", ("revoked", "expired"))
async def test_missing_wait_cleanup_without_disclosure(
    stores, tmp_path, request, monkeypatch, cleanup_scenario
):
    await test_public_execution_wait_authenticates_and_binds_the_complete_wait(
        stores,
        tmp_path,
        request,
        monkeypatch,
        False,
        "registration",
        cleanup_scenario=cleanup_scenario,
    )


@pytest.mark.parametrize("namespace_transition", ("seal", "rotate"))
async def test_public_missing_wait_cleanup_after_namespace_transition(
    stores, tmp_path, request, monkeypatch, namespace_transition
):
    await test_public_execution_wait_authenticates_and_binds_the_complete_wait(
        stores, tmp_path, request, monkeypatch, False, "registration", namespace_transition
    )


@pytest.mark.parametrize("expired", (False, True))
async def test_missing_registration_cleanup_fences_late_registration(stores, expired):
    """Foreign transaction qualification; public interruption is exercised below."""
    from datetime import UTC, datetime, timedelta

    from cayu.collaboration._contracts import CollaborationConflict
    from cayu.collaboration.waits import request_object_ref
    from cayu.runtime._session_continuation import (
        ContinuationNamespace,
        ContinuationTicket,
        continuation_namespace_id,
    )

    source = stores()
    try:
        application, resolver, values = await public_setup(source)
        initial = values[1]
        receipt = await application.accept_collaboration_request(
            values[4], context=resolver.context
        )
        wait = wait_for(receipt, initial).model_copy(
            update={
                "deadline": (datetime.now(UTC) + timedelta(days=-1 if expired else 1)).isoformat(),
            }
        )
        ticket = ContinuationTicket(
            namespace=ContinuationNamespace(
                session_id="destination",
                session_instance_id="instance",
                owner=initial.owner,
                namespace_id=continuation_namespace_id("destination", "instance", initial.owner),
            ),
            session_id="destination",
            session_instance_id="instance",
            owner=initial.owner,
            registration_key="wait-ticket",
            targets=(request_object_ref(receipt.expected.intent.selection.reference),),
            predicate_kind=wait.predicate,
            predicate_version=1,
            deadline=wait.deadline,
            failure_policy=wait.failure_policy,
            service_policy=wait.service_policy,
            wait_edge_revision=1,
            interaction_id="interaction",
            writer_generation=1,
            purpose="cleanup",
            state="ARMING",
            revision=1,
        )
        bound = wait.model_copy(update={"delivery_ticket": ticket})
        retained = await application._wait_coordinator.cancel_prepared_registration(
            bound,
            context=resolver.context,
        )
        assert retained.state == ("expired" if expired else "cancelled")
        assert retained.delivery == "pending"  # not yet native exclusion
        assert (
            await source.register_wait(
                initial,
                bound,
                redactor=application._secret_redactor,
            )
            == retained
        )
        assert (
            await application._wait_coordinator.cancel_prepared_registration(
                bound,
                context=resolver.context,
            )
            == retained
        )
        with pytest.raises(CollaborationConflict):
            await source.register_wait(
                initial,
                bound.model_copy(
                    update={
                        "wait_edge_revision": 2,
                        "delivery_ticket": ticket.model_copy(update={"wait_edge_revision": 2}),
                    }
                ),
                redactor=application._secret_redactor,
            )
    finally:
        await source.close()


@pytest.mark.parametrize(
    "with_tool_round,park_failure",
    (
        (False, False),
        (True, False),
        (False, True),
        (False, "cancelled"),
        (False, "registration"),
        (False, "registration_cancelled"),
    ),
)
async def test_public_execution_wait_authenticates_and_binds_the_complete_wait(
    stores,
    tmp_path,
    request,
    monkeypatch,
    with_tool_round,
    park_failure,
    namespace_transition=None,
    cleanup_scenario=None,
):
    source = stores()
    if isinstance(source, InMemoryCollaborationStore):
        store = InMemorySessionStore()
    elif isinstance(source, SQLiteCollaborationStore):
        store = SQLiteSessionStore(tmp_path / "execution-wait.sqlite")
    else:
        store = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    try:
        application, resolver, values = await public_setup(source, session_store=store)
        initialized = values[1]
        receipt = await application.accept_collaboration_request(
            values[4], context=resolver.context
        )
        wait = wait_for(receipt, initialized).model_copy(
            update={"service_policy": "clarification", "failure_policy": "return_and_report"}
        )
        replies = [
            [
                ModelStreamEvent.text_delta("Ready to wait."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
        tool_calls = []

        class ObserveTurn(Tool):
            spec = ToolSpec(
                name="observe_turn",
                description="Observe the active turn.",
                input_schema={"type": "object", "properties": {}},
            )

            async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
                active_session = await store.load(ctx.session_id)
                assert active_session is not None
                preparing = await store.load_continuation_ticket(
                    ctx.session_id,
                    session_instance_id=active_session.instance_id,
                    registration_key="execution-wait:" + continuation_digest(wait.operation),
                )
                assert preparing is not None
                assert preparing.ticket.state == "ARMING"
                assert active_session.run_epoch == preparing.ticket.writer_generation
                tool_calls.append(ctx.session_id)
                return ToolResult(content="The tool round completed.")

        if with_tool_round:
            replies.insert(
                0,
                [
                    ModelStreamEvent.tool_call(id="observe", name="observe_turn", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
            )
        provider = ScriptedModelProvider(replies)
        application.register_provider(provider, default=True)
        application.register_agent(
            AgentSpec(name="root", model="model"), tools=[ObserveTurn()] if with_tool_round else []
        )
        creation = ParticipantSessionCreationRequest(
            creation_key="execution-wait:" + initialized.binding.application_scope,
            request=RunRequest(
                agent_name="root", messages=[Message.text("user", "Review then wait.")]
            ),
        )
        participant = values[2].reference
        session, _ = await application.create_participant_session(
            creation, participant=participant, context=CONTEXT
        )
        execution = ParticipantSessionExecutionRequest(
            request=creation.request.model_copy(update={"session_id": session.id}),
            session_instance_id=session.instance_id,
            execution_key="execute-and-wait",
        )
        before = await store.load(session.id)
        before_events = await store.load_events(session.id)
        with pytest.raises(PermissionError):
            [
                event
                async for event in application.execute_participant_session_to_wait(
                    execution,
                    wait,
                    participant=participant,
                    context=CONTEXT,
                    wait_context=resolver.context.model_copy(update={"principal": "unrelated"}),
                )
            ]
        assert provider.requests == []
        assert await store.load(session.id) == before
        assert await store.load_events(session.id) == before_events
        if park_failure:
            from cayu.runtime._execution_to_wait import _ExecutionToWait

            park_entered = asyncio.Event()

            async def fail_park(self, invocation):
                if park_failure == "cancelled":
                    park_entered.set()
                    await asyncio.Future()
                raise OSError("Wait publication unavailable before commit")

            monkeypatch.setattr(_ExecutionToWait, "park", fail_park)
            if park_failure in {"registration", "registration_cancelled"}:

                async def fail_registration(*args, **kwargs):
                    if park_failure == "registration_cancelled":
                        park_entered.set()
                        await asyncio.Future()
                    raise OSError("Wait registration unavailable before commit")

                monkeypatch.setattr(application._wait_coordinator, "register", fail_registration)
        events = []

        async def consume_execution():
            async for event in application.execute_participant_session_to_wait(
                execution,
                wait,
                participant=participant,
                context=CONTEXT,
                wait_context=resolver.context,
            ):
                events.append(event)

        if park_failure in {"cancelled", "registration_cancelled"}:
            task = asyncio.create_task(consume_execution())
            await asyncio.wait_for(park_entered.wait(), timeout=30)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
            assert task.cancelling() == 1
        else:
            await consume_execution()
        expected_dispatches = (
            0 if park_failure in {"registration", "registration_cancelled"} else 1 + with_tool_round
        )
        assert len(provider.requests) == expected_dispatches
        assert tool_calls == ([session.id] if with_tool_round else [])
        if park_failure:
            from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
            from cayu.runtime._execution_to_wait import prepare_execution_wait
            from cayu.runtime._invocation_lifecycle import (
                _invocation_lifecycle_receipt_from_checkpoint,
            )
            from cayu.runtime._session_continuation import (
                ContinuationConflict,
                ContinuationReleasedRetirement,
                ContinuationRetirement,
            )

            if park_failure is True:
                assert any(event.type == EventType.SESSION_FAILED for event in events)
            armed = await store.load_continuation_ticket(
                session.id,
                session_instance_id=session.instance_id,
                registration_key="execution-wait:" + continuation_digest(wait.operation),
            )
            assert armed is not None
            assert armed.ticket.state == "ARMING"
            observed = await store.load(session.id)
            assert observed is not None
            assert observed.run_epoch == armed.ticket.writer_generation + 1
            checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint(session.id)
            admission = _invocation_lifecycle_receipt_from_checkpoint(
                checkpoint,
                command_identity=f"admit:{session.id}:{session.instance_id}:{armed.ticket.writer_generation}",
            )
            assert admission is not None
            assert admission.participant_permit_operation is not None
            assert admission.participant_permit_commitment is not None
            handoff = await prepare_execution_wait(
                application, execution, wait, context=resolver.context
            )
            with pytest.raises(ContinuationConflict):
                await application.exclude_participant_session_wait(
                    ParticipantSessionExecutionRequest(
                        request=execution.request,
                        session_instance_id=execution.session_instance_id,
                        execution_key="different-execution-under-the-same-wait-key",
                    ),
                    wait,
                    participant=participant,
                    context=CONTEXT,
                    wait_context=resolver.context,
                )
            bound = wait.model_copy(update={"delivery_ticket": armed.preparation.intent})
            wrong = ContinuationReleasedRetirement(
                retirement=ContinuationRetirement(
                    ticket=armed.ticket,
                    control_id="invalid-proof-probe",
                    reason="cancelled",
                    retired_at="2026-09-23T00:00:00+00:00",
                ),
                permit_operation=admission.participant_permit_operation,
                permit_commitment="f" * 64,
            )
            with pytest.raises(ContinuationConflict):
                await handoff.owner.retire_released(wrong)
            with pytest.raises(ContinuationConflict):
                await application.exclude_participant_session_wait(
                    execution,
                    wait.model_copy(update={"wait_edge_revision": 2}),
                    participant=participant,
                    context=CONTEXT,
                    wait_context=resolver.context,
                )
            foreign = await application.inspect_collaboration_wait(bound, context=resolver.context)
            if park_failure in {"registration", "registration_cancelled"}:
                assert foreign is None
            else:
                assert foreign.state == "pending"
            if namespace_transition is not None:
                from cayu.collaboration.lifecycle import NamespaceRotate, NamespaceSeal

                current = await application.inspect_collaboration_namespace(context=CONTEXT)
                transition = NamespaceSeal if namespace_transition == "seal" else NamespaceRotate
                method = (
                    application.seal_collaboration_namespace
                    if namespace_transition == "seal"
                    else application.rotate_collaboration_namespace
                )
                await method(
                    transition(
                        operation=current.current.reference.operation("seal-before-cleanup"),
                        namespace=current.current.reference,
                        expected_revision=current.current.revision,
                    ),
                    context=CONTEXT,
                )
            if cleanup_scenario in {"revoked", "expired"}:
                if cleanup_scenario == "revoked":
                    resolver.denied = True
                else:
                    resolver.resolution = resolver.resolution.model_copy(
                        update={
                            "principal": resolver.resolution.principal.model_copy(
                                update={"expires_at_ms": 1}
                            )
                        }
                    )
                with pytest.raises(PermissionError):
                    await application.inspect_collaboration_wait(bound, context=resolver.context)
                before_cleanup = await store.load_continuation_ticket(
                    session.id,
                    session_instance_id=session.instance_id,
                    registration_key=armed.ticket.registration_key,
                )
                with pytest.raises(PermissionError):
                    await application.exclude_participant_session_wait(
                        execution,
                        wait,
                        participant=participant,
                        context=CONTEXT.model_copy(update={"principal": "unrelated"}),
                        wait_context=resolver.context,
                    )
                assert (
                    await store.load_continuation_ticket(
                        session.id,
                        session_instance_id=session.instance_id,
                        registration_key=armed.ticket.registration_key,
                    )
                    == before_cleanup
                )
            original_delivery = source.record_wait_delivery

            async def unavailable_delivery(*args, **kwargs):
                pending = await store.load_continuation_ticket(
                    session.id,
                    session_instance_id=session.instance_id,
                    registration_key=armed.ticket.registration_key,
                )
                assert pending is not None
                assert pending.ticket.state == "RETIRED"
                assert not pending.retirement_acknowledged
                with pytest.raises(PermissionError, match="has not acknowledged"):
                    await application.collaboration_wait_latch_receiver().authenticate_continuation_retirement(
                        bound, pending
                    )
                with pytest.raises(ContinuationConflict, match="acknowledgement is pending"):
                    await store.delete_session(session.id)
                if park_failure == "cancelled":
                    await original_delivery(*args, **kwargs)
                raise OSError("Foreign exclusion acknowledgement lost")

            with monkeypatch.context() as overrides:
                overrides.setattr(source, "record_wait_delivery", unavailable_delivery)
                with pytest.raises(OSError, match="acknowledgement lost"):
                    await application.exclude_participant_session_wait(
                        execution,
                        wait,
                        participant=participant,
                        context=CONTEXT,
                        wait_context=resolver.context,
                    )
            if cleanup_scenario == "pruned":
                from tests.core._execution_wait_cleanup_flow import prune_and_reopen

                application, source, store = await prune_and_reopen(
                    application,
                    source,
                    store,
                    stores,
                    tmp_path,
                    request,
                    resolver,
                    initialized,
                    receipt,
                    bound,
                )
                from cayu.runtime._execution_to_wait import _build_execution_wait

                handoff = await _build_execution_wait(
                    application, execution, wait, context=resolver.context
                )
            settled = await application.exclude_participant_session_wait(
                execution,
                wait,
                participant=participant,
                context=CONTEXT,
                wait_context=resolver.context,
            )
            retired = await store.load_continuation_ticket(
                session.id,
                session_instance_id=session.instance_id,
                registration_key=armed.ticket.registration_key,
            )
            assert retired is not None
            assert retired.retirement is not None
            assert retired.released_retirement is not None
            assert retired.retirement_acknowledged
            command = ContinuationReleasedRetirement(
                retirement=retired.retirement,
                permit_operation=admission.participant_permit_operation,
                permit_commitment=admission.participant_permit_commitment,
            )
            assert retired.ticket.state == "RETIRED"
            assert retired.released_retirement.permit_commitment == command.permit_commitment
            with monkeypatch.context() as overrides:

                def unavailable_original_ledger(*args, **kwargs):
                    raise AssertionError("Exact retirement replay must use its retained proof")

                overrides.setattr(
                    "cayu.sessions._session_continuation_store.require_released_wait_invocation",
                    unavailable_original_ledger,
                )
                assert await handoff.owner.retire_released(command) == retired
            with pytest.raises(ContinuationConflict):
                await handoff.owner.retire_released(wrong)
            assert settled.delivery == "excluded"
            if cleanup_scenario == "pruned_after_ack":
                from tests.core._execution_wait_cleanup_flow import prune_and_reopen

                application, source, store = await prune_and_reopen(
                    application,
                    source,
                    store,
                    stores,
                    tmp_path,
                    request,
                    resolver,
                    initialized,
                    receipt,
                    bound,
                )
            # A delayed exact registration cannot reopen the cancelled wait.
            if cleanup_scenario not in {"pruned", "pruned_after_ack"}:
                assert (
                    await source.register_wait(
                        initialized, bound, redactor=application._secret_redactor
                    )
                ).delivery == "excluded"
            assert (
                await application.exclude_participant_session_wait(
                    execution,
                    wait,
                    participant=participant,
                    context=CONTEXT,
                    wait_context=resolver.context,
                )
                == settled
            )
            with pytest.raises(ContinuationConflict):
                await application.exclude_participant_session_wait(
                    execution,
                    wait.model_copy(update={"wait_edge_revision": 2}),
                    participant=participant,
                    context=CONTEXT,
                    wait_context=resolver.context,
                )
            with pytest.raises((ContinuationConflict, PermissionError)):
                await application._wait_coordinator.exclude_released(
                    bound,
                    context=resolver.context,
                    continuation_owner=handoff.owner,
                    permit_operation=command.permit_operation,
                    permit_commitment="f" * 64,
                )
            after_cleanup = await store.load(session.id)
            assert after_cleanup is not None
            assert after_cleanup.run_epoch == observed.run_epoch
            assert len(provider.requests) == expected_dispatches
            await store.delete_session(session.id)
            assert await store.load(session.id) is None
            return
        assert any(event.type == EventType.SESSION_INTERRUPTED for event in events)
        observed = await store.load(session.id)
        assert observed is not None
        assert observed.status is SessionStatus.INTERRUPTED
        parked = await store.load_continuation_ticket(
            session.id,
            session_instance_id=session.instance_id,
            registration_key="execution-wait:" + continuation_digest(wait.operation),
        )
        assert parked is not None
        assert parked.ticket.state == "WAITING"
        assert observed.run_epoch == parked.ticket.writer_generation + 1
        bound = wait.model_copy(update={"delivery_ticket": parked.preparation.intent})
        snapshot = await application.inspect_collaboration_wait(bound, context=resolver.context)
        assert snapshot is not None
        assert snapshot.registration.wait == bound
        before_events = await store.load_events(session.id)
        [
            event
            async for event in application.execute_participant_session_to_wait(
                execution,
                wait,
                participant=participant,
                context=CONTEXT,
                wait_context=resolver.context,
            )
        ]
        assert len(provider.requests) == 1 + with_tool_round
        assert await store.load(session.id) == observed
        assert await store.load_events(session.id) == before_events
        with pytest.raises(ValueError, match="retained permit"):
            [
                event
                async for event in application.execute_participant_session_to_wait(
                    execution,
                    wait.model_copy(update={"wait_edge_revision": 2}),
                    participant=participant,
                    context=CONTEXT,
                    wait_context=resolver.context,
                )
            ]
        assert len(provider.requests) == 1 + with_tool_round
        assert await store.load(session.id) == observed
        assert await store.load_events(session.id) == before_events
    finally:
        if not isinstance(store, InMemorySessionStore):
            await store.close()
