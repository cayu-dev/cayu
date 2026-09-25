"""Planning uses the existing real source/budget-authorized question owner."""

import asyncio

import pytest
from tests.core.test_clarification_public import (
    test_public_question_uses_real_assistant_export as run_public_question,
)
from tests.core.test_request_planning_contracts import _policy
from tests.core.test_request_planning_public import complete_plan

from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import CollaborationConflict, ExactMatch, ExactNotFound
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.planning import (
    RequestPlanningClarify,
    RequestPlanningControl,
    RequestPlanningDecline,
    RequestPlanningDefer,
    RequestPlanningPredecessor,
    RequestPlanningRequest,
    RequestPlanningTimer,
    planning_policy_commitment,
)
from cayu.vaults.redaction import SecretRedactor


def configured_question(initialized, command, source):
    redactor = SecretRedactor()
    policy = _policy()
    policy = policy.model_copy(
        update={
            "reference": policy.reference.model_copy(update={"owner": initialized.owner}),
            "rules": (),
            "default": RequestPlanningClarify(opening=command, source=source),
        }
    )
    planned = RequestPlanningRequest(
        operation=initialized.operation("public-question-plan"),
        expected=command.expected,
        expected_revision=command.expected_revision - 1,
        expected_input_revision=command.question.input_revision,
        expected_input_sha256=clarification_commitment(command.expected, redactor),
        planning_generation=1,
        admission_operation=command.question.admission,
        admission_generation=1,
        initiator=command.question.initiator,
        policy=policy.reference,
        policy_sha256=planning_policy_commitment(policy, redactor=redactor),
        limits=policy.limits,
        deadline_at_ms=command.question.deadline_at_ms,
        predecessor=None,
    )
    return planned, policy


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_planning_successor_consumes_real_reply_without_completing_original_wait(
    backend, tmp_path, request, monkeypatch
):
    planners = []

    async def journey(
        *,
        application_for,
        collaboration,
        sessions,
        initialized,
        command,
        source,
        context,
        payloads,
        reopen_collaboration,
        backend,
    ):
        planned, policy = configured_question(initialized, command, source)
        planner = application_for(collaboration, sessions, planning_policies=(policy,))
        planners.append(planner)
        await planner.initialize_collaboration()
        record = await complete_plan(planner, planned, context.mandate)
        opened = await planner.lookup_clarification(command, context=context.mandate)
        assert isinstance(opened, ExactMatch)

        async def after_reply(*, current, reply, wait, parked, wait_context):
            before_calls = len(payloads)
            snapshot = await current.inspect_collaboration_request(
                command.expected, context=context.mandate
            )
            assert snapshot.state == "open"
            assert snapshot.clarification.input_revision == reply.input_revision == 1
            # Read the original native wait before and after the new generation;
            # accepting clarification must not consume final-result continuation.
            native_before = await sessions.load_continuation_ticket(
                parked.ticket.session_id,
                session_instance_id=parked.ticket.session_instance_id,
                registration_key=parked.ticket.registration_key,
            )
            assert native_before is not None and native_before.ticket.state == "WAITING"
            bound = wait.model_copy(update={"delivery_ticket": native_before.preparation.intent})
            wait_before = await current.inspect_collaboration_wait(bound, context=wait_context)
            next_policy = policy.model_copy(
                update={
                    "reference": policy.reference.model_copy(update={"revision": 2}),
                    "default": RequestPlanningDefer(
                        prerequisite=RequestPlanningTimer(
                            not_before_ms=command.expected.intent.selection.accepted_at_ms,
                            deadline_at_ms=planned.deadline_at_ms,
                        )
                    ),
                }
            )
            successor = planned.model_copy(
                update={
                    "operation": initialized.operation("plan-after-real-reply"),
                    "admission_operation": initialized.operation("admit-after-real-reply"),
                    "expected_revision": snapshot.revision,
                    "expected_input_revision": snapshot.clarification.input_revision,
                    "expected_input_sha256": snapshot.clarification.input_sha256,
                    "planning_generation": 2,
                    "admission_generation": snapshot.admission_generation + 1,
                    "policy": next_policy.reference,
                    "policy_sha256": planning_policy_commitment(
                        next_policy, redactor=SecretRedactor()
                    ),
                    "predecessor": RequestPlanningPredecessor(
                        operation=planned.operation, revision=record.revision
                    ),
                }
            )
            reopened = reopen_collaboration()
            second = application_for(reopened, sessions, planning_policies=(next_policy,))
            try:
                await second.initialize_collaboration()
                stale = successor.model_copy(
                    update={
                        "expected_input_revision": planned.expected_input_revision,
                        "expected_input_sha256": planned.expected_input_sha256,
                    }
                )
                with pytest.raises(CollaborationConflict):
                    await second.plan_collaboration_request(stale, context=context.mandate)
                assert (
                    await second.lookup_collaboration_plan(planned, context=context.mandate)
                ).receipt == record
                result = await complete_plan(second, successor, context.mandate)
                assert result.state == "deferred"
                assert result.receipt.command.expected_input_revision == 1
                assert (
                    result.receipt.command.expected_input_sha256
                    == snapshot.clarification.input_sha256
                )
                assert (
                    await second.reconcile_collaboration_plan(successor, context=context.mandate)
                    == result
                )
                assert (
                    await second.inspect_collaboration_wait(bound, context=wait_context)
                    == wait_before
                )
                assert (
                    await sessions.load_continuation_ticket(
                        parked.ticket.session_id,
                        session_instance_id=parked.ticket.session_instance_id,
                        registration_key=parked.ticket.registration_key,
                    )
                    == native_before
                )
                # A later explicit decline can settle the original request only
                # after the native question/service/delivery history is terminal.
                latest = await second.inspect_collaboration_request(
                    command.expected, context=context.mandate
                )
                decline_policy = next_policy.model_copy(
                    update={
                        "reference": policy.reference.model_copy(update={"revision": 3}),
                        "default": RequestPlanningDecline(reason="unsupported_request"),
                    }
                )
                final_command = successor.model_copy(
                    update={
                        "operation": initialized.operation("decline-after-real-reply"),
                        "admission_operation": initialized.operation("admit-decline-after-reply"),
                        "expected_revision": latest.revision,
                        "planning_generation": 3,
                        "admission_generation": latest.admission_generation + 1,
                        "policy": decline_policy.reference,
                        "policy_sha256": planning_policy_commitment(
                            decline_policy, redactor=SecretRedactor()
                        ),
                        "predecessor": RequestPlanningPredecessor(
                            operation=successor.operation, revision=result.revision
                        ),
                    }
                )
                final = application_for(reopened, sessions, planning_policies=(decline_policy,))
                planners.append(final)
                await final.initialize_collaboration()
                declined = await complete_plan(final, final_command, context.mandate)
                assert declined.state == "declined"
                assert (
                    await final.inspect_collaboration_request(
                        command.expected, context=context.mandate
                    )
                ).state == "declined"
                assert (
                    await final.reconcile_collaboration_plan(final_command, context=context.mandate)
                    == declined
                )
                assert len(payloads) == before_calls
            finally:
                await second.drain_collaboration_requests()
                await second.drain_session_exports()
                if backend != "memory":
                    await reopened.close()

        return opened.receipt, after_reply

    try:
        await run_public_question(
            backend,
            tmp_path,
            request,
            monkeypatch,
            True,
            True,
            False,
            False,
            planning_journey=journey,
        )
    finally:
        for planner in planners:
            await planner.drain_collaboration_requests()
            await planner.drain_session_exports()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("cancel_pending", [False, True])
async def test_public_planned_question_uses_export_owner_and_survives_reconstruction(
    backend, tmp_path, request, monkeypatch, cancel_pending
):
    async def driver(
        *,
        application_for,
        collaboration,
        sessions,
        initialized,
        command,
        source,
        context,
        payloads,
        reopen_collaboration,
        backend,
        export_policy,
    ):
        planned, policy = configured_question(initialized, command, source)
        application = application_for(collaboration, sessions, planning_policies=(policy,))
        reopened = None
        reader = None
        release = asyncio.Event()
        try:
            await application.initialize_collaboration()
            before_dispatches = len(payloads)
            if cancel_pending:
                entered = asyncio.Event()
                original_open = application._clarification_coordinator._open_planned

                async def blocked_open(*args, **kwargs):
                    entered.set()
                    await release.wait()
                    return await original_open(*args, **kwargs)

                monkeypatch.setattr(
                    application._clarification_coordinator, "_open_planned", blocked_open
                )
                # Isolate real caller cancellation from the independently
                # covered acknowledgement timeout; do not change owner authority.
                application._request_coordinator._owners.observation_timeout = 120
                caller = asyncio.create_task(
                    application.plan_collaboration_request(planned, context=context.mandate)
                )
                async with asyncio.timeout(180):
                    await entered.wait()
                    caller.cancel("private-cancellation-marker")
                    caller.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await caller
                    assert caller.cancelled() and caller.cancelling() == 2
                    pending_tasks = tuple(application._request_coordinator._owners.pending)
                    assert pending_tasks
                    reopened = reopen_collaboration()
                    reader = application_for(reopened, sessions)
                    await reader.initialize_collaboration()
                    pending = await reader.lookup_collaboration_plan(
                        planned, context=context.mandate
                    )
                    assert isinstance(pending, ExactMatch) and pending.receipt.pending_stages == 1
                    await export_policy.revoke()
                    control = RequestPlanningControl(
                        expected=planned,
                        expected_revision=pending.receipt.revision,
                        kind="cancelled",
                        initiator=planned.initiator,
                    )
                    cleaned = await complete_plan(reader, control, context.mandate, control=True)
                    assert cleaned.state == "cancelled" and cleaned.pending_stages == 0
                    release.set()
                    outcomes = await asyncio.gather(*pending_tasks, return_exceptions=True)
                    assert all(
                        isinstance(outcome, CollaborationAccessDenied) for outcome in outcomes
                    )
                    assert isinstance(
                        await reader.lookup_clarification(command, context=context.mandate),
                        ExactNotFound,
                    )
                    assert (
                        await reader.control_collaboration_plan(control, context=context.mandate)
                        == cleaned
                    )
                    assert (
                        await reader.plan_collaboration_request(planned, context=context.mandate)
                        == cleaned
                    )
                    assert len(payloads) == before_dispatches
                    return
            original_open = application._clarification_coordinator._open_planned
            opening_calls = 0

            async def lose_opening_ack(*args, **kwargs):
                nonlocal opening_calls
                opening_calls += 1
                await original_open(*args, **kwargs)
                raise ConnectionError("Question and stage committed; acknowledgement lost.")

            monkeypatch.setattr(
                application._clarification_coordinator, "_open_planned", lose_opening_ack
            )
            # The fixture resolver uses an ordinary non-reentrant lock.
            async with asyncio.timeout(180):
                with pytest.raises((ConnectionError, CollaborationUnavailable)):
                    await application.plan_collaboration_request(planned, context=context.mandate)
                pending = tuple(application._request_coordinator._owners.pending)
                if pending:
                    outcomes = await asyncio.gather(*pending, return_exceptions=True)
                    assert all(isinstance(outcome, ConnectionError) for outcome in outcomes)
                # Recovery must read the committed native stage, never call the
                # opening again (the fault remains installed to prove this).
                result = await complete_plan(application, planned, context.mandate, reconcile=True)
                assert opening_calls == 1
            assert result.state == "clarifying"
            assert result.stage_count == 1 and result.pending_stages == 0
            assert len(payloads) == before_dispatches
            opened = await application.lookup_clarification(command, context=context.mandate)
            assert isinstance(opened, ExactMatch) and opened.receipt.command == command
            # Matching caller values do not authenticate a planner-owned opening.
            with pytest.raises(CollaborationConflict):
                await application.open_clarification(command, source=source, context=context)
            reopened = reopen_collaboration()
            reader = application_for(reopened, sessions)
            await reader.initialize_collaboration()
            found = await reader.lookup_collaboration_plan(planned, context=context.mandate)
            assert isinstance(found, ExactMatch) and found.receipt == result
            assert (
                await reader.plan_collaboration_request(planned, context=context.mandate) == result
            )
            assert len(payloads) == before_dispatches
        finally:
            release.set()
            await application.drain_collaboration_requests()
            await application.drain_session_exports()
            if reader is not None:
                await reader.drain_collaboration_requests()
                await reader.drain_session_exports()
            if reopened is not None and backend != "memory":
                await reopened.close()

    await run_public_question(
        backend,
        tmp_path,
        request,
        monkeypatch,
        False,
        False,
        False,
        False,
        planning_driver=driver,
    )
