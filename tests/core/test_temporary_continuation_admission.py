"""Native invocation/service atomicity; registered foreign receiver follows separately."""

import asyncio

import pytest
from tests.core._execution_profile_fixtures import runtime_interaction_started_event
from tests.core.test_temporary_continuation_store import (
    _latch,
    prepared_service,
    run_owned,
)
from tests.core.test_temporary_continuation_store import store_factory as store_factory

from cayu.applications import CayuApp
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._invocation_lifecycle import (
    AdmitInvocationCommand,
    invocation_admission_command_sha256,
    invocation_checkpoint_state_sha256,
)
from cayu.runtime._session_continuation import ContinuationConflict, continuation_digest
from cayu.runtime._session_continuation_scope import authenticated_latch_scope
from cayu.runtime._temporary_continuation import (
    TemporaryServiceRecord,
    temporary_admission_payload_sha256,
    temporary_service_key,
)
from cayu.runtime._temporary_continuation_scope import temporary_admission_scope
from cayu.runtime._temporary_service_budget import admitted_service_budget_constraint
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.sessions.records import SessionStatus
from cayu.tools.exposure import tool_capability_ceiling_from_session_metadata


async def native_command(store, *, template=None, side_session=False, require_peer_append=False):
    reserved = await prepared_service(store, template=template)
    session = await store.load(reserved.intent.ticket.session_id)
    if side_session:
        from uuid import uuid4

        from tests.core._execution_profile_fixtures import (
            create_admitted_session,
            interrupt_and_release_test_invocation,
        )

        from cayu.messages import Message
        from cayu.sessions.requests import RunRequest

        created = await create_admitted_session(
            store,
            request=RunRequest(
                agent_name="side",
                session_id="side-" + uuid4().hex,
                messages=[Message.text("user", "Existing side session.")],
            ),
            provider_name="provider",
            model="model",
        )
        await interrupt_and_release_test_invocation(store, created.session.id)
        session = await store.load(created.session.id)
        selected = reserved.intent.model_copy(
            update={
                "mode": "side_session",
                "target": reserved.intent.target.model_copy(
                    update={"object_id": session.id, "incarnation": session.instance_id}
                ),
            }
        )
        reserved = reserved.model_copy(
            update={
                "admission": reserved.admission.model_copy(
                    update={
                        "dispatch": reserved.admission.dispatch.model_copy(
                            update={"intent": selected, "expected_run_epoch": session.run_epoch}
                        )
                    }
                )
            }
        )
    checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint(session.id)
    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    assert active is not None
    app = CayuApp(session_store=store, enable_logging=False)
    event = runtime_interaction_started_event(
        app,
        session_id=session.id,
        interaction_id=reserved.intent.invocation_id,
        agent_name=session.agent_name,
    )
    command = AdmitInvocationCommand(
        session_id=session.id,
        expected_session_instance_id=session.instance_id,
        expected_statuses=(SessionStatus.INTERRUPTED,),
        expected_run_epoch=session.run_epoch,
        expected_checkpoint_sha256=invocation_checkpoint_state_sha256(checkpoint),
        target_active_profile=active.model_copy(
            update={
                "interaction_id": event.interaction_id,
                "run_epoch": session.run_epoch + 1,
            }
        ),
        interaction_started_event=event,
        tool_capability_ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
        expected_active_profile=active,
        temporary_service_operation_key=temporary_service_key(reserved.intent.operation),
    )
    intent = reserved.intent.model_copy(
        update={"execution_profile_sha256": active.profile.fingerprint}
    )
    if require_peer_append:
        from tests.core.test_clarification_deliveries import delivery_record

        peer = delivery_record().intent.append
        key = peer.append_key.model_copy(
            update={
                "target_session_id": session.id,
                "target_session_instance_id": session.instance_id,
            }
        )
        peer = peer.model_copy(
            update={
                "append_key": key,
                "attempt_key": peer.attempt_key.model_copy(
                    update={"append_key": key, "target_run_epoch": session.run_epoch + 1}
                ),
            }
        )
        intent = intent.model_copy(update={"required_peer_append": peer})
    dispatch = reserved.admission.dispatch.model_copy(
        update={
            "intent": intent,
            "admission_payload_sha256": temporary_admission_payload_sha256(command),
        }
    )
    permit = reserved.admission.permit
    permit = permit.model_copy(
        update={
            "intent": permit.intent.model_copy(
                update={
                    "request": permit.intent.request.model_copy(
                        update={
                            "admission_commitment": continuation_digest(dispatch),
                            "target": intent.target,
                        }
                    ),
                }
            )
        }
    )
    command = command.model_copy(
        update={
            "participant_permit_operation": permit.operation.caller_key,
            "participant_permit_commitment": reserved.admission.permit_receipt_sha256,
        }
    )
    admission = reserved.admission.model_copy(
        update={
            "dispatch": dispatch,
            "permit": permit,
            "admission_command_sha256": invocation_admission_command_sha256(command),
        }
    )
    return admission, command


@pytest.mark.parametrize("require_peer_append", [False, True])
def test_native_service_and_invocation_commit_together(store_factory, require_peer_append):
    async def run(store):
        admission, command = await native_command(store, require_peer_append=require_peer_append)
        # A caller-shaped command is not an execution entrance.
        before = await store.load(command.session_id)
        with pytest.raises(PermissionError):
            await store.apply_invocation_lifecycle_command(command)
        assert await store.load(command.session_id) == before
        with temporary_admission_scope(admission, command):
            result = await store.apply_invocation_lifecycle_command(command)
        assert result.session.status is SessionStatus.RUNNING
        assert result.session.run_epoch == command.expected_run_epoch + 1
        retained = await store._load_temporary_continuation_service(admission)
        assert retained.state == "admitted"
        assert retained.execution.admission_command_sha256 == invocation_admission_command_sha256(
            command
        )
        assert retained.execution.run_epoch == result.session.run_epoch

        async def check_constraint(candidate):
            if require_peer_append:
                with pytest.raises(ContinuationConflict, match="exact durable peer append"):
                    await admitted_service_budget_constraint(
                        candidate, session_id=command.session_id
                    )
                assert (
                    await candidate.read_peer_content_attempt(
                        admission.dispatch.intent.required_peer_append
                    )
                    is None
                )
            else:
                assert (
                    await admitted_service_budget_constraint(
                        candidate, session_id=command.session_id
                    )
                    == admission.dispatch.intent
                )

        await check_constraint(store)
        reconstructed = store_factory()
        try:
            await check_constraint(reconstructed)
        finally:
            if reconstructed is not store:
                await reconstructed.close()
        # No second invocation or service generation on acknowledgement replay.
        with temporary_admission_scope(admission, command):
            replay = await store.apply_invocation_lifecycle_command(command)
        assert replay.session == result.session
        assert await store._load_temporary_continuation_service(admission) == retained
        with pytest.raises(PermissionError):
            await store.apply_invocation_lifecycle_command(command)

    asyncio.run(run_owned(store_factory, run))


def test_admission_reconciliation_replays_competing_native_publication(store_factory):
    async def run(store):
        admission, command = await native_command(store, side_session=True)
        await store._prepare_temporary_side_service(admission.preparation)
        prepared = await store._load_temporary_continuation_service(admission.preparation)
        reserved = TemporaryServiceRecord(admission=admission, state="reserved")
        await store._publish_temporary_continuation_service(previous=prepared, proposed=reserved)
        other = store_factory()
        try:
            # The independent recovery owner reads before native admission wins.
            previous = await other._load_temporary_continuation_service(admission)
            assert previous == reserved
            with temporary_admission_scope(admission, command):
                await store.apply_invocation_lifecycle_command(command)
            admitted = await other._read_temporary_continuation_outcome(admission)
            assert admitted is not None and admitted.state == "admitted"
            await store._publish_temporary_continuation_service(
                previous=previous, proposed=admitted
            )
            ticket = admission.dispatch.intent.ticket
            parent = await store.load_continuation_ticket(
                ticket.session_id,
                session_instance_id=ticket.session_instance_id,
                registration_key=ticket.registration_key,
            )
            session = await store.load(ticket.session_id)
            events = await store.load_events(ticket.session_id)
            assert await asyncio.gather(
                store._publish_temporary_continuation_service(previous=previous, proposed=admitted),
                other._publish_temporary_continuation_service(previous=previous, proposed=admitted),
            ) == [admitted, admitted]
            assert admitted.execution is not None
            changed = admitted.model_copy(
                update={"execution": admitted.execution.model_copy(update={"receipt_id": "wrong"})}
            )
            with pytest.raises(ContinuationConflict):
                await other._publish_temporary_continuation_service(
                    previous=previous, proposed=changed
                )
            assert await other._load_temporary_continuation_service(admission) == admitted
            assert await store.load_events(ticket.session_id) == events
            after = await store.load(ticket.session_id)
            assert after.model_dump(
                exclude={"last_activity_at", "updated_at"}
            ) == session.model_dump(exclude={"last_activity_at", "updated_at"})
            assert (
                await store.load_continuation_ticket(
                    ticket.session_id,
                    session_instance_id=ticket.session_instance_id,
                    registration_key=ticket.registration_key,
                )
                == parent
            )
        finally:
            if other is not store:
                await other.close()

    asyncio.run(run_owned(store_factory, run))


def test_prepared_exclusion_and_native_admission_have_one_winner(store_factory):
    from cayu.collaboration._permits import ReceivingSettlementReceipt
    from cayu.sessions.base import InMemorySessionStore

    async def run(store):
        admission, command = await native_command(store)
        prepared = TemporaryServiceRecord(admission=admission.preparation, state="prepared")
        await store._publish_temporary_continuation_service(previous=None, proposed=prepared)
        excluded = TemporaryServiceRecord(
            admission=admission.preparation,
            state="excluded",
            settlement=ReceivingSettlementReceipt(
                expected=admission.permit,
                receiving_owner=admission.dispatch.intent.target.owner,
                receipt_id="native-exclusion-race",
                outcome="quiescent",
                admission_excluded=True,
            ),
        )
        other = store_factory()
        try:

            async def admit():
                with temporary_admission_scope(admission, command):
                    return await store.apply_invocation_lifecycle_command(command)

            admitted, fenced = await asyncio.gather(
                admit(),
                other._publish_temporary_continuation_service(previous=prepared, proposed=excluded),
                return_exceptions=True,
            )
            assert isinstance(admitted, BaseException) != isinstance(fenced, BaseException)
            retained = await store._load_temporary_continuation_service(admission.preparation)
            session = await store.load(command.session_id)
            if isinstance(admitted, BaseException):
                assert isinstance(admitted, ContinuationConflict)
                assert retained == excluded
                assert session.run_epoch == command.expected_run_epoch
                with (
                    temporary_admission_scope(admission, command),
                    pytest.raises(ContinuationConflict),
                ):
                    await store.apply_invocation_lifecycle_command(command)
            else:
                assert isinstance(fenced, ContinuationConflict)
                assert retained.state == "admitted"
                assert session.run_epoch == command.expected_run_epoch + 1
        finally:
            if not isinstance(other, InMemorySessionStore):
                await other.close()

    asyncio.run(run_owned(store_factory, run))


def test_latch_winner_prevents_native_service_admission_atomically(store_factory):
    async def run(store):
        admission, command = await native_command(store)
        ticket = admission.dispatch.intent.ticket
        latch = _latch(ticket)
        with authenticated_latch_scope(latch):
            await store.latch_continuation(latch)
        before = await store.load(ticket.session_id)
        # Refresh only the ordinary checkpoint CAS to exercise the source-owner
        # latch check, rather than stopping at an unrelated stale-checkpoint gate.
        checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint(
            ticket.session_id
        )
        command = command.model_copy(
            update={"expected_checkpoint_sha256": invocation_checkpoint_state_sha256(checkpoint)}
        )
        dispatch = admission.dispatch.model_copy(
            update={"admission_payload_sha256": temporary_admission_payload_sha256(command)}
        )
        permit = admission.permit
        permit = permit.model_copy(
            update={
                "intent": permit.intent.model_copy(
                    update={
                        "request": permit.intent.request.model_copy(
                            update={"admission_commitment": continuation_digest(dispatch)}
                        )
                    }
                )
            }
        )
        admission = admission.model_copy(
            update={
                "dispatch": dispatch,
                "permit": permit,
                "admission_command_sha256": invocation_admission_command_sha256(command),
            }
        )
        with temporary_admission_scope(admission, command), pytest.raises(ContinuationConflict):
            await store.apply_invocation_lifecycle_command(command)
        assert await store.load(ticket.session_id) == before
        assert await store._load_temporary_continuation_service(admission) is None
        retained = await store.load_continuation_ticket(
            ticket.session_id,
            session_instance_id=ticket.session_instance_id,
            registration_key=ticket.registration_key,
        )
        assert retained.ticket.state == "WAITING"
        assert retained.latch == latch

    asyncio.run(run_owned(store_factory, run))


def test_service_admission_failure_rolls_back_invocation_and_owner(store_factory, monkeypatch):
    from cayu.sessions import _temporary_continuation_store as owner

    async def run(store):
        admission, command = await native_command(store)
        session_before = await store.load(command.session_id)
        checkpoint_before = await runtime_checkpoint_session_store(store).load_checkpoint(
            command.session_id
        )
        events_before = await store.load_events(command.session_id)
        original = owner.compose_same_session_admission

        def fail_after_preparation(**kwargs):
            original(**kwargs)
            raise OSError("injected failure after service and native receipt preparation")

        monkeypatch.setattr(owner, "compose_same_session_admission", fail_after_preparation)
        with (
            temporary_admission_scope(admission, command),
            pytest.raises(OSError, match="after service"),
        ):
            await store.apply_invocation_lifecycle_command(command)
        assert await store.load(command.session_id) == session_before
        assert (
            await runtime_checkpoint_session_store(store).load_checkpoint(command.session_id)
            == checkpoint_before
        )
        assert await store.load_events(command.session_id) == events_before
        assert await store._load_temporary_continuation_service(admission) is None

    asyncio.run(run_owned(store_factory, run))


@pytest.mark.parametrize("interruption", ("acknowledgement_loss", "cancellation"))
def test_admission_commit_is_recoverable_after_observer_interruption(
    store_factory, interruption, monkeypatch
):
    from cayu.sessions._session_continuation_store import pending_admission_receipt_identities

    async def run(store):
        admission, command = await native_command(store)
        original = type(store).admit_session_invocation
        committed = asyncio.Event()
        blocked = asyncio.Event()

        async def interrupt_after_commit(self, *args, **kwargs):
            result = await original(self, *args, **kwargs)
            committed.set()
            if interruption == "acknowledgement_loss":
                raise OSError("lost native admission acknowledgement")
            await blocked.wait()
            return result

        monkeypatch.setattr(type(store), "admit_session_invocation", interrupt_after_commit)
        assert store._supports_invocation_lifecycle_command_protocol()

        async def admit():
            with temporary_admission_scope(admission, command):
                return await store.apply_invocation_lifecycle_command(command)

        task = asyncio.create_task(admit())
        ready = asyncio.create_task(committed.wait())
        try:
            await asyncio.wait((task, ready), return_when=asyncio.FIRST_COMPLETED)
            if not committed.is_set():
                await task  # Surface an early failure instead of hanging on the barrier.
                pytest.fail("Admission exited without crossing its commit barrier.")
        finally:
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)
        if interruption == "cancellation":
            task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
            assert task.cancelling() == 1
        else:
            # The existing lifecycle owner reconciles its committed receipt.
            result = await task
            assert result.session.status is SessionStatus.RUNNING
        monkeypatch.setattr(type(store), "admit_session_invocation", original)
        retained = await store._load_temporary_continuation_service(admission)
        assert retained.state == "admitted"
        assert retained.settlement is None
        session = await store.load(command.session_id)
        checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint(
            command.session_id
        )
        pinned = pending_admission_receipt_identities(session, checkpoint)
        assert retained.execution.receipt_id in pinned
        assert f"release:{session.id}:{session.instance_id}:{session.run_epoch}" in pinned
        with temporary_admission_scope(admission, command):
            replay = await store.apply_invocation_lifecycle_command(command)
        assert replay.session.run_epoch == session.run_epoch
        assert await store._load_temporary_continuation_service(admission) == retained

    asyncio.run(run_owned(store_factory, run))


def test_native_release_returns_service_and_preserves_original_latch(store_factory):
    from tests.core._execution_profile_fixtures import interrupt_and_release_test_invocation

    from cayu.collaboration._contracts import ExactMatch, ExactUnavailable
    from cayu.runtime._temporary_continuation_permits import TemporaryServiceSettlementReader
    from cayu.sessions._session_continuation_store import pending_admission_receipt_identities
    from cayu.vaults.redaction import SecretRedactor

    async def run(store):
        admission, command = await native_command(store)
        ticket = admission.dispatch.intent.ticket
        reader = TemporaryServiceSettlementReader(store, admission, redactor=SecretRedactor())
        assert isinstance(await reader.lookup(admission.permit), ExactUnavailable)
        assert await store._read_temporary_continuation_outcome(admission) is None
        with temporary_admission_scope(admission, command):
            await store.apply_invocation_lifecycle_command(command)
        admitted = await store._load_temporary_continuation_service(admission)
        assert isinstance(await reader.lookup(admission.permit), ExactUnavailable)
        assert await store._reconcile_temporary_continuation_service(admission) == admitted
        latch = _latch(ticket)
        with authenticated_latch_scope(latch):
            await store.latch_continuation(latch)
        await interrupt_and_release_test_invocation(store, command.session_id)
        positive = await reader.lookup(admission.permit)
        assert isinstance(positive, ExactMatch)
        assert positive.receipt.outcome == "quiescent"
        # Writer release is positive receiving evidence, but source responsibility
        # remains until its separate return transaction commits.
        retained = await store._load_temporary_continuation_service(admission)
        assert retained.state == "admitted"
        returned = await store._reconcile_temporary_continuation_service(admission)
        assert returned.state == "returned"
        assert returned.released_session_status == "interrupted"
        assert returned.execution == admitted.execution
        assert returned.settlement.outcome == "quiescent"
        assert returned.settlement == positive.receipt
        assert await reader.lookup(admission.permit) == positive
        assert returned.returned_writer_generation == command.expected_run_epoch + 2
        original = await store.load_continuation_ticket(
            ticket.session_id,
            session_instance_id=ticket.session_instance_id,
            registration_key=ticket.registration_key,
        )
        assert original.ticket.state == "WAITING"
        assert original.ticket.writer_generation == ticket.writer_generation
        assert original.latch == latch
        session = await store.load(ticket.session_id)
        checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint(
            ticket.session_id
        )
        # The service is settled, but the original WAITING ticket still owns
        # its admission/release evidence for cancellation or expiry cleanup.
        assert pending_admission_receipt_identities(session, checkpoint) == frozenset(
            f"{kind}:{session.id}:{session.instance_id}:{ticket.writer_generation}"
            for kind in ("admit", "release")
        )
        assert await store._reconcile_temporary_continuation_service(admission) == returned

    asyncio.run(run_owned(store_factory, run))


def test_native_service_return_reconciles_after_store_reconstruction(store_factory):
    from tests.core._execution_profile_fixtures import interrupt_and_release_test_invocation

    from cayu.runtime._temporary_continuation import TemporaryServiceAdmission
    from cayu.sessions.base import InMemorySessionStore

    async def run():
        store = store_factory()
        try:
            admission, command = await native_command(store)
            with temporary_admission_scope(admission, command):
                await store.apply_invocation_lifecycle_command(command)
            admitted = await store._load_temporary_continuation_service(admission)
            await interrupt_and_release_test_invocation(store, command.session_id)
            # Lose the return observer after native release commits. Reconstruct
            # the complete expected tuple, not private in-process provenance.
            expected = TemporaryServiceAdmission.model_validate_json(admission.model_dump_json())
            if not isinstance(store, InMemorySessionStore):
                await store.close()
                store = store_factory()
            assert await store._load_temporary_continuation_service(expected) == admitted
            returned = await store._reconcile_temporary_continuation_service(expected)
            assert returned.state == "returned"
            assert returned.released_session_status == "interrupted"
            assert returned.execution == admitted.execution
            assert returned.settlement.outcome == "quiescent"
            events = await store.load_events(command.session_id)
            session = await store.load(command.session_id)
            if not isinstance(store, InMemorySessionStore):
                await store.close()
                store = store_factory()
            assert await store._reconcile_temporary_continuation_service(expected) == returned
            assert await store.load(command.session_id) == session
            assert await store.load_events(command.session_id) == events
        finally:
            if not isinstance(store, InMemorySessionStore):
                await store.close()

    asyncio.run(run())
