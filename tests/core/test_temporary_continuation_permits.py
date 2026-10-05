"""Real foreign permit readback; complete runtime/public journey follows separately."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.test_clarification_commands import question_for
from tests.core.test_collaboration_request_foundation import accept, setup
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import stores as stores
from tests.core.test_participant_lifecycle import change
from tests.core.test_temporary_continuation_contracts import selection

from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import CollaborationConflict, ObjectRef
from cayu.collaboration._permits import PermitCommand, PermitIntent, PermitRegistration
from cayu.collaboration.waits import request_object_ref
from cayu.runtime._temporary_continuation_permits import TemporaryServicePermitAuthority
from cayu.sessions._session_continuation import (
    ContinuationNamespace,
    continuation_digest,
    continuation_namespace_id,
)
from cayu.sessions._temporary_continuation import (
    TemporaryServiceAdmission,
    TemporaryServiceDispatch,
    temporary_service_invocation_id,
)
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio
REDACTOR = SecretRedactor()


@pytest.mark.parametrize("worker_offset", (-3600, 3600))
@pytest.mark.parametrize("event_only", [False, True])
async def test_owner_deadline_ignores_worker_clock_and_bounds_active_stream(
    stores, monkeypatch, worker_offset, event_only
):
    from datetime import UTC, datetime, timedelta

    import cayu.deadlines as deadlines
    from cayu.runtime._temporary_service_execution import drive_temporary_service

    store = stores()
    _, initial, admission = await prepared(store)
    authority = TemporaryServicePermitAuthority(store, initial, redactor=REDACTOR)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        now = await tx.now_ms()
    candidate = admission.dispatch.intent
    candidate = candidate.model_copy(
        update={
            "question": candidate.question.model_copy(update={"deadline_at_ms": now + 1000}),
            "ticket": candidate.ticket.model_copy(update={"deadline": None})
            if event_only
            else candidate.ticket,
        }
    )

    class WorkerClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(seconds=worker_offset)

    monkeypatch.setattr(deadlines, "datetime", WorkerClock)
    boundary = await authority.execution_deadline(candidate)
    remaining = boundary.remaining_seconds()
    assert remaining is not None and 0 < remaining <= 1
    entered, closed = [], []

    async def blocked():
        try:
            entered.append(True)
            await asyncio.Future()
        finally:
            closed.append(True)
        if False:
            yield

    with pytest.raises(TimeoutError):
        await drive_temporary_service(blocked(), boundary)
    assert entered == closed == [True]
    assert boundary.expired
    assert datetime.now(UTC).timestamp() * 1000 >= now + 900


async def test_owner_deadline_charges_clock_read_acknowledgement_latency(stores, monkeypatch):
    store = stores()
    _, initial, admission = await prepared(store)
    authority = TemporaryServicePermitAuthority(store, initial, redactor=REDACTOR)
    transaction = store._transaction
    sampled = []

    @asynccontextmanager
    async def delayed(*args, **kwargs):
        async with transaction(*args, **kwargs) as tx:
            sampled.append(await tx.now_ms())
            yield tx
        await asyncio.sleep(0.1)

    candidate = admission.dispatch.intent
    # A short policy bound isolates acknowledgement charging from setup time.
    candidate = candidate.model_copy(
        update={
            "question": candidate.question.model_copy(
                update={
                    "policy": candidate.question.policy.model_copy(
                        update={"service_timeout_ms": 250}
                    ),
                }
            ),
        }
    )
    monkeypatch.setattr(store, "_transaction", delayed)
    boundary = await authority.execution_deadline(candidate)
    remaining = boundary.remaining_seconds()
    assert sampled and remaining is not None and 0 <= remaining <= 0.15


@pytest.mark.parametrize("expiry", [1, True])
async def test_registration_guard_expiry_is_checked_before_native_mutation(stores, expiry):
    store = stores()
    _, initial, admission = await prepared(store, open_question=True)

    @asynccontextmanager
    async def guard(dispatch):
        assert dispatch == admission.dispatch
        yield expiry

    authority = TemporaryServicePermitAuthority(
        store, initial, redactor=REDACTOR, admission_guard=guard
    )
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        before = await store._anchor(tx, initial, REDACTOR)
    with pytest.raises(PermissionError if expiry is True else CollaborationConflict):
        await authority.register(admission.dispatch)
    assert await authority.lookup(admission.dispatch.intent) is None
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        assert await store._anchor(tx, initial, REDACTOR) == before


async def test_cancellation_does_not_release_inflight_registration_guard(stores, monkeypatch):
    store = stores()
    _, initial, admission = await prepared(store, open_question=True)
    lock = asyncio.Lock()
    entered = asyncio.Event()
    release = asyncio.Event()
    revoked = asyncio.Event()

    @asynccontextmanager
    async def guard(dispatch):
        async with lock:
            if revoked.is_set():
                raise PermissionError("disclosure revoked")
            yield dispatch.intent.question.deadline_at_ms

    authority = TemporaryServicePermitAuthority(
        store, initial, redactor=REDACTOR, admission_guard=guard
    )
    transaction = store._transaction

    @asynccontextmanager
    async def held_transaction(scope, *, write):
        async with transaction(scope, write=write) as tx:
            if write:
                entered.set()
                await release.wait()
            yield tx

    async def revoke():
        async with lock:
            revoked.set()

    with monkeypatch.context() as patch:
        patch.setattr(store, "_transaction", held_transaction)
        caller = asyncio.create_task(authority.register(admission.dispatch))
        revoker = None
        try:
            await asyncio.wait_for(entered.wait(), 10)
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert caller.cancelled() and caller.cancelling() == 1
            assert lock.locked() and store._owners.pending
            revoker = asyncio.create_task(revoke())
            await asyncio.sleep(0)
            assert not revoker.done()
            release.set()
            await asyncio.wait_for(revoker, 10)
        finally:
            release.set()
            await asyncio.gather(
                caller, *(() if revoker is None else (revoker,)), return_exceptions=True
            )
    receipt = await authority.lookup(admission.dispatch.intent)
    assert receipt is not None
    # A permit durably granted before revocation remains its exact historical
    # authority. Replay must not reacquire a fresh disclosure grant.
    assert await authority.register(admission.dispatch) == receipt


async def prepared(
    store, *, open_question=False, session_store=None, budget_binding=None, service_timeout_ms=None
):
    from tests.core.test_participant_identity import registration as participant_registration

    class BudgetReceiver:
        def __init__(self):
            self.binding = budget_binding

        async def resolve_budget_binding(self, *, request):
            return self.binding

    values = await setup(
        store,
        session_store=session_store,
        reg=None
        if budget_binding is None
        else participant_registration(scope=budget_binding.application_scope),
        application_options=(
            None
            if budget_binding is None
            else {
                "enable_common_root_budget_binding": True,
                "budget_binding_receiver": BudgetReceiver(),
            }
        ),
    )
    application, initial, sender, *_ = values
    if service_timeout_ms is not None:
        # Bound the original request consistently with the selected service
        # policy; preparation/real root execution also consumes this lifetime.
        values = (
            *values[:4],
            values[4].model_copy(update={"ttl_ms": max(60_000, service_timeout_ms + 60_000)}),
            values[5],
        )
    accepted = (await accept(store, values)).receipt
    question = question_for(accepted.expected)
    if service_timeout_ms is not None:
        question = question.model_copy(
            update={
                "deadline_at_ms": accepted.expected.intent.selection.accepted_at_ms
                + service_timeout_ms
                - 1_000,
                "policy": question.policy.model_copy(
                    update={"service_timeout_ms": service_timeout_ms}
                ),
            }
        )
    if budget_binding is not None:
        question = question.model_copy(
            update={
                "budget_binding": ObjectRef(
                    owner=initial.owner,
                    kind="budget_binding",
                    object_id=budget_binding.binding_id,
                    incarnation=budget_binding.authority_digest,
                    revision=1,
                ),
                "budget_authority_sha256": budget_binding.authority_digest,
            }
        )
    if open_question:
        from cayu.collaboration._clarification_commands import ClarificationOpenCommand
        from cayu.collaboration._clarification_store import open_in_transaction
        from cayu.collaboration._request_arbitration import admit_in_transaction
        from cayu.collaboration.requests import RequestAdmissionCommand

        question = question.model_copy(
            update={"policy": question.policy.model_copy(update={"max_service_turns": 1})}
        )
        async with store._transaction(initial.binding.application_scope, write=True) as tx:
            decision = await admit_in_transaction(
                store,
                tx,
                initial,
                RequestAdmissionCommand(
                    operation=initial.operation("admission"),
                    expected=accepted.expected,
                    expected_revision=1,
                    expected_input_revision=0,
                    expected_input_sha256=clarification_commitment(accepted.expected, REDACTOR),
                    generation=1,
                    decision="clarify",
                    evidence=(),
                    initiator=accepted.expected.initiator,
                ),
                redactor=REDACTOR,
            )
            await open_in_transaction(
                store,
                tx,
                initial,
                ClarificationOpenCommand(
                    operation=question.operation,
                    expected=accepted.expected,
                    expected_revision=decision.revision,
                    question=question,
                ),
                redactor=REDACTOR,
            )
    original = selection()
    namespace = ContinuationNamespace(
        session_id=original.ticket.session_id,
        session_instance_id=original.ticket.session_instance_id,
        owner=initial.owner,
        namespace_id=continuation_namespace_id(
            original.ticket.session_id, original.ticket.session_instance_id, initial.owner
        ),
    )
    intent = original.model_copy(
        update={
            "operation": initial.operation("service"),
            "invocation_id": temporary_service_invocation_id(initial.operation("service")),
            "initiator": question.initiator,
            "question": question,
            "ticket": original.ticket.model_copy(
                update={
                    "namespace": namespace,
                    "owner": initial.owner,
                    "targets": (request_object_ref(question.request),),
                }
            ),
            "target": ObjectRef(
                owner=initial.owner, kind="session", object_id="waiting", incarnation="one"
            ),
            "budget_binding": question.budget_binding,
            "budget_authority_sha256": question.budget_authority_sha256,
        }
    )
    dispatch = TemporaryServiceDispatch(
        intent=intent, admission_payload_sha256="a" * 64, expected_run_epoch=2
    )
    registration = PermitRegistration(
        operation=initial.operation("service-permit"),
        participant=sender.reference,
        expected_lifecycle_revision=sender.lifecycle_revision,
        expected_configuration_revision=sender.configuration_revision,
        admission_generation=sender.admission_generation,
        admission_commitment=continuation_digest(dispatch),
        source_operation=intent.operation,
        target=intent.target,
        target_state="existing",
        effect_scope="clarification_service",
        required_settlement="quiescence",
        settlement_operation=initial.operation("settle-service-permit"),
    )
    permit = PermitCommand(
        operation=registration.operation,
        source=initial.owner,
        destination=initial.owner,
        initiator=question.initiator,
        intent=PermitIntent(request=registration, limits=initial.binding.limits),
    )
    admission = TemporaryServiceAdmission(
        dispatch=dispatch,
        permit=permit,
        permit_receipt_sha256="b" * 64,
        admission_command_sha256="c" * 64,
    )
    return application, initial, admission


@pytest.mark.parametrize("registered_first", (True, False))
async def test_exact_foreign_permit_orders_against_disablement(stores, registered_first):
    from cayu.collaboration._clarification_service_store import register_service_in_transaction

    store = stores()
    application, initial, admission = await prepared(store, open_question=True)

    async def register():
        async with store._transaction(initial.binding.application_scope, write=True) as tx:
            record = await register_service_in_transaction(
                store, tx, initial, admission.dispatch, admission.permit, redactor=REDACTOR
            )
            return record.permit

    authority = TemporaryServicePermitAuthority(stores(), initial, redactor=REDACTOR)
    with pytest.raises(PermissionError):
        await authority.authenticate(admission)
    if registered_first:
        receipt = await register()
        admission = admission.model_copy(
            update={"permit_receipt_sha256": continuation_digest(receipt)}
        )
        assert await authority.authenticate(admission) == admission
    await application.change_participant_lifecycle(
        change(
            initial,
            admission.permit.intent.request.participant,
            key="disable",
            revision=1,
            state="disabled",
        ),
        context=CONTEXT,
    )
    reconstructed = TemporaryServicePermitAuthority(stores(), initial, redactor=REDACTOR)
    if registered_first:
        assert await reconstructed.authenticate(admission) == admission
        assert await register() == receipt
        with pytest.raises(PermissionError):
            await reconstructed.authenticate(
                admission.model_copy(update={"permit_receipt_sha256": "d" * 64})
            )
    else:
        with pytest.raises(CollaborationConflict):
            await register()
        with pytest.raises(PermissionError):
            await reconstructed.authenticate(admission)
