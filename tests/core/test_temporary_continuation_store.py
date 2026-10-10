"""Native continuation-child transactions, not public receiver qualification."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from tests.core._execution_profile_fixtures import (
    create_admitted_session,
    interrupt_and_release_test_invocation,
)
from tests.core.test_session_continuation import _park, _preparation, _prepare_command
from tests.core.test_temporary_continuation_contracts import admission

from cayu.messages import Message
from cayu.runtime._session_continuation import (
    ContinuationConflict,
    ContinuationLatch,
    ContinuationNamespace,
    continuation_digest,
    continuation_namespace_id,
)
from cayu.runtime._session_continuation_scope import authenticated_latch_scope
from cayu.runtime._temporary_continuation import TemporaryServiceRecord, temporary_service_key
from cayu.sessions._temporary_continuation_store import require_service_deadline
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.requests import RunRequest


@pytest.fixture(params=("memory", "sqlite", "postgres"))
def store_factory(request, tmp_path):
    if request.param == "memory":
        store = InMemorySessionStore()
        return lambda: store
    if request.param == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        return lambda: SQLiteSessionStore(tmp_path / "temporary-service.sqlite")
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    dsn = request.getfixturevalue("postgres_dsn")
    return lambda: PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)


async def run_owned(factory, scenario):
    store = factory()
    try:
        await scenario(store)
    finally:
        if not isinstance(store, InMemorySessionStore):
            await store.close()


def _latch(ticket):
    return ContinuationLatch(
        ticket=ticket,
        wait_receipt_digest="wait-receipt",
        outcome_kind="success",
        selected_manifest=(),
        disclosure_digest="disclosure",
        latch_key="latch-1",
        outcome_digest="outcome",
        accepted_at="2026-09-23T00:00:00+00:00",
    )


@pytest.mark.parametrize("boundary", ("question", "original_wait"))
@pytest.mark.parametrize("offset_ms", (-1, 0, 1))
def test_service_admission_deadline_boundary(boundary, offset_ms):
    """Native predicate characterization; transaction tests below prove no mutation."""
    template = admission()
    intent = template.dispatch.intent
    expiry = datetime(2029, 1, 1, tzinfo=UTC)
    later = expiry + timedelta(days=1)
    intent = intent.model_copy(
        update={
            "ticket": intent.ticket.model_copy(
                update={"deadline": (expiry if boundary == "original_wait" else later).isoformat()}
            ),
            "question": intent.question.model_copy(
                update={
                    "deadline_at_ms": int(
                        (expiry if boundary == "question" else later).timestamp() * 1000
                    )
                }
            ),
        }
    )
    dispatch = template.dispatch.model_copy(update={"intent": intent})
    permit = template.permit
    registration = permit.intent.request.model_copy(
        update={"admission_commitment": continuation_digest(dispatch)}
    )
    permit = permit.model_copy(
        update={"intent": permit.intent.model_copy(update={"request": registration})}
    )
    proposed = TemporaryServiceRecord(
        admission=template.model_copy(update={"dispatch": dispatch, "permit": permit}),
        state="reserved",
    )
    now = expiry + timedelta(milliseconds=offset_ms)
    if offset_ms < 0:
        require_service_deadline(proposed, now)
    else:
        with pytest.raises(ContinuationConflict, match="deadline"):
            require_service_deadline(proposed, now)


async def prepared_service(store, *, template=None, retained_tickets=1):
    custom = template is not None
    template = admission() if template is None else template
    admitted = await create_admitted_session(
        store,
        request=RunRequest(
            agent_name="owner",
            session_id="temporary-service-" + uuid4().hex,
            messages=[Message.text("user", "wait")],
        ),
        provider_name="provider",
        model="model",
    )
    session = admitted.session
    old_ticket = template.dispatch.intent.ticket
    namespace = ContinuationNamespace(
        session_id=session.id,
        session_instance_id=session.instance_id,
        owner=old_ticket.owner,
        namespace_id=continuation_namespace_id(session.id, session.instance_id, old_ticket.owner),
    )
    ticket = old_ticket.model_copy(
        update={
            "namespace": namespace,
            "session_id": session.id,
            "session_instance_id": session.instance_id,
            "interaction_id": admitted.active_invocation_profile.interaction_id,
            "writer_generation": session.run_epoch,
            "state": "ARMING",
            "revision": 1,
        }
    )
    await _prepare_command(store, await _preparation(store, ticket))
    parked = await _park(store, ticket)
    if retained_tickets > 1:
        from tests.core.test_session_continuation import _context, _QualifiedReceiver

        from cayu.collaboration._capabilities import CapabilityDescriptor
        from cayu.runtime._session_continuation import ContinuationRetirement
        from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner
        from cayu.vaults.redaction import SecretRedactor

        owner = SessionContinuationOwner(
            store=store,
            owner=ticket.owner,
            receiver=_QualifiedReceiver(),
            receiver_capability=CapabilityDescriptor(
                owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
            ),
            redactor=SecretRedactor(),
        )
        invocation = await _context(store, session.id)
        for index in range(1, retained_tickets):
            extra = ticket.model_copy(update={"registration_key": f"retained-{index}"})
            await _prepare_command(store, await _preparation(store, extra))
            await owner.exclude(
                ContinuationRetirement(
                    ticket=extra,
                    control_id=f"retire-{index}",
                    reason="cancelled",
                    retired_at=datetime.now(UTC).isoformat(),
                ),
                invocation=invocation,
            )
        await owner.drain()
    await interrupt_and_release_test_invocation(store, session.id)
    intent = template.dispatch.intent.model_copy(
        update={
            "ticket": parked.ticket,
            "question": template.dispatch.intent.question
            if custom
            else template.dispatch.intent.question.model_copy(
                update={"deadline_at_ms": int(datetime.now(UTC).timestamp() * 1000) + 60_000}
            ),
            "target": template.dispatch.intent.target.model_copy(
                update={"object_id": session.id, "incarnation": session.instance_id}
            ),
        }
    )
    dispatch = template.dispatch.model_copy(
        update={"intent": intent, "expected_run_epoch": session.run_epoch + 1}
    )
    registration = template.permit.intent.request.model_copy(
        update={"target": intent.target, "admission_commitment": continuation_digest(dispatch)}
    )
    permit = template.permit.model_copy(
        update={"intent": template.permit.intent.model_copy(update={"request": registration})}
    )
    return TemporaryServiceRecord(
        admission=template.model_copy(update={"dispatch": dispatch, "permit": permit}),
        state="reserved",
    )


def test_erasure_collects_children_beyond_ticket_only_bound(store_factory):
    from tests.core.test_session_continuation import _context, _QualifiedReceiver

    from cayu.collaboration._capabilities import CapabilityDescriptor
    from cayu.collaboration._permits import ReceivingSettlementReceipt
    from cayu.runtime._session_continuation import ContinuationRetirement
    from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner
    from cayu.sessions._session_continuation_store import MAX_RETAINED_TICKETS
    from cayu.vaults.redaction import SecretRedactor

    async def run(store):
        proposed = await prepared_service(store, retained_tickets=MAX_RETAINED_TICKETS)
        await store._publish_temporary_continuation_service(previous=None, proposed=proposed)
        excluded = proposed.model_copy(
            update={
                "state": "excluded",
                "settlement": ReceivingSettlementReceipt(
                    expected=proposed.admission.permit,
                    receiving_owner=proposed.intent.target.owner,
                    receipt_id="excluded-before-admission",
                    outcome="quiescent",
                    admission_excluded=True,
                ),
            }
        )
        await store._publish_temporary_continuation_service(previous=proposed, proposed=excluded)
        ticket = proposed.intent.ticket
        owner = SessionContinuationOwner(
            store=store,
            owner=ticket.owner,
            receiver=_QualifiedReceiver(),
            receiver_capability=CapabilityDescriptor(
                owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
            ),
            redactor=SecretRedactor(),
        )
        current = await store.load_continuation_ticket(
            ticket.session_id,
            session_instance_id=ticket.session_instance_id,
            registration_key=ticket.registration_key,
        )
        await owner.exclude(
            ContinuationRetirement(
                ticket=current.ticket,
                control_id="final-retirement",
                reason="cancelled",
                retired_at=datetime.now(UTC).isoformat(),
            ),
            invocation=await _context(store, ticket.session_id),
        )
        # All 66 records must reach the shared semantic validator, which still
        # rejects the real pending acknowledgement rather than a false quota.
        with pytest.raises(ContinuationConflict, match="settlement acknowledgement"):
            await store.validate_session_closure_admission(ticket.session_id)
        acknowledged = excluded.model_copy(update={"settlement_acknowledged": True})
        await store._publish_temporary_continuation_service(
            previous=excluded,
            proposed=acknowledged,
        )
        await store.validate_session_closure_admission(ticket.session_id)
        await store.delete_session(ticket.session_id)
        assert await store.load(ticket.session_id) is None
        await owner.drain()

    asyncio.run(run_owned(store_factory, run))


def test_terminal_ack_replays_after_competing_owner_commit(store_factory):
    from cayu.collaboration._permits import ReceivingSettlementReceipt

    async def run(store):
        other = store_factory()
        try:
            proposed = await prepared_service(store)
            await store._publish_temporary_continuation_service(previous=None, proposed=proposed)
            excluded = proposed.model_copy(
                update={
                    "state": "excluded",
                    "settlement": ReceivingSettlementReceipt(
                        expected=proposed.admission.permit,
                        receiving_owner=proposed.intent.target.owner,
                        receipt_id="excluded-before-admission",
                        outcome="quiescent",
                        admission_excluded=True,
                    ),
                }
            )
            await store._publish_temporary_continuation_service(
                previous=proposed, proposed=excluded
            )
            assert (
                await other._publish_temporary_continuation_service(
                    previous=proposed, proposed=excluded
                )
                == excluded
            )
            acknowledged = excluded.model_copy(update={"settlement_acknowledged": True})
            # Both independent owners prepared against the same pre-ACK record.
            results = await asyncio.gather(
                store._publish_temporary_continuation_service(
                    previous=excluded, proposed=acknowledged
                ),
                other._publish_temporary_continuation_service(
                    previous=excluded, proposed=acknowledged
                ),
            )
            assert results == [acknowledged, acknowledged]
            ticket = proposed.intent.ticket
            terminal_parent = await store.load_continuation_ticket(
                ticket.session_id,
                session_instance_id=ticket.session_instance_id,
                registration_key=ticket.registration_key,
            )
            assert (
                await other._publish_temporary_continuation_service(
                    previous=excluded, proposed=acknowledged
                )
                == acknowledged
            )
            # A delayed settlement publisher must not remove an existing ACK.
            assert (
                await other._publish_temporary_continuation_service(
                    previous=proposed, proposed=excluded
                )
                == acknowledged
            )
            assert excluded.settlement is not None
            conflicting = acknowledged.model_copy(
                update={
                    "settlement": excluded.settlement.model_copy(update={"receipt_id": "wrong"})
                }
            )
            with pytest.raises(ContinuationConflict):
                await other._publish_temporary_continuation_service(
                    previous=excluded, proposed=conflicting
                )
            retained = await store.load_session_operation(
                proposed.intent.ticket.session_id, temporary_service_key(proposed.intent.operation)
            )
            assert TemporaryServiceRecord.model_validate(retained) == acknowledged
            assert (
                await other.load_continuation_ticket(
                    ticket.session_id,
                    session_instance_id=ticket.session_instance_id,
                    registration_key=ticket.registration_key,
                )
                == terminal_parent
            )
        finally:
            if other is not store:
                await other.close()

    asyncio.run(run_owned(store_factory, run))


def test_native_service_reservation_and_latch_arbitrate_and_replay(store_factory):
    async def run(store):
        proposed = await prepared_service(store)
        ticket = proposed.intent.ticket
        result = await store._publish_temporary_continuation_service(
            previous=None, proposed=proposed
        )
        assert result == proposed
        assert await store._load_temporary_continuation_service(proposed.admission) == proposed
        retained = await store.load_continuation_ticket(
            ticket.session_id,
            session_instance_id=ticket.session_instance_id,
            registration_key=ticket.registration_key,
        )
        assert retained.ticket.state == "SERVICING"
        assert retained.ticket.writer_generation == ticket.writer_generation
        assert retained.services[0].key == temporary_service_key(proposed.intent.operation)
        assert (
            await store._publish_temporary_continuation_service(
                previous=proposed, proposed=proposed
            )
            == proposed
        )
        latch = _latch(ticket)
        with authenticated_latch_scope(latch):
            latched = await store.latch_continuation(latch)
        assert latched.latch == latch
        assert latched.ticket.state == "SERVICING"
        with pytest.raises(ContinuationConflict):
            await store.delete_session(ticket.session_id)
        with pytest.raises(ContinuationConflict):
            await store._publish_temporary_continuation_service(previous=None, proposed=proposed)

    asyncio.run(run_owned(store_factory, run))


def test_native_latch_first_rejects_service_without_child_write(store_factory):
    async def run(store):
        proposed = await prepared_service(store)
        ticket = proposed.intent.ticket
        latch = _latch(ticket)
        with authenticated_latch_scope(latch):
            await store.latch_continuation(latch)
        with pytest.raises(ContinuationConflict):
            await store._publish_temporary_continuation_service(previous=None, proposed=proposed)
        assert (
            await store.load_session_operation(
                ticket.session_id, temporary_service_key(proposed.intent.operation)
            )
            is None
        )
        retained = await store.load_continuation_ticket(
            ticket.session_id,
            session_instance_id=ticket.session_instance_id,
            registration_key=ticket.registration_key,
        )
        assert retained.ticket.state == "WAITING"
        assert retained.services == ()

    asyncio.run(run_owned(store_factory, run))


@pytest.mark.parametrize("capacity_only", (False, True))
def test_native_service_restart_and_exact_exclusion_preserve_latch(store_factory, capacity_only):
    from cayu.collaboration._permits import ReceivingSettlementReceipt

    async def run():
        first = store_factory()
        try:
            proposed = await prepared_service(first)
            if capacity_only:
                proposed = TemporaryServiceRecord(
                    admission=proposed.acknowledged_admission.preparation, state="prepared"
                )
            await first._publish_temporary_continuation_service(previous=None, proposed=proposed)
            ticket = proposed.intent.ticket
            parent = await first.load_continuation_ticket(
                ticket.session_id,
                session_instance_id=ticket.session_instance_id,
                registration_key=ticket.registration_key,
            )
            assert parent.ticket.state == ("WAITING" if capacity_only else "SERVICING")
            latch = _latch(ticket)
            with authenticated_latch_scope(latch):
                await first.latch_continuation(latch)
        finally:
            if not isinstance(first, InMemorySessionStore):
                await first.close()
        second = store_factory()
        try:
            raw = await second.load_session_operation(
                ticket.session_id, temporary_service_key(proposed.intent.operation)
            )
            assert TemporaryServiceRecord.model_validate(raw) == proposed
            assert await second._load_temporary_continuation_service(proposed.admission) == proposed
            changed = (
                proposed.admission.model_copy(
                    update={
                        "permit": proposed.admission.permit.model_copy(
                            update={
                                "intent": proposed.admission.permit.intent.model_copy(
                                    update={
                                        "request": proposed.admission.permit.intent.request.model_copy(
                                            update={"expected_lifecycle_revision": 2}
                                        )
                                    }
                                )
                            }
                        )
                    }
                )
                if capacity_only
                else proposed.admission.model_copy(update={"permit_receipt_sha256": "b" * 64})
            )
            with pytest.raises(ContinuationConflict):
                await second._load_temporary_continuation_service(changed)
            excluded = proposed.model_copy(
                update={
                    "state": "excluded",
                    "settlement": ReceivingSettlementReceipt(
                        expected=proposed.admission.permit,
                        receiving_owner=proposed.intent.target.owner,
                        receipt_id="excluded-before-admission",
                        outcome="quiescent",
                        admission_excluded=True,
                    ),
                }
            )
            await second._publish_temporary_continuation_service(
                previous=proposed, proposed=excluded
            )
            restored = await second.load_continuation_ticket(
                ticket.session_id,
                session_instance_id=ticket.session_instance_id,
                registration_key=ticket.registration_key,
            )
            assert restored.ticket.state == "WAITING"
            assert restored.latch == latch
            assert restored.ticket.writer_generation == ticket.writer_generation
            assert (
                await second._publish_temporary_continuation_service(
                    previous=excluded, proposed=excluded
                )
                == excluded
            )
        finally:
            if not isinstance(second, InMemorySessionStore):
                await second.close()

    asyncio.run(run())


def test_independent_service_and_latch_workers_share_native_arbitration(store_factory):
    async def run(first):
        proposed = await prepared_service(first)
        ticket = proposed.intent.ticket
        latch = _latch(ticket)
        second = store_factory()
        try:
            start = asyncio.Event()

            async def reserve():
                await start.wait()
                return await first._publish_temporary_continuation_service(
                    previous=None, proposed=proposed
                )

            async def publish_latch():
                await start.wait()
                with authenticated_latch_scope(latch):
                    return await second.latch_continuation(latch)

            service_task = asyncio.create_task(reserve())
            latch_task = asyncio.create_task(publish_latch())
            start.set()
            service_result, latch_result = await asyncio.gather(
                service_task, latch_task, return_exceptions=True
            )
            assert not isinstance(latch_result, BaseException)
            retained = await second.load_continuation_ticket(
                ticket.session_id,
                session_instance_id=ticket.session_instance_id,
                registration_key=ticket.registration_key,
            )
            assert retained.latch == latch
            if isinstance(service_result, ContinuationConflict):
                assert retained.ticket.state == "WAITING"
                assert retained.services == ()
                assert await second._load_temporary_continuation_service(proposed.admission) is None
            else:
                assert service_result == proposed
                assert retained.ticket.state == "SERVICING"
                assert len(retained.services) == 1
                assert (
                    await second._load_temporary_continuation_service(proposed.admission)
                    == proposed
                )
        finally:
            if not isinstance(second, InMemorySessionStore):
                await second.close()

    asyncio.run(run_owned(store_factory, run))


@pytest.mark.parametrize("interruption", ("acknowledgement_loss", "cancellation"))
def test_committed_service_retains_responsibility_after_interruption(
    store_factory, interruption, monkeypatch
):
    async def run(first):
        proposed = await prepared_service(first)
        original = first.publish_session_operation_guarded_with_store_time
        committed = asyncio.Event()
        blocked = asyncio.Event()

        async def lose_acknowledgement(*args, **kwargs):
            await original(*args, **kwargs)
            committed.set()
            if interruption == "acknowledgement_loss":
                raise OSError("injected acknowledgement loss after commit")
            await blocked.wait()

        monkeypatch.setattr(
            first, "publish_session_operation_guarded_with_store_time", lose_acknowledgement
        )
        task = asyncio.create_task(
            first._publish_temporary_continuation_service(previous=None, proposed=proposed)
        )
        await committed.wait()
        if interruption == "cancellation":
            task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
            assert task.cancelling() == 1
        else:
            with pytest.raises(OSError, match="after commit"):
                await task
        monkeypatch.setattr(first, "publish_session_operation_guarded_with_store_time", original)
        second = store_factory()
        try:
            observed = await second._load_temporary_continuation_service(proposed.admission)
            assert observed == proposed
            assert observed.state == "reserved"
            assert observed.settlement is None
            assert (
                await second._publish_temporary_continuation_service(
                    previous=observed, proposed=observed
                )
                == proposed
            )
            ticket = proposed.intent.ticket
            retained = await second.load_continuation_ticket(
                ticket.session_id,
                session_instance_id=ticket.session_instance_id,
                registration_key=ticket.registration_key,
            )
            assert retained.ticket.state == "SERVICING"
            assert len(retained.services) == 1
        finally:
            if not isinstance(second, InMemorySessionStore):
                await second.close()

    asyncio.run(run_owned(store_factory, run))


@pytest.mark.parametrize("expired_boundary", ("question", "original_wait"))
def test_native_expired_service_is_rejected_without_source_mutation(
    store_factory, expired_boundary
):
    async def run(store):
        template = None
        if expired_boundary == "original_wait":
            template = admission()
            selected = template.dispatch.intent
            selected = selected.model_copy(
                update={
                    "ticket": selected.ticket.model_copy(
                        update={"deadline": "2000-01-01T00:00:00+00:00"}
                    ),
                    "question": selected.question.model_copy(
                        update={
                            "deadline_at_ms": int(datetime.now(UTC).timestamp() * 1000) + 60_000
                        }
                    ),
                }
            )
            template = template.model_copy(
                update={"dispatch": template.dispatch.model_copy(update={"intent": selected})}
            )
        original = await prepared_service(store, template=template)
        dispatch = original.admission.dispatch
        intent = dispatch.intent
        if expired_boundary == "question":
            intent = intent.model_copy(
                update={"question": intent.question.model_copy(update={"deadline_at_ms": 1})}
            )
        dispatch = dispatch.model_copy(update={"intent": intent})
        permit = original.admission.permit
        registration = permit.intent.request.model_copy(
            update={"admission_commitment": continuation_digest(dispatch)}
        )
        permit = permit.model_copy(
            update={"intent": permit.intent.model_copy(update={"request": registration})}
        )
        expired = TemporaryServiceRecord(
            admission=original.admission.model_copy(
                update={"dispatch": dispatch, "permit": permit}
            ),
            state="reserved",
        )
        ticket = intent.ticket
        before = await store.load(ticket.session_id)
        retained_before = await store.load_continuation_ticket(
            ticket.session_id,
            session_instance_id=ticket.session_instance_id,
            registration_key=ticket.registration_key,
        )
        with pytest.raises(ContinuationConflict, match="deadline"):
            await store._publish_temporary_continuation_service(previous=None, proposed=expired)
        assert await store.load(ticket.session_id) == before
        assert await store._load_temporary_continuation_service(expired.admission) is None
        assert (
            await store.load_continuation_ticket(
                ticket.session_id,
                session_instance_id=ticket.session_instance_id,
                registration_key=ticket.registration_key,
            )
            == retained_before
        )

    asyncio.run(run_owned(store_factory, run))


def test_native_command_payload_has_no_permit_hash_cycle():
    from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
    from cayu.runtime._invocation_lifecycle import (
        AdmitInvocationCommand,
        invocation_admission_command_sha256,
        invocation_checkpoint_state_sha256,
    )
    from cayu.runtime._temporary_continuation import (
        require_temporary_service_command,
        temporary_admission_payload_sha256,
    )
    from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
    from cayu.sessions.records import SessionStatus
    from cayu.tools.exposure import tool_capability_ceiling_from_session_metadata

    async def run():
        store = InMemorySessionStore()
        proposed = await prepared_service(store)
        session = await store.load(proposed.intent.ticket.session_id)
        assert session is not None
        checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint(session.id)
        active = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert active is not None
        command = AdmitInvocationCommand(
            session_id=session.id,
            expected_session_instance_id=session.instance_id,
            expected_statuses=(SessionStatus.INTERRUPTED,),
            expected_run_epoch=session.run_epoch,
            expected_checkpoint_sha256=invocation_checkpoint_state_sha256(checkpoint),
            target_active_profile=active.model_copy(
                update={
                    "interaction_id": proposed.intent.invocation_id,
                    "run_epoch": session.run_epoch + 1,
                }
            ),
            continued_interaction_id=proposed.intent.invocation_id,
            tool_capability_ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
            expected_active_profile=active,
            temporary_service_operation_key=temporary_service_key(proposed.intent.operation),
        )
        dispatch = proposed.admission.dispatch.model_copy(
            update={
                "admission_payload_sha256": temporary_admission_payload_sha256(command),
                "intent": proposed.intent.model_copy(
                    update={
                        "execution_profile_sha256": active.profile.fingerprint,
                    }
                ),
            }
        )
        permit = proposed.admission.permit
        permit = permit.model_copy(
            update={
                "intent": permit.intent.model_copy(
                    update={
                        "request": permit.intent.request.model_copy(
                            update={
                                "admission_commitment": continuation_digest(dispatch),
                            }
                        ),
                    }
                )
            }
        )
        command = command.model_copy(
            update={
                "participant_permit_operation": permit.operation.caller_key,
                "participant_permit_commitment": proposed.admission.permit_receipt_sha256,
            }
        )
        assert temporary_admission_payload_sha256(command) == dispatch.admission_payload_sha256
        admission = proposed.admission.model_copy(
            update={
                "dispatch": dispatch,
                "permit": permit,
                "admission_command_sha256": invocation_admission_command_sha256(command),
            }
        )
        require_temporary_service_command(admission, command)
        for field, value in (
            ("defer_interaction_source", True),
            ("participant_permit_commitment", "a" * 64),
            ("participant_permit_operation", "different-permit"),
        ):
            changed = command.model_copy(update={field: value})
            # Recomputing the final command hash must not bypass the separately
            # frozen payload or the authenticated permit receipt/operation.
            changed_admission = admission.model_copy(
                update={
                    "admission_command_sha256": invocation_admission_command_sha256(changed),
                }
            )
            with pytest.raises(ContinuationConflict):
                require_temporary_service_command(changed_admission, changed)

    asyncio.run(run())
