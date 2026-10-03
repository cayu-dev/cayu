"""Target transition contracts; native transaction/public qualification is separate."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.core._execution_profile_fixtures import (
    create_admitted_session,
    interrupt_and_release_test_invocation,
)
from tests.core.test_temporary_continuation_contracts import admission
from tests.core.test_temporary_continuation_store import run_owned
from tests.core.test_temporary_continuation_store import store_factory as store_factory

from cayu.collaboration._permits import ReceivingSettlementReceipt
from cayu.messages import Message
from cayu.runtime._session_continuation import ContinuationConflict, continuation_digest
from cayu.runtime._temporary_continuation import TemporaryServiceExecution, TemporaryServiceRecord
from cayu.sessions._temporary_service_target import (
    TemporaryServiceTarget,
    acknowledge_side_target,
    exclude_side_target,
    return_side_target,
)
from cayu.sessions.base import RunRequest


def side_admission(*, session=None):
    original = admission()
    intent = original.dispatch.intent
    intent = intent.model_copy(
        update={
            "mode": "side_session",
            "target": intent.target.model_copy(
                update={
                    "object_id": "existing-side" if session is None else session.id,
                    "incarnation": intent.target.incarnation
                    if session is None
                    else session.instance_id,
                }
            ),
            "question": intent.question.model_copy(
                update={"deadline_at_ms": int(datetime.now(UTC).timestamp() * 1000) + 60_000}
            ),
        }
    )
    dispatch = original.dispatch.model_copy(
        update={
            "intent": intent,
            "expected_run_epoch": original.dispatch.expected_run_epoch
            if session is None
            else session.run_epoch,
        }
    )
    permit = original.permit.model_copy(
        update={
            "intent": original.permit.intent.model_copy(
                update={
                    "request": original.permit.intent.request.model_copy(
                        update={
                            "target": intent.target,
                            "admission_commitment": continuation_digest(dispatch),
                        }
                    )
                }
            )
        }
    )
    return original.model_validate(
        original.model_copy(update={"dispatch": dispatch, "permit": permit}).model_dump()
    )


def test_target_exclusion_replay_and_acknowledgement_are_exact():
    expected = side_admission().preparation
    target = TemporaryServiceTarget(
        service=TemporaryServiceRecord(admission=expected, state="prepared")
    )
    excluded = exclude_side_target(target, expected)
    assert excluded.service.settlement is not None
    assert excluded.service.settlement.proves_exclusion
    assert not excluded.source_acknowledged
    assert exclude_side_target(excluded, expected) == excluded
    reconstructed = TemporaryServiceTarget.model_validate_json(excluded.model_dump_json())
    assert exclude_side_target(reconstructed, expected) == excluded
    acknowledged = acknowledge_side_target(reconstructed, excluded.service)
    assert acknowledged.source_acknowledged
    assert acknowledge_side_target(acknowledged, excluded.service) == acknowledged
    with pytest.raises(ContinuationConflict):
        acknowledge_side_target(target, excluded.service)
    changed = expected.model_copy(
        update={"dispatch": expected.dispatch.model_copy(update={"expected_run_epoch": 4})}
    )
    with pytest.raises(ContinuationConflict):
        exclude_side_target(reconstructed, changed)


def test_admitted_target_requires_native_return_not_exclusion():
    expected = side_admission()
    intent = expected.dispatch.intent
    execution = TemporaryServiceExecution(
        receipt_id="native-admit",
        receipt_sha256="a" * 64,
        admission_command_sha256=expected.admission_command_sha256,
        session_id=intent.target.object_id,
        session_instance_id=intent.target.incarnation,
        invocation_id=intent.invocation_id,
        run_epoch=expected.dispatch.expected_run_epoch + 1,
    )
    admitted = TemporaryServiceTarget(
        service=TemporaryServiceRecord(admission=expected, state="admitted", execution=execution)
    )
    with pytest.raises(ContinuationConflict):
        exclude_side_target(admitted, expected.preparation)
    with pytest.raises(ContinuationConflict):
        acknowledge_side_target(admitted, admitted.service)
    returned = admitted.service.model_copy(
        update={
            "state": "returned",
            "returned_writer_generation": execution.run_epoch + 1,
            "released_session_status": "completed",
            "settlement": ReceivingSettlementReceipt(
                expected=expected.permit,
                receiving_owner=intent.target.owner,
                receipt_id="native-release",
                outcome="quiescent",
            ),
        }
    )
    settled = return_side_target(admitted, returned)
    assert not settled.source_acknowledged
    assert return_side_target(settled, returned) == settled
    assert acknowledge_side_target(settled, returned).source_acknowledged
    with pytest.raises(ContinuationConflict):
        return_side_target(admitted, admitted.service)


def test_same_session_cannot_create_a_second_target_fence():
    with pytest.raises(ValueError):
        TemporaryServiceTarget(
            service=TemporaryServiceRecord(admission=admission().preparation, state="prepared")
        )


def test_native_target_fence_replay_exclusion_and_retention(store_factory):
    async def run(store):
        admitted = await create_admitted_session(
            store,
            request=RunRequest(
                agent_name="target",
                session_id="side-" + uuid4().hex,
                messages=[Message.text("user", "Prepare this existing target.")],
            ),
            provider_name="provider",
            model="model",
        )
        await interrupt_and_release_test_invocation(store, admitted.session.id)
        session = await store.load(admitted.session.id)
        assert session is not None
        expected = side_admission(session=session).preparation
        prepared = TemporaryServiceTarget(
            service=TemporaryServiceRecord(admission=expected, state="prepared")
        )
        assert (
            await store._publish_temporary_service_target(previous=None, proposed=prepared)
            == prepared
        )
        assert await store._load_temporary_service_target(expected) == prepared
        reconstructed = store_factory()
        try:
            assert await reconstructed._load_temporary_service_target(expected) == prepared
        finally:
            if reconstructed is not store:
                await reconstructed.close()
        assert (
            await store._publish_temporary_service_target(previous=prepared, proposed=prepared)
            == prepared
        )
        with pytest.raises(ContinuationConflict):
            await store.delete_session(session.id)
        changed = expected.model_copy(
            update={
                "dispatch": expected.dispatch.model_copy(
                    update={"expected_run_epoch": session.run_epoch + 1}
                )
            }
        )
        with pytest.raises(ValueError):
            await store._load_temporary_service_target(changed)
        excluded = exclude_side_target(prepared, expected)
        assert (
            await store._publish_temporary_service_target(previous=prepared, proposed=excluded)
            == excluded
        )
        with pytest.raises(ContinuationConflict):
            await store.delete_session(session.id)
        settled = acknowledge_side_target(excluded, excluded.service)
        assert (
            await store._publish_temporary_service_target(previous=excluded, proposed=settled)
            == settled
        )
        assert await store._load_temporary_service_target(expected) == settled
        # Independent retry retains the exact terminal outcome and existing ACK.
        reconstructed = store_factory()
        try:
            for previous, proposed in (
                (excluded, settled),
                (prepared, excluded),
                (excluded, excluded),
            ):
                assert (
                    await reconstructed._publish_temporary_service_target(
                        previous=previous, proposed=proposed
                    )
                    == settled
                )
            assert excluded.service.settlement is not None
            changed = settled.model_copy(
                update={
                    "service": settled.service.model_copy(
                        update={
                            "settlement": excluded.service.settlement.model_copy(
                                update={"receipt_id": "different"}
                            )
                        }
                    )
                }
            )
            with pytest.raises(ContinuationConflict):
                await reconstructed._publish_temporary_service_target(
                    previous=excluded, proposed=changed
                )
            assert await reconstructed._load_temporary_service_target(expected) == settled
        finally:
            if reconstructed is not store:
                await reconstructed.close()
        await store.delete_session(session.id)
        assert await store.load(session.id) is None

    asyncio.run(run_owned(store_factory, run))


@pytest.mark.parametrize("ordering", ("concurrent", "admission_first", "exclusion_first"))
def test_native_side_target_admission_and_exclusion_race(store_factory, ordering):
    from tests.core.test_temporary_continuation_admission import native_command

    from cayu.runtime._temporary_continuation_scope import temporary_admission_scope
    from cayu.sessions.base import InMemorySessionStore

    async def run(store):
        expected, command = await native_command(store, side_session=True)
        source = TemporaryServiceRecord(admission=expected, state="reserved")
        await store._publish_temporary_continuation_service(previous=None, proposed=source)
        original_source = await store.load(expected.dispatch.intent.ticket.session_id)
        prepared = TemporaryServiceTarget(
            service=TemporaryServiceRecord(admission=expected.preparation, state="prepared")
        )
        await store._publish_temporary_service_target(previous=None, proposed=prepared)
        excluded = exclude_side_target(prepared, expected.preparation)
        other = store_factory()
        try:

            async def admit():
                with temporary_admission_scope(expected, command):
                    return await store.apply_invocation_lifecycle_command(command)

            async def exclude():
                return await other._publish_temporary_service_target(
                    previous=prepared, proposed=excluded
                )

            if ordering == "concurrent":
                admitted, fenced = await asyncio.gather(admit(), exclude(), return_exceptions=True)
            elif ordering == "admission_first":
                admitted = await admit()
                [fenced] = await asyncio.gather(exclude(), return_exceptions=True)
            else:
                fenced = await exclude()
                [admitted] = await asyncio.gather(admit(), return_exceptions=True)
            assert isinstance(admitted, BaseException) != isinstance(fenced, BaseException)
            observed = await store._load_temporary_service_target(expected.preparation)
            assert observed is not None
            if isinstance(admitted, BaseException):
                assert isinstance(admitted, ContinuationConflict)
                assert observed == excluded
                with (
                    temporary_admission_scope(expected, command),
                    pytest.raises(ContinuationConflict),
                ):
                    await store.apply_invocation_lifecycle_command(command)
            else:
                assert observed.service.state == "admitted"
                with temporary_admission_scope(expected, command):
                    assert (
                        await store.apply_invocation_lifecycle_command(command)
                    ).session == admitted.session
            assert await store.load(expected.dispatch.intent.ticket.session_id) == original_source
            assert await store._load_temporary_continuation_service(expected) == source
            if not isinstance(admitted, BaseException):
                await interrupt_and_release_test_invocation(store, command.session_id)
                returned = await store._reconcile_temporary_continuation_service(expected)
                assert returned.state == "returned"
                target = await store._load_temporary_service_target(expected)
                assert target is not None and target.source_acknowledged
                assert target.service == returned
                assert await store._reconcile_temporary_continuation_service(expected) == returned
                source_after_return = await store.load(expected.dispatch.intent.ticket.session_id)
                assert source_after_return is not None and original_source is not None
                # Reconciliation publishes the source receipt and updates its
                # mutation timestamp, but cannot replace the source writer.
                assert source_after_return.model_dump(
                    exclude={"updated_at", "last_activity_at"}
                ) == original_source.model_dump(exclude={"updated_at", "last_activity_at"})
        finally:
            if other is not store and not isinstance(other, InMemorySessionStore):
                await other.close()

    asyncio.run(run_owned(store_factory, run))


def test_native_side_target_exclusion_requires_preparation(store_factory):
    from tests.core.test_temporary_continuation_admission import native_command

    async def run(store):
        admission_value, _ = await native_command(store, side_session=True)
        expected = admission_value.preparation
        prepared = TemporaryServiceTarget(
            service=TemporaryServiceRecord(admission=expected, state="prepared")
        )
        excluded = exclude_side_target(prepared, expected)
        with pytest.raises(ContinuationConflict):
            await store._publish_temporary_service_target(previous=None, proposed=excluded)
        assert await store._load_temporary_service_target(expected) is None
        await store._prepare_temporary_side_service(expected)
        assert await store._load_temporary_service_target(expected) == prepared
        assert (
            await store._publish_temporary_service_target(previous=prepared, proposed=excluded)
            == excluded
        )
        with pytest.raises(ContinuationConflict):
            await store._prepare_temporary_side_service(expected)
        assert await store._load_temporary_service_target(expected) == excluded

    asyncio.run(run_owned(store_factory, run))
