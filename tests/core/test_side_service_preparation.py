"""Native pair reservation must not leave half a side-service preparation."""

import asyncio

import pytest
from tests.core.test_temporary_continuation_admission import native_command
from tests.core.test_temporary_continuation_store import run_owned
from tests.core.test_temporary_continuation_store import store_factory as store_factory

from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._session_continuation import ContinuationConflict, continuation_digest
from cayu.runtime._temporary_continuation import (
    TemporaryServiceRecord,
    temporary_service_invocation_id,
)
from cayu.sessions._temporary_service_target import MAX_TARGET_SERVICES, TemporaryServiceTarget
from cayu.sessions.base import InMemorySessionStore


def test_joint_preparation_exact_replay(store_factory):
    async def run(store):
        admission, _ = await native_command(store, side_session=True)
        preparation = admission.preparation
        intent = preparation.dispatch.intent
        ids = (intent.ticket.session_id, intent.target.object_id)
        before = [await store.load(key) for key in ids]
        await store._prepare_temporary_side_service(preparation)
        source = await store._load_temporary_continuation_service(preparation)
        target = await store._load_temporary_service_target(preparation)
        assert source is not None and source.state == "prepared"
        assert target is not None and target.service == source
        after = [await store.load(key) for key in ids]
        assert [(item.status, item.run_epoch) for item in after] == [
            (item.status, item.run_epoch) for item in before
        ]
        checkpoints = [
            await runtime_checkpoint_session_store(store).load_checkpoint(key) for key in ids
        ]
        await store._prepare_temporary_side_service(preparation)
        assert [await store.load(key) for key in ids] == after
        assert [
            await runtime_checkpoint_session_store(store).load_checkpoint(key) for key in ids
        ] == checkpoints

    asyncio.run(run_owned(store_factory, run))


def test_target_incarnation_rejection_preserves_source(store_factory):
    async def run(store):
        admission, _ = await native_command(store, side_session=True)
        preparation = admission.preparation
        dispatch = preparation.dispatch
        intent = dispatch.intent
        source_id = intent.ticket.session_id
        before = await runtime_checkpoint_session_store(store).load_checkpoint(source_id)
        changed_intent = intent.model_copy(
            update={"target": intent.target.model_copy(update={"incarnation": "wrong-instance"})}
        )
        changed_dispatch = dispatch.model_copy(update={"intent": changed_intent})
        permit = preparation.permit
        changed = preparation.model_copy(
            update={
                "dispatch": changed_dispatch,
                "permit": permit.model_copy(
                    update={
                        "intent": permit.intent.model_copy(
                            update={
                                "request": permit.intent.request.model_copy(
                                    update={
                                        "target": changed_intent.target,
                                        "admission_commitment": continuation_digest(
                                            changed_dispatch
                                        ),
                                    }
                                )
                            }
                        )
                    }
                ),
            }
        )
        with pytest.raises(ContinuationConflict):
            await store._prepare_temporary_side_service(changed)
        assert await runtime_checkpoint_session_store(store).load_checkpoint(source_id) == before
        assert await store._load_temporary_continuation_service(preparation) is None
        assert await store._load_temporary_service_target(preparation) is None

    asyncio.run(run_owned(store_factory, run))


def test_full_target_rejection_preserves_source(store_factory):
    async def run(store):
        admission, _ = await native_command(store, side_session=True)
        preparation = admission.preparation
        intent = preparation.dispatch.intent
        source_id = intent.ticket.session_id
        # Fill the real native target ceiling, not a monkeypatched small limit.
        # These are receiving-fence characterization fixtures, not public grants.
        for index in range(MAX_TARGET_SERVICES):
            operation = intent.operation.model_copy(update={"caller_key": f"capacity-{index}"})
            dispatch = preparation.dispatch.model_copy(
                update={
                    "intent": intent.model_copy(
                        update={
                            "operation": operation,
                            "invocation_id": temporary_service_invocation_id(operation),
                        }
                    )
                }
            )
            permit = preparation.permit
            candidate = preparation.model_copy(
                update={
                    "dispatch": dispatch,
                    "permit": permit.model_copy(
                        update={
                            "intent": permit.intent.model_copy(
                                update={
                                    "request": permit.intent.request.model_copy(
                                        update={
                                            "source_operation": operation,
                                            "admission_commitment": continuation_digest(dispatch),
                                        }
                                    )
                                }
                            )
                        }
                    ),
                }
            )
            await store._publish_temporary_service_target(
                previous=None,
                proposed=TemporaryServiceTarget(
                    service=TemporaryServiceRecord(admission=candidate, state="prepared")
                ),
            )
        before = await runtime_checkpoint_session_store(store).load_checkpoint(source_id)
        with pytest.raises(ContinuationConflict, match="capacity"):
            await store._prepare_temporary_side_service(preparation)
        assert await runtime_checkpoint_session_store(store).load_checkpoint(source_id) == before
        assert await store._load_temporary_continuation_service(preparation) is None
        assert await store._load_temporary_service_target(preparation) is None

    asyncio.run(run_owned(store_factory, run))


def test_deleted_target_rejection_preserves_source(store_factory):
    async def run(store):
        admission, _ = await native_command(store, side_session=True)
        preparation = admission.preparation
        intent = preparation.dispatch.intent
        source_id = intent.ticket.session_id
        before = await runtime_checkpoint_session_store(store).load_checkpoint(source_id)
        await store.delete_session(intent.target.object_id)
        with pytest.raises(KeyError):
            await store._prepare_temporary_side_service(preparation)
        assert await runtime_checkpoint_session_store(store).load_checkpoint(source_id) == before
        assert await store._load_temporary_continuation_service(preparation) is None

    asyncio.run(run_owned(store_factory, run))


def test_two_store_preparation_replay(store_factory):
    async def run(store):
        admission, _ = await native_command(store, side_session=True)
        preparation = admission.preparation
        other = store_factory()
        try:
            await asyncio.gather(
                store._prepare_temporary_side_service(preparation),
                other._prepare_temporary_side_service(preparation),
            )
            source = await other._load_temporary_continuation_service(preparation)
            target = await other._load_temporary_service_target(preparation)
            assert source is not None and source.state == "prepared"
            assert target is not None and target.service == source
        finally:
            if not isinstance(other, InMemorySessionStore):
                await other.close()

    asyncio.run(run_owned(store_factory, run))


def test_second_preparation_failure_has_no_partial_write(store_factory, monkeypatch):
    async def run(store):
        from cayu.sessions import _temporary_service_target

        admission, _ = await native_command(store, side_session=True)
        preparation = admission.preparation
        intent = preparation.dispatch.intent
        ids = (intent.ticket.session_id, intent.target.object_id)
        sessions = [await store.load(key) for key in ids]
        checkpoints = [
            await runtime_checkpoint_session_store(store).load_checkpoint(key) for key in ids
        ]

        def fail_target(*args, **kwargs):
            raise RuntimeError("Injected target preparation failure")

        with monkeypatch.context() as patch:
            patch.setattr(_temporary_service_target, "publish_target_record", fail_target)
            with pytest.raises(RuntimeError, match="Injected target"):
                await store._prepare_temporary_side_service(preparation)
        assert [await store.load(key) for key in ids] == sessions
        assert [
            await runtime_checkpoint_session_store(store).load_checkpoint(key) for key in ids
        ] == checkpoints
        assert await store._load_temporary_continuation_service(preparation) is None
        assert await store._load_temporary_service_target(preparation) is None
        # Failure did not strand a native capacity reservation or poison retry.
        await store._prepare_temporary_side_service(preparation)
        assert await store._load_temporary_continuation_service(preparation) is not None

    asyncio.run(run_owned(store_factory, run))
