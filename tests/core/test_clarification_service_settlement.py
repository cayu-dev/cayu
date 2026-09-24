"""Joined native permit/receiving settlement, not the public provider journey."""

import asyncio
import json
import os
import sys

import pytest
from tests.core._execution_profile_fixtures import interrupt_and_release_test_invocation
from tests.core.test_temporary_continuation_admission import native_command
from tests.core.test_temporary_continuation_permits import prepared
from tests.core.test_temporary_continuation_store import store_factory as store_factory

from cayu.collaboration._clarification_service_store import (
    discover_services_in_transaction,
    register_runtime_service_in_transaction,
)
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.runtime._invocation_lifecycle import invocation_admission_command_sha256
from cayu.runtime._session_continuation import continuation_digest
from cayu.runtime._temporary_continuation import TemporaryServiceAdmission
from cayu.runtime._temporary_continuation_permits import (
    TemporaryServicePermitAuthority,
    TemporaryServiceSettlementReader,
)
from cayu.runtime._temporary_continuation_scope import temporary_admission_scope
from cayu.sessions.base import InMemorySessionStore
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize("store_factory", ("sqlite", "postgres"), indirect=True)
def test_native_return_settlement_in_fresh_process(store_factory, tmp_path, request, monkeypatch):
    test_native_return_settles_registered_service_and_lineage(
        store_factory, tmp_path, request, monkeypatch, "process_restart"
    )


@pytest.mark.parametrize("interruption", (None, "lost_acknowledgement", "cancellation"))
def test_native_return_settles_registered_service_and_lineage(
    store_factory, tmp_path, request, monkeypatch, interruption
):
    session_store = store_factory()
    if isinstance(session_store, InMemorySessionStore):
        memory = InMemoryCollaborationStore()

        def source_factory():
            return memory
    elif isinstance(session_store, SQLiteSessionStore):

        def source_factory():
            return SQLiteCollaborationStore(tmp_path / "clarification-source.sqlite")
    else:
        dsn = request.getfixturevalue("postgres_dsn")

        def source_factory():
            return PostgresCollaborationStore(dsn, schema_mode=SchemaMode.CREATE)

    redactor = SecretRedactor()

    async def run():
        nonlocal session_store
        source = source_factory()
        try:
            _, initial, template = await prepared(source, open_question=True)
            admission, command = await native_command(session_store, template=template)
            async with source._transaction(initial.binding.application_scope, write=True) as tx:
                registered = await register_runtime_service_in_transaction(
                    source, tx, initial, admission.dispatch, redactor=redactor
                )
            receipt_hash = continuation_digest(registered.permit)
            command = command.model_copy(
                update={
                    "participant_permit_commitment": receipt_hash,
                    "participant_permit_operation": registered.permit.expected.operation.caller_key,
                }
            )
            admission = admission.model_copy(
                update={
                    "permit_receipt_sha256": receipt_hash,
                    "permit": registered.permit.expected,
                    "admission_command_sha256": invocation_admission_command_sha256(command),
                }
            )
            authority = TemporaryServicePermitAuthority(source, initial, redactor=redactor)
            authenticated = await authority.authenticate(admission)
            with temporary_admission_scope(authenticated, command):
                await session_store.apply_invocation_lifecycle_command(command)
            await interrupt_and_release_test_invocation(session_store, command.session_id)
            returned = await session_store._reconcile_temporary_continuation_service(admission)
            assert returned.state == "returned"
            reader = TemporaryServiceSettlementReader(session_store, admission, redactor=redactor)
            if interruption == "process_restart":
                value = {
                    "parent_pid": os.getpid(),
                    "initial": initial.model_dump(mode="json"),
                    "admission": admission.model_dump(mode="json"),
                }
                if isinstance(session_store, SQLiteSessionStore):
                    value.update(
                        backend="sqlite",
                        source_path=str(tmp_path / "clarification-source.sqlite"),
                        session_path=str(tmp_path / "temporary-service.sqlite"),
                    )
                else:
                    value.update(backend="postgres", dsn=request.getfixturevalue("postgres_dsn"))
                await source.close()
                await session_store.close()
                child = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "tests.core._clarification_recovery_process",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        child.communicate(json.dumps(value).encode()), timeout=90
                    )
                    assert child.returncode == 0, stderr.decode()
                    assert stdout == b""
                finally:
                    if child.returncode is None:
                        child.kill()
                        await child.wait()
                    source = source_factory()
                    session_store = store_factory()
                authority = TemporaryServicePermitAuthority(source, initial, redactor=redactor)
                reader = TemporaryServiceSettlementReader(
                    session_store, admission, redactor=redactor
                )
            elif interruption is not None:
                original = type(source)._settle_permit
                committed = asyncio.Event()
                blocked = asyncio.Event()

                async def lose_acknowledgement(self, *args, **kwargs):
                    result = await original(self, *args, **kwargs)
                    committed.set()
                    if interruption == "lost_acknowledgement":
                        raise OSError("permit settlement acknowledgement lost")
                    await blocked.wait()
                    return result

                monkeypatch.setattr(type(source), "_settle_permit", lose_acknowledgement)
                task = asyncio.create_task(authority.settle(admission, reader=reader))
                ready = asyncio.create_task(committed.wait())
                try:
                    await asyncio.wait((task, ready), return_when=asyncio.FIRST_COMPLETED)
                    if not committed.is_set():
                        await task
                        pytest.fail("Settlement never reached commit.")
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
                    with pytest.raises(OSError, match="acknowledgement lost"):
                        await task
                monkeypatch.setattr(type(source), "_settle_permit", original)
                if not isinstance(source, InMemoryCollaborationStore):
                    await source.close()
                    source = source_factory()
                    await session_store.close()
                    session_store = store_factory()
                admission = TemporaryServiceAdmission.model_validate_json(
                    admission.model_dump_json()
                )
                authority = TemporaryServicePermitAuthority(source, initial, redactor=redactor)
                reader = TemporaryServiceSettlementReader(
                    session_store, admission, redactor=redactor
                )
                async with source._transaction(
                    initial.binding.application_scope, write=False
                ) as tx:
                    assert await discover_services_in_transaction(
                        source, tx, initial, after=None, limit=1, redactor=redactor
                    ) == (registered,)
                    lineage = await tx.get(
                        "clarification_lineages",
                        operation_key(admission.dispatch.intent.question.lineage),
                    )
                    assert lineage["usage"]["pending"] == 2
            await authority.settle(admission, reader=reader)
            from tests.core.test_session_continuation import _QualifiedReceiver

            from cayu.collaboration._capabilities import CapabilityDescriptor
            from cayu.runtime._session_continuation_owner import (
                LATCH_FAMILY,
                SessionContinuationOwner,
            )

            receiving = SessionContinuationOwner(
                store=session_store,
                owner=admission.dispatch.intent.ticket.owner,
                receiver=_QualifiedReceiver(),
                receiver_capability=CapabilityDescriptor(
                    owner=admission.dispatch.intent.ticket.owner,
                    mutations=(),
                    readbacks=(LATCH_FAMILY,),
                ),
                redactor=redactor,
                temporary_permits=authority,
            )
            if interruption is None:
                from cayu.runtime._session_continuation import ContinuationUnavailable

                async def lose_native_ack(retained):
                    raise OSError("native settlement acknowledgement unavailable")

                with monkeypatch.context() as patch:
                    patch.setattr(receiving, "_acknowledge_temporary_settlement", lose_native_ack)
                    with pytest.raises(ContinuationUnavailable) as failure:
                        await receiving.reconcile_temporary(admission)
                    assert isinstance(failure.value.__cause__, OSError)
                    assert (
                        str(failure.value.__cause__)
                        == "native settlement acknowledgement unavailable"
                    )
                unacknowledged = await session_store._load_temporary_continuation_service(admission)
                assert unacknowledged is not None
                assert not unacknowledged.settlement_acknowledged
                assert unacknowledged.settlement == returned.settlement
            acknowledged = await receiving.reconcile_temporary(admission)
            assert acknowledged.settlement_acknowledged
            assert await receiving.reconcile_temporary(admission) == acknowledged
            await receiving.drain()
            async with source._transaction(initial.binding.application_scope, write=False) as tx:
                assert (
                    await discover_services_in_transaction(
                        source, tx, initial, after=None, limit=1, redactor=redactor
                    )
                    == ()
                )
                lineage = await tx.get(
                    "clarification_lineages",
                    operation_key(admission.dispatch.intent.question.lineage),
                )
                assert lineage["usage"]["pending"] == 1  # only the question remains
                assert lineage["usage"]["service_turns"] == 1
                before = await source._anchor(tx, initial, redactor)
            await authority.settle(admission, reader=reader)
            async with source._transaction(initial.binding.application_scope, write=False) as tx:
                assert await source._anchor(tx, initial, redactor) == before
            if interruption is None:
                from cayu.collaboration._clarification_retention import prune_service_record
                from cayu.collaboration._permits import PermitSettlement
                from cayu.collaboration._preparation import contract_bytes
                from cayu.collaboration.participants import CollaborationUnavailable

                async with source._transaction(
                    initial.binding.application_scope, write=False
                ) as tx:
                    settlement = PermitSettlement.model_validate(
                        await tx.get(
                            "operations",
                            operation_key(admission.permit.intent.request.settlement_operation),
                        )
                    )
                    retained_service = await tx.get(
                        "clarification_services", operation_key(admission.dispatch.intent.operation)
                    )
                changed = settlement.model_copy(
                    update={
                        "receiving_receipt": settlement.receiving_receipt.model_copy(
                            update={"receipt_id": "different-native-return"}
                        )
                    }
                )
                with pytest.raises(CollaborationUnavailable):
                    async with source._transaction(
                        initial.binding.application_scope, write=True
                    ) as tx:
                        await prune_service_record(tx, registered.permit, changed, redactor)
                with pytest.raises(OSError):
                    async with source._transaction(
                        initial.binding.application_scope, write=True
                    ) as tx:
                        await prune_service_record(tx, registered.permit, settlement, redactor)
                        raise OSError("before prune transaction commit")
                async with source._transaction(initial.binding.application_scope, write=True) as tx:
                    from cayu.collaboration._clarification_services import (
                        ClarificationServiceRecord,
                    )

                    assert (
                        await tx.get(
                            "clarification_services",
                            operation_key(admission.dispatch.intent.operation),
                        )
                        == retained_service
                    )
                    released = await prune_service_record(
                        tx, registered.permit, settlement, redactor
                    )
                    assert released == len(
                        contract_bytes(
                            ClarificationServiceRecord.model_validate(retained_service),
                            redactor=redactor,
                        )
                    )
        finally:
            await source.close()
            if not isinstance(session_store, InMemorySessionStore):
                await session_store.close()

    asyncio.run(run())
