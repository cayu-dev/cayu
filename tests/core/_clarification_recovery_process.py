"""Fresh-process qualification of retained native service settlement."""

import asyncio
import json
import os
import sys

from cayu.collaboration._clarification_service_store import discover_services_in_transaction
from cayu.collaboration.participants import CollaborationInitialization
from cayu.runtime._temporary_continuation import TemporaryServiceAdmission
from cayu.runtime._temporary_continuation_permits import (
    TemporaryServicePermitAuthority,
    TemporaryServiceSettlementReader,
)
from cayu.vaults.redaction import SecretRedactor


async def recover(value):
    assert os.getpid() != value["parent_pid"]
    if value["backend"] == "sqlite":
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
        from cayu.storage.sqlite import SQLiteSessionStore

        source = SQLiteCollaborationStore(value["source_path"])
        sessions = SQLiteSessionStore(value["session_path"])
    else:
        from cayu.storage.collaboration_postgres import PostgresCollaborationStore
        from cayu.storage.postgres import PostgresSessionStore

        source = PostgresCollaborationStore(value["dsn"])
        sessions = PostgresSessionStore(value["dsn"])
    try:
        initial = CollaborationInitialization.model_validate(value["initial"])
        admission = TemporaryServiceAdmission.model_validate(value["admission"])
        redactor = SecretRedactor()
        async with source._transaction(initial.binding.application_scope, write=False) as tx:
            pending = await discover_services_in_transaction(
                source, tx, initial, after=None, limit=1, redactor=redactor
            )
        assert len(pending) == 1
        assert pending[0].dispatch == admission.dispatch
        assert pending[0].permit.expected == admission.permit
        returned = await sessions._reconcile_temporary_continuation_service(admission)
        assert returned.state == "returned"
        assert returned.released_session_status == "interrupted"
        authority = TemporaryServicePermitAuthority(source, initial, redactor=redactor)
        reader = TemporaryServiceSettlementReader(sessions, admission, redactor=redactor)
        await authority.settle(admission, reader=reader)
        await authority.settle(admission, reader=reader)
        async with source._transaction(initial.binding.application_scope, write=False) as tx:
            assert not await discover_services_in_transaction(
                source, tx, initial, after=None, limit=1, redactor=redactor
            )
    finally:
        await source.close()
        await sessions.close()


if __name__ == "__main__":
    asyncio.run(recover(json.load(sys.stdin)))
