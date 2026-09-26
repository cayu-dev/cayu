"""Reopen the real source/native stores and prove retirement survives process loss."""

import asyncio
import json
import sys
from pathlib import Path

from tests.core.test_collaboration_request_foundation import RequestResolver
from tests.core.test_participant_identity import CONTEXT, app, registration

from cayu.collaboration._producer_contracts import ProducerOutputRecord
from cayu.collaboration.lifecycle import NamespaceRef
from cayu.collaboration.request_access import PreparedAdmissionRegistration, RequestRegistration
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


async def main():
    material = json.loads(sys.stdin.read())
    record = ProducerOutputRecord.model_validate(material["record"])
    namespace = NamespaceRef.model_validate(material["namespace"])
    command = record.command
    if material["backend"] == "sqlite":
        source = SQLiteCollaborationStore(material["address"])
        native = SQLiteSessionStore(Path(material["address"]).with_name("sessions.sqlite"))
    else:
        source = PostgresCollaborationStore(material["address"])
        native = PostgresSessionStore(material["address"])
    application = app(
        source,
        registration(
            scope=command.operation.application_scope,
            limits=command.admission.expected.intent.limits,
        ),
        session_store=native,
        collaboration_requests=RequestRegistration(
            mandates=RequestResolver(command.admission.expected.intent.request),
            prepared_admission=PreparedAdmissionRegistration(receiver=command.receiver),
            max_ttl_ms=300_000,
        ),
    )
    try:
        await application.initialize_collaboration()
        application._request_coordinator._owners.observation_timeout = 60
        assert not application._providers
        # A read of the old receipt must distinguish retired history from a
        # missing acknowledgement that could justify another cleanup dispatch.
        try:
            await native._read_completed_native_producer_cleanup(record)
        except ValueError as error:
            assert "retired" in str(error)
        else:
            raise AssertionError("Native retirement was lost across process restart")
        result = await application.reclaim_producer_cleanup(namespace, context=CONTEXT, limit=1)
        assert result.removed == 0 and not result.remaining
        print(result.model_dump_json())
    finally:
        await application.drain_collaboration_requests()
        await source.close()
        await native.close()


if __name__ == "__main__":
    asyncio.run(main())
