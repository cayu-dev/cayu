"""Repair producer completion from native evidence without model or budget dispatch."""

import asyncio
import json
import sys
from pathlib import Path

from tests.core.test_collaboration_request_foundation import RequestResolver
from tests.core.test_participant_identity import CONTEXT, app, registration

from cayu import ProducerOutputRegistration
from cayu.collaboration._contracts import ExactMatch, ExactNotFound
from cayu.collaboration.request_access import PreparedAdmissionRegistration, RequestRegistration
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


async def main():
    material = json.loads(sys.stdin.read())
    command = ProducerOutputRegistration.model_validate(material["expected"])
    if material["backend"] == "sqlite":
        collaboration = SQLiteCollaborationStore(material["address"])
        sessions = SQLiteSessionStore(Path(material["address"]).with_name("sessions.sqlite"))
    else:
        collaboration = PostgresCollaborationStore(
            material["address"], schema_mode=SchemaMode.CREATE
        )
        sessions = PostgresSessionStore(material["address"], schema_mode=SchemaMode.CREATE)
    application = app(
        collaboration,
        registration(
            scope=command.operation.application_scope,
            limits=command.admission.expected.intent.limits,
        ),
        session_store=sessions,
        collaboration_requests=RequestRegistration(
            mandates=RequestResolver(command.admission.expected.intent.request),
            prepared_admission=PreparedAdmissionRegistration(receiver=command.receiver),
            max_ttl_ms=300_000,
        ),
    )
    try:
        await application.initialize_collaboration()
        assert not application._providers
        retained = await application.lookup_producer_registration(command, context=CONTEXT)
        assert isinstance(retained, ExactMatch) and retained.receipt == command
        if material.get("successor", False):
            completion = await application.retain_producer_completion(command, context=CONTEXT)
            release = await sessions._read_native_producer_release(command)
            final = await application.settle_producer_output(command, context=CONTEXT)
            print(
                json.dumps(
                    {
                        "completion": completion.model_dump(mode="json"),
                        "release": release.model_dump(mode="json"),
                        "cleanup": final.model_dump(mode="json"),
                    }
                )
            )
        elif material.get("cleanup", False):
            application._request_coordinator._owners.observation_timeout = 60
            if not material.get("excluded", False):
                assert await sessions.load(command.admission.prepared.target.session_id) is None
            final = await application.settle_producer_output(command, context=CONTEXT)
            assert await application.settle_producer_output(command, context=CONTEXT) == final
            found = await application.lookup_producer_completion(command, context=CONTEXT)
            if material.get("excluded", False):
                assert isinstance(found, ExactNotFound)
                assert final.native_receipt.mode == "exclusion"
            else:
                completion = await application.retain_producer_completion(command, context=CONTEXT)
                assert isinstance(found, ExactMatch) and found.receipt == completion
            print(final.model_dump_json())
        else:
            completion = await application.retain_producer_completion(command, context=CONTEXT)
            found = await application.lookup_producer_completion(command, context=CONTEXT)
            assert isinstance(found, ExactMatch) and found.receipt == completion
            print(completion.model_dump_json())
    finally:
        await application.drain_collaboration_requests()
        await collaboration.close()
        await sessions.close()


if __name__ == "__main__":
    asyncio.run(main())
