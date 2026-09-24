"""Read admission in a fresh interpreter with no recipient or budget registration."""

import asyncio
import json
import sys

from tests.core.test_collaboration_request_foundation import RequestResolver
from tests.core.test_participant_identity import app, registration

from cayu.collaboration.request_access import RequestRegistration
from cayu.collaboration.requests import RequestAdmissionCommand
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode


async def main():
    material = json.loads(sys.stdin.read())
    expected = RequestAdmissionCommand.model_validate(material["expected"])
    resolver = RequestResolver(expected.expected.intent.request)
    store = (
        SQLiteCollaborationStore(material["address"])
        if material["backend"] == "sqlite"
        else PostgresCollaborationStore(material["address"], schema_mode=SchemaMode.CREATE)
    )
    application = app(
        store,
        registration(
            scope=expected.operation.application_scope, limits=expected.expected.intent.limits
        ),
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
    )
    try:
        await application.initialize_collaboration()
        result = await application.collaboration_admission_reader().lookup(
            expected,
            context=resolver.context,
        )
        print(result.model_dump_json())
    finally:
        await application.drain_collaboration_requests()
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
