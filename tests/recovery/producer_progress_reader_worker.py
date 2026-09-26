"""Replay a durable progress occurrence in a fresh process without native work."""

import asyncio
import json
import sys

from tests.core.test_participant_identity import app, registration
from tests.core.test_prepared_admission_public import PreparationResolver

from cayu import ProducerOutputRegistration, ProducerProgressOccurrence
from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration.request_access import PreparedAdmissionRegistration, RequestRegistration
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode


async def main():
    material = json.loads(sys.stdin.read())
    expected = ProducerOutputRegistration.model_validate(material["expected"])
    occurrence = ProducerProgressOccurrence.model_validate(material["occurrence"])
    store = (
        SQLiteCollaborationStore(material["address"])
        if material["backend"] == "sqlite"
        else PostgresCollaborationStore(material["address"], schema_mode=SchemaMode.CREATE)
    )
    resolver = PreparationResolver(
        expected.admission.expected.intent.request, expected.admission.prepared.recipient
    )
    application = app(
        store,
        registration(
            scope=expected.operation.application_scope,
            limits=expected.admission.expected.intent.limits,
        ),
        collaboration_requests=RequestRegistration(
            mandates=resolver,
            prepared_admission=PreparedAdmissionRegistration(receiver=expected.receiver),
            max_ttl_ms=300_000,
        ),
    )
    try:
        await application.initialize_collaboration()
        application._request_coordinator._owners.observation_timeout = 60
        assert not application._providers
        assert (
            await application.session_store.load(expected.admission.prepared.target.session_id)
            is None
        )
        first = await application.record_producer_progress(
            expected, occurrence, context=resolver.recipient.context
        )
        assert (
            await application.record_producer_progress(
                expected, occurrence, context=resolver.recipient.context
            )
            == first
        )
        try:
            await application.record_producer_progress(
                expected,
                occurrence.model_copy(update={"kind": "started"}),
                context=resolver.recipient.context,
            )
        except CollaborationConflict:
            pass
        else:
            raise AssertionError("Changed progress identity must conflict after restart")
        print(first.model_dump_json())
    finally:
        await application.drain_collaboration_requests()
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
