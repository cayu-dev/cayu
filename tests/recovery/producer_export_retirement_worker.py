"""Reconstruct exact excluded-delivery cleanup with no disclosure or providers."""

import asyncio
import json
import sys
from pathlib import Path

from tests.core.test_collaboration_request_foundation import RequestResolver
from tests.core.test_participant_identity import CONTEXT, app, registration
from tests.core.test_producer_output_contracts import output_exports

from cayu import ProducerOutputRegistration
from cayu.collaboration.request_access import PreparedAdmissionRegistration, RequestRegistration
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


async def main():
    material = json.loads(sys.stdin.read())
    command = ProducerOutputRegistration.model_validate(material["expected"])
    if material["backend"] == "sqlite":
        collaboration = SQLiteCollaborationStore(material["address"])
        sessions = SQLiteSessionStore(Path(material["address"]).with_name("sessions.sqlite"))
    else:
        collaboration = PostgresCollaborationStore(material["address"])
        sessions = PostgresSessionStore(material["address"])
    reg = registration(
        scope=command.operation.application_scope,
        limits=command.admission.expected.intent.limits,
    )
    resolver = RequestResolver(command.admission.expected.intent.request)
    resolver.denied = True
    if material.get("failure_resolution") is not None:
        from tests.core.test_prepared_admission_public import PreparationResolver

        from cayu.collaboration.mandates import MandateResolution

        resolver = PreparationResolver(
            command.admission.expected.intent.request, command.admission.prepared.recipient
        )
        resolver.recipient.resolution = MandateResolution.model_validate(
            material["failure_resolution"]
        )
    first = app(collaboration, reg, session_store=sessions)
    initialized = await first.initialize_collaboration()
    exports = output_exports(
        initialized,
        command.admission.expected.intent.request,
        resolver,
        collaboration_store=collaboration,
        session_store=sessions,
    )
    exports.policy.denied.update(
        {"source", "readback", "export", "expose", "append", "retire", "release"}
    )
    application = app(
        collaboration,
        reg,
        session_store=sessions,
        session_exports=exports,
        collaboration_requests=RequestRegistration(
            mandates=resolver,
            prepared_admission=PreparedAdmissionRegistration(receiver=command.receiver),
            max_ttl_ms=300_000,
        ),
    )
    try:
        await application.initialize_collaboration()
        assert not application._provider_registry.registrations
        application._request_coordinator._owners.observation_timeout = 60
        if material.get("failure_resolution") is not None:
            from cayu.collaboration.exports import SessionExportAccessContext

            context = SessionExportAccessContext(
                principal=resolver.recipient.context.principal, mandate=resolver.recipient.context
            )
            from cayu.collaboration._contracts import ExactUnavailable
            from cayu.collaboration._producer_export_store import read_export

            async with collaboration._transaction(
                initialized.owner.application_scope, write=False
            ) as tx:
                retained = await read_export(
                    tx, command, command.destinations[0], redactor=application._secret_redactor
                )
            assert retained is not None
            from cayu.collaboration.exports import SessionExportDenied

            try:
                await application.lookup_session_export(retained.request, context=context)
            except SessionExportDenied:
                pass
            else:
                raise AssertionError("Current readback denial must remain an exception")
            exports.policy.denied.remove("readback")
            assert isinstance(
                await application.lookup_session_export(retained.request, context=context),
                ExactUnavailable,
            )
            result = await application.publish_producer_outcome(command, context=context)
            assert result.command.outcome == "failed"
            assert await application.publish_producer_outcome(command, context=context) == result
            assert exports.projectors[0].calls == 0
            print(result.model_dump_json())
            return
        result = await application.retire_producer_export(
            command, command.destinations[0].operation, context=CONTEXT
        )
        assert (
            await application.retire_producer_export(
                command, command.destinations[0].operation, context=CONTEXT
            )
            == result
        )
        assert exports.projectors[0].calls == 0
        print(result.model_dump_json())
    finally:
        await application.drain_collaboration_requests()
        await collaboration.close()
        await sessions.close()


if __name__ == "__main__":
    asyncio.run(main())
