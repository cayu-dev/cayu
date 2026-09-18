"""Real export, committed answer loss, and exact replay in a fresh process."""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from tests.core.test_collaboration_request_exports import (
    _admit,
    _integration,
    _Policy,
    _PublicExportReader,
)
from tests.core.test_collaboration_request_foundation import RequestResolver
from tests.core.test_participant_identity import CONTEXT, app, registration

from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration._session_export_participant import SessionExportRequestReceivingOwner
from cayu.collaboration.exports import (
    ExportLimits,
    SessionExportAccessContext,
    SessionExportRegistration,
)
from cayu.collaboration.request_access import RequestRegistration
from cayu.collaboration.requests import RequestOutcomeCommand
from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode
from cayu.vaults.redaction import SecretRedactor


async def main():
    backend, mode, filename = sys.argv[1:]
    path = Path(filename)
    address = os.environ["CAYU_REQUEST_TEST_STORE"]
    store = (
        SQLiteCollaborationStore(address)
        if backend == "sqlite"
        else PostgresCollaborationStore(address, schema_mode=SchemaMode.CREATE)
    )
    redactor = SecretRedactor()
    if mode == "publish":
        case = await _integration(store, path.parent, address if backend == "postgres" else None)
        accepted, initiator, _ = await _admit(case)
        application, context = case.app, case.context
        initialized = case.values[1]
        outcome = RequestOutcomeCommand(
            operation=initialized.operation("answer"),
            expected=accepted.expected,
            expected_revision=2,
            outcome="answered",
            commitment=case.source.expected.intent.output_commitment,
            source_receipt=case.source,
            initiator=initiator,
        )
        path.write_text(json.dumps({"outcome": outcome.model_dump(mode="json")}))
        original = store._transaction

        @asynccontextmanager
        async def transaction(scope, *, write):
            answer_published = False
            async with original(scope, write=write) as tx:
                put = tx.put

                async def capture(family, key, value, *, insert):
                    nonlocal answer_published
                    await put(family, key, value, insert=insert)
                    if family == "request_events" and value.type == "request_answered":
                        answer_published = True

                tx.put = capture
                yield tx
            if answer_published:
                os._exit(19)

        store._transaction = transaction
        await application.publish_collaboration_outcome(outcome, context=context)
        raise AssertionError("Answer acknowledgement must be lost")

    saved = json.loads(path.read_text())
    outcome = prepare_contract(RequestOutcomeCommand, saved["outcome"], redactor=redactor)
    expected = outcome.expected
    scope = expected.operation.application_scope
    resolver = RequestResolver(expected.intent.request)
    owner = expected.destination
    export_context = SessionExportAccessContext(principal="operator")
    reader = _PublicExportReader(owner, export_context)
    # Committed answer replay needs neither settlement again nor a projector.
    reader.allow_settlement = False
    session_store = (
        SQLiteSessionStore(path.parent / "request-exports.sqlite")
        if backend == "sqlite"
        else PostgresSessionStore(address, schema_mode=SchemaMode.CREATE)
    )
    application = app(
        store,
        registration(scope=scope),
        session_store=session_store,
        collaboration_requests=RequestRegistration(
            mandates=resolver,
            max_ttl_ms=300_000,
            receiving_owner=SessionExportRequestReceivingOwner(audience=owner, reader=reader),
        ),
        session_exports=SessionExportRegistration(
            owner=owner,
            policy=_Policy(owner, outcome.source_receipt.expected.intent.request.policy),
            projectors=(),
            limits=ExportLimits(max_exports=16, max_pending=8, max_retained_bytes=1024 * 1024),
            readers=(reader,),
        ),
    )
    reader.app = application
    initialized = await application.initialize_collaboration()
    context = resolver.context.model_copy(
        update={"participant": expected.intent.selection.recipient.reference}
    )
    try:
        before = await application.inspect_collaboration_request(expected, context=resolver.context)
        assert before.state == "answered" and before.outcome.command == outcome
        assert (await reader.lookup(outcome.source_receipt)).status == "match"
        for _ in range(2):
            assert (
                await application.publish_collaboration_outcome(outcome, context=context)
                == before.outcome
            )
        assert (
            await application.inspect_collaboration_request(expected, context=resolver.context)
            == before
        )
        participant = await application.inspect_participant(
            expected.intent.selection.recipient.reference, context=CONTEXT
        )
        assert participant.issued_permit_frontier == 1
        assert participant.outstanding_obligations == 0
        async with store._transaction(scope, write=False) as tx:
            events = await tx.scan_request_events(after=0, limit=64)
            assert sum(row["type"] == "request_answered" for row in events) == 1
            anchor = await store._anchor(tx, initialized, redactor)
            assert anchor.reserved_operations == 0 and anchor.reserved_events == 0
        print("answer-replay-ok", flush=True)
    finally:
        await application.drain_collaboration_requests()
        await application.drain_session_exports()
        await session_store.close()
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
