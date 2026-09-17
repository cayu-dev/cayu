"""Process-loss fixture: stop after native commit, before its owner acknowledges."""

from __future__ import annotations

import asyncio
import json
import sys

from tests.core.test_collaboration_permits import Receiver, permit
from tests.core.test_participant_identity import CONTEXT, app, create, registration

from cayu.collaboration import _namespace_store, _permit_store
from cayu.collaboration._permits import PermitCommand
from cayu.collaboration.lifecycle import (
    LifecycleCommand,
    NamespacePrune,
    NamespaceRetire,
    NamespaceRotate,
    ParticipantLifecycleChange,
)
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode
from cayu.vaults.redaction import SecretRedactor

REDACTOR = SecretRedactor()


def hold_after_commit(module, name):
    original = getattr(module, name)

    async def committed(*args, **kwargs):
        result = await original(*args, **kwargs)
        print(result.model_dump_json(), flush=True)
        await asyncio.Event().wait()
        return result

    setattr(module, name, committed)


async def execute_lifecycle(application, request):
    method = {
        "participant_lifecycle": application.change_participant_lifecycle,
        "namespace_retire": application.retire_collaboration_namespace,
        "namespace_prune": application.prune_collaboration_namespace,
    }[request.kind]
    return await method(request, context=CONTEXT)


async def main():
    backend, address, scope, phase = sys.argv[1:]
    store = (
        SQLiteCollaborationStore(address)
        if backend == "sqlite"
        else PostgresCollaborationStore(address, schema_mode=SchemaMode.CREATE)
    )
    application = app(store, registration(scope=scope))
    initialized = await application.initialize_collaboration()
    if phase == "replay":
        data = json.loads(sys.stdin.read())
        if data["expected"]["mode"] == "permit":
            expected = PermitCommand.model_validate(data["expected"])
            result = await store._settle_permit(
                initialized,
                expected,
                reader=Receiver(expected),
                redactor=REDACTOR,
            )
        else:
            expected = LifecycleCommand.model_validate(data["expected"])
            result = await execute_lifecycle(application, expected.intent.request)
        events = await application.list_participant_events(context=CONTEXT)
        assert sum(event.id == result.event.id for event in events.events) == 1
        assert (await application.discover_participants(context=CONTEXT)).participants
        print(result.model_dump_json(), flush=True)
        await store.close()
        return

    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    expected = permit(initialized, ref)
    await store._register_permit(initialized, expected, redactor=REDACTOR)
    if phase == "settlement":
        hold_after_commit(_permit_store, "settle_permit")
        await store._settle_permit(
            initialized, expected, reader=Receiver(expected), redactor=REDACTOR
        )
        return
    disable = ParticipantLifecycleChange(
        operation=initialized.operation("disable"),
        participant=ref,
        expected_lifecycle_revision=1,
        state="disabled",
    )
    if phase == "disable":
        hold_after_commit(_namespace_store, "apply_lifecycle")
        await execute_lifecycle(application, disable)
        return
    await execute_lifecycle(application, disable)
    await store._settle_permit(initialized, expected, reader=Receiver(expected), redactor=REDACTOR)
    before = await application.inspect_collaboration_namespace(context=CONTEXT)
    rotated = await application.rotate_collaboration_namespace(
        NamespaceRotate(
            operation=before.current.reference.operation("rotate"),
            namespace=before.current.reference,
            expected_revision=before.current.revision,
        ),
        context=CONTEXT,
    )
    retire = NamespaceRetire(
        operation=rotated.successor.reference.operation("retire"),
        namespace=rotated.namespace.reference,
        expected_revision=rotated.namespace.revision,
        expected_retired_through=0,
    )
    if phase == "retirement":
        hold_after_commit(_namespace_store, "apply_lifecycle")
        await execute_lifecycle(application, retire)
        return
    assert phase == "pruning"
    await execute_lifecycle(application, retire)
    before = await application.inspect_collaboration_namespace(context=CONTEXT)
    hold_after_commit(_namespace_store, "apply_lifecycle")
    await execute_lifecycle(
        application,
        NamespacePrune(
            operation=rotated.successor.reference.operation("prune"),
            namespace=rotated.namespace.reference,
            expected_retention_revision=before.retention_revision,
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
