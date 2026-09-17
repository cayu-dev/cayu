"""Bounded, exact pruning of retired operations and unreferenced history."""

from __future__ import annotations

from typing import cast
from uuid import uuid4

from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._contracts import CollaborationConflict, ExactMatch
from cayu.collaboration._history_references import HistoryKey, history_references
from cayu.collaboration._namespace_store import (
    lifecycle_replay,
    load_namespace,
    require_maintenance_namespace,
)
from cayu.collaboration._permit_store import (
    prepare_permit_record,
    registered_receipt,
    require_event,
)
from cayu.collaboration._permits import (
    PermitExclusion,
    PermitSettlement,
    PermitSnapshot,
    ReservedPermitSettlement,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration.base import CollaborationStore, _Anchor, _key, _Repository, _stored_mode
from cayu.collaboration.lifecycle import LifecycleCommand, LifecycleReceipt, NamespacePrune
from cayu.collaboration.participants import (
    CollaborationCapacityExceeded,
    CollaborationUnavailable,
    ParticipantConfigurationEvidence,
    ParticipantEvent,
    ParticipantLifecycleEvidence,
    ParticipantReceipt,
    ParticipantSnapshot,
)
from cayu.vaults.redaction import SecretRedactor


async def release_unused_history(
    tx: _Repository, refs: tuple[HistoryKey, ...], redactor: SecretRedactor
) -> int:
    released = 0
    for family, participant_id, revision in sorted(set(refs)):
        current_raw = await tx.get("participants", (participant_id,))
        if current_raw is None:
            raise CollaborationUnavailable("Current participant history authority is unavailable.")
        current = prepare_contract(ParticipantSnapshot, current_raw, redactor=redactor)
        live_revision = (
            current.configuration_revision
            if family == "configurations"
            else current.lifecycle_revision
        )
        if live_revision == revision or await tx.history_in_use(family, participant_id, revision):
            continue
        raw = await tx.get(family, (participant_id, revision))
        if raw is None:
            raise CollaborationUnavailable("Referenced participant history is unavailable.")
        schema = (
            ParticipantConfigurationEvidence
            if family == "configurations"
            else ParticipantLifecycleEvidence
        )
        value = prepare_contract(schema, raw, redactor=redactor)
        stored_revision = (
            value.configuration_revision
            if isinstance(value, ParticipantConfigurationEvidence)
            else value.lifecycle_revision
        )
        if value.reference != current.reference or stored_revision != revision:
            raise CollaborationUnavailable("Historical participant incarnation conflicts.")
        released += len(contract_bytes(value, redactor=redactor))
        await tx.delete(family, (participant_id, revision))
    return released


async def prune_namespace(
    store: CollaborationStore,
    tx: _Repository,
    anchor: _Anchor,
    expected: LifecycleCommand,
    redactor: SecretRedactor,
) -> LifecycleReceipt:
    request = expected.intent.request
    assert isinstance(request, NamespacePrune)
    await require_maintenance_namespace(tx, anchor, expected.operation, redactor)
    if (
        request.expected_retention_revision != anchor.retention_revision
        or request.namespace.generation != anchor.pruned_through + 1
        or request.namespace.generation > anchor.retired_through
    ):
        raise CollaborationConflict(
            "Pruning requires the next retired generation and exact retention revision."
        )
    namespace = await load_namespace(tx, anchor, request.namespace.generation, redactor)
    if (
        namespace.reference != request.namespace
        or namespace.state != "retired"
        or namespace.outstanding_obligations
    ):
        raise CollaborationConflict("Namespace does not have settled retirement authority.")
    records = await tx.scan_operations(
        request.namespace.namespace_incarnation,
        request.namespace.generation,
        limit=request.max_records,
    )
    released = removed = events_removed = permits_removed = 0
    processed = set()
    for raw in records:
        if removed >= request.max_records:
            break
        mode = _stored_mode(raw)
        if mode == "identity":
            item = prepare_contract(ParticipantReceipt, raw, redactor=redactor)
            found = await store._receipt(tx, item.expected, redactor)
            if not isinstance(found, ExactMatch) or found.receipt != item:
                raise CollaborationUnavailable("Pruning lacks exact participant evidence.")
            bundle = (item,)
        elif mode == "lifecycle":
            item = prepare_contract(LifecycleReceipt, raw, redactor=redactor)
            if await lifecycle_replay(store, tx, item.expected, redactor) != item:
                raise CollaborationUnavailable("Pruning lacks exact lifecycle evidence.")
            bundle = (item,)
        elif (
            mode == "permit"
            and isinstance(raw, dict)
            and cast("dict[object, object]", raw).get("record_type") == "permit_excluded"
        ):
            item = prepare_contract(PermitExclusion, raw, redactor=redactor)
            await require_event(tx, item.event, redactor)
            bundle = (item,)
        elif mode == "permit":
            item = prepare_permit_record(raw, redactor)
            parent = _key(item.expected)
            if parent in processed:
                continue
            if isinstance(item, ReservedPermitSettlement):
                raise CollaborationUnavailable(
                    "Retired namespace still retains pending responsibility."
                )
            registered = await registered_receipt(tx, item.expected, redactor)
            if registered is None:
                raise CollaborationUnavailable("Pruning lacks exact permit registration.")
            operation = item.expected.intent.request.settlement_operation
            settled = prepare_permit_record(
                await tx.get(
                    "operations",
                    (
                        operation.namespace_incarnation,
                        operation.generation,
                        operation.caller_key,
                    ),
                ),
                redactor,
            )
            if not isinstance(settled, PermitSettlement):
                raise CollaborationUnavailable("Pruning cannot remove an unsettled permit.")
            if removed + 2 > request.max_records:
                if not removed:
                    raise CollaborationCapacityExceeded(
                        "A settled permit requires a pruning batch of at least two records."
                    )
                break
            snapshot = prepare_contract(
                PermitSnapshot, await tx.get("permits", parent), redactor=redactor
            )
            if snapshot.state != "settled":
                raise CollaborationUnavailable("Pruning requires positive permit settlement.")
            released += len(contract_bytes(snapshot, redactor=redactor))
            await tx.delete("permits", parent)
            permits_removed += 1
            processed.add(parent)
            bundle = (registered, settled)
        else:
            raise CollaborationUnavailable("Unknown retained operation cannot be pruned.")
        references: list[HistoryKey] = []
        for record in bundle:
            operation = (
                record.expected.intent.request.settlement_operation
                if isinstance(record, PermitSettlement)
                else record.expected.operation
            )
            await tx.delete(
                "operations",
                (operation.namespace_incarnation, operation.generation, operation.caller_key),
            )
            await tx.delete("events", (record.event.sequence,))
            released += len(contract_bytes(record, redactor=redactor)) + len(
                contract_bytes(record.event, redactor=redactor)
            )
            references.extend(history_references(record))
            removed += 1
            events_removed += 1
        released += await release_unused_history(tx, tuple(references), redactor)
    remaining = await tx.scan_operations(
        request.namespace.namespace_incarnation, request.namespace.generation, limit=1
    )
    complete = not remaining
    if complete:
        await tx.delete(
            "namespaces", (request.namespace.namespace_incarnation, request.namespace.generation)
        )
        released += len(contract_bytes(namespace, redactor=redactor))
    elif namespace.content != "partial":
        partial_namespace = prepare_contract(
            type(namespace), namespace.model_copy(update={"content": "partial"}), redactor=redactor
        )
        released += len(contract_bytes(namespace, redactor=redactor)) - len(
            contract_bytes(partial_namespace, redactor=redactor)
        )
        namespace = partial_namespace
        await tx.put(
            "namespaces",
            (request.namespace.namespace_incarnation, request.namespace.generation),
            namespace,
            insert=False,
        )
    event = ParticipantEvent(
        id=uuid4().hex,
        sequence=anchor.event_sequence + 1,
        operation=expected.operation,
        type="namespace_pruned",
        participants=(),
    )
    receipt = prepare_contract(
        LifecycleReceipt,
        LifecycleReceipt(
            expected=expected,
            namespace=namespace,
            event=event,
            removed_records=removed,
            pruned_through=anchor.pruned_through + int(complete),
            retention_revision=anchor.retention_revision + 1,
            complete=complete,
        ),
        redactor=redactor,
    )
    charge = (
        len(contract_bytes(receipt, redactor=redactor))
        + len(contract_bytes(event, redactor=redactor))
        - released
    )
    updated = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "pruned_through": receipt.pruned_through,
                "retention_revision": receipt.retention_revision,
                "retained_generations": anchor.retained_generations - int(complete),
                "operation_count": anchor.operation_count - removed + 1,
                "event_count": anchor.event_count - events_removed + 1,
                "event_sequence": event.sequence,
                "retained_bytes": anchor.retained_bytes + charge,
                "permit_count": anchor.permit_count - permits_removed,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated, ordinary=False)
    await tx.put("operations", _key(expected), receipt, insert=True)
    await tx.put("events", (event.sequence,), event, insert=True)
    await tx.put("anchors", (), updated, insert=False)
    return receipt
