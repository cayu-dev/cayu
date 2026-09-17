"""Namespace transitions inside the collaboration owner's native transaction.

No application callback, policy lookup, or receiving-owner I/O runs here.
"""

from __future__ import annotations

from uuid import uuid4

from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._contracts import CollaborationConflict, OperationRef
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration.base import CollaborationStore, _Anchor, _key, _Repository, _stored_mode
from cayu.collaboration.lifecycle import (
    CollaborationNamespaceRetired,
    LifecycleCommand,
    LifecycleReceipt,
    NamespaceInspection,
    NamespacePrune,
    NamespaceRef,
    NamespaceRetire,
    NamespaceRetirementEvidence,
    NamespaceRotate,
    NamespaceSeal,
    NamespaceSnapshot,
    ParticipantLifecycleChange,
)
from cayu.collaboration.participants import (
    CollaborationInitialization,
    CollaborationUnavailable,
    ParticipantEvent,
    ParticipantReceipt,
)
from cayu.vaults.redaction import SecretRedactor


def prepare_lifecycle(
    initialized: CollaborationInitialization,
    expected: LifecycleCommand,
    redactor: SecretRedactor,
) -> LifecycleCommand:
    expected = prepare_contract(LifecycleCommand, expected, redactor=redactor)
    if (
        expected.source != initialized.owner
        or expected.operation.application_scope != initialized.binding.application_scope
        or expected.operation.namespace_incarnation != initialized.namespace_incarnation
        or expected.intent.limits != initialized.binding.limits
    ):
        raise CollaborationConflict("Lifecycle command conflicts with initialized authority.")
    return expected


async def load_namespace(
    tx: _Repository, anchor: _Anchor, generation: int, redactor: SecretRedactor
) -> NamespaceSnapshot:
    initial = anchor.initialization
    if generation > anchor.current_generation:
        raise CollaborationConflict("Namespace generation has not been elected.")
    if generation <= anchor.pruned_through:
        raise CollaborationNamespaceRetired("Namespace content has been pruned.")
    raw = await tx.get("namespaces", (initial.namespace_incarnation, generation))
    if raw is None:
        raise CollaborationUnavailable("Namespace authority is unavailable.")
    result = prepare_contract(NamespaceSnapshot, raw, redactor=redactor)
    if result.reference != NamespaceRef(
        owner=initial.owner,
        namespace_incarnation=initial.namespace_incarnation,
        generation=generation,
    ) or (result.state == "retired") != (generation <= anchor.retired_through):
        raise CollaborationUnavailable("Namespace index or retirement authority conflicts.")
    return result


async def require_open_namespace(
    tx: _Repository, anchor: _Anchor, operation: OperationRef, redactor: SecretRedactor
) -> NamespaceSnapshot:
    if operation.generation <= anchor.retired_through:
        raise CollaborationNamespaceRetired("Namespace no longer admits operations.")
    namespace = await load_namespace(tx, anchor, operation.generation, redactor)
    if namespace.state != "open" or operation.generation != anchor.current_generation:
        raise CollaborationConflict("Namespace is sealed against new admission.")
    return namespace


async def require_maintenance_namespace(
    tx: _Repository, anchor: _Anchor, operation: OperationRef, redactor: SecretRedactor
) -> NamespaceSnapshot:
    """Keep reclamation possible after sealing, without admitting business work."""
    namespace = await load_namespace(tx, anchor, operation.generation, redactor)
    if operation.generation != anchor.current_generation or namespace.state not in (
        "open",
        "sealed",
    ):
        raise CollaborationConflict("Maintenance requires the current control namespace.")
    return namespace


async def inspect_namespace(
    store: CollaborationStore, initialized: CollaborationInitialization, redactor: SecretRedactor
) -> NamespaceInspection:
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        anchor = await store._anchor(tx, initialized, redactor)
        return NamespaceInspection(
            current=await load_namespace(tx, anchor, anchor.current_generation, redactor),
            retired_through=anchor.retired_through,
            pruned_through=anchor.pruned_through,
            retention_revision=anchor.retention_revision,
            retained_generations=anchor.retained_generations,
        )


async def inspect_retirement(
    store: CollaborationStore,
    initialized: CollaborationInitialization,
    namespace: NamespaceRef,
    redactor: SecretRedactor,
) -> NamespaceRetirementEvidence | None:
    if (
        namespace.owner != initialized.owner
        or namespace.namespace_incarnation != initialized.namespace_incarnation
    ):
        raise CollaborationConflict("Retirement query belongs to another namespace owner.")
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        anchor = await store._anchor(tx, initialized, redactor)
        if namespace.generation <= anchor.pruned_through:
            content = "pruned"
        else:
            current = await load_namespace(tx, anchor, namespace.generation, redactor)
            if current.state != "retired":
                return None
            content = current.content
        return NamespaceRetirementEvidence(
            namespace=namespace,
            retired_through=anchor.retired_through,
            pruned_through=anchor.pruned_through,
            content=content,
        )


async def lifecycle_replay(
    store: CollaborationStore, tx: _Repository, expected: LifecycleCommand, redactor: SecretRedactor
) -> LifecycleReceipt | None:
    raw = await tx.get("operations", _key(expected))
    if raw is None:
        return None
    if _stored_mode(raw) == "permit":
        from cayu.collaboration._permit_store import prepare_permit_record

        prepare_permit_record(raw, redactor)
        raise CollaborationConflict("Operation key already carries permit responsibility.")
    if _stored_mode(raw) == "identity":
        prepare_contract(ParticipantReceipt, raw, redactor=redactor)
        raise CollaborationConflict("Operation key already has different intent.")
    receipt = prepare_contract(LifecycleReceipt, raw, redactor=redactor)
    require_exact_contract(expected, receipt.expected, redactor=redactor)
    event = await tx.get("events", (receipt.event.sequence,))
    if event is None:
        raise CollaborationUnavailable("Lifecycle receipt event is unavailable.")
    require_exact_contract(
        receipt.event,
        prepare_contract(ParticipantEvent, event, redactor=redactor),
        redactor=redactor,
    )
    if receipt.participant is not None:
        await store._require_snapshot_history(tx, receipt.participant, redactor)
    return receipt


async def apply_lifecycle(
    store: CollaborationStore,
    initialized: CollaborationInitialization,
    expected: LifecycleCommand,
    redactor: SecretRedactor,
) -> LifecycleReceipt:
    async with store._transaction(initialized.binding.application_scope, write=True) as tx:
        anchor = await store._anchor(tx, initialized, redactor)
        replay = await lifecycle_replay(store, tx, expected, redactor)
        if replay is not None:
            return replay
        request = expected.intent.request
        if isinstance(request, NamespacePrune):
            from cayu.collaboration._retention_store import prune_namespace

            return await prune_namespace(store, tx, anchor, expected, redactor)
        if isinstance(request, ParticipantLifecycleChange):
            from cayu.collaboration._participant_lifecycle_store import elect_participant_lifecycle

            return await elect_participant_lifecycle(store, tx, anchor, expected, redactor)
        if not isinstance(request, (NamespaceSeal, NamespaceRotate, NamespaceRetire)):
            raise CollaborationUnavailable("This lifecycle transition is not implemented yet.")
        if isinstance(request, NamespaceRetire):
            await require_maintenance_namespace(tx, anchor, expected.operation, redactor)
        elif request.namespace.generation != anchor.current_generation:
            raise CollaborationConflict("Only the current namespace can elect a transition.")
        namespace = await load_namespace(tx, anchor, request.namespace.generation, redactor)
        if (
            namespace.reference != request.namespace
            or namespace.revision != request.expected_revision
        ):
            raise CollaborationConflict("Namespace revision changed.")
        successor: NamespaceSnapshot | None = None
        retired_through = anchor.retired_through
        if isinstance(request, NamespaceRetire):
            if (
                request.expected_retired_through != anchor.retired_through
                or namespace.state != "sealed"
                or namespace.outstanding_obligations
            ):
                raise CollaborationConflict("Namespace retirement requires contiguous settlement.")
            retired_through = namespace.reference.generation
            state = "retired"
            event_type = "namespace_retired"
        else:
            if namespace.state == "retired" or (
                isinstance(request, NamespaceSeal) and namespace.state != "open"
            ):
                raise CollaborationConflict("Namespace transition is not admissible.")
            state = "sealed"
            event_type = "namespace_sealed"
            if isinstance(request, NamespaceRotate):
                successor = NamespaceSnapshot(
                    reference=NamespaceRef(
                        owner=initialized.owner,
                        namespace_incarnation=initialized.namespace_incarnation,
                        generation=anchor.current_generation + 1,
                    ),
                    revision=1,
                    state="open",
                    outstanding_obligations=0,
                )
                event_type = "namespace_rotated"
        updated_namespace = prepare_contract(
            NamespaceSnapshot,
            namespace.model_copy(
                update={
                    "revision": namespace.revision + 1,
                    "state": state,
                }
            ),
            redactor=redactor,
        )
        event = prepare_contract(
            ParticipantEvent,
            {
                "id": uuid4().hex,
                "sequence": anchor.event_sequence + 1,
                "operation": expected.operation,
                "type": event_type,
                "participants": (),
            },
            redactor=redactor,
        )
        receipt = prepare_contract(
            LifecycleReceipt,
            LifecycleReceipt(
                expected=expected,
                namespace=updated_namespace,
                successor=successor,
                event=event,
            ),
            redactor=redactor,
        )
        charge = (
            len(contract_bytes(receipt, redactor=redactor))
            + len(contract_bytes(event, redactor=redactor))
            + len(contract_bytes(updated_namespace, redactor=redactor))
            - len(contract_bytes(namespace, redactor=redactor))
            + (0 if successor is None else len(contract_bytes(successor, redactor=redactor)))
        )
        updated = prepare_contract(
            _Anchor,
            anchor.model_copy(
                update={
                    "current_generation": anchor.current_generation + int(successor is not None),
                    "retained_generations": anchor.retained_generations
                    + int(successor is not None),
                    "retired_through": retired_through,
                    "operation_count": anchor.operation_count + 1,
                    "event_count": anchor.event_count + 1,
                    "event_sequence": event.sequence,
                    "retained_bytes": anchor.retained_bytes + charge,
                }
            ),
            redactor=redactor,
        )
        require_capacity(updated, ordinary=False)
        await tx.put(
            "namespaces",
            (
                initialized.namespace_incarnation,
                namespace.reference.generation,
            ),
            updated_namespace,
            insert=False,
        )
        if successor is not None:
            await tx.put(
                "namespaces",
                (
                    initialized.namespace_incarnation,
                    successor.reference.generation,
                ),
                successor,
                insert=True,
            )
        await tx.put("operations", _key(expected), receipt, insert=True)
        await tx.put("events", (event.sequence,), event, insert=True)
        await tx.put("anchors", (), updated, insert=False)
        return receipt
