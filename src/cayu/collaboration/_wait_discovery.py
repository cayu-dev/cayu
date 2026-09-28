"""Authenticated indexed wait discovery and exact restart reconstruction."""

from pydantic import Field, StrictInt, StrictStr

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.collaboration._contracts import (
    MAX_DEPTH,
    MAX_ENVELOPE_BYTES,
    MAX_NODES,
    ContractValue,
    ExactConflict,
    ExactMatch,
    ExactUnavailable,
    Generation,
    OperationRef,
    snapshot_input,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.waits import (
    CollaborationWait,
    WaitDelivery,
    WaitSnapshot,
    WaitState,
    wait_identity_digest,
)


class WaitDiscoveryCursor(ContractValue):
    operation: OperationRef


class WaitRecovery(ContractValue):
    operation: OperationRef
    registration_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class DiscoveredWait(ContractValue):
    recovery: WaitRecovery
    # These are historical scan observations, never permission to act.
    revision: Generation
    state: WaitState
    delivery: WaitDelivery


class WaitDiscoveryPage(ContractValue):
    items: tuple[DiscoveredWait, ...] = Field(max_length=32)
    next_cursor: WaitDiscoveryCursor | None


class _Query(ContractValue):
    cursor: WaitDiscoveryCursor | None = None
    limit: StrictInt = Field(ge=1, le=32)


def wait_projection(raw, *, scope, key):
    """Verify source identity before writing or trusting its relational projection."""
    from cayu.vaults.redaction import SecretRedactor

    record = prepare_contract(WaitSnapshot, raw, redactor=SecretRedactor())
    operation = record.registration.wait.operation
    if operation.application_scope != scope or key != (
        operation.namespace_incarnation,
        operation.generation,
        operation.caller_key,
    ):
        raise ValueError("Wait discovery projection contradicts its source identity.")
    return record, (record.state, record.delivery)


def _require_store(store):
    from cayu.collaboration.memory import InMemoryCollaborationStore
    from cayu.storage.collaboration_postgres import PostgresCollaborationStore
    from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

    if type(store) not in (
        InMemoryCollaborationStore,
        SQLiteCollaborationStore,
        PostgresCollaborationStore,
    ) or "_transaction" in vars(store):
        raise CollaborationUnavailable("Wait discovery requires a qualified native owner.")


async def discover_waits(coordinator, *, context, cursor=None, limit=32):
    redactor = coordinator._redactor
    query = prepare_contract(_Query, {"cursor": cursor, "limit": limit}, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    participants = coordinator._participants
    store, initialized = participants._ready()
    _require_store(store)
    participants._capability(store, initialized, mutation=False, family=REQUEST_FAMILY)
    _, grant = participants._authorize(context, "request_readback")
    participants._require_refs(grant, (), create=True)
    after = None if query.cursor is None else query.cursor.operation
    if after is not None and (
        after.application_scope != initialized.owner.application_scope
        or after.namespace_incarnation != initialized.namespace_incarnation
    ):
        raise ValueError("Wait discovery cursor belongs to another owner.")
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        await store._anchor(tx, initialized, redactor)
        rows = await tx.scan_waits(
            namespace=initialized.namespace_incarnation,
            after=None if after is None else (after.generation, after.caller_key),
            limit=query.limit,
        )
        if len(rows) > query.limit:
            raise CollaborationUnavailable("Wait discovery exceeded its source page bound.")
        items = []
        previous = None if after is None else (after.generation, after.caller_key)
        last = None
        truncated = False
        for raw in rows:
            snapshot = prepare_contract(WaitSnapshot, raw, redactor=redactor)
            wait = snapshot.registration.wait
            position = (wait.operation.generation, wait.operation.caller_key)
            if (
                wait.source_owner != initialized.owner
                or wait.operation.namespace_incarnation != initialized.namespace_incarnation
                or (previous is not None and position <= previous)
            ):
                raise CollaborationUnavailable("Wait discovery source ordering conflicts.")
            candidate_cursor = WaitDiscoveryCursor(operation=wait.operation)
            item = DiscoveredWait(
                recovery=WaitRecovery(
                    operation=wait.operation,
                    registration_digest=wait_identity_digest(wait),
                ),
                revision=snapshot.revision,
                state=snapshot.state,
                delivery=snapshot.delivery,
            )
            try:
                canonical_bounded_durable_json_bytes(
                    {
                        "items": [snapshot_input(entry) for entry in (*items, item)],
                        "next_cursor": snapshot_input(candidate_cursor),
                    },
                    "wait discovery page",
                    max_bytes=MAX_ENVELOPE_BYTES,
                    max_nodes=MAX_NODES,
                    max_nesting=MAX_DEPTH,
                )
            except ValueError:
                if not items:
                    raise CollaborationUnavailable(
                        "Wait discovery item exceeds page capacity."
                    ) from None
                truncated = True
                break
            items.append(item)
            previous = position
            last = candidate_cursor
    return WaitDiscoveryPage(
        items=tuple(items), next_cursor=last if truncated or len(rows) == query.limit else None
    )


async def resolve_wait(coordinator, expected, *, context, wait_for_settlement=False):
    """Resolve full original intent only after current per-source read authorization."""
    redactor = coordinator._redactor
    expected = prepare_contract(WaitRecovery, expected, redactor=redactor)
    context = prepare_contract(MandateAccessContext, context, redactor=redactor)
    participants = coordinator._participants
    store, initialized = participants._ready()
    _require_store(store)
    participants._capability(store, initialized, mutation=False, family=REQUEST_FAMILY)
    _, grant = participants._authorize(
        CollaborationAccessContext(principal=context.principal), "request_readback"
    )
    participants._require_refs(grant, (), create=True)
    operation = expected.operation
    if (
        operation.application_scope != initialized.owner.application_scope
        or operation.namespace_incarnation != initialized.namespace_incarnation
    ):
        return ExactConflict()
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        anchor = await store._anchor(tx, initialized, redactor)
        raw = await tx.get(
            "operations",
            (operation.namespace_incarnation, operation.generation, operation.caller_key),
        )
        if raw is None:
            return await store._missing_operation(tx, anchor, operation.generation, redactor)
    try:
        snapshot = prepare_contract(WaitSnapshot, raw, redactor=redactor)
    except ValueError:
        return ExactUnavailable()
    wait = snapshot.registration.wait
    if wait.source_owner != initialized.owner:
        return ExactUnavailable()
    # Foreign authorization remains outside the source transaction. Retained
    # registration is immutable; subsequent observe/deliver rechecks its state.
    for target in wait.targets:
        await coordinator._requests._authorize_retained_source(
            target, context=context, wait_for_settlement=wait_for_settlement
        )
    if wait.operation != operation or wait_identity_digest(wait) != expected.registration_digest:
        return ExactConflict()
    return ExactMatch[CollaborationWait](receipt=wait)
