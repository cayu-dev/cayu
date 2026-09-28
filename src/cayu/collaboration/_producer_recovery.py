"""Bounded producer discovery through existing durable participant responsibilities."""

from hashlib import sha256
from typing import Annotated

from pydantic import Field, StrictInt

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.collaboration._contracts import (
    MAX_DEPTH,
    MAX_ENVELOPE_BYTES,
    MAX_NODES,
    CollaborationConflict,
    ContractValue,
    OperationRef,
    snapshot_input,
)
from cayu.collaboration._permits import PermitSnapshot
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import (
    MAX_OUTPUT_DESTINATIONS,
    ProducerOutputRecord,
)
from cayu.collaboration._producer_store import output_permit, read_output_registration
from cayu.collaboration._request_coordinator import _safe_request_failure
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.participants import CollaborationUnavailable, Counter, ParticipantRef
from cayu.collaboration.prepared_admission import NativeCommitment


class ProducerOutputRecovery(ContractValue):
    registration: OperationRef
    registration_commitment: NativeCommitment


class ProducerPendingOutput(ContractValue):
    recovery: ProducerOutputRecovery
    position: Counter
    destinations: tuple[OperationRef, ...] = Field(max_length=MAX_OUTPUT_DESTINATIONS)


class ProducerPendingPage(ContractValue):
    """Cursor advances scanned permits, including unrelated responsibilities."""

    items: tuple[ProducerPendingOutput, ...] = Field(max_length=32)
    next_cursor: Counter | None = None


class _Query(ContractValue):
    participant: ParticipantRef
    after: Counter = 0
    limit: Annotated[StrictInt, Field(ge=1, le=32)] = 32


async def pending_producer_outputs(
    app, participant, *, context, after=0, limit=32, wait_for_settlement=False
):
    """Find live responsibility even after answer election, without content access.

    Uses the existing (scope, participant, state, position) permit index. Each
    call inspects at most limit rows; an empty page with a cursor is not EOF.
    No request-open filter, private retry handle, or full operation scan is used.
    """
    requests = app._request_coordinator
    participants = app._participant_coordinator
    redactor = app._secret_redactor
    query = prepare_contract(
        _Query, {"participant": participant, "after": after, "limit": limit}, redactor=redactor
    )
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    store, initialized = participants._ready()

    async def discover():
        participants._capability(store, initialized, mutation=False, family=REQUEST_FAMILY)
        _, grant = participants._authorize(context, "request_readback")
        participants._require_refs(grant, (query.participant,))
        if query.participant.owner != initialized.owner:
            raise CollaborationConflict("Producer discovery belongs to another owner.")
        items = []
        scanned = query.after
        truncated = False
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            await store._participant(tx, query.participant, initialized.owner, redactor)
            rows = await tx.scan_permits(
                query.participant.participant_id,
                after=query.after,
                limit=query.limit,
                pending_only=True,
            )
            if len(rows) > query.limit:
                raise CollaborationUnavailable("Producer responsibility scan exceeded its bound.")
            for raw in rows:
                permit = prepare_contract(PermitSnapshot, raw, redactor=redactor)
                intent = permit.expected.intent.request
                if (
                    permit.state != "pending"
                    or intent.participant != query.participant
                    or permit.position <= scanned
                ):
                    raise CollaborationConflict(
                        "Producer responsibility scan contradicts its index."
                    )
                if intent.effect_scope != "producer_output":
                    if intent.target.kind == "producer_output":
                        raise CollaborationConflict(
                            "Producer responsibility classification conflicts."
                        )
                    scanned = permit.position
                    continue
                stored = await tx.get("operations", operation_key(intent.source_operation))
                if stored is None:
                    raise CollaborationUnavailable(
                        "Producer responsibility lacks its retained registration."
                    )
                candidate = prepare_contract(ProducerOutputRecord, stored, redactor=redactor)
                require_exact_contract(
                    output_permit(candidate.command, redactor), permit.expected, redactor=redactor
                )
                retained = await read_output_registration(tx, candidate.command, redactor=redactor)
                if retained != candidate or candidate.cleanup_ack is not None:
                    raise CollaborationUnavailable(
                        "Pending producer responsibility contradicts its owner."
                    )
                item = ProducerPendingOutput(
                    recovery=ProducerOutputRecovery(
                        registration=candidate.command.operation,
                        registration_commitment="sha256:"
                        + sha256(contract_bytes(candidate.command, redactor=redactor)).hexdigest(),
                    ),
                    position=permit.position,
                    destinations=tuple(
                        destination.operation for destination in candidate.command.destinations
                    ),
                )
                # Only aggregate envelope capacity may shorten an otherwise
                # validated page; do not hide malformed durable rows as overflow.
                proposed = {
                    "items": [snapshot_input(entry) for entry in (*items, item)],
                    "next_cursor": permit.position,
                }
                try:
                    canonical_bounded_durable_json_bytes(
                        proposed,
                        "producer recovery page",
                        max_bytes=MAX_ENVELOPE_BYTES,
                        max_nodes=MAX_NODES,
                        max_nesting=MAX_DEPTH,
                    )
                except ValueError:
                    if not items:
                        raise CollaborationUnavailable(
                            "Producer recovery item exceeds page capacity."
                        ) from None
                    truncated = True
                    break
                items.append(item)
                scanned = permit.position
        return prepare_contract(
            ProducerPendingPage,
            {
                "items": tuple(items),
                "next_cursor": scanned if truncated or len(rows) == query.limit else None,
            },
            redactor=redactor,
        )

    async def owned():
        return await requests._dependency(discover)

    return await requests._observe(
        requests._owners.run(
            owned,
            key=("producer_discovery", object()),
            expectation=contract_bytes(query, redactor=redactor)
            + contract_bytes(context, redactor=redactor),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
            wait_for_settlement=wait_for_settlement,
        )
    )
