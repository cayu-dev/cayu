"""Bounded native participant-session inventory for registered runtime servicing."""

from pydantic import Field, StrictInt

from cayu.collaboration._contracts import ContractValue, Identifier
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.participants import ParticipantRef
from cayu.sessions.context_views import ParticipantSessionCreationReceipt
from cayu.vaults.redaction import SecretRedactor


class ParticipantSessionScan(ContractValue):
    participant: ParticipantRef
    after: Identifier | None = None
    limit: StrictInt = Field(ge=1, le=32)


class ParticipantSessionReference(ContractValue):
    participant: ParticipantRef
    creation_key: Identifier
    session_id: Identifier
    session_instance_id: Identifier
    receipt_commitment: Identifier


class ParticipantSessionCursor(ContractValue):
    participant: ParticipantRef
    after: Identifier


async def discover_participant_sessions(app, participant, *, context, cursor=None, limit=32):
    from cayu.collaboration.participants import CollaborationUnavailable
    from cayu.sessions.base import InMemorySessionStore
    from cayu.storage.postgres import PostgresSessionStore
    from cayu.storage.sqlite import SQLiteSessionStore

    redactor = app._secret_redactor
    if participant is None:
        raise ValueError("Session discovery requires an exact participant.")
    participant = prepare_contract(ParticipantRef, participant, redactor=redactor)
    if cursor is not None:
        cursor = prepare_contract(ParticipantSessionCursor, cursor, redactor=redactor)
        if cursor.participant != participant:
            raise ValueError("Session discovery cursor belongs to another participant.")
    query = prepare_scan(participant, None if cursor is None else cursor.after, limit)
    store = app.session_store
    if (
        type(store) not in (InMemorySessionStore, SQLiteSessionStore, PostgresSessionStore)
        or "_scan_participant_session_bindings" in vars(store)
        or not store._supports_session_continuation_protocol()
    ):
        raise CollaborationUnavailable("Session discovery requires a qualified native owner.")
    await app.inspect_participant(participant, context=context)
    rows = await store._scan_participant_session_bindings(
        query.participant, after=query.after, limit=query.limit
    )
    if len(rows) > query.limit:
        raise CollaborationUnavailable("Session discovery exceeded its page bound.")
    items = []
    previous = query.after
    for raw in rows:
        item = prepare_contract(ParticipantSessionReference, raw, redactor=redactor)
        if item.participant != participant or (
            previous is not None and item.creation_key <= previous
        ):
            raise CollaborationUnavailable("Session discovery source ordering conflicts.")
        items.append(item)
        previous = item.creation_key
    return tuple(items), (
        ParticipantSessionCursor(participant=participant, after=previous)
        if len(items) == query.limit and previous is not None
        else None
    )


def prepare_scan(participant, after, limit):
    return prepare_contract(
        ParticipantSessionScan,
        {"participant": participant, "after": after, "limit": limit},
        redactor=SecretRedactor(),
    )


def scan_parameters(query):
    participant = query.participant
    return (
        participant.owner.application_scope,
        participant.owner.owner_id,
        participant.owner.incarnation,
        participant.participant_id,
        participant.incarnation,
        query.after or "",
        query.limit,
    )


def reference(receipt, query):
    receipt = prepare_contract(
        ParticipantSessionCreationReceipt, receipt, redactor=SecretRedactor()
    )
    binding = receipt.binding
    if binding.participant != query.participant or (
        query.after is not None and binding.creation_key <= query.after
    ):
        raise ValueError("Participant inventory contradicts its exact owner or cursor.")
    return ParticipantSessionReference(
        participant=binding.participant,
        creation_key=binding.creation_key,
        session_id=binding.session_id,
        session_instance_id=binding.session_instance_id,
        receipt_commitment=receipt.receipt_commitment,
    )
