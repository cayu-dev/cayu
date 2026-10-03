"""Source-owned continuation recovery, including after wait-source pruning.

Session creation inventory selects the exact incarnation. The native protected
checkpoint is the ticket index; no host-maintained queue or retry handle is used.
Every value returned here is readback, never a launch or disclosure capability.
"""

from dataclasses import dataclass

from pydantic import Field, StrictInt, StrictStr

from cayu.collaboration._contracts import ContractValue, Identifier
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration.access import CollaborationAccessContext
from cayu.runtime._session_continuation_store import (
    ROOT_KEY,
    ContinuationRoot,
    digest,
    require_history,
)
from cayu.sessions._participant_discovery import ParticipantSessionReference
from cayu.sessions._session_continuation import (
    CONTINUATION_NAMESPACE_KEY,
    ContinuationConflict,
    ContinuationRecord,
    ContinuationUnavailable,
    continuation_digest,
    continuation_operation_key,
)
from cayu.sessions._session_continuation_scope import publication_scope
from cayu.sessions.base import _invocation_lifecycle_authority_read_scope
from cayu.sessions.context_views import ParticipantSessionCreationReceipt


class ContinuationRecovery(ContractValue):
    session: ParticipantSessionReference
    registration_key: Identifier
    ticket_key: StrictStr = Field(pattern=r"^session-continuation:[0-9a-f]{64}$")
    preparation_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class _Scan(ContractValue):
    session: ParticipantSessionReference
    after: StrictStr | None = Field(default=None, pattern=r"^session-continuation:[0-9a-f]{64}$")
    limit: StrictInt = Field(ge=1, le=32)


@dataclass(frozen=True, slots=True)
class ContinuationDiscoveryPage:
    items: tuple[ContinuationRecovery, ...]
    next_cursor: str | None


async def _require_session(app, expected, context):
    from cayu.sessions.base import InMemorySessionStore
    from cayu.storage.postgres import PostgresSessionStore
    from cayu.storage.sqlite import SQLiteSessionStore

    store = app.session_store
    if (
        type(store) not in (InMemorySessionStore, SQLiteSessionStore, PostgresSessionStore)
        or not store._supports_session_continuation_protocol()
    ):
        raise ContinuationUnavailable("Continuation discovery requires a qualified native store.")
    await app.inspect_participant(expected.participant, context=context)
    raw = await store.load_participant_session_creation_receipt(expected.session_id)
    if raw is None:
        raise ContinuationUnavailable("Continuation source creation is unavailable.")
    receipt = prepare_contract(
        ParticipantSessionCreationReceipt, raw, redactor=app._secret_redactor
    )
    binding = receipt.binding
    if (
        binding.participant != expected.participant
        or binding.session_id != expected.session_id
        or binding.session_instance_id != expected.session_instance_id
        or binding.creation_key != expected.creation_key
        or receipt.receipt_commitment != expected.receipt_commitment
    ):
        raise ContinuationConflict("Continuation discovery belongs to another native creation.")


async def discover_session_continuations(app, session, *, context, after=None, limit=32):
    redactor = app._secret_redactor
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    query = prepare_contract(
        _Scan, {"session": session, "after": after, "limit": limit}, redactor=redactor
    )
    await _require_session(app, query.session, context)
    store = app.session_store
    with (
        publication_scope(CONTINUATION_NAMESPACE_KEY),
        _invocation_lifecycle_authority_read_scope(),
    ):
        checkpoint = await store.load_checkpoint(query.session.session_id)
        namespace = await store.load_session_operation(
            query.session.session_id, CONTINUATION_NAMESPACE_KEY
        )
    raw_root = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if raw_root is None:
        if namespace is not None:
            raise ContinuationUnavailable("Continuation index lacks its retained namespace.")
        return ContinuationDiscoveryPage((), None)
    root = prepare_contract(ContinuationRoot, raw_root, redactor=redactor)
    if (
        root.namespace.session_id != query.session.session_id
        or root.namespace.session_instance_id != query.session.session_instance_id
        or root.namespace.model_dump(mode="json") != namespace
    ):
        raise ContinuationUnavailable("Continuation index contradicts its exact namespace.")
    entries = sorted(
        (entry for entry in root.entries if query.after is None or entry.ticket_key > query.after),
        key=lambda entry: entry.ticket_key,
    )[: query.limit]
    items = []
    used = 0
    truncated = False
    for entry in entries:
        raw = await store.load_session_operation(query.session.session_id, entry.ticket_key)
        if raw is None:
            raise ContinuationUnavailable("Indexed continuation has no retained record.")
        record = prepare_contract(ContinuationRecord, raw, redactor=redactor)
        require_history(record)
        if (
            record.namespace != root.namespace
            or continuation_operation_key(record.ticket) != entry.ticket_key
            or digest(record.model_dump(mode="json")) != entry.record_sha256
        ):
            # A concurrent native transition is a retryable observation conflict,
            # never evidence that the indexed responsibility has disappeared.
            raise ContinuationUnavailable("Continuation changed during discovery.")
        token = ContinuationRecovery(
            session=query.session,
            registration_key=record.ticket.registration_key,
            ticket_key=entry.ticket_key,
            preparation_digest=continuation_digest(record.preparation),
        )
        size = len(contract_bytes(token, redactor=redactor))
        if items and used + size > 60 * 1024:
            truncated = True
            break
        items.append(token)
        used += size
    await _require_session(app, query.session, context)
    return ContinuationDiscoveryPage(
        tuple(items),
        items[-1].ticket_key if items and (truncated or len(entries) == query.limit) else None,
    )


async def recover_session_continuation(app, expected, *, context):
    expected = prepare_contract(ContinuationRecovery, expected, redactor=app._secret_redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=app._secret_redactor)
    await _require_session(app, expected.session, context)
    record = await app.session_store.load_continuation_ticket(
        expected.session.session_id,
        session_instance_id=expected.session.session_instance_id,
        registration_key=expected.registration_key,
    )
    if record is None:
        raise ContinuationUnavailable("Discovered continuation record is unavailable.")
    record = prepare_contract(ContinuationRecord, record, redactor=app._secret_redactor)
    if (
        continuation_operation_key(record.ticket) != expected.ticket_key
        or continuation_digest(record.preparation) != expected.preparation_digest
    ):
        raise ContinuationConflict("Continuation recovery conflicts with its original preparation.")
    await _require_session(app, expected.session, context)
    return record
