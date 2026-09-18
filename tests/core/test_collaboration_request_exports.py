"""Real session-export evidence through the public collaboration request APIs."""

from __future__ import annotations

from contextlib import asynccontextmanager, suppress
from uuid import uuid4

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_collaboration_request_foundation import public_setup

from cayu.applications import CayuApp
from cayu.collaboration._contracts import CollaborationConflict, ExactMatch, ObjectRef, OperationRef
from cayu.collaboration._permits import ReceivingSettlementReceipt
from cayu.collaboration._session_export_participant import (
    SessionExportRequestReceivingOwner,
)
from cayu.collaboration.exports import (
    ExportLimits,
    SessionExportAcceptance,
    SessionExportAcceptanceReader,
    SessionExportAccessContext,
    SessionExportAuthorization,
    SessionExportPolicy,
    SessionExportProjector,
    SessionExportRef,
    SessionExportRegistration,
    SessionExportRequest,
    SessionExportUnavailable,
)
from cayu.collaboration.mandates import ResourceSelector
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.request_access import RequestRegistration
from cayu.collaboration.requests import (
    RequestAdmissionCommand,
    RequestControl,
    RequestOutcomeCommand,
    RequestProgressCommand,
)
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionIdentity, SessionStatus
from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.migrations import SchemaMode

pytestmark = pytest.mark.anyio
stores = identity_tests.stores


class _Policy(SessionExportPolicy):
    def __init__(self, owner, ref):
        self.owner = owner
        self._ref = ref

    @property
    def ref(self):
        return self._ref

    @asynccontextmanager
    async def acquire(self, context, *, session_id, session_instance_id, actions, audience=None):
        if context.principal != "operator" or not actions:
            raise PermissionError("unqualified export access")
        yield SessionExportAuthorization(
            issuer=self.owner,
            principal=context.principal,
            policy=self.ref,
            revision=1,
            expires_at_ms=4102444800000,
        )


class _Projector(SessionExportProjector):
    def __init__(self, owner, ref):
        self.owner = owner
        self._ref = ref

    @property
    def ref(self):
        return self._ref

    def project(self, source):
        return {
            "count": sum(
                len(part.text)
                for row in source
                for part in row.message.content
                if part.type == "text"
            )
        }

    def validate(self, source, output, audience):
        return audience == self.owner and output == self.project(source)


class _PublicExportReader(SessionExportAcceptanceReader):
    """A controlled receiver: public readback authenticates the source.

    The permit is deliberately retained from the accepted request snapshot. It
    is not manufactured from the terminal command supplied by the caller.
    """

    def __init__(self, owner, context):
        self._owner = owner
        self._context = context
        self.app = None
        self.permit = None
        self.allow_settlement = True
        self.lookup_calls = 0

    @property
    def owner(self):
        return self._owner

    async def lookup(self, receipt):
        self.lookup_calls += 1
        found = await self.app.lookup_session_export(
            receipt.expected.intent.request, context=self._context
        )
        if found.status != "match":
            return found
        return ExactMatch[SessionExportAcceptance](
            receipt=SessionExportAcceptance(
                export_receipt=found.receipt,
                receiving_owner=self.owner,
                receipt_id="public-acceptance",
            )
        )

    async def settlement(self, receipt, expected):
        found = await self.lookup(receipt)
        if (
            not self.allow_settlement
            or not isinstance(found, ExactMatch)
            or expected != self.permit
        ):
            from cayu.collaboration._contracts import ExactUnavailable

            return ExactUnavailable()
        return ExactMatch[ReceivingSettlementReceipt](
            receipt=ReceivingSettlementReceipt(
                expected=self.permit,
                receiving_owner=self.owner,
                receipt_id="public-settlement",
                outcome="quiescent",
            )
        )


class _Integration:
    def __init__(
        self, app, session_store, values, resolver, source, reader, receiving_owner, export_context
    ):
        self.app = app
        self.session_store = session_store
        self.values = values
        self.resolver = resolver
        self.source = source
        self.reader = reader
        self.receiving_owner = receiving_owner
        self.export_context = export_context
        self.context = None


async def _integration(store, tmp_path, postgres_dsn, *, lose_ack=False):
    base, resolver, values = await public_setup(store)
    export_resolver = type(resolver)(values[4])
    collaboration_registration = base._participant_coordinator._registration
    initialized = values[1]
    owner = initialized.owner
    if type(store).__name__ == "InMemoryCollaborationStore":
        session_type, session_args, session_kwargs = InMemorySessionStore, (), {}
    elif type(store).__name__ == "SQLiteCollaborationStore":
        session_type = SQLiteSessionStore
        session_args, session_kwargs = (tmp_path / "request-exports.sqlite",), {}
    else:
        session_type = PostgresSessionStore
        session_args, session_kwargs = (postgres_dsn,), {"schema_mode": SchemaMode.CREATE}
    if lose_ack:
        base = session_type

        class LostAcknowledgementStore(base):
            session_export_version = 1

            async def publish_session_operation_guarded_with_store_time(self, *args, **kwargs):
                result = await super().publish_session_operation_guarded_with_store_time(
                    *args, **kwargs
                )
                if any(
                    event.type == EventType.SESSION_EXPORT_PUBLISHED for event in kwargs["events"]
                ):
                    self.lost_ack = True
                    raise OSError("lost session-export publication acknowledgement")
                return result

        session_type = LostAcknowledgementStore
    session_store = session_type(*session_args, **session_kwargs)
    session_store.lost_ack = False

    session_id = "request-export-" + uuid4().hex
    await session_store.create(
        RunRequest(session_id=session_id, agent_name="assistant", messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake"),
    )
    await session_store.append_transcript_messages(session_id, [Message.text("user", "hello")])
    await session_store.update_status(session_id, SessionStatus.COMPLETED)
    await session_store.append_event(
        session_id, Event(type=EventType.SESSION_COMPLETED, session_id=session_id)
    )

    request_contract = values[4].output_contract
    assert request_contract is not None
    export_context = SessionExportAccessContext(
        principal="operator", mandate=export_resolver.context
    )
    policy = _Policy(owner, values[4].disclosure_policy)
    projector = _Projector(owner, request_contract)
    reader = _PublicExportReader(owner, export_context)
    receiving_owner = SessionExportRequestReceivingOwner(audience=owner, reader=reader)
    registration = SessionExportRegistration(
        owner=owner,
        policy=policy,
        projectors=(projector,),
        limits=ExportLimits(max_exports=16, max_pending=8, max_retained_bytes=1024 * 1024),
        readers=(reader,),
        mandates=export_resolver,
    )
    app = CayuApp(
        session_store=session_store,
        collaboration_store=store,
        collaboration=collaboration_registration,
        collaboration_requests=RequestRegistration(
            mandates=resolver,
            max_ttl_ms=300_000,
            receiving_owner=receiving_owner,
        ),
        session_exports=registration,
        enable_logging=False,
    )
    reader.app = app
    await app.initialize_collaboration()
    namespace = await app.initialize_session_exports(session_id, context=export_context)
    request = SessionExportRequest(
        ref=SessionExportRef(
            session_id=session_id,
            session_instance_id=namespace.session_instance_id,
            operation=OperationRef(
                application_scope=owner.application_scope,
                namespace_incarnation=namespace.namespace_incarnation,
                generation=1,
                caller_key="source-export",
            ),
        ),
        source_indices=(0,),
        audience=owner,
        projector=projector.ref,
        policy=policy.ref,
    )
    source_selector = ResourceSelector(
        resource=ObjectRef(
            owner=owner,
            kind="session_transcript_row",
            object_id=session_id,
            incarnation=namespace.session_instance_id,
            revision=1,
        )
    )
    source_actions = ("consult", "readback", "administer", "source", "publish")
    export_resolver.resolution = export_resolver.resolution.model_copy(
        update={
            "principal": export_resolver.resolution.principal.model_copy(
                update={"actions": source_actions}
            ),
            "chain": export_resolver.resolution.chain.model_copy(
                update={
                    "entries": (
                        export_resolver.resolution.chain.entries[-1].model_copy(
                            update={
                                "actions": source_actions,
                                "resources": (source_selector,),
                                "restrictions": export_resolver.resolution.chain.entries[
                                    -1
                                ].restrictions.model_copy(
                                    update={"channels": ("prompt", "source")}
                                ),
                            }
                        ),
                    )
                }
            ),
        }
    )
    if lose_ack:
        with suppress(SessionExportUnavailable):
            await app.export_session(request, context=export_context)
        lookup = await app.lookup_session_export(request, context=export_context)
        assert isinstance(lookup, ExactMatch)
        source = lookup.receipt
    else:
        source = await app.export_session(request, context=export_context)
    return _Integration(
        app, session_store, values, resolver, source, reader, receiving_owner, export_context
    )


async def _admit(case, *, decision="continue", with_source=True):
    app, resolver, values = case.app, case.resolver, case.values
    request = values[4]
    accepted = await app.accept_collaboration_request(request, context=resolver.context)
    snapshot = await app.inspect_collaboration_request(accepted.expected, context=resolver.context)
    case.reader.permit = snapshot.permit
    initiator = values[5].model_copy(
        update={
            "participant": ObjectRef(
                owner=values[3].reference.owner,
                kind="participant",
                object_id=values[3].reference.participant_id,
                incarnation=values[3].reference.incarnation,
            )
        }
    )
    context = resolver.context.model_copy(update={"participant": values[3].reference})
    case.context = context
    resolver.context = context
    resolution = resolver.resolution
    resolver.resolution = resolution.model_copy(
        update={
            "principal": resolution.principal.model_copy(
                update={"participants": (values[3].reference,)}
            ),
            "chain": resolution.chain.model_copy(
                update={
                    "entries": (
                        resolution.chain.entries[-1].model_copy(
                            update={"participant": values[3].reference}
                        ),
                    )
                }
            ),
        }
    )
    admission = RequestAdmissionCommand(
        operation=values[1].operation("admission"),
        expected=accepted.expected,
        expected_revision=1,
        generation=1,
        decision=decision,
        source_export=case.source.expected.intent.request.ref if with_source else None,
        source_receipt=case.source if with_source else None,
        evidence=(
            ObjectRef(
                owner=values[3].reference.owner,
                kind="participant",
                object_id=values[3].reference.participant_id,
                incarnation=values[3].reference.incarnation,
            ),
        ),
        initiator=initiator,
    )
    async with case.receiving_owner.acquire(admission, context=context):
        pass
    return accepted, initiator, await app.admit_collaboration_request(admission, context=context)


@pytest.mark.parametrize("decision", ["defer", "clarify", "decline"])
async def test_real_session_export_receiver_accepts_source_free_nonexecution_decisions(
    stores, tmp_path, request, decision
):
    store = stores()
    dsn = request.getfixturevalue("postgres_dsn") if "Postgres" in type(store).__name__ else None
    case = await _integration(store, tmp_path, dsn)
    try:
        accepted, initiator, admission = await _admit(case, decision=decision, with_source=False)
        assert (
            admission.state
            == {
                "defer": "deferred",
                "clarify": "clarifying",
                "decline": "closed",
            }[decision]
        )
        if decision in {"defer", "clarify"}:
            before = await case.app.inspect_collaboration_request(
                accepted.expected, context=case.context
            )
            for outcome in ("answered", "failed"):
                command = RequestOutcomeCommand(
                    operation=case.values[1].operation(outcome),
                    expected=accepted.expected,
                    expected_revision=2,
                    outcome=outcome,
                    commitment=case.source.expected.intent.output_commitment
                    if outcome == "answered"
                    else None,
                    source_receipt=case.source,
                    initiator=initiator,
                )
                with pytest.raises(CollaborationConflict):
                    await case.app.publish_collaboration_outcome(command, context=case.context)
            assert (
                await case.app.inspect_collaboration_request(
                    accepted.expected, context=case.context
                )
                == before
            )
            expire = RequestControl(
                operation=case.values[1].operation("expire"),
                expected=accepted.expected,
                expected_revision=2,
                kind="expire",
                source_receipt=case.source,
            )
            with pytest.raises(CollaborationConflict):
                await case.app.control_collaboration_request(expire, context=case.context)
    finally:
        await case.app.drain_collaboration_requests()
        await case.app.drain_session_exports()
        if hasattr(case.session_store, "close"):
            await case.session_store.close()


async def test_progress_rejects_a_different_authenticated_export(stores, tmp_path, request):
    store = stores()
    dsn = request.getfixturevalue("postgres_dsn") if "Postgres" in type(store).__name__ else None
    case = await _integration(store, tmp_path, dsn)
    try:
        original = case.source.expected.intent.request
        foreign_request = original.model_copy(
            update={
                "ref": original.ref.model_copy(
                    update={
                        "operation": original.ref.operation.model_copy(
                            update={"caller_key": "other-export"}
                        )
                    }
                )
            }
        )
        foreign = await case.app.export_session(foreign_request, context=case.export_context)
        assert (
            foreign.expected.intent.output_commitment
            == case.source.expected.intent.output_commitment
        )
        accepted, initiator, _ = await _admit(case)
        before = await case.app.inspect_collaboration_request(
            accepted.expected, context=case.context
        )
        command = RequestProgressCommand(
            operation=case.values[1].operation("progress"),
            expected=accepted.expected,
            expected_revision=2,
            admission_generation=1,
            publisher=initiator,
            sequence=1,
            kind="producing",
            commitment=foreign.expected.intent.output_commitment,
            source_receipt=foreign,
        )
        with pytest.raises(CollaborationConflict):
            await case.app.record_collaboration_progress(command, context=case.context)
        assert (
            await case.app.inspect_collaboration_request(accepted.expected, context=case.context)
            == before
        )
        result = await case.app.record_collaboration_progress(
            command.model_copy(update={"source_receipt": case.source}), context=case.context
        )
        assert result.command.source_receipt == case.source
    finally:
        await case.app.drain_collaboration_requests()
        await case.app.drain_session_exports()
        if hasattr(case.session_store, "close"):
            await case.session_store.close()


async def test_source_evidence_is_bound_to_the_request_contract_and_producer(
    stores, tmp_path, request
):
    store = stores()
    dsn = request.getfixturevalue("postgres_dsn") if "Postgres" in type(store).__name__ else None
    case = await _integration(store, tmp_path, dsn)
    try:
        app, resolver, values = case.app, case.resolver, case.values
        accepted = await app.accept_collaboration_request(values[4], context=resolver.context)
        context = resolver.context.model_copy(update={"participant": values[3].reference})
        initiator = values[5].model_copy(
            update={
                "participant": ObjectRef(
                    owner=values[3].reference.owner,
                    kind="participant",
                    object_id=values[3].reference.participant_id,
                    incarnation=values[3].reference.incarnation,
                )
            }
        )
        changed_source = case.source.model_copy(
            deep=True,
            update={
                "expected": case.source.expected.model_copy(
                    deep=True,
                    update={
                        "intent": case.source.expected.intent.model_copy(
                            deep=True,
                            update={
                                "request": case.source.expected.intent.request.model_copy(
                                    update={
                                        "projector": ObjectRef(
                                            owner=values[1].owner,
                                            kind="projector",
                                            object_id="unrelated",
                                            incarnation="one",
                                            revision=1,
                                        )
                                    }
                                )
                            },
                        )
                    },
                )
            },
        )
        with pytest.raises(ValueError):
            admission = RequestAdmissionCommand(
                operation=values[1].operation("admission-mismatch"),
                expected=accepted.expected,
                expected_revision=1,
                generation=1,
                decision="continue",
                source_export=changed_source.expected.intent.request.ref,
                source_receipt=changed_source,
                evidence=(),
                initiator=initiator,
            )
            await app.admit_collaboration_request(admission, context=context)
    finally:
        await case.app.drain_collaboration_requests()
        await case.app.drain_session_exports()
        if hasattr(case.session_store, "close"):
            await case.session_store.close()


@pytest.mark.parametrize("terminal", ["answer", "failure", "cancel"])
async def test_public_session_export_admission_and_terminal_settlement(
    stores, tmp_path, request, terminal
):
    store = stores()
    dsn = request.getfixturevalue("postgres_dsn") if "Postgres" in type(store).__name__ else None
    case = await _integration(store, tmp_path, dsn)
    accepted, initiator, admission = await _admit(case)
    assert admission.state == "admitted"
    if terminal == "cancel":
        intent = RequestControl(
            operation=case.values[1].operation("cancel"),
            expected=accepted.expected,
            expected_revision=2,
            kind="cancel",
            source_receipt=case.source,
        )
        result = await case.app.control_collaboration_request(intent, context=case.context)
        assert result.state == "cancelled"
    else:
        command = RequestOutcomeCommand(
            operation=case.values[1].operation("outcome"),
            expected=accepted.expected,
            expected_revision=2,
            outcome="answered" if terminal == "answer" else "failed",
            commitment=(
                case.source.expected.intent.output_commitment if terminal == "answer" else None
            ),
            source_receipt=case.source,
            initiator=initiator,
        )
        result = await case.app.publish_collaboration_outcome(command, context=case.context)
        assert result.event.type == (
            "request_answered" if terminal == "answer" else "request_failed"
        )
    assert (
        await case.app.lookup_session_export(
            case.source.expected.intent.request,
            context=case.export_context,
        )
    ).receipt == case.source
    if hasattr(case.session_store, "close"):
        await case.session_store.close()


async def test_missing_authenticated_settlement_is_rejected_without_terminal_mutation(
    stores, tmp_path, request
):
    store = stores()
    dsn = request.getfixturevalue("postgres_dsn") if "Postgres" in type(store).__name__ else None
    case = await _integration(store, tmp_path, dsn)
    accepted, initiator, _ = await _admit(case)
    case.reader.allow_settlement = False
    command = RequestOutcomeCommand(
        operation=case.values[1].operation("failed"),
        expected=accepted.expected,
        expected_revision=2,
        outcome="failed",
        source_receipt=case.source,
        initiator=initiator,
    )
    with pytest.raises(CollaborationUnavailable):
        await case.app.publish_collaboration_outcome(command, context=case.context)
    snapshot = await case.app.inspect_collaboration_request(accepted.expected, context=case.context)
    assert snapshot.state == "open" and snapshot.revision == 2
    if hasattr(case.session_store, "close"):
        await case.session_store.close()


async def test_lost_export_ack_is_reconciled_before_request_authentication(
    stores, tmp_path, request
):
    store = stores()
    dsn = request.getfixturevalue("postgres_dsn") if "Postgres" in type(store).__name__ else None
    case = await _integration(store, tmp_path, dsn, lose_ack=True)
    assert case.session_store.lost_ack
    assert (await case.reader.lookup(case.source)).status == "match"
    await _admit(case)
    if hasattr(case.session_store, "close"):
        await case.session_store.close()


@pytest.mark.parametrize("terminal", ["answer", "decline"])
async def test_terminal_replay_does_not_require_settlement_again(
    stores, tmp_path, request, terminal
):
    store = stores()
    dsn = request.getfixturevalue("postgres_dsn") if "Postgres" in type(store).__name__ else None
    case = await _integration(store, tmp_path, dsn)
    try:
        accepted, initiator, admission = await _admit(
            case, decision="decline" if terminal == "decline" else "continue"
        )
        if terminal == "decline":
            command, first = admission.command, admission
            publish = case.app.admit_collaboration_request
            changed = command.model_copy(update={"proposal_commitment": "changed"})
        else:
            command = RequestOutcomeCommand(
                operation=case.values[1].operation("answer"),
                expected=accepted.expected,
                expected_revision=2,
                outcome="answered",
                commitment=case.source.expected.intent.output_commitment,
                source_receipt=case.source,
                initiator=initiator,
            )
            publish = case.app.publish_collaboration_outcome
            first = await publish(command, context=case.context)
            changed = command.model_copy(update={"expected_revision": 3})

        async def unavailable(*args, **kwargs):
            raise AssertionError("Committed replay must not query settlement")

        case.reader.settlement = unavailable
        case.reader.lookup = unavailable
        before = await case.app.inspect_collaboration_request(
            accepted.expected, context=case.context
        )
        case.reader.lookup_calls = 0
        assert await publish(command, context=case.context) == first
        assert case.reader.lookup_calls == 0
        with pytest.raises(CollaborationConflict):
            await publish(changed, context=case.context)
        assert (
            await case.app.inspect_collaboration_request(accepted.expected, context=case.context)
            == before
        )
    finally:
        await case.app.drain_collaboration_requests()
        await case.app.drain_session_exports()
        if hasattr(case.session_store, "close"):
            await case.session_store.close()
