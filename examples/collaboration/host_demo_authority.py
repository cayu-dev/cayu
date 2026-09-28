"""Finite application authority for the credential-free host example.

This is example policy, not a new Cayu receiving owner. Only the local demo
administrator installs grants. A receipt alone never installs a disclosure
grant. Restarting an application requires re-registering its exact policy and
reconciling retained owner evidence before installing equivalent grants.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from uuid import uuid4

from cayu import (
    CollaborationAccessDenied,
    CollaborationAccessGrant,
    CollaborationAccessPolicy,
    CollaborationMandate,
    MandateAccessContext,
    MandateChain,
    MandateDenied,
    MandateResolution,
    MandateResolver,
    MandateRestrictions,
    PrincipalResolution,
    SessionExportAuthorization,
    SessionExportDenied,
    SessionExportPolicy,
    SessionExportProjector,
)
from cayu.collaboration._contracts import ExactMatch, ObjectRef
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration.participants import ParticipantRef
from cayu.collaboration.peer_content import PeerContentAppendAuthorization
from cayu.sessions.creation_fence import SessionCreationDecision
from cayu.vaults.redaction import SecretRedactor

PRINCIPAL = "host-example-admin"


def commitment(value) -> str:
    """Commit a previously validated, bounded owner value using canonical JSON."""
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def reference(owner, kind, key):
    return ObjectRef(owner=owner, kind=kind, object_id=key, incarnation="demo-v1", revision=1)


class Administration(CollaborationAccessPolicy):
    def __init__(self, scope):
        self._scope = scope

    def authorize(self, context, *, application_scope, action):
        if (
            context.principal != PRINCIPAL
            or application_scope != self._scope
            or action
            not in {
                "discover",
                "inspect",
                "administration",
                "readback",
                "create",
                "configure",
                "alias",
                "namespace_inspect",
                "namespace_seal",
                "namespace_rotate",
                "namespace_retire",
                "namespace_prune",
                "participant_lifecycle",
                "obligations",
                "request_accept",
                "request_readback",
                "request_control",
            }
        ):
            raise CollaborationAccessDenied("Example administration denied.")
        return CollaborationAccessGrant(application_scope=application_scope, participants=None)


class DemoMandates(MandateResolver):
    """A held, exact local principal catalogue with explicit source grants."""

    def __init__(self, owner, participants, contract, *, expires_at_ms):
        self._reference = reference(owner, "mandate_resolver", "host-example")
        self._condition = asyncio.Condition()
        self._readers = {}
        self.contexts = {}
        self._resolutions = {}
        self._question_contexts = {}
        actions = (
            "consult",
            "prepare",
            "execute",
            "publish",
            "administer",
            "readback",
            "source",
            "expose",
            "release",
            "retire",
        )
        for participant in participants:
            mandate_ref = reference(owner, "mandate", participant.participant_id)
            context = MandateAccessContext(
                issuer=owner, principal=PRINCIPAL, participant=participant, mandate=mandate_ref
            )
            mandate = CollaborationMandate(
                reference=mandate_ref,
                root=mandate_ref,
                parent=None,
                issuer=owner,
                principal=PRINCIPAL,
                participant=participant,
                audiences=(owner,),
                scopes=(owner.application_scope,),
                actions=actions,
                resources=(),
                remaining_delegations=0,
                sponsor=None,
                budgets=(),
                restrictions=MandateRestrictions(
                    channels=("prompt", "source"),
                    excluded_sources=(),
                    independence_policy=contract,
                    disclosure_policy=contract,
                ),
                expires_at_ms=expires_at_ms,
                revocation_generation=1,
            )
            self.contexts[participant] = context
            self._resolutions[context] = MandateResolution(
                principal=PrincipalResolution(
                    resolver=self.ref,
                    issuer=owner,
                    principal=PRINCIPAL,
                    participants=(participant,),
                    audiences=(owner,),
                    scopes=(owner.application_scope,),
                    actions=actions,
                    expires_at_ms=expires_at_ms,
                ),
                chain=MandateChain(entries=(mandate,)),
            )

    @property
    def ref(self):
        return self._reference

    @asynccontextmanager
    async def acquire(self, context):
        async with self._condition:
            resolution = self._resolutions.get(context)
            if resolution is None:
                raise MandateDenied()
            self._readers[context] = self._readers.get(context, 0) + 1
        try:
            # Independent native owners may authenticate the same principal
            # concurrently. All readers pin revocation; they do not serialize
            # one another across cross-owner callbacks.
            yield resolution
        finally:
            async with self._condition:
                self._readers[context] -= 1
                if not self._readers[context]:
                    del self._readers[context]
                self._condition.notify_all()

    async def question_context(self, participant, key):
        """Administrator-issued producer authority, separate for each question.

        Reusing a specialist does not merge independent questions' source grants.
        The base participant mandate remains resource-free; retained question
        contexts are replayed, never silently replaced during reconstruction.
        """
        if participant not in self.contexts or type(key) is not str or not 1 <= len(key) <= 32:
            raise ValueError("Example question authority requires a registered bounded selection.")
        async with self._condition:
            selected = (participant, key)
            if selected in self._question_contexts:
                return self._question_contexts[selected]
            if len(self._question_contexts) >= 16:
                raise ValueError("Example question authority capacity exhausted.")
            base = self.contexts[participant]
            original = self._resolutions[base]
            mandate = reference(base.issuer, "mandate", uuid4().hex)
            context = base.model_copy(update={"mandate": mandate})
            entry = original.chain.entries[0].model_copy(
                update={"reference": mandate, "root": mandate}
            )
            self._resolutions[context] = MandateResolution(
                principal=original.principal, chain=MandateChain(entries=(entry,))
            )
            self._question_contexts[selected] = context
            return context

    async def grant_source(self, context, resource, audience):
        """Explicit administrator decision after authenticating native source identity."""
        async with self._condition:
            if context not in self._question_contexts.values():
                raise MandateDenied()
            # Only users of this exact mandate can observe the changed grant.
            # Unrelated questions must not pin one another's administration.
            await self._condition.wait_for(lambda: not self._readers.get(context))
            current = self._resolutions[context]
            entry = current.chain.entries[0]
            if resource in entry.resources and audience in entry.audiences:
                return
            if resource not in entry.resources and len(entry.resources) >= 16:
                raise ValueError("Example source grant capacity exhausted.")
            audiences = tuple(dict.fromkeys((*entry.audiences, audience)))
            self._resolutions[context] = MandateResolution(
                principal=current.principal.model_copy(update={"audiences": audiences}),
                chain=MandateChain(
                    entries=(
                        entry.model_copy(
                            update={
                                "audiences": audiences,
                                "resources": tuple(dict.fromkeys((*entry.resources, resource))),
                                "revocation_generation": entry.revocation_generation + 1,
                            }
                        ),
                    )
                ),
            )


class VisibleText(SessionExportProjector):
    def __init__(self, ref):
        self._reference = ref

    @property
    def ref(self):
        return self._reference

    def project(self, source):
        if any(part.type != "text" for row in source for part in row.message.content):
            raise ValueError("The example exports visible text only.")
        return {"text": "".join(part.text for row in source for part in row.message.content)}

    def validate(self, source, output, audience):
        return output == self.project(source)


@dataclass(frozen=True)
class QuestionInput:
    """Application-selected visible work product, not a native execution grant."""

    source_receipt: str
    content_commitment: str
    text: str
    question_key: str
    recipient: ParticipantRef


class ExactDisclosure(SessionExportPolicy):
    """Application-selected exact source/target grants; no receipt-to-grant inference."""

    def __init__(self, ref, *, expires_at_ms, session_store=None):
        self._reference = ref
        self._expiry = expires_at_ms
        self._lock = asyncio.Lock()
        self._sources = set()
        self._grants = {}
        self._revoked = False
        self._sessions = session_store
        self._question_inputs = {}

    @property
    def ref(self):
        return self._reference

    async def allow_source(self, session_id, instance_id):
        async with self._lock:
            if (session_id, instance_id) not in self._sources and len(self._sources) >= 16:
                raise ValueError("Example source capacity exhausted.")
            self._sources.add((session_id, instance_id))

    async def revoke(self):
        """Wait for held permission uses; deny subsequent disclosure."""
        async with self._lock:
            self._revoked = True

    async def allow_question_input(
        self, *, source_receipt, text, question_key, recipient, forward_from=None
    ):
        """Explicit administrator decision to reuse one already selected visible reply.

        The source must have traversed authenticated export/readback and the
        original destination grant. By default only that recipient can reuse it.
        Forwarding requires an explicit administrator decision naming the exact
        original recipient as forward_from and the new recipient in this owner.
        The new input is application-owned text, not private transcript history.
        """
        async with self._lock:
            grant = self._grants.get(source_receipt)
            original_recipient = recipient if forward_from is None else forward_from
            if (
                grant is None
                or self._revoked
                or time.time_ns() // 1_000_000 >= self._expiry
                or type(text) is not str
                or not text
                or len(text.encode()) > 2048
                or type(question_key) is not str
                or not 1 <= len(question_key) <= 32
                or type(recipient) is not ParticipantRef
                or type(original_recipient) is not ParticipantRef
                or recipient.owner != original_recipient.owner
                or (
                    original_recipient.participant_id,
                    original_recipient.incarnation,
                    original_recipient.owner,
                )
                != (grant[2].consumer_id, grant[2].consumer_participant_incarnation, grant[4])
                or commitment({"text": text, "artifact_commitments": []}) != grant[1]
            ):
                raise SessionExportDenied()
            value = QuestionInput(source_receipt, grant[1], text, question_key, recipient)
            # Keep private immutable values, not the caller-visible dataclass
            # or a shared nested model. Frozen containers are not provenance.
            key = (question_key, contract_bytes(recipient, redactor=SecretRedactor()))
            authority = (source_receipt, grant[1], text)
            previous = self._question_inputs.get(key)
            if previous is not None and previous != authority:
                raise SessionExportDenied()
            if previous is None and len(self._question_inputs) >= 16:
                raise ValueError("Example question-input capacity exhausted.")
            self._question_inputs[key] = authority
            return value

    async def question_input(self, value, *, question_key, recipient):
        """Check current application permission before accepting derived input."""
        async with self._lock:
            if (
                type(value) is not QuestionInput
                or any(
                    type(field) is not str
                    for field in (
                        value.source_receipt,
                        value.content_commitment,
                        value.text,
                        value.question_key,
                    )
                )
                or value.question_key != question_key
                or value.recipient != recipient
                or self._question_inputs.get(
                    (question_key, contract_bytes(recipient, redactor=SecretRedactor()))
                )
                != (value.source_receipt, value.content_commitment, value.text)
                or self._revoked
                or time.time_ns() // 1_000_000 >= self._expiry
            ):
                raise SessionExportDenied()
            return (
                "\nPRECEDING_REPLY_SOURCE:"
                + value.source_receipt
                + "\nPRECEDING_REPLY_COMMITMENT:"
                + value.content_commitment
                + "\nPRECEDING_REPLY:"
                + value.text
            )

    async def allow_delivery(self, *, receipt, payload, destination, producer_key):
        """Called only after application readback authenticates this exact export.

        Selecting the destination here is the explicit new disclosure decision;
        no other candidate or target can reuse the grant.
        """
        expected = receipt.expected.intent.request
        if (
            (expected.ref.session_id, expected.ref.session_instance_id) not in self._sources
            or expected.audience.owner_id != destination.recipient.participant_id
            or expected.audience.incarnation != destination.recipient.incarnation
            or expected.projector != destination.projector
            or expected.policy != self.ref
        ):
            raise SessionExportDenied()
        value = (
            receipt,
            commitment({"text": payload["text"], "artifact_commitments": []}),
            destination.attempt.append_key,
            producer_key,
            destination.recipient.owner,
        )
        async with self._lock:
            prior = self._grants.get(receipt.event_id)
            if prior is not None and prior != value:
                raise SessionExportDenied()
            if prior is None and len(self._grants) >= 16:
                raise ValueError("Example disclosure capacity exhausted.")
            self._grants[receipt.event_id] = value

    @asynccontextmanager
    async def acquire(self, context, *, session_id, session_instance_id, actions, audience=None):
        async with self._lock:
            if (
                context.principal != PRINCIPAL
                or not actions
                or any(
                    action
                    not in {
                        "initialize",
                        "readback",
                        "source",
                        "export",
                        "expose",
                        "release",
                        "retire",
                    }
                    for action in actions
                )
                or (session_id, session_instance_id) not in self._sources
                or (self._revoked and set(actions) - {"readback", "release", "retire"})
            ):
                raise SessionExportDenied()
            yield SessionExportAuthorization(
                issuer=self.ref.owner,
                principal=PRINCIPAL,
                policy=self.ref,
                revision=1,
                expires_at_ms=self._expiry,
            )

    def _require_occurrence(self, key, occurrence):
        grant = self._grants.get(occurrence.source_export_receipt_id)
        if grant is None or self._revoked or time.time_ns() // 1_000_000 >= self._expiry:
            raise SessionExportDenied()
        receipt, payload_digest, expected_key, producer_key, _receiving_owner = grant
        source = receipt.expected.intent.request.ref
        if (
            key != expected_key
            or occurrence.producer_receipt_id != producer_key
            or occurrence.payload.content_sha256 != payload_digest
            or (occurrence.sender_session_id, occurrence.sender_session_instance_id)
            != (source.session_id, source.session_instance_id)
            or occurrence.audience != (key.consumer_id,)
        ):
            raise SessionExportDenied()
        return grant

    @asynccontextmanager
    async def acquire_peer_append(self, context, *, request, append_key, occurrence):
        async with self._lock:
            if context.principal != PRINCIPAL:
                raise SessionExportDenied()
            self._require_occurrence(append_key, occurrence)
            yield PeerContentAppendAuthorization(
                source_export_receipt_id=occurrence.source_export_receipt_id,
                producer_receipt_id=occurrence.producer_receipt_id,
                source_session_id=occurrence.sender_session_id,
                source_session_instance_id=occurrence.sender_session_instance_id,
                content_sha256=occurrence.payload.content_sha256,
                audience=occurrence.audience,
            )

    @asynccontextmanager
    async def acquire_peer_read(self, context, *, append_key, receipt):
        async with self._lock:
            if context.principal != PRINCIPAL or receipt.occurrence is None:
                raise SessionExportDenied()
            self._require_occurrence(append_key, receipt.occurrence)
            yield

    @asynccontextmanager
    async def acquire_peer_exposures(self, context, *, items):
        async with self._lock:
            if not 1 <= len(items) <= 16:
                raise SessionExportDenied()
            for item in items:
                origin = item.origin
                key = origin.append_key
                grant = self._require_occurrence(key, item.occurrence)
                target = (key.target_session_id, key.target_session_instance_id)
                if key.creation_target is not None:
                    if self._sessions is None:
                        raise SessionExportDenied()
                    found = await self._sessions.read_session_creation_decision(key.creation_target)
                    if not isinstance(found, ExactMatch):
                        raise SessionExportDenied()
                    decision = prepare_contract(
                        SessionCreationDecision, found.receipt, redactor=SecretRedactor()
                    )
                    require_exact_contract(
                        decision.target, key.creation_target, redactor=SecretRedactor()
                    )
                    if decision.state != "created":
                        raise SessionExportDenied()
                    target = (decision.session_id, decision.session_instance_id)
                if (
                    (origin.target_session_id, origin.target_session_instance_id) != target
                    or origin.provider_name != "offline"
                    or origin.model != "gpt-5.6"
                    or origin.requester_principal != PRINCIPAL
                    # The runtime authenticates the participant separately;
                    # exposure names its collaboration receiving owner, not a
                    # synthetic OwnerRef made from the participant's public ID.
                    or item.audience != grant[4]
                ):
                    raise SessionExportDenied()
            yield tuple(item.occurrence.payload for item in items)
