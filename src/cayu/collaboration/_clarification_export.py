"""Clarification projection under the existing registered session-export owner."""

from __future__ import annotations

import json
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

from cayu.collaboration._contracts import ObjectRef, OwnerRef
from cayu.collaboration._preparation import contract_bytes
from cayu.collaboration.clarifications import ClarificationSource
from cayu.collaboration.exports import (
    SessionExportAccessContext,
    SessionExportAuthorization,
    SessionExportConflict,
    SessionExportDenied,
    SessionExportReceipt,
    SessionExportRequest,
    SessionExportUnavailable,
)
from cayu.collaboration.participants import ParticipantRef

if TYPE_CHECKING:
    from cayu.collaboration._session_export_coordinator import SessionExportCoordinator


@dataclass(frozen=True, slots=True)
class ClarificationSourceProjection:
    """Guard-local projection, not transferable or caller-authored authorization."""

    source: ClarificationSource
    receipt: SessionExportReceipt
    text: str
    expires_at_ms: int
    authorization: SessionExportAuthorization


@asynccontextmanager
async def acquire_clarification_source(
    owner: SessionExportCoordinator,
    request: SessionExportRequest,
    *,
    context: SessionExportAccessContext,
    sender: ParticipantRef,
    audience: ParticipantRef,
    expected: ClarificationSource | None = None,
):
    """Hold fresh source disclosure until the clarification owner finishes use.

    This does not register a new export or invoke a projector. The registered
    export owner reconstructs an existing content-bound publication and checks
    current policy/mandate authority. Callers must additionally check this
    guard's expiry with their mutation owner's authoritative transaction clock.
    """
    owner.ready(access="readback")
    request = owner.prepare(SessionExportRequest, request)
    context = owner.prepare(SessionExportAccessContext, context)
    sender = owner.prepare(ParticipantRef, sender)
    audience = owner.prepare(ParticipantRef, audience)
    if expected is not None:
        expected = owner.prepare(ClarificationSource, expected)
    if request.audience != OwnerRef(
        application_scope=audience.owner.application_scope,
        owner_id=audience.participant_id,
        incarnation=audience.incarnation,
    ):
        raise SessionExportDenied()
    async with (
        owner.acquire(
            context,
            session_id=request.ref.session_id,
            session_instance_id=request.ref.session_instance_id,
            actions=("readback", "expose"),
            audience=request.audience,
            request=request,
        ) as raw,
        AsyncExitStack() as release_guard,
    ):
        authorization = owner.auth(raw, context)
        session = await owner.session(request.ref.session_id, request.ref.session_instance_id)
        binding = await owner.store.load_participant_session_binding(session.id)
        creation = await owner.store.load_participant_session_creation_receipt(session.id)
        if (
            binding is None
            or binding.session_id != session.id
            or binding.session_instance_id != session.instance_id
            or binding.participant != sender
            or creation is None
            or creation.binding != binding
        ):
            raise SessionExportDenied()
        record = await owner.record(session, request, context, authorization, replay=False)
        if record is None or record.state == "retired":
            raise SessionExportUnavailable()
        output = owner.output(json.loads(record.payload_json))
        deadlines = [authorization.expires_at_ms]
        if authorization.mandate is not None:
            deadlines.extend(
                (
                    authorization.mandate.principal.expires_at_ms,
                    *(entry.expires_at_ms for entry in authorization.mandate.chain.entries),
                )
            )
        if request.mode == "reviewed_prose":
            released, release = await owner.released_output(
                request, record.receipt.expected.intent.source_commitment, release_guard
            )
            if released != output or release != record.receipt.expected.intent.release_receipt:
                raise SessionExportConflict()
            await owner.check_current_time(session, authorization, release)
            deadlines.append(release.expires_at_ms)
        if set(output) != {"text"} or type(output["text"]) is not str or not output["text"]:
            raise SessionExportDenied()
        text = output["text"]
        encoded = text.encode("utf-8")
        receipt = record.receipt
        source = owner.prepare(
            ClarificationSource,
            {
                "export": request.ref,
                "export_receipt_sha256": sha256(
                    contract_bytes(receipt, redactor=owner.redactor)
                ).hexdigest(),
                "producer": ObjectRef(
                    owner=receipt.expected.source,
                    kind="session_export",
                    object_id=receipt.event_id,
                    incarnation=request.ref.session_instance_id,
                    revision=1,
                ),
                "content_sha256": sha256(encoded).hexdigest(),
                "content_bytes": len(encoded),
                "selection": request.source_selection,
                "projector": request.projector,
                "policy": request.policy,
                "audience": ObjectRef(
                    owner=audience.owner,
                    kind="participant",
                    object_id=audience.participant_id,
                    incarnation=audience.incarnation,
                ),
            },
        )
        if expected is not None and expected != source:
            raise SessionExportConflict()
        await owner.check_current_time(session, authorization)
        yield ClarificationSourceProjection(
            source=source,
            receipt=receipt,
            text=text,
            expires_at_ms=min(deadlines),
            authorization=authorization,
        )
