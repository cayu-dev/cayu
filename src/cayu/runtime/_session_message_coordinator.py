"""Application-owned access and public projection for durable session messages.

The store owns queue state and compare-and-set. This boundary never treats a
public alias, a matching tenant string, or a copied source snapshot as a grant.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from cayu.core.events import Event, event_with_durable_sequence
from cayu.runtime._event_projection import public_event_id
from cayu.runtime._message_redaction import redact_untrusted_message_for_boundary
from cayu.runtime._session_request_boundary import require_secret_free_session_authority
from cayu.runtime.approvals import ResolutionActor, ResolutionActorSource
from cayu.runtime.session_message_lifecycle import (
    SessionMessageAccessContext,
    SessionMessageAccessDenied,
    SessionMessageAccessPolicy,
    SessionMessageActionRequest,
    SessionMessageConflict,
    SessionMessageQuery,
    SessionMessageSource,
)
from cayu.runtime.sessions import (
    EnqueueSessionMessageRequest,
    EnqueueSessionMessageResult,
    EventQuery,
    Session,
    SessionMessageActionResult,
    SessionMessageInspection,
    SessionMessageInspectionRecord,
    SessionQueuedMessage,
    SessionStore,
    copy_enqueue_session_message_request,
)
from cayu.vaults import SecretRedactor


class SessionMessageAuthorizationUnavailable(RuntimeError):
    """The policy could not decide; this is not a negative permission decision."""


class _InvalidEventLinkage(SessionMessageConflict):
    """A completed lookup disproved the expected event linkage, not a read failure."""


class _Enqueue(Protocol):
    def __call__(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        store_resolved_session_id: str | None = None,
        store_resolved_source_session_id: str | None = None,
        expected_authorized_target_instance_id: str | None = None,
    ) -> Awaitable[EnqueueSessionMessageResult]: ...


@dataclass(frozen=True)
class _PreparedEnqueue:
    request: EnqueueSessionMessageRequest
    resolved_target: str | None
    resolved_source: str | None
    expected_authorized_target_instance_id: str | None


class SessionMessageCoordinator:
    def __init__(
        self,
        *,
        store: SessionStore,
        policy: SessionMessageAccessPolicy | None,
        redactor: SecretRedactor,
        resolve_session: Callable[[str], Awaitable[tuple[str, str | None]]],
        project_session: Callable[[str], str],
        project_event: Callable[[Event], Awaitable[Event]],
        enqueue: _Enqueue,
        fan_out: Callable[[list[Event]], Awaitable[list[Event]]],
    ) -> None:
        self._store = store
        self._policy = policy
        self._redactor = redactor
        self._resolve_session = resolve_session
        self._project_session = project_session
        self._project_event = project_event
        self._enqueue = enqueue
        self._fan_out = fan_out

    def _context(self, context: SessionMessageAccessContext | None) -> SessionMessageAccessContext:
        if self._policy is None or type(context) is not SessionMessageAccessContext:
            raise SessionMessageAccessDenied()
        return SessionMessageAccessContext(subject=context.subject, tenant=context.tenant)

    def _require_lifecycle_capability(self, *operations: str) -> None:
        """Require an exact attestation from each operation's implementation owner.

        A subclass may inherit unchanged implementations. Overriding one does
        not inherit its guarantee: the class defining that override must attest
        the complete v1 contract itself, just like queued profile handoffs.
        """
        store_type = type(self._store)
        version = getattr(store_type, "session_message_lifecycle_version", None)
        if type(version) is not int or version != 1:
            raise NotImplementedError(
                "This SessionStore does not attest session-message lifecycle v1."
            )
        for operation in operations:
            owner = next(
                (
                    candidate
                    for candidate in type.__getattribute__(store_type, "__mro__")
                    if operation in type.__getattribute__(candidate, "__dict__")
                ),
                None,
            )
            declared = {} if owner is None else type.__getattribute__(owner, "__dict__")
            owner_version = declared.get("session_message_lifecycle_version")
            method = getattr(self._store, operation, None)
            if (
                type(owner_version) is not int
                or owner_version != 1
                or getattr(method, "__self__", None) is not self._store
                or getattr(method, "__func__", None) is not declared.get(operation)
            ):
                raise NotImplementedError(
                    "This SessionStore operation does not attest session-message lifecycle v1."
                )

    async def _authorize(
        self,
        session_id: str,
        context: SessionMessageAccessContext,
        action: Literal["inspect", "enqueue", "source", "withdraw", "quarantine"],
    ) -> tuple[Session, str | None]:
        context = self._context(context)
        try:
            private_id, resolved = await self._resolve_session(session_id)
            require_secret_free_session_authority(
                # Finding a reserved-looking raw ID is not an alias translation.
                session_id if resolved is None or session_id == private_id else None,
                field_name="session_id",
                redactor=self._redactor,
            )
        except ValueError:
            raise SessionMessageAccessDenied() from None
        # A failed store read is operational failure, not evidence of absence or
        # denial. Preserve it for the trusted SDK caller; HTTP sanitizes it.
        session = await self._store.load(private_id)
        if session is None or self._policy is None:
            raise SessionMessageAccessDenied()
        try:
            allowed = self._policy.authorize(
                context,
                session_id=session.id,
                session_instance_id=session.instance_id,
                action=action,
            )
        except SessionMessageAccessDenied:
            raise
        except Exception as exc:
            raise SessionMessageAuthorizationUnavailable(
                "Session-message authorization could not be evaluated."
            ) from exc
        if allowed is not True:
            raise SessionMessageAccessDenied()
        return session, resolved

    def _authority(self, value: str | None, field_name: str) -> None:
        require_secret_free_session_authority(
            value,
            field_name=field_name,
            redactor=self._redactor,
        )

    def _actor(
        self,
        actor: ResolutionActor | None,
        context: SessionMessageAccessContext,
        source: ResolutionActorSource,
    ) -> ResolutionActor:
        self._authority(context.subject, "context.subject")
        self._authority(context.tenant, "context.tenant")
        if actor is not None:
            # Matching who/tenant never authenticates how, claims, or an actor
            # object. Only the trusted entrance chooses the provenance source.
            raise SessionMessageAccessDenied()
        return ResolutionActor(
            subject=context.subject,
            tenant=context.tenant,
            source=source,
        )

    async def prepare_enqueue(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        context: SessionMessageAccessContext | None = None,
        _actor_source: ResolutionActorSource = ResolutionActorSource.REQUEST,
    ) -> _PreparedEnqueue:
        request = copy_enqueue_session_message_request(request)
        resolved_source = None
        expected_authorized_target_instance_id = None
        if self._policy is not None or request.conditions.source is not None or context is not None:
            context = self._context(context)
            target, resolved = await self._authorize(request.session_id, context, "enqueue")
            expected_authorized_target_instance_id = target.instance_id
            self._authority(expected_authorized_target_instance_id, "session_instance_id")
            self._require_lifecycle_capability("enqueue_session_message")
            request = request.model_copy(
                update={
                    "session_id": target.id,
                    "requested_by": self._actor(request.requested_by, context, _actor_source),
                },
                deep=True,
            )
            source = request.conditions.source
            if source is not None:
                source_session, resolved_source = await self._authorize(
                    source.session_id, context, "source"
                )
                if source.session_instance_id != source_session.instance_id:
                    raise SessionMessageConflict()
                source = source.model_copy(update={"session_id": source_session.id})
                request = request.model_copy(
                    update={
                        "conditions": request.conditions.model_copy(update={"source": source}),
                    },
                    deep=True,
                )
        else:
            session_id, resolved = await self._resolve_session(request.session_id)
            actor = request.requested_by
            if actor is not None:
                actor = ResolutionActor(
                    subject=actor.subject,
                    tenant=actor.tenant,
                    source=_actor_source,
                )
            request = request.model_copy(
                update={"session_id": session_id, "requested_by": actor}, deep=True
            )
        if (
            request.conditions.source is not None
            or request.conditions.target is not None
            or request.conditions.expires_at is not None
        ):
            # Admission must not accept a promise that the later delivery owner
            # can silently ignore. Check both owners before the first write.
            self._require_lifecycle_capability(
                "enqueue_session_message",
                "deliver_queued_session_messages",
            )
        return _PreparedEnqueue(
            request, resolved, resolved_source, expected_authorized_target_instance_id
        )

    async def enqueue(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        context: SessionMessageAccessContext | None = None,
    ) -> EnqueueSessionMessageResult:
        return await self._enqueue_with_source(request, context, ResolutionActorSource.REQUEST)

    async def enqueue_from_http(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        context: SessionMessageAccessContext,
    ) -> EnqueueSessionMessageResult:
        """Server-only entrance: actor provenance is not a request field."""
        return await self._enqueue_with_source(
            request, self._context(context), ResolutionActorSource.HTTP_AUTH
        )

    async def enqueue_from_scenario(
        self, request: EnqueueSessionMessageRequest
    ) -> EnqueueSessionMessageResult:
        """Runtime-only actor derivation; scenario data grants no scoped access."""
        request = copy_enqueue_session_message_request(request)
        if request.requested_by is not None:
            raise SessionMessageAccessDenied()
        request = request.model_copy(
            update={
                "requested_by": ResolutionActor(
                    subject="cayu:eval-scenario", source=ResolutionActorSource.SYSTEM
                )
            },
            deep=True,
        )
        return await self._enqueue_with_source(request, None, ResolutionActorSource.SYSTEM)

    async def _enqueue_with_source(
        self,
        request: EnqueueSessionMessageRequest,
        context: SessionMessageAccessContext | None,
        source: ResolutionActorSource,
    ) -> EnqueueSessionMessageResult:
        prepared = await self.prepare_enqueue(request, context=context, _actor_source=source)
        request = prepared.request
        result = await self._enqueue(
            request,
            store_resolved_session_id=prepared.resolved_target,
            store_resolved_source_session_id=prepared.resolved_source,
            expected_authorized_target_instance_id=prepared.expected_authorized_target_instance_id,
        )
        event = await self._project_lifecycle_event(result.event)
        accepted_event_id = await self._project_event_reference(
            request.session_id,
            result.message.queue_id,
            result.message.accepted_event_id,
            "session.message.queued",
        )
        if accepted_event_id != event.id:
            raise SessionMessageConflict()
        # Acceptance replay returns the actual current message status, not a
        # fabricated inspection record requiring a terminal receipt it lacks.
        message = await self._project_message(
            result.message, session_id=request.session_id, accepted_event_id=accepted_event_id
        )
        return result.model_copy(
            update={"event": event, "message": message},
            deep=True,
        )

    async def _public_session(self, value: str) -> str:
        public = self._project_session(value)
        if public != value:
            await self._store.register_public_authority_alias(
                public,
                field_name="session_id",
                private_value=value,
            )
        return public

    async def _project_source(self, source: SessionMessageSource) -> SessionMessageSource:
        for name in ("session_instance_id", "transcript_sha256", "checkpoint_sha256"):
            self._authority(getattr(source, name), name)
        return source.model_copy(
            update={"session_id": await self._public_session(source.session_id)}
        )

    async def _event_linkage(
        self, session_id: str, queue_id: str, event_id: str, event_type: str
    ) -> tuple[int, Event]:
        records = await self._store.query_events(
            EventQuery(session_id=session_id, event_id=event_id, limit=2)
        )
        # Side-effect receipts bind only session/event ID and sequence, not
        # queue identity or event kind. They cannot authenticate this linkage.
        if len(records) != 1:
            raise _InvalidEventLinkage()
        record = records[0]
        if (
            record.event.session_id != session_id
            or record.event.id != event_id
            or record.event.type != event_type
            or record.event.payload.get("queue_id") != queue_id
        ):
            raise _InvalidEventLinkage()
        return record.sequence, record.event

    async def _project_lifecycle_event(self, event: Event) -> Event:
        sequence, retained = await self._event_linkage(
            event.session_id, event.payload["queue_id"], event.id, str(event.type)
        )
        return await self._project_event(event_with_durable_sequence(retained, sequence))

    async def _project_event_reference(
        self, session_id: str, queue_id: str, event_id: str | None, event_type: str
    ) -> str | None:
        if event_id is None:
            return None
        sequence, _ = await self._event_linkage(session_id, queue_id, event_id, event_type)
        return public_event_id(sequence)

    async def _project_record(
        self, record: SessionMessageInspectionRecord, *, session_id: str
    ) -> SessionMessageInspectionRecord:
        for name in ("queue_id", "revision"):
            self._authority(getattr(record, name), name)
        if record.status in {"delivered", "withdrawn", "quarantined", "stale", "expired"} and (
            record.terminal_event_id is None
        ):
            raise SessionMessageConflict()
        terminal_event_id = await self._project_event_reference(
            session_id,
            record.queue_id,
            record.terminal_event_id,
            f"session.message.{record.status}",
        )
        if record.validity == "unreadable":
            return record.model_copy(
                update={"message": None, "terminal_event_id": terminal_event_id}, deep=True
            )
        message = record.message
        if message is None:
            raise SessionMessageConflict()
        try:
            accepted_event_id = await self._project_event_reference(
                session_id, record.queue_id, message.accepted_event_id, "session.message.queued"
            )
        except _InvalidEventLinkage:
            # Keep the exact raw revision and damaged pointer in storage. Only
            # the public representation becomes unreadable; terminal linkage
            # was positively verified above and operational errors escape.
            return record.model_copy(
                update={
                    "validity": "unreadable",
                    "message": None,
                    "terminal_event_id": terminal_event_id,
                },
                deep=True,
            )
        projected = await self._project_message(
            message, session_id=session_id, accepted_event_id=accepted_event_id
        )
        return record.model_copy(
            update={"message": projected, "terminal_event_id": terminal_event_id}, deep=True
        )

    async def _project_message(
        self,
        message: SessionQueuedMessage,
        *,
        session_id: str,
        accepted_event_id: str | None,
    ) -> SessionQueuedMessage:
        """Project a message with verified acceptance, independently of terminal receipts."""
        self._authority(message.queue_id, "queue_id")
        if accepted_event_id is None or message.session_id != session_id:
            raise SessionMessageConflict()
        if message.status == "delivered" and message.delivered_event_id is None:
            raise SessionMessageConflict()
        conditions = message.conditions
        if conditions.source is not None:
            conditions = conditions.model_copy(
                update={"source": await self._project_source(conditions.source)}
            )
        if conditions.target is not None:
            self._authority(conditions.target.session_instance_id, "target.session_instance_id")
        self._authority(message.idempotency_key, "idempotency_key")
        delivered_event_id = await self._project_event_reference(
            session_id, message.queue_id, message.delivered_event_id, "session.message.delivered"
        )
        actor = message.requested_by
        if actor is not None:
            # Actor identity remains exact. Do not turn a secret into a different actor.
            self._authority(actor.subject, "requested_by.subject")
            self._authority(actor.tenant, "requested_by.tenant")
            actor = actor.model_copy(update={"claims": {}}, deep=True)
        return message.model_copy(
            update={
                "session_id": await self._public_session(message.session_id),
                "content": self._redactor.redact_text(message.content),
                "message": None
                if message.message is None
                else redact_untrusted_message_for_boundary(
                    message.message,
                    redactor=self._redactor,
                    field_name="message",
                ),
                "conditions": conditions,
                "requested_by": actor,
                "accepted_event_id": accepted_event_id,
                "delivered_event_id": delivered_event_id,
            },
            deep=True,
        )

    async def inspect(
        self,
        query: SessionMessageQuery,
        *,
        context: SessionMessageAccessContext,
    ) -> SessionMessageInspection:
        if type(query) is not SessionMessageQuery:
            raise TypeError("query must be SessionMessageQuery.")
        query = SessionMessageQuery(
            session_id=query.session_id,
            cursor=query.cursor,
            limit=query.limit,
        )
        session, _ = await self._authorize(query.session_id, context, "inspect")
        if query.cursor is not None:
            self._authority(query.cursor.session_instance_id, "cursor.session_instance_id")
            if query.cursor.session_instance_id != session.instance_id:
                raise SessionMessageConflict()
        self._require_lifecycle_capability("inspect_session_messages")
        self._authority(session.instance_id, "session_instance_id")
        result = await self._store.inspect_session_messages(
            query.model_copy(update={"session_id": session.id}),
            expected_authorized_session_instance_id=session.instance_id,
        )
        if result.session_instance_id != session.instance_id or result.session_id != session.id:
            raise SessionMessageAccessDenied()
        self._authority(result.session_instance_id, "session_instance_id")
        records = tuple(
            [await self._project_record(record, session_id=session.id) for record in result.records]
        )
        return result.model_copy(
            update={
                "session_id": await self._public_session(result.session_id),
                "records": records,
            },
            deep=True,
        )

    async def apply_action(
        self,
        request: SessionMessageActionRequest,
        *,
        context: SessionMessageAccessContext,
    ) -> SessionMessageActionResult:
        return await self._apply_action(request, context, ResolutionActorSource.REQUEST)

    async def apply_action_from_http(
        self,
        request: SessionMessageActionRequest,
        *,
        context: SessionMessageAccessContext,
    ) -> SessionMessageActionResult:
        """Server-only entrance; the body cannot supply the actor or its source."""
        return await self._apply_action(request, context, ResolutionActorSource.HTTP_AUTH)

    async def _apply_action(
        self,
        request: SessionMessageActionRequest,
        context: SessionMessageAccessContext,
        source: ResolutionActorSource,
    ) -> SessionMessageActionResult:
        if type(request) is not SessionMessageActionRequest:
            raise TypeError("request must be SessionMessageActionRequest.")
        request = SessionMessageActionRequest(
            **{name: getattr(request, name) for name in SessionMessageActionRequest.model_fields}
        )
        context = self._context(context)
        session, _ = await self._authorize(request.session_id, context, request.action)
        self._require_lifecycle_capability("apply_session_message_action")
        if request.session_instance_id != session.instance_id:
            raise SessionMessageConflict()
        for name in ("session_instance_id", "queue_id", "idempotency_key", "expected_revision"):
            self._authority(getattr(request, name), name)
        request = request.model_copy(
            update={
                "session_id": session.id,
                "requested_by": self._actor(request.requested_by, context, source),
            },
            deep=True,
        )
        result = await self._store.apply_session_message_action(request)
        if not result.replayed:
            await self._fan_out([result.event])
        return result.model_copy(
            update={
                "event": await self._project_lifecycle_event(result.event),
                "record": await self._project_record(result.record, session_id=session.id),
            },
            deep=True,
        )

    async def snapshot_source(
        self,
        session_id: str,
        *,
        context: SessionMessageAccessContext,
        include_transcript_digest: bool = False,
        include_checkpoint_digest: bool = False,
    ) -> SessionMessageSource:
        if (
            type(include_transcript_digest) is not bool
            or type(include_checkpoint_digest) is not bool
        ):
            raise TypeError("Snapshot digest selectors must be bools.")
        session, _ = await self._authorize(session_id, context, "source")
        self._require_lifecycle_capability("snapshot_session_message_source")
        self._authority(session.instance_id, "session_instance_id")
        result = await self._store.snapshot_session_message_source(
            session.id,
            expected_authorized_session_instance_id=session.instance_id,
            include_transcript_digest=include_transcript_digest,
            include_checkpoint_digest=include_checkpoint_digest,
        )
        if result.session_id != session.id or result.session_instance_id != session.instance_id:
            raise SessionMessageAccessDenied()
        return await self._project_source(result)
