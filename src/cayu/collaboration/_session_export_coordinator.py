"""Bounded application owner for exact, source-owned session exports."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, TypeVar
from uuid import uuid4

from cayu.collaboration._contracts import (
    CollaborationConflict,
    CollaborationContractError,
    ContractValue,
    ExactConflict,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
    ExpectedOperation,
    InitiatorBinding,
    ObjectRef,
    OwnerRef,
    snapshot_input,
)
from cayu.collaboration._diagnostics import safe_failure
from cayu.collaboration._ownership import _MutationOwners
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration._session_export_store import (
    NAMESPACE_KEY,
    ROOT_KEY,
    ExportMutation,
    ExportRecord,
    ExportRoot,
    SettlementRecord,
    digest,
    encoded,
    mutation_scope,
    operation_key,
    read_scope,
    source_digest,
)
from cayu.collaboration.exports import (
    ExportLimits,
    SessionExportAcceptance,
    SessionExportAccessContext,
    SessionExportAuthorization,
    SessionExportCapacityExceeded,
    SessionExportConflict,
    SessionExportDenied,
    SessionExportIntent,
    SessionExportNamespace,
    SessionExportReceipt,
    SessionExportRegistration,
    SessionExportRequest,
    SessionExportSettlementReceipt,
    SessionExportSettlementRequest,
    SessionExportUnavailable,
)
from cayu.collaboration.participants import CollaborationCapacityExceeded
from cayu.events import Event, EventType, event_with_runtime_payload_authority
from cayu.sessions.base import Session, SessionOperationPublication, SessionStore
from cayu.vaults.redaction import SecretRedactor

T = TypeVar("T")
V = TypeVar("V", bound=ContractValue)
_RESERVATION_BYTES = 64 * 1024
_EXPORT_ERRORS = (
    SessionExportConflict,
    SessionExportDenied,
    SessionExportCapacityExceeded,
    SessionExportUnavailable,
)


def _safe_export_error(error: BaseException, redactor: SecretRedactor) -> BaseException:
    """Detach typed dependency failures and their bounded diagnostic graph."""
    seen: set[int] = set()

    def copy(value: BaseException, depth: int) -> BaseException | None:
        if id(value) in seen:
            return None
        if len(seen) >= 64 or depth >= 16:
            return RuntimeError("Additional export failure evidence withheld.")
        seen.add(id(value))
        if isinstance(value, BaseExceptionGroup):
            children = []
            for child in value.exceptions:
                detached = copy(child, depth + 1)
                if detached is not None:
                    children.append(detached)
                if len(seen) >= 64:
                    break
            result = (
                BaseExceptionGroup("Session export dependency failures.", children)
                if children
                else RuntimeError("Shared export failure evidence already represented.")
            )
        else:
            kind = next((kind for kind in _EXPORT_ERRORS if isinstance(value, kind)), None)
            result = kind() if kind is not None else safe_failure(value, redactor=redactor)
        if value.__cause__ is not None:
            result.__cause__ = copy(value.__cause__, depth + 1)
        if value.__context__ is not None:
            result.__context__ = copy(value.__context__, depth + 1)
        result.__suppress_context__ = value.__suppress_context__
        return result

    result = copy(error, 0)
    assert result is not None
    return result


class _Output(ContractValue):
    payload_json: str


class SessionExportCoordinator:
    def __init__(
        self,
        *,
        store: SessionStore,
        registration: SessionExportRegistration | None,
        redactor: SecretRedactor,
    ) -> None:
        self.store = store
        self.registration = registration
        self.redactor = redactor
        self.owners = _MutationOwners()
        self.owner: OwnerRef | None = None
        self.limits: ExportLimits | None = None
        self.policy_ref: ObjectRef | None = None
        self.projectors = {}
        self.readers = {}
        if registration is not None:
            if type(registration) is not SessionExportRegistration:
                raise SessionExportDenied()
            self.owner = self.prepare(OwnerRef, registration.owner)
            self.limits = self.prepare(ExportLimits, registration.limits)
            self.policy_ref = self.prepare(ObjectRef, registration.policy.ref)
            if len(registration.projectors) > 64 or len(registration.readers) > 64:
                raise SessionExportCapacityExceeded()
            for projector in registration.projectors:
                ref = self.prepare(ObjectRef, projector.ref)
                if ref in self.projectors:
                    raise SessionExportConflict()
                self.projectors[ref] = projector
            for reader in registration.readers:
                owner = self.prepare(OwnerRef, reader.owner)
                if owner in self.readers:
                    raise SessionExportConflict()
                self.readers[owner] = reader

    def prepare(self, schema: type[V], value: object) -> V:
        return prepare_contract(schema, value, redactor=self.redactor)

    def output(self, value: object) -> dict[str, Any]:
        plain = snapshot_input(value)
        if type(plain) is not dict:
            raise SessionExportDenied()

        def check(item: object) -> None:
            if type(item) is str:
                self.prepare(_Output, {"payload_json": item})
            elif type(item) is dict:
                for key, child in item.items():
                    check(key)
                    check(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    check(child)

        check(plain)
        canonical = encoded(plain)
        if len(canonical) > 8192:
            raise SessionExportCapacityExceeded()
        prepared = self.prepare(_Output, {"payload_json": canonical.decode()})
        return json.loads(prepared.payload_json)

    def ready(self) -> SessionExportRegistration:
        if self.registration is None or not self.store._supports_session_export_protocol():
            raise SessionExportUnavailable()
        return self.registration

    async def observed(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        key: tuple[object, ...],
        expected: bytes,
    ) -> T:
        failure: BaseException | None = None
        try:

            async def run() -> T:
                owner = asyncio.current_task()
                cancellation_baseline = 0 if owner is None else owner.cancelling()
                try:
                    return await operation()
                except asyncio.CancelledError as error:
                    if owner is not None and owner.cancelling() > cancellation_baseline:
                        raise
                    # A cancelled dependency is not caller cancellation. Snapshot
                    # its graph before Task's cancelled state hides the evidence.
                    unavailable = SessionExportUnavailable()
                    unavailable.__cause__ = _safe_export_error(error, self.redactor)
                # Do not attach the original, potentially secret-bearing context.
                raise unavailable

            return await self.owners.run(
                run,
                key=key,
                expectation=expected,
                redactor=self.redactor,
                failure_snapshot=lambda error: _safe_export_error(error, self.redactor),
            )
        except _EXPORT_ERRORS as error:
            failure = _safe_export_error(error, self.redactor)
        except asyncio.CancelledError as error:
            failure = asyncio.CancelledError(
                "Session export observation cancelled; reconcile exact state."
            )
            if error.__cause__ is not None:
                failure.__cause__ = _safe_export_error(error.__cause__, self.redactor)
        except CollaborationConflict:
            failure = SessionExportConflict()
        except CollaborationCapacityExceeded:
            failure = SessionExportCapacityExceeded()
        except BaseExceptionGroup as error:

            def fatal(value):
                if isinstance(value, BaseExceptionGroup):
                    return any(fatal(child) for child in value.exceptions)
                return not isinstance(value, (Exception, asyncio.CancelledError))

            if fatal(error):
                raise
            failure = SessionExportUnavailable()
            failure.__cause__ = _safe_export_error(error, self.redactor)
        except Exception as error:
            failure = SessionExportUnavailable()
            failure.__cause__ = _safe_export_error(error, self.redactor)
        assert failure is not None
        raise failure

    async def close(self) -> None:
        await self.owners.drain()

    async def session(self, session_id: str, instance: str | None = None) -> Session:
        session = await self.store.load(session_id)
        if session is None:
            raise SessionExportUnavailable()
        if instance is not None and session.instance_id != instance:
            raise SessionExportConflict()
        return session

    def auth(
        self,
        raw: object,
        context: SessionExportAccessContext,
    ) -> SessionExportAuthorization:
        result = self.prepare(SessionExportAuthorization, raw)
        if (
            result.principal != context.principal
            or result.policy != self.policy_ref
            or self.policy_ref is None
            or result.issuer != self.policy_ref.owner
        ):
            raise SessionExportDenied()
        return result

    @staticmethod
    def check_time(authorization: SessionExportAuthorization, now: datetime) -> None:
        if (
            now.tzinfo is None
            or int(now.astimezone(UTC).timestamp() * 1000) >= authorization.expires_at_ms
        ):
            raise SessionExportDenied()

    async def check_current_time(
        self, session: Session, authorization: SessionExportAuthorization
    ) -> None:
        def check(current, _checkpoint, now):
            if current.instance_id != session.instance_id:
                raise SessionExportConflict()
            self.check_time(authorization, now)
            return None

        await self.store.transform_checkpoint_with_store_time(session.id, check)

    async def root(self, session: Session) -> ExportRoot | None:
        with read_scope(session.id):
            checkpoint = await self.store.load_checkpoint(session.id)
        if checkpoint is None or ROOT_KEY not in checkpoint:
            return None
        root = self.prepare(ExportRoot, checkpoint[ROOT_KEY])
        if (
            root.namespace.session_id != session.id
            or root.namespace.session_instance_id != session.instance_id
            or root.namespace.owner != self.owner
        ):
            raise SessionExportConflict()
        return root

    async def initialize(
        self,
        session_id: str,
        *,
        context: SessionExportAccessContext,
    ) -> SessionExportNamespace:
        registration = self.ready()
        context = self.prepare(SessionExportAccessContext, context)
        # Prepare identity before creating an owned task; no repr/serialization of arbitrary values.
        session_id = self.prepare(
            OwnerRef,
            {"application_scope": "export", "owner_id": session_id, "incarnation": "input"},
        ).owner_id

        async def operation():
            session = await self.session(session_id)
            async with registration.policy.acquire(
                context,
                session_id=session.id,
                session_instance_id=session.instance_id,
                actions=("initialize",),
                audience=None,
            ) as raw:
                authorization = self.auth(raw, context)
                await self.check_current_time(session, authorization)
                existing = await self.root(session)
                await self.check_current_time(session, authorization)
                if existing is not None:
                    if existing.namespace.limits != self.limits:
                        raise SessionExportConflict()
                    return existing.namespace
                namespace = self.prepare(
                    SessionExportNamespace,
                    {
                        "owner": self.owner,
                        "session_id": session.id,
                        "session_instance_id": session.instance_id,
                        "namespace_incarnation": uuid4().hex,
                        "generation": 1,
                        "limits": self.limits,
                    },
                )
                desired = ExportRoot(
                    namespace=namespace, export_count=0, pending_count=0, retained_bytes=0
                )
                try:
                    await self.publish(
                        session,
                        None,
                        desired,
                        NAMESPACE_KEY,
                        namespace.model_dump(mode="json"),
                        authorization,
                        [],
                    )
                except Exception:
                    reconciled = await self.root(session)
                    await self.check_current_time(session, authorization)
                    if reconciled is None or reconciled.namespace.limits != self.limits:
                        raise
                    return reconciled.namespace
                return namespace

        return await self.observed(
            operation,
            key=("initialize", session_id),
            expected=encoded(
                {
                    "principal": context.principal,
                    "owner": snapshot_input(self.owner),
                    "limits": snapshot_input(self.limits),
                }
            ),
        )

    def validate_namespace(self, root: ExportRoot, request: SessionExportRequest) -> None:
        namespace, operation = root.namespace, request.ref.operation
        if (
            operation.application_scope != namespace.owner.application_scope
            or operation.namespace_incarnation != namespace.namespace_incarnation
            or operation.generation != namespace.generation
        ):
            raise SessionExportConflict()

    def parse_record(self, schema: type[V], raw: object) -> V:
        # A valid sibling operation at this key is a permanent kind conflict.
        # Malformed evidence remains unavailable, never guessed into a kind.
        try:
            return self.prepare(schema, raw)
        except CollaborationContractError:
            sibling = SettlementRecord if schema is ExportRecord else ExportRecord
            self.prepare(sibling, raw)
        raise SessionExportConflict()

    async def record(
        self,
        session: Session,
        request: SessionExportRequest,
        context: SessionExportAccessContext,
        authorization: SessionExportAuthorization,
        *,
        replay: bool = True,
    ) -> ExportRecord | None:
        try:
            result = await self._record(session, request)
            if (
                replay
                and result is not None
                and result.receipt.expected.initiator != self.initiator(authorization)
            ):
                raise SessionExportConflict()
        except SessionExportConflict:
            await self.check_current_time(session, authorization)
            raise
        await self.check_current_time(session, authorization)
        return result

    @staticmethod
    def initiator(authorization: SessionExportAuthorization) -> InitiatorBinding:
        return InitiatorBinding(
            issuer=authorization.issuer,
            principal=authorization.principal,
            participant=None,
            mandate=None,
            invocation_id=None,
            interaction_id=None,
        )

    async def _record(
        self,
        session: Session,
        request: SessionExportRequest,
    ) -> ExportRecord | None:
        root = await self.root(session)
        if root is None:
            raise SessionExportUnavailable()
        self.validate_namespace(root, request)
        with read_scope(session.id):
            raw = await self.store.load_session_operation(
                session.id, operation_key(request.ref.operation)
            )
        if raw is None:
            return None
        record = self.parse_record(ExportRecord, raw)
        expected = record.receipt.expected
        if (
            expected.intent.request != request
            or expected.source != self.owner
            or expected.destination != request.audience
            or expected.kind != "session_export"
            or expected.mode != "deterministic"
            or expected.schema_version != 1
            or expected.receipt_stage != "published"
            or expected.intent.limits != root.namespace.limits
            or record.reserved_bytes != _RESERVATION_BYTES
            or expected.intent.output_commitment != digest(json.loads(record.payload_json))
        ):
            raise SessionExportConflict()
        return record

    async def export(
        self, request: SessionExportRequest, *, context: SessionExportAccessContext
    ) -> SessionExportReceipt:
        registration = self.ready()
        request = self.prepare(SessionExportRequest, request)
        context = self.prepare(SessionExportAccessContext, context)

        async def operation():
            session = await self.session(request.ref.session_id, request.ref.session_instance_id)
            # Historical replay needs current readback, not new-effect permission or old projector.
            async with registration.policy.acquire(
                context,
                session_id=session.id,
                session_instance_id=session.instance_id,
                actions=("readback",),
                audience=request.audience,
            ) as raw:
                authorization = self.auth(raw, context)
                await self.check_current_time(session, authorization)
                existing = await self.record(session, request, context, authorization)
                if existing is not None:
                    return existing.receipt
            if request.policy != self.policy_ref or request.projector not in self.projectors:
                raise SessionExportUnavailable()
            async with registration.policy.acquire(
                context,
                session_id=session.id,
                session_instance_id=session.instance_id,
                actions=("readback", "source", "export"),
                audience=request.audience,
            ) as raw:
                authorization = self.auth(raw, context)
                await self.check_current_time(session, authorization)
                # A peer may have published while this caller waited for the guard.
                # Historical replay must not depend on source or projector availability.
                existing = await self.record(session, request, context, authorization)
                if existing is not None:
                    return existing.receipt
                source = []
                for index in request.source_indices:
                    page = await self.store.load_transcript_window(
                        session.id, start_index=index, limit=1
                    )
                    if not page.records or page.records[0].index != index:
                        raise SessionExportUnavailable()
                    record = page.records[0]
                    # Initial source adapter permits text and structured tool results, never hidden thinking/assets.
                    if any(
                        part.type not in {"text", "tool_result"} for part in record.message.content
                    ):
                        raise SessionExportDenied()
                    source.append(record.model_copy(deep=True))
                source_tuple = tuple(source)
                source_commitment = source_digest(source_tuple)
                if len(encoded([r.model_dump(mode="json") for r in source_tuple])) > 32 * 1024:
                    raise SessionExportCapacityExceeded()
                projector = self.projectors[request.projector]

                def project():
                    # Detach callback values from the retained source commitment and output.
                    selected = tuple(record.model_copy(deep=True) for record in source_tuple)
                    raw_output = projector.project(selected)
                    output = self.output(raw_output)
                    if len(encoded(output)) > 8192:
                        raise SessionExportCapacityExceeded()
                    checked = self.output(output)
                    if (
                        projector.validate(
                            tuple(r.model_copy(deep=True) for r in source_tuple),
                            checked,
                            request.audience,
                        )
                        is not True
                    ):
                        raise SessionExportDenied()
                    if checked != output:
                        raise SessionExportConflict()
                    return output

                output = await asyncio.to_thread(project)
                assert self.owner is not None and self.limits is not None
                expected = self.prepare(
                    ExpectedOperation[SessionExportIntent],
                    {
                        "operation": request.ref.operation,
                        "kind": "session_export",
                        "schema_version": 1,
                        "mode": "deterministic",
                        "source": self.owner,
                        "destination": request.audience,
                        "initiator": InitiatorBinding(
                            issuer=authorization.issuer,
                            principal=context.principal,
                            participant=None,
                            mandate=None,
                            invocation_id=None,
                            interaction_id=None,
                        ),
                        "receipt_stage": "published",
                        "intent": SessionExportIntent(
                            request=request,
                            limits=self.limits,
                            source_commitment=source_commitment,
                            output_commitment=digest(output),
                            authorization=authorization,
                        ),
                    },
                )
                event = self.event(session.id, EventType.SESSION_EXPORT_PUBLISHED, expected)
                receipt = self.prepare(
                    SessionExportReceipt, {"expected": expected, "event_id": event.id}
                )
                record = self.prepare(
                    ExportRecord,
                    {
                        "receipt": receipt,
                        "payload_json": encoded(output).decode(),
                        "reserved_bytes": _RESERVATION_BYTES,
                    },
                )
                self.preflight_settlement(record)
                for _ in range(4):
                    root = await self.root(session)
                    if root is None:
                        raise SessionExportUnavailable()
                    self.validate_namespace(root, request)
                    existing = await self.record(session, request, context, authorization)
                    if existing is not None:
                        return existing.receipt
                    if root.namespace.limits != self.limits:
                        raise SessionExportConflict()
                    if (
                        root.export_count >= self.limits.max_exports
                        or root.pending_count >= self.limits.max_pending
                        or root.retained_bytes + record.reserved_bytes
                        > self.limits.max_retained_bytes
                    ):
                        raise SessionExportCapacityExceeded()
                    desired = root.model_copy(
                        update={
                            "export_count": root.export_count + 1,
                            "pending_count": root.pending_count + 1,
                            "retained_bytes": root.retained_bytes + record.reserved_bytes,
                        }
                    )
                    try:
                        await self.publish(
                            session,
                            root,
                            desired,
                            operation_key(request.ref.operation),
                            record.model_dump(mode="json"),
                            authorization,
                            [event],
                            source_tuple,
                        )
                        return receipt
                    except Exception as error:
                        existing = await self.record(session, request, context, authorization)
                        if existing is not None:
                            return existing.receipt
                        if not isinstance(error, SessionExportConflict):
                            raise
                raise SessionExportConflict()

        return await self.observed(
            operation,
            key=(request.ref.session_id, operation_key(request.ref.operation)),
            expected=encoded({"request": snapshot_input(request), "principal": context.principal}),
        )

    def event(
        self, session_id: str, kind: EventType, expected: ExpectedOperation[SessionExportIntent]
    ) -> Event:
        return event_with_runtime_payload_authority(
            Event(
                type=kind,
                session_id=session_id,
                payload={
                    "export_commitment": digest(snapshot_input(expected)),
                    "output_commitment": expected.intent.output_commitment,
                },
            ),
            "export_commitment",
            "output_commitment",
        )

    def preflight_settlement(self, record: ExportRecord) -> None:
        # Every admitted export reserves its largest mandatory future representation.
        request = record.receipt.expected.intent.request
        operation = request.ref.operation.model_copy(update={"caller_key": "\x01" * 512})
        acceptance = SessionExportAcceptance(
            export_receipt=record.receipt, receiving_owner=request.audience, receipt_id="\x01" * 512
        )
        settlement = SessionExportSettlementReceipt(
            request=SessionExportSettlementRequest(
                request=request, operation=operation, mode="release"
            ),
            initiator=InitiatorBinding(
                issuer=OwnerRef(
                    application_scope="\x01" * 512,
                    owner_id="\x01" * 512,
                    incarnation="\x01" * 512,
                ),
                principal="\x01" * 512,
                participant=None,
                mandate=None,
                invocation_id=None,
                interaction_id=None,
            ),
            acceptance=acceptance,
            event_id="f" * 36,
        )
        # Synthetic bound probes are not business identity and do not use workload secrets.
        future = prepare_contract(
            ExportRecord,
            {**snapshot_input(record), "state": "released", "settlement": settlement},
            redactor=SecretRedactor(),
        )
        encoded(
            {
                operation_key(request.ref.operation): snapshot_input(future),
                operation_key(operation): snapshot_input(SettlementRecord(settlement=settlement)),
            }
        )

    async def publish(
        self,
        session,
        before,
        after,
        key,
        record,
        authorization,
        events,
        source=(),
        additional_records=None,
    ):
        records = {key: record, **(additional_records or {})}
        mutation = ExportMutation(
            session.id,
            session.instance_id,
            None if before is None else encoded(before.model_dump(mode="json")),
            encoded(after.model_dump(mode="json")),
            tuple((k, encoded(v)) for k, v in records.items()),
            tuple(r.index for r in source),
            source_digest(source) if source else None,
            tuple(encoded(e.model_dump(mode="json")) for e in events),
        )

        def transform(current, checkpoint, old, now):
            if current.instance_id != session.instance_id or old is not None:
                raise SessionExportConflict()
            self.check_time(authorization, now)
            updated = {} if checkpoint is None else dict(checkpoint)
            updated[ROOT_KEY] = after.model_dump(mode="json")
            return SessionOperationPublication(checkpoint=updated, operation_records=records)

        with mutation_scope(mutation):
            await self.store.publish_session_operation_guarded_with_store_time(
                session.id,
                idempotency_key=key,
                operation_transform=transform,
                commit_guard=lambda: None,
                commit_time_guard=lambda now: self.check_time(authorization, now),
                events=events,
            )

    async def lookup(
        self,
        request: SessionExportRequest,
        *,
        context: SessionExportAccessContext,
        expose: bool = False,
    ):
        registration = self.ready()
        request = self.prepare(SessionExportRequest, request)
        context = self.prepare(SessionExportAccessContext, context)

        async def operation():
            readback_failure: BaseException | None = None
            outcome: ExactConflict | ExactUnavailable | None = None
            try:
                async with registration.policy.acquire(
                    context,
                    session_id=request.ref.session_id,
                    session_instance_id=request.ref.session_instance_id,
                    actions=("readback", "expose") if expose else ("readback",),
                    audience=request.audience,
                ) as raw:
                    authorization = self.auth(raw, context)
                    record: ExportRecord | None = None
                    try:
                        session = await self.session(
                            request.ref.session_id, request.ref.session_instance_id
                        )
                        record = await self.record(
                            session, request, context, authorization, replay=False
                        )
                    except SessionExportConflict as error:
                        if expose:
                            raise
                        readback_failure = _safe_export_error(error, self.redactor)
                        outcome = ExactConflict()
                    except (SessionExportDenied, asyncio.CancelledError):
                        raise
                    except Exception as error:
                        if expose:
                            raise
                        readback_failure = _safe_export_error(error, self.redactor)
                        outcome = ExactUnavailable()
                    # Keep sanitized evidence in flight through guard cleanup.
                    # Raise outside the handler to avoid attaching the raw error.
                    # Only this exact failure may become a four-way lookup result;
                    # a replacement/aggregate from cleanup remains authoritative.
                    if readback_failure is not None:
                        raise readback_failure
                    if record is None:
                        if expose:
                            raise SessionExportUnavailable()
                        return ExactNotFound()
                    if expose:
                        if record.state == "retired":
                            raise SessionExportUnavailable()
                        return self.output(json.loads(record.payload_json))
                    return ExactMatch[SessionExportReceipt](receipt=record.receipt)
            except BaseException as error:
                if error is not readback_failure:
                    raise
            if outcome is None:
                # A policy guard must not turn a suppressed denial into success.
                raise SessionExportUnavailable()
            return outcome

        return await self.observed(operation, key=("read", uuid4().hex), expected=b"read")

    async def settle(
        self, request: SessionExportSettlementRequest, *, context: SessionExportAccessContext
    ) -> SessionExportSettlementReceipt:
        registration = self.ready()
        request = self.prepare(SessionExportSettlementRequest, request)
        context = self.prepare(SessionExportAccessContext, context)
        original = request.request
        if (
            request.operation == original.ref.operation
            or request.operation.application_scope != original.ref.operation.application_scope
            or request.operation.namespace_incarnation
            != original.ref.operation.namespace_incarnation
            or request.operation.generation != original.ref.operation.generation
        ):
            raise SessionExportConflict()
        key = operation_key(request.operation)

        async def read_settlement(session, authorization):
            root = await self.root(session)
            if root is None:
                raise SessionExportUnavailable()
            self.validate_namespace(root, original)
            with read_scope(session.id):
                raw = await self.store.load_session_operation(session.id, key)
            await self.check_current_time(session, authorization)
            if raw is None:
                return None
            result = self.parse_record(SettlementRecord, raw).settlement
            if result.request != request or result.initiator != self.initiator(authorization):
                raise SessionExportConflict()
            return result

        async def operation():
            session = await self.session(original.ref.session_id, original.ref.session_instance_id)
            async with registration.policy.acquire(
                context,
                session_id=session.id,
                session_instance_id=session.instance_id,
                actions=("readback",),
                audience=original.audience,
            ) as raw:
                authorization = self.auth(raw, context)
                await self.check_current_time(session, authorization)
                replay = await read_settlement(session, authorization)
                if replay is not None:
                    return replay
            async with registration.policy.acquire(
                context,
                session_id=session.id,
                session_instance_id=session.instance_id,
                actions=("readback", request.mode),
                audience=original.audience,
            ) as raw:
                authorization = self.auth(raw, context)
                record = await self.record(session, original, context, authorization, replay=False)
                if record is None:
                    raise SessionExportUnavailable()
                if record.state != "pending":
                    replay = await read_settlement(session, authorization)
                    if replay is not None:
                        return replay
                    raise SessionExportConflict()
                acceptance = None
                if request.mode == "release":
                    reader = self.readers.get(original.audience)
                    if reader is None:
                        raise SessionExportUnavailable()
                    lookup = await reader.lookup(record.receipt)
                    checked = prepare_contract(
                        ExactMatch[SessionExportAcceptance], lookup, redactor=self.redactor
                    )
                    acceptance = checked.receipt
                    if (
                        acceptance.export_receipt != record.receipt
                        or acceptance.receiving_owner != original.audience
                    ):
                        raise SessionExportConflict()
                event = self.event(
                    session.id,
                    EventType.SESSION_EXPORT_RELEASED
                    if request.mode == "release"
                    else EventType.SESSION_EXPORT_RETIRED,
                    record.receipt.expected,
                )
                receipt = self.prepare(
                    SessionExportSettlementReceipt,
                    {
                        "request": request,
                        "initiator": self.initiator(authorization),
                        "acceptance": acceptance,
                        "event_id": event.id,
                    },
                )
                updated = self.prepare(
                    ExportRecord,
                    {
                        **snapshot_input(record),
                        "state": "released" if request.mode == "release" else "retired",
                        "settlement": receipt,
                    },
                )
                settlement = SettlementRecord(settlement=receipt)
                for _ in range(4):
                    root = await self.root(session)
                    replay = await read_settlement(session, authorization)
                    if replay is not None:
                        return replay
                    if root is None or root.pending_count == 0:
                        raise SessionExportConflict()
                    # Recheck exact pending state after every concurrent counter change.
                    current = await self.record(
                        session, original, context, authorization, replay=False
                    )
                    if current != record:
                        replay = await read_settlement(session, authorization)
                        if replay is not None:
                            return replay
                        raise SessionExportConflict()
                    desired = root.model_copy(update={"pending_count": root.pending_count - 1})
                    try:
                        await self.publish(
                            session,
                            root,
                            desired,
                            key,
                            settlement.model_dump(mode="json"),
                            authorization,
                            [event],
                            additional_records={
                                operation_key(original.ref.operation): updated.model_dump(
                                    mode="json"
                                )
                            },
                        )
                        return receipt
                    except Exception as error:
                        replay = await read_settlement(session, authorization)
                        if replay is not None:
                            return replay
                        if not isinstance(error, SessionExportConflict):
                            raise
                raise SessionExportConflict()

        return await self.observed(
            operation,
            key=(original.ref.session_id, key),
            expected=encoded({"request": snapshot_input(request), "principal": context.principal}),
        )
