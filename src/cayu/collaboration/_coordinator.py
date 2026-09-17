"""Authenticated SDK identity administration; never an execution coordinator."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Annotated, TypeVar

from pydantic import Field, StrictBool, StrictInt

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.collaboration._capabilities import FamilyVersion, require_capability
from cayu.collaboration._contracts import (
    MAX_DEPTH,
    MAX_ENVELOPE_BYTES,
    MAX_NODES,
    CollaborationConflict,
    CollaborationContractError,
    ContractValue,
    ExactConflict,
    ExactLookup,
    ExactMatch,
    ExpectedOperation,
    Generation,
    InitiatorBinding,
    snapshot_input,
)
from cayu.collaboration._diagnostics import safe_cancelled_group_failure, safe_failure
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration.access import (
    CollaborationAccessContext,
    CollaborationAccessDenied,
    CollaborationAccessGrant,
    CollaborationAccessPolicy,
    CollaborationRegistration,
    ParticipantAction,
)
from cayu.collaboration.base import IDENTITY_FAMILY, LIFECYCLE_FAMILY, CollaborationStore
from cayu.collaboration.lifecycle import (
    CollaborationHistoryUnavailable,
    CollaborationNamespaceRetired,
    LifecycleCommand,
    LifecycleReceipt,
    NamespaceInspection,
    NamespaceRef,
    NamespaceRetirementEvidence,
    ParticipantLifecycleChange,
)
from cayu.collaboration.obligations import (
    ParticipantObligation,
    ParticipantObligationCursor,
    ParticipantObligationPage,
)
from cayu.collaboration.participants import (
    CollaborationBootstrap,
    CollaborationCapacityExceeded,
    CollaborationInitialization,
    CollaborationNotInitialized,
    CollaborationUnavailable,
    ParticipantAlias,
    ParticipantAliasChange,
    ParticipantCommand,
    ParticipantConfiguration,
    ParticipantConfigure,
    ParticipantCreate,
    ParticipantCursor,
    ParticipantEvent,
    ParticipantEventCursor,
    ParticipantEventPage,
    ParticipantInspection,
    ParticipantIntent,
    ParticipantPage,
    ParticipantReceipt,
    ParticipantRef,
    ParticipantSnapshot,
)
from cayu.vaults.redaction import SecretRedactor

T = TypeVar("T", bound=ContractValue)
ResultT = TypeVar("ResultT")
RecordT = TypeVar("RecordT", bound=ContractValue)


class _Readback(ContractValue):
    result: ExactLookup[ParticipantReceipt]


class _LifecycleReadback(ContractValue):
    result: ExactLookup[LifecycleReceipt]


class _ObligationQuery(ContractValue):
    participant: ParticipantRef
    cursor: ParticipantObligationCursor | None
    pending_only: StrictBool
    limit: Annotated[StrictInt, Field(ge=1, le=64)]


class _RetentionRevision(ContractValue):
    revision: Generation


class ParticipantCoordinator:
    def __init__(
        self,
        *,
        store: CollaborationStore | None,
        registration: CollaborationRegistration | None,
        redactor: SecretRedactor,
    ) -> None:
        self._store = store
        self._registration = registration
        self._redactor = redactor
        self._initialized: CollaborationInitialization | None = None
        if (store is None) != (registration is None):
            raise TypeError("Collaboration requires both a store and registration.")
        if registration is not None:
            if type(registration) is not CollaborationRegistration or not isinstance(
                registration.access_policy, CollaborationAccessPolicy
            ):
                raise TypeError("Collaboration requires a trusted registration and access policy.")
            self._binding = self._prepare(CollaborationBootstrap, registration.bootstrap)
            if (
                type(registration.configurations) is not tuple
                or len(registration.configurations) > 64
            ):
                raise ValueError("Collaboration configuration registry must be bounded.")
            self._configurations = tuple(
                self._prepare(ParticipantConfiguration, c) for c in registration.configurations
            )
            # The same versioned name must never denote conflicting configuration.
            if len({contract_bytes(c, redactor=redactor) for c in self._configurations}) != len(
                self._configurations
            ):
                raise ValueError("Duplicate participant configuration registration.")

    def _prepare(self, schema: type[T], value: object) -> T:
        return prepare_contract(schema, value, redactor=self._redactor)

    async def _store_result(self, operation: Awaitable[ResultT]) -> ResultT:
        """Detach extension diagnostics; loss of acknowledgement is not rejection."""
        task = asyncio.current_task()
        failure: BaseException
        diagnostic: BaseException | None = None
        try:
            return await operation
        except asyncio.CancelledError as error:
            if error.__cause__ is not None:
                diagnostic = self._safe_failure(error.__cause__)
            failure = (
                asyncio.CancelledError("Collaboration observation cancelled.")
                if task is not None and task.cancelling()
                else CollaborationUnavailable(
                    "Collaboration dependency cancelled; exact readback is required."
                )
            )
        except CollaborationConflict:
            failure = CollaborationConflict("Collaboration authority or intent conflicts.")
        except CollaborationCapacityExceeded:
            failure = CollaborationCapacityExceeded(
                "Collaboration admission capacity is exhausted."
            )
        except CollaborationContractError:
            failure = CollaborationContractError("Invalid collaboration owner evidence.")
        except CollaborationNotInitialized:
            failure = CollaborationNotInitialized("Collaboration owner is not initialized.")
        except CollaborationNamespaceRetired:
            failure = CollaborationNamespaceRetired("Namespace no longer admits operations.")
        except CollaborationHistoryUnavailable:
            failure = CollaborationHistoryUnavailable("Requested history is no longer retained.")
        except CollaborationUnavailable:
            failure = CollaborationUnavailable(
                "Collaboration acknowledgement is unavailable; exact readback is required."
            )
        except BaseExceptionGroup as error:
            fatal = error.subgroup(
                lambda value: (
                    not isinstance(value, (Exception, asyncio.CancelledError, BaseExceptionGroup))
                )
            )
            if fatal is not None:
                raise
            cancellations = error.subgroup(asyncio.CancelledError)
            if cancellations is not None and task is not None and task.cancelling():
                failure = asyncio.CancelledError("Collaboration observation cancelled.")
                diagnostic = safe_cancelled_group_failure(error, redactor=self._redactor)
            else:
                failure = CollaborationUnavailable(
                    "Collaboration dependency failed; exact readback is required."
                )
                diagnostic = self._safe_failure(error)
        except Exception as error:
            failure = CollaborationUnavailable(
                "Collaboration acknowledgement is unavailable; exact readback is required."
            )
            diagnostic = self._safe_failure(error)
        # Neither cause nor context may retain raw store/driver material.
        raise failure from diagnostic

    def _safe_failure(self, error: BaseException) -> BaseException:
        return safe_failure(error, redactor=self._redactor)

    def _page(
        self,
        schema: type[T],
        field: str,
        records: tuple[RecordT, ...],
        cursor: Callable[[RecordT], dict],
        limit: int,
        *,
        evidence: dict | None = None,
    ) -> T:
        # Every fetched record has already passed schema, secret, authority and
        # ordering checks. Only aggregate bounds may shorten this trusted page.
        plain = [snapshot_input(record) for record in records]
        for count in range(len(records), -1, -1):
            if records and count == 0:
                break
            continuation = (
                cursor(records[count - 1])
                if count and (count < len(records) or len(records) == limit)
                else None
            )
            value = {**(evidence or {}), field: plain[:count], "next_cursor": continuation}
            try:
                canonical_bounded_durable_json_bytes(
                    value,
                    "participant page",
                    max_bytes=MAX_ENVELOPE_BYTES,
                    max_nodes=MAX_NODES,
                    max_nesting=MAX_DEPTH,
                )
            except ValueError:
                continue
            return self._prepare(schema, value)
        raise CollaborationUnavailable("Participant record and cursor exceed page capacity.")

    def _ready(self) -> tuple[CollaborationStore, CollaborationInitialization]:
        if self._store is None or self._initialized is None:
            raise CollaborationNotInitialized(
                "Call initialize_collaboration before participant operations."
            )
        return self._store, self._initialized

    def _capability(
        self,
        store: CollaborationStore,
        initialized: CollaborationInitialization,
        *,
        mutation: bool,
        family: FamilyVersion = IDENTITY_FAMILY,
    ) -> None:
        require_capability(
            store.capabilities(initialized.owner),
            expected_owner=initialized.owner,
            required=family,
            supported=(IDENTITY_FAMILY, LIFECYCLE_FAMILY),
            access="mutation" if mutation else "readback",
            redactor=self._redactor,
        )

    async def initialize(self) -> CollaborationInitialization:
        store = self._store
        if store is None or self._registration is None:
            raise CollaborationNotInitialized("No collaboration registration is configured.")
        if type(store.identity_contract_version) is not int or store.identity_contract_version != 1:
            raise CollaborationUnavailable("Collaboration identity contract is unsupported.")
        result = self._prepare(
            CollaborationInitialization,
            await self._store_result(store.initialize(self._binding, redactor=self._redactor)),
        )
        if (
            result.binding != self._binding
            or result.owner.application_scope != self._binding.application_scope
            or result.owner.owner_id != self._binding.owner_name
        ):
            raise CollaborationUnavailable(
                "Initialization evidence conflicts with configured ownership."
            )
        if self._initialized is not None and self._initialized != result:
            raise CollaborationUnavailable("Initialization changed its durable authority.")
        self._initialized = result
        return result

    async def inspect_namespace(
        self, *, context: CollaborationAccessContext
    ) -> NamespaceInspection:
        store, initialized = self._ready()
        _, grant = self._authorize(context, "namespace_inspect")
        self._require_refs(grant, (), create=True)
        self._capability(store, initialized, mutation=False, family=LIFECYCLE_FAMILY)
        result = self._prepare(
            NamespaceInspection,
            await self._store_result(store.inspect_namespace(initialized, redactor=self._redactor)),
        )
        if (
            result.current.reference.owner != initialized.owner
            or result.current.reference.namespace_incarnation != initialized.namespace_incarnation
        ):
            raise CollaborationUnavailable("Namespace inspection authority conflicts.")
        return result

    async def inspect_retirement(
        self,
        namespace: NamespaceRef,
        *,
        context: CollaborationAccessContext,
    ) -> NamespaceRetirementEvidence | None:
        store, initialized = self._ready()
        namespace = self._prepare(NamespaceRef, namespace)
        _, grant = self._authorize(context, "namespace_inspect")
        self._require_refs(grant, (), create=True)
        self._capability(store, initialized, mutation=False, family=LIFECYCLE_FAMILY)
        if (
            namespace.owner != initialized.owner
            or namespace.namespace_incarnation != initialized.namespace_incarnation
        ):
            raise CollaborationConflict("Retirement query belongs to another namespace owner.")
        result = await self._store_result(
            store.inspect_retirement(initialized, namespace, redactor=self._redactor)
        )
        if result is None:
            return None
        result = self._prepare(NamespaceRetirementEvidence, result)
        if result.namespace != namespace:
            raise CollaborationUnavailable("Retirement evidence belongs to another generation.")
        return result

    async def obligations(
        self,
        participant: ParticipantRef,
        *,
        context: CollaborationAccessContext,
        cursor: ParticipantObligationCursor | None = None,
        pending_only: bool = True,
        limit: int = 32,
    ) -> ParticipantObligationPage:
        store, initialized = self._ready()
        query = self._prepare(
            _ObligationQuery,
            {
                "participant": participant,
                "cursor": cursor,
                "pending_only": pending_only,
                "limit": limit,
            },
        )
        context, grant = self._authorize(context, "obligations")
        self._require_refs(grant, (query.participant,))
        if query.participant.owner != initialized.owner:
            raise CollaborationConflict("Obligation query belongs to another owner.")
        cursor = query.cursor
        if cursor is not None and (
            cursor.scope != initialized.binding.application_scope
            or cursor.principal != context.principal
            or cursor.participant != query.participant
            or cursor.pending_only != query.pending_only
        ):
            raise CollaborationConflict("Obligation cursor conflicts with its authorized query.")
        self._capability(store, initialized, mutation=False, family=LIFECYCLE_FAMILY)
        raw = await self._store_result(
            store.scan_obligations(
                initialized,
                query.participant,
                after=0 if cursor is None else cursor.after_position,
                limit=query.limit,
                pending_only=query.pending_only,
                retention_revision=None if cursor is None else cursor.retention_revision,
                redactor=self._redactor,
            )
        )
        if (
            type(raw) is not tuple
            or len(raw) != 2
            or type(raw[1]) is not tuple
            or len(raw[1]) > query.limit
        ):
            raise CollaborationUnavailable("Invalid bounded obligation query response.")
        revision = self._prepare(_RetentionRevision, {"revision": raw[0]}).revision
        if cursor is not None and revision != cursor.retention_revision:
            raise CollaborationHistoryUnavailable("Obligation cursor predates retained history.")
        values = tuple(self._prepare(ParticipantObligation, value) for value in raw[1])
        previous = 0 if cursor is None else cursor.after_position
        for value in values:
            if (
                value.participant != query.participant
                or value.position <= previous
                or (query.pending_only and value.state != "pending")
            ):
                raise CollaborationUnavailable("Obligation query evidence conflicts.")
            previous = value.position
        return self._page(
            ParticipantObligationPage,
            "obligations",
            values,
            lambda value: {
                "scope": initialized.binding.application_scope,
                "principal": context.principal,
                "participant": snapshot_input(query.participant),
                "pending_only": query.pending_only,
                "retention_revision": revision,
                "after_position": value.position,
            },
            query.limit,
        )

    async def mutate_lifecycle(self, schema, request, *, context):
        store, initialized = self._ready()
        request = self._prepare(schema, request)
        context, read_grant = self._authorize(context, "readback")
        refs = (request.participant,) if isinstance(request, ParticipantLifecycleChange) else ()
        scope_control = not isinstance(request, ParticipantLifecycleChange)
        self._require_refs(read_grant, refs, create=scope_control)
        expected = self._prepare(
            LifecycleCommand,
            {
                "operation": request.operation,
                "kind": request.kind,
                "source": initialized.owner,
                "destination": initialized.owner,
                "initiator": {
                    "issuer": initialized.owner,
                    "principal": context.principal,
                    "participant": None,
                    "mandate": None,
                    "invocation_id": None,
                    "interaction_id": None,
                },
                "intent": {"request": request, "limits": initialized.binding.limits},
            },
        )
        self._capability(store, initialized, mutation=False, family=LIFECYCLE_FAMILY)
        replay = self._prepare(
            _LifecycleReadback,
            {
                "result": await self._store_result(
                    store.lookup_lifecycle(initialized, expected, redactor=self._redactor)
                )
            },
        ).result
        if isinstance(replay, ExactMatch):
            return self._lifecycle_receipt(expected, replay.receipt)
        if isinstance(replay, ExactConflict):
            raise CollaborationConflict("Lifecycle operation conflicts with retained intent.")
        if replay.status != "not_found":
            raise CollaborationUnavailable("Lifecycle exact readback is unavailable.")
        _, grant = self._authorize(context, request.kind)
        self._require_refs(grant, refs, create=scope_control)
        self._capability(store, initialized, mutation=True, family=LIFECYCLE_FAMILY)
        return self._lifecycle_receipt(
            expected,
            await self._store_result(
                store.apply_lifecycle(initialized, expected, redactor=self._redactor)
            ),
        )

    def _lifecycle_receipt(self, expected: LifecycleCommand, raw: object) -> LifecycleReceipt:
        receipt = self._prepare(LifecycleReceipt, raw)
        if receipt.expected != expected:
            raise CollaborationUnavailable("Lifecycle receipt conflicts with expected operation.")
        return receipt

    async def lookup_lifecycle(
        self, expected: LifecycleCommand, *, context: CollaborationAccessContext
    ) -> ExactLookup[LifecycleReceipt]:
        store, initialized = self._ready()
        expected = self._prepare(LifecycleCommand, expected)
        _, grant = self._authorize(context, "readback")
        request = expected.intent.request
        participant_control = isinstance(request, ParticipantLifecycleChange)
        refs = (request.participant,) if isinstance(request, ParticipantLifecycleChange) else ()
        self._require_refs(grant, refs, create=not participant_control)
        self._capability(store, initialized, mutation=False, family=LIFECYCLE_FAMILY)
        result = self._prepare(
            _LifecycleReadback,
            {
                "result": await self._store_result(
                    store.lookup_lifecycle(initialized, expected, redactor=self._redactor)
                )
            },
        ).result
        if isinstance(result, ExactMatch):
            self._lifecycle_receipt(expected, result.receipt)
        return result

    def _authorize(
        self, context: CollaborationAccessContext, action: ParticipantAction
    ) -> tuple[CollaborationAccessContext, CollaborationAccessGrant]:
        _, initialized = self._ready()
        if type(context) is not CollaborationAccessContext:
            raise CollaborationAccessDenied("A trusted application context is required.")
        context = self._prepare(CollaborationAccessContext, context)
        assert self._registration is not None
        failed = False
        try:
            raw = self._registration.access_policy.authorize(
                context, application_scope=initialized.binding.application_scope, action=action
            )
            grant = self._prepare(CollaborationAccessGrant, raw)
        except Exception:
            failed = True
        if failed:
            raise CollaborationAccessDenied("Participant access was not authorized.")
        if grant.application_scope != initialized.binding.application_scope:
            raise CollaborationAccessDenied("Participant access scope conflicts.")
        if grant.participants is not None:
            if any(ref.owner != initialized.owner for ref in grant.participants) or len(
                set(grant.participants)
            ) != len(grant.participants):
                raise CollaborationAccessDenied("Participant access references conflict.")
            grant = CollaborationAccessGrant(
                application_scope=grant.application_scope,
                participants=tuple(
                    sorted(
                        grant.participants, key=lambda ref: (ref.participant_id, ref.incarnation)
                    )
                ),
            )
        return context, grant

    def _require_refs(
        self,
        grant: CollaborationAccessGrant,
        refs: tuple[ParticipantRef, ...],
        *,
        create: bool = False,
    ) -> None:
        if (create and grant.participants is not None) or (
            grant.participants is not None and any(ref not in grant.participants for ref in refs)
        ):
            raise CollaborationAccessDenied(
                "Participant operation is outside the authorized selection."
            )

    def _expected(
        self, request, context: CollaborationAccessContext
    ) -> ExpectedOperation[ParticipantIntent]:
        _, initialized = self._ready()
        if request.operation.application_scope != initialized.binding.application_scope:
            raise CollaborationConflict("Participant operation belongs to another scope.")
        return self._prepare(
            ParticipantCommand,
            ParticipantCommand(
                operation=request.operation,
                kind=request.kind,
                schema_version=1,
                mode="identity",
                source=initialized.owner,
                destination=initialized.owner,
                initiator=InitiatorBinding(
                    issuer=initialized.owner,
                    principal=context.principal,
                    participant=None,
                    mandate=None,
                    invocation_id=None,
                    interaction_id=None,
                ),
                receipt_stage="committed",
                intent=ParticipantIntent(request=request, limits=initialized.binding.limits),
            ),
        )

    async def mutate(self, schema, request, *, context):
        store, initialized = self._ready()
        request = self._prepare(schema, request)
        context, read_grant = self._authorize(context, "readback")
        refs = self._request_refs(request)
        self._require_refs(read_grant, refs, create=isinstance(request, ParticipantCreate))
        expected = self._expected(request, context)
        self._capability(store, initialized, mutation=False)
        replay = self._prepare(
            _Readback,
            {
                "result": await self._store_result(
                    store.lookup(initialized, expected, redactor=self._redactor)
                )
            },
        ).result
        if isinstance(replay, ExactMatch):
            return self._receipt(expected, replay.receipt)
        if isinstance(replay, ExactConflict):
            raise CollaborationConflict("Participant operation conflicts with retained intent.")
        if replay.status != "not_found":
            raise CollaborationUnavailable("Participant exact readback is unavailable.")
        _, grant = self._authorize(context, request.kind)
        self._require_refs(grant, refs, create=isinstance(request, ParticipantCreate))
        if (
            isinstance(request, (ParticipantCreate, ParticipantConfigure))
            and request.configuration not in self._configurations
        ):
            raise CollaborationUnavailable("Exact participant configuration is not registered.")
        self._capability(store, initialized, mutation=True)
        return self._receipt(
            expected,
            await self._store_result(store.apply(initialized, expected, redactor=self._redactor)),
        )

    def _request_refs(self, request) -> tuple[ParticipantRef, ...]:
        if isinstance(request, ParticipantConfigure):
            return (request.participant,)
        if isinstance(request, ParticipantAliasChange):
            return tuple(
                ref for ref in (request.expected_target, request.target) if ref is not None
            )
        return ()

    def _receipt(self, expected, receipt) -> ParticipantReceipt:
        receipt = self._prepare(ParticipantReceipt, receipt)
        if receipt.expected != expected or receipt.event.operation != expected.operation:
            raise CollaborationUnavailable("Participant receipt conflicts with expected operation.")
        return receipt

    async def lookup(
        self, expected: ExpectedOperation[ParticipantIntent], *, context: CollaborationAccessContext
    ):
        store, initialized = self._ready()
        expected = self._prepare(ParticipantCommand, expected)
        _, grant = self._authorize(context, "readback")
        self._require_refs(
            grant,
            self._request_refs(expected.intent.request),
            create=isinstance(expected.intent.request, ParticipantCreate),
        )
        self._capability(store, initialized, mutation=False)
        result = self._prepare(
            _Readback,
            {
                "result": await self._store_result(
                    store.lookup(initialized, expected, redactor=self._redactor)
                )
            },
        ).result
        if isinstance(result, ExactMatch):
            self._receipt(expected, result.receipt)
        return result

    async def inspect(self, participant: ParticipantRef, *, context: CollaborationAccessContext):
        store, initialized = self._ready()
        participant = self._prepare(ParticipantRef, participant)
        _, grant = self._authorize(context, "inspect")
        self._require_refs(grant, (participant,))
        self._capability(store, initialized, mutation=False)
        result = self._prepare(
            ParticipantInspection,
            await self._store_result(
                store.inspect(initialized, participant, redactor=self._redactor)
            ),
        )
        if result.participant.reference != participant:
            raise CollaborationUnavailable("Participant inspection returned another identity.")
        return result

    async def resolve_alias(self, alias: str, *, context: CollaborationAccessContext):
        store, initialized = self._ready()
        _, grant = self._authorize(context, "discover")
        self._capability(store, initialized, mutation=False)
        result = await self._store_result(
            store.resolve_alias(initialized, alias, redactor=self._redactor)
        )
        if result is not None:
            result = self._prepare(ParticipantAlias, result)
            if result.alias != alias or result.target.owner != initialized.owner:
                raise CollaborationUnavailable("Alias lookup returned another identity.")
        if (
            result is not None
            and grant.participants is not None
            and result.target not in grant.participants
        ):
            return None
        return result

    async def discover(
        self,
        *,
        context: CollaborationAccessContext,
        cursor: ParticipantCursor | None = None,
        limit: int = 32,
    ) -> ParticipantPage:
        store, initialized = self._ready()
        if type(limit) is not int or not 1 <= limit <= 64:
            raise CollaborationContractError("Invalid participant page limit.")
        context, grant = self._authorize(context, "discover")
        self._capability(store, initialized, mutation=False)
        after = ""
        if cursor is not None:
            cursor = self._prepare(ParticipantCursor, cursor)
            if (cursor.scope, cursor.principal, cursor.allowed) != (
                grant.application_scope,
                context.principal,
                grant.participants,
            ):
                raise CollaborationAccessDenied(
                    "Participant cursor does not match current authorized query."
                )
            after = cursor.after_id
        records = await self._store_result(
            store.scan(
                initialized,
                table="participants",
                after=after,
                limit=limit,
                allowed=grant.participants,
                redactor=self._redactor,
            )
        )
        if type(records) is not tuple or len(records) > limit:
            raise CollaborationUnavailable("Participant page has invalid evidence.")
        records = tuple(self._prepare(ParticipantSnapshot, p) for p in records)
        previous = after
        for record in records:
            if (
                record.reference.owner != initialized.owner
                or record.reference.participant_id <= previous
            ):
                raise CollaborationUnavailable("Participant page authority or ordering conflicts.")
            self._require_refs(grant, (record.reference,))
            previous = record.reference.participant_id
        return self._page(
            ParticipantPage,
            "participants",
            records,
            lambda record: {
                "scope": grant.application_scope,
                "principal": context.principal,
                "allowed": None
                if grant.participants is None
                else [snapshot_input(ref) for ref in grant.participants],
                "after_id": record.reference.participant_id,
            },
            limit,
        )

    async def events(
        self,
        *,
        context: CollaborationAccessContext,
        cursor: ParticipantEventCursor | None = None,
        limit: int = 32,
    ) -> ParticipantEventPage:
        store, initialized = self._ready()
        if type(limit) is not int or not 1 <= limit <= 64:
            raise CollaborationContractError("Invalid participant event page limit.")
        context, grant = self._authorize(context, "inspect")
        self._capability(store, initialized, mutation=False)
        after = 0
        if cursor is not None:
            cursor = self._prepare(ParticipantEventCursor, cursor)
            if (cursor.scope, cursor.principal, cursor.allowed) != (
                grant.application_scope,
                context.principal,
                grant.participants,
            ):
                raise CollaborationAccessDenied(
                    "Event cursor does not match current authorized query."
                )
            after = cursor.after_sequence
        batch = await self._store_result(
            store.scan_events(
                initialized,
                after=after,
                limit=limit,
                allowed=grant.participants,
                retention_revision=None if cursor is None else cursor.retention_revision,
                redactor=self._redactor,
            )
        )
        if (
            type(batch) is not tuple
            or len(batch) != 2
            or type(batch[0]) is not int
            or not 1 <= batch[0] <= 2**53 - 1
        ):
            raise CollaborationUnavailable("Event retention evidence is unavailable.")
        revision, records = batch
        if cursor is not None and revision != cursor.retention_revision:
            raise CollaborationHistoryUnavailable("Event cursor predates retained history.")
        if type(records) is not tuple or len(records) > limit:
            raise CollaborationUnavailable("Participant event page has invalid evidence.")
        records = tuple(self._prepare(ParticipantEvent, p) for p in records)
        previous = after
        for record in records:
            if record.sequence <= previous or any(
                ref.owner != initialized.owner for ref in record.participants
            ):
                raise CollaborationUnavailable("Participant event authority or ordering conflicts.")
            if not record.participants and grant.participants is not None:
                raise CollaborationAccessDenied("Scope event is outside the authorized selection.")
            self._require_refs(grant, record.participants)
            previous = record.sequence
        return self._page(
            ParticipantEventPage,
            "events",
            records,
            lambda record: {
                "scope": grant.application_scope,
                "principal": context.principal,
                "allowed": None
                if grant.participants is None
                else [snapshot_input(ref) for ref in grant.participants],
                "after_sequence": record.sequence,
                "retention_revision": revision,
            },
            limit,
            evidence={"retention_revision": revision, "history_complete": revision == 1},
        )
