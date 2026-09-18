"""Authenticated owner facade. No model, session or delivery dispatcher is held."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Annotated, TypeVar

from pydantic import Field, StrictInt

from cayu.collaboration._contracts import (
    CollaborationConflict,
    CollaborationContractError,
    ContractValue,
    ExactConflict,
    ExactLookup,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
)
from cayu.collaboration._coordinator import ParticipantCoordinator
from cayu.collaboration._diagnostics import safe_failure
from cayu.collaboration._mandate_validation import (
    MandateInput,
    MandateUse,
    validate_mandate_resolution,
)
from cayu.collaboration._ownership import _MutationOwners
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_arbitration import (
    admit_in_transaction,
    outcome_in_transaction,
    progress_in_transaction,
    register_observation_in_transaction,
)
from cayu.collaboration._request_store import (
    accept_in_transaction,
    control_in_transaction,
    observation_operation,
    require_request_absence,
    require_request_event,
    retained_request,
)
from cayu.collaboration.access import (
    CollaborationAccessContext,
    CollaborationAccessDenied,
    CollaborationAccessGrant,
)
from cayu.collaboration.base import REQUEST_FAMILY, _key, _Repository, _stored_mode
from cayu.collaboration.lifecycle import (
    CollaborationHistoryUnavailable,
    CollaborationNamespaceRetired,
)
from cayu.collaboration.mandates import (
    MandateAccessContext,
    MandateDenied,
    MandateResolution,
    MandateResolver,
    ResourceSelector,
    ResourceSelectorOwner,
)
from cayu.collaboration.participants import (
    CollaborationCapacityExceeded,
    CollaborationNotInitialized,
    CollaborationUnavailable,
    ParticipantAlias,
    ParticipantRef,
)
from cayu.collaboration.request_access import (
    RequestReceivingAuthorization,
    RequestReceivingOwner,
    RequestRegistration,
)
from cayu.collaboration.requests import (
    CollaborationRequest,
    RequestAdmissionCommand,
    RequestAdmissionReceipt,
    RequestAlias,
    RequestCommand,
    RequestControl,
    RequestControlCommand,
    RequestControlReceipt,
    RequestDueCursor,
    RequestDuePage,
    RequestEvent,
    RequestObservation,
    RequestObservationPage,
    RequestObservationReceipt,
    RequestOutcomeCommand,
    RequestOutcomeReceipt,
    RequestProgressCommand,
    RequestProgressReceipt,
    RequestReceipt,
    RequestSnapshot,
)
from cayu.vaults.redaction import SecretRedactor

T = TypeVar("T")

_REQUEST_ERRORS = (
    CollaborationAccessDenied,
    CollaborationConflict,
    CollaborationContractError,
    CollaborationCapacityExceeded,
    CollaborationNotInitialized,
    CollaborationNamespaceRetired,
    CollaborationHistoryUnavailable,
    CollaborationUnavailable,
)


def _safe_request_failure(error: BaseException, redactor: SecretRedactor) -> BaseException:
    """Copy causal evidence before any public classification can discard it."""
    seen: set[int] = set()

    def copy(value: BaseException, depth: int) -> BaseException | None:
        if id(value) in seen:
            return None
        if len(seen) >= 64 or depth >= 16:
            return RuntimeError("Additional request failure evidence withheld.")
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
                BaseExceptionGroup("Request dependency failures.", children)
                if children
                else RuntimeError("Shared request failure already represented.")
            )
        else:
            result = safe_failure(value, redactor=redactor)
        if value.__cause__ is not None:
            result.__cause__ = copy(value.__cause__, depth + 1)
        if value.__context__ is not None:
            result.__context__ = copy(value.__context__, depth + 1)
        result.__suppress_context__ = value.__suppress_context__
        return result

    result = copy(error, 0)
    assert result is not None
    return result


class _Submission(ContractValue):
    request: CollaborationRequest | RequestCommand | RequestControl | RequestControlCommand
    context: MandateAccessContext
    observation: RequestObservation | None = None


class _DueQuery(ContractValue):
    context: MandateAccessContext
    cursor: RequestDueCursor
    limit: Annotated[StrictInt, Field(ge=1, le=64)]


class _ReceivingSubmission(ContractValue):
    command: RequestAdmissionCommand | RequestProgressCommand | RequestOutcomeCommand
    context: MandateAccessContext


def _initiator(context: MandateAccessContext) -> InitiatorBinding:
    participant = context.participant
    return InitiatorBinding(
        issuer=context.issuer,
        principal=context.principal,
        mandate=context.mandate,
        participant=None
        if participant is None
        else ObjectRef(
            owner=participant.owner,
            kind="participant",
            object_id=participant.participant_id,
            incarnation=participant.incarnation,
        ),
        invocation_id=None,
        interaction_id=None,
    )


class RequestCoordinator:
    def __init__(
        self,
        *,
        participants: ParticipantCoordinator,
        registration: RequestRegistration | None,
        redactor: SecretRedactor,
    ):
        self._participants = participants
        self._registration = registration
        self._redactor = redactor
        self._owners = (
            participants._store._owners
            if registration is not None and participants._store is not None
            else _MutationOwners()
        )
        self._resource_owners = {}
        self._resolver_ref = None
        self._receiving_ref = None
        if registration is not None:
            if (
                type(registration) is not RequestRegistration
                or not isinstance(registration.mandates, MandateResolver)
                or type(registration.max_ttl_ms) is not int
                or not 1 <= registration.max_ttl_ms <= 2**53 - 1
                or type(registration.resource_owners) is not tuple
                or len(registration.resource_owners) > 32
            ):
                raise CollaborationContractError("Invalid request owner registration.")
            self._resolver_ref = prepare_contract(
                ObjectRef, registration.mandates.ref, redactor=redactor
            )
            if registration.receiving_owner is not None:
                if not isinstance(registration.receiving_owner, RequestReceivingOwner):
                    raise CollaborationContractError("Invalid receiving owner registration.")
                self._receiving_ref = prepare_contract(
                    ObjectRef, registration.receiving_owner.ref, redactor=redactor
                )
                if self._receiving_ref.revision is None:
                    raise CollaborationContractError("Receiving owner must have a pinned revision.")
            for owner in registration.resource_owners:
                if (
                    not isinstance(owner, ResourceSelectorOwner)
                    or owner.owner in self._resource_owners
                ):
                    raise CollaborationContractError("Invalid request resource owner registration.")
                self._resource_owners[owner.owner] = owner

    async def close(self) -> None:
        await self._owners.drain()

    async def _dependency(self, operation: Callable[[], Awaitable[T]]) -> T:
        task = asyncio.current_task()
        baseline = 0 if task is None else task.cancelling()
        try:
            return await operation()
        except asyncio.CancelledError as error:
            if task is not None and task.cancelling() > baseline:
                raise
            # Snapshot before the retained task's cancelled state loses the
            # dependency's causal graph. This is not observer cancellation.
            failure = RuntimeError("Request dependency cancelled; reconcile exact state.")
            failure.__cause__ = _safe_request_failure(error, self._redactor)
        # Leave the handler before raising: raw dependency context is private.
        raise failure

    async def _observe(self, operation: Awaitable[T]) -> T:
        failure: BaseException
        try:
            return await operation
        except asyncio.CancelledError as error:
            failure = asyncio.CancelledError(
                "Request observation cancelled; reconcile exact state."
            )
            if error.__cause__ is not None:
                failure.__cause__ = _safe_request_failure(error.__cause__, self._redactor)
        except MandateDenied as error:
            failure = CollaborationAccessDenied("Request authority was denied.")
            failure.__cause__ = _safe_request_failure(error, self._redactor)
        except _REQUEST_ERRORS as error:
            kind = next(kind for kind in _REQUEST_ERRORS if isinstance(error, kind))
            failure = kind("Request owner refused the operation; reconcile exact state.")
            failure.__cause__ = _safe_request_failure(error, self._redactor)
        except BaseExceptionGroup as error:
            fatal = error.subgroup(
                lambda value: (
                    not isinstance(value, (Exception, asyncio.CancelledError, BaseExceptionGroup))
                )
            )
            if fatal is not None:
                raise
            failure = CollaborationUnavailable("Request dependency failed; reconcile exact state.")
            failure.__cause__ = _safe_request_failure(error, self._redactor)
        except Exception as error:
            failure = CollaborationUnavailable("Request dependency failed; reconcile exact state.")
            failure.__cause__ = _safe_request_failure(error, self._redactor)
        raise failure

    async def due(
        self,
        *,
        context: MandateAccessContext,
        cursor: RequestDueCursor | None = None,
        limit: int = 32,
    ) -> RequestDuePage:
        value = prepare_contract(
            _DueQuery,
            {
                "context": context,
                "cursor": RequestDueCursor(after=0) if cursor is None else cursor,
                "limit": limit,
            },
            redactor=self._redactor,
        )

        async def owned():
            return await self._dependency(lambda: self._due(value))

        return await self._observe(
            self._owners.run(
                owned,
                key=("request_due_observation", object()),
                expectation=contract_bytes(value, redactor=self._redactor),
                redactor=self._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, self._redactor),
            )
        )

    async def _due(self, query: _DueQuery) -> RequestDuePage:
        store, initialized = self._participants._ready()
        self._participants._capability(store, initialized, mutation=False, family=REQUEST_FAMILY)
        registration = self._registration
        resolver_ref = self._resolver_ref
        if registration is None or resolver_ref is None:
            raise CollaborationNotInitialized("No collaboration request owner is registered.")
        _, grant = self._participants._authorize(
            CollaborationAccessContext(principal=query.context.principal), "request_readback"
        )
        # Due inspection is a scope-wide maintenance API, not participant discovery.
        self._participants._require_refs(grant, (), create=True)
        require_exact_contract(
            resolver_ref,
            prepare_contract(ObjectRef, registration.mandates.ref, redactor=self._redactor),
            redactor=self._redactor,
        )
        async with registration.mandates.acquire(query.context) as raw:
            resolution = prepare_contract(MandateResolution, raw, redactor=self._redactor)
            async with store._transaction(initialized.binding.application_scope, write=False) as tx:
                now = await tx.now_ms()
            await asyncio.to_thread(
                partial(
                    validate_mandate_resolution,
                    resolution,
                    context=query.context,
                    resolver=resolver_ref,
                    use=MandateUse(
                        audience=initialized.owner,
                        scope=initialized.binding.application_scope,
                        actions=("readback",),
                        resources=(),
                        inputs=(),
                    ),
                    now_ms=now,
                    resource_owners=self._resource_owners,
                    redactor=self._redactor,
                )
            )
            async with store._transaction(initialized.binding.application_scope, write=False) as tx:
                await store._anchor(tx, initialized, self._redactor)
                now = await tx.now_ms()
                rows = await tx.scan_due_requests(
                    after=query.cursor.after, now_ms=now, limit=query.limit
                )
                records = []
                previous = query.cursor.after
                for raw in rows:
                    record = prepare_contract(RequestSnapshot, raw, redactor=self._redactor)
                    if (
                        record.state != "open"
                        or record.receipt.event.sequence <= previous
                        or record.next_due_at_ms > now
                    ):
                        raise CollaborationUnavailable("Due index conflicts with request evidence.")
                    expected = record.receipt.expected
                    exact = await retained_request(
                        store,
                        tx,
                        initialized,
                        expected.intent.request,
                        expected.initiator,
                        self._redactor,
                    )
                    if exact != record:
                        raise CollaborationUnavailable("Due request contradicts its acceptance.")
                    records.append(record)
                    previous = record.receipt.event.sequence
                if await tx.now_ms() >= min(
                    resolution.principal.expires_at_ms,
                    *(item.expires_at_ms for item in resolution.chain.entries),
                ):
                    raise CollaborationAccessDenied("Inspection authority expired during readback.")
            return self._participants._page(
                RequestDuePage,
                "items",
                tuple(records),
                lambda item: {"after": item.receipt.event.sequence},
                query.limit,
                evidence={"observed_at_ms": now},
            )

    async def accept(
        self, request: CollaborationRequest, *, context: MandateAccessContext
    ) -> RequestReceipt:
        value = prepare_contract(
            _Submission, {"request": request, "context": context}, redactor=self._redactor
        )
        if not isinstance(value.request, CollaborationRequest):
            raise CollaborationContractError("Expected original request intent.")
        result = await self._run(value, mode="accept")
        assert isinstance(result, RequestSnapshot)
        return result.receipt

    async def inspect(
        self, expected: RequestCommand, *, context: MandateAccessContext
    ) -> RequestSnapshot | None:
        value = prepare_contract(
            _Submission, {"request": expected, "context": context}, redactor=self._redactor
        )
        if not isinstance(value.request, RequestCommand):
            raise CollaborationContractError("Expected the complete accepted command.")
        result = await self._run(value, mode="inspect")
        assert result is None or isinstance(result, RequestSnapshot)
        return result

    async def lookup(
        self, expected: RequestCommand | RequestControlCommand, *, context: MandateAccessContext
    ) -> ExactLookup[RequestReceipt | RequestControlReceipt]:
        # Authorization failures remain denials; only qualified readback failure
        # becomes the fourth exact-lookup alternative.
        schema = (
            RequestControlCommand if isinstance(expected, RequestControlCommand) else RequestCommand
        )
        expected = prepare_contract(schema, expected, redactor=self._redactor)
        context = prepare_contract(MandateAccessContext, context, redactor=self._redactor)
        try:
            if isinstance(expected, RequestControlCommand):
                result = await self._run(
                    prepare_contract(
                        _Submission,
                        {"request": expected, "context": context},
                        redactor=self._redactor,
                    ),
                    mode="lookup_control",
                )
                return (
                    ExactNotFound()
                    if result is None
                    else ExactMatch[RequestControlReceipt](receipt=result)
                )
            result = await self.inspect(expected, context=context)
        except CollaborationConflict:
            return ExactConflict()
        except (CollaborationUnavailable, CollaborationContractError):
            return ExactUnavailable()
        return (
            ExactNotFound()
            if result is None
            else ExactMatch[RequestReceipt](receipt=result.receipt)
        )

    async def control(
        self, request: RequestControl, *, context: MandateAccessContext
    ) -> RequestControlReceipt:
        value = prepare_contract(
            _Submission, {"request": request, "context": context}, redactor=self._redactor
        )
        if not isinstance(value.request, RequestControl):
            raise CollaborationContractError("Expected a request control.")
        result = await self._run(value, mode="control")
        assert isinstance(result, RequestControlReceipt)
        return result

    async def admit(
        self, command: RequestAdmissionCommand, *, context: MandateAccessContext
    ) -> RequestAdmissionReceipt:
        return await self._trusted_mutation(
            command, context=context, operation=admit_in_transaction
        )

    async def progress(
        self, command: RequestProgressCommand, *, context: MandateAccessContext
    ) -> RequestProgressReceipt:
        return await self._trusted_mutation(
            command, context=context, operation=progress_in_transaction
        )

    async def outcome(
        self, command: RequestOutcomeCommand, *, context: MandateAccessContext
    ) -> RequestOutcomeReceipt:
        return await self._trusted_mutation(
            command, context=context, operation=outcome_in_transaction
        )

    async def observe(
        self,
        expected: RequestCommand,
        observation: RequestObservation,
        *,
        context: MandateAccessContext,
    ) -> RequestObservationReceipt:
        expected = prepare_contract(RequestCommand, expected, redactor=self._redactor)
        observation = prepare_contract(RequestObservation, observation, redactor=self._redactor)
        value = prepare_contract(
            _Submission,
            {"request": expected, "context": context, "observation": observation},
            redactor=self._redactor,
        )
        return await self._run(value, mode="register_observation")

    async def read_observation(
        self,
        expected: RequestCommand,
        observation: RequestObservation,
        *,
        context: MandateAccessContext,
    ) -> RequestObservationPage:
        expected = prepare_contract(RequestCommand, expected, redactor=self._redactor)
        observation = prepare_contract(RequestObservation, observation, redactor=self._redactor)
        value = prepare_contract(
            _Submission,
            {"request": expected, "context": context, "observation": observation},
            redactor=self._redactor,
        )
        result = await self._run(value, mode="read_observation")
        assert isinstance(result, RequestObservationPage)
        return result

    async def _trusted_mutation(self, command, *, context, operation):
        if self._registration is None:
            raise CollaborationNotInitialized("No collaboration request owner is registered.")
        value = prepare_contract(
            _ReceivingSubmission,
            {"command": command, "context": context},
            redactor=self._redactor,
        )

        async def owned():
            return await self._dependency(lambda: self._receive_held(value, operation))

        return await self._observe(
            self._owners.run(
                owned,
                key=("request_receiver", object()),
                expectation=contract_bytes(value, redactor=self._redactor),
                redactor=self._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, self._redactor),
            )
        )

    async def _receive_held(self, value: _ReceivingSubmission, operation):
        registration = self._registration
        assert registration is not None
        receiver = registration.receiving_owner
        if receiver is None or self._receiving_ref is None:
            raise CollaborationUnavailable("No qualified receiving owner is registered.")
        command, context = value.command, value.context
        initiating = (
            command.publisher if isinstance(command, RequestProgressCommand) else command.initiator
        )
        require_exact_contract(initiating, _initiator(context), redactor=self._redactor)
        selected = command.expected.intent.selection
        if context.participant != selected.recipient.reference:
            raise CollaborationAccessDenied("Receiving principal is not the selected recipient.")
        store, initialized = self._participants._ready()
        self._participants._capability(store, initialized, mutation=False, family=REQUEST_FAMILY)
        _, grant = self._participants._authorize(
            CollaborationAccessContext(principal=context.principal), "request_readback"
        )
        self._participants._require_refs(
            grant, (selected.sender.reference, selected.recipient.reference)
        )
        require_exact_contract(
            self._receiving_ref,
            prepare_contract(ObjectRef, receiver.ref, redactor=self._redactor),
            redactor=self._redactor,
        )
        # A committed operation is replayed from the collaboration store before
        # consulting the producer.  Producer authority may have expired or
        # been retired after the durable receipt was published; replay must not
        # turn that acknowledgement-loss/retry path into a new source read.
        key = (
            command.operation.namespace_incarnation,
            command.operation.generation,
            command.operation.caller_key,
        )
        async with store._transaction(initialized.binding.application_scope, write=False) as tx:
            if await tx.get("operations", key) is not None:
                return await operation(store, tx, initialized, command, redactor=self._redactor)
        async with receiver.acquire(command, context=context) as raw:
            authority = prepare_contract(
                RequestReceivingAuthorization, raw, redactor=self._redactor
            )
            require_exact_contract(command, authority.command, redactor=self._redactor)
            require_exact_contract(self._receiving_ref, authority.receiver, redactor=self._redactor)
            async with store._transaction(initialized.binding.application_scope, write=False) as tx:
                if await tx.now_ms() >= authority.expires_at_ms:
                    raise CollaborationAccessDenied("Receiving read authority expired.")
            settlement = authority.settlement
            if settlement is None and (
                isinstance(command, RequestOutcomeCommand)
                or (isinstance(command, RequestAdmissionCommand) and command.decision == "decline")
            ):
                async with store._transaction(
                    initialized.binding.application_scope, write=False
                ) as tx:
                    prior = await retained_request(
                        store,
                        tx,
                        initialized,
                        command.expected.intent.request,
                        command.expected.initiator,
                        self._redactor,
                    )
                if prior is None:
                    raise CollaborationUnavailable("Request responsibility is unavailable.")
                settlement = await receiver.settlement(command, prior.permit, context=context)
            self._participants._capability(store, initialized, mutation=True, family=REQUEST_FAMILY)
            async with store._transaction(initialized.binding.application_scope, write=True) as tx:
                if await tx.now_ms() >= authority.expires_at_ms:
                    raise CollaborationAccessDenied(
                        "Receiving authority expired before publication."
                    )
                if isinstance(command, RequestAdmissionCommand):
                    return await operation(
                        store,
                        tx,
                        initialized,
                        command,
                        settlement=settlement,
                        redactor=self._redactor,
                    )
                if isinstance(command, RequestOutcomeCommand):
                    return await operation(
                        store,
                        tx,
                        initialized,
                        command,
                        settlement=settlement,
                        redactor=self._redactor,
                    )
                return await operation(store, tx, initialized, command, redactor=self._redactor)

    async def _run(self, value: _Submission, *, mode: str):
        if self._registration is None:
            raise CollaborationNotInitialized("No collaboration request owner is registered.")
        operation = value.request.operation

        async def owned():
            return await self._dependency(lambda: self._held(value, mode=mode))

        return await self._observe(
            self._owners.run(
                owned,
                key=(
                    mode,
                    operation.application_scope,
                    operation.namespace_incarnation,
                    operation.generation,
                    operation.caller_key,
                    # An identical claim does not authenticate another observer.
                    # Each retained task runs its own guard; the transaction owns
                    # exact mutation replay across applications and processes.
                    object(),
                ),
                expectation=contract_bytes(value, redactor=self._redactor),
                redactor=self._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, self._redactor),
            )
        )

    async def _require_retained_read_grant(
        self, tx: _Repository, operation: OperationRef, grant: CollaborationAccessGrant
    ) -> None:
        """Authorize frozen participants before revealing any exact-key outcome.

        Called inside the same read transaction as comparison. No alias lookup
        may replace the participants selected by the retained operation.
        """
        if grant.participants is None:
            return
        raw = await tx.get(
            "operations",
            (operation.namespace_incarnation, operation.generation, operation.caller_key),
        )
        if raw is None:
            return
        mode = _stored_mode(raw)
        if mode == "request":
            receipt = prepare_contract(RequestReceipt, raw, redactor=self._redactor)
            selected = receipt.expected.intent.selection
        elif mode == "request_control":
            control = prepare_contract(RequestControlReceipt, raw, redactor=self._redactor)
            selected = control.expected.intent.expected.intent.selection
        else:
            # Another operation family is not evidence of request read access.
            raise CollaborationAccessDenied("Operation is outside request read authority.")
        self._participants._require_refs(
            grant, (selected.sender.reference, selected.recipient.reference)
        )

    async def _held(self, value: _Submission, *, mode: str):
        store, initialized = self._participants._ready()
        self._participants._capability(
            store,
            initialized,
            mutation=False,
            family=REQUEST_FAMILY,
        )
        registration = self._registration
        assert registration is not None and self._resolver_ref is not None
        resolver_ref = self._resolver_ref
        context = value.context
        raw_request = value.request
        command = (
            raw_request
            if isinstance(raw_request, RequestCommand)
            else raw_request.intent.expected
            if isinstance(raw_request, RequestControlCommand)
            else raw_request.expected
            if isinstance(raw_request, RequestControl)
            else None
        )
        if isinstance(raw_request, CollaborationRequest):
            request = raw_request
        else:
            assert isinstance(command, RequestCommand)
            request = command.intent.request
            selected = command.intent.selection
        assert isinstance(request, CollaborationRequest)
        if command is None:
            selected = None
        if request.sender.owner != initialized.owner:
            raise CollaborationConflict("Request belongs to another collaboration owner.")
        _, read_grant = self._participants._authorize(
            CollaborationAccessContext(principal=context.principal), "request_readback"
        )
        declared = (
            (
                selected.sender.reference,
                selected.recipient.reference,
            )
            if selected is not None
            else (request.sender, request.target)
            if isinstance(request.target, ParticipantRef)
            else (request.sender,)
        )
        self._participants._require_refs(read_grant, declared)
        require_exact_contract(
            self._resolver_ref,
            prepare_contract(ObjectRef, registration.mandates.ref, redactor=self._redactor),
            redactor=self._redactor,
        )
        async with registration.mandates.acquire(context) as raw_resolution:
            resolution = prepare_contract(
                MandateResolution, raw_resolution, redactor=self._redactor
            )
            permission_deadline = min(
                resolution.principal.expires_at_ms,
                *(entry.expires_at_ms for entry in resolution.chain.entries),
            )

            async def validate(actions):
                async with store._transaction(
                    initialized.binding.application_scope, write=False
                ) as tx:
                    now = await tx.now_ms()
                resources = (
                    tuple(ResourceSelector(resource=ref) for ref in request.inputs)
                    if "consult" in actions
                    else ()
                )
                use = MandateUse(
                    audience=initialized.owner,
                    scope=initialized.binding.application_scope,
                    actions=actions,
                    resources=resources,
                    inputs=tuple(
                        MandateInput(source=ref, channel="prompt") for ref in request.inputs
                    )
                    if resources
                    else (),
                )
                return await asyncio.to_thread(
                    partial(
                        validate_mandate_resolution,
                        resolution,
                        context=context,
                        resolver=resolver_ref,
                        use=use,
                        now_ms=now,
                        resource_owners=self._resource_owners,
                        redactor=self._redactor,
                    )
                )

            await validate(("readback",))
            original_initiator = _initiator(context) if command is None else command.initiator
            async with store._transaction(initialized.binding.application_scope, write=False) as tx:
                await self._require_retained_read_grant(tx, request.operation, read_grant)
                found = await retained_request(
                    store, tx, initialized, request, original_initiator, self._redactor
                )
                if await tx.now_ms() >= permission_deadline:
                    raise CollaborationAccessDenied("Read authority expired during inspection.")
            if found is not None:
                chosen = found.receipt.expected.intent.selection
                self._participants._require_refs(
                    read_grant, (chosen.sender.reference, chosen.recipient.reference)
                )
                if command is not None:
                    require_exact_contract(command, found.receipt.expected, redactor=self._redactor)
                if mode in ("inspect", "accept"):
                    return found
            elif mode == "inspect":
                assert command is not None
                assert selected is not None
                self._participants._require_refs(
                    read_grant,
                    (
                        selected.sender.reference,
                        selected.recipient.reference,
                    ),
                )
                return None
            if mode == "lookup_control":
                assert isinstance(raw_request, RequestControlCommand)
                self._participants._require_refs(
                    read_grant,
                    (
                        raw_request.intent.expected.intent.selection.sender.reference,
                        raw_request.intent.expected.intent.selection.recipient.reference,
                    ),
                )
                async with store._transaction(
                    initialized.binding.application_scope, write=False
                ) as tx:
                    await self._require_retained_read_grant(tx, raw_request.operation, read_grant)
                    raw = await tx.get("operations", _key(raw_request))
                    if raw is None:
                        await require_request_absence(
                            store, tx, initialized, raw_request.operation, self._redactor
                        )
                        return None
                    if _stored_mode(raw) != "request_control":
                        raise CollaborationConflict("Control key has another operation.")
                    replay = prepare_contract(RequestControlReceipt, raw, redactor=self._redactor)
                    require_exact_contract(raw_request, replay.expected, redactor=self._redactor)
                    await require_request_event(tx, replay.event, self._redactor)
                    if found is None or found.terminal != replay:
                        raise CollaborationUnavailable("Control receipt contradicts request state.")
                    if await tx.now_ms() >= permission_deadline:
                        raise CollaborationAccessDenied("Read authority expired during replay.")
                    return replay
            if mode == "register_observation":
                if found is None or command is None or value.observation is None:
                    raise CollaborationUnavailable(
                        "Observation requires retained request authority."
                    )
                self._participants._capability(
                    store, initialized, mutation=True, family=REQUEST_FAMILY
                )
                async with store._transaction(
                    initialized.binding.application_scope, write=True
                ) as tx:
                    if await tx.now_ms() >= permission_deadline:
                        raise CollaborationAccessDenied("Observation authority expired.")
                    return await register_observation_in_transaction(
                        store,
                        tx,
                        initialized,
                        command,
                        value.observation,
                        initiator=_initiator(context),
                        redactor=self._redactor,
                    )
            if mode == "read_observation":
                if found is None or command is None or value.observation is None:
                    raise CollaborationUnavailable(
                        "Observation requires retained request authority."
                    )
                wanted = value.observation
                operation = observation_operation(command, wanted.key)
                async with store._transaction(
                    initialized.binding.application_scope, write=False
                ) as tx:
                    # Reload under the same transaction as the source frontier.
                    # The earlier permission lookup is not a coverage snapshot.
                    anchor = await store._anchor(tx, initialized, self._redactor)
                    current_request = await retained_request(
                        store, tx, initialized, request, original_initiator, self._redactor
                    )
                    if current_request is None:
                        raise CollaborationUnavailable("Observation request is unavailable.")
                    current = next(
                        (item for item in current_request.observations if item.key == wanted.key),
                        None,
                    )
                    if current is None:
                        raise CollaborationUnavailable("Observation registration is unavailable.")
                    raw = await tx.get(
                        "operations",
                        (
                            operation.namespace_incarnation,
                            operation.generation,
                            operation.caller_key,
                        ),
                    )
                    if raw is None:
                        raise CollaborationUnavailable("Observation receipt is unavailable.")
                    registration = prepare_contract(
                        RequestObservationReceipt, raw, redactor=self._redactor
                    )
                    require_exact_contract(registration.expected, command, redactor=self._redactor)
                    require_exact_contract(registration.intent, wanted, redactor=self._redactor)
                    require_exact_contract(
                        registration.observation, current, redactor=self._redactor
                    )
                    await require_request_event(tx, registration.event, self._redactor)
                    known_sequences = tuple(
                        sequence
                        for sequence in current_request.event_sequences
                        if wanted.after_sequence < sequence <= anchor.event_sequence
                    )
                    if len(known_sequences) > 64:
                        raise CollaborationUnavailable("Observation frontier is too large to read.")
                    events_list = []
                    for sequence in known_sequences:
                        raw_event = await tx.get("request_events", (sequence,))
                        if raw_event is None:
                            raise CollaborationUnavailable(
                                "Observation frontier has a durable gap."
                            )
                        event = prepare_contract(RequestEvent, raw_event, redactor=self._redactor)
                        if event.type != "request_observation_registered":
                            events_list.append(event)
                    events = tuple(events_list)
                    if await tx.now_ms() >= permission_deadline:
                        raise CollaborationAccessDenied(
                            "Observation authority expired during read."
                        )
                    return RequestObservationPage(
                        registration=registration,
                        events=events,
                        coverage_sequence=anchor.event_sequence,
                        complete=True,
                    )
            if mode == "control":
                assert isinstance(raw_request, RequestControl)
                if found is None:
                    raise CollaborationUnavailable("Request acceptance is unavailable.")
                control = prepare_contract(
                    RequestControlCommand,
                    {
                        "operation": raw_request.operation,
                        "kind": raw_request.kind,
                        "source": initialized.owner,
                        "destination": initialized.owner,
                        "initiator": _initiator(context),
                        "intent": raw_request,
                    },
                    redactor=self._redactor,
                )
                async with store._transaction(
                    initialized.binding.application_scope, write=False
                ) as tx:
                    await self._require_retained_read_grant(tx, control.operation, read_grant)
                    raw = await tx.get("operations", _key(control))
                    if raw is not None:
                        if _stored_mode(raw) != "request_control":
                            raise CollaborationConflict("Control key has another operation.")
                        replay = prepare_contract(
                            RequestControlReceipt, raw, redactor=self._redactor
                        )
                        require_exact_contract(control, replay.expected, redactor=self._redactor)
                        await require_request_event(tx, replay.event, self._redactor)
                        if await tx.now_ms() >= permission_deadline:
                            raise CollaborationAccessDenied("Read authority expired during replay.")
                        return replay
                self._participants._capability(
                    store, initialized, mutation=True, family=REQUEST_FAMILY
                )
                await validate(("readback", "administer"))
                _, grant = self._participants._authorize(
                    CollaborationAccessContext(principal=context.principal), "request_control"
                )
                chosen = found.receipt.expected.intent.selection
                self._participants._require_refs(
                    grant, (chosen.sender.reference, chosen.recipient.reference)
                )
                if found.admission != "undecided" or found.delivery != "pending":
                    receiver = registration.receiving_owner
                    if receiver is None or self._receiving_ref is None:
                        raise CollaborationUnavailable(
                            "No qualified receiving owner is registered."
                        )
                    require_exact_contract(
                        self._receiving_ref,
                        prepare_contract(ObjectRef, receiver.ref, redactor=self._redactor),
                        redactor=self._redactor,
                    )
                    # Keep both guards in the same retained task. Control is
                    # authorized by the administrator's mandate, not by posing
                    # as the producer; the receiver authenticates settlement.
                    async with receiver.acquire(control, context=context) as raw:
                        authority = prepare_contract(
                            RequestReceivingAuthorization, raw, redactor=self._redactor
                        )
                        require_exact_contract(control, authority.command, redactor=self._redactor)
                        require_exact_contract(
                            self._receiving_ref, authority.receiver, redactor=self._redactor
                        )
                        settlement = authority.settlement
                        if settlement is None:
                            settlement = await receiver.settlement(
                                control, found.permit, context=context
                            )
                        async with store._transaction(
                            initialized.binding.application_scope, write=True
                        ) as tx:
                            await self._require_retained_read_grant(
                                tx, control.operation, read_grant
                            )
                            return await control_in_transaction(
                                store,
                                tx,
                                initialized,
                                control,
                                authority_expires_at_ms=min(
                                    permission_deadline, authority.expires_at_ms
                                ),
                                settlement=settlement,
                                redactor=self._redactor,
                            )
                expiry = min(
                    resolution.principal.expires_at_ms,
                    *(item.expires_at_ms for item in resolution.chain.entries),
                )
                async with store._transaction(
                    initialized.binding.application_scope, write=True
                ) as tx:
                    await self._require_retained_read_grant(tx, control.operation, read_grant)
                    return await control_in_transaction(
                        store,
                        tx,
                        initialized,
                        control,
                        authority_expires_at_ms=expiry,
                        redactor=self._redactor,
                    )
            self._participants._capability(store, initialized, mutation=True, family=REQUEST_FAMILY)
            if context.participant != request.sender:
                raise CollaborationAccessDenied(
                    "New requests require the exact initiating participant."
                )
            await validate(("readback", "consult"))
            leaf = resolution.chain.entries[-1]
            if (
                leaf.restrictions.independence_policy != request.independence_policy
                or leaf.restrictions.disclosure_policy != request.disclosure_policy
                or request.ttl_ms > registration.max_ttl_ms
            ):
                raise CollaborationAccessDenied(
                    "Request restrictions conflict with initiating authority."
                )
            _, grant = self._participants._authorize(
                CollaborationAccessContext(principal=context.principal), "request_accept"
            )
            self._participants._require_refs(grant, (request.sender,))
            async with store._transaction(initialized.binding.application_scope, write=False) as tx:
                sender = await store._participant(
                    tx, request.sender, initialized.owner, self._redactor
                )
                if isinstance(request.target, RequestAlias):
                    raw_alias = await tx.get("aliases", (request.target.alias,))
                    if raw_alias is None:
                        # Missing and inaccessible aliases must not disclose
                        # different lookup outcomes to an initiating caller.
                        raise CollaborationAccessDenied(
                            "Participant operation is outside the authorized selection."
                        )
                    alias = prepare_contract(
                        ParticipantAlias,
                        raw_alias,
                        redactor=self._redactor,
                    )
                    target = alias.target
                else:
                    target = request.target
                self._participants._require_refs(grant, (target,))
                self._participants._require_refs(read_grant, (target,))
                recipient = await store._participant(tx, target, initialized.owner, self._redactor)
            self._participants._require_refs(grant, (sender.reference, recipient.reference))
            self._participants._require_refs(read_grant, (sender.reference, recipient.reference))
            if recipient.configuration not in self._participants._configurations:
                raise CollaborationUnavailable(
                    "Pinned participant configuration is not registered."
                )
            expiry = min(
                resolution.principal.expires_at_ms,
                *(item.expires_at_ms for item in resolution.chain.entries),
            )
            async with store._transaction(initialized.binding.application_scope, write=True) as tx:
                await self._require_retained_read_grant(tx, request.operation, read_grant)
                return await accept_in_transaction(
                    store,
                    tx,
                    initialized,
                    request,
                    _initiator(context),
                    sender=sender,
                    recipient=recipient,
                    authority=resolution,
                    authority_expires_at_ms=expiry,
                    redactor=self._redactor,
                )
