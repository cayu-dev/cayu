"""Join request publication to the existing source, mandate and budget owners.

No foreign transaction is held inside a collaboration transaction. The export
owner's live guard authenticates disclosure; the request transaction arbitrates
the exact question with terminal outcomes and participant lifecycle changes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from decimal import Decimal
from functools import partial

from pydantic import StrictBool  # noqa: TC002 -- Pydantic resolves the contract at runtime.

from cayu._validation import revalidate_model_input
from cayu.budgets.binding import BudgetBinding
from cayu.collaboration._clarification_commands import (
    ClarificationOpenCommand,
    ClarificationOpenReceipt,
)
from cayu.collaboration._clarification_deliveries import (
    ClarificationDeliveryIntent,
    ClarificationDeliveryReceipt,
)
from cayu.collaboration._clarification_delivery_store import (
    reconcile_delivery,
    register_delivery_in_transaction,
)
from cayu.collaboration._clarification_export import acquire_clarification_source
from cayu.collaboration._clarification_service_api import (
    ClarificationServiceReceipt,
    ClarificationServiceRequest,
)
from cayu.collaboration._clarification_store import open_in_transaction
from cayu.collaboration._contracts import CollaborationConflict, ContractValue, ObjectRef
from cayu.collaboration._mandate_validation import (
    MandateInput,
    MandateUse,
    validate_mandate_resolution,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_coordinator import (
    RequestCoordinator,
    _initiator,
    _safe_request_failure,
)
from cayu.collaboration._request_store import operation_key
from cayu.collaboration._session_export_coordinator import SessionExportCoordinator
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.exports import (
    SessionExportAccessContext,
    SessionExportDenied,
    SessionExportRequest,
)
from cayu.collaboration.mandates import ResourceSelector
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.peer_content import PeerContentReceipt


class _Opening(ContractValue):
    command: ClarificationOpenCommand
    source: SessionExportRequest
    context: SessionExportAccessContext


class _Delivery(ContractValue):
    intent: ClarificationDeliveryIntent
    context: SessionExportAccessContext
    prepare_only: StrictBool = False


class ClarificationCoordinator:
    def __init__(
        self,
        *,
        requests: RequestCoordinator,
        exports: SessionExportCoordinator,
        resolve_budget: Callable[..., Awaitable[BudgetBinding]],
        common_root_enabled: bool,
        append_peer: Callable[..., Awaitable[PeerContentReceipt]],
    ):
        self.requests = requests
        self.exports = exports
        self.resolve_budget = resolve_budget
        self.common_root_enabled = common_root_enabled
        self.append_peer = append_peer

    async def reply(self, app, request, *, context):
        from cayu.collaboration._clarification_reply_api import (
            ClarificationReplyRequest,
            accept_reply,
        )

        requests = self.requests
        request = prepare_contract(ClarificationReplyRequest, request, redactor=requests._redactor)
        context = prepare_contract(SessionExportAccessContext, context, redactor=requests._redactor)

        async def owned():
            try:
                return await requests._dependency(
                    lambda: accept_reply(self, app, request, context=context)
                )
            except SessionExportDenied as error:
                failure = CollaborationAccessDenied("Clarification reply disclosure was denied.")
                failure.__cause__ = _safe_request_failure(error, requests._redactor)
            raise failure

        return await requests._observe(
            requests._owners.run(
                owned,
                key=("clarification-reply", object()),
                expectation=contract_bytes(request, redactor=requests._redactor)
                + contract_bytes(context, redactor=requests._redactor),
                redactor=requests._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
            )
        )

    async def service(
        self,
        app,
        request: ClarificationServiceRequest,
        *,
        context: SessionExportAccessContext,
        delivery_context: SessionExportAccessContext | None = None,
    ) -> ClarificationServiceReceipt:
        return await self._service(
            app,
            request,
            context=context,
            delivery_context=delivery_context,
            wait_for_settlement=False,
        )

    async def _service_owned(self, app, request, *, context, delivery_context=None):
        return await self._service(
            app,
            request,
            context=context,
            delivery_context=delivery_context,
            wait_for_settlement=True,
        )

    async def _service(
        self,
        app,
        request: ClarificationServiceRequest,
        *,
        context: SessionExportAccessContext,
        delivery_context: SessionExportAccessContext | None = None,
        wait_for_settlement: bool,
    ) -> ClarificationServiceReceipt:
        from cayu.collaboration._clarification_service_api import service_clarification
        from cayu.collaboration._preparation_progress import (
            PreparationProgress,
            PreparationReadFailure,
        )
        from cayu.runtime._session_continuation import ContinuationConflict

        requests = self.requests
        request = prepare_contract(
            ClarificationServiceRequest, request, redactor=requests._redactor
        )
        context = prepare_contract(SessionExportAccessContext, context, redactor=requests._redactor)

        if delivery_context is not None:
            delivery_context = prepare_contract(
                SessionExportAccessContext, delivery_context, redactor=requests._redactor
            )

        async def owned():
            progress = PreparationProgress()
            try:
                return await requests._dependency(
                    lambda: service_clarification(
                        self,
                        app,
                        request,
                        context=context,
                        delivery_context=delivery_context,
                        _preparation_progress=progress,
                    )
                )
            except SessionExportDenied as error:
                failure = CollaborationAccessDenied("Clarification service disclosure was denied.")
                failure.__cause__ = _safe_request_failure(error, requests._redactor)
            except ContinuationConflict as error:
                failure = CollaborationConflict("Clarification service selection conflicts.")
                failure.__cause__ = _safe_request_failure(error, requests._redactor)
            except Exception as error:
                evidence = progress.read_failure(error)
                if wait_for_settlement and evidence is not None:
                    return evidence
                raise
            evidence = progress.read_failure(failure)
            if wait_for_settlement and evidence is not None:
                return evidence
            raise failure

        return await requests._observe(
            requests._owners.run(
                owned,
                # A fresh observer must authenticate its own disclosure grant;
                # native stores own operation deduplication across observers.
                key=("clarification-service", object()),
                expectation=contract_bytes(request, redactor=requests._redactor)
                + contract_bytes(context, redactor=requests._redactor)
                + (
                    b"none"
                    if delivery_context is None
                    else contract_bytes(delivery_context, redactor=requests._redactor)
                ),
                redactor=requests._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
                result_failure=lambda result: (
                    result.error if type(result) is PreparationReadFailure else None
                ),
                wait_for_settlement=wait_for_settlement,
            )
        )

    async def _inspect_service_owned(self, app, request, *, context):
        """Retain one authenticated read through its exact native outcome."""
        from cayu.collaboration._clarification_recovery import inspect_settled_service
        from cayu.runtime._session_continuation import ContinuationConflict, ContinuationUnavailable

        requests = self.requests
        request = prepare_contract(
            ClarificationServiceRequest, request, redactor=requests._redactor
        )
        context = prepare_contract(CollaborationAccessContext, context, redactor=requests._redactor)

        async def owned():
            try:
                return await requests._dependency(
                    lambda: inspect_settled_service(self, app, request, context=context)
                )
            except ContinuationConflict as error:
                failure = CollaborationConflict("Service inspection selection conflicts.")
                failure.__cause__ = _safe_request_failure(error, requests._redactor)
            except ContinuationUnavailable as error:
                failure = CollaborationUnavailable("Service inspection remains unresolved.")
                failure.__cause__ = _safe_request_failure(error, requests._redactor)
            raise failure

        return await requests._observe(
            requests._owners.run(
                owned,
                key=("clarification-service-inspection", object()),
                expectation=contract_bytes(request, redactor=requests._redactor)
                + contract_bytes(context, redactor=requests._redactor),
                redactor=requests._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
                wait_for_settlement=True,
            )
        )

    async def _inspect_maintenance_owned(self, app, expected, *, context):
        """Retained exact readback, without repeating a maintenance mutation."""
        from cayu.collaboration._clarification_delivery_recovery import reconcile_pending_delivery
        from cayu.collaboration._clarification_question_recovery import expire_question
        from cayu.collaboration._clarification_recovery import inspect_settled_service
        from cayu.collaboration._clarification_recovery_types import (
            ClarificationDeliveryRecovery,
            ClarificationExpiryRequest,
            ClarificationServiceRecovery,
        )

        if type(expected) not in (
            ClarificationDeliveryRecovery,
            ClarificationExpiryRequest,
            ClarificationServiceRecovery,
        ):
            raise TypeError("Maintenance inspection requires an exact recovery selector.")
        requests = self.requests
        expected = prepare_contract(type(expected), expected, redactor=requests._redactor)
        context = prepare_contract(CollaborationAccessContext, context, redactor=requests._redactor)

        async def owned():
            if type(expected) is ClarificationExpiryRequest:
                return await expire_question(self, expected, context=context, read_only=True)
            if type(expected) is ClarificationDeliveryRecovery:
                return await reconcile_pending_delivery(
                    self, expected, context=context, read_only=True
                )
            return await inspect_settled_service(self, app, expected, context=context)

        return await requests._observe(
            requests._owners.run(
                owned,
                key=("clarification-maintenance-inspection", object()),
                expectation=contract_bytes(expected, redactor=requests._redactor)
                + contract_bytes(context, redactor=requests._redactor),
                redactor=requests._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
                wait_for_settlement=True,
            )
        )

    async def reconcile_service(
        self, app, request, *, context, exclude=False, wait_for_settlement=False
    ):
        from cayu.collaboration._clarification_recovery import (
            ServiceRecoveryInput,
            reconcile_service,
        )
        from cayu.runtime._session_continuation import ContinuationConflict, ContinuationUnavailable

        requests = self.requests
        request = prepare_contract(
            ServiceRecoveryInput, {"request": request}, redactor=requests._redactor
        ).request
        context = prepare_contract(CollaborationAccessContext, context, redactor=requests._redactor)

        async def owned():
            try:
                return await requests._dependency(
                    lambda: reconcile_service(self, app, request, context=context, exclude=exclude)
                )
            except ContinuationConflict as error:
                failure = CollaborationConflict("Service reconciliation selection conflicts.")
                failure.__cause__ = _safe_request_failure(error, requests._redactor)
            except ContinuationUnavailable as error:
                failure = CollaborationUnavailable("Service reconciliation remains unresolved.")
                failure.__cause__ = _safe_request_failure(error, requests._redactor)
            raise failure

        return await requests._observe(
            requests._owners.run(
                owned,
                key=(
                    "clarification-service-exclusion"
                    if exclude
                    else "clarification-service-reconciliation",
                    object(),
                ),
                expectation=contract_bytes(request, redactor=requests._redactor)
                + contract_bytes(context, redactor=requests._redactor)
                + (b"exclude" if exclude else b"reconcile"),
                redactor=requests._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
                wait_for_settlement=wait_for_settlement,
            )
        )

    async def due_questions(self, *, context, cursor=None, limit=32, wait_for_settlement=False):
        from cayu.collaboration._clarification_question_recovery import due_questions
        from cayu.collaboration._clarification_recovery_types import (
            ClarificationPendingServiceQuery,
        )

        requests = self.requests
        context = prepare_contract(CollaborationAccessContext, context, redactor=requests._redactor)
        query = prepare_contract(
            ClarificationPendingServiceQuery,
            {"cursor": cursor, "limit": limit},
            redactor=requests._redactor,
        )
        return await requests._observe(
            requests._owners.run(
                lambda: requests._dependency(lambda: due_questions(self, query, context=context)),
                key=("clarification-due-questions", object()),
                wait_for_settlement=wait_for_settlement,
                expectation=contract_bytes(query, redactor=requests._redactor)
                + contract_bytes(context, redactor=requests._redactor),
                redactor=requests._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
            )
        )

    async def expire_question(self, request, *, context, wait_for_settlement=False):
        from cayu.collaboration._clarification_question_recovery import expire_question
        from cayu.collaboration._clarification_recovery_types import ClarificationExpiryRequest

        requests = self.requests
        request = prepare_contract(ClarificationExpiryRequest, request, redactor=requests._redactor)
        context = prepare_contract(CollaborationAccessContext, context, redactor=requests._redactor)
        return await requests._observe(
            requests._owners.run(
                lambda: requests._dependency(
                    lambda: expire_question(self, request, context=context)
                ),
                key=("clarification-question-expiry", object()),
                expectation=contract_bytes(request, redactor=requests._redactor)
                + contract_bytes(context, redactor=requests._redactor),
                redactor=requests._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
                wait_for_settlement=wait_for_settlement,
            )
        )

    async def pending_deliveries(
        self, *, context, cursor=None, limit=32, wait_for_settlement=False
    ):
        from cayu.collaboration._clarification_delivery_recovery import pending_deliveries
        from cayu.collaboration._clarification_recovery_types import (
            ClarificationPendingServiceQuery,
        )

        requests = self.requests
        context = prepare_contract(CollaborationAccessContext, context, redactor=requests._redactor)
        query = prepare_contract(
            ClarificationPendingServiceQuery,
            {"cursor": cursor, "limit": limit},
            redactor=requests._redactor,
        )
        return await requests._observe(
            requests._owners.run(
                lambda: requests._dependency(
                    lambda: pending_deliveries(
                        self, context=context, cursor=query.cursor, limit=query.limit
                    )
                ),
                key=("clarification-pending-deliveries", object()),
                wait_for_settlement=wait_for_settlement,
                expectation=contract_bytes(query, redactor=requests._redactor)
                + contract_bytes(context, redactor=requests._redactor),
                redactor=requests._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
            )
        )

    async def reconcile_delivery(
        self, recovery, *, context, exclude=False, wait_for_settlement=False
    ):
        from cayu.collaboration._clarification_delivery_recovery import reconcile_pending_delivery
        from cayu.collaboration._clarification_recovery_types import ClarificationDeliveryRecovery

        requests = self.requests
        recovery = prepare_contract(
            ClarificationDeliveryRecovery, recovery, redactor=requests._redactor
        )
        context = prepare_contract(CollaborationAccessContext, context, redactor=requests._redactor)
        return await requests._observe(
            requests._owners.run(
                lambda: requests._dependency(
                    lambda: reconcile_pending_delivery(
                        self, recovery, context=context, exclude=exclude
                    )
                ),
                key=(
                    "clarification-delivery-exclusion"
                    if exclude
                    else "clarification-delivery-reconciliation",
                    object(),
                ),
                expectation=(b"exclude:" if exclude else b"reconcile:")
                + contract_bytes(recovery, redactor=requests._redactor)
                + contract_bytes(context, redactor=requests._redactor),
                redactor=requests._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
                wait_for_settlement=wait_for_settlement,
            )
        )

    async def pending_services(self, *, context, cursor=None, limit=32, wait_for_settlement=False):
        from cayu.collaboration._clarification_recovery import pending_services
        from cayu.collaboration._clarification_recovery_types import (
            ClarificationPendingServiceQuery,
        )

        requests = self.requests
        context = prepare_contract(CollaborationAccessContext, context, redactor=requests._redactor)
        query = prepare_contract(
            ClarificationPendingServiceQuery,
            {"cursor": cursor, "limit": limit},
            redactor=requests._redactor,
        )

        async def owned():
            return await requests._dependency(
                lambda: pending_services(
                    self, context=context, cursor=query.cursor, limit=query.limit
                )
            )

        return await requests._observe(
            requests._owners.run(
                owned,
                key=("clarification-service-discovery", object()),
                wait_for_settlement=wait_for_settlement,
                expectation=contract_bytes(query, redactor=requests._redactor)
                + contract_bytes(context, redactor=requests._redactor),
                redactor=requests._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
            )
        )

    async def inspect_services(self, app, ticket, *, context, cursor=None, limit=32):
        from cayu.collaboration._clarification_recovery import inspect_services
        from cayu.collaboration._clarification_recovery_types import (
            ClarificationPendingServiceQuery,
        )
        from cayu.runtime._session_continuation import (
            ContinuationConflict,
            ContinuationTicket,
            ContinuationUnavailable,
        )

        requests = self.requests
        ticket = prepare_contract(ContinuationTicket, ticket, redactor=requests._redactor)
        context = prepare_contract(CollaborationAccessContext, context, redactor=requests._redactor)
        query = prepare_contract(
            ClarificationPendingServiceQuery,
            {"cursor": cursor, "limit": limit},
            redactor=requests._redactor,
        )

        async def owned():
            try:
                return await requests._dependency(
                    lambda: inspect_services(
                        self, app, ticket, context=context, cursor=query.cursor, limit=query.limit
                    )
                )
            except ContinuationConflict as error:
                failure = CollaborationConflict("Native service inspection conflicts.")
                failure.__cause__ = _safe_request_failure(error, requests._redactor)
            except ContinuationUnavailable as error:
                failure = CollaborationUnavailable("Native service inspection remains unavailable.")
                failure.__cause__ = _safe_request_failure(error, requests._redactor)
            raise failure

        return await requests._observe(
            requests._owners.run(
                owned,
                key=("clarification-native-service-inspection", object()),
                expectation=contract_bytes(ticket, redactor=requests._redactor)
                + contract_bytes(query, redactor=requests._redactor)
                + contract_bytes(context, redactor=requests._redactor),
                redactor=requests._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
            )
        )

    async def deliver(
        self,
        intent: ClarificationDeliveryIntent,
        *,
        context: SessionExportAccessContext,
        prepare_only: bool = False,
        wait_for_settlement: bool = False,
    ) -> ClarificationDeliveryReceipt:
        requests = self.requests
        value = prepare_contract(
            _Delivery,
            {"intent": intent, "context": context, "prepare_only": prepare_only},
            redactor=requests._redactor,
        )

        async def owned():
            try:
                return await requests._dependency(lambda: self._deliver_held(value))
            except SessionExportDenied as error:
                failure = CollaborationAccessDenied("Clarification delivery disclosure was denied.")
                failure.__cause__ = _safe_request_failure(error, requests._redactor)
            raise failure

        return await requests._observe(
            requests._owners.run(
                owned,
                key=("clarification-delivery", object()),
                expectation=contract_bytes(value, redactor=requests._redactor),
                redactor=requests._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
                wait_for_settlement=wait_for_settlement,
            )
        )

    async def _deliver_held(self, value: _Delivery) -> ClarificationDeliveryReceipt:
        requests, exports = self.requests, self.exports
        redactor = requests._redactor
        intent, context = value.intent, value.context
        if (
            context.mandate is None
            or context.mandate.participant != intent.sender
            or requests._resolver_ref is None
            or exports.mandate_ref != requests._resolver_ref
        ):
            raise CollaborationAccessDenied("Delivery requires current source mandate authority.")
        require_exact_contract(intent.initiator, _initiator(context.mandate), redactor=redactor)
        participants = requests._participants
        store, initialized = participants._ready()
        participants._capability(store, initialized, mutation=True, family=REQUEST_FAMILY)
        peer_context = CollaborationAccessContext(principal=context.principal)
        _, grant = participants._authorize(peer_context, "request_accept")
        participants._require_refs(grant, (intent.sender, intent.recipient))
        source = intent.question.source if intent.reply is None else intent.reply.source
        record = None
        async with acquire_clarification_source(
            exports,
            intent.export,
            context=context,
            sender=intent.sender,
            audience=intent.recipient,
            expected=source,
        ) as projection:
            async with store._transaction(initialized.binding.application_scope, write=False) as tx:
                opening = prepare_contract(
                    ClarificationOpenReceipt,
                    await tx.get("operations", operation_key(intent.question.operation)),
                    redactor=redactor,
                )
                require_exact_contract(intent.question, opening.command.question, redactor=redactor)
            await self._validate_source_use(
                projection, context, opening.command.expected, store, initialized
            )
            async with store._transaction(initialized.binding.application_scope, write=True) as tx:
                record = await register_delivery_in_transaction(
                    store,
                    tx,
                    initialized,
                    intent,
                    authority_expires_at_ms=projection.expires_at_ms,
                    redactor=redactor,
                )
        # Do not recursively acquire a policy's non-reentrant revocation lock.
        # Pending ownership is durable before releasing the first source guard;
        # actual peer append independently reacquires current export authority.
        if record is None:
            raise CollaborationUnavailable("Delivery guard produced no durable responsibility.")
        if value.prepare_only:
            return ClarificationDeliveryReceipt.from_record(record)
        if record.state == "pending":
            record = await reconcile_delivery(
                store, initialized, intent, exports.store, redactor=redactor
            )
        if record.state == "pending":
            await self.append_peer(intent.append, context=peer_context)
            record = await reconcile_delivery(
                store, initialized, intent, exports.store, redactor=redactor
            )
        return ClarificationDeliveryReceipt.from_record(record)

    async def _validate_source_use(self, projection, context, expected, store, initialized):
        requests = self.requests
        resolution = projection.authorization.mandate
        if resolution is None or requests._resolver_ref is None:
            raise CollaborationAccessDenied("Clarification requires current mandate authority.")
        async with store._transaction(initialized.binding.application_scope, write=False) as tx:
            now = await tx.now_ms()
        # Reuse the held export/mandate guard, including non-reentrant resolvers.
        await asyncio.to_thread(
            partial(
                validate_mandate_resolution,
                resolution,
                context=context.mandate,
                resolver=requests._resolver_ref,
                use=MandateUse(
                    audience=initialized.owner,
                    scope=initialized.binding.application_scope,
                    actions=("consult", "readback"),
                    resources=tuple(
                        ResourceSelector(resource=ref) for ref in expected.intent.request.inputs
                    ),
                    inputs=tuple(
                        MandateInput(source=ref, channel="prompt")
                        for ref in expected.intent.request.inputs
                    ),
                ),
                now_ms=now,
                resource_owners=requests._resource_owners,
                redactor=requests._redactor,
            )
        )

    async def open(
        self,
        command: ClarificationOpenCommand,
        source: SessionExportRequest,
        *,
        context: SessionExportAccessContext,
    ) -> ClarificationOpenReceipt:
        requests = self.requests
        value = prepare_contract(
            _Opening,
            {"command": command, "source": source, "context": context},
            redactor=requests._redactor,
        )

        async def owned():
            try:
                return await requests._dependency(lambda: self._open_held(value))
            except SessionExportDenied as error:
                failure = CollaborationAccessDenied("Clarification source disclosure was denied.")
                failure.__cause__ = _safe_request_failure(error, requests._redactor)
            raise failure

        return await requests._observe(
            requests._owners.run(
                owned,
                # Authorization is per observer; only the transaction shares replay.
                key=("clarification-open", object()),
                expectation=contract_bytes(value, redactor=requests._redactor),
                redactor=requests._redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, requests._redactor),
            )
        )

    async def _open_planned(self, command, source, *, context, planned):
        """Called only inside the planning coordinator's owned mutation task."""
        from cayu.collaboration._planning_stages import _PlannedStage

        if type(planned) is not _PlannedStage:
            raise CollaborationAccessDenied("Planning opening lacks its internal owner.")
        value = prepare_contract(
            _Opening,
            {"command": command, "source": source, "context": context},
            redactor=self.requests._redactor,
        )
        try:
            return await self._open_held(value, _planned_stage=planned)
        except SessionExportDenied as error:
            failure = CollaborationAccessDenied("Clarification source disclosure was denied.")
            failure.__cause__ = _safe_request_failure(error, self.requests._redactor)
        raise failure

    async def _open_held(self, value: _Opening, *, _planned_stage=None) -> ClarificationOpenReceipt:
        requests, exports = self.requests, self.exports
        redactor = requests._redactor
        command, context = value.command, value.context
        question = command.question
        selected = command.expected.intent.selection
        registration = requests._registration
        if (
            registration is None
            or registration.receiving_owner is None
            or requests._receiving_ref is None
            or context.mandate is None
            or context.mandate.participant != selected.recipient.reference
            or exports.mandate_ref != requests._resolver_ref
            or self.common_root_enabled is not True
        ):
            raise CollaborationAccessDenied("Clarification receiving authority is unavailable.")
        require_exact_contract(question.initiator, _initiator(context.mandate), redactor=redactor)
        require_exact_contract(question.receiver, requests._receiving_ref, redactor=redactor)
        require_exact_contract(
            question.receiver,
            prepare_contract(ObjectRef, registration.receiving_owner.ref, redactor=redactor),
            redactor=redactor,
        )
        policy = requests._require_clarification_policy(question.policy)
        participants = requests._participants
        store, initialized = participants._ready()
        participants._capability(store, initialized, mutation=True, family=REQUEST_FAMILY)
        _, grant = participants._authorize(
            CollaborationAccessContext(principal=context.principal), "request_accept"
        )
        participants._require_refs(grant, (selected.sender.reference, selected.recipient.reference))
        async with acquire_clarification_source(
            exports,
            value.source,
            context=context,
            sender=selected.recipient.reference,
            audience=selected.sender.reference,
            expected=question.source,
        ) as projection:
            await self._validate_source_use(
                projection, context, command.expected, store, initialized
            )
            binding = await self.resolve_budget(
                request={
                    "kind": "clarification",
                    "session_id": value.source.ref.session_id,
                    "session_instance_id": value.source.ref.session_instance_id,
                    "operation": question.operation,
                    "lineage": question.lineage,
                }
            )
            if type(binding) is not BudgetBinding:
                raise CollaborationUnavailable("The budget receiver returned invalid authority.")
            # Revalidate before digest serialization: a trusted extension can
            # still accidentally return a post-construction-mutated model.
            binding = revalidate_model_input(binding, BudgetBinding)
            if (
                binding.application_scope != initialized.binding.application_scope
                or binding.authority_digest != question.budget_authority_sha256
                or question.budget_binding
                != ObjectRef(
                    owner=initialized.owner,
                    kind="budget_binding",
                    object_id=binding.binding_id,
                    incarnation=binding.authority_digest,
                    revision=1,
                )
            ):
                raise CollaborationConflict("Clarification common-root authority conflicts.")
            required = {binding.root_budget_id, *binding.ancestor_budget_ids}
            ceilings = tuple(
                limit
                for limit in binding.limits
                if limit.scope == "causal" and limit.key in required
            )
            if (
                len(ceilings) != len(required)
                or any(
                    limit.currency != "USD"
                    or limit.window.kind != "all_time"
                    or limit.reservation is None
                    or limit.allow_unpriced
                    for limit in ceilings
                )
                or next(
                    limit for limit in ceilings if limit.key == binding.root_budget_id
                ).max_estimated_cost
                > Decimal(policy.max_spend_usd)
            ):
                raise CollaborationAccessDenied(
                    "Clarification lacks its finite cumulative ceiling."
                )
            async with store._transaction(initialized.binding.application_scope, write=True) as tx:
                if await tx.now_ms() >= projection.expires_at_ms:
                    raise CollaborationAccessDenied(
                        "Clarification authority expired before publication."
                    )
                if await tx.get("operations", operation_key(command.operation)) is None:
                    for selected_participant in (selected.sender, selected.recipient):
                        current = await store._participant(
                            tx, selected_participant.reference, initialized.owner, redactor
                        )
                        if current.lifecycle != "active" or (
                            current.configuration_revision,
                            current.lifecycle_revision,
                            current.admission_generation,
                        ) != (
                            selected_participant.configuration_revision,
                            selected_participant.lifecycle_revision,
                            selected_participant.admission_generation,
                        ):
                            raise CollaborationAccessDenied(
                                "Clarification participant authority changed."
                            )
                return await open_in_transaction(
                    store,
                    tx,
                    initialized,
                    command,
                    redactor=redactor,
                    _planned_stage=_planned_stage,
                )
        raise CollaborationUnavailable("Clarification authorization did not produce a decision.")
