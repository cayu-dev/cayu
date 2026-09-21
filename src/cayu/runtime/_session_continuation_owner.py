"""Registered session receiving owner for durable continuation evidence.

Registration is application/runtime configuration, never request data. Foreign
authentication executes outside private store publication scopes. Observation
can stop while the bounded owner retains in-flight receiving work.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from hashlib import sha256
from typing import TYPE_CHECKING, TypeVar

from cayu.collaboration._capabilities import CapabilityDescriptor, FamilyVersion, require_capability
from cayu.collaboration._contracts import (
    CollaborationConflict,
    ExpectedOperation,
    HandoffIntent,
    HandoffSlot,
    InitiatorBinding,
    OperationRef,
    OwnerRef,
)
from cayu.collaboration._diagnostics import safe_failure
from cayu.collaboration._ownership import _MutationOwners
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration.participants import CollaborationCapacityExceeded, CollaborationUnavailable
from cayu.runtime._invocation_lifecycle import (
    AdmitInvocationCommand,
    AdmittedInvocationBinding,
    InvocationContext,
    InvocationMutationResult,
    PreparedInvocationBinding,
    copy_invocation_lifecycle_command,
)
from cayu.runtime._session_continuation import (
    ContinuationConflict,
    ContinuationConsumption,
    ContinuationLatch,
    ContinuationLatchReceiver,
    ContinuationNamespace,
    ContinuationPreparation,
    ContinuationRecord,
    ContinuationRetirement,
    ContinuationService,
    ContinuationTicket,
    ContinuationUnavailable,
    ContinuationWait,
    admit_continuation,
    continuation_admission_digest,
    continuation_admission_inputs,
    continuation_namespace_id,
    continuation_operation_key,
    continuation_registration_operation,
    require_latch_identity,
    require_ticket_identity,
)
from cayu.runtime._session_continuation_scope import (
    authenticated_latch_scope,
    consumption_scope,
    park_scope,
    preparation_scope,
    require_ticket_invocation,
    retirement_scope,
)
from cayu.sessions.base import ResumeRequest, SessionStore, copy_resume_request
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from cayu.applications import CayuApp

LATCH_FAMILY = FamilyVersion(family="session.continuation.latch", version=1)
T = TypeVar("T")


def _failure_graph(error: BaseException, redactor: SecretRedactor) -> BaseException:
    seen: set[int] = set()

    def copy(value: BaseException, depth: int) -> BaseException | None:
        if id(value) in seen:
            return None
        if len(seen) >= 64 or depth >= 16:
            return RuntimeError("Additional continuation failure evidence withheld.")
        seen.add(id(value))
        if isinstance(value, BaseExceptionGroup):
            children = []
            for child in value.exceptions:
                if len(seen) >= 64:
                    children.append(RuntimeError("Additional failure evidence withheld."))
                    break
                copied = copy(child, depth + 1)
                if copied is not None:
                    children.append(copied)
            result = (
                BaseExceptionGroup("Continuation dependency failures.", children)
                if children
                else RuntimeError("Shared failure evidence already represented.")
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


class SessionContinuationOwner:
    """Owner-level receiving service, not a wait arbiter or automatic worker."""

    def __init__(
        self,
        *,
        store: SessionStore,
        owner: OwnerRef,
        receiver: ContinuationLatchReceiver,
        receiver_capability: CapabilityDescriptor,
        redactor: SecretRedactor,
    ) -> None:
        if not store._supports_session_continuation_protocol():
            raise ContinuationUnavailable("Session store does not qualify continuation ownership.")
        self.store = store
        self.redactor = redactor
        self.owner = prepare_contract(OwnerRef, owner, redactor=redactor)
        self.receiver = receiver
        self.receiver_capability = prepare_contract(
            CapabilityDescriptor, receiver_capability, redactor=redactor
        )
        require_capability(
            self.receiver_capability,
            expected_owner=self.receiver_capability.owner,
            required=LATCH_FAMILY,
            supported=(LATCH_FAMILY,),
            access="readback",
            redactor=redactor,
        )
        self.owners = _MutationOwners()

    async def _observe(
        self, operation: Callable[[], Awaitable[T]], *, key: tuple[object, ...], expected: bytes
    ) -> T:
        async def run() -> T:
            failure: BaseException
            try:
                return await operation()
            except asyncio.CancelledError as error:
                failure = ContinuationUnavailable(
                    "Receiving dependency cancelled; reconcile the latch."
                )
                failure.__cause__ = _failure_graph(error, self.redactor)
            except BaseExceptionGroup as error:
                fatal, _ = error.split(
                    lambda value: (
                        not isinstance(
                            value, (Exception, asyncio.CancelledError, BaseExceptionGroup)
                        )
                    )
                )
                if fatal is not None:
                    raise
                failure = ContinuationUnavailable(
                    "Receiving dependency failed; reconcile the latch."
                )
                failure.__cause__ = _failure_graph(error, self.redactor)
            except Exception as error:
                kind = (
                    ContinuationConflict
                    if isinstance(error, ContinuationConflict)
                    else PermissionError
                    if isinstance(error, PermissionError)
                    else ContinuationUnavailable
                )
                failure = kind("Continuation receiving operation failed.")
                failure.__cause__ = _failure_graph(error, self.redactor)
            raise failure

        try:
            return await self.owners.run(
                run,
                key=key,
                expectation=expected,
                redactor=self.redactor,
                failure_snapshot=lambda error: _failure_graph(error, self.redactor),
            )
        except CollaborationConflict:
            failure = ContinuationConflict("An owned continuation operation has different intent.")
        except (CollaborationUnavailable, CollaborationCapacityExceeded):
            failure = ContinuationUnavailable("Continuation acknowledgement remains pending.")
        raise failure

    async def latch(self, candidate: ContinuationLatch) -> ContinuationRecord:
        latch = prepare_contract(ContinuationLatch, candidate, redactor=self.redactor)
        if latch.ticket.owner != self.owner:
            raise PermissionError("Continuation belongs to a different session owner.")

        async def receive() -> ContinuationRecord:
            retained = await self.store.load_continuation_ticket(
                latch.ticket.session_id,
                registration_key=latch.ticket.registration_key,
                session_instance_id=latch.ticket.session_instance_id,
            )
            if retained is None:
                raise ContinuationConflict("Continuation is not durably prepared.")
            require_ticket_identity(retained.ticket, latch.ticket)
            if (
                retained.preparation.registration.child.destination
                != self.receiver_capability.owner
            ):
                raise PermissionError("Continuation wait belongs to another registered receiver.")
            if retained.latch is not None:
                require_latch_identity(retained.latch, latch)
                return retained
            authenticated = prepare_contract(
                ContinuationLatch,
                await self.receiver.authenticate_continuation_latch(latch),
                redactor=self.redactor,
            )
            require_latch_identity(latch, authenticated)
            with authenticated_latch_scope(authenticated):
                return await self.store.latch_continuation(authenticated)

        return await self._observe(
            receive,
            key=(latch.ticket.session_id, continuation_operation_key(latch.ticket), "latch"),
            expected=contract_bytes(latch, redactor=self.redactor),
        )

    async def prepare(
        self, intent: ContinuationWait, *, invocation: InvocationContext
    ) -> ContinuationRecord:
        """Retain ticket and registration intent while the actual writer is active."""
        if (
            type(invocation) is not InvocationContext
            or type(invocation.binding) is not AdmittedInvocationBinding
        ):
            raise PermissionError(
                "Continuation preparation requires an admitted runtime invocation."
            )
        invocation.require_runtime_authority()
        intent = prepare_contract(ContinuationWait, intent, redactor=self.redactor)
        binding = invocation.binding
        namespace = ContinuationNamespace(
            owner=self.owner,
            session_id=binding.session_id,
            session_instance_id=binding.session_instance_id,
            namespace_id=continuation_namespace_id(
                binding.session_id, binding.session_instance_id, self.owner
            ),
        )
        ticket = prepare_contract(
            ContinuationTicket,
            {
                **intent.model_dump(mode="json"),
                "namespace": namespace,
                "owner": self.owner,
                "session_id": binding.session_id,
                "session_instance_id": binding.session_instance_id,
                "interaction_id": binding.interaction_id,
                "writer_generation": binding.run_epoch,
                "state": "ARMING",
                "revision": 1,
            },
            redactor=self.redactor,
        )

        async def prepare() -> ContinuationRecord:
            session = await self.store.load(binding.session_id)
            if session is None or session.instance_id != binding.session_instance_id:
                raise PermissionError("Continuation session incarnation is unavailable.")
            operation = OperationRef(
                application_scope=self.owner.application_scope,
                namespace_incarnation=namespace.namespace_id,
                generation=namespace.generation,
                caller_key=intent.registration_key,
            )
            initiator = InitiatorBinding(
                issuer=self.owner,
                principal=self.owner.owner_id,
                invocation_id=session.invocation.root_invocation_id,
                interaction_id=binding.interaction_id,
                participant=None,
                mandate=None,
            )
            command = ContinuationPreparation(
                operation=operation,
                source=self.owner,
                destination=self.owner,
                initiator=initiator,
                intent=ticket,
                registration=HandoffIntent[ContinuationTicket](
                    slot=HandoffSlot(source=self.owner, parent=operation, slot="wait-registration"),
                    child=ExpectedOperation[ContinuationTicket](
                        operation=continuation_registration_operation(operation),
                        kind="wait.register",
                        schema_version=1,
                        mode="wait",
                        source=self.owner,
                        destination=self.receiver_capability.owner,
                        initiator=initiator,
                        receipt_stage="registered",
                        intent=ticket,
                    ),
                ),
            )
            with preparation_scope(command, invocation):
                return await self.store.prepare_continuation_ticket(command)

        return await self._observe(
            prepare,
            key=(binding.session_id, continuation_operation_key(ticket), "prepare"),
            expected=contract_bytes(ticket, redactor=self.redactor)
            + invocation.profile.fingerprint.encode(),
        )

    async def park(
        self, candidate: ContinuationTicket, *, invocation: InvocationContext
    ) -> ContinuationRecord:
        """Acknowledge parking without releasing or replacing the session writer."""
        ticket = prepare_contract(ContinuationTicket, candidate, redactor=self.redactor)
        if ticket.owner != self.owner:
            raise PermissionError("Continuation belongs to another registered session owner.")
        require_ticket_invocation(ticket, invocation)

        async def publish() -> ContinuationRecord:
            with park_scope(ticket, invocation):
                return await self.store.mark_continuation_waiting(ticket)

        return await self._observe(
            publish,
            key=(ticket.session_id, continuation_operation_key(ticket), "park"),
            expected=contract_bytes(ticket, redactor=self.redactor),
        )

    async def retire(
        self, candidate: ContinuationRetirement, *, invocation: InvocationContext | None = None
    ) -> ContinuationRecord:
        """Retire with writer authority, or exclude an overtaken unconsumed wait.

        Registered owners may recover supersession without retaining an old
        in-process invocation. The store must prove writer advancement atomically.
        This never authorizes admission or settlement of an in-flight consumption.
        """
        retirement = prepare_contract(ContinuationRetirement, candidate, redactor=self.redactor)
        if retirement.ticket.owner != self.owner:
            raise PermissionError("Continuation belongs to another registered session owner.")
        # Validate caller provenance before scheduling, but grant mutation scope
        # only inside the retained task that owns publication and readback.
        if invocation is None:
            if retirement.reason != "superseded":
                raise PermissionError("Continuation retirement requires its original invocation.")
        else:
            require_ticket_invocation(retirement.ticket, invocation)

        async def publish() -> ContinuationRecord:
            with retirement_scope(retirement, invocation):
                return await self.store.retire_continuation(retirement)

        return await self._observe(
            publish,
            key=(
                retirement.ticket.session_id,
                continuation_operation_key(retirement.ticket),
                "retire",
            ),
            expected=contract_bytes(retirement, redactor=self.redactor),
        )

    async def exclude(
        self, candidate: ContinuationRetirement, *, invocation: InvocationContext
    ) -> ContinuationRecord:
        """Record an exact destination refusal without dispatching admission."""
        retirement = prepare_contract(ContinuationRetirement, candidate, redactor=self.redactor)
        if retirement.reason not in {"failed", "expired", "unavailable", "superseded"}:
            raise ContinuationConflict("Continuation exclusion requires a refusal reason.")
        return await self.retire(retirement, invocation=invocation)

    async def admit(
        self,
        candidate: ContinuationConsumption,
        command: AdmitInvocationCommand,
        *,
        invocation: InvocationContext,
    ) -> tuple[InvocationMutationResult, ContinuationRecord]:
        """Receive a runtime-prepared invocation; never compute policy gates here."""
        if (
            type(invocation) is not InvocationContext
            or type(invocation.binding) is not PreparedInvocationBinding
        ):
            raise PermissionError("Continuation admission requires runtime preparation authority.")
        invocation.require_runtime_authority()
        if type(command) is not AdmitInvocationCommand:
            raise PermissionError("Continuation admission requires the typed receiving command.")
        failure = None
        try:
            copied_command = copy_invocation_lifecycle_command(command)
        except Exception as error:
            failure = ContinuationConflict("Continuation admission command is invalid.")
            failure.__cause__ = _failure_graph(error, self.redactor)
        if failure is not None:
            raise failure
        if type(copied_command) is not AdmitInvocationCommand:
            raise PermissionError("Continuation admission requires the typed receiving command.")
        command = copied_command
        expected = prepare_contract(ContinuationConsumption, candidate, redactor=self.redactor)
        binding = invocation.binding
        if (
            expected.ticket.owner != self.owner
            or binding.session_id != expected.ticket.session_id
            or binding.session_instance_id != expected.ticket.session_instance_id
            or invocation.active_profile != command.target_active_profile
            or invocation.tool_capability_ceiling != command.tool_capability_ceiling
            or binding.run_epoch != command.expected_run_epoch + 1
        ):
            raise PermissionError("Continuation admission conflicts with runtime authority.")
        if (
            expected.input_digest,
            expected.profile_digest,
            expected.budget_digest,
        ) != continuation_admission_inputs(
            command
        ) or expected.admission_command_digest != continuation_admission_digest(command):
            raise ContinuationConflict(
                "Continuation admission attribution differs from its command."
            )

        async def dispatch() -> tuple[InvocationMutationResult, ContinuationRecord]:
            from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store

            with consumption_scope(expected):
                return await admit_continuation(
                    runtime_checkpoint_session_store(self.store), expected, command
                )

        return await self._observe(
            dispatch,
            key=(expected.ticket.session_id, continuation_operation_key(expected.ticket), "admit"),
            expected=contract_bytes(expected, redactor=self.redactor),
        )

    async def service(
        self, app: CayuApp, request: ResumeRequest, candidate: ContinuationService
    ) -> ContinuationRecord:
        """Run the existing resume path, or reconcile its exact prior admission.

        This is an explicitly invoked host service, not a scheduler. The supplied
        application owns all normal session gates and execution. Event output
        remains in the session's durable event stream rather than an unbounded
        secondary buffer here.
        """
        from cayu.runtime._session_continuation_resume import _ContinuationResumeHandoff

        service = prepare_contract(ContinuationService, candidate, redactor=self.redactor)
        if app.session_store is not self.store or service.ticket.owner != self.owner:
            raise PermissionError("Continuation service belongs to another application owner.")
        failure = None
        try:
            copied_request = copy_resume_request(request)
            request_digest = app._session_engine.work_attempt_source_request_sha256(
                copied_request, kind="continuation"
            )
        except Exception as error:
            failure = ContinuationConflict("Continuation resume request is invalid.")
            failure.__cause__ = _failure_graph(error, self.redactor)
        if failure is not None:
            raise failure
        if copied_request.session_id != service.ticket.session_id:
            raise PermissionError("Continuation request belongs to another session.")
        service_digest = sha256(
            contract_bytes(service, redactor=self.redactor) + request_digest.encode("ascii")
        ).hexdigest()

        async def receive() -> ContinuationRecord:
            retained = await self.store.load_continuation_ticket(
                service.ticket.session_id,
                registration_key=service.ticket.registration_key,
                session_instance_id=service.ticket.session_instance_id,
            )
            if retained is None or retained.latch is None:
                raise ContinuationConflict("Continuation has no retained ready wait.")
            require_ticket_identity(service.ticket, retained.ticket)
            require_latch_identity(service.latch, retained.latch)
            if retained.consumption is not None:
                consumption = retained.consumption
                if (
                    consumption.service_digest != service_digest
                    or consumption.continuation_id != service.continuation_id
                    or consumption.mode != service.mode
                    or consumption.accepted_at != service.accepted_at
                ):
                    raise ContinuationConflict("Continuation service was accepted differently.")
                if consumption.receipt_stage in {"admitted", "excluded"}:
                    return retained
                if consumption.receipt_stage == "prepared":
                    from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
                    from cayu.runtime._invocation_lifecycle import (
                        reconcile_invocation_admission_from_state,
                        superseding_invocation_admission_digest_from_state,
                    )

                    session = await self.store.load(service.ticket.session_id)
                    checkpoint = await runtime_checkpoint_session_store(self.store).load_checkpoint(
                        service.ticket.session_id
                    )
                    if session is None:
                        raise ContinuationUnavailable("Continuation session is unavailable.")
                    if (
                        superseding_invocation_admission_digest_from_state(
                            session,
                            checkpoint,
                            session_id=service.ticket.session_id,
                            session_instance_id=service.ticket.session_instance_id,
                            expected_run_epoch=consumption.admission_expected_run_epoch,
                            command_sha256=consumption.admission_command_digest,
                        )
                        is not None
                        or reconcile_invocation_admission_from_state(
                            session,
                            checkpoint,
                            session_id=service.ticket.session_id,
                            session_instance_id=service.ticket.session_instance_id,
                            expected_run_epoch=consumption.admission_expected_run_epoch,
                            command_sha256=consumption.admission_command_digest,
                            profile_sha256=consumption.profile_digest,
                        )
                        is not None
                    ):
                        return await self.reconcile_admission(
                            consumption.model_copy(update={"receipt_stage": "prepared"})
                        )
                    # Release rechecks the receipt in the admission transaction.
                    # Late dispatches check this exact claim ID and are fenced;
                    # absence alone is not being used as proof of worker death.
                    if consumption.admission_claimed:
                        with consumption_scope(consumption):
                            await self.store._release_continuation_admission_claim(consumption)
                # Re-enter normal gates with the exact retained command identity.
                # If the gates now refuse, responsibility remains unclaimed and
                # can be explicitly excluded rather than stranded as in-flight.
                handoff = _ContinuationResumeHandoff(self, service, service_digest)
                stream = app._resume_private(
                    copied_request,
                    store_resolved_session_id=service.ticket.session_id,
                    continuation_handoff=handoff,
                )
                try:
                    async for _ in stream:
                        pass
                finally:
                    await stream.aclose()
                result = await self.store.load_continuation_ticket(
                    service.ticket.session_id,
                    registration_key=service.ticket.registration_key,
                    session_instance_id=service.ticket.session_instance_id,
                )
                if result is None:
                    raise ContinuationUnavailable("Continuation service readback is unavailable.")
                return result
            if (
                retained.ticket.state != "WAITING"
                or retained.ticket.revision != service.ticket.revision
            ):
                raise ContinuationConflict("Continuation is no longer eligible for service.")
            handoff = _ContinuationResumeHandoff(self, service, service_digest)
            stream = app._resume_private(
                copied_request,
                store_resolved_session_id=service.ticket.session_id,
                continuation_handoff=handoff,
            )
            try:
                async for _ in stream:
                    pass
            finally:
                await stream.aclose()
            result = await self.store.load_continuation_ticket(
                service.ticket.session_id,
                registration_key=service.ticket.registration_key,
                session_instance_id=service.ticket.session_instance_id,
            )
            if result is None:
                raise ContinuationUnavailable("Continuation service readback is unavailable.")
            return result

        return await self._observe(
            receive,
            key=(service.ticket.session_id, continuation_operation_key(service.ticket), "service"),
            expected=service_digest.encode("ascii"),
        )

    async def reconcile_admission(self, candidate: ContinuationConsumption) -> ContinuationRecord:
        """Settle a retained handoff only from positive typed admission evidence."""
        expected = prepare_contract(ContinuationConsumption, candidate, redactor=self.redactor)
        if expected.ticket.owner != self.owner:
            raise PermissionError("Continuation belongs to another registered session owner.")
        if expected.receipt_stage != "prepared":
            raise ContinuationConflict("Reconciliation requires the original prepared handoff.")
        retained = await self.store.load_continuation_ticket(
            expected.ticket.session_id,
            registration_key=expected.ticket.registration_key,
            session_instance_id=expected.ticket.session_instance_id,
        )
        if retained is None or retained.consumption is None:
            raise ContinuationUnavailable("Continuation admission claim is unavailable.")
        require_ticket_identity(expected.ticket, retained.ticket)
        require_latch_identity(expected.latch, retained.consumption.latch)
        claimless = {
            "receipt_stage": "prepared",
            "admission_claimed": False,
            "admission_claim_id": None,
        }
        if retained.consumption.model_copy(
            update={"ticket": expected.ticket, "latch": expected.latch, **claimless}
        ) != expected.model_copy(update=claimless):
            raise ContinuationConflict("Continuation admission differs from its retained claim.")
        if retained.consumption.receipt_stage in {"admitted", "excluded"}:
            return retained
        expected = retained.consumption

        async def reconcile() -> ContinuationRecord:
            with consumption_scope(expected):
                from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
                from cayu.runtime._invocation_lifecycle import (
                    superseding_invocation_admission_digest_from_state,
                )

                session = await self.store.load(expected.ticket.session_id)
                checkpoint = await runtime_checkpoint_session_store(self.store).load_checkpoint(
                    expected.ticket.session_id
                )
                if session is None:
                    raise ContinuationUnavailable("Continuation session is unavailable.")
                if (
                    superseding_invocation_admission_digest_from_state(
                        session,
                        checkpoint,
                        session_id=expected.ticket.session_id,
                        session_instance_id=expected.ticket.session_instance_id,
                        expected_run_epoch=expected.admission_expected_run_epoch,
                        command_sha256=expected.admission_command_digest,
                    )
                    is not None
                ):
                    return await self.store._release_continuation_admission_claim(expected)
                if not expected.admission_claimed:
                    raise ContinuationUnavailable("Continuation admission claim is unavailable.")
                return await self.store._finalize_continuation_admission(
                    expected.model_copy(
                        update={"receipt_stage": "admitted", "admission_claimed": True}
                    )
                )

        return await self._observe(
            reconcile,
            key=(
                expected.ticket.session_id,
                continuation_operation_key(expected.ticket),
                "reconcile",
            ),
            expected=contract_bytes(expected, redactor=self.redactor),
        )

    async def drain(self) -> None:
        await self.owners.drain()
