"""Registered session receiving owner for durable continuation evidence.

Registration is application/runtime configuration, never request data. Foreign
authentication executes outside private store publication scopes. Observation
can stop while the bounded owner retains in-flight receiving work.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any, TypeVar

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
    InvocationContext,
)
from cayu.runtime._session_continuation import admit_continuation
from cayu.runtime._session_continuation_scope import (
    park_scope,
    preparation_scope,
    require_ticket_invocation,
    retirement_scope,
)
from cayu.runtime._temporary_continuation_permits import (
    TemporaryServicePermitAuthority,
    TemporaryServiceSettlementReader,
)
from cayu.runtime._temporary_continuation_scope import temporary_admission_scope
from cayu.sessions._invocation_lifecycle import (
    AdmitInvocationCommand,
    AdmittedInvocationBinding,
    InvocationMutationResult,
    PreparedInvocationBinding,
    copy_invocation_lifecycle_command,
    invocation_admission_command_sha256,
)
from cayu.sessions._session_continuation import (
    ContinuationConflict,
    ContinuationConsumption,
    ContinuationLatch,
    ContinuationLatchReceiver,
    ContinuationNamespace,
    ContinuationPreparation,
    ContinuationRecord,
    ContinuationReleasedRetirement,
    ContinuationRetirement,
    ContinuationService,
    ContinuationTicket,
    ContinuationUnavailable,
    ContinuationWait,
    continuation_admission_digest,
    continuation_admission_inputs,
    continuation_digest,
    continuation_namespace_id,
    continuation_operation_key,
    continuation_registration_operation,
    require_latch_identity,
    require_ticket_identity,
)
from cayu.sessions._session_continuation_scope import authenticated_latch_scope, consumption_scope
from cayu.sessions._temporary_continuation import (
    TemporaryServiceAdmission,
    TemporaryServiceDispatch,
    TemporaryServiceIntent,
    TemporaryServicePreparation,
    TemporaryServiceRecord,
    require_temporary_service_command,
    temporary_admission_payload_sha256,
    temporary_service_key,
)
from cayu.sessions.base import ResumeRequest, SessionStore, copy_resume_request
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.collaboration.access import CollaborationAccessContext

LATCH_FAMILY = FamilyVersion(family="session.continuation.latch", version=1)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class _ContinuationServiceResult:
    """Native service distinguishes its own execution from admission readback.

    An admitted replay can belong to a still-running invocation supervised by
    another owner. It is not release evidence for that invocation.
    """

    record: ContinuationRecord
    dispatched: bool


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


class _ContinuationServiceNotStarted(Exception):
    """Private evidence that this service finished before entering admission."""

    def __init__(self, failure: Exception):
        super().__init__("Continuation service did not enter admission.")
        self.failure = failure


@dataclass(frozen=True)
class _ContinuationServiceReadFailure:
    """Completed initial read, with no admission dispatched by this service turn."""

    failure: Exception


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
        temporary_permits: TemporaryServicePermitAuthority | None = None,
        track: Callable[[asyncio.Task[Any]], None] | None = None,
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
        # Lets the owning application's shutdown wait for work this owner retains;
        # None when the owner is used on its own, outside an application.
        self._track = track
        self.temporary_permits = temporary_permits

    async def _observe(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        key: tuple[object, ...],
        expected: bytes,
        wait_for_settlement: bool = False,
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
                track=self._track,
                failure_snapshot=lambda error: _failure_graph(error, self.redactor),
                result_failure=lambda result: (
                    result.failure if type(result) is _ContinuationServiceReadFailure else None
                ),
                wait_for_settlement=wait_for_settlement,
            )
        except CollaborationConflict:
            failure = ContinuationConflict("An owned continuation operation has different intent.")
        except (CollaborationUnavailable, CollaborationCapacityExceeded):
            failure = ContinuationUnavailable("Continuation acknowledgement remains pending.")
        raise failure

    async def latch(self, candidate: ContinuationLatch) -> ContinuationRecord:
        return await self._receive_latch(candidate, wait_for_settlement=False)

    async def _latch_owned(self, candidate: ContinuationLatch) -> ContinuationRecord:
        """Retained runtime observation; authentication is identical to latch()."""
        return await self._receive_latch(candidate, wait_for_settlement=True)

    async def _read_latch_record(self, latch):
        retained = await self.store.load_continuation_ticket(
            latch.ticket.session_id,
            registration_key=latch.ticket.registration_key,
            session_instance_id=latch.ticket.session_instance_id,
        )
        if retained is None:
            raise ContinuationConflict("Continuation is not durably prepared.")
        require_ticket_identity(retained.ticket, latch.ticket)
        if retained.preparation.registration.child.destination != self.receiver_capability.owner:
            raise PermissionError("Continuation wait belongs to another registered receiver.")
        if retained.latch is not None:
            require_latch_identity(retained.latch, latch)
        return retained

    async def _inspect_latch_owned(self, candidate: ContinuationLatch):
        """Exact retained latch readback; cannot create a latch or start execution."""
        from cayu.collaboration._wait_coordinator import CollaborationWaitLatchReceiver

        latch = prepare_contract(ContinuationLatch, candidate, redactor=self.redactor)
        if (
            latch.ticket.owner != self.owner
            or type(self.receiver) is not CollaborationWaitLatchReceiver
        ):
            raise PermissionError("Retained latch readback requires its registered receiver.")

        async def inspect():
            retained = await self._read_latch_record(latch)
            return retained if retained.latch is not None else None

        return await self._observe(
            inspect,
            key=(
                latch.ticket.session_id,
                continuation_operation_key(latch.ticket),
                "inspect-latch",
            ),
            expected=contract_bytes(latch, redactor=self.redactor),
            wait_for_settlement=True,
        )

    async def _receive_latch(self, candidate, *, wait_for_settlement):
        latch = prepare_contract(ContinuationLatch, candidate, redactor=self.redactor)
        if latch.ticket.owner != self.owner:
            raise PermissionError("Continuation belongs to a different session owner.")
        authenticate = self.receiver.authenticate_continuation_latch
        if wait_for_settlement:
            from cayu.collaboration._wait_coordinator import CollaborationWaitLatchReceiver

            if type(self.receiver) is not CollaborationWaitLatchReceiver:
                raise PermissionError("Retained latch observation requires its native receiver.")
            authenticate = self.receiver._authenticate_latch_owned

        async def receive() -> ContinuationRecord:
            retained = await self._read_latch_record(latch)
            if retained.latch is not None:
                return retained
            authenticated = prepare_contract(
                ContinuationLatch,
                await authenticate(latch),
                redactor=self.redactor,
            )
            require_latch_identity(latch, authenticated)
            with authenticated_latch_scope(authenticated):
                return await self.store.latch_continuation(authenticated)

        return await self._observe(
            receive,
            key=(latch.ticket.session_id, continuation_operation_key(latch.ticket), "latch"),
            expected=contract_bytes(latch, redactor=self.redactor),
            wait_for_settlement=wait_for_settlement,
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
        if retirement.reason not in {
            "cancelled",
            "failed",
            "expired",
            "unavailable",
            "superseded",
        }:
            raise ContinuationConflict("Continuation exclusion requires a refusal reason.")
        return await self.retire(retirement, invocation=invocation)

    async def retire_released(
        self, candidate: ContinuationReleasedRetirement
    ) -> ContinuationRecord:
        """Retire only through native release and acknowledged service evidence."""
        from cayu.sessions._session_continuation_scope import released_retirement_scope
        from cayu.sessions._temporary_continuation import (
            TemporaryServiceRecord,
            reference_for_service,
        )

        expected = prepare_contract(
            ContinuationReleasedRetirement, candidate, redactor=self.redactor
        )
        ticket = expected.retirement.ticket
        if ticket.owner != self.owner:
            raise PermissionError("Released retirement belongs to another session owner.")
        retained = await self.store.load_continuation_ticket(
            ticket.session_id,
            session_instance_id=ticket.session_instance_id,
            registration_key=ticket.registration_key,
        )
        if retained is None:
            raise ContinuationUnavailable("Released retirement responsibility is unavailable.")
        settlements = []
        for reference in retained.services:
            raw = await self.store.load_session_operation(ticket.session_id, reference.key)
            if raw is None:
                raise ContinuationUnavailable("Service settlement evidence is unavailable.")
            service = prepare_contract(TemporaryServiceRecord, raw, redactor=self.redactor)
            require_ticket_identity(retained.ticket, service.intent.ticket)
            if (
                reference_for_service(service, retained.services) != reference
                or service.state not in {"returned", "excluded"}
                or not service.settlement_acknowledged
            ):
                raise ContinuationConflict("Temporary service settlement remains unresolved.")
            settlements.append((reference.key, reference.record_sha256))
        expected = expected.model_copy(update={"settled_services": tuple(settlements)})

        async def retire():
            with released_retirement_scope(expected):
                return await self.store.retire_continuation(expected.retirement)

        return await self._observe(
            retire,
            key=(ticket.session_id, continuation_operation_key(ticket), "released-retirement"),
            expected=contract_bytes(expected, redactor=self.redactor),
        )

    async def admit(
        self,
        candidate: ContinuationConsumption,
        command: AdmitInvocationCommand,
        *,
        invocation: InvocationContext,
    ) -> tuple[InvocationMutationResult, ContinuationRecord]:
        return await self._admit(
            candidate, command, invocation=invocation, wait_for_settlement=False
        )

    async def _admit_owned(
        self,
        candidate,
        command,
        *,
        invocation,
    ) -> tuple[InvocationMutationResult, ContinuationRecord]:
        return await self._admit(
            candidate, command, invocation=invocation, wait_for_settlement=True
        )

    async def _admit(
        self,
        candidate: ContinuationConsumption,
        command: AdmitInvocationCommand,
        *,
        invocation: InvocationContext,
        wait_for_settlement: bool,
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
            wait_for_settlement=wait_for_settlement,
        )

    async def prepare_temporary_admission(
        self,
        candidate: TemporaryServiceIntent,
        command: AdmitInvocationCommand,
        *,
        invocation: InvocationContext,
    ) -> tuple[InvocationMutationResult, TemporaryServiceRecord]:
        """Join runtime preparation to durable registration, then native admission.

        No public caller may supply this invocation authority. The configured
        clarification coordinator must configure current source disclosure
        authorization on the permit owner's admission guard. That guard belongs
        to the durable registration task, not its cancellable observer.
        """
        if self.temporary_permits is None:
            raise PermissionError("Temporary service permit owner is not registered.")
        intent = prepare_contract(TemporaryServiceIntent, candidate, redactor=self.redactor)
        copied = self._temporary_runtime_command(intent, command, invocation)
        await self._require_temporary_participant_binding(intent)
        copied = copied.model_copy(
            update={"temporary_service_operation_key": temporary_service_key(intent.operation)}
        )
        dispatch = TemporaryServiceDispatch(
            intent=intent,
            admission_payload_sha256=temporary_admission_payload_sha256(copied),
            expected_run_epoch=copied.expected_run_epoch,
        )
        preparation = await self.temporary_permits.prepare(dispatch)
        if intent.mode == "side_session":
            # Both native capacities precede the separate foreign permit handoff.
            # A rejected target must not strand a source-only reservation.
            await self.store._prepare_temporary_side_service(preparation)
        else:
            prepared = await self.store._load_temporary_continuation_service(preparation)
            if prepared is None:
                await self.store._publish_temporary_continuation_service(
                    previous=None,
                    proposed=TemporaryServiceRecord(admission=preparation, state="prepared"),
                )
            elif prepared.state != "prepared":
                raise ContinuationConflict("Temporary service preparation was already decided.")
        record = await self.temporary_permits.register(preparation)
        receipt_digest = continuation_digest(record.permit)
        copied = copied.model_copy(
            update={
                "participant_permit_operation": record.permit.expected.operation.caller_key,
                "participant_permit_commitment": receipt_digest,
            }
        )
        admission = TemporaryServiceAdmission(
            dispatch=dispatch,
            permit=record.permit.expected,
            permit_receipt_sha256=receipt_digest,
            admission_command_sha256=invocation_admission_command_sha256(copied),
        )
        # This is the runtime dependency, not the public observer. A committed
        # admission must reach the driver even if its acknowledgement is slow;
        # the surrounding service execution deadline still bounds the wait.
        return await self.admit_temporary(
            admission, copied, invocation=invocation, wait_for_settlement=True
        )

    def _temporary_runtime_command(
        self,
        intent: TemporaryServiceIntent,
        command: AdmitInvocationCommand,
        invocation: InvocationContext,
    ) -> AdmitInvocationCommand:
        if (
            type(invocation) is not InvocationContext
            or type(invocation.binding) is not PreparedInvocationBinding
        ):
            raise PermissionError("Temporary service requires runtime preparation authority.")
        invocation.require_runtime_authority()
        if type(command) is not AdmitInvocationCommand:
            raise PermissionError("Temporary service requires the typed admission command.")
        failure = None
        try:
            copied = copy_invocation_lifecycle_command(command)
        except Exception as error:
            failure = ContinuationConflict("Temporary service admission command is invalid.")
            failure.__cause__ = _failure_graph(error, self.redactor)
        if failure is not None:
            raise failure
        if type(copied) is not AdmitInvocationCommand:
            raise PermissionError("Temporary service requires the typed admission command.")
        binding = invocation.binding
        if (
            intent.ticket.owner != self.owner
            or intent.target.owner != self.owner
            or binding.session_id != intent.target.object_id
            or binding.session_instance_id != intent.target.incarnation
            or copied.session_id != binding.session_id
            or copied.expected_session_instance_id != binding.session_instance_id
            or binding.run_epoch != copied.expected_run_epoch + 1
            or invocation.active_profile != copied.target_active_profile
            or invocation.active_profile.interaction_id != intent.invocation_id
            or copied.interaction_started_event is None
            or copied.interaction_started_event.timestamp != intent.prepared_at
            or invocation.active_profile.profile.fingerprint != intent.execution_profile_sha256
            or invocation.tool_capability_ceiling != copied.tool_capability_ceiling
        ):
            raise PermissionError("Temporary service conflicts with runtime authority.")
        return copied

    async def _require_temporary_participant_binding(self, intent: TemporaryServiceIntent) -> None:
        binding = await self.store.load_participant_session_binding(intent.target.object_id)
        if (
            binding is None
            or binding.session_instance_id != intent.target.incarnation
            or binding.participant != intent.question.responder
            or continuation_digest(binding) != intent.participant_binding_sha256
        ):
            raise PermissionError("Temporary service participant binding conflicts.")

    async def admit_temporary(
        self,
        candidate: TemporaryServiceAdmission,
        command: AdmitInvocationCommand,
        *,
        invocation: InvocationContext,
        wait_for_settlement: bool = False,
    ) -> tuple[InvocationMutationResult, TemporaryServiceRecord]:
        """Consume a registered permit after ordinary runtime preparation gates.

        This is not a public caller receipt entrance. Both the runtime context
        and the foreign durable permit must authenticate the complete command.
        """
        if self.temporary_permits is None:
            raise PermissionError("Temporary service permit owner is not registered.")
        expected = prepare_contract(TemporaryServiceAdmission, candidate, redactor=self.redactor)
        intent = expected.dispatch.intent
        copied = self._temporary_runtime_command(intent, command, invocation)
        require_temporary_service_command(expected, copied)

        async def dispatch() -> tuple[InvocationMutationResult, TemporaryServiceRecord]:
            assert self.temporary_permits is not None
            authenticated = await self.temporary_permits.authenticate(expected)
            await self._require_temporary_participant_binding(intent)
            if intent.mode == "side_session":
                previous = await self.store._load_temporary_continuation_service(
                    authenticated.preparation
                )
                if previous is None:
                    raise ContinuationUnavailable("Side-session source responsibility is missing.")
                if previous.state == "prepared":
                    await self.store._publish_temporary_continuation_service(
                        previous=previous,
                        proposed=TemporaryServiceRecord(admission=authenticated, state="reserved"),
                    )
                elif previous.state != "reserved" or previous.admission != authenticated:
                    raise ContinuationConflict(
                        "Side-session source reservation was already decided."
                    )
            with temporary_admission_scope(authenticated, copied):
                result = await self.store.apply_invocation_lifecycle_command(copied)
            if intent.mode == "side_session":
                await self.store._reconcile_temporary_continuation_service(authenticated)
            retained = await self.store._load_temporary_continuation_service(authenticated)
            if retained is None:
                raise ContinuationUnavailable("Temporary service admission readback is pending.")
            return result, retained

        return await self._observe(
            dispatch,
            key=(intent.target.object_id, temporary_service_key(intent.operation), "admit"),
            expected=contract_bytes(expected, redactor=self.redactor),
            wait_for_settlement=wait_for_settlement,
        )

    async def exclude_temporary(
        self, candidate: TemporaryServicePreparation
    ) -> TemporaryServiceRecord:
        """Fence a prepared native operation before settling foreign responsibility.

        This is an explicit coordinator decision, never an inference from timeout
        or cancellation. The native compare-and-publication races admission under
        the same session transaction; admitted work cannot take this transition.
        """
        from cayu.collaboration._permits import ReceivingSettlementReceipt

        if self.temporary_permits is None:
            raise PermissionError("Temporary service permit owner is not registered.")
        expected = prepare_contract(TemporaryServicePreparation, candidate, redactor=self.redactor)
        intent = expected.dispatch.intent
        if intent.ticket.owner != self.owner or intent.target.owner != self.owner:
            raise PermissionError("Temporary service belongs to another receiving owner.")

        async def exclude():
            assert self.temporary_permits is not None
            retained = await self.store._load_temporary_continuation_service(expected)
            if retained is None:
                raise ContinuationUnavailable("Temporary service has no reserved receiving fence.")
            target_record = None
            if intent.mode == "side_session":
                from cayu.sessions._temporary_service_target import (
                    exclude_side_target,
                )

                if retained.state == "excluded":
                    target_session = await self.store.load(intent.target.object_id)
                    if (
                        target_session is None
                        or target_session.instance_id != intent.target.incarnation
                    ):
                        # Positive exclusion is retained by the source; a missing
                        # target is not being used to create exclusion evidence.
                        await self.temporary_permits.exclude(
                            expected,
                            reader=TemporaryServiceSettlementReader(
                                self.store, expected, redactor=self.redactor
                            ),
                        )
                        return await self._acknowledge_temporary_settlement(retained)
                target_record = await self.store._load_temporary_service_target(expected)
                if target_record is None:
                    raise ContinuationUnavailable(
                        "Joint preparation is missing its exact receiving fence."
                    )
                target_record = await self.store._publish_temporary_service_target(
                    previous=target_record,
                    proposed=exclude_side_target(target_record, expected),
                )
            if retained.state == "prepared" or (
                target_record is not None and retained.state == "reserved"
            ):
                excluded = TemporaryServiceRecord(
                    admission=retained.admission,
                    state="excluded",
                    settlement=target_record.service.settlement
                    if target_record is not None
                    else ReceivingSettlementReceipt(
                        expected=expected.permit,
                        receiving_owner=self.owner,
                        receipt_id="clarification-exclusion:" + continuation_digest(expected),
                        outcome="quiescent",
                        admission_excluded=True,
                    ),
                )
                retained = await self.store._publish_temporary_continuation_service(
                    previous=retained, proposed=excluded
                )
            elif retained.state != "excluded":
                raise ContinuationConflict("Temporary service was admitted and cannot be excluded.")
            if target_record is not None:
                from cayu.sessions._temporary_service_target import acknowledge_side_target

                await self.store._publish_temporary_service_target(
                    previous=target_record,
                    proposed=acknowledge_side_target(target_record, retained),
                )
            await self.temporary_permits.exclude(
                expected,
                reader=TemporaryServiceSettlementReader(
                    self.store, expected, redactor=self.redactor
                ),
            )
            return await self._acknowledge_temporary_settlement(retained)

        return await self._observe(
            exclude,
            key=(intent.target.object_id, temporary_service_key(intent.operation), "exclude"),
            expected=contract_bytes(expected, redactor=self.redactor),
        )

    async def _acknowledge_temporary_settlement(
        self, retained: TemporaryServiceRecord
    ) -> TemporaryServiceRecord:
        """Retain foreign settlement acknowledgement before permitting erasure."""
        if retained.settlement_acknowledged:
            return retained
        return await self.store._publish_temporary_continuation_service(
            previous=retained,
            proposed=retained.model_copy(update={"settlement_acknowledged": True}),
        )

    async def reconcile_temporary(
        self, candidate: TemporaryServiceAdmission
    ) -> TemporaryServiceRecord:
        """Reconcile native return before discharging the foreign obligation.

        If the foreign acknowledgement is lost, the source's exact returned
        record remains available to the registered settlement reader. Neither
        missing admission nor caller cancellation is an exclusion witness.
        """
        if self.temporary_permits is None:
            raise PermissionError("Temporary service permit owner is not registered.")
        expected = prepare_contract(TemporaryServiceAdmission, candidate, redactor=self.redactor)
        intent = expected.dispatch.intent
        if intent.ticket.owner != self.owner or intent.target.owner != self.owner:
            raise PermissionError("Temporary service belongs to another receiving owner.")

        async def reconcile() -> TemporaryServiceRecord:
            assert self.temporary_permits is not None
            retained = await self.store._load_temporary_continuation_service(expected)
            if retained is None or retained.settlement is None:
                authenticated = await self.temporary_permits.authenticate(expected)
                retained = await self.store._reconcile_temporary_continuation_service(authenticated)
            if retained.settlement is not None:
                await self.temporary_permits.settle(
                    expected,
                    reader=TemporaryServiceSettlementReader(
                        self.store, expected, redactor=self.redactor
                    ),
                )
                retained = await self._acknowledge_temporary_settlement(retained)
            return retained

        return await self._observe(
            reconcile,
            key=(intent.target.object_id, temporary_service_key(intent.operation), "reconcile"),
            expected=contract_bytes(expected, redactor=self.redactor),
        )

    async def service_temporary(
        self,
        app: CayuApp,
        request: ResumeRequest,
        candidate: TemporaryServiceIntent,
        *,
        participant_context: CollaborationAccessContext,
        delivery: Callable[[InvocationContext], Awaitable[None]] | None = None,
        wait_for_settlement: bool = False,
    ) -> TemporaryServiceRecord:
        """Explicit internal service, with durable reconciliation before dispatch.

        The registered clarification coordinator configures the permit owner's
        admission guard to retain current source disclosure through registration.
        The guard ends before provider serialization acquires its own disclosure
        authority; holding a non-reentrant policy guard around this entire call
        would deadlock. This is not an independently authorized SDK entrance.
        Cancelling observation does not prove the owned service stopped.
        """
        from cayu.collaboration.access import CollaborationAccessContext
        from cayu.runtime._temporary_continuation_resume import _TemporaryContinuationResumeHandoff
        from cayu.runtime._temporary_service_execution import drive_temporary_service

        if self.temporary_permits is None or app.session_store is not self.store:
            raise PermissionError("Temporary service requires its registered application owner.")
        intent = prepare_contract(TemporaryServiceIntent, candidate, redactor=self.redactor)
        participant_context = prepare_contract(
            CollaborationAccessContext, participant_context, redactor=self.redactor
        )
        if intent.ticket.owner != self.owner or intent.target.owner != self.owner:
            raise PermissionError("Temporary service belongs to another receiving owner.")
        failure = None
        try:
            copied = copy_resume_request(request)
            # The source digest distinguishes omitted and explicitly supplied
            # controls. Ordinary copying materializes defaults; preserve the
            # validated caller field-presence tuple for this exact handoff.
            fields = request.model_fields_set
            if type(fields) is not set or any(
                type(name) is not str or name not in ResumeRequest.model_fields for name in fields
            ):
                raise ValueError("Temporary resume field presence is invalid.")
            object.__setattr__(copied, "__pydantic_fields_set__", set(fields))
            digest = app._session_engine.work_attempt_source_request_sha256(
                copied, kind="continuation"
            )
        except Exception as error:
            failure = ContinuationConflict("Temporary service resume request is invalid.")
            failure.__cause__ = _failure_graph(error, self.redactor)
        if failure is not None:
            raise failure
        if copied.session_id != intent.target.object_id or digest != intent.resume_sha256:
            raise ContinuationConflict(
                "Temporary service resume request conflicts with its intent."
            )

        async def read_native() -> TemporaryServiceRecord | None:
            parent = await self.store.load_continuation_ticket(
                intent.ticket.session_id,
                session_instance_id=intent.ticket.session_instance_id,
                registration_key=intent.ticket.registration_key,
            )
            if parent is None:
                raise ContinuationUnavailable("Temporary service source ticket is unavailable.")
            require_ticket_identity(intent.ticket, parent.ticket)
            raw = await self.store.load_session_operation(
                intent.ticket.session_id, temporary_service_key(intent.operation)
            )
            if raw is None:
                if any(
                    item.key == temporary_service_key(intent.operation) for item in parent.services
                ):
                    raise ContinuationUnavailable(
                        "Temporary service receiving index lost its record."
                    )
                return None
            record = prepare_contract(TemporaryServiceRecord, raw, redactor=self.redactor)
            if record.intent != intent:
                raise ContinuationConflict("Temporary service receiving identity conflicts.")
            # Re-read through the exact receiving owner to authenticate its parent
            # index, incarnation and complete admission tuple, not just this JSON.
            return await self.store._load_temporary_continuation_service(record.admission)

        async def receive() -> TemporaryServiceRecord:
            assert self.temporary_permits is not None
            registered = await self.temporary_permits.lookup(intent)
            retained = await read_native()
            if retained is not None:
                if retained.state == "excluded":
                    preparation = (
                        retained.admission.preparation
                        if isinstance(retained.admission, TemporaryServiceAdmission)
                        else retained.admission
                    )
                    return await self.exclude_temporary(preparation)
                if registered is None or retained.admission.dispatch != registered.dispatch:
                    raise ContinuationUnavailable(
                        "Temporary service source registration conflicts."
                    )
                if retained.state == "prepared":
                    raise ContinuationUnavailable(
                        "Temporary service preparation requires exact admission or exclusion."
                    )
                return await self.reconcile_temporary(retained.acknowledged_admission)
            if registered is not None:
                raise ContinuationUnavailable(
                    "Temporary service admission is unresolved; reconcile or obtain exact exclusion."
                )
            await self._require_temporary_participant_binding(intent)
            boundary = await self.temporary_permits.execution_deadline(intent)
            stream = app._resume_private(
                copied,
                store_resolved_session_id=intent.target.object_id,
                continuation_handoff=_TemporaryContinuationResumeHandoff(self, intent, delivery),
                participant_context=participant_context,
            )
            await drive_temporary_service(stream, boundary)
            retained = await read_native()
            if retained is None:
                raise ContinuationUnavailable("Temporary service admission remains unacknowledged.")
            return await self.reconcile_temporary(retained.acknowledged_admission)

        return await self._observe(
            receive,
            key=(intent.target.object_id, temporary_service_key(intent.operation), "service"),
            expected=contract_bytes(intent, redactor=self.redactor)
            + contract_bytes(participant_context, redactor=self.redactor),
            wait_for_settlement=wait_for_settlement,
        )

    async def service(
        self,
        app: CayuApp,
        request: ResumeRequest,
        candidate: ContinuationService,
        *,
        participant_context: CollaborationAccessContext | None = None,
    ) -> ContinuationRecord:
        """Observe normal resume or exact reconciliation with a bounded waiter."""
        result = await self._service(
            app,
            request,
            candidate,
            participant_context=participant_context,
            wait_for_settlement=False,
        )
        return result.record

    async def _service_owned(
        self,
        app,
        request,
        candidate,
        *,
        participant_context,
    ) -> _ContinuationServiceResult:
        """Runtime-owned observation; all service authentication remains shared."""
        return await self._service(
            app,
            request,
            candidate,
            participant_context=participant_context,
            wait_for_settlement=True,
        )

    def _prepare_service_input(
        self,
        app: CayuApp,
        request: ResumeRequest,
        candidate: ContinuationService,
        *,
        participant_context: CollaborationAccessContext | None,
    ):
        """Snapshot identical service input without granting execution or read access."""
        service = prepare_contract(ContinuationService, candidate, redactor=self.redactor)
        if app.session_store is not self.store or service.ticket.owner != self.owner:
            raise PermissionError("Continuation service belongs to another application owner.")
        if participant_context is not None:
            from cayu.collaboration.access import CollaborationAccessContext

            participant_context = prepare_contract(
                CollaborationAccessContext, participant_context, redactor=self.redactor
            )
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
        return copied_request, service, participant_context, service_digest

    async def _prepare_service_observation(self, app, request, candidate, *, participant_context):
        prepared = self._prepare_service_input(
            app, request, candidate, participant_context=participant_context
        )
        _, service, context, _ = prepared
        source = await self.store.load(service.ticket.session_id)
        if source is None or source.instance_id != service.ticket.session_instance_id:
            raise ContinuationConflict("Continuation session incarnation is unavailable.")
        # Public execution and replay continue to require current execution
        # authority. The separate inspection entrance cannot dispatch work.
        await app._require_participant_execution(source, context)
        return prepared

    async def _read_service_record(self, service, service_digest):
        retained = await self.store.load_continuation_ticket(
            service.ticket.session_id,
            registration_key=service.ticket.registration_key,
            session_instance_id=service.ticket.session_instance_id,
        )
        if retained is None or retained.latch is None:
            raise ContinuationConflict("Continuation has no retained ready wait.")
        require_ticket_identity(service.ticket, retained.ticket)
        require_latch_identity(service.latch, retained.latch)
        consumption = retained.consumption
        if consumption is not None and (
            consumption.service_digest != service_digest
            or consumption.continuation_id != service.continuation_id
            or consumption.mode != service.mode
            or consumption.accepted_at != service.accepted_at
        ):
            raise ContinuationConflict("Continuation service was accepted differently.")
        return retained

    async def _inspect_service(self, app, request, candidate, *, expected, participant_context):
        """Read exact service acceptance without admitting or replaying execution."""
        from cayu.runtime._host_continuation_discovery import recover_session_continuation

        _, service, context, service_digest = self._prepare_service_input(
            app, request, candidate, participant_context=participant_context
        )
        # Readback authenticates the exact participant/creation/incarnation and
        # original preparation, without requiring that participant to be active.
        observed = await recover_session_continuation(app, expected, context=context)
        require_ticket_identity(service.ticket, observed.ticket)
        retained = await self._read_service_record(service, service_digest)
        if retained != observed:
            raise ContinuationUnavailable("Continuation changed during service inspection.")
        return retained

    async def _service(
        self,
        app: CayuApp,
        request: ResumeRequest,
        candidate: ContinuationService,
        *,
        participant_context: CollaborationAccessContext | None,
        wait_for_settlement: bool,
    ) -> _ContinuationServiceResult:
        """Run normal resume, or reconcile its exact prior admission.

        The application owns session gates and execution; this service is not a
        scheduler and retains no secondary event buffer.
        """
        from cayu.runtime._session_continuation_resume import _ContinuationResumeHandoff

        try:
            (
                copied_request,
                service,
                participant_context,
                service_digest,
            ) = await self._prepare_service_observation(
                app, request, candidate, participant_context=participant_context
            )
        except Exception as error:
            if wait_for_settlement:
                # Only the internal host entrance consumes this proof. The
                # public entrance preserves its original exception contract.
                # Control signals and failures after entering receive remain
                # subject to the normal retained-ownership boundary.
                evidence = _failure_graph(error, self.redactor)
                assert isinstance(evidence, Exception)
                raise _ContinuationServiceNotStarted(evidence) from None
            raise

        async def receive() -> _ContinuationServiceResult | _ContinuationServiceReadFailure:
            try:
                retained = await self._read_service_record(service, service_digest)
            except Exception as error:
                # Carry positive completion evidence through the owned observer.
                # Ordinary ExceptionGroups are completed failures too. Groups
                # containing control signals are not Exceptions and still pass
                # through _observe()'s existing control-signal classification.
                # Only this initial read is covered: later failures may follow
                # admission/dispatch and must retain their normal recovery fence.
                return _ContinuationServiceReadFailure(error)
            # The read above authenticates the complete latch identity, excluding
            # mutable ticket state/revision. Admission must carry the native
            # representation that consume_continuation retains, while the original
            # service digest continues to bind the caller's exact request.
            native_service = service.model_copy(update={"latch": retained.latch})
            if retained.consumption is not None:
                consumption = retained.consumption
                if consumption.receipt_stage in {"admitted", "excluded"}:
                    return _ContinuationServiceResult(retained, False)
                if consumption.receipt_stage == "prepared":
                    from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
                    from cayu.sessions._invocation_lifecycle import (
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
                        return _ContinuationServiceResult(
                            await self.reconcile_admission(
                                consumption.model_copy(update={"receipt_stage": "prepared"})
                            ),
                            False,
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
                handoff = _ContinuationResumeHandoff(self, native_service, service_digest)
                stream = app._resume_private(
                    copied_request,
                    store_resolved_session_id=service.ticket.session_id,
                    continuation_handoff=handoff,
                    participant_context=participant_context,
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
                return _ContinuationServiceResult(result, True)
            if (
                retained.ticket.state != "WAITING"
                or retained.ticket.revision != service.ticket.revision
            ):
                raise ContinuationConflict("Continuation is no longer eligible for service.")
            handoff = _ContinuationResumeHandoff(self, native_service, service_digest)
            stream = app._resume_private(
                copied_request,
                store_resolved_session_id=service.ticket.session_id,
                continuation_handoff=handoff,
                participant_context=participant_context,
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
            return _ContinuationServiceResult(result, True)

        observed = await self._observe(
            receive,
            key=(service.ticket.session_id, continuation_operation_key(service.ticket), "service"),
            expected=service_digest.encode("ascii")
            + (
                b""
                if participant_context is None
                else contract_bytes(participant_context, redactor=self.redactor)
            ),
            wait_for_settlement=wait_for_settlement,
        )
        if isinstance(observed, _ContinuationServiceReadFailure):
            # Public callers retain the same sanitized exception contract as
            # _observe(). Conversion is observer-local, including joined calls.
            error = observed.failure
            kind = (
                ContinuationConflict
                if isinstance(error, ContinuationConflict)
                else PermissionError
                if isinstance(error, PermissionError)
                else ContinuationUnavailable
            )
            failure = kind(
                "Receiving dependency failed; reconcile the latch."
                if isinstance(error, ExceptionGroup)
                else "Continuation receiving operation failed."
            )
            failure.__cause__ = _failure_graph(error, self.redactor)
            if wait_for_settlement:
                raise _ContinuationServiceNotStarted(failure) from None
            raise failure
        return observed

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
                from cayu.sessions._invocation_lifecycle import (
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
