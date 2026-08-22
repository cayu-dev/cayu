"""Durable CAS state for governed egress-authority installation."""

from __future__ import annotations

import asyncio
import contextlib
import weakref
from abc import ABC, abstractmethod
from hashlib import sha256
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._exception_groups import exception_tree_contains
from cayu._task_wait import (
    await_shielded_task_outcome,
    restore_task_cancellation_requests,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_value,
    require_durable_clean_nonblank,
    require_durable_nonblank,
)
from cayu.core.events import Event, EventType, event_with_runtime_generated_id
from cayu.egress.adapter import (
    DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
    EgressAuthorityCutoverRequest,
    EgressAuthorityCutoverResult,
    SandboxEgressAdapter,
)
from cayu.egress.authority import (
    EgressAuthorityChangeKind,
    EgressAuthorityCutoverReceipt,
    EgressAuthorityCutoverStrategy,
    EgressAuthorityIdentity,
    EgressAuthorityTransitionState,
    _egress_authority_cutover_receipt_is_adapter_verified,
)
from cayu.egress.errors import EgressAuthorityCutoverNeedsAttention
from cayu.environments.factory import (
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryResult,
)
from cayu.runtime._event_projection import prepare_new_runtime_event
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime.approvals import (
    ResolutionActor,
    copy_resolution_actor,
    resolution_actor_payload,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileAuthorityDecision,
    ExecutionProfileDecision,
    ExecutionProfileDecisionKind,
    _has_runtime_execution_profile_decision_authority,
)
from cayu.runtime.sessions import Session, SessionStatus, SessionStore
from cayu.vaults.redaction import SecretRedactor

EGRESS_AUTHORITY_TRANSITION_CHECKPOINT_KEY = "cayu:egress_authority_transition"
EGRESS_AUTHORITY_TRANSITION_SCHEMA_VERSION = 3
_EGRESS_AUTHORITY_PARKED_OUTCOME = "cayu_egress_authority_parked"
_TEXT_MAX_CHARS = 4096


class _EgressAuthorityTransitionReplay(RuntimeError):
    """Abort an atomic publication whose exact revision is already durable."""


class EgressAuthorityTransitionConflict(RuntimeError):
    """A stale worker attempted to advance a different transition revision."""


class EgressAuthorityAdoptionHandler(ABC):
    """Application-owned installer invoked by real profile admission.

    The runtime passes the exact sealed decision produced by its profile policy.
    Implementations own their retained environment, private worker token, and
    transition identity. They must use the supplied coordinator to durably finish
    the corresponding transition; the runtime independently reloads and verifies
    ``ACTIVE`` before admission.
    """

    @abstractmethod
    async def adopt(
        self,
        decision: ExecutionProfileDecision,
        *,
        coordinator: EgressAuthorityTransitionCoordinator,
        expected_environment_fingerprint: str,
        factory_result: EnvironmentFactoryResult,
    ) -> EgressAuthorityAdoptionResult:
        """Return proof for the exact retained pre-cutover environment."""


class _ParkedEgressAllocation:
    """One runtime-deposited live allocation awaiting exact profile adoption."""

    __slots__ = (
        "draining",
        "environment_name",
        "factory_result",
        "fingerprint",
        "ready",
        "release_task",
        "session_id",
    )

    def __init__(
        self,
        *,
        session_id: str,
        environment_name: str,
        fingerprint: str,
        factory_result: EnvironmentFactoryResult,
    ) -> None:
        self.session_id = session_id
        self.environment_name = environment_name
        self.fingerprint = fingerprint
        self.factory_result = factory_result
        self.ready = False
        self.draining = False
        self.release_task: asyncio.Task[None] | None = None


_PARKED_EGRESS_ALLOCATIONS: dict[
    int,
    tuple[
        EgressAuthorityAdoptionHandler,
        dict[tuple[str, str], _ParkedEgressAllocation],
    ],
] = {}


def _parked_egress_allocations_for_handler(
    handler: EgressAuthorityAdoptionHandler,
    *,
    create: bool,
) -> dict[tuple[str, str], _ParkedEgressAllocation] | None:
    """Address a public handler by identity without requiring hash/weakref support."""

    handler_id = id(handler)
    entry = _PARKED_EGRESS_ALLOCATIONS.get(handler_id)
    if entry is not None:
        if entry[0] is not handler:
            raise AssertionError("Parked egress handler identity was reused while owned.")
        return entry[1]
    if not create:
        return None
    allocations: dict[tuple[str, str], _ParkedEgressAllocation] = {}
    # The registry is the live allocation owner between invocations, so it
    # deliberately retains the application handler until the last allocation
    # is claimed. Dropping a weak key would orphan the external allocation.
    _PARKED_EGRESS_ALLOCATIONS[handler_id] = (handler, allocations)
    return allocations


def _reserve_egress_authority_allocation_parking(
    handler: EgressAuthorityAdoptionHandler,
    *,
    session_id: str,
    environment_name: str,
    fingerprint: str,
    factory_result: EnvironmentFactoryResult,
    max_parked_allocations: int,
) -> _ParkedEgressAllocation:
    """Reserve one bounded owner before parking can mutate a live allocation."""

    if not isinstance(handler, EgressAuthorityAdoptionHandler):
        raise TypeError("Parked egress allocation requires an adoption handler.")
    if type(factory_result) is not EnvironmentFactoryResult:
        raise TypeError("Parked egress allocation requires an EnvironmentFactoryResult.")
    if type(max_parked_allocations) is not int or max_parked_allocations <= 0:
        raise ValueError("max_parked_allocations must be a positive integer.")
    key = (
        require_durable_clean_nonblank(session_id, "session_id"),
        require_durable_clean_nonblank(environment_name, "environment_name"),
    )
    if (
        type(fingerprint) is not str
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("Parked egress allocation fingerprint must be a SHA-256 digest.")
    allocations = _parked_egress_allocations_for_handler(handler, create=True)
    if allocations is None:
        raise AssertionError("Egress parking did not create its handler registry.")
    existing = allocations.get(key)
    if existing is not None:
        if existing.factory_result is factory_result and existing.fingerprint == fingerprint:
            return existing
        raise EgressAuthorityTransitionConflict(
            "The adoption handler already owns another allocation for this session environment."
        )
    if len(allocations) >= max_parked_allocations:
        raise EgressAuthorityTransitionConflict(
            "The adoption handler parked-allocation capacity is exhausted."
        )
    reservation = _ParkedEgressAllocation(
        session_id=key[0],
        environment_name=key[1],
        fingerprint=fingerprint,
        factory_result=factory_result,
    )
    allocations[key] = reservation
    return reservation


def _complete_egress_authority_allocation_parking(
    handler: EgressAuthorityAdoptionHandler,
    *,
    reservation: _ParkedEgressAllocation,
) -> None:
    """Expose one exact allocation only after its runner is positively parked."""

    allocations = _parked_egress_allocations_for_handler(handler, create=False)
    key = (reservation.session_id, reservation.environment_name)
    if allocations is None or allocations.get(key) is not reservation:
        raise EgressAuthorityTransitionConflict(
            "Egress allocation parking lost its exact runtime reservation."
        )
    reservation.ready = True


def _discard_egress_authority_allocation_parking_reservation(
    handler: EgressAuthorityAdoptionHandler,
    *,
    reservation: _ParkedEgressAllocation,
) -> None:
    """Release an unused reservation only while no parked allocation depends on it."""

    if reservation.ready:
        return
    allocations = _parked_egress_allocations_for_handler(handler, create=False)
    key = (reservation.session_id, reservation.environment_name)
    if allocations is None or allocations.get(key) is not reservation:
        return
    del allocations[key]
    if not allocations:
        del _PARKED_EGRESS_ALLOCATIONS[id(handler)]


def _park_egress_authority_allocation(
    handler: EgressAuthorityAdoptionHandler,
    *,
    session_id: str,
    environment_name: str,
    fingerprint: str,
    factory_result: EnvironmentFactoryResult,
    max_parked_allocations: int,
) -> None:
    """Test/internal shortcut for depositing an already-quiescent allocation."""

    reservation = _reserve_egress_authority_allocation_parking(
        handler,
        session_id=session_id,
        environment_name=environment_name,
        fingerprint=fingerprint,
        factory_result=factory_result,
        max_parked_allocations=max_parked_allocations,
    )
    _complete_egress_authority_allocation_parking(
        handler,
        reservation=reservation,
    )


def _require_parked_egress_authority_allocation(
    handler: EgressAuthorityAdoptionHandler,
    *,
    session_id: str,
    environment_name: str,
    expected_fingerprint: str,
) -> EnvironmentFactoryResult:
    """Return the exact live owner without consuming its post-admission claim."""

    allocations = _parked_egress_allocations_for_handler(handler, create=False)
    parked = None if allocations is None else allocations.get((session_id, environment_name))
    if parked is None:
        raise EgressAuthorityTransitionConflict(
            "Egress profile adoption has no runtime-parked environment allocation."
        )
    if not parked.ready:
        raise EgressAuthorityTransitionConflict(
            "Egress environment allocation parking has not reached its verified boundary."
        )
    if parked.draining:
        raise EgressAuthorityTransitionConflict(
            "Egress environment allocation cleanup already owns this parked resource."
        )
    if parked.fingerprint != expected_fingerprint:
        raise EgressAuthorityTransitionConflict(
            "The runtime-parked environment allocation conflicts with durable identity."
        )
    return parked.factory_result


def _find_parked_egress_authority_allocation(
    handler: EgressAuthorityAdoptionHandler,
    *,
    session_id: str,
    environment_name: str,
    expected_fingerprint: str,
) -> EnvironmentFactoryResult | None:
    """Return an exact parked owner when this process still retains one."""

    allocations = _parked_egress_allocations_for_handler(handler, create=False)
    if allocations is None or (session_id, environment_name) not in allocations:
        return None
    return _require_parked_egress_authority_allocation(
        handler,
        session_id=session_id,
        environment_name=environment_name,
        expected_fingerprint=expected_fingerprint,
    )


def _claim_parked_egress_authority_allocation(
    handler: EgressAuthorityAdoptionHandler,
    *,
    session_id: str,
    environment_name: str,
    expected_fingerprint: str,
    expected_factory_result: EnvironmentFactoryResult,
) -> EnvironmentFactoryResult:
    """Consume one exact parked allocation after durable invocation admission."""

    result = _require_parked_egress_authority_allocation(
        handler,
        session_id=session_id,
        environment_name=environment_name,
        expected_fingerprint=expected_fingerprint,
    )
    if result is not expected_factory_result:
        raise EgressAuthorityTransitionConflict(
            "Egress adoption returned a different allocation than the runtime parked."
        )
    allocations = _parked_egress_allocations_for_handler(handler, create=False)
    if allocations is None:
        raise EgressAuthorityTransitionConflict("Egress adoption lost its parked-allocation owner.")
    del allocations[(session_id, environment_name)]
    if not allocations:
        del _PARKED_EGRESS_ALLOCATIONS[id(handler)]
    return result


async def _drain_parked_egress_authority_allocations(
    handler: EgressAuthorityAdoptionHandler,
    *,
    timeout_s: float,
    session_id: str | None = None,
) -> bool:
    """Discard selected parked allocations while retaining uncertain cleanup ownership."""

    if not isinstance(handler, EgressAuthorityAdoptionHandler):
        raise TypeError("Parked egress cleanup requires an adoption handler.")
    if type(timeout_s) not in {int, float} or not isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be a finite positive number.")
    selected_session_id = (
        None if session_id is None else require_durable_clean_nonblank(session_id, "session_id")
    )
    allocations = _parked_egress_allocations_for_handler(handler, create=False)
    if not allocations:
        return True
    selected = tuple(
        allocation
        for allocation in allocations.values()
        if selected_session_id is None or allocation.session_id == selected_session_id
    )
    if not selected:
        return True
    if any(
        allocation.factory_result.release is None and allocation.release_task is None
        for allocation in selected
    ):
        raise RuntimeError(
            "A parked egress allocation has no provider cleanup owner and cannot be discarded."
        )
    for allocation in selected:
        allocation.draining = True
        release = allocation.factory_result.release
        if release is not None and allocation.release_task is None:

            async def discard(
                release=release,
            ) -> None:
                await release(EnvironmentFactoryReleaseAction.DISCARD)

            allocation.release_task = asyncio.create_task(
                discard(),
                name=(
                    "cayu-egress-parked-allocation-discard-"
                    f"{allocation.session_id}-{allocation.environment_name}"
                ),
            )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + float(timeout_s)
    for allocation in selected:
        task = allocation.release_task
        if task is not None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            outcome = await await_shielded_task_outcome(task, timeout_s=remaining)
            if outcome.timed_out:
                if outcome.cancellation is not None:
                    restore_task_cancellation_requests(
                        outcome.cancellation_requests_consumed,
                        cancellation=outcome.cancellation,
                    )
                    raise outcome.cancellation from TimeoutError(
                        "Parked egress allocation cleanup remains in flight."
                    )
                return False
            if outcome.error is not None:
                allocation.release_task = None
                if outcome.cancellation is not None:
                    restore_task_cancellation_requests(
                        outcome.cancellation_requests_consumed,
                        cancellation=outcome.cancellation,
                    )
                    raise outcome.cancellation from outcome.error
                raise outcome.error
        current = _parked_egress_allocations_for_handler(handler, create=False)
        key = (allocation.session_id, allocation.environment_name)
        if current is not None and current.get(key) is allocation:
            del current[key]
            if not current:
                del _PARKED_EGRESS_ALLOCATIONS[id(handler)]
        if task is not None and outcome.cancellation is not None:
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed,
                cancellation=outcome.cancellation,
            )
            raise outcome.cancellation
    return True


_EGRESS_AUTHORITY_ADOPTION_RESULT_TOKEN = object()


class EgressAuthorityAdoptionResult:
    """Runtime-owned handoff from backend cutover to invocation lifecycle.

    The result remains owned by the application handler until the invocation
    claims it after durable admission.  Claiming is single-use, so the exact
    factory result cannot be adopted by two invocations or cleaned twice.
    """

    __slots__ = (
        "_claimed",
        "_coordinator_authority",
        "_factory_result",
        "_token",
        "transition",
    )

    def __init__(
        self,
        *,
        transition: EgressAuthorityTransitionRecord,
        factory_result: EnvironmentFactoryResult,
        coordinator_authority: object,
        _token: object,
    ) -> None:
        if _token is not _EGRESS_AUTHORITY_ADOPTION_RESULT_TOKEN:
            raise TypeError("Egress authority adoption results are runtime-owned.")
        if type(transition) is not EgressAuthorityTransitionRecord:
            raise TypeError("transition must be EgressAuthorityTransitionRecord.")
        if type(factory_result) is not EnvironmentFactoryResult:
            raise TypeError("factory_result must be EnvironmentFactoryResult.")
        self.transition = transition
        self._factory_result = factory_result
        self._coordinator_authority = coordinator_authority
        self._claimed = False
        self._token = _token

    @property
    def environment(self):
        """Return the retained environment for runtime verification only."""

        return self._factory_result.environment

    def _claim_factory_result(self) -> EnvironmentFactoryResult:
        if self._token is not _EGRESS_AUTHORITY_ADOPTION_RESULT_TOKEN:
            raise EgressAuthorityTransitionConflict(
                "Egress authority environment handoff is not runtime-owned."
            )
        if self._claimed:
            raise EgressAuthorityTransitionConflict(
                "Egress authority environment handoff was already claimed."
            )
        self._claimed = True
        return self._factory_result


def _runtime_egress_authority_adoption_result(
    *,
    transition: EgressAuthorityTransitionRecord,
    factory_result: EnvironmentFactoryResult,
    coordinator: EgressAuthorityTransitionCoordinator,
) -> EgressAuthorityAdoptionResult:
    """Seal one adapter-verified environment ownership handoff."""

    if not isinstance(coordinator, EgressAuthorityTransitionCoordinator):
        raise TypeError("coordinator must be EgressAuthorityTransitionCoordinator.")
    if not _has_runtime_verified_active_transition(
        transition,
        coordinator_authority=coordinator._handoff_authority,
    ):
        raise EgressAuthorityTransitionConflict(
            "Egress authority adoption requires runtime-owned backend proof."
        )
    return EgressAuthorityAdoptionResult(
        transition=transition,
        factory_result=factory_result,
        coordinator_authority=coordinator._handoff_authority,
        _token=_EGRESS_AUTHORITY_ADOPTION_RESULT_TOKEN,
    )


def _has_runtime_egress_authority_adoption_result(
    result: object,
    *,
    coordinator: EgressAuthorityTransitionCoordinator | None = None,
) -> bool:
    valid = (
        type(result) is EgressAuthorityAdoptionResult
        and result._token is _EGRESS_AUTHORITY_ADOPTION_RESULT_TOKEN
        and not result._claimed
        and _has_runtime_verified_active_transition(
            result.transition,
            coordinator_authority=result._coordinator_authority,
        )
    )
    if not valid or coordinator is None:
        return valid
    return result._coordinator_authority is coordinator._handoff_authority


_CUTOVER_CANCELLATION_REQUESTS_ATTRIBUTE = "_cayu_egress_cutover_cancellation_requests"


def _carry_cutover_cancellation_requests(
    cancellation: asyncio.CancelledError,
    count: int,
) -> asyncio.CancelledError:
    """Carry shield-consumed request ownership to the outer live-resource owner."""

    setattr(cancellation, _CUTOVER_CANCELLATION_REQUESTS_ATTRIBUTE, count)
    return cancellation


def _cutover_cancellation_requests(error: BaseException) -> int:
    """Read the request count attached by the cutover coordinator."""

    value = getattr(error, _CUTOVER_CANCELLATION_REQUESTS_ATTRIBUTE, 0)
    return value if type(value) is int and value >= 0 else 0


def _egress_transition_fatal_signal(error: BaseException) -> BaseException | None:
    """Return the first process-control signal cancellation must not replace."""

    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            fatal = _egress_transition_fatal_signal(child)
            if fatal is not None:
                return fatal
        return None
    if not isinstance(error, (Exception, asyncio.CancelledError)):
        return error
    return None


def _egress_transition_fatal_failure(error: BaseException) -> bool:
    """Identify process-control failures that cancellation must not replace."""

    return _egress_transition_fatal_signal(error) is not None


class EgressAuthorityTransitionRecord(BaseModel):
    """One reconstructable state in an egress-authority adoption lifecycle."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.egress-authority-transition"] = "cayu.egress-authority-transition"
    schema_version: Literal[3] = EGRESS_AUTHORITY_TRANSITION_SCHEMA_VERSION
    transition_id: str = Field(max_length=256)
    session_id: str = Field(max_length=256)
    environment_name: str = Field(max_length=256)
    revision: StrictInt = Field(ge=1)
    state: EgressAuthorityTransitionState
    owner_fingerprint: str
    expected_authority: EgressAuthorityIdentity
    target_authority: EgressAuthorityIdentity
    classification: EgressAuthorityChangeKind
    policy_identity: str = Field(max_length=256)
    actor: ResolutionActor
    authorization_reason: str = Field(max_length=_TEXT_MAX_CHARS)
    reason: str = Field(max_length=_TEXT_MAX_CHARS)
    strategy: EgressAuthorityCutoverStrategy
    source_environment_fingerprint: str
    environment_fingerprint: str | None = None
    receipt: EgressAuthorityCutoverReceipt | None = None
    fingerprint: str

    @field_validator("transition_id", "session_id", "environment_name", "policy_identity")
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("authorization_reason", "reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return require_durable_nonblank(value, "reason")

    @field_validator("actor", mode="before")
    @classmethod
    def copy_actor(cls, value: object) -> ResolutionActor:
        if isinstance(value, ResolutionActor):
            copied = copy_resolution_actor(value)
            if copied is None:
                raise ValueError("Egress authority transition actor is required.")
            return copied
        return ResolutionActor.model_validate(value)

    @field_validator("expected_authority", "target_authority", mode="before")
    @classmethod
    def copy_authority(cls, value: object) -> EgressAuthorityIdentity:
        if isinstance(value, EgressAuthorityIdentity):
            value = value.model_dump(mode="json")
        return EgressAuthorityIdentity.model_validate(value)

    @field_validator("receipt", mode="before")
    @classmethod
    def copy_receipt(cls, value: object) -> EgressAuthorityCutoverReceipt | None:
        if value is None:
            return None
        if isinstance(value, EgressAuthorityCutoverReceipt):
            value = value.model_dump(mode="json")
        return EgressAuthorityCutoverReceipt.model_validate(value)

    @field_validator(
        "owner_fingerprint",
        "source_environment_fingerprint",
        "environment_fingerprint",
        "fingerprint",
    )
    @classmethod
    def validate_digest(cls, value: str | None, info) -> str | None:
        if value is None and info.field_name == "environment_fingerprint":
            return None
        if (
            value is None
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @model_validator(mode="after")
    def validate_transition(self) -> EgressAuthorityTransitionRecord:
        if self.expected_authority.runner_kind != self.target_authority.runner_kind:
            raise ValueError("Egress authority transition cannot change runner kind in place.")
        if self.target_authority.generation <= self.expected_authority.generation:
            raise ValueError("Egress authority transition generation must increase.")
        if self.strategy != self.target_authority.cutover_strategy:
            raise ValueError("Transition strategy does not match the target authority.")
        terminal_with_receipt = self.state is EgressAuthorityTransitionState.ACTIVE
        if terminal_with_receipt != (self.receipt is not None):
            raise ValueError("Only an active transition carries an activation receipt.")
        if self.receipt is not None and (
            self.receipt.from_fingerprint != self.expected_authority.fingerprint
            or self.receipt.to_fingerprint != self.target_authority.fingerprint
            or self.receipt.from_generation != self.expected_authority.generation
            or self.receipt.to_generation != self.target_authority.generation
            or self.receipt.runner_kind != self.target_authority.runner_kind
            or self.receipt.strategy != self.strategy
            or self.receipt.environment_fingerprint != self.environment_fingerprint
        ):
            raise ValueError("Cutover receipt does not prove this transition.")
        if (
            self.environment_fingerprint is not None
            and self.environment_fingerprint != self.source_environment_fingerprint
        ):
            raise ValueError("Egress authority transition changed the retained backend allocation.")
        if (
            self.state
            in {
                EgressAuthorityTransitionState.AUTHORIZED,
                EgressAuthorityTransitionState.REFUSED,
            }
            and self.environment_fingerprint is not None
        ):
            raise ValueError("Unverified transition states cannot claim environment identity.")
        if self.fingerprint != _transition_fingerprint(self):
            raise ValueError("Egress authority transition fingerprint is inconsistent.")
        return self


_RUNTIME_VERIFIED_ACTIVE_TRANSITIONS: dict[
    int,
    tuple[weakref.ReferenceType[EgressAuthorityTransitionRecord], str, object],
] = {}


def _with_runtime_verified_active_transition(
    transition: EgressAuthorityTransitionRecord,
    *,
    coordinator: EgressAuthorityTransitionCoordinator,
) -> EgressAuthorityTransitionRecord:
    """Attest one exact active record after adapter-owned backend verification."""

    if type(transition) is not EgressAuthorityTransitionRecord:
        raise TypeError("transition must be an EgressAuthorityTransitionRecord.")
    if not isinstance(coordinator, EgressAuthorityTransitionCoordinator):
        raise TypeError("coordinator must be EgressAuthorityTransitionCoordinator.")
    if transition.state is not EgressAuthorityTransitionState.ACTIVE or transition.receipt is None:
        raise ValueError("Only a durably active transition can carry runtime verification.")
    identity = id(transition)

    def forget(reference: weakref.ReferenceType[EgressAuthorityTransitionRecord]) -> None:
        current = _RUNTIME_VERIFIED_ACTIVE_TRANSITIONS.get(identity)
        if current is not None and current[0] is reference:
            _RUNTIME_VERIFIED_ACTIVE_TRANSITIONS.pop(identity, None)

    reference = weakref.ref(transition, forget)
    _RUNTIME_VERIFIED_ACTIVE_TRANSITIONS[identity] = (
        reference,
        _transition_fingerprint(transition),
        coordinator._handoff_authority,
    )
    return transition


def _has_runtime_verified_active_transition(
    transition: EgressAuthorityTransitionRecord,
    *,
    coordinator_authority: object | None = None,
) -> bool:
    """Return positive in-process proof from an adapter verification boundary."""

    attestation = _RUNTIME_VERIFIED_ACTIVE_TRANSITIONS.get(id(transition))
    if attestation is None or attestation[0]() is not transition:
        return False
    try:
        observed_fingerprint = _transition_fingerprint(transition)
    except Exception:
        return False
    return observed_fingerprint == attestation[1] and (
        coordinator_authority is None or coordinator_authority is attestation[2]
    )


def authorized_egress_authority_transition(
    *,
    decision: ExecutionProfileDecision,
    transition_id: str,
    environment_name: str,
    owner_fingerprint: str,
    source_environment_fingerprint: str,
) -> EgressAuthorityTransitionRecord:
    """Convert one trusted adopted profile decision into installation authority."""

    if type(decision) is not ExecutionProfileDecision:
        raise TypeError("Egress authority authorization requires ExecutionProfileDecision.")
    if not _has_runtime_execution_profile_decision_authority(decision):
        raise ValueError(
            "Egress authority authorization requires a runtime-owned profile decision."
        )
    if decision.kind is not ExecutionProfileDecisionKind.ADOPTED:
        raise ValueError("Only an adopted profile decision can authorize egress installation.")
    expected = decision.expected_profile.egress_authority
    target = decision.candidate_profile.egress_authority
    classification = decision.egress_authority_change
    if expected is None or target is None or classification is None:
        raise ValueError("Profile decision does not contain a typed egress authority change.")
    if target.cutover_strategy is not EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH:
        raise ValueError("This coordinator requires fresh-path same-allocation cutover.")
    if (
        classification
        in {
            EgressAuthorityChangeKind.WIDER,
            EgressAuthorityChangeKind.INCOMPARABLE,
        }
        and decision.authority_decision is not ExecutionProfileAuthorityDecision.AUTHORIZED
    ):
        raise ValueError("Wider or ambiguous egress authority requires trusted authorization.")
    if decision.actor is None or decision.actor.source is None:
        raise ValueError("Egress authority installation requires an attributable trusted actor.")
    return _build_transition_record(
        transition_id=transition_id,
        session_id=decision.event.session_id,
        environment_name=environment_name,
        revision=1,
        state=EgressAuthorityTransitionState.AUTHORIZED,
        owner_fingerprint=owner_fingerprint,
        expected_authority=expected,
        target_authority=target,
        classification=classification,
        policy_identity=decision.policy_identity,
        actor=ResolutionActor.model_validate(resolution_actor_payload(decision.actor)),
        authorization_reason=decision.reason,
        reason=decision.reason,
        strategy=target.cutover_strategy,
        source_environment_fingerprint=source_environment_fingerprint,
    )


def advance_egress_authority_transition(
    current: EgressAuthorityTransitionRecord,
    *,
    state: EgressAuthorityTransitionState,
    reason: str,
    receipt: EgressAuthorityCutoverReceipt | None = None,
    environment_fingerprint: str | None = None,
) -> EgressAuthorityTransitionRecord:
    """Build the next legal revision without mutating the current record."""

    allowed = {
        EgressAuthorityTransitionState.AUTHORIZED: {
            EgressAuthorityTransitionState.INSTALLING,
            EgressAuthorityTransitionState.REFUSED,
        },
        EgressAuthorityTransitionState.INSTALLING: {
            EgressAuthorityTransitionState.ACTIVE,
            EgressAuthorityTransitionState.REFUSED,
            EgressAuthorityTransitionState.AMBIGUOUS,
        },
        EgressAuthorityTransitionState.AMBIGUOUS: {
            EgressAuthorityTransitionState.ACTIVE,
            EgressAuthorityTransitionState.REFUSED,
        },
    }
    if state not in allowed.get(current.state, set()):
        raise ValueError(f"Illegal egress authority transition: {current.state} -> {state}.")
    return _build_transition_record(
        transition_id=current.transition_id,
        session_id=current.session_id,
        environment_name=current.environment_name,
        revision=current.revision + 1,
        state=state,
        owner_fingerprint=current.owner_fingerprint,
        expected_authority=current.expected_authority,
        target_authority=current.target_authority,
        classification=current.classification,
        policy_identity=current.policy_identity,
        actor=current.actor,
        authorization_reason=current.authorization_reason,
        reason=reason,
        strategy=current.strategy,
        source_environment_fingerprint=current.source_environment_fingerprint,
        environment_fingerprint=environment_fingerprint,
        receipt=receipt,
    )


class SessionCheckpointEgressAuthorityTransitionStore:
    """CAS transition storage using every SessionStore's atomic checkpoint seam."""

    def __init__(
        self,
        session_store: SessionStore,
        *,
        event_writer: RuntimeEventWriter | None = None,
    ) -> None:
        if not all(
            callable(getattr(session_store, method, None))
            for method in ("load_checkpoint", "publish_checkpoint_and_events")
        ):
            raise TypeError("Egress authority transition store requires a SessionStore.")
        if event_writer is not None and not isinstance(event_writer, RuntimeEventWriter):
            raise TypeError("event_writer must be a RuntimeEventWriter.")
        self._session_store = session_store
        self._event_writer = event_writer

    async def load(self, session_id: str) -> EgressAuthorityTransitionRecord | None:
        checkpoint = await self._session_store.load_checkpoint(session_id)
        return _transition_from_checkpoint(checkpoint)

    async def compare_and_set(
        self,
        *,
        expected: EgressAuthorityTransitionRecord | None,
        replacement: EgressAuthorityTransitionRecord,
    ) -> EgressAuthorityTransitionRecord:
        if type(replacement) is not EgressAuthorityTransitionRecord:
            raise TypeError("replacement must be EgressAuthorityTransitionRecord.")

        def transform(session: Session, checkpoint: dict | None) -> dict:
            if session.id != replacement.session_id:
                raise ValueError("Egress authority transition belongs to another session.")
            if session.status in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
                raise EgressAuthorityTransitionConflict(
                    "Egress authority can change only while the session is parked."
                )
            current = _transition_from_checkpoint(checkpoint)
            if current == replacement:
                raise _EgressAuthorityTransitionReplay
            if current != expected:
                raise EgressAuthorityTransitionConflict(
                    "Egress authority transition changed before compare-and-set."
                )
            if expected is None:
                if replacement.revision != 1:
                    raise ValueError("Initial egress authority transition revision must be one.")
            elif replacement.revision != expected.revision + 1:
                raise ValueError("Egress authority transition revisions must be monotonic.")
            updated = (
                {}
                if checkpoint is None
                else copy_durable_json_value(
                    checkpoint,
                    "checkpoint",
                )
            )
            updated[EGRESS_AUTHORITY_TRANSITION_CHECKPOINT_KEY] = replacement.model_dump(
                mode="json"
            )
            return updated

        raw_events = list(egress_authority_transition_events(replacement))
        prepared_events = (
            [prepare_new_runtime_event(event, redactor=SecretRedactor()) for event in raw_events]
            if self._event_writer is None
            else self._event_writer.prepare_many(raw_events)
        )
        with contextlib.suppress(_EgressAuthorityTransitionReplay):
            await self._session_store.publish_checkpoint_and_events(
                replacement.session_id,
                checkpoint_transform=transform,
                events=prepared_events,
            )
        persisted = await self.load(replacement.session_id)
        if persisted is None or persisted != replacement:
            raise RuntimeError("Egress authority transition acknowledgement was not durable.")
        if self._event_writer is not None:
            await self._event_writer.fan_out_persisted(prepared_events)
        return persisted


class EgressAuthorityTransitionCoordinator:
    """Install one authorized target and durably classify every outcome."""

    def __init__(
        self,
        store: SessionCheckpointEgressAuthorityTransitionStore,
        *,
        expected_source_environment_fingerprint: str | None = None,
    ) -> None:
        if not isinstance(store, SessionCheckpointEgressAuthorityTransitionStore):
            raise TypeError("Egress authority coordinator requires a transition store.")
        if expected_source_environment_fingerprint is not None and (
            type(expected_source_environment_fingerprint) is not str
            or len(expected_source_environment_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_source_environment_fingerprint
            )
        ):
            raise ValueError(
                "expected_source_environment_fingerprint must be a lowercase SHA-256 digest."
            )
        self._store = store
        self._expected_source_environment_fingerprint = expected_source_environment_fingerprint
        # Handoffs are accepted only when they were completed through the exact
        # coordinator supplied by the invocation-admission boundary. This keeps
        # application handlers from silently bypassing the production event
        # preparation and side-effect fan-out path with a sibling store.
        self._handoff_authority = object()

    async def _compare_and_set_owned(
        self,
        *,
        expected: EgressAuthorityTransitionRecord | None,
        replacement: EgressAuthorityTransitionRecord,
        cancellation: asyncio.CancelledError | None = None,
    ) -> tuple[EgressAuthorityTransitionRecord, asyncio.CancelledError | None, int]:
        """Own one durable transition write and reconcile commit-then-raise."""

        publication_task = asyncio.create_task(
            self._store.compare_and_set(
                expected=expected,
                replacement=replacement,
            ),
            name=(
                f"cayu-egress-authority-publish-{replacement.transition_id}-{replacement.revision}"
            ),
        )
        publication = await await_shielded_task_outcome(
            publication_task,
            cancellation=cancellation,
        )
        carried_cancellation = publication.cancellation or cancellation
        consumed_requests = publication.cancellation_requests_consumed
        if publication.error is None:
            persisted = publication.result
            if type(persisted) is not EgressAuthorityTransitionRecord:
                raise RuntimeError("Egress transition publication returned no durable record.")
            return persisted, carried_cancellation, consumed_requests

        readback_task = asyncio.create_task(
            self._store.load(replacement.session_id),
            name=(
                f"cayu-egress-authority-readback-{replacement.transition_id}-{replacement.revision}"
            ),
        )
        readback = await await_shielded_task_outcome(
            readback_task,
            cancellation=carried_cancellation,
        )
        carried_cancellation = readback.cancellation or carried_cancellation
        consumed_requests += readback.cancellation_requests_consumed
        if readback.error is None and readback.result == replacement:
            if _egress_transition_fatal_failure(publication.error):
                if carried_cancellation is not None:
                    raise publication.error from carried_cancellation
                raise publication.error
            return replacement, carried_cancellation, consumed_requests

        publication_error = publication.error
        if isinstance(publication_error, asyncio.CancelledError) and carried_cancellation is None:
            publication_error = unexpected_child_cancellation_error(
                publication_error,
                operation="Egress transition publication",
            )
        failures: list[BaseException] = [publication_error]
        if readback.error is not None:
            readback_error = readback.error
            if isinstance(readback_error, asyncio.CancelledError) and carried_cancellation is None:
                readback_error = unexpected_child_cancellation_error(
                    readback_error,
                    operation="Egress transition readback",
                )
            failures.append(readback_error)
        else:
            failures.append(
                EgressAuthorityTransitionConflict(
                    "Egress transition publication failed without exact durable readback."
                )
            )
        failure: BaseException = (
            failures[0]
            if len(failures) == 1
            else BaseExceptionGroup(
                "Egress transition publication and readback both failed.",
                failures,
            )
        )
        if any(_egress_transition_fatal_failure(item) for item in failures):
            raise failure from carried_cancellation
        if carried_cancellation is not None:
            raise _carry_cutover_cancellation_requests(
                carried_cancellation,
                consumed_requests,
            ) from failure
        raise failure

    async def authorize(
        self,
        record: EgressAuthorityTransitionRecord,
    ) -> EgressAuthorityTransitionRecord:
        if record.state is not EgressAuthorityTransitionState.AUTHORIZED:
            raise ValueError("Initial egress authority record must be authorized.")
        if (
            self._expected_source_environment_fingerprint is not None
            and record.source_environment_fingerprint
            != self._expected_source_environment_fingerprint
        ):
            raise EgressAuthorityTransitionConflict(
                "The egress transition does not target the retained backend allocation."
            )
        current = await self._store.load(record.session_id)
        if current is not None:
            if _same_egress_authority_transition(current, record):
                return current
            if current.state is EgressAuthorityTransitionState.ACTIVE:
                authoritative = current.target_authority
            elif current.state is EgressAuthorityTransitionState.REFUSED:
                authoritative = current.expected_authority
            else:
                raise EgressAuthorityTransitionConflict(
                    "Another egress authority transition is still active for this session."
                )
            authoritative_source_environment_fingerprint = (
                current.environment_fingerprint or current.source_environment_fingerprint
            )
            if (
                current.environment_name != record.environment_name
                or record.expected_authority != authoritative
                or record.source_environment_fingerprint
                != authoritative_source_environment_fingerprint
            ):
                raise EgressAuthorityTransitionConflict(
                    "The next egress transition does not continue the authoritative generation."
                )
            record = _build_transition_record(
                transition_id=record.transition_id,
                session_id=record.session_id,
                environment_name=record.environment_name,
                revision=current.revision + 1,
                state=record.state,
                owner_fingerprint=record.owner_fingerprint,
                expected_authority=record.expected_authority,
                target_authority=record.target_authority,
                classification=record.classification,
                policy_identity=record.policy_identity,
                actor=record.actor,
                authorization_reason=record.authorization_reason,
                reason=record.reason,
                strategy=record.strategy,
                source_environment_fingerprint=(record.source_environment_fingerprint),
            )
            return await self._store.compare_and_set(expected=current, replacement=record)
        return await self._store.compare_and_set(expected=None, replacement=record)

    async def install(
        self,
        *,
        authorized: EgressAuthorityTransitionRecord,
        adapter: SandboxEgressAdapter,
        request: EgressAuthorityCutoverRequest,
        owner_token: str,
    ) -> tuple[EgressAuthorityTransitionRecord, EgressAuthorityCutoverResult | None]:
        if authorized.state is not EgressAuthorityTransitionState.AUTHORIZED:
            raise EgressAuthorityTransitionConflict(
                "Only the exact durable authorized revision can begin installation."
            )
        if (
            authorized.owner_fingerprint != request.owner_fingerprint
            or authorized.owner_fingerprint != egress_authority_owner_fingerprint(owner_token)
        ):
            raise EgressAuthorityTransitionConflict(
                "Cutover request does not hold the authorized transition owner."
            )
        if (
            authorized.session_id != request.session_id
            or authorized.environment_name != request.environment_name
            or authorized.expected_authority != request.expected_authority
            or authorized.target_authority != request.target_authority
            or authorized.source_environment_fingerprint != request.environment_fingerprint
        ):
            raise EgressAuthorityTransitionConflict(
                "Cutover request does not match the durable authorized transition."
            )
        installing = advance_egress_authority_transition(
            authorized,
            state=EgressAuthorityTransitionState.INSTALLING,
            reason="The authorized target is being installed while agent work is fenced.",
            environment_fingerprint=request.environment_fingerprint,
        )
        (
            installing,
            publication_cancellation,
            publication_cancellation_requests,
        ) = await self._compare_and_set_owned(
            expected=authorized,
            replacement=installing,
        )
        if publication_cancellation is not None:
            refused = advance_egress_authority_transition(
                installing,
                state=EgressAuthorityTransitionState.REFUSED,
                reason="Installation was cancelled before adapter dispatch began.",
            )
            _, publication_cancellation, additional_requests = await self._compare_and_set_owned(
                expected=installing,
                replacement=refused,
                cancellation=publication_cancellation,
            )
            if publication_cancellation is None:
                raise RuntimeError("Egress transition publication lost caller cancellation.")
            raise _carry_cutover_cancellation_requests(
                publication_cancellation,
                publication_cancellation_requests + additional_requests,
            )
        try:
            installation_task = asyncio.create_task(
                adapter.cutover_authority(request),
                name=f"cayu-egress-authority-cutover-{authorized.transition_id}",
            )
            installation_outcome = await await_shielded_task_outcome(
                installation_task,
                timeout_s=DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
                timeout_after_cancellation_s=DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
            )
            if installation_outcome.timed_out:
                timeout_error = EgressAuthorityCutoverNeedsAttention(
                    "Egress cutover did not settle within the bounded ownership window.",
                    replacement_binding=None,
                    environment_fingerprint=request.environment_fingerprint,
                    target_authority_installed=True,
                    settlement_task=installation_task,
                    cancellation=installation_outcome.cancellation,
                    cancellation_requests_consumed=(
                        installation_outcome.cancellation_requests_consumed
                    ),
                )
                raise timeout_error
            if installation_outcome.error is not None:
                if isinstance(
                    installation_outcome.error,
                    EgressAuthorityCutoverNeedsAttention,
                ):
                    needs_attention = installation_outcome.error
                    if installation_outcome.cancellation is not None:
                        needs_attention.cancellation = installation_outcome.cancellation
                        needs_attention.cancellation_requests_consumed = (
                            installation_outcome.cancellation_requests_consumed
                        )
                    raise needs_attention
                if _egress_transition_fatal_failure(installation_outcome.error):
                    if installation_outcome.cancellation is not None:
                        raise installation_outcome.error from installation_outcome.cancellation
                    raise installation_outcome.error
                if exception_tree_contains(
                    installation_outcome.error,
                    (asyncio.CancelledError,),
                ):
                    child_cancellation = (
                        unexpected_child_cancellation_error(
                            installation_outcome.error,
                            operation="Egress authority adapter cutover",
                        )
                        if isinstance(installation_outcome.error, asyncio.CancelledError)
                        else installation_outcome.error
                    )
                    raise EgressAuthorityCutoverNeedsAttention(
                        "The egress adapter ended with child cancellation after dispatch; "
                        "the environment must remain fenced until backend readback.",
                        replacement_binding=None,
                        environment_fingerprint=request.environment_fingerprint,
                        target_authority_installed=True,
                        settlement_task=installation_task,
                        cancellation=installation_outcome.cancellation,
                        cancellation_requests_consumed=(
                            installation_outcome.cancellation_requests_consumed
                        ),
                    ) from child_cancellation
                if installation_outcome.cancellation is not None:
                    raise _carry_cutover_cancellation_requests(
                        installation_outcome.cancellation,
                        installation_outcome.cancellation_requests_consumed,
                    ) from installation_outcome.error
                raise installation_outcome.error
            result = installation_outcome.result
            if result is None:
                raise RuntimeError("Egress adapter returned no cutover result.")
            if installation_outcome.cancellation is not None:
                result = EgressAuthorityCutoverResult(
                    binding=result.binding,
                    receipt=result.receipt,
                    cancellation=installation_outcome.cancellation,
                    cancellation_requests_consumed=(
                        installation_outcome.cancellation_requests_consumed
                    ),
                )
            if (
                not _egress_authority_cutover_receipt_is_adapter_verified(result.receipt)
                or result.receipt.environment_fingerprint != request.environment_fingerprint
            ):
                raise EgressAuthorityCutoverNeedsAttention(
                    "Egress adapter returned caller-constructible activation evidence.",
                    replacement_binding=result.binding,
                    environment_fingerprint=result.receipt.environment_fingerprint,
                    target_authority_installed=True,
                )
        except EgressAuthorityCutoverNeedsAttention as exc:
            ambiguous = advance_egress_authority_transition(
                installing,
                state=EgressAuthorityTransitionState.AMBIGUOUS,
                reason="A backend mutation was dispatched but its exact authority outcome "
                "is unproven.",
                environment_fingerprint=request.environment_fingerprint,
            )
            try:
                (
                    _,
                    publication_cancellation,
                    additional_requests,
                ) = await self._compare_and_set_owned(
                    expected=installing,
                    replacement=ambiguous,
                    cancellation=exc.cancellation,
                )
            except BaseException as persistence_error:
                persistence_cancellation = (
                    persistence_error
                    if isinstance(persistence_error, asyncio.CancelledError)
                    else exc.cancellation
                )
                persistence_cancellation_requests = (
                    _cutover_cancellation_requests(persistence_error)
                    if isinstance(persistence_error, asyncio.CancelledError)
                    else exc.cancellation_requests_consumed
                )
                publication_failure = (
                    persistence_error.__cause__
                    if isinstance(persistence_error, asyncio.CancelledError)
                    and persistence_error.__cause__ is not None
                    else persistence_error
                )
                raise EgressAuthorityCutoverNeedsAttention(
                    "The old egress path was retired and durable ambiguity evidence "
                    "could not be acknowledged; the environment must remain fenced.",
                    replacement_binding=exc.replacement_binding,
                    environment_fingerprint=exc.environment_fingerprint,
                    target_authority_installed=exc.target_authority_installed,
                    settlement_task=exc.settlement_task,
                    cancellation=persistence_cancellation,
                    cancellation_requests_consumed=(persistence_cancellation_requests),
                ) from publication_failure
            if publication_cancellation is not None:
                exc.cancellation = publication_cancellation
                exc.cancellation_requests_consumed += additional_requests
            raise
        except BaseException as installation_error:
            refused = advance_egress_authority_transition(
                installing,
                state=EgressAuthorityTransitionState.REFUSED,
                reason="Installation failed before a safe target activation was acknowledged.",
            )
            try:
                source_cancellation = (
                    installation_error
                    if isinstance(installation_error, asyncio.CancelledError)
                    else None
                )
                (
                    _,
                    publication_cancellation,
                    additional_requests,
                ) = await self._compare_and_set_owned(
                    expected=installing,
                    replacement=refused,
                    cancellation=source_cancellation,
                )
            except BaseException as persistence_error:
                if isinstance(persistence_error, asyncio.CancelledError):
                    if persistence_error is installation_error:
                        raise
                    publication_failure = persistence_error.__cause__
                    failures = [installation_error]
                    if publication_failure is not None:
                        failures.append(publication_failure)
                    raise persistence_error from BaseExceptionGroup(
                        "Egress cutover failure preceded cancellation during durable refusal.",
                        failures,
                    )
                raise EgressAuthorityTransitionConflict(
                    "Pre-cutover failure could not be durably acknowledged; work remains fenced."
                ) from BaseExceptionGroup(
                    "Egress cutover and durable refusal acknowledgement both failed.",
                    [installation_error, persistence_error],
                )
            if publication_cancellation is not None:
                carried = _carry_cutover_cancellation_requests(
                    publication_cancellation,
                    _cutover_cancellation_requests(installation_error) + additional_requests,
                )
                if carried is installation_error:
                    raise
                raise carried from installation_error
            raise
        active = advance_egress_authority_transition(
            installing,
            state=EgressAuthorityTransitionState.ACTIVE,
            reason="The exact target authority and environment identity were backend-verified.",
            receipt=result.receipt,
            environment_fingerprint=result.receipt.environment_fingerprint,
        )
        activation_task = asyncio.create_task(
            self._store.compare_and_set(
                expected=installing,
                replacement=active,
            ),
            name=f"cayu-egress-authority-activate-{authorized.transition_id}",
        )
        activation_outcome = await await_shielded_task_outcome(
            activation_task,
            cancellation=result.cancellation,
            timeout_s=DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
            timeout_after_cancellation_s=DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
        )
        carried_cancellation = activation_outcome.cancellation or result.cancellation
        carried_cancellation_requests = (
            result.cancellation_requests_consumed
            + activation_outcome.cancellation_requests_consumed
        )
        if activation_outcome.timed_out:
            raise EgressAuthorityCutoverNeedsAttention(
                "The target egress path passed backend verification but its durable active "
                "acknowledgement remains in flight; the environment must remain fenced.",
                replacement_binding=result.binding,
                environment_fingerprint=result.receipt.environment_fingerprint,
                settlement_task=activation_task,
                cancellation=carried_cancellation,
                cancellation_requests_consumed=carried_cancellation_requests,
            )
        persistence_error = activation_outcome.error
        if persistence_error is not None:
            ambiguous = advance_egress_authority_transition(
                installing,
                state=EgressAuthorityTransitionState.AMBIGUOUS,
                reason=("Backend activation succeeded but its durable acknowledgement was lost."),
                environment_fingerprint=result.receipt.environment_fingerprint,
            )
            with contextlib.suppress(BaseException):
                await self._store.compare_and_set(
                    expected=installing,
                    replacement=ambiguous,
                )
            raise EgressAuthorityCutoverNeedsAttention(
                "The target egress path passed backend verification but its durable active "
                "acknowledgement is unproven; the environment must remain fenced.",
                replacement_binding=result.binding,
                environment_fingerprint=result.receipt.environment_fingerprint,
                cancellation=carried_cancellation,
                cancellation_requests_consumed=carried_cancellation_requests,
            ) from persistence_error
        persisted_active = activation_outcome.result
        if type(persisted_active) is not EgressAuthorityTransitionRecord:
            raise RuntimeError("Egress authority activation returned no durable transition.")
        return _with_runtime_verified_active_transition(
            persisted_active,
            coordinator=self,
        ), EgressAuthorityCutoverResult(
            binding=result.binding,
            receipt=result.receipt,
            cancellation=carried_cancellation,
            cancellation_requests_consumed=carried_cancellation_requests,
        )

    async def reconcile(
        self,
        *,
        current: EgressAuthorityTransitionRecord,
        adapter: SandboxEgressAdapter,
        request: EgressAuthorityCutoverRequest,
        owner_token: str,
    ) -> EgressAuthorityTransitionRecord:
        """Activate only exact backend proof; otherwise persist needs-attention."""

        persisted = await self._store.load(current.session_id)
        if persisted != current:
            raise EgressAuthorityTransitionConflict(
                "Egress authority reconciliation does not own the current durable revision."
            )
        if current.state not in {
            EgressAuthorityTransitionState.INSTALLING,
            EgressAuthorityTransitionState.AMBIGUOUS,
            EgressAuthorityTransitionState.ACTIVE,
        }:
            return current
        if (
            egress_authority_owner_fingerprint(owner_token) != current.owner_fingerprint
            or request.owner_fingerprint != current.owner_fingerprint
            or request.session_id != current.session_id
            or request.environment_name != current.environment_name
            or request.expected_authority != current.expected_authority
            or request.target_authority != current.target_authority
            or request.environment_fingerprint != current.environment_fingerprint
            or request.environment_fingerprint != current.source_environment_fingerprint
        ):
            raise EgressAuthorityTransitionConflict(
                "Egress authority reconciliation owner or operation identity conflicts."
            )
        receipt = await adapter.reconcile_authority_cutover(request)
        if receipt is None:
            if current.state is EgressAuthorityTransitionState.ACTIVE:
                raise EgressAuthorityTransitionConflict(
                    "The durable active egress transition is not proven by backend readback."
                )
            if current.state is EgressAuthorityTransitionState.AMBIGUOUS:
                return current
            ambiguous = advance_egress_authority_transition(
                current,
                state=EgressAuthorityTransitionState.AMBIGUOUS,
                reason="Recovery could not prove which egress generation became active.",
                environment_fingerprint=current.environment_fingerprint,
            )
            return await self._store.compare_and_set(expected=current, replacement=ambiguous)
        if not _egress_authority_cutover_receipt_is_adapter_verified(receipt):
            raise EgressAuthorityTransitionConflict(
                "Egress authority recovery did not return adapter-owned backend proof."
            )
        if (
            current.environment_fingerprint is not None
            and receipt.environment_fingerprint != current.environment_fingerprint
        ):
            raise EgressAuthorityTransitionConflict(
                "Recovery receipt belongs to a different backend environment."
            )
        if current.state is EgressAuthorityTransitionState.ACTIVE:
            if receipt != current.receipt:
                raise EgressAuthorityTransitionConflict(
                    "Backend readback conflicts with the durable activation receipt."
                )
            return _with_runtime_verified_active_transition(
                current,
                coordinator=self,
            )
        active = advance_egress_authority_transition(
            current,
            state=EgressAuthorityTransitionState.ACTIVE,
            reason="Recovery reconciled the exact backend/environment activation receipt.",
            receipt=receipt,
            environment_fingerprint=receipt.environment_fingerprint,
        )
        persisted_active = await self._store.compare_and_set(
            expected=current,
            replacement=active,
        )
        return _with_runtime_verified_active_transition(
            persisted_active,
            coordinator=self,
        )


def egress_authority_owner_fingerprint(owner_token: str) -> str:
    """Commit a private worker ownership token without persisting it."""

    token = require_durable_clean_nonblank(owner_token, "owner_token")
    return sha256(token.encode("utf-8")).hexdigest()


def egress_authority_transition_events(
    record: EgressAuthorityTransitionRecord,
) -> tuple[Event, ...]:
    """Project bounded, deterministic, credential-free evidence for one revision."""

    if type(record) is not EgressAuthorityTransitionRecord:
        raise TypeError("record must be EgressAuthorityTransitionRecord.")
    event_types = {
        EgressAuthorityTransitionState.AUTHORIZED: (
            EventType.EGRESS_AUTHORITY_REQUESTED,
            EventType.EGRESS_AUTHORITY_AUTHORIZED,
        ),
        EgressAuthorityTransitionState.INSTALLING: (EventType.EGRESS_AUTHORITY_INSTALLING,),
        EgressAuthorityTransitionState.ACTIVE: (EventType.EGRESS_AUTHORITY_ACTIVATED,),
        EgressAuthorityTransitionState.REFUSED: (EventType.EGRESS_AUTHORITY_REFUSED,),
        EgressAuthorityTransitionState.AMBIGUOUS: (EventType.EGRESS_AUTHORITY_AMBIGUOUS,),
    }[record.state]
    payload = {
        "schema_version": EGRESS_AUTHORITY_TRANSITION_SCHEMA_VERSION,
        "transition_id": record.transition_id,
        "transition_fingerprint": record.fingerprint,
        "revision": record.revision,
        "state": record.state.value,
        "classification": record.classification.value,
        "from_authority": _authority_evidence(record.expected_authority),
        "to_authority": _authority_evidence(record.target_authority),
        "policy_identity": record.policy_identity,
        "actor": resolution_actor_payload(record.actor),
        "authorization_reason": record.authorization_reason,
        "reason": record.reason,
        "adapter_strategy": record.strategy.value,
        "source_environment_fingerprint": record.source_environment_fingerprint,
        "environment_fingerprint": record.environment_fingerprint,
        "receipt": (None if record.receipt is None else record.receipt.model_dump(mode="json")),
    }
    return tuple(
        event_with_runtime_generated_id(
            Event(
                type=event_type,
                session_id=record.session_id,
                environment_name=record.environment_name,
                id=(
                    f"egress-authority:{record.fingerprint}:"
                    f"{event_type.value.rsplit('.', maxsplit=1)[-1]}"
                ),
                payload=payload,
            )
        )
        for event_type in event_types
    )


def _authority_evidence(identity: EgressAuthorityIdentity) -> dict:
    """Return the documented safe bounded projection, never raw policy/provider data."""

    return {
        "schema_version": identity.schema_version,
        "generation": identity.generation,
        "fingerprint": identity.fingerprint,
        "authority_source": identity.authority_source,
        "authority_scope": identity.authority_scope,
        "policy_version": identity.policy_version,
        "runner_kind": identity.runner_kind,
        "cutover_strategy": identity.cutover_strategy.value,
        "comparison_available": identity.comparison_available,
        "policies": [policy.model_dump(mode="json") for policy in identity.policies],
        "bindings": [binding.model_dump(mode="json") for binding in identity.bindings],
    }


def _transition_from_checkpoint(
    checkpoint: dict | None,
) -> EgressAuthorityTransitionRecord | None:
    if checkpoint is None:
        return None
    raw = checkpoint.get(EGRESS_AUTHORITY_TRANSITION_CHECKPOINT_KEY)
    if raw is None:
        return None
    return EgressAuthorityTransitionRecord.model_validate(
        copy_durable_json_value(raw, "egress_authority_transition")
    )


def require_exact_egress_authority_transition(
    checkpoint: dict | None,
    expected: EgressAuthorityTransitionRecord,
) -> None:
    """Bind invocation admission to the exact backend-verified transition revision."""

    if type(expected) is not EgressAuthorityTransitionRecord:
        raise TypeError("expected must be an EgressAuthorityTransitionRecord.")
    if _transition_from_checkpoint(checkpoint) != expected:
        raise EgressAuthorityTransitionConflict(
            "Egress authority transition changed before invocation admission."
        )


def require_egress_authority_transition_compatible_with_profile(
    checkpoint: dict | None,
    profile_authority: EgressAuthorityIdentity | None,
) -> None:
    """Fail closed when durable cutover state cannot govern the proposed invocation."""

    current = _transition_from_checkpoint(checkpoint)
    if current is None:
        return
    if current.state in {
        EgressAuthorityTransitionState.INSTALLING,
        EgressAuthorityTransitionState.AMBIGUOUS,
    }:
        raise EgressAuthorityTransitionConflict(
            "Egress authority transition requires reconciliation before invocation admission."
        )
    authoritative = (
        current.target_authority
        if current.state is EgressAuthorityTransitionState.ACTIVE
        else current.expected_authority
    )
    if profile_authority != authoritative:
        raise EgressAuthorityTransitionConflict(
            "The invocation profile does not match durable egress authority."
        )


def _build_transition_record(**values: Any) -> EgressAuthorityTransitionRecord:
    provisional = EgressAuthorityTransitionRecord.model_construct(
        record_type="cayu.egress-authority-transition",
        schema_version=EGRESS_AUTHORITY_TRANSITION_SCHEMA_VERSION,
        fingerprint="0" * 64,
        **values,
    )
    return EgressAuthorityTransitionRecord.model_validate(
        {
            **provisional.model_dump(mode="json"),
            "fingerprint": _transition_fingerprint(provisional),
        }
    )


def _same_egress_authority_transition(
    current: EgressAuthorityTransitionRecord,
    candidate: EgressAuthorityTransitionRecord,
) -> bool:
    """Compare the immutable authority tuple for exact request replay."""

    return (
        current.transition_id == candidate.transition_id
        and current.session_id == candidate.session_id
        and current.environment_name == candidate.environment_name
        and current.owner_fingerprint == candidate.owner_fingerprint
        and current.expected_authority == candidate.expected_authority
        and current.target_authority == candidate.target_authority
        and current.classification == candidate.classification
        and current.policy_identity == candidate.policy_identity
        and current.actor == candidate.actor
        and current.authorization_reason == candidate.authorization_reason
        and current.strategy == candidate.strategy
        and current.source_environment_fingerprint == candidate.source_environment_fingerprint
    )


def _transition_fingerprint(record: EgressAuthorityTransitionRecord) -> str:
    material = record.model_dump(mode="json", exclude={"fingerprint"})
    return sha256(canonical_durable_json_bytes(material, "egress_authority_transition")).hexdigest()
