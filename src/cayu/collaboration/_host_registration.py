"""Inert finite configuration for native collaboration-host scheduling."""

from __future__ import annotations

from dataclasses import dataclass

from cayu.collaboration._contracts import OperationRef
from cayu.collaboration._host_clarification_maintenance import (
    _ClarificationMaintenanceSource,
    snapshot_maintenance_source,
)
from cayu.collaboration._host_clarifications import (
    _ClarificationRule,
    snapshot_clarification_rule,
)
from cayu.collaboration._host_continuations import (
    _ContinuationRule,
    snapshot_continuation_rule,
)
from cayu.collaboration._host_ownership import (
    HostOwnershipLimits,
    _positive_integer,
    _seconds,
)
from cayu.collaboration._host_planned_producer import (
    _PlannedProducerRule,
    snapshot_planned_producer_rule,
)
from cayu.collaboration._host_producer_execution import (
    HostProducerExecution,
)
from cayu.collaboration._host_producer_maintenance import (
    _DISCLOSURE_ACTIONS,
    HostProducerMaintenance,
)
from cayu.collaboration._host_requests import (
    _RequestMaintenanceSource,
    snapshot_request_source,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._producer_contracts import ProducerOutputRegistration
from cayu.collaboration._producer_recovery import ProducerOutputRecovery
from cayu.collaboration._wait_discovery import WaitRecovery
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.exports import SessionExportAccessContext
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import ParticipantRef
from cayu.collaboration.planning import RequestPlanningRequest


@dataclass(frozen=True, slots=True)
class _ProducerSource:
    participant: ParticipantRef
    context: CollaborationAccessContext


@dataclass(frozen=True, slots=True)
class _ProducerMaintenanceRule:
    intent: HostProducerMaintenance
    context: CollaborationAccessContext
    disclosure_context: SessionExportAccessContext | None = None


@dataclass(frozen=True, slots=True)
class _ProducerExecutionRule:
    intent: HostProducerExecution
    context: CollaborationAccessContext
    producer_context: MandateAccessContext


@dataclass(frozen=True, slots=True)
class _ProducerDisclosure:
    destination: OperationRef
    context: SessionExportAccessContext


@dataclass(frozen=True, slots=True)
class _ProducerOutputRule:
    recovery: ProducerOutputRecovery
    context: CollaborationAccessContext
    failure_context: SessionExportAccessContext
    destinations: tuple[_ProducerDisclosure, ...]


@dataclass(frozen=True, slots=True)
class _PlanningRule:
    expected: RequestPlanningRequest
    context: MandateAccessContext


@dataclass(frozen=True, slots=True)
class _ProducerRegistrationRule:
    expected: ProducerOutputRegistration
    context: CollaborationAccessContext
    producer_context: MandateAccessContext


@dataclass(frozen=True, slots=True)
class _WaitRule:
    expected: WaitRecovery
    context: MandateAccessContext


@dataclass(frozen=True, slots=True)
class _HostRegistration:
    """Finite local configuration, never a durable work queue or authority grant."""

    limits: HostOwnershipLimits
    producer_sources: tuple[_ProducerSource, ...]
    producer_rules: tuple[_ProducerMaintenanceRule, ...]
    batch_size: int = 8
    page_bytes: int = 65536
    observation_timeout_s: float = 5
    poll_interval_s: float = 0.1
    shutdown_timeout_s: float = 5
    producer_execution_rules: tuple[_ProducerExecutionRule, ...] = ()
    producer_output_rules: tuple[_ProducerOutputRule, ...] = ()
    planning_rules: tuple[_PlanningRule, ...] = ()
    producer_registration_rules: tuple[_ProducerRegistrationRule, ...] = ()
    wait_rules: tuple[_WaitRule, ...] = ()
    continuation_rules: tuple[_ContinuationRule, ...] = ()
    clarification_rules: tuple[_ClarificationRule, ...] = ()
    clarification_maintenance_sources: tuple[_ClarificationMaintenanceSource, ...] = ()
    planned_producer_rules: tuple[_PlannedProducerRule, ...] = ()
    request_maintenance_sources: tuple[_RequestMaintenanceSource, ...] = ()
    discovery_slots: int = 4
    discovery_bytes: int = 512 * 1024


def _snapshot_registration(app, registration):
    if type(registration) is not _HostRegistration:
        raise TypeError("Host requires its exact runtime registration.")
    if type(registration.limits) is not HostOwnershipLimits:
        raise TypeError("Host requires finite ownership limits.")
    limits = HostOwnershipLimits(
        registration.limits.execution_slots,
        registration.limits.maintenance_slots,
        registration.limits.retained_operations,
        registration.limits.retained_bytes,
    )
    _positive_integer(registration.batch_size, 32, "batch_size")
    _positive_integer(registration.page_bytes, 65536, "page_bytes")
    _positive_integer(registration.discovery_slots, 32, "discovery_slots")
    _positive_integer(registration.discovery_bytes, 4 * 1024 * 1024, "discovery_bytes")
    if registration.discovery_bytes < 131072:
        raise ValueError("Host discovery must reserve one complete query and owner result.")
    for interval in (
        registration.observation_timeout_s,
        registration.poll_interval_s,
        registration.shutdown_timeout_s,
    ):
        _seconds(interval)
    if type(registration.producer_sources) is not tuple or len(registration.producer_sources) > 32:
        raise ValueError("Host producer sources exceed their finite bound.")
    if type(registration.producer_rules) is not tuple or len(registration.producer_rules) > 32:
        raise ValueError("Host producer rules exceed their finite bound.")
    if (
        type(registration.producer_execution_rules) is not tuple
        or len(registration.producer_execution_rules) > 32
    ):
        raise ValueError("Host producer execution rules exceed their finite bound.")
    if (
        type(registration.producer_output_rules) is not tuple
        or len(registration.producer_output_rules) > 32
    ):
        raise ValueError("Host producer output rules exceed their finite bound.")
    if type(registration.planning_rules) is not tuple or len(registration.planning_rules) > 32:
        raise ValueError("Host planning rules exceed their finite bound.")
    if (
        type(registration.producer_registration_rules) is not tuple
        or len(registration.producer_registration_rules) > 32
    ):
        raise ValueError("Host producer registration rules exceed their finite bound.")
    if type(registration.wait_rules) is not tuple or len(registration.wait_rules) > 32:
        raise ValueError("Host wait rules exceed their finite bound.")
    if (
        type(registration.continuation_rules) is not tuple
        or len(registration.continuation_rules) > 32
    ):
        raise ValueError("Host continuation rules exceed their finite bound.")
    if (
        type(registration.clarification_rules) is not tuple
        or len(registration.clarification_rules) > 32
    ):
        raise ValueError("Host clarification rules exceed their finite bound.")
    if (
        type(registration.clarification_maintenance_sources) is not tuple
        or len(registration.clarification_maintenance_sources) > 32
    ):
        raise ValueError("Host clarification maintenance sources exceed their bound.")
    if (
        type(registration.planned_producer_rules) is not tuple
        or len(registration.planned_producer_rules) > 32
    ):
        raise ValueError("Host planned producer rules exceed their finite bound.")
    if (
        type(registration.request_maintenance_sources) is not tuple
        or len(registration.request_maintenance_sources) > 32
    ):
        raise ValueError("Host request maintenance sources exceed their finite bound.")
    if (
        not registration.producer_sources
        and not registration.planning_rules
        and not registration.producer_registration_rules
        and not registration.wait_rules
        and not registration.continuation_rules
        and not registration.clarification_rules
        and not registration.clarification_maintenance_sources
        and not registration.planned_producer_rules
        and not registration.request_maintenance_sources
    ):
        raise ValueError("Host requires at least one configured source or planning request.")
    redactor = app._secret_redactor
    sources = []
    rules = []
    execution_rules = []
    output_rules = []
    planning_rules = []
    registration_rules = []
    wait_rules = []
    continuation_rules = []
    clarification_rules = []
    maintenance_sources = []
    planned_producer_rules = []
    seen_sources = set()
    seen_rules = set()
    retained = 0
    for source in registration.producer_sources:
        if type(source) is not _ProducerSource:
            raise TypeError("Host source must be a typed participant binding.")
        participant = prepare_contract(ParticipantRef, source.participant, redactor=redactor)
        context = prepare_contract(CollaborationAccessContext, source.context, redactor=redactor)
        key = contract_bytes(participant, redactor=redactor)
        if key in seen_sources:
            raise ValueError("Host producer sources must be distinct.")
        seen_sources.add(key)
        retained += len(key) + len(contract_bytes(context, redactor=redactor))
        sources.append(_ProducerSource(participant, context))
    for rule in registration.producer_rules:
        if type(rule) is not _ProducerMaintenanceRule:
            raise TypeError("Host maintenance selection must be typed.")
        intent = prepare_contract(HostProducerMaintenance, rule.intent, redactor=redactor)
        context = prepare_contract(CollaborationAccessContext, rule.context, redactor=redactor)
        disclosure = (
            None
            if rule.disclosure_context is None
            else prepare_contract(
                SessionExportAccessContext, rule.disclosure_context, redactor=redactor
            )
        )
        key = contract_bytes(intent, redactor=redactor)
        if (intent.action in _DISCLOSURE_ACTIONS) != (disclosure is not None):
            raise ValueError("Host maintenance requires role-specific current access.")
        if key in seen_rules:
            raise ValueError("Host maintenance rules must be distinct.")
        seen_rules.add(key)
        retained += len(key) + len(contract_bytes(context, redactor=redactor))
        if disclosure is not None:
            retained += len(contract_bytes(disclosure, redactor=redactor))
        rules.append(_ProducerMaintenanceRule(intent, context, disclosure))
    for rule in registration.producer_execution_rules:
        if type(rule) is not _ProducerExecutionRule:
            raise TypeError("Host execution selection must be typed.")
        intent = prepare_contract(HostProducerExecution, rule.intent, redactor=redactor)
        context = prepare_contract(CollaborationAccessContext, rule.context, redactor=redactor)
        producer_context = prepare_contract(
            MandateAccessContext, rule.producer_context, redactor=redactor
        )
        key = contract_bytes(intent, redactor=redactor)
        if key in seen_rules:
            raise ValueError("Host execution rules must be distinct.")
        seen_rules.add(key)
        retained += len(key) + len(contract_bytes(context, redactor=redactor))
        retained += len(contract_bytes(producer_context, redactor=redactor))
        execution_rules.append(_ProducerExecutionRule(intent, context, producer_context))
    seen_outputs = set()
    for rule in registration.producer_output_rules:
        if type(rule) is not _ProducerOutputRule:
            raise TypeError("Host output selection must be typed.")
        recovery = prepare_contract(ProducerOutputRecovery, rule.recovery, redactor=redactor)
        context = prepare_contract(CollaborationAccessContext, rule.context, redactor=redactor)
        failure_context = prepare_contract(
            SessionExportAccessContext, rule.failure_context, redactor=redactor
        )
        key = contract_bytes(recovery, redactor=redactor)
        if key in seen_outputs:
            raise ValueError("Host output rules must be distinct.")
        seen_outputs.add(key)
        retained += len(key) + len(contract_bytes(context, redactor=redactor))
        retained += len(contract_bytes(failure_context, redactor=redactor))
        if type(rule.destinations) is not tuple or not 1 <= len(rule.destinations) <= 32:
            raise ValueError("Host output destinations exceed their finite bound.")
        destinations = []
        seen_destinations = set()
        for destination in rule.destinations:
            if type(destination) is not _ProducerDisclosure:
                raise TypeError("Host disclosure selection must be typed.")
            operation = prepare_contract(OperationRef, destination.destination, redactor=redactor)
            access = prepare_contract(
                SessionExportAccessContext, destination.context, redactor=redactor
            )
            if operation in seen_destinations:
                raise ValueError("Host disclosure destinations must be distinct.")
            seen_destinations.add(operation)
            retained += len(contract_bytes(operation, redactor=redactor))
            retained += len(contract_bytes(access, redactor=redactor))
            destinations.append(_ProducerDisclosure(operation, access))
        output_rules.append(
            _ProducerOutputRule(recovery, context, failure_context, tuple(destinations))
        )
    seen_plans = set()
    for rule in registration.planning_rules:
        if type(rule) is not _PlanningRule:
            raise TypeError("Host planning selection must be typed.")
        expected = prepare_contract(RequestPlanningRequest, rule.expected, redactor=redactor)
        context = prepare_contract(MandateAccessContext, rule.context, redactor=redactor)
        if expected.operation in seen_plans:
            raise ValueError("Host planning operations must be distinct.")
        seen_plans.add(expected.operation)
        retained += len(contract_bytes(expected, redactor=redactor))
        retained += len(contract_bytes(context, redactor=redactor))
        planning_rules.append(_PlanningRule(expected, context))
    seen_registrations = set()
    for rule in registration.producer_registration_rules:
        if type(rule) is not _ProducerRegistrationRule:
            raise TypeError("Host producer registration selection must be typed.")
        expected = prepare_contract(ProducerOutputRegistration, rule.expected, redactor=redactor)
        context = prepare_contract(CollaborationAccessContext, rule.context, redactor=redactor)
        producer_context = prepare_contract(
            MandateAccessContext, rule.producer_context, redactor=redactor
        )
        if expected.operation in seen_registrations:
            raise ValueError("Host producer registration operations must be distinct.")
        seen_registrations.add(expected.operation)
        retained += sum(
            len(contract_bytes(value, redactor=redactor))
            for value in (expected, context, producer_context)
        )
        registration_rules.append(_ProducerRegistrationRule(expected, context, producer_context))
    seen_waits = set()
    for rule in registration.wait_rules:
        if type(rule) is not _WaitRule:
            raise TypeError("Host wait selection must be typed.")
        expected = prepare_contract(WaitRecovery, rule.expected, redactor=redactor)
        context = prepare_contract(MandateAccessContext, rule.context, redactor=redactor)
        if expected.operation in seen_waits:
            raise ValueError("Host wait operations must be distinct.")
        seen_waits.add(expected.operation)
        retained += len(contract_bytes(expected, redactor=redactor)) + len(
            contract_bytes(context, redactor=redactor)
        )
        wait_rules.append(_WaitRule(expected, context))
    seen_continuations = set()
    for rule in registration.continuation_rules:
        copied, size = snapshot_continuation_rule(app, rule)
        if copied.expected in seen_continuations:
            raise ValueError("Host continuation operations must be distinct.")
        seen_continuations.add(copied.expected)
        retained += size
        continuation_rules.append(copied)
    seen_clarifications = set()
    for rule in registration.clarification_rules:
        copied, size = snapshot_clarification_rule(app, rule)
        if copied.request.operation in seen_clarifications:
            raise ValueError("Host clarification operations must be distinct.")
        seen_clarifications.add(copied.request.operation)
        retained += size
        clarification_rules.append(copied)
    for source in registration.clarification_maintenance_sources:
        copied, size = snapshot_maintenance_source(app, source)
        if copied in maintenance_sources:
            raise ValueError("Host maintenance sources must be distinct.")
        maintenance_sources.append(copied)
        retained += size
    seen_planned = set()
    for rule in registration.planned_producer_rules:
        copied, size = snapshot_planned_producer_rule(app, rule)
        if copied.selected.operation in seen_planned:
            raise ValueError("Host planned producer operations must be distinct.")
        seen_planned.add(copied.selected.operation)
        retained += size
        planned_producer_rules.append(copied)
    request_sources = []
    for source in registration.request_maintenance_sources:
        copied, size = snapshot_request_source(app, source)
        if copied in request_sources:
            raise ValueError("Host request maintenance sources must be distinct.")
        request_sources.append(copied)
        retained += size
    if retained > 65536:
        raise ValueError("Host registration exceeds its retained-byte bound.")
    return _HostRegistration(
        limits,
        tuple(sources),
        tuple(rules),
        registration.batch_size,
        registration.page_bytes,
        registration.observation_timeout_s,
        registration.poll_interval_s,
        registration.shutdown_timeout_s,
        tuple(execution_rules),
        tuple(output_rules),
        tuple(planning_rules),
        tuple(registration_rules),
        tuple(wait_rules),
        tuple(continuation_rules),
        tuple(clarification_rules),
        tuple(maintenance_sources),
        tuple(planned_producer_rules),
        tuple(request_sources),
        registration.discovery_slots,
        registration.discovery_bytes,
    )
