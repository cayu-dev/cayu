"""Explicit host composition over existing native owners.

Scheduling composes the existing native owners. Finite local discovery cannot
prove deployment-wide quiescence; inspection reports partial coverage.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cayu.applications import CayuApp

from cayu._exception_groups import (
    exception_cause,
    exception_context,
    failure_control_cause,
    set_exception_cause,
    set_exception_context,
)
from cayu.collaboration._contracts import ExactMatch, ExactNotFound
from cayu.collaboration._host_clarification_maintenance import (
    HostClarificationMaintenanceResult,
    HostClarificationMaintenanceSweep,
    acknowledge_clarification_maintenance,
)
from cayu.collaboration._host_clarifications import (
    HostClarificationResult,
    acknowledge_clarification,
    start_clarification,
)
from cayu.collaboration._host_continuation_readiness import continuation_delivery_ready
from cayu.collaboration._host_continuations import (
    HostContinuationOwner,
    HostContinuationResult,
)
from cayu.collaboration._host_lifecycle import HostLifecycle, HostPassOutcome
from cayu.collaboration._host_ownership import (
    HostCapacityExceeded,
    HostOwnership,
)
from cayu.collaboration._host_planned_producer import (
    HostPlannedProducerResult,
    acknowledge_planned_producer,
    planned_producer_identity,
    start_planned_producer,
)
from cayu.collaboration._host_planning import (
    HostPlanningResult,
    acknowledge_planning,
    lookup_host_plan,
    planning_needs_service,
    start_planning,
)
from cayu.collaboration._host_producer_execution import (
    HostExecutionResult,
    acknowledge_producer_execution,
    start_producer_execution,
)
from cayu.collaboration._host_producer_maintenance import (
    _DISCLOSURE_ACTIONS,
    HostMaintenanceResult,
)
from cayu.collaboration._host_producer_observation import observe_producer_work
from cayu.collaboration._host_producer_registration import (
    HostRegistrationResult,
    acknowledge_producer_registration,
    producer_registration_ready,
    start_producer_registration,
)
from cayu.collaboration._host_producer_tasks import (
    acknowledge_producer_maintenance,
    start_producer_maintenance,
)
from cayu.collaboration._host_reads import HostReads
from cayu.collaboration._host_registration import (
    _HostRegistration as _HostRegistration,
)
from cayu.collaboration._host_registration import (
    _PlanningRule as _PlanningRule,
)
from cayu.collaboration._host_registration import (
    _ProducerDisclosure as _ProducerDisclosure,
)
from cayu.collaboration._host_registration import (
    _ProducerExecutionRule as _ProducerExecutionRule,
)
from cayu.collaboration._host_registration import (
    _ProducerMaintenanceRule as _ProducerMaintenanceRule,
)
from cayu.collaboration._host_registration import (
    _ProducerOutputRule as _ProducerOutputRule,
)
from cayu.collaboration._host_registration import (
    _ProducerRegistrationRule as _ProducerRegistrationRule,
)
from cayu.collaboration._host_registration import (
    _ProducerSource as _ProducerSource,
)
from cayu.collaboration._host_registration import (
    _snapshot_registration,
)
from cayu.collaboration._host_registration import (
    _WaitRule as _WaitRule,
)
from cayu.collaboration._host_requests import (
    HostRequestMaintenanceResult,
    HostRequestMaintenanceSweep,
    acknowledge_request_maintenance,
)
from cayu.collaboration._host_waits import HostWaitOwner, HostWaitResult
from cayu.collaboration._preparation import contract_bytes


@dataclass(frozen=True, slots=True)
class _HostInspection:
    closing: bool
    servicing_pending: bool
    active: int
    uncertain: int
    failed: int
    source_failures: int
    serviced: int
    observed_blocked: int
    discovery_pending: int = 0
    coverage_complete: bool = False

    @property
    def pending(self) -> bool:
        """Local ownership remains, including reads and active native execution."""
        return bool(
            self.servicing_pending or self.discovery_pending or self.active or self.uncertain
        )


class CollaborationHost:
    """Explicit lifetime with retained native tasks, not another execution loop.

    Construction validates and snapshots configuration only. No store, policy,
    model, or resource operation occurs until explicit servicing starts.
    """

    def __init__(self, app: CayuApp, registration: _HostRegistration) -> None:
        self._app = app
        self._registration = _snapshot_registration(app, registration)
        self._planned_identities = tuple(
            planned_producer_identity(
                rule.selected, rule.context, rule.producer_context, redactor=app._secret_redactor
            )[0]
            for rule in self._registration.planned_producer_rules
        )
        # Bounded by immutable local configuration, never a durable authority
        # cache. Completion suppresses only redundant attachment work; native
        # execution and disclosure still authenticate their own current grants.
        self._attached_planned = set()
        self._owned = HostOwnership(self._registration.limits)
        self._reads = HostReads(
            slots=self._registration.discovery_slots,
            bytes_limit=self._registration.discovery_bytes,
        )
        # Mandatory control discovery must remain possible when ordinary reads
        # occupy every configured slot or byte. Each configured source has one
        # stable read key and enough reserved capacity for its complete query
        # and result; a blocked source cannot consume another source's slot.
        request_sources = max(1, len(self._registration.request_maintenance_sources))
        self._maintenance_reads = HostReads(
            slots=request_sources, bytes_limit=131072 * request_sources
        )
        clarification_sources = max(1, len(self._registration.clarification_maintenance_sources))
        self._clarification_maintenance_reads = HostReads(
            slots=clarification_sources, bytes_limit=131072 * clarification_sources
        )
        # One bounded read per configured source prevents a blocked participant
        # from consuming another participant's cleanup discovery reservation.
        maintenance_sources = max(1, len(self._registration.producer_sources))
        self._producer_maintenance_reads = HostReads(
            slots=maintenance_sources, bytes_limit=131072 * maintenance_sources
        )
        self._read_close_errors: tuple[BaseException, ...] = ()
        self._cursors = [None] * len(self._registration.producer_sources)
        self._next_source = 0
        self._maintenance_cursors = [None] * len(self._registration.producer_sources)
        self._next_maintenance_source = 0
        self._next_rule = 0
        self._next_plan = 0
        self._next_registration = 0
        self._next_wait = 0
        self._wait_owner = HostWaitOwner(app)
        self._next_continuation = 0
        self._continuation_owner = HostContinuationOwner(app)
        self._next_clarification = 0
        self._next_planned_producer = 0
        self._clarification_maintenance = HostClarificationMaintenanceSweep(
            app,
            self._registration.clarification_maintenance_sources,
            self._clarification_maintenance_reads,
        )
        self._request_maintenance = HostRequestMaintenanceSweep(
            app, self._registration.request_maintenance_sources, self._maintenance_reads
        )
        self._next_family = 0
        self._source_errors: dict[int, Exception] = {}
        self._serviced = 0
        self._observed_blocked = 0
        self._closing_observer = False
        self._lifecycle = HostLifecycle(
            self._step,
            observation_timeout_s=self._registration.observation_timeout_s,
            poll_interval_s=self._registration.poll_interval_s,
        )

    def inspect(self) -> _HostInspection:
        owned = self._owned.inspect()
        return _HostInspection(
            closing=self._lifecycle.closing,
            servicing_pending=self._lifecycle.inspect().pending,
            active=len(owned.active),
            uncertain=len(owned.uncertain),
            failed=sum(outcome.error is not None for outcome in owned.completed),
            source_failures=(
                len(self._source_errors)
                + len(self._clarification_maintenance.errors)
                + len(self._request_maintenance.errors)
                + len(self._read_close_errors)
            ),
            serviced=self._serviced,
            observed_blocked=self._observed_blocked,
            discovery_pending=(
                self._reads.pending
                + self._maintenance_reads.pending
                + self._clarification_maintenance_reads.pending
                + self._producer_maintenance_reads.pending
            ),
        )

    async def _collect(self, timeout):
        await self._owned.observe(timeout)
        serviced, errors = self._acknowledge_completed()
        self._serviced = serviced
        self._raise_reconciled_failures(errors)
        return serviced

    @staticmethod
    def _raise_reconciled_failures(errors):
        for signal in errors:
            if isinstance(
                signal, (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit)
            ):
                cause = exception_cause(signal)
                set_exception_cause(
                    signal,
                    failure_control_cause([*errors, *(() if cause is None else (cause,))], signal),
                )
                raise signal
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Reconciled collaboration host operations failed", errors)

    def _acknowledge_completed(self):
        # Revisit every retained outcome, including one observed before a prior
        # collector failed. Reporting a task result is not an acknowledgement.
        serviced = 0
        errors = self._owned.take_control_failures()
        errors.extend(self._owned.take_reconciliation_failures())
        handlers = {
            HostExecutionResult: acknowledge_producer_execution,
            HostMaintenanceResult: acknowledge_producer_maintenance,
            HostPlanningResult: acknowledge_planning,
            HostRegistrationResult: acknowledge_producer_registration,
            HostWaitResult: self._wait_owner.acknowledge,
            HostContinuationResult: self._continuation_owner.acknowledge,
            HostClarificationResult: acknowledge_clarification,
            HostClarificationMaintenanceResult: acknowledge_clarification_maintenance,
            HostPlannedProducerResult: acknowledge_planned_producer,
            HostRequestMaintenanceResult: acknowledge_request_maintenance,
        }
        for outcome in self._owned.inspect().completed:
            if outcome.error is not None and not outcome.reconciled:
                continue
            # Control signals were delivered independently of settlement above.
            # A later positive read must not replay historical cancellation.
            failure = outcome.error if isinstance(outcome.error, Exception) else None
            if outcome.reconciled:
                # Only an internal native adapter can supply this positive
                # recovery result. Validate it through the normal handoff path;
                # never clear an uncertain failure just because its task ended.
                outcome = replace(outcome, error=None)
            try:
                handler = handlers.get(type(outcome.value))
                if handler is None:
                    raise RuntimeError("Host owner returned an unknown handoff type.")
                acknowledged = handler(self._owned, outcome)
            except Exception as error:
                # Preserve already-reconciled siblings' original errors too.
                # This operation remains retained because validation failed.
                errors.append(error)
                continue
            if acknowledged:
                if (
                    type(outcome.value) is HostPlannedProducerResult
                    and outcome.value.command is not None
                    and outcome.identity in self._planned_identities
                ):
                    self._attached_planned.add(outcome.identity)
                serviced += int(outcome.value.dispatched)
                if failure is not None:
                    errors.append(failure)
        return serviced, errors

    async def _service_plans(self, deadline, stop):
        if not self._owned.has_slot("maintenance"):
            # The retained native task already supervises this capacity. Do
            # not contend with its transactions just to rediscover work that
            # cannot be admitted in this pass.
            return
        rules = self._registration.planning_rules
        loop = asyncio.get_running_loop()
        for _ in range(min(self._registration.batch_size, len(rules))):
            if stop.is_set() or loop.time() >= deadline:
                break
            index = self._next_plan
            self._next_plan = (index + 1) % len(rules)
            rule = rules[index]
            try:
                observed = await self._reads.observe(
                    "plan:" + str(index),
                    expectation=contract_bytes(rule.expected, redactor=self._app._secret_redactor)
                    + contract_bytes(rule.context, redactor=self._app._secret_redactor),
                    reserved_bytes=131072,
                    read=lambda rule=rule: lookup_host_plan(
                        self._app, rule.expected, context=rule.context
                    ),
                )
                if observed is None:
                    continue
                found = observed.value
                if isinstance(found, ExactMatch):
                    if not planning_needs_service(found.receipt):
                        self._source_errors.pop(-index - 1, None)
                        continue
                elif not isinstance(found, ExactNotFound):
                    raise RuntimeError("Host planning source evidence is unavailable.")
                self._source_errors.pop(-index - 1, None)
                if stop.is_set() or loop.time() >= deadline:
                    break
                start_planning(
                    self._app,
                    self._owned,
                    rule.expected,
                    context=rule.context,
                    observation_deadline=deadline,
                )
            except HostCapacityExceeded:
                self._next_plan = index
                break
            except Exception as error:
                self._source_errors[-index - 1] = error

    async def _service_registrations(self, deadline, stop):
        if not self._owned.has_slot("maintenance"):
            return
        rules = self._registration.producer_registration_rules
        loop = asyncio.get_running_loop()
        for _ in range(min(self._registration.batch_size, len(rules))):
            if stop.is_set() or loop.time() >= deadline:
                break
            index = self._next_registration
            self._next_registration = (index + 1) % len(rules)
            rule = rules[index]
            error_key = -33 - index
            try:
                observed = await self._reads.observe(
                    "registration:" + str(index),
                    expectation=contract_bytes(rule.expected, redactor=self._app._secret_redactor)
                    + contract_bytes(rule.context, redactor=self._app._secret_redactor),
                    reserved_bytes=131072,
                    read=lambda rule=rule: producer_registration_ready(
                        self._app, rule.expected, context=rule.context
                    ),
                )
                if observed is None:
                    continue
                ready = observed.value
                if ready is True:
                    self._source_errors.pop(error_key, None)
                    continue
                if ready is not False:
                    raise RuntimeError("Host registration source evidence is unavailable.")
                self._source_errors.pop(error_key, None)
                if stop.is_set() or loop.time() >= deadline:
                    break
                start_producer_registration(
                    self._app,
                    self._owned,
                    rule.expected,
                    context=rule.context,
                    producer_context=rule.producer_context,
                    observation_deadline=deadline,
                )
            except HostCapacityExceeded:
                self._next_registration = index
                break
            except Exception as error:
                self._source_errors[error_key] = error

    async def _service_waits(self, deadline, stop):
        rules = self._registration.wait_rules
        loop = asyncio.get_running_loop()
        for _ in range(min(self._registration.batch_size, len(rules))):
            if stop.is_set() or loop.time() >= deadline:
                break
            index = self._next_wait
            self._next_wait = (index + 1) % len(rules)
            rule = rules[index]
            error_key = -65 - index
            try:
                self._wait_owner.start(
                    self._owned,
                    rule.expected,
                    context=rule.context,
                    observation_deadline=deadline,
                )
                self._source_errors.pop(error_key, None)
            except HostCapacityExceeded:
                # Capacity refusal did not service this rule. Start here next
                # time rather than repeatedly favoring the preceding wait.
                self._next_wait = index
                break
            except Exception as error:
                self._source_errors[error_key] = error

    async def _service_continuations(self, deadline, stop):
        if not self._owned.has_slot("execution"):
            # Native supervision already owns the occupied capacity. Re-reading
            # every complete wait/producer tuple cannot admit another invocation
            # and competes with that owner's durable progress. Maintenance and
            # dispatch-window renewal still run through their separate paths.
            return
        rules = self._registration.continuation_rules
        loop = asyncio.get_running_loop()
        for _ in range(min(self._registration.batch_size, len(rules))):
            if stop.is_set() or loop.time() >= deadline:
                break
            index = self._next_continuation
            self._next_continuation = (index + 1) % len(rules)
            rule = rules[index]
            error_key = -97 - index
            try:
                observed = await self._reads.observe(
                    "continuation-readiness:" + str(index),
                    expectation=contract_bytes(rule.expected, redactor=self._app._secret_redactor)
                    + contract_bytes(rule.service, redactor=self._app._secret_redactor)
                    + contract_bytes(rule.context, redactor=self._app._secret_redactor),
                    reserved_bytes=131072,
                    read=lambda rule=rule: continuation_delivery_ready(
                        self._app, rule.expected, rule.service, context=rule.context
                    ),
                )
                if observed is None:
                    continue
                self._source_errors.pop(error_key, None)
                if not observed.value:
                    self._observed_blocked += 1
                    continue
                if stop.is_set() or loop.time() >= deadline:
                    break
                self._continuation_owner.start(
                    self._owned,
                    rule.expected,
                    rule.request,
                    rule.service,
                    context=rule.context,
                    observation_deadline=deadline,
                    recovery_inactive_for_seconds=rule.recovery_inactive_for_seconds,
                )
                self._source_errors.pop(error_key, None)
            except HostCapacityExceeded:
                self._next_continuation = index
                break
            except Exception as error:
                self._source_errors[error_key] = error

    async def _service_clarifications(self, deadline, stop):
        rules = self._registration.clarification_rules
        loop = asyncio.get_running_loop()
        for _ in range(min(self._registration.batch_size, len(rules))):
            if stop.is_set() or loop.time() >= deadline:
                break
            index = self._next_clarification
            self._next_clarification = (index + 1) % len(rules)
            rule = rules[index]
            error_key = -129 - index
            try:
                start_clarification(
                    self._app,
                    self._owned,
                    rule.request,
                    context=rule.context,
                    delivery_context=rule.delivery_context,
                    observation_deadline=deadline,
                )
                self._source_errors.pop(error_key, None)
            except HostCapacityExceeded:
                self._next_clarification = index
                break
            except Exception as error:
                self._source_errors[error_key] = error

    async def _service_planned_producers(self, deadline, stop):
        if not self._owned.has_slot("maintenance"):
            # The retained attachment already owns this slot and its dispatch
            # window is renewed by _step. Revalidating a large frozen plan here
            # cannot admit work and only competes with that owner's progress.
            return
        rules = self._registration.planned_producer_rules
        loop = asyncio.get_running_loop()
        for _ in range(min(self._registration.batch_size, len(rules))):
            if stop.is_set() or loop.time() >= deadline or not self._owned.has_slot("maintenance"):
                break
            index = self._next_planned_producer
            self._next_planned_producer = (index + 1) % len(rules)
            if self._planned_identities[index] in self._attached_planned:
                continue
            rule = rules[index]
            error_key = -161 - index
            try:
                start_planned_producer(
                    self._app,
                    self._owned,
                    rule.selected,
                    context=rule.context,
                    producer_context=rule.producer_context,
                    observation_deadline=deadline,
                )
                self._source_errors.pop(error_key, None)
            except HostCapacityExceeded:
                self._next_planned_producer = index
                break
            except Exception as error:
                self._source_errors[error_key] = error

    async def _service_clarification_maintenance(self, deadline, stop):
        registration = self._registration
        await self._clarification_maintenance.step(
            self._owned,
            deadline,
            stop,
            batch_size=registration.batch_size,
            page_bytes=registration.page_bytes,
        )

    async def _step(self, deadline, stop):
        loop = asyncio.get_running_loop()
        self._owned.renew_dispatch_windows(deadline)
        serviced = await self._collect(0.001)
        families = (
            self._service_request_maintenance,
            self._service_clarification_maintenance,
            self._service_clarifications,
            self._service_continuations,
            self._service_waits,
            self._service_plans,
            self._service_registrations,
            self._service_planned_producers,
            self._service_producers,
            self._service_producer_maintenance,
        )
        first = self._next_family
        # Advance after an actual ownership grant, not after observing an
        # occupied slot. Polling must not erase the successor's turn while a
        # slow operation runs: otherwise release timing can repeatedly give
        # the same family the sole maintenance slot. Source cursors continue
        # rotating independently and every family is still visited each pass.
        granted = False
        self._observed_blocked = 0
        for offset in range(len(families)):
            if stop.is_set() or loop.time() >= deadline:
                break
            index = (first + offset) % len(families)
            before = self._owned.pending
            try:
                await families[index](deadline, stop)
            finally:
                if self._owned.pending > before:
                    granted = True
                    self._next_family = (index + 1) % len(families)
        if not granted and not self._owned.pending:
            # No occupied slot has a successor to preserve. Rotate read-only
            # discovery too, including when a bounded pass cannot visit all
            # families before its observation deadline.
            self._next_family = (first + 1) % len(families)
        return HostPassOutcome(serviced, False)

    async def _service_request_maintenance(self, deadline, stop):
        registration = self._registration
        await self._request_maintenance.step(
            self._owned,
            deadline,
            stop,
            batch_size=registration.batch_size,
            page_bytes=registration.page_bytes,
        )

    async def _service_producers(self, deadline, stop):
        registration = self._registration
        if (
            (registration.producer_rules or registration.producer_output_rules)
            and not registration.producer_execution_rules
            and not registration.planned_producer_rules
        ):
            return
        await self._service_producer_role(deadline, stop, maintenance=False)

    async def _service_producer_maintenance(self, deadline, stop):
        if self._registration.producer_rules or self._registration.producer_output_rules:
            await self._service_producer_role(deadline, stop, maintenance=True)

    async def _service_producer_role(self, deadline, stop, *, maintenance):
        loop = asyncio.get_running_loop()
        registration = self._registration
        execution_enabled = not maintenance and bool(
            registration.producer_execution_rules or registration.planned_producer_rules
        )
        maintenance_enabled = maintenance and bool(
            registration.producer_rules or registration.producer_output_rules
        )
        if (execution_enabled or maintenance_enabled) and not (
            (execution_enabled and self._owned.has_slot("execution"))
            or (maintenance_enabled and self._owned.has_slot("maintenance"))
        ):
            # Native work already owns this capacity. Re-reading its complete
            # authority while it is settling only contends with its transactions;
            # the retained task and next source scan still own progress/recovery.
            return
        scanned = 0
        blocked = 0
        # Advance before awaiting each source: a failing source cannot monopolize
        # subsequent passes. Cursors are only bounded scan hints, never claims.
        while scanned < min(registration.batch_size, len(registration.producer_sources)):
            if stop.is_set() or loop.time() >= deadline:
                break
            index = self._next_maintenance_source if maintenance else self._next_source
            if maintenance:
                self._next_maintenance_source = (index + 1) % len(registration.producer_sources)
            else:
                self._next_source = (index + 1) % len(registration.producer_sources)
            cursors = self._maintenance_cursors if maintenance else self._cursors
            error_key = len(registration.producer_sources) + index if maintenance else index
            scanned += 1
            source = registration.producer_sources[index]
            try:
                found = await observe_producer_work(
                    self._producer_maintenance_reads if maintenance else self._reads,
                    "producer-source:" + str(index),
                    self._app,
                    registration,
                    source,
                    cursors[index],
                    maintenance_only=maintenance,
                )
                if found is None:
                    continue
                page = found.value
            except Exception as error:
                self._source_errors[error_key] = error
                continue
            self._source_errors.pop(error_key, None)
            cursors[index] = page.next_cursor
            if page.errors:
                self._source_errors[error_key] = ExceptionGroup(
                    "Host producer observations failed", list(page.errors)
                )
            blocked += page.blocked
            for rule in page.execution:
                if stop.is_set() or loop.time() >= deadline:
                    break
                try:
                    start_producer_execution(
                        self._app,
                        self._owned,
                        rule.intent,
                        context=rule.context,
                        producer_context=rule.producer_context,
                        observation_deadline=deadline,
                    )
                except HostCapacityExceeded:
                    break
            for offset in range(len(registration.producer_rules)):
                rule_index = (self._next_rule + offset) % len(registration.producer_rules)
                if rule_index not in page.maintenance:
                    continue
                if stop.is_set() or loop.time() >= deadline:
                    break
                rule = registration.producer_rules[rule_index]
                try:
                    start_producer_maintenance(
                        self._app,
                        self._owned,
                        rule.intent,
                        context=rule.context,
                        disclosure_context=rule.disclosure_context,
                        observation_deadline=deadline,
                    )
                except HostCapacityExceeded:
                    break
                self._next_rule = (rule_index + 1) % len(registration.producer_rules)
                break
            for rule_index, intent in page.output:
                if stop.is_set() or loop.time() >= deadline:
                    break
                rule = registration.producer_output_rules[rule_index]
                access = None
                if intent.action in _DISCLOSURE_ACTIONS:
                    access = (
                        rule.failure_context
                        if intent.destination is None
                        else next(
                            entry.context
                            for entry in rule.destinations
                            if entry.destination == intent.destination
                        )
                    )
                try:
                    start_producer_maintenance(
                        self._app,
                        self._owned,
                        intent,
                        context=rule.context,
                        disclosure_context=access,
                        observation_deadline=deadline,
                    )
                except HostCapacityExceeded:
                    break
        self._observed_blocked += blocked

    async def service_once(self) -> _HostInspection:
        await self._lifecycle.service_once()
        return self.inspect()

    async def run(self) -> None:
        await self._lifecycle.run()

    async def aclose(self) -> _HostInspection:
        if self._closing_observer:
            raise RuntimeError("Collaboration host close observation is already active.")
        self._closing_observer = True
        deadline = asyncio.get_running_loop().time() + self._registration.shutdown_timeout_s
        self._owned.request_close()
        self._reads.request_close()
        self._maintenance_reads.request_close()
        self._clarification_maintenance_reads.request_close()
        self._producer_maintenance_reads.request_close()
        try:
            observed = await self._lifecycle.aclose(timeout_s=self._registration.shutdown_timeout_s)
            remaining = deadline - asyncio.get_running_loop().time()
            errors = []
            collected = False
            if not observed.pending and remaining > 0:
                await self._owned.close(remaining)
                collected = True
            if not observed.pending:
                for reads in (
                    self._reads,
                    self._maintenance_reads,
                    self._clarification_maintenance_reads,
                    self._producer_maintenance_reads,
                ):
                    read_errors = await reads.close(
                        max(0, deadline - asyncio.get_running_loop().time())
                    )
                    self._read_close_errors += read_errors
                    # Report each drained pool before another await can cancel
                    # observation and discard its original errors. A subsequent
                    # close still owns every undrained sibling pool.
                    if collected:
                        _, errors = self._acknowledge_completed()
                        collected = False
                    errors.extend(read_errors)
                    self._raise_reconciled_failures(errors)
            self._raise_reconciled_failures(errors)
            return self.inspect()
        finally:
            self._closing_observer = False

    async def __aenter__(self) -> CollaborationHost:
        if self._lifecycle.closing:
            raise RuntimeError("Collaboration host is closing.")
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        failure = None
        try:
            await self.aclose()
        except BaseException as error:
            failure = error
        if failure is not None:
            if exc is None or failure is exc:
                raise failure
            # Remove only Python's immediate implicit primary -> cleanup link
            # before making the ordered relation explicit. Preserve original
            # objects, nested groups and any independently authored causes.
            if exception_context(failure) is exc:
                set_exception_context(failure, None)
            controls = (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit)
            if isinstance(exc, controls):
                prior = exception_cause(exc)
                combined = (
                    failure
                    if prior is None or prior is failure
                    else BaseExceptionGroup("Host shutdown failures", [prior, failure])
                )
                set_exception_cause(exc, combined)
                raise exc
            if isinstance(failure, controls):
                prior = exception_cause(failure)
                combined = (
                    exc
                    if prior is None or prior is exc
                    else BaseExceptionGroup("Host body and shutdown failures", [exc, prior])
                )
                set_exception_cause(failure, combined)
                raise failure
            raise BaseExceptionGroup("Host body and shutdown failures", [exc, failure])
        return False
