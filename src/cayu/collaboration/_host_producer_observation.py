"""Retained producer discovery and scheduling observations, never dispatch grants.

Keep the source query and its dependent reads under one read owner. A slow
inspection must not block a host pass or require another source query to find
the handle needed to drain it. Receiving mutations revalidate all authority.
"""

from dataclasses import dataclass

from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration._host_discovery import discover_host_source
from cayu.collaboration._host_output_selection import select_producer_output
from cayu.collaboration._host_planned_producer import execution_for_planned_producer
from cayu.collaboration._host_producer_maintenance import HostProducerMaintenance
from cayu.collaboration._host_registration import _ProducerExecutionRule
from cayu.collaboration._preparation import contract_bytes
from cayu.collaboration._producer_inspection import inspect_producer_output
from cayu.collaboration._producer_recovery import ProducerPendingOutput


@dataclass(frozen=True, slots=True)
class ProducerWorkObservation:
    next_cursor: int | None
    execution: tuple[_ProducerExecutionRule, ...]
    maintenance: tuple[int, ...]
    output: tuple[tuple[int, HostProducerMaintenance], ...]
    blocked: int
    errors: tuple[Exception, ...]


async def observe_producer_work(
    reads, key, app, registration, source, cursor, *, maintenance_only=False
):
    redactor = app._secret_redactor
    expectation = b"\0".join(
        (
            contract_bytes(source.context, redactor=redactor),
            contract_bytes(source.participant, redactor=redactor),
            str(cursor).encode("ascii"),
            str(registration.page_bytes).encode("ascii"),
            str(maintenance_only).encode("ascii"),
        )
    )

    async def read():
        # One bounded source item plus one bounded inspection can coexist. Do
        # not retain full inspections for a page; retain only scheduling intent.
        page = await discover_host_source(
            app,
            "producers",
            participant=source.participant,
            context=source.context,
            cursor=cursor,
            limit=1,
            max_bytes=registration.page_bytes,
        )
        executions, maintenance, output, errors = [], [], [], []
        blocked = 0
        retained = len(expectation) + page.retained_bytes
        for item in page.items:
            if type(item) is not ProducerPendingOutput:
                raise TypeError("Native producer discovery returned another source type.")
            rules = [] if maintenance_only else list(registration.producer_execution_rules)
            for rule in () if maintenance_only else registration.planned_producer_rules:
                if retained > 65536:
                    raise ValueError("Host producer scheduling observation exceeds its byte bound.")
                try:
                    intent = await execution_for_planned_producer(app, rule, item.recovery)
                    if intent is not None:
                        retained += len(contract_bytes(intent, redactor=redactor))
                        rules.append(
                            _ProducerExecutionRule(intent, rule.context, rule.producer_context)
                        )
                except Exception as error:
                    errors.append(error)
            for rule in rules:
                if retained > 65536:
                    raise ValueError("Host producer scheduling observation exceeds its byte bound.")
                if rule.intent.recovery != item.recovery:
                    continue
                try:
                    current = await inspect_producer_output(
                        app, item.recovery, context=rule.context, wait_for_settlement=True
                    )
                    if not isinstance(current, ExactMatch):
                        raise RuntimeError("Host execution source evidence is unavailable.")
                    if current.receipt.completion is None and (
                        current.receipt.state == "registered"
                        or (
                            current.receipt.state == "launch_claimed"
                            and current.receipt.request_state == "open"
                            and rule.intent.recovery_inactive_for_seconds is not None
                        )
                    ):
                        executions.append(rule)
                    del current
                except Exception as error:
                    errors.append(error)
            if not maintenance_only:
                if retained > 65536:
                    raise ValueError("Host producer scheduling observation exceeds its byte bound.")
                continue
            maintenance.extend(
                index
                for index, rule in enumerate(registration.producer_rules)
                if rule.intent.recovery == item.recovery
            )
            for index, rule in enumerate(registration.producer_output_rules):
                if retained > 65536:
                    raise ValueError("Host producer scheduling observation exceeds its byte bound.")
                if rule.recovery != item.recovery:
                    continue
                try:
                    current = await inspect_producer_output(
                        app, item.recovery, context=rule.context, wait_for_settlement=True
                    )
                    if not isinstance(current, ExactMatch):
                        raise RuntimeError("Host output source evidence is unavailable.")
                    if {entry.destination for entry in rule.destinations} != {
                        entry.destination for entry in current.receipt.destinations
                    }:
                        raise ValueError("Host output destinations conflict with registration.")
                    selected = select_producer_output(current.receipt)
                    del current
                    blocked += int(selected.blocked is not None)
                    if selected.intent is not None:
                        retained += len(contract_bytes(selected.intent, redactor=redactor))
                        output.append((index, selected.intent))
                except Exception as error:
                    errors.append(error)
            if retained > 65536:
                raise ValueError("Host producer scheduling observation exceeds its byte bound.")
        next_cursor = page.next_cursor
        if next_cursor is not None and type(next_cursor) is not int:
            raise TypeError("Native producer discovery returned another cursor type.")
        return ProducerWorkObservation(
            next_cursor,
            tuple(executions),
            tuple(maintenance),
            tuple(output),
            blocked,
            tuple(errors),
        )

    return await reads.observe(key, expectation=expectation, reserved_bytes=131072, read=read)
