from __future__ import annotations

from decimal import Decimal

from cayu.core import Event, EventType
from cayu.runtime.budgets import (
    BudgetCheck,
    budget_check_payload,
    project_budget_inspection_event,
    session_budget_inspection,
)
from cayu.runtime.costs import SessionCostSummary


def _budget_limit_id(value: int) -> str:
    return f"blim_{value:064x}"


def _model_attempt_identity(value: int) -> dict[str, str]:
    return {
        "model_step_id": f"mstep_{value:032x}",
        "model_attempt_id": f"matt_{value:032x}",
    }


def test_budget_inspection_uses_latest_fully_priced_checks_without_reservations() -> None:
    def checked_event(actual: str) -> Event:
        amount = Decimal(actual)
        summary = SessionCostSummary(
            session_id="sess_checked",
            currency="USD",
            model_steps=1,
            priced_model_steps=1,
            unpriced_model_steps=0,
            total_cost=amount,
        )
        check = BudgetCheck(
            budget_limit_id=_budget_limit_id(1),
            scope="session",
            key="sess_checked",
            currency="USD",
            maximum=Decimal("1"),
            actual=amount,
            action="interrupt",
            model_steps=1,
            unpriced_model_steps=0,
            limit_reached=False,
            message="priced",
            cost_summary=summary,
        )
        return Event(
            type=EventType.BUDGET_CHECKED,
            session_id="sess_checked",
            payload=budget_check_payload(check),
        )

    for event_type in (EventType.BUDGET_CHECKED, EventType.BUDGET_LIMIT_REACHED):
        events = [
            checked_event(actual).model_copy(update={"type": event_type})
            for actual in ("0.10", "0.25")
        ]

        inspection = session_budget_inspection(
            [project_budget_inspection_event(event) for event in events]
        )

        assert inspection.cost_state == "priced"
        assert inspection.amount == "0.25"
        assert inspection.currency == "USD"


def test_budget_inspection_does_not_double_count_parallel_limit_ledgers() -> None:
    for identities in (
        (("1", "interrupt"), ("2", "interrupt")),
        (("1", "interrupt"), ("1", "notify")),
        (("1", "interrupt"), ("1", "interrupt")),
    ):
        events: list[Event] = []
        for index, (maximum, action) in enumerate(identities):
            reservation_id = f"reservation-{index}-{maximum}-{action}"
            budget_limit_id = _budget_limit_id(index + 1)
            events.extend(
                [
                    Event(
                        type=EventType.BUDGET_RESERVED,
                        session_id="sess_parallel_limits",
                        payload={
                            "reservation_id": reservation_id,
                            "budget_limit_id": budget_limit_id,
                            **_model_attempt_identity(1),
                            "scope": "session",
                            "key": "sess_parallel_limits",
                            "window": "all_time",
                            "currency": "USD",
                            "maximum": maximum,
                            "action": action,
                            "requested": "0.50",
                        },
                    ),
                    Event(
                        type=EventType.BUDGET_RECONCILED,
                        session_id="sess_parallel_limits",
                        payload={
                            "reservation_id": reservation_id,
                            "budget_limit_id": budget_limit_id,
                            **_model_attempt_identity(1),
                            "actual_amount": "0.25",
                            "pricing": {"provider_name": "fake", "model": "model"},
                        },
                    ),
                ]
            )

        inspection = session_budget_inspection(events)

        assert inspection.cost_state == "priced"
        assert inspection.amount == "0.25"
        assert inspection.currency == "USD"


def test_budget_inspection_does_not_leak_a_currency_for_mixed_attempts() -> None:
    events: list[Event] = []
    for index, currency in enumerate(("USD", "EUR"), start=1):
        reservation_id = f"reservation-{index}"
        budget_limit_id = _budget_limit_id(index)
        identity = _model_attempt_identity(index)
        events.extend(
            [
                Event(
                    type=EventType.BUDGET_RESERVED,
                    session_id="sess_mixed_attempt_currencies",
                    payload={
                        "reservation_id": reservation_id,
                        "budget_limit_id": budget_limit_id,
                        **identity,
                        "scope": "session",
                        "key": "sess_mixed_attempt_currencies",
                        "window": "all_time",
                        "currency": currency,
                        "maximum": "1",
                        "action": "interrupt",
                        "requested": "0.50",
                    },
                ),
                Event(
                    type=EventType.BUDGET_RECONCILED,
                    session_id="sess_mixed_attempt_currencies",
                    payload={
                        "reservation_id": reservation_id,
                        "budget_limit_id": budget_limit_id,
                        **identity,
                        "actual_amount": "0.25",
                        "pricing": {"provider_name": "fake", "model": "model"},
                    },
                ),
            ]
        )

    inspection = session_budget_inspection(events)

    assert inspection.cost_state == "mixed_currency"
    assert inspection.amount is None
    assert inspection.currency is None


def test_budget_inspection_marks_malformed_reservation_evidence_partial() -> None:
    budget_limit_id = _budget_limit_id(1)
    valid_events = [
        Event(
            type=EventType.BUDGET_RESERVED,
            session_id="sess_malformed_evidence",
            payload={
                "reservation_id": "reservation-valid",
                "budget_limit_id": budget_limit_id,
                **_model_attempt_identity(1),
                "scope": "session",
                "key": "sess_malformed_evidence",
                "window": "all_time",
                "currency": "USD",
                "maximum": "1",
                "action": "interrupt",
                "requested": "0.50",
            },
        ),
        Event(
            type=EventType.BUDGET_RECONCILED,
            session_id="sess_malformed_evidence",
            payload={
                "reservation_id": "reservation-valid",
                "budget_limit_id": budget_limit_id,
                **_model_attempt_identity(1),
                "actual_amount": "0.25",
                "pricing": {"provider_name": "fake", "model": "model"},
            },
        ),
    ]
    malformed_events = (
        Event(
            type=EventType.BUDGET_RESERVED,
            session_id="sess_malformed_evidence",
            payload={"reservation_id": 42},
        ),
        Event(
            type=EventType.BUDGET_RECONCILED,
            session_id="sess_malformed_evidence",
            payload={"reservation_id": 42, "actual_amount": "0.25", "pricing": {}},
        ),
        Event(
            type=EventType.BUDGET_RESERVATION_RELEASED,
            session_id="sess_malformed_evidence",
            payload={"reservation_id": 42},
        ),
    )

    for malformed_event in malformed_events:
        inspection = session_budget_inspection([*valid_events, malformed_event])

        assert inspection.cost_state == "partial"
        assert inspection.amount is None
        assert inspection.currency is None


def test_budget_inspection_rejects_cross_limit_settlement() -> None:
    reservation_limit_id = _budget_limit_id(1)
    settlement_limit_id = _budget_limit_id(2)
    inspection = session_budget_inspection(
        [
            Event(
                type=EventType.BUDGET_RESERVED,
                session_id="sess_cross_limit",
                payload={
                    "reservation_id": "reservation-cross-limit",
                    "budget_limit_id": reservation_limit_id,
                    **_model_attempt_identity(1),
                    "scope": "session",
                    "key": "sess_cross_limit",
                    "window": "all_time",
                    "currency": "USD",
                    "maximum": "1",
                    "action": "interrupt",
                    "requested": "0.50",
                },
            ),
            Event(
                type=EventType.BUDGET_RECONCILED,
                session_id="sess_cross_limit",
                payload={
                    "reservation_id": "reservation-cross-limit",
                    "budget_limit_id": settlement_limit_id,
                    **_model_attempt_identity(1),
                    "actual_amount": "0.25",
                    "pricing": {"provider_name": "fake", "model": "model"},
                },
            ),
        ]
    )

    assert inspection.cost_state == "partial"
    assert inspection.amount is None


def test_budget_inspection_sums_distinct_attempts_without_amount_heuristics() -> None:
    events: list[Event] = []
    reservations = (
        (1, 1, "0.25"),
        (1, 2, "0.25"),
        (2, 3, "0.50"),
    )
    for limit_number, attempt_number, amount in reservations:
        reservation_id = f"reservation-{limit_number}-{attempt_number}"
        budget_limit_id = _budget_limit_id(limit_number)
        identity = _model_attempt_identity(attempt_number)
        events.extend(
            [
                Event(
                    type=EventType.BUDGET_RESERVED,
                    session_id="sess_distinct_attempts",
                    payload={
                        "reservation_id": reservation_id,
                        "budget_limit_id": budget_limit_id,
                        **identity,
                        "scope": "session",
                        "key": "sess_distinct_attempts",
                        "window": "all_time",
                        "currency": "USD",
                        "maximum": str(limit_number),
                        "action": "interrupt",
                        "requested": "0.50",
                    },
                ),
                Event(
                    type=EventType.BUDGET_RECONCILED,
                    session_id="sess_distinct_attempts",
                    payload={
                        "reservation_id": reservation_id,
                        "budget_limit_id": budget_limit_id,
                        **identity,
                        "actual_amount": amount,
                        "pricing": {"provider_name": "fake", "model": "model"},
                    },
                ),
            ]
        )

    inspection = session_budget_inspection(events)

    assert inspection.cost_state == "priced"
    assert inspection.amount == "1.00"
    assert inspection.currency == "USD"


def test_budget_inspection_rejects_conflicting_costs_for_one_attempt() -> None:
    events: list[Event] = []
    identity = _model_attempt_identity(1)
    for limit_number, amount in ((1, "0.25"), (2, "0.30")):
        reservation_id = f"reservation-{limit_number}"
        budget_limit_id = _budget_limit_id(limit_number)
        events.extend(
            [
                Event(
                    type=EventType.BUDGET_RESERVED,
                    session_id="sess_conflicting_attempt_cost",
                    payload={
                        "reservation_id": reservation_id,
                        "budget_limit_id": budget_limit_id,
                        **identity,
                        "scope": "session",
                        "key": "sess_conflicting_attempt_cost",
                        "window": "all_time",
                        "currency": "USD",
                        "maximum": str(limit_number),
                        "action": "interrupt",
                        "requested": "0.50",
                    },
                ),
                Event(
                    type=EventType.BUDGET_RECONCILED,
                    session_id="sess_conflicting_attempt_cost",
                    payload={
                        "reservation_id": reservation_id,
                        "budget_limit_id": budget_limit_id,
                        **identity,
                        "actual_amount": amount,
                        "pricing": {"provider_name": "fake", "model": "model"},
                    },
                ),
            ]
        )

    inspection = session_budget_inspection(events)

    assert inspection.cost_state == "partial"
    assert inspection.amount is None


def test_budget_inspection_rejects_one_attempt_id_attached_to_two_steps() -> None:
    events: list[Event] = []
    shared_attempt_id = f"matt_{1:032x}"
    for value in (1, 2):
        reservation_id = f"reservation-{value}"
        budget_limit_id = _budget_limit_id(value)
        identity = {
            "model_step_id": f"mstep_{value:032x}",
            "model_attempt_id": shared_attempt_id,
        }
        events.extend(
            [
                Event(
                    type=EventType.BUDGET_RESERVED,
                    session_id="sess_conflicting_attempt_parent",
                    payload={
                        "reservation_id": reservation_id,
                        "budget_limit_id": budget_limit_id,
                        **identity,
                        "scope": "session",
                        "key": "sess_conflicting_attempt_parent",
                        "window": "all_time",
                        "currency": "USD",
                        "maximum": str(value),
                        "action": "interrupt",
                    },
                ),
                Event(
                    type=EventType.BUDGET_RECONCILED,
                    session_id="sess_conflicting_attempt_parent",
                    payload={
                        "reservation_id": reservation_id,
                        "budget_limit_id": budget_limit_id,
                        **identity,
                        "actual_amount": "0.25",
                        "pricing": {"provider_name": "fake", "model": "model"},
                    },
                ),
            ]
        )

    inspection = session_budget_inspection(events)

    assert inspection.cost_state == "partial"
    assert inspection.amount is None


def test_budget_inspection_requires_attempt_identity_on_reservation_failure() -> None:
    failure = Event(
        type=EventType.BUDGET_RESERVATION_FAILED,
        session_id="sess_failed_reservation",
        payload={
            "budget_limit_id": _budget_limit_id(1),
            **_model_attempt_identity(1),
            "scope": "session",
            "key": "sess_failed_reservation",
            "window": "all_time",
            "currency": "USD",
            "maximum": "1",
            "action": "interrupt",
        },
    )

    projected = project_budget_inspection_event(failure)
    assert projected.payload["model_step_id"] == failure.payload["model_step_id"]
    assert projected.payload["model_attempt_id"] == failure.payload["model_attempt_id"]
    assert session_budget_inspection([projected]).cost_state == "priced"

    projected.payload.pop("model_attempt_id")
    inspection = session_budget_inspection([projected])

    assert inspection.cost_state == "partial"
    assert inspection.amount is None


def test_budget_inspection_rejects_one_limit_id_with_two_definitions() -> None:
    budget_limit_id = _budget_limit_id(1)
    events = [
        Event(
            type=EventType.BUDGET_CHECKED,
            session_id="sess_conflicting_limit",
            payload={
                "budget_limit_id": budget_limit_id,
                "scope": "session",
                "key": "sess_conflicting_limit",
                "window": "all_time",
                "currency": "USD",
                "maximum": maximum,
                "actual": "0.25",
                "action": "interrupt",
                "unpriced_model_steps": 0,
                "cost_summary": {},
            },
        )
        for maximum in ("1", "2")
    ]

    inspection = session_budget_inspection(events)

    assert inspection.cost_state == "partial"
    assert inspection.amount is None
