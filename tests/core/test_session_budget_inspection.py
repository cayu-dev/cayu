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
    for identities, expected_state, expected_amount in (
        ((("1", "interrupt"), ("2", "interrupt")), "priced", "0.25"),
        ((("1", "interrupt"), ("1", "notify")), "priced", "0.25"),
        ((("1", "interrupt"), ("1", "interrupt")), "partial", None),
    ):
        events: list[Event] = []
        for index, (maximum, action) in enumerate(identities):
            reservation_id = f"reservation-{index}-{maximum}-{action}"
            events.extend(
                [
                    Event(
                        type=EventType.BUDGET_RESERVED,
                        session_id="sess_parallel_limits",
                        payload={
                            "reservation_id": reservation_id,
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
                            "actual_amount": "0.25",
                            "pricing": {"provider_name": "fake", "model": "model"},
                        },
                    ),
                ]
            )

        inspection = session_budget_inspection(events)

        assert inspection.cost_state == expected_state
        assert inspection.amount == expected_amount
        assert inspection.currency == ("USD" if expected_amount is not None else None)


def test_budget_inspection_marks_malformed_reservation_evidence_partial() -> None:
    valid_events = [
        Event(
            type=EventType.BUDGET_RESERVED,
            session_id="sess_malformed_evidence",
            payload={
                "reservation_id": "reservation-valid",
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
