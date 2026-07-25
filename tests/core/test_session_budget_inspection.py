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


def test_budget_inspection_marks_malformed_reservation_evidence_partial() -> None:
    budget_limit_id = _budget_limit_id(1)
    valid_events = [
        Event(
            type=EventType.BUDGET_RESERVED,
            session_id="sess_malformed_evidence",
            payload={
                "reservation_id": "reservation-valid",
                "budget_limit_id": budget_limit_id,
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
                    "actual_amount": "0.25",
                    "pricing": {"provider_name": "fake", "model": "model"},
                },
            ),
        ]
    )

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
