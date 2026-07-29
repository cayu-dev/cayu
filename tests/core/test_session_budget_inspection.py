from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from cayu.core import Event, EventType
from cayu.runtime.budgets import (
    BudgetCheck,
    BudgetReconciliation,
    budget_check_payload,
    budget_reconciliation_payload,
    budget_settlement_id,
    project_budget_inspection_event,
    project_budget_model_attempt_inspection_event,
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


def _pricing_evidence(*, model: str = "model") -> dict[str, object]:
    return {
        "provider_name": "fake",
        "model": model,
        "match": "exact",
        "provenance": {
            "source": "application",
            "url": "application://test-price-book",
            "as_of": "2026-07-25",
        },
        "effective_from": None,
        "effective_through": None,
        "tier_max_input_tokens": None,
    }


def _reservation_event(*, value: int) -> Event:
    return Event(
        type=EventType.BUDGET_RESERVED,
        session_id="sess_settlement_states",
        payload={
            "reservation_id": f"reservation-state-{value}",
            "budget_limit_id": _budget_limit_id(value),
            **_model_attempt_identity(value),
            "scope": "session",
            "key": None,
            "window": "all_time",
            "currency": "USD",
            "maximum": "1",
            "action": "interrupt",
        },
    )


def _reconciliation(
    *,
    value: int,
    kind: Literal["completed", "conservative"],
) -> BudgetReconciliation:
    reservation_id = f"reservation-state-{value}"
    return BudgetReconciliation(
        reservation_id=reservation_id,
        settlement_id=budget_settlement_id(reservation_id),
        settlement_kind=kind,
        budget_limit_id=_budget_limit_id(value),
        **_model_attempt_identity(value),
        status="reconciled",
        reserved_amount=Decimal("0.50"),
        actual_amount=Decimal("0.25"),
        released_amount=Decimal("0.25"),
        settled_at=datetime(2026, 7, 29, tzinfo=UTC),
    )


def test_budget_inspection_distinguishes_all_settlement_states() -> None:
    pending = _reservation_event(value=1)
    completed_unsettled = _reconciliation(value=2, kind="completed")
    completed = _reconciliation(value=3, kind="completed")
    conservative = _reconciliation(value=4, kind="conservative")
    released = _reservation_event(value=5)

    budget_events = [
        pending,
        *(_reservation_event(value=value) for value in range(2, 5)),
        Event(
            type=EventType.BUDGET_RECONCILED,
            session_id="sess_settlement_states",
            payload={
                **budget_reconciliation_payload(completed),
                "pricing": _pricing_evidence(),
            },
        ),
        Event(
            type=EventType.BUDGET_RECONCILED,
            session_id="sess_settlement_states",
            payload={
                **budget_reconciliation_payload(conservative),
                "pricing": _pricing_evidence(),
            },
        ),
        released,
        Event(
            type=EventType.BUDGET_RESERVATION_RELEASED,
            session_id="sess_settlement_states",
            payload={
                "reservation_id": "reservation-state-5",
                "settlement_kind": "released",
                "budget_limit_id": _budget_limit_id(5),
                **_model_attempt_identity(5),
            },
        ),
    ]
    terminal_events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="sess_settlement_states",
            payload={
                **_model_attempt_identity(2),
                "budget_settlements": [budget_reconciliation_payload(completed_unsettled)],
            },
        ),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="sess_settlement_states",
            payload={
                **_model_attempt_identity(3),
                "budget_settlements": [budget_reconciliation_payload(completed)],
            },
        ),
        Event(
            type=EventType.MODEL_ERROR,
            session_id="sess_settlement_states",
            payload=_model_attempt_identity(4),
        ),
    ]

    inspection = session_budget_inspection(
        [project_budget_inspection_event(event) for event in budget_events],
        model_attempt_terminal_events=[
            project_budget_model_attempt_inspection_event(event) for event in terminal_events
        ],
    )

    assert inspection.pending_reservation_count == 1
    assert inspection.completed_unsettled_reservation_count == 1
    assert inspection.reconciled_reservation_count == 1
    assert inspection.conservative_reconciliation_count == 1
    assert inspection.released_reservation_count == 1
    assert inspection.cost_state == "partial"


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
            key=None,
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


def test_budget_inspection_retains_prior_spend_when_reservation_fails() -> None:
    actual = Decimal("0.25")
    summary = SessionCostSummary(
        session_id="sess_checked_then_failed",
        currency="USD",
        model_steps=1,
        priced_model_steps=1,
        unpriced_model_steps=0,
        total_cost=actual,
    )
    check = BudgetCheck(
        budget_limit_id=_budget_limit_id(1),
        scope="session",
        key=None,
        currency="USD",
        maximum=Decimal("1"),
        actual=actual,
        action="interrupt",
        model_steps=1,
        unpriced_model_steps=0,
        limit_reached=False,
        message="priced",
        cost_summary=summary,
    )
    failure = Event(
        type=EventType.BUDGET_RESERVATION_FAILED,
        session_id="sess_checked_then_failed",
        payload={
            "budget_limit_id": _budget_limit_id(1),
            **_model_attempt_identity(1),
            "scope": "session",
            "key": None,
            "window": "all_time",
            "currency": "USD",
            "maximum": "1",
            "action": "interrupt",
        },
    )

    inspection = session_budget_inspection(
        [
            project_budget_inspection_event(
                Event(
                    type=EventType.BUDGET_CHECKED,
                    session_id="sess_checked_then_failed",
                    payload=budget_check_payload(check),
                )
            ),
            project_budget_inspection_event(failure),
        ]
    )

    assert inspection.cost_state == "priced"
    assert inspection.amount == "0.25"
    assert inspection.currency == "USD"

    conflicting_actual = Decimal("0.50")
    conflicting_check = BudgetCheck(
        budget_limit_id=_budget_limit_id(2),
        scope="session",
        key=None,
        currency="USD",
        maximum=Decimal("1"),
        actual=conflicting_actual,
        action="interrupt",
        model_steps=1,
        unpriced_model_steps=0,
        limit_reached=False,
        message="priced",
        cost_summary=summary.model_copy(update={"total_cost": conflicting_actual}),
    )
    contradictory = session_budget_inspection(
        [
            project_budget_inspection_event(
                Event(
                    type=EventType.BUDGET_CHECKED,
                    session_id="sess_checked_then_failed",
                    payload=budget_check_payload(check),
                )
            ),
            project_budget_inspection_event(
                Event(
                    type=EventType.BUDGET_CHECKED,
                    session_id="sess_checked_then_failed",
                    payload=budget_check_payload(conflicting_check),
                )
            ),
            project_budget_inspection_event(failure),
        ]
    )

    assert contradictory.cost_state == "partial"
    assert contradictory.amount is None
    assert contradictory.currency is None


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
                            "key": None,
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
                            "settlement_kind": "completed",
                            "budget_limit_id": budget_limit_id,
                            **_model_attempt_identity(1),
                            "actual_amount": "0.25",
                            "pricing": _pricing_evidence(),
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
                        "key": None,
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
                        "settlement_kind": "completed",
                        "budget_limit_id": budget_limit_id,
                        **identity,
                        "actual_amount": "0.25",
                        "pricing": _pricing_evidence(),
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
                "key": None,
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
                "settlement_kind": "completed",
                "budget_limit_id": budget_limit_id,
                **_model_attempt_identity(1),
                "actual_amount": "0.25",
                "pricing": _pricing_evidence(),
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


def test_budget_inspection_projection_fails_closed_for_malformed_accounting() -> None:
    budget_limit_id = _budget_limit_id(1)
    identity = _model_attempt_identity(1)

    def reservation(
        *,
        reservation_id: str = "reservation-valid",
        attempt_identity: dict[str, str] | None = None,
    ) -> Event:
        attempt_identity = identity if attempt_identity is None else attempt_identity
        return Event(
            type=EventType.BUDGET_RESERVED,
            session_id="sess_malformed_projection",
            payload={
                "reservation_id": reservation_id,
                "settlement_kind": "completed",
                "budget_limit_id": budget_limit_id,
                **attempt_identity,
                "scope": "session",
                "key": None,
                "window": "all_time",
                "currency": "USD",
                "maximum": "1",
                "action": "interrupt",
            },
        )

    def reconciliation(
        *,
        reservation_id: str = "reservation-valid",
        actual_amount: object = "0.25",
        pricing: object = None,
    ) -> Event:
        return Event(
            type=EventType.BUDGET_RECONCILED,
            session_id="sess_malformed_projection",
            payload={
                "reservation_id": reservation_id,
                "settlement_kind": "completed",
                "budget_limit_id": budget_limit_id,
                **identity,
                "actual_amount": actual_amount,
                "pricing": pricing,
            },
        )

    malformed_cases = (
        (
            reservation(
                attempt_identity={
                    "model_step_id": identity["model_step_id"],
                    "model_attempt_id": "bad",
                }
            ),
        ),
        (
            reservation(reservation_id=" reservation-with-whitespace "),
            reconciliation(
                reservation_id=" reservation-with-whitespace ",
                pricing=_pricing_evidence(),
            ),
        ),
        (
            reservation(),
            reconciliation(
                actual_amount="not-a-number",
                pricing=_pricing_evidence(),
            ),
        ),
        (
            reservation(),
            reconciliation(
                actual_amount={"unbounded": ["caller-controlled"]},
                pricing=_pricing_evidence(),
            ),
        ),
        (
            reservation(),
            reconciliation(actual_amount="0.25", pricing="not-an-object"),
        ),
        (
            reservation(),
            reconciliation(actual_amount="0.25", pricing={}),
        ),
        (
            reservation(),
            reconciliation(actual_amount="0.25", pricing={"provider_name": 42}),
        ),
        (
            reservation(),
            reconciliation(
                actual_amount="0.25",
                pricing={**_pricing_evidence(), "match": []},
            ),
        ),
        (
            reservation(),
            reconciliation(
                actual_amount="0.25",
                pricing={
                    **_pricing_evidence(),
                    "effective_from": "2026-07-26",
                    "effective_through": "2026-07-25",
                },
            ),
        ),
    )

    for events in malformed_cases:
        inspection = session_budget_inspection(
            [project_budget_inspection_event(event) for event in events]
        )

        assert inspection.cost_state == "partial"
        assert inspection.amount is None
        assert inspection.currency is None


def test_budget_inspection_requires_exactly_one_terminal_settlement_per_reservation() -> None:
    identity = _model_attempt_identity(1)

    def reservation(
        index: int,
        *,
        attempt_identity: dict[str, str] | None = None,
    ) -> Event:
        attempt_identity = identity if attempt_identity is None else attempt_identity
        return Event(
            type=EventType.BUDGET_RESERVED,
            session_id="sess_terminal_coverage",
            payload={
                "reservation_id": f"reservation-{index}",
                "settlement_kind": "released",
                "budget_limit_id": _budget_limit_id(index),
                **attempt_identity,
                "scope": "session",
                "key": None,
                "window": "all_time",
                "currency": "USD",
                "maximum": "1",
                "action": "interrupt",
            },
        )

    def terminal(index: int, event_type: EventType) -> Event:
        payload: dict[str, object] = {
            "reservation_id": f"reservation-{index}",
            "settlement_kind": (
                "completed" if event_type == EventType.BUDGET_RECONCILED else "released"
            ),
            "budget_limit_id": _budget_limit_id(index),
            **identity,
        }
        if event_type == EventType.BUDGET_RECONCILED:
            payload.update({"actual_amount": "0.25", "pricing": None})
        return Event(
            type=event_type,
            session_id="sess_terminal_coverage",
            payload=payload,
        )

    second_identity = _model_attempt_identity(2)
    malformed_cases = (
        (reservation(1),),
        (
            reservation(1),
            terminal(1, EventType.BUDGET_RECONCILED),
            reservation(2, attempt_identity=second_identity),
        ),
        (
            reservation(1),
            terminal(1, EventType.BUDGET_RECONCILED),
            terminal(1, EventType.BUDGET_RECONCILED),
        ),
        (
            reservation(1),
            terminal(1, EventType.BUDGET_RESERVATION_RELEASED),
            terminal(1, EventType.BUDGET_RESERVATION_RELEASED),
        ),
        (
            reservation(1),
            reservation(2),
            terminal(1, EventType.BUDGET_RECONCILED),
            terminal(2, EventType.BUDGET_RESERVATION_RELEASED),
        ),
    )

    for events in malformed_cases:
        inspection = session_budget_inspection(
            [project_budget_inspection_event(event) for event in events]
        )

        assert inspection.cost_state == "partial"
        assert inspection.amount is None
        assert inspection.currency is None


def test_budget_inspection_rejects_noncanonical_failure_currencies() -> None:
    def failure(index: int, *, currency: str, model_attempt_id: str | None = None) -> Event:
        identity = _model_attempt_identity(index)
        if model_attempt_id is not None:
            identity["model_attempt_id"] = model_attempt_id
        return Event(
            type=EventType.BUDGET_RESERVATION_FAILED,
            session_id="sess_failure_currencies",
            payload={
                "budget_limit_id": _budget_limit_id(index),
                **identity,
                "scope": "session",
                "key": None,
                "window": "all_time",
                "currency": currency,
                "maximum": "1",
                "action": "interrupt",
            },
        )

    noncanonical = session_budget_inspection(
        [
            project_budget_inspection_event(event)
            for event in (failure(1, currency="USD"), failure(2, currency="usd"))
        ]
    )
    invalid = session_budget_inspection(
        [project_budget_inspection_event(failure(1, currency="USD", model_attempt_id="bad"))]
    )
    mixed = session_budget_inspection(
        [
            project_budget_inspection_event(event)
            for event in (failure(1, currency="USD"), failure(2, currency="EUR"))
        ]
    )

    assert noncanonical.cost_state == "partial"
    assert noncanonical.amount is None
    assert noncanonical.currency is None
    assert invalid.cost_state == "partial"
    assert invalid.amount is None
    assert invalid.currency is None
    assert mixed.cost_state == "mixed_currency"
    assert mixed.amount is None
    assert mixed.currency is None


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
                    "key": None,
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
                    "settlement_kind": "completed",
                    "budget_limit_id": settlement_limit_id,
                    **_model_attempt_identity(1),
                    "actual_amount": "0.25",
                    "pricing": _pricing_evidence(),
                },
            ),
        ]
    )

    assert inspection.cost_state == "partial"
    assert inspection.amount is None


def test_budget_inspection_requires_exact_model_terminal_join() -> None:
    identity = _model_attempt_identity(1)
    budget_limit_id = _budget_limit_id(1)
    budget_events = [
        Event(
            type=EventType.BUDGET_RESERVED,
            session_id="sess_model_terminal_join",
            payload={
                "reservation_id": "reservation-model-terminal-join",
                "budget_limit_id": budget_limit_id,
                **identity,
                "scope": "session",
                "key": None,
                "window": "all_time",
                "currency": "USD",
                "maximum": "1",
                "action": "interrupt",
            },
        ),
        Event(
            type=EventType.BUDGET_RECONCILED,
            session_id="sess_model_terminal_join",
            payload={
                "reservation_id": "reservation-model-terminal-join",
                "settlement_kind": "completed",
                "budget_limit_id": budget_limit_id,
                **identity,
                "actual_amount": "0.25",
                "pricing": _pricing_evidence(),
            },
        ),
    ]

    for event_type in (EventType.MODEL_COMPLETED, EventType.MODEL_ERROR):
        matching_terminal = project_budget_model_attempt_inspection_event(
            Event(
                type=event_type,
                session_id="sess_model_terminal_join",
                payload=identity,
            )
        )
        inspection = session_budget_inspection(
            budget_events,
            model_attempt_terminal_events=[matching_terminal],
        )

        assert inspection.cost_state == "priced"
        assert inspection.amount == "0.25"
        assert inspection.currency == "USD"

    for terminal_payload in (
        {
            **identity,
            "model_step_id": f"mstep_{2:032x}",
        },
        {
            "model_step_id": "malformed",
            "model_attempt_id": identity["model_attempt_id"],
        },
    ):
        terminal = project_budget_model_attempt_inspection_event(
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_model_terminal_join",
                payload=terminal_payload,
            )
        )
        inspection = session_budget_inspection(
            budget_events,
            model_attempt_terminal_events=[terminal],
        )

        assert inspection.cost_state == "partial"
        assert inspection.amount is None
        assert inspection.currency is None

    missing = session_budget_inspection(
        budget_events,
        model_attempt_terminal_events=[],
    )
    assert missing.cost_state == "partial"
    assert missing.amount is None
    assert missing.currency is None

    matching_terminal = project_budget_model_attempt_inspection_event(
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="sess_model_terminal_join",
            payload=identity,
        )
    )
    duplicate = session_budget_inspection(
        budget_events,
        model_attempt_terminal_events=[matching_terminal, matching_terminal],
    )
    assert duplicate.cost_state == "partial"
    assert duplicate.amount is None
    assert duplicate.currency is None


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
                        "key": None,
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
                        "settlement_kind": "completed",
                        "budget_limit_id": budget_limit_id,
                        **identity,
                        "actual_amount": amount,
                        "pricing": _pricing_evidence(),
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
                        "settlement_kind": "completed",
                        "budget_limit_id": budget_limit_id,
                        **identity,
                        "scope": "session",
                        "key": None,
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
                        "settlement_kind": "completed",
                        "budget_limit_id": budget_limit_id,
                        **identity,
                        "actual_amount": amount,
                        "pricing": _pricing_evidence(),
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
                        "key": None,
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
                        "settlement_kind": "completed",
                        "budget_limit_id": budget_limit_id,
                        **identity,
                        "actual_amount": "0.25",
                        "pricing": _pricing_evidence(),
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
            "key": None,
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


def test_budget_inspection_rejects_malformed_failure_alongside_priced_reservation() -> None:
    identity = _model_attempt_identity(1)
    events = [
        Event(
            type=EventType.BUDGET_RESERVED,
            session_id="sess_malformed_failure_with_reservation",
            payload={
                "reservation_id": "reservation-valid",
                "budget_limit_id": _budget_limit_id(1),
                **identity,
                "scope": "session",
                "key": None,
                "window": "all_time",
                "currency": "USD",
                "maximum": "1",
                "action": "interrupt",
            },
        ),
        Event(
            type=EventType.BUDGET_RECONCILED,
            session_id="sess_malformed_failure_with_reservation",
            payload={
                "reservation_id": "reservation-valid",
                "settlement_kind": "completed",
                "budget_limit_id": _budget_limit_id(1),
                **identity,
                "actual_amount": "0.25",
                "pricing": _pricing_evidence(),
            },
        ),
        Event(
            type=EventType.BUDGET_RESERVATION_FAILED,
            session_id="sess_malformed_failure_with_reservation",
            payload={
                "budget_limit_id": _budget_limit_id(2),
                "model_step_id": identity["model_step_id"],
                "model_attempt_id": "malformed",
                "scope": "session",
                "key": None,
                "window": "all_time",
                "currency": "USD",
                "maximum": "2",
                "action": "interrupt",
            },
        ),
    ]

    inspection = session_budget_inspection(
        [project_budget_inspection_event(event) for event in events]
    )

    assert inspection.cost_state == "partial"
    assert inspection.amount is None
    assert inspection.currency is None


def test_budget_inspection_rejects_failure_outcomes_that_contradict_an_attempt() -> None:
    identity = _model_attempt_identity(1)

    def reservation(limit_index: int) -> Event:
        return Event(
            type=EventType.BUDGET_RESERVED,
            session_id="sess_contradictory_failure",
            payload={
                "reservation_id": f"reservation-{limit_index}",
                "budget_limit_id": _budget_limit_id(limit_index),
                **identity,
                "scope": "session",
                "key": None,
                "window": "all_time",
                "currency": "USD",
                "maximum": str(limit_index),
                "action": "interrupt",
            },
        )

    def reconciliation(limit_index: int) -> Event:
        return Event(
            type=EventType.BUDGET_RECONCILED,
            session_id="sess_contradictory_failure",
            payload={
                "reservation_id": f"reservation-{limit_index}",
                "settlement_kind": "completed",
                "budget_limit_id": _budget_limit_id(limit_index),
                **identity,
                "actual_amount": "0.25",
                "pricing": _pricing_evidence(),
            },
        )

    def failure(limit_index: int) -> Event:
        return Event(
            type=EventType.BUDGET_RESERVATION_FAILED,
            session_id="sess_contradictory_failure",
            payload={
                "budget_limit_id": _budget_limit_id(limit_index),
                **identity,
                "scope": "session",
                "key": None,
                "window": "all_time",
                "currency": "USD",
                "maximum": str(limit_index),
                "action": "interrupt",
            },
        )

    contradictory_cases = (
        (reservation(1), reconciliation(1), failure(1)),
        (reservation(1), reconciliation(1), failure(2)),
        (failure(1), failure(2)),
    )

    for events in contradictory_cases:
        inspection = session_budget_inspection(
            [project_budget_inspection_event(event) for event in events]
        )

        assert inspection.cost_state == "partial"
        assert inspection.amount is None
        assert inspection.currency is None


def test_budget_inspection_accepts_released_siblings_before_reservation_failure() -> None:
    identity = _model_attempt_identity(1)
    events = [
        Event(
            type=EventType.BUDGET_RESERVED,
            session_id="sess_released_before_failure",
            payload={
                "reservation_id": "reservation-released",
                "settlement_kind": "released",
                "budget_limit_id": _budget_limit_id(1),
                **identity,
                "scope": "session",
                "key": None,
                "window": "all_time",
                "currency": "USD",
                "maximum": "1",
                "action": "interrupt",
            },
        ),
        Event(
            type=EventType.BUDGET_RESERVATION_FAILED,
            session_id="sess_released_before_failure",
            payload={
                "budget_limit_id": _budget_limit_id(2),
                **identity,
                "scope": "session",
                "key": None,
                "window": "all_time",
                "currency": "USD",
                "maximum": "2",
                "action": "interrupt",
            },
        ),
        Event(
            type=EventType.BUDGET_RESERVATION_RELEASED,
            session_id="sess_released_before_failure",
            payload={
                "reservation_id": "reservation-released",
                "settlement_kind": "released",
                "budget_limit_id": _budget_limit_id(1),
                **identity,
            },
        ),
    ]

    inspection = session_budget_inspection(
        [project_budget_inspection_event(event) for event in events]
    )

    assert inspection.cost_state == "priced"
    assert inspection.amount == "0"
    assert inspection.currency == "USD"


def test_budget_inspection_rejects_one_limit_id_with_two_definitions() -> None:
    budget_limit_id = _budget_limit_id(1)
    events = [
        Event(
            type=EventType.BUDGET_CHECKED,
            session_id="sess_conflicting_limit",
            payload={
                "budget_limit_id": budget_limit_id,
                "scope": "session",
                "key": None,
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


def test_budget_inspection_rejects_invalid_limit_descriptor_contracts() -> None:
    invalid_descriptors = (
        {"scope": "app", "key": "forbidden", "window": "all_time"},
        {"scope": "session", "key": None, "window": "not-a-budget-window"},
        {"scope": "agent", "key": None, "window": "all_time"},
    )

    for index, invalid_descriptor in enumerate(invalid_descriptors, start=1):
        budget_limit_id = _budget_limit_id(index)
        reservation_id = f"reservation-invalid-descriptor-{index}"
        identity = _model_attempt_identity(index)
        events = [
            Event(
                type=EventType.BUDGET_RESERVED,
                session_id="sess_invalid_descriptor",
                payload={
                    "reservation_id": reservation_id,
                    "budget_limit_id": budget_limit_id,
                    **identity,
                    **invalid_descriptor,
                    "currency": "USD",
                    "maximum": "1",
                    "action": "interrupt",
                },
            ),
            Event(
                type=EventType.BUDGET_RECONCILED,
                session_id="sess_invalid_descriptor",
                payload={
                    "reservation_id": reservation_id,
                    "settlement_kind": "completed",
                    "budget_limit_id": budget_limit_id,
                    **identity,
                    "actual_amount": "0.25",
                    "pricing": _pricing_evidence(),
                },
            ),
        ]

        inspection = session_budget_inspection(
            [project_budget_inspection_event(event) for event in events]
        )

        assert inspection.cost_state == "partial"
        assert inspection.amount is None
        assert inspection.currency is None


def test_budget_inspection_rejects_unrepresentable_decimal_totals() -> None:
    budget_limit_id = _budget_limit_id(1)
    reservation_id = "reservation-extreme-decimal"
    identity = _model_attempt_identity(1)
    events = [
        Event(
            type=EventType.BUDGET_RESERVED,
            session_id="sess_extreme_decimal",
            payload={
                "reservation_id": reservation_id,
                "budget_limit_id": budget_limit_id,
                **identity,
                "scope": "session",
                "key": None,
                "window": "all_time",
                "currency": "USD",
                "maximum": "1",
                "action": "interrupt",
            },
        ),
        Event(
            type=EventType.BUDGET_RECONCILED,
            session_id="sess_extreme_decimal",
            payload={
                "reservation_id": reservation_id,
                "settlement_kind": "completed",
                "budget_limit_id": budget_limit_id,
                **identity,
                "actual_amount": "1E+999999999",
                "pricing": _pricing_evidence(),
            },
        ),
    ]

    inspection = session_budget_inspection(
        [project_budget_inspection_event(event) for event in events]
    )

    assert inspection.cost_state == "partial"
    assert inspection.amount is None
    assert inspection.currency is None
