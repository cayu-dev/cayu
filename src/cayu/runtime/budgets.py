from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, DecimalException, Inexact, localcontext
from hashlib import sha256
from itertools import chain
from threading import Lock
from typing import Any, Literal, NamedTuple, cast
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    MIN_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_execution_unit_id,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.core.billing import (
    UNRESOLVED_BILLING_IDENTITY,
    BillingIdentity,
    BillingIdentityState,
    PricingContext,
    ResolvedBillingIdentity,
    billing_identity_value,
    completed_billing_identity,
    copy_billing_identity,
)
from cayu.core.events import Event, EventType, copy_event
from cayu.runtime.costs import (
    CostLineItem,
    PriceBook,
    Provenance,
    SessionCostSummary,
    _normalize_provider,
    _resolve_price_book,
    _ResolvedPrice,
    estimate_session_cost,
    resolve_price_book,
)
from cayu.runtime.execution_units import (
    BudgetLimitIdentity,
    ModelAttemptIdentity,
    copy_model_attempt_identity,
)

BudgetScope = Literal["app", "agent", "causal", "session", "run"]
BudgetWindowKind = Literal["all_time", "rolling", "calendar"]
BudgetCalendarPeriod = Literal["day", "week", "month"]
BudgetAction = Literal["interrupt", "notify"]
BudgetReservationStatus = Literal["active", "reconciled", "released"]
BudgetSettlementKind = Literal["completed", "conservative", "released"]
_TOKENS_PER_MILLION = Decimal("1000000")
_MICROSECONDS_PER_SECOND = 1_000_000
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
DEFAULT_RESERVATION_TTL_SECONDS = 3600
MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY = "budget_settlements"
PUBLICATION_FALLBACK_BUDGET_REASON = (
    "model completion settlement evidence was not publishable; charged reserved amount"
)
_ALL_TIME_WINDOW = "all_time"
_ROLLING_PREFIX = "rolling:"
_ROLLING_SUFFIX = "s"
_CALENDAR_PREFIX = "calendar:"
_BUDGET_LEDGER_IDENTITY_STATE_ATTRIBUTE = "_cayu_runtime_budget_reservation_identity_state_v1"
_BUDGET_LEDGER_IDENTITY_STATE_INIT_LOCK = Lock()

_BUDGET_INSPECTION_EVENT_TYPES = frozenset(
    {
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.BUDGET_RESERVED,
        EventType.BUDGET_RECONCILED,
        EventType.BUDGET_RESERVATION_FAILED,
        EventType.BUDGET_RESERVATION_RELEASED,
    }
)
_MODEL_ATTEMPT_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.MODEL_COMPLETED,
        EventType.MODEL_ERROR,
    }
)
_BUDGET_RECONCILIATION_PRICING_KEYS = frozenset(
    {
        "provider_name",
        "model",
        "match",
        "provenance",
        "effective_from",
        "effective_through",
        "tier_max_input_tokens",
    }
)
_BUDGET_INSPECTION_PRICING_STATE_KEY = "_cayu_pricing_state"
_BUDGET_INSPECTION_COMPLETED_SETTLEMENT_IDS_KEY = (
    "_cayu_completed_budget_settlement_reservation_ids"
)


class BudgetReservationIdentityConflict(RuntimeError):
    """A reservation identity was already linked to another durable event."""


class _BudgetLedgerReservationIdentityState:
    """Process-local fallback ownership registry for one ledger instance."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.claims: dict[str, tuple[str, str]] = {}


def _budget_ledger_reservation_identity_state(
    ledger: BudgetLedger,
) -> _BudgetLedgerReservationIdentityState:
    """Return the runtime-owned fallback registry attached to ``ledger``."""

    with _BUDGET_LEDGER_IDENTITY_STATE_INIT_LOCK:
        attributes = object.__getattribute__(ledger, "__dict__")
        state = attributes.get(_BUDGET_LEDGER_IDENTITY_STATE_ATTRIBUTE)
        if state is None:
            state = _BudgetLedgerReservationIdentityState()
            attributes[_BUDGET_LEDGER_IDENTITY_STATE_ATTRIBUTE] = state
        elif type(state) is not _BudgetLedgerReservationIdentityState:
            raise RuntimeError("Budget ledger reservation identity state is invalid.")
        return state


class SessionBudgetInspection(BaseModel):
    """Availability of durable budget and price-ledger evidence."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    event_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    reservation_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    reconciliation_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    pending_reservation_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    completed_unsettled_reservation_count: StrictInt = Field(
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    reconciled_reservation_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    conservative_reconciliation_count: StrictInt = Field(
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    released_reservation_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    cost_state: Literal["unknown", "unpriced", "partial", "mixed_currency", "priced"]
    amount: str | None = None
    currency: str | None = None


def session_budget_inspection(
    events: list[Event],
    *,
    model_attempt_terminal_events: Iterable[Event] | None = None,
) -> SessionBudgetInspection:
    """Project durable budget events into one honest session cost state.

    When bounded model terminal evidence is supplied, every reconciled
    reservation must join to a completion or error carrying the same exact
    model-step and model-attempt identity. Standalone callers that only have a
    budget-event projection may omit that evidence and retain the budget-local
    validation contract.
    """
    budget_events = [event for event in events if event.type in _BUDGET_INSPECTION_EVENT_TYPES]
    terminal_step_ids_by_attempt_id: dict[str, list[str]] = {}
    invalid_terminal_attempt_ids: set[str] = set()
    terminal_events_by_identity: dict[ModelAttemptIdentity, list[Event]] = {}
    if model_attempt_terminal_events is not None:
        for event in model_attempt_terminal_events:
            if event.type not in _MODEL_ATTEMPT_TERMINAL_EVENT_TYPES:
                continue
            model_attempt_id = _inspection_execution_unit_id(
                event.payload.get("model_attempt_id"),
                "model_attempt_id",
            )
            if model_attempt_id is None:
                continue
            identity = _inspection_model_attempt_identity(event.payload)
            if identity is None:
                invalid_terminal_attempt_ids.add(model_attempt_id)
                continue
            terminal_step_ids_by_attempt_id.setdefault(model_attempt_id, []).append(
                identity.model_step_id
            )
            terminal_events_by_identity.setdefault(identity, []).append(event)
    checks = [
        event
        for event in budget_events
        if event.type in {EventType.BUDGET_CHECKED, EventType.BUDGET_LIMIT_REACHED}
    ]
    reservations = [event for event in budget_events if event.type == EventType.BUDGET_RESERVED]
    reconciliations = [
        event for event in budget_events if event.type == EventType.BUDGET_RECONCILED
    ]
    releases = [
        event for event in budget_events if event.type == EventType.BUDGET_RESERVATION_RELEASED
    ]
    failures = [
        event for event in budget_events if event.type == EventType.BUDGET_RESERVATION_FAILED
    ]
    limit_descriptors: dict[str, tuple[str, str | None, str, Decimal, str, str]] = {}
    contradictory_limit_identity = False
    for event in chain(checks, reservations, failures):
        budget_limit_id = _inspection_budget_limit_id(event.payload.get("budget_limit_id"))
        descriptor = _inspection_budget_limit_descriptor(event.payload)
        if budget_limit_id is None or descriptor is None:
            contradictory_limit_identity = True
            continue
        previous_descriptor = limit_descriptors.setdefault(budget_limit_id, descriptor)
        if previous_descriptor != descriptor:
            contradictory_limit_identity = True

    step_ids_by_attempt_id: dict[str, str] = {}
    failure_keys: set[tuple[str, ModelAttemptIdentity]] = set()
    failed_attempt_identities: set[ModelAttemptIdentity] = set()
    invalid_failure_identity = False
    for event in failures:
        budget_limit_id = _inspection_budget_limit_id(event.payload.get("budget_limit_id"))
        model_attempt_identity = _inspection_model_attempt_identity(event.payload)
        if budget_limit_id is None or model_attempt_identity is None:
            invalid_failure_identity = True
            continue
        failure_key = (budget_limit_id, model_attempt_identity)
        if failure_key in failure_keys:
            invalid_failure_identity = True
        failure_keys.add(failure_key)
        if model_attempt_identity in failed_attempt_identities:
            invalid_failure_identity = True
        failed_attempt_identities.add(model_attempt_identity)
        previous_step_id = step_ids_by_attempt_id.setdefault(
            model_attempt_identity.model_attempt_id,
            model_attempt_identity.model_step_id,
        )
        if previous_step_id != model_attempt_identity.model_step_id:
            invalid_failure_identity = True

    reservation_ids: set[str] = set()
    reservation_ids_by_attempt: dict[ModelAttemptIdentity, set[str]] = {}
    limit_ids_by_reservation: dict[str, str] = {}
    attempt_identities_by_reservation: dict[str, ModelAttemptIdentity] = {}
    reservation_ids_by_limit_attempt: dict[tuple[str, ModelAttemptIdentity], str] = {}
    currencies_by_reservation: dict[str, str] = {}
    invalid_reservation_limit_identity = False
    invalid_reservation_evidence = False
    for event in reservations:
        reservation_id = _inspection_reservation_id(event.payload.get("reservation_id"))
        budget_limit_id = _inspection_budget_limit_id(event.payload.get("budget_limit_id"))
        model_attempt_identity = _inspection_model_attempt_identity(event.payload)
        currency = event.payload.get("currency")
        if reservation_id is None:
            invalid_reservation_evidence = True
            continue
        if reservation_id in reservation_ids:
            invalid_reservation_evidence = True
        reservation_ids.add(reservation_id)
        scope = event.payload.get("scope")
        key = event.payload.get("key")
        window = event.payload.get("window")
        maximum = _inspection_decimal(event.payload.get("maximum"), positive=True)
        action = event.payload.get("action")
        if (
            budget_limit_id is not None
            and model_attempt_identity is not None
            and type(scope) is str
            and (key is None or type(key) is str)
            and type(window) is str
            and maximum is not None
            and type(action) is str
            and action in {"interrupt", "notify"}
            and type(currency) is str
            and bool(currency.strip())
        ):
            previous_limit_id = limit_ids_by_reservation.setdefault(
                reservation_id,
                budget_limit_id,
            )
            if previous_limit_id != budget_limit_id:
                invalid_reservation_evidence = True
            previous_attempt_identity = attempt_identities_by_reservation.setdefault(
                reservation_id,
                model_attempt_identity,
            )
            if previous_attempt_identity != model_attempt_identity:
                invalid_reservation_evidence = True
            previous_step_id = step_ids_by_attempt_id.setdefault(
                model_attempt_identity.model_attempt_id,
                model_attempt_identity.model_step_id,
            )
            if previous_step_id != model_attempt_identity.model_step_id:
                invalid_reservation_evidence = True
            reservation_ids_by_attempt.setdefault(
                model_attempt_identity,
                set(),
            ).add(reservation_id)
            limit_attempt_key = (
                budget_limit_id,
                model_attempt_identity,
            )
            previous_reservation_id = reservation_ids_by_limit_attempt.setdefault(
                limit_attempt_key,
                reservation_id,
            )
            if previous_reservation_id != reservation_id:
                invalid_reservation_evidence = True
        else:
            invalid_reservation_limit_identity = True
        if type(currency) is str:
            currencies_by_reservation[reservation_id] = currency.upper()

    priced_amount_by_reservation: dict[str, Decimal] = {}
    priced_reservation_ids: set[str] = set()
    reconciled_reservation_ids: set[str] = set()
    exactly_reconciled_reservation_ids: set[str] = set()
    conservatively_reconciled_reservation_ids: set[str] = set()
    terminal_count_by_reservation: dict[str, int] = {}
    for event in reconciliations:
        reservation_id = _inspection_reservation_id(event.payload.get("reservation_id"))
        budget_limit_id = _inspection_budget_limit_id(event.payload.get("budget_limit_id"))
        model_attempt_identity = _inspection_model_attempt_identity(event.payload)
        if (
            reservation_id is None
            or budget_limit_id is None
            or model_attempt_identity is None
            or limit_ids_by_reservation.get(reservation_id) != budget_limit_id
            or attempt_identities_by_reservation.get(reservation_id) != model_attempt_identity
        ):
            invalid_reservation_evidence = True
            continue
        settlement_kind = event.payload.get("settlement_kind")
        if settlement_kind == "completed":
            exactly_reconciled_reservation_ids.add(reservation_id)
        elif settlement_kind == "conservative":
            conservatively_reconciled_reservation_ids.add(reservation_id)
        else:
            invalid_reservation_evidence = True
            continue
        reconciled_reservation_ids.add(reservation_id)
        terminal_count_by_reservation[reservation_id] = (
            terminal_count_by_reservation.get(reservation_id, 0) + 1
        )
        amount = event.payload.get("actual_amount")
        pricing_state = _inspection_pricing_state(event.payload)
        parsed_amount = _inspection_decimal(amount)
        if parsed_amount is None:
            invalid_reservation_evidence = True
            continue
        if pricing_state == "unpriced":
            continue
        if pricing_state == "invalid":
            invalid_reservation_evidence = True
            continue
        currency = currencies_by_reservation.get(reservation_id)
        if currency is None:
            invalid_reservation_evidence = True
            continue
        priced_amount_by_reservation[reservation_id] = parsed_amount
        priced_reservation_ids.add(reservation_id)

    released_reservation_ids: set[str] = set()
    for event in releases:
        reservation_id = _inspection_reservation_id(event.payload.get("reservation_id"))
        budget_limit_id = _inspection_budget_limit_id(event.payload.get("budget_limit_id"))
        model_attempt_identity = _inspection_model_attempt_identity(event.payload)
        if (
            reservation_id is None
            or budget_limit_id is None
            or model_attempt_identity is None
            or event.payload.get("settlement_kind") != "released"
            or limit_ids_by_reservation.get(reservation_id) != budget_limit_id
            or attempt_identities_by_reservation.get(reservation_id) != model_attempt_identity
        ):
            invalid_reservation_evidence = True
            continue
        released_reservation_ids.add(reservation_id)
        terminal_count_by_reservation[reservation_id] = (
            terminal_count_by_reservation.get(reservation_id, 0) + 1
        )
    settled_reservation_ids = reconciled_reservation_ids | released_reservation_ids
    completed_reservation_ids: set[str] = set()
    invalid_completion_settlement_evidence = False
    for identity, terminal_events in terminal_events_by_identity.items():
        expected_attempt_reservation_ids = reservation_ids_by_attempt.get(identity, set())
        for event in terminal_events:
            raw_ids = event.payload.get(_BUDGET_INSPECTION_COMPLETED_SETTLEMENT_IDS_KEY)
            if raw_ids is None:
                continue
            if (
                type(raw_ids) is not list
                or any(type(reservation_id) is not str for reservation_id in raw_ids)
                or len(set(raw_ids)) != len(raw_ids)
                or set(raw_ids) != expected_attempt_reservation_ids
                or bool(completed_reservation_ids & set(raw_ids))
            ):
                invalid_completion_settlement_evidence = True
                continue
            completed_reservation_ids.update(raw_ids)
    complete_terminal_coverage = (
        bool(reservation_ids)
        and not invalid_reservation_limit_identity
        and not invalid_reservation_evidence
        and settled_reservation_ids == reservation_ids
        and all(
            terminal_count_by_reservation.get(reservation_id) == 1
            for reservation_id in reservation_ids
        )
    )
    contradictory_attempt_terminal_outcome = any(
        bool(attempt_reservation_ids & reconciled_reservation_ids)
        and bool(attempt_reservation_ids & released_reservation_ids)
        for attempt_reservation_ids in reservation_ids_by_attempt.values()
    )
    contradictory_failure_outcome = bool(
        failure_keys & reservation_ids_by_limit_attempt.keys()
    ) or any(
        bool(reservation_ids_by_attempt.get(attempt_identity, set()) & reconciled_reservation_ids)
        for attempt_identity in failed_attempt_identities
    )
    complete_priced_coverage = (
        complete_terminal_coverage and priced_reservation_ids == reconciled_reservation_ids
    )
    invalid_model_attempt_terminal_join = False
    if model_attempt_terminal_events is not None:
        reconciled_attempt_identities = {
            attempt_identities_by_reservation[reservation_id]
            for reservation_id in reconciled_reservation_ids
            if reservation_id in attempt_identities_by_reservation
        }
        invalid_model_attempt_terminal_join = any(
            identity.model_attempt_id in invalid_terminal_attempt_ids
            or terminal_step_ids_by_attempt_id.get(identity.model_attempt_id)
            != [identity.model_step_id]
            for identity in reconciled_attempt_identities
        )

    check_cost_state: Literal["unknown", "unpriced", "partial", "mixed_currency", "priced"]
    check_cost_state, check_amount, check_currency = (
        _inspection_latest_check_cost(checks) if not reservations else ("unknown", None, None)
    )
    cost_state: Literal["unknown", "unpriced", "partial", "mixed_currency", "priced"]
    amount: str | None = None
    currency: str | None = None
    if (
        contradictory_limit_identity
        or invalid_reservation_limit_identity
        or invalid_reservation_evidence
        or invalid_failure_identity
        or invalid_completion_settlement_evidence
        or contradictory_failure_outcome
        or invalid_model_attempt_terminal_join
        or (
            reservations
            and (not complete_terminal_coverage or contradictory_attempt_terminal_outcome)
        )
    ):
        cost_state = "partial"
    elif complete_priced_coverage:
        attempt_costs: list[tuple[Decimal, str]] = []
        mixed_currency = False
        contradictory_attempt_cost = False
        for attempt_reservation_ids in reservation_ids_by_attempt.values():
            attempt_reconciled_ids = attempt_reservation_ids & reconciled_reservation_ids
            attempt_released_ids = attempt_reservation_ids & released_reservation_ids
            currencies = {
                currencies_by_reservation[reservation_id]
                for reservation_id in attempt_reservation_ids
                if reservation_id in currencies_by_reservation
            }
            if len(currencies) != 1:
                mixed_currency = True
                continue
            attempt_currency = next(iter(currencies))
            if attempt_released_ids:
                attempt_costs.append((Decimal("0"), attempt_currency))
                continue
            priced_amounts = {
                priced_amount_by_reservation[reservation_id]
                for reservation_id in attempt_reconciled_ids
            }
            if len(priced_amounts) != 1:
                contradictory_attempt_cost = True
                continue
            attempt_costs.append((next(iter(priced_amounts)), attempt_currency))
        if contradictory_attempt_cost:
            cost_state = "partial"
        elif mixed_currency or len({item_currency for _, item_currency in attempt_costs}) > 1:
            cost_state = "mixed_currency"
        elif attempt_costs:
            total = _inspection_decimal_sum(item_amount for item_amount, _ in attempt_costs)
            if total is None:
                cost_state = "partial"
            else:
                cost_state = "priced"
                amount = str(total)
                currency = attempt_costs[0][1]
        else:
            cost_state = "partial"
    elif not reservations and checks and not (reconciliations or releases or failures):
        cost_state = check_cost_state
        amount = check_amount
        currency = check_currency
    elif not reservation_ids and failures:
        failure_currencies = {
            descriptor[5]
            for event in failures
            if (descriptor := _inspection_budget_limit_descriptor(event.payload)) is not None
        }
        if checks:
            if check_cost_state == "partial":
                cost_state = "partial"
            elif check_cost_state == "mixed_currency":
                cost_state = "mixed_currency"
            elif (
                check_cost_state == "priced"
                and check_currency is not None
                and failure_currencies == {check_currency}
            ):
                cost_state = "priced"
                amount = check_amount
                currency = check_currency
            elif (
                check_cost_state == "priced"
                and check_currency is not None
                and len(failure_currencies | {check_currency}) > 1
            ):
                cost_state = "mixed_currency"
            else:
                cost_state = check_cost_state
                amount = check_amount
                currency = check_currency
        elif len(failure_currencies) == 1:
            cost_state = "priced"
            amount = "0"
            currency = next(iter(failure_currencies))
        elif len(failure_currencies) > 1:
            cost_state = "mixed_currency"
        else:
            cost_state = "partial"
    elif priced_amount_by_reservation:
        cost_state = "partial"
    elif budget_events:
        cost_state = "unpriced"
    else:
        # No ledger evidence is unknown even when observed token counters are zero.
        # A zero-priced reconciliation is represented above as amount="0".
        cost_state = "unknown"

    return SessionBudgetInspection(
        event_count=len(budget_events),
        reservation_count=len(reservations),
        reconciliation_count=len(reconciliations),
        pending_reservation_count=len(
            reservation_ids - settled_reservation_ids - completed_reservation_ids
        ),
        completed_unsettled_reservation_count=len(
            completed_reservation_ids - settled_reservation_ids
        ),
        reconciled_reservation_count=len(exactly_reconciled_reservation_ids),
        conservative_reconciliation_count=len(conservatively_reconciled_reservation_ids),
        released_reservation_count=len(released_reservation_ids),
        cost_state=cost_state,
        amount=amount,
        currency=currency,
    )


def _inspection_latest_check_cost(
    checks: list[Event],
) -> tuple[
    Literal["unknown", "unpriced", "partial", "mixed_currency", "priced"],
    str | None,
    str | None,
]:
    """Return the agreed latest cumulative spend reported by budget checks."""

    if not checks:
        return "unknown", None, None
    latest_checks: dict[str, tuple[Decimal, str, int]] = {}
    invalid_check = False
    for event in checks:
        budget_limit_id = _inspection_budget_limit_id(event.payload.get("budget_limit_id"))
        descriptor = _inspection_budget_limit_descriptor(event.payload)
        actual = _inspection_decimal(event.payload.get("actual"))
        unpriced_steps = event.payload.get("unpriced_model_steps")
        cost_summary = event.payload.get("cost_summary")
        if (
            budget_limit_id is None
            or descriptor is None
            or actual is None
            or type(unpriced_steps) is not int
            or unpriced_steps < 0
            or type(cost_summary) is not dict
        ):
            invalid_check = True
            continue
        latest_checks[budget_limit_id] = (
            actual,
            descriptor[5],
            unpriced_steps,
        )

    if invalid_check:
        return "partial", None, None
    if any(unpriced_steps > 0 for _, _, unpriced_steps in latest_checks.values()):
        state = (
            "unpriced"
            if all(unpriced_steps > 0 for _, _, unpriced_steps in latest_checks.values())
            else "partial"
        )
        return state, None, None
    check_totals = {
        (actual, checked_currency) for actual, checked_currency, _ in latest_checks.values()
    }
    if len({checked_currency for _, checked_currency in check_totals}) > 1:
        return "mixed_currency", None, None
    if len(check_totals) != 1:
        return "partial", None, None
    checked_amount, checked_currency = next(iter(check_totals))
    return "priced", str(checked_amount), checked_currency


def _inspection_decimal(value: Any, *, positive: bool = False) -> Decimal | None:
    if type(value) is not str:
        return None
    try:
        parsed = Decimal(value)
    except ArithmeticError:
        return None
    if not parsed.is_finite() or parsed < 0 or (positive and parsed == 0):
        return None
    return parsed


def _inspection_decimal_sum(values: Iterable[Decimal]) -> Decimal | None:
    """Add inspected amounts without rounding or leaking Decimal failures."""

    try:
        with localcontext() as context:
            # Inspection must never manufacture a confident rounded cost. The
            # default context already traps invalid operations and overflow;
            # additionally reject an otherwise silent precision loss.
            context.traps[Inexact] = True
            total = Decimal("0")
            for value in values:
                total += value
    except DecimalException:
        return None
    return total if total.is_finite() and total >= 0 else None


def _inspection_budget_limit_id(value: Any) -> str | None:
    if type(value) is not str:
        return None
    try:
        return BudgetLimitIdentity(budget_limit_id=value).budget_limit_id
    except ValueError:
        return None


def _inspection_reservation_id(value: Any) -> str | None:
    if type(value) is not str:
        return None
    try:
        return require_clean_nonblank(value, "reservation_id")
    except ValueError:
        return None


def is_complete_budget_reconciliation_pricing(value: Any) -> bool:
    """Whether a reconciliation carries complete runtime-owned pricing evidence."""

    if (
        type(value) is not dict
        or len(value) != len(_BUDGET_RECONCILIATION_PRICING_KEYS)
        or value.keys() != _BUDGET_RECONCILIATION_PRICING_KEYS
    ):
        return False
    try:
        require_clean_nonblank(value.get("provider_name"), "provider_name")
        require_clean_nonblank(value.get("model"), "model")
    except ValueError:
        return False
    match = value.get("match")
    if type(match) is not str or match not in {"exact", "prefix", "resource_mapping"}:
        return False

    provenance = value.get("provenance")
    if type(provenance) is not dict:
        return False
    try:
        Provenance.model_validate(provenance)
    except (TypeError, ValueError):
        return False

    parsed_dates: list[date | None] = []
    for field_name in ("effective_from", "effective_through"):
        raw_date = value.get(field_name)
        if raw_date is None:
            parsed_dates.append(None)
            continue
        if type(raw_date) is not str:
            return False
        try:
            parsed_date = date.fromisoformat(raw_date)
        except ValueError:
            return False
        if parsed_date.isoformat() != raw_date:
            return False
        parsed_dates.append(parsed_date)
    effective_from, effective_through = parsed_dates
    if (
        effective_from is not None
        and effective_through is not None
        and effective_from > effective_through
    ):
        return False

    tier_max_input_tokens = value.get("tier_max_input_tokens")
    return tier_max_input_tokens is None or (
        type(tier_max_input_tokens) is int and 0 < tier_max_input_tokens <= MAX_DURABLE_JSON_INTEGER
    )


def _inspection_pricing_state(
    payload: Mapping[str, Any],
) -> Literal["unpriced", "priced", "invalid"]:
    """Classify raw pricing evidence or the bounded projector's trusted state."""

    projected_state = payload.get(_BUDGET_INSPECTION_PRICING_STATE_KEY)
    if projected_state is not None:
        if (
            "pricing" in payload
            or type(projected_state) is not str
            or projected_state not in {"priced", "invalid"}
        ):
            return "invalid"
        return cast("Literal['priced', 'invalid']", projected_state)

    pricing = payload.get("pricing")
    if pricing is None:
        return "unpriced"
    return "priced" if is_complete_budget_reconciliation_pricing(pricing) else "invalid"


def _inspection_model_attempt_identity(
    payload: Mapping[str, Any],
) -> ModelAttemptIdentity | None:
    try:
        return ModelAttemptIdentity.model_validate(
            {
                "model_step_id": payload.get("model_step_id"),
                "model_attempt_id": payload.get("model_attempt_id"),
            }
        )
    except (TypeError, ValueError):
        return None


def _inspection_execution_unit_id(value: Any, field_name: str) -> str | None:
    try:
        return require_execution_unit_id(value, field_name)
    except (TypeError, ValueError):
        return None


def _inspection_budget_limit_descriptor(
    payload: Mapping[str, Any],
) -> tuple[str, str | None, str, Decimal, str, str] | None:
    scope = payload.get("scope")
    key = payload.get("key")
    window = payload.get("window")
    maximum = _inspection_decimal(payload.get("maximum"), positive=True)
    action = payload.get("action")
    currency = payload.get("currency")
    if (
        type(scope) is not str
        or scope not in {"app", "agent", "causal", "session", "run"}
        or (key is not None and (type(key) is not str or not key or key.strip() != key))
        or type(window) is not str
        or not window
        or window.strip() != window
        or maximum is None
        or type(action) is not str
        or action not in {"interrupt", "notify"}
        or type(currency) is not str
        or not currency
        or currency.strip() != currency
        or currency != currency.upper()
    ):
        return None
    if scope in {"app", "session", "run"} and key is not None:
        return None
    if scope in {"agent", "causal"} and key is None:
        return None
    try:
        parsed_window = _budget_window_from_string(window)
    except (TypeError, ValueError):
        return None
    if parsed_window.storage_key != window:
        return None
    return scope, key, window, maximum, action, currency


def is_budget_inspection_event(event: Event) -> bool:
    """Whether an event contributes to the session budget inspection projection."""
    return event.type in _BUDGET_INSPECTION_EVENT_TYPES


def project_budget_model_attempt_inspection_event(event: Event) -> Event:
    """Retain only bounded model-attempt identity used by budget inspection."""

    if event.type not in _MODEL_ATTEMPT_TERMINAL_EVENT_TYPES:
        raise ValueError("Budget model-attempt inspection requires a model terminal event.")
    payload: dict[str, Any] = {}
    for field_name in ("model_step_id", "model_attempt_id"):
        value = _inspection_execution_unit_id(event.payload.get(field_name), field_name)
        if value is not None:
            payload[field_name] = value
    if (
        event.type == EventType.MODEL_COMPLETED
        and MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY in event.payload
    ):
        try:
            settlements = tuple(
                budget_reconciliation_from_payload(raw)
                for raw in event.payload[MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY]
            )
            identity = _inspection_model_attempt_identity(payload)
            if (
                type(event.payload[MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY]) is not list
                or identity is None
                or len({item.reservation_id for item in settlements}) != len(settlements)
                or any(
                    item.status != "reconciled"
                    or item.settlement_kind == "released"
                    or item.model_step_id != identity.model_step_id
                    or item.model_attempt_id != identity.model_attempt_id
                    for item in settlements
                )
            ):
                raise ValueError("Invalid model completion budget settlements.")
            payload[_BUDGET_INSPECTION_COMPLETED_SETTLEMENT_IDS_KEY] = [
                item.reservation_id for item in settlements
            ]
        except (TypeError, ValueError):
            payload[_BUDGET_INSPECTION_COMPLETED_SETTLEMENT_IDS_KEY] = False
    return event.model_copy(update={"payload": payload})


def project_budget_inspection_event(event: Event) -> Event:
    """Retain only the durable fields consumed by budget inspection."""
    retained_keys: tuple[str, ...]
    if event.type in {EventType.BUDGET_CHECKED, EventType.BUDGET_LIMIT_REACHED}:
        retained_keys = (
            "budget_limit_id",
            "scope",
            "key",
            "window",
            "currency",
            "maximum",
            "actual",
            "action",
            "unpriced_model_steps",
        )
    elif event.type == EventType.BUDGET_RESERVED:
        retained_keys = (
            "reservation_id",
            "budget_limit_id",
            "model_step_id",
            "model_attempt_id",
            "scope",
            "key",
            "window",
            "currency",
            "maximum",
            "action",
        )
    elif event.type == EventType.BUDGET_RECONCILED:
        retained_keys = (
            "reservation_id",
            "settlement_kind",
            "budget_limit_id",
            "model_step_id",
            "model_attempt_id",
            "actual_amount",
        )
    elif event.type == EventType.BUDGET_RESERVATION_RELEASED:
        retained_keys = (
            "reservation_id",
            "settlement_kind",
            "budget_limit_id",
            "model_step_id",
            "model_attempt_id",
        )
    elif event.type == EventType.BUDGET_RESERVATION_FAILED:
        retained_keys = (
            "budget_limit_id",
            "model_step_id",
            "model_attempt_id",
            "scope",
            "key",
            "window",
            "currency",
            "maximum",
            "action",
        )
    else:
        retained_keys = ()

    payload = {key: event.payload[key] for key in retained_keys if key in event.payload}
    if (
        event.type
        in {
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_LIMIT_REACHED,
        }
        and type(event.payload.get("cost_summary")) is dict
    ):
        payload["cost_summary"] = {}
    if event.type == EventType.BUDGET_RECONCILED:
        actual_amount = event.payload.get("actual_amount")
        if type(actual_amount) is not str:
            # Retain a bounded invalid-shape marker instead of caller-controlled data.
            payload["actual_amount"] = False
        pricing = event.payload.get("pricing")
        if is_complete_budget_reconciliation_pricing(pricing):
            payload[_BUDGET_INSPECTION_PRICING_STATE_KEY] = "priced"
        elif pricing is not None:
            # Preserve only the fact that malformed pricing evidence was present.
            payload[_BUDGET_INSPECTION_PRICING_STATE_KEY] = "invalid"
    return event.model_copy(update={"payload": payload})


class BudgetWindow(BaseModel):
    """Time window used when selecting events for budget accounting."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: BudgetWindowKind = "all_time"
    duration_seconds: StrictInt | None = Field(default=None, ge=1, le=MAX_DURABLE_JSON_INTEGER)
    period: BudgetCalendarPeriod | None = None
    timezone: str | None = None

    @classmethod
    def all_time(cls) -> BudgetWindow:
        return cls(kind="all_time")

    @classmethod
    def rolling(cls, *, seconds: int) -> BudgetWindow:
        return cls(kind="rolling", duration_seconds=seconds)

    @classmethod
    def calendar(
        cls,
        *,
        period: BudgetCalendarPeriod,
        timezone: str = "UTC",
    ) -> BudgetWindow:
        return cls(kind="calendar", period=period, timezone=timezone)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        timezone = require_clean_nonblank(value, info.field_name)
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown budget window timezone: {timezone}") from exc
        return timezone

    @model_validator(mode="after")
    def validate_window_fields(self) -> BudgetWindow:
        if self.kind == "all_time":
            if (
                self.duration_seconds is not None
                or self.period is not None
                or self.timezone is not None
            ):
                raise ValueError("All-time budget windows must not set window details.")
        elif self.kind == "rolling":
            if self.duration_seconds is None:
                raise ValueError("Rolling budget windows require duration_seconds.")
            if self.period is not None or self.timezone is not None:
                raise ValueError("Rolling budget windows must not set calendar details.")
        elif self.kind == "calendar":
            if self.duration_seconds is not None:
                raise ValueError("Calendar budget windows must not set duration_seconds.")
            if self.period is None:
                raise ValueError("Calendar budget windows require period.")
            if self.timezone is None:
                raise ValueError("Calendar budget windows require timezone.")
        return self

    @property
    def storage_key(self) -> str:
        if self.kind == "all_time":
            return "all_time"
        if self.kind == "rolling":
            return f"rolling:{self.duration_seconds}s"
        return f"calendar:{self.period}:{self.timezone}"

    def since(self, now: datetime | None = None) -> datetime | None:
        return self.bounds(now=now)[0]

    def until(self, now: datetime | None = None) -> datetime | None:
        return self.bounds(now=now)[1]

    def bounds(self, now: datetime | None = None) -> tuple[datetime | None, datetime | None]:
        reference = datetime.now(UTC) if now is None else _utc_datetime(now, "now")
        if self.kind == "all_time":
            return None, None
        if self.kind == "rolling":
            return reference - timedelta(seconds=self.duration_seconds or 0), reference
        return _calendar_window_bounds(
            reference,
            period=self.period or "day",
            timezone=self.timezone or "UTC",
        )


class BudgetReservation(BaseModel):
    """Conservative per-model-step reservation configured by the app."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    max_input_tokens: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    max_output_tokens: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    max_cache_read_input_tokens: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    max_cache_write_input_tokens: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)

    @model_validator(mode="after")
    def validate_nonzero_reservation(self) -> BudgetReservation:
        if (
            self.max_input_tokens
            + self.max_output_tokens
            + self.max_cache_read_input_tokens
            + self.max_cache_write_input_tokens
            <= 0
        ):
            raise ValueError("Budget reservation must reserve at least one token.")
        return self


class BudgetLimit(BaseModel):
    """Estimated-cost budget that applies across one durable runtime scope."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    scope: BudgetScope = "session"
    max_estimated_cost: Decimal = Field(gt=0)
    pricing: PriceBook
    currency: str = "USD"
    window: BudgetWindow = Field(default_factory=BudgetWindow.all_time)
    key: str | None = None
    allow_unpriced: StrictBool = False
    reservation: BudgetReservation | None = None
    action: BudgetAction = "interrupt"

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name).upper()

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("window", mode="before")
    @classmethod
    def copy_window(cls, value) -> BudgetWindow:
        return copy_budget_window(value)

    @field_validator("max_estimated_cost")
    @classmethod
    def validate_cost(cls, value: Decimal, info) -> Decimal:
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value

    @model_validator(mode="after")
    def validate_scope_key(self) -> BudgetLimit:
        if self.scope in ("app", "session", "run") and self.key is not None:
            raise ValueError(f"{self.scope.title()} budget limits must not set key.")
        if self.scope in ("agent", "causal") and self.key is None:
            raise ValueError(f"{self.scope.title()} budget limits require key.")
        if self.reservation is not None and self.allow_unpriced:
            raise ValueError("Budget reservations require priced model usage.")
        if self.reservation is not None and self.action != "interrupt":
            raise ValueError("Budget reservations require action='interrupt'.")
        return self

    @field_validator("reservation")
    @classmethod
    def copy_reservation(cls, value: BudgetReservation | None) -> BudgetReservation | None:
        if value is None:
            return None
        if type(value) is not BudgetReservation:
            raise TypeError("reservation must be a BudgetReservation.")
        return copy_budget_reservation(value)


class _EffectiveBudgetLimit(BudgetLimit):
    """A configured limit after the runtime assigns its durable ledger identity."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    budget_limit_id: str
    identity_namespace: str = Field(exclude=True, repr=False)
    identity_definition_sha256: str = Field(exclude=True, repr=False)
    identity_occurrences: StrictInt = Field(ge=1, exclude=True, repr=False)
    identity_occurrence: StrictInt = Field(ge=0, exclude=True, repr=False)

    @field_validator("budget_limit_id")
    @classmethod
    def validate_budget_limit_id(cls, value: str) -> str:
        return BudgetLimitIdentity(budget_limit_id=value).budget_limit_id

    @field_validator("identity_namespace")
    @classmethod
    def validate_identity_namespace(cls, value: str) -> str:
        return require_clean_nonblank(value, "identity_namespace")

    @field_validator("identity_definition_sha256")
    @classmethod
    def validate_identity_definition_sha256(cls, value: str) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("identity_definition_sha256 must be a lowercase SHA-256 digest.")
        return value

    @model_validator(mode="after")
    def validate_runtime_identity(self) -> _EffectiveBudgetLimit:
        if self.identity_occurrence >= self.identity_occurrences:
            raise ValueError("Budget limit identity occurrence is outside its duplicate set.")
        definition_digest = _budget_limit_definition_digest(self)
        if definition_digest != self.identity_definition_sha256:
            raise ValueError("Budget limit identity does not match its configured definition.")
        expected_id = _budget_limit_id_from_material(
            namespace=self.identity_namespace,
            definition_digest=definition_digest,
            definition_occurrences=self.identity_occurrences,
            occurrence=self.identity_occurrence,
        )
        if self.budget_limit_id != expected_id:
            raise ValueError("Budget limit identity does not match its runtime-owned material.")
        return self


class BudgetPolicy(BaseModel):
    """App-level budget policy applied automatically by the runtime."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)

    @field_validator("limits", mode="before")
    @classmethod
    def copy_limits(
        cls, value: Iterable[BudgetLimit | Mapping[str, Any]] | None
    ) -> tuple[BudgetLimit, ...]:
        if value is None:
            return ()
        if isinstance(value, str | bytes):
            raise ValueError("Budget policy limits must be an iterable of BudgetLimit values.")
        return tuple(_coerce_budget_limit(limit) for limit in value)

    @model_validator(mode="after")
    def validate_policy_scopes(self) -> BudgetPolicy:
        for limit in self.limits:
            if limit.scope in {"session", "run"}:
                raise ValueError(
                    f"{limit.scope.title()} budget limits are request-scoped, "
                    "not app policy limits."
                )
        return self


class BudgetCheck(BaseModel):
    """Result of evaluating one budget limit."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    budget_limit_id: str
    scope: BudgetScope
    key: str | None = None
    window: BudgetWindow = Field(default_factory=BudgetWindow.all_time)
    currency: str
    maximum: Decimal = Field(gt=0)
    actual: Decimal = Field(ge=0)
    action: BudgetAction = "interrupt"
    model_steps: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    unpriced_model_steps: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    limit_reached: StrictBool
    message: str
    cost_summary: SessionCostSummary

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("budget_limit_id")
    @classmethod
    def validate_budget_limit_id(cls, value: str) -> str:
        return BudgetLimitIdentity(budget_limit_id=value).budget_limit_id

    @field_validator("window", mode="before")
    @classmethod
    def copy_window(cls, value) -> BudgetWindow:
        return copy_budget_window(value)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name).upper()

    @field_validator("maximum", "actual")
    @classmethod
    def validate_decimal(cls, value: Decimal, info) -> Decimal:
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value


class BudgetSettlementFallback(BaseModel):
    """Pre-dispatch authority for a publishable conservative terminal outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    settled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reconciliation_reason: str = PUBLICATION_FALLBACK_BUDGET_REASON
    release_reason: str = "reservation released before provider dispatch"
    expiration_reason: str | None = None

    @field_validator("settled_at")
    @classmethod
    def validate_settled_at(cls, value: datetime) -> datetime:
        return _utc_datetime(value, "settled_at")

    @field_validator("reconciliation_reason", "release_reason")
    @classmethod
    def validate_required_reason(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("expiration_reason")
    @classmethod
    def validate_expiration_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "expiration_reason")


def copy_budget_settlement_fallback(
    fallback: BudgetSettlementFallback,
) -> BudgetSettlementFallback:
    if type(fallback) is not BudgetSettlementFallback:
        raise TypeError("settlement_fallback must be a BudgetSettlementFallback.")
    return BudgetSettlementFallback.model_validate(fallback.model_dump(mode="python"))


class BudgetReservationRecord(BaseModel):
    """One reserved budget amount for a model step."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    reservation_id: str = Field(default_factory=lambda: f"bres_{uuid4().hex}")
    budget_limit_id: str
    model_step_id: str
    model_attempt_id: str
    scope: BudgetScope
    key: str | None = None
    window: BudgetWindow = Field(default_factory=BudgetWindow.all_time)
    currency: str
    session_id: str
    agent_name: str
    environment_name: str | None = None
    provider_name: str
    model: str
    billing_identity: BillingIdentity | None = None
    settlement_event_payload: dict[str, Any] = Field(default_factory=dict)
    settlement_fallback: BudgetSettlementFallback = Field(default_factory=BudgetSettlementFallback)
    dispatch_id: str | None = None
    dispatched_at: datetime | None = None
    reserved_amount: Decimal = Field(ge=0)
    actual_amount: Decimal | None = Field(default=None, ge=0)
    status: BudgetReservationStatus = "active"
    reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator(
        "reservation_id",
        "budget_limit_id",
        "currency",
        "session_id",
        "agent_name",
        "provider_name",
        "model",
    )
    @classmethod
    def validate_nonblank_strings(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if info.field_name == "budget_limit_id":
            return BudgetLimitIdentity(budget_limit_id=value).budget_limit_id
        return value

    @field_validator("key", "reason", "environment_name", "dispatch_id")
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("settlement_event_payload", mode="before")
    @classmethod
    def copy_settlement_event_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "settlement_event_payload")

    @field_validator("settlement_fallback", mode="before")
    @classmethod
    def copy_settlement_fallback(
        cls,
        value: BudgetSettlementFallback | Mapping[str, Any],
    ) -> BudgetSettlementFallback:
        if type(value) is BudgetSettlementFallback:
            return copy_budget_settlement_fallback(value)
        return BudgetSettlementFallback.model_validate(value)

    @field_validator("window", mode="before")
    @classmethod
    def copy_window(cls, value) -> BudgetWindow:
        return copy_budget_window(value)

    @field_validator("currency")
    @classmethod
    def validate_record_currency(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name).upper()

    @field_validator("reserved_amount", "actual_amount")
    @classmethod
    def validate_record_decimal(cls, value: Decimal | None, info) -> Decimal | None:
        if value is None:
            return None
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value

    @field_validator("created_at", "updated_at", "dispatched_at")
    @classmethod
    def validate_record_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return _utc_datetime(value, info.field_name)

    @model_validator(mode="after")
    def validate_model_attempt_identity(self) -> BudgetReservationRecord:
        ModelAttemptIdentity(
            model_step_id=self.model_step_id,
            model_attempt_id=self.model_attempt_id,
        )
        if (self.dispatch_id is None) != (self.dispatched_at is None):
            raise ValueError("dispatch_id and dispatched_at must be set together.")
        return self


class BudgetReservationResult(BaseModel):
    """Result of attempting to reserve budget before a model step."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    accepted: StrictBool
    budget_limit_id: str
    model_step_id: str
    model_attempt_id: str
    scope: BudgetScope
    key: str | None = None
    window: BudgetWindow = Field(default_factory=BudgetWindow.all_time)
    currency: str
    maximum: Decimal = Field(gt=0)
    action: BudgetAction
    requested: Decimal = Field(ge=0)
    actual: Decimal = Field(ge=0)
    message: str
    record: BudgetReservationRecord | None = None

    @field_validator("key")
    @classmethod
    def validate_result_key(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("budget_limit_id")
    @classmethod
    def validate_budget_limit_id(cls, value: str) -> str:
        return BudgetLimitIdentity(budget_limit_id=value).budget_limit_id

    @field_validator("window", mode="before")
    @classmethod
    def copy_window(cls, value) -> BudgetWindow:
        return copy_budget_window(value)

    @field_validator("currency", "message")
    @classmethod
    def validate_result_strings(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        return value.upper() if info.field_name == "currency" else value

    @field_validator("maximum", "requested", "actual")
    @classmethod
    def validate_result_decimal(cls, value: Decimal, info) -> Decimal:
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value

    @model_validator(mode="after")
    def validate_model_attempt_identity(self) -> BudgetReservationResult:
        identity = ModelAttemptIdentity(
            model_step_id=self.model_step_id,
            model_attempt_id=self.model_attempt_id,
        )
        if self.record is not None and (
            self.record.budget_limit_id != self.budget_limit_id
            or self.record.model_step_id != identity.model_step_id
            or self.record.model_attempt_id != identity.model_attempt_id
        ):
            raise ValueError(
                "Budget reservation result identity does not match its reservation record."
            )
        return self


class BudgetReconciliationPricing(BaseModel):
    """Bounded pricing provenance committed with one budget settlement."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    provider_name: str
    model: str
    match: Literal["exact", "prefix", "resource_mapping"]
    provenance: Provenance
    effective_from: date | None = None
    effective_through: date | None = None
    tier_max_input_tokens: StrictInt | None = Field(default=None, gt=0, le=MAX_DURABLE_JSON_INTEGER)
    billing_identity: BillingIdentity | None = Field(default=None, exclude=True)

    @field_validator("provider_name", "model")
    @classmethod
    def validate_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_effective_window(self) -> BudgetReconciliationPricing:
        if (
            self.effective_from is not None
            and self.effective_through is not None
            and self.effective_from > self.effective_through
        ):
            raise ValueError("reconciliation pricing effective window is reversed")
        return self


class BudgetReconciliation(BaseModel):
    """Result of reconciling a reservation after the model step completes."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    reservation_id: str
    settlement_id: str
    settlement_kind: BudgetSettlementKind
    budget_limit_id: str
    model_step_id: str
    model_attempt_id: str
    status: BudgetReservationStatus
    reserved_amount: Decimal = Field(ge=0)
    actual_amount: Decimal | None = Field(default=None, ge=0)
    released_amount: Decimal = Field(ge=0)
    reason: str | None = None
    settled_at: datetime
    billing_identity: BillingIdentity | None = None
    pricing_provider_name: str | None = None
    pricing_model: str | None = None
    pricing_match: Literal["exact", "prefix", "resource_mapping"] | None = None
    pricing_provenance: Provenance | None = None
    pricing_effective_from: date | None = None
    pricing_effective_through: date | None = None
    pricing_tier_max_input_tokens: StrictInt | None = Field(
        default=None, gt=0, le=MAX_DURABLE_JSON_INTEGER
    )

    @field_validator("reservation_id", "settlement_id")
    @classmethod
    def validate_reconciliation_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("budget_limit_id")
    @classmethod
    def validate_budget_limit_id(cls, value: str) -> str:
        return BudgetLimitIdentity(budget_limit_id=value).budget_limit_id

    @field_validator("reason", "pricing_provider_name", "pricing_model")
    @classmethod
    def validate_reconciliation_reason(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_pricing_evidence(self) -> BudgetReconciliation:
        ModelAttemptIdentity(
            model_step_id=self.model_step_id,
            model_attempt_id=self.model_attempt_id,
        )
        if self.settlement_id != budget_settlement_id(self.reservation_id):
            raise ValueError("reconciliation settlement identity is not canonical")
        if self.status == "released":
            if (
                self.settlement_kind != "released"
                or self.actual_amount is not None
                or self.released_amount != self.reserved_amount
            ):
                raise ValueError("released reconciliation has contradictory accounting")
        elif self.status == "reconciled":
            expected_released = (
                Decimal("0")
                if self.actual_amount is None or self.actual_amount >= self.reserved_amount
                else self.reserved_amount - self.actual_amount
            )
            if (
                self.settlement_kind == "released"
                or self.actual_amount is None
                or self.released_amount != expected_released
            ):
                raise ValueError("charged reconciliation has contradictory accounting")
        else:
            raise ValueError("budget reconciliation must be terminal")
        identity = (
            self.pricing_provider_name,
            self.pricing_model,
            self.pricing_match,
            self.pricing_provenance,
        )
        evidence = (
            *identity,
            self.pricing_effective_from,
            self.pricing_effective_through,
            self.pricing_tier_max_input_tokens,
        )
        if any(value is not None for value in evidence) and not all(
            value is not None for value in identity
        ):
            raise ValueError("reconciliation pricing identity and provenance must be complete")
        if (
            self.pricing_effective_from is not None
            and self.pricing_effective_through is not None
            and self.pricing_effective_from > self.pricing_effective_through
        ):
            raise ValueError("reconciliation pricing effective window is reversed")
        return self

    @field_validator("settled_at")
    @classmethod
    def validate_settled_at(cls, value: datetime) -> datetime:
        return _utc_datetime(value, "settled_at")

    @field_validator("reserved_amount", "actual_amount", "released_amount")
    @classmethod
    def validate_reconciliation_decimal(cls, value: Decimal | None, info) -> Decimal | None:
        if value is None:
            return None
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value


class BudgetSettlementRecord(BaseModel):
    """One ledger-committed terminal transition and its durable audit event."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    settlement_id: str
    reservation_id: str
    settlement_kind: BudgetSettlementKind
    session_id: str
    agent_name: str
    environment_name: str | None = None
    reconciliation: BudgetReconciliation
    event: Event
    event_published: StrictBool = False

    @field_validator("settlement_id", "reservation_id", "session_id", "agent_name")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("environment_name")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_event_binding(self) -> BudgetSettlementRecord:
        reconciliation = self.reconciliation
        expected_type = (
            EventType.BUDGET_RESERVATION_RELEASED
            if self.settlement_kind == "released"
            else EventType.BUDGET_RECONCILED
        )
        if (
            self.settlement_id != reconciliation.settlement_id
            or self.reservation_id != reconciliation.reservation_id
            or self.settlement_kind != reconciliation.settlement_kind
            or self.event.id != budget_settlement_event_id(self.settlement_id)
            or self.event.type != expected_type
            or self.event.session_id != self.session_id
            or self.event.agent_name != self.agent_name
            or self.event.environment_name != self.environment_name
            or self.event.timestamp != reconciliation.settled_at
        ):
            raise ValueError("Budget settlement audit event has conflicting identity.")
        expected_payload = budget_reconciliation_payload(reconciliation)
        if any(self.event.payload.get(key) != value for key, value in expected_payload.items()):
            raise ValueError("Budget settlement audit event conflicts with committed accounting.")
        return self


class BudgetSettlementCursor(BaseModel):
    """Stable keyset position in the ledger settlement outbox."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    settled_at: datetime
    settlement_id: str

    @field_validator("settled_at")
    @classmethod
    def validate_settled_at(cls, value: datetime) -> datetime:
        return _utc_datetime(value, "settled_at")

    @field_validator("settlement_id")
    @classmethod
    def validate_settlement_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "settlement_id")


class BudgetStore(ABC):
    """Durable source for cross-session budget accounting."""

    @abstractmethod
    async def append_event(self, event: Event) -> None:
        """Observe one cost-bearing event, idempotently by session and event id.

        Runtime delivery is at-least-once across crashes. Implementations must
        return without adding a second charge when the same immutable event is
        retried, and reject a conflicting event with the same identity.
        """

    @abstractmethod
    async def load_events_for_budget(
        self,
        *,
        scope: BudgetScope,
        key: str | None,
        window: BudgetWindow,
    ) -> list[Event]:
        """Return events that contribute to the given budget scope."""


class BudgetLedger(ABC):
    """Atomic reservation ledger for strict budget enforcement."""

    async def claim_reservation_identity(
        self,
        *,
        reservation_id: str,
        publication_session_id: str,
        publication_id: str,
    ) -> None:
        """Permanently bind one reservation id to its exact event publication.

        The default protects all apps sharing this ledger instance in one
        process. Durable multi-worker ledgers must override this method and
        enforce the same idempotent claim in their shared storage.
        """

        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        publication_session_id = require_clean_nonblank(
            publication_session_id,
            "publication_session_id",
        )
        publication_id = require_clean_nonblank(publication_id, "publication_id")
        state = _budget_ledger_reservation_identity_state(self)
        publication = (publication_session_id, publication_id)
        with state.lock:
            existing = state.claims.setdefault(reservation_id, publication)
            if existing != publication:
                raise BudgetReservationIdentityConflict(
                    "Budget ledger reused a reservation identity."
                )

    @property
    def reservation_ttl_seconds(self) -> int | None:
        """Active-reservation lease duration, or ``None`` for non-expiring ledgers.

        Custom ledgers remain non-expiring by default. A ledger that advertises
        a finite TTL must also implement :meth:`heartbeat` so the runtime can
        keep live provider calls reserved.
        """

        return None

    async def heartbeat(self, *, reservation_id: str) -> bool:
        """Renew one active reservation lease.

        Return ``False`` when the reservation is terminal or its lease already
        expired. Unknown reservation ids raise ``KeyError``. The default is
        intentionally unsupported because the base ledger advertises no TTL.
        """

        raise NotImplementedError("This budget ledger does not use expiring reservations.")

    @abstractmethod
    async def mark_dispatched(
        self,
        *,
        reservation_ids: tuple[str, ...],
        dispatch_id: str,
        dispatched_at: datetime | None = None,
    ) -> tuple[BudgetReservationRecord, ...]:
        """Atomically fence one attempt's reservations before provider code.

        An identical retry is idempotent. Once this transition commits, TTL
        cleanup must retain every reservation because provider dispatch is no
        longer known not to have occurred. Implementations must update the
        complete set or none of it.
        """

    @abstractmethod
    async def reserve(
        self,
        *,
        reservation_id: str | None = None,
        limit: BudgetLimit,
        session_id: str,
        agent_name: str,
        provider_name: str,
        model: str,
        model_attempt_identity: ModelAttemptIdentity,
        environment_name: str | None = None,
        settlement_event_payload: dict[str, Any] | None = None,
        settlement_fallback: BudgetSettlementFallback | None = None,
        requested_amount: Decimal | None = None,
        billing_identity: BillingIdentity | None = None,
        effective_at: datetime | None = None,
    ) -> BudgetReservationResult:
        """Reserve budget for one provider dispatch if capacity remains.

        ``reservation_id`` lets the runtime validate durable authority before
        asking the ledger to mutate. Direct ledger callers may omit it and let
        the ledger allocate one.
        ``effective_at`` fixes the pricing instant selected by the runtime.
        Direct ledger callers may omit it to use the ledger's current clock.
        ``settlement_fallback`` binds a publication-safe conservative outcome
        before provider dispatch. Direct callers may omit it when no workload
        secret publication policy applies.
        ``requested_amount`` lets the runtime supply the amount priced from
        transient request evidence before ``billing_identity`` is normalized
        for durable storage. Direct callers normally omit it and let the ledger
        price from ``billing_identity``.
        Implementations must reject a generated ``reservation_id`` that is
        already owned by another record before mutating the existing record.
        """

    @abstractmethod
    async def reconcile(
        self,
        *,
        reservation_id: str,
        actual_amount: Decimal,
        settlement_kind: Literal["completed", "conservative"] = "completed",
        reason: str | None = None,
        occurred_at: datetime | None = None,
        billing_identity: BillingIdentity | None = None,
        pricing: BudgetReconciliationPricing | None = None,
    ) -> BudgetReconciliation:
        """Replace a reservation with the charged amount.

        Implementations must return the existing result for an identical retry
        without changing its accounting timestamp, and reject a conflicting
        terminal outcome for the same reservation.
        """

    @abstractmethod
    async def release(
        self,
        *,
        reservation_id: str,
        reason: str,
        occurred_at: datetime | None = None,
    ) -> BudgetReconciliation:
        """Release an active reservation without charging it.

        Implementations must return the existing result for an identical retry
        and reject a conflicting terminal outcome for the same reservation.
        A reservation with a committed dispatch fence is not releasable.
        """

    @abstractmethod
    async def load_settlement(self, settlement_id: str) -> BudgetSettlementRecord | None:
        """Load one exact durable terminal transition by stable identity."""

    @abstractmethod
    async def list_pending_settlements(
        self,
        *,
        session_id: str | None = None,
        after: BudgetSettlementCursor | None = None,
        limit: int = 100,
    ) -> list[BudgetSettlementRecord]:
        """Return a bounded deterministic page of unpublished audit events.

        Omitting ``session_id`` returns a ledger-wide page so a reservation in
        one session can publish releases created while another session reaps
        its expired capacity. ``after`` advances through unavailable publication
        domains without acknowledging or starving their pending records.
        """

    @abstractmethod
    async def mark_settlement_event_published(
        self,
        *,
        settlement_id: str,
        event_id: str,
    ) -> BudgetSettlementRecord:
        """Acknowledge that the settlement's exact audit event is durable."""


class InMemoryBudgetStore(BudgetStore):
    """In-memory budget store for tests and local apps."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._events_by_id: dict[tuple[str, str], Event] = {}
        self._lock = asyncio.Lock()

    async def append_event(self, event: Event) -> None:
        copied = copy_event(event)
        async with self._lock:
            identity = (copied.session_id, copied.id)
            existing = self._events_by_id.get(identity)
            if existing is not None:
                if existing != copied:
                    raise ValueError(
                        "Budget event identity was reused with conflicting contents: "
                        f"{copied.session_id}/{copied.id}"
                    )
                return
            self._events_by_id[identity] = copied
            self._events.append(copied)

    async def load_events_for_budget(
        self,
        *,
        scope: BudgetScope,
        key: str | None,
        window: BudgetWindow,
    ) -> list[Event]:
        async with self._lock:
            events = [copy_event(event) for event in self._events]
        events = events_for_budget_window(events, window)
        if scope == "app":
            return events
        if scope == "agent":
            budget_key = require_clean_nonblank(key or "", "key")
            return [event for event in events if event.agent_name == budget_key]
        if scope == "causal":
            raise ValueError(
                "InMemoryBudgetStore cannot resolve causal budget scope. "
                "Use SessionBudgetStore for causal budgets."
            )
        raise ValueError(f"Unsupported budget scope: {scope}")


class SessionBudgetStore(BudgetStore):
    """Budget store backed by the existing durable session event stream."""

    def __init__(self, session_store: Any) -> None:
        from cayu.runtime.sessions import SessionStore

        if not isinstance(session_store, SessionStore):
            raise TypeError("session_store must be a SessionStore.")
        self._session_store = session_store

    async def append_event(self, event: Event) -> None:
        if type(event) is not Event:
            raise TypeError("event must be an Event.")

    async def load_events_for_budget(
        self,
        *,
        scope: BudgetScope,
        key: str | None,
        window: BudgetWindow,
    ) -> list[Event]:
        from cayu.runtime.sessions import EventQuery

        window = copy_budget_window(window)
        since, until = window.bounds()
        agent_name: str | None = None
        causal_budget_id: str | None = None
        if scope == "agent":
            agent_name = require_clean_nonblank(key or "", "key")
        elif scope == "causal":
            causal_budget_id = require_clean_nonblank(key or "", "key")
        elif scope != "app":
            raise ValueError(f"Unsupported budget scope: {scope}")

        records = []
        after_sequence: int | None = None
        while True:
            page = await self._session_store.query_events(
                EventQuery(
                    event_type=EventType.MODEL_COMPLETED,
                    causal_budget_id=causal_budget_id,
                    agent_name=agent_name,
                    since=since,
                    until=until,
                    after_sequence=after_sequence,
                    limit=5000,
                )
            )
            if not page:
                break
            records.extend(page)
            after_sequence = page[-1].sequence
            if len(page) < 5000:
                break
        return [copy_event(record.event) for record in records]


class InMemoryBudgetLedger(BudgetLedger):
    """In-memory reservation ledger for single-process apps and tests."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        reservation_ttl_seconds: int | None = DEFAULT_RESERVATION_TTL_SECONDS,
    ) -> None:
        self._records: dict[str, BudgetReservationRecord] = {}
        self._settlements: dict[str, BudgetSettlementRecord] = {}
        self._lock = asyncio.Lock()
        self._clock = _clock_or_utc_now(clock)
        self._reservation_ttl_seconds = _validate_reservation_ttl(reservation_ttl_seconds)

    @property
    def reservation_ttl_seconds(self) -> int | None:
        return self._reservation_ttl_seconds

    async def reserve(
        self,
        *,
        reservation_id: str | None = None,
        limit: BudgetLimit,
        session_id: str,
        agent_name: str,
        provider_name: str,
        model: str,
        model_attempt_identity: ModelAttemptIdentity,
        environment_name: str | None = None,
        settlement_event_payload: dict[str, Any] | None = None,
        settlement_fallback: BudgetSettlementFallback | None = None,
        requested_amount: Decimal | None = None,
        billing_identity: BillingIdentity | None = None,
        effective_at: datetime | None = None,
    ) -> BudgetReservationResult:
        reservation_id = (
            new_budget_reservation_id()
            if reservation_id is None
            else require_clean_nonblank(reservation_id, "reservation_id")
        )
        limit = _ensure_effective_budget_limit(
            limit,
            identity_namespace="app_policy",
        )
        session_id = require_clean_nonblank(session_id, "session_id")
        agent_name = require_clean_nonblank(agent_name, "agent_name")
        provider_name = require_clean_nonblank(provider_name, "provider_name")
        model = require_clean_nonblank(model, "model")
        if environment_name is not None:
            environment_name = require_clean_nonblank(environment_name, "environment_name")
        durable_billing_identity = copy_billing_identity(billing_identity)
        durable_settlement_payload = copy_durable_json_object(
            settlement_event_payload or {},
            "settlement_event_payload",
        )
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        async with self._lock:
            now = self._clock()
            durable_settlement_fallback = (
                BudgetSettlementFallback(
                    settled_at=now,
                    expiration_reason=(
                        None
                        if self._reservation_ttl_seconds is None
                        else _expired_reservation_reason(self._reservation_ttl_seconds)
                    ),
                )
                if settlement_fallback is None
                else copy_budget_settlement_fallback(settlement_fallback)
            )
            pricing_effective_at = (
                now if effective_at is None else _utc_datetime(effective_at, "effective_at")
            )
            request = (
                _budget_reservation_amount(
                    limit=limit,
                    provider_name=provider_name,
                    model=model,
                    effective_at=pricing_effective_at,
                    billing_identity=durable_billing_identity,
                )
                if requested_amount is None
                else _validate_amount(requested_amount, "requested_amount")
            )
            self._reap_expired_unlocked(now, limit=limit)
            current = _ledger_used_amount(
                self._records.values(),
                limit=limit,
                now=now,
            )
            projected = current + request
            if projected > limit.max_estimated_cost:
                return _reservation_result(
                    limit=limit,
                    model_attempt_identity=model_attempt_identity,
                    accepted=False,
                    requested=request,
                    actual=projected,
                    message=(
                        "Budget reservation failed: "
                        f"{projected} > {limit.max_estimated_cost} {limit.currency}."
                    ),
                )
            record = BudgetReservationRecord(
                reservation_id=reservation_id,
                budget_limit_id=limit.budget_limit_id,
                model_step_id=model_attempt_identity.model_step_id,
                model_attempt_id=model_attempt_identity.model_attempt_id,
                scope=limit.scope,
                key=limit.key,
                window=limit.window,
                currency=limit.currency,
                session_id=session_id,
                agent_name=agent_name,
                environment_name=environment_name,
                provider_name=provider_name,
                model=model,
                billing_identity=durable_billing_identity,
                settlement_event_payload=durable_settlement_payload,
                settlement_fallback=durable_settlement_fallback,
                reserved_amount=request,
                created_at=now,
                updated_at=now,
            )
            if record.reservation_id in self._records:
                raise BudgetReservationIdentityConflict(
                    "Budget ledger reused a reservation identity."
                )
            self._records[record.reservation_id] = record.model_copy(deep=True)
            return _reservation_result(
                limit=limit,
                model_attempt_identity=model_attempt_identity,
                accepted=True,
                requested=request,
                actual=projected,
                message=(
                    f"Budget reserved: {request} {limit.currency} for {provider_name}/{model}."
                ),
                record=record,
            )

    async def mark_dispatched(
        self,
        *,
        reservation_ids: tuple[str, ...],
        dispatch_id: str,
        dispatched_at: datetime | None = None,
    ) -> tuple[BudgetReservationRecord, ...]:
        reservation_ids = _validate_reservation_id_batch(reservation_ids)
        dispatch_id = require_clean_nonblank(dispatch_id, "dispatch_id")
        marked_at = (
            _utc_datetime(dispatched_at, "dispatched_at")
            if dispatched_at is not None
            else self._clock()
        )
        async with self._lock:
            records: list[BudgetReservationRecord] = []
            for reservation_id in reservation_ids:
                record = self._records.get(reservation_id)
                if record is None:
                    raise KeyError(f"Budget reservation not found: {reservation_id}")
                if record.dispatch_id is not None and record.dispatch_id != dispatch_id:
                    raise ValueError(
                        f"Budget reservation has a conflicting dispatch: {reservation_id}"
                    )
                if record.dispatch_id is None and record.status != "active":
                    raise ValueError(f"Budget reservation is not active: {reservation_id}")
                records.append(record)
            dispatched_records = tuple(
                (
                    record
                    if record.dispatch_id is not None
                    else record.model_copy(
                        update={"dispatch_id": dispatch_id, "dispatched_at": marked_at},
                        deep=True,
                    )
                )
                for record in records
            )
            for record in dispatched_records:
                self._records[record.reservation_id] = record
            return tuple(record.model_copy(deep=True) for record in dispatched_records)

    async def heartbeat(self, *, reservation_id: str) -> bool:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        now = self._clock()
        async with self._lock:
            record = self._records.get(reservation_id)
            if record is None:
                raise KeyError(f"Budget reservation not found: {reservation_id}")
            if record.status != "active" or _reservation_is_expired(
                record,
                now=now,
                ttl_seconds=self._reservation_ttl_seconds,
            ):
                return False
            self._records[reservation_id] = record.model_copy(
                update={"updated_at": now},
                deep=True,
            )
            return True

    async def reconcile(
        self,
        *,
        reservation_id: str,
        actual_amount: Decimal,
        settlement_kind: Literal["completed", "conservative"] = "completed",
        reason: str | None = None,
        occurred_at: datetime | None = None,
        billing_identity: BillingIdentity | None = None,
        pricing: BudgetReconciliationPricing | None = None,
    ) -> BudgetReconciliation:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        actual_amount = _validate_amount(actual_amount, "actual_amount")
        reconciled_at = _utc_datetime(occurred_at, "occurred_at") if occurred_at else self._clock()
        async with self._lock:
            record = self._reconcilable_record(reservation_id)
            reconciled = _reconciled_record(
                record,
                actual_amount=actual_amount,
                reason=reason,
                updated_at=reconciled_at,
                billing_identity=billing_identity,
            )
            reconciliation = _reconciliation_from_record(
                reconciled,
                settlement_kind=settlement_kind,
                pricing=pricing,
            )
            settlement = _budget_settlement_record(record, reconciliation)
            self._store_settlement_unlocked(settlement)
            self._records[reservation_id] = reconciled
            return reconciliation.model_copy(deep=True)

    async def release(
        self,
        *,
        reservation_id: str,
        reason: str,
        occurred_at: datetime | None = None,
    ) -> BudgetReconciliation:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        reason = require_clean_nonblank(reason, "reason")
        released_at = (
            _utc_datetime(occurred_at, "occurred_at") if occurred_at is not None else self._clock()
        )
        async with self._lock:
            record = self._releasable_record(reservation_id)
            released = _released_record(
                record,
                reason=reason,
                updated_at=released_at,
            )
            reconciliation = _reconciliation_from_record(
                released,
                settlement_kind="released",
            )
            settlement = _budget_settlement_record(record, reconciliation)
            self._store_settlement_unlocked(settlement)
            self._records[reservation_id] = released
            return reconciliation.model_copy(deep=True)

    async def load_settlement(self, settlement_id: str) -> BudgetSettlementRecord | None:
        settlement_id = require_clean_nonblank(settlement_id, "settlement_id")
        async with self._lock:
            settlement = self._settlements.get(settlement_id)
            return None if settlement is None else settlement.model_copy(deep=True)

    async def list_pending_settlements(
        self,
        *,
        session_id: str | None = None,
        after: BudgetSettlementCursor | None = None,
        limit: int = 100,
    ) -> list[BudgetSettlementRecord]:
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        after = _copy_budget_settlement_cursor(after)
        limit = _validate_settlement_page_limit(limit)
        async with self._lock:
            settlements = sorted(
                (
                    settlement
                    for settlement in self._settlements.values()
                    if not settlement.event_published
                    and (session_id is None or settlement.session_id == session_id)
                    and (
                        after is None
                        or (
                            settlement.reconciliation.settled_at,
                            settlement.settlement_id,
                        )
                        > (after.settled_at, after.settlement_id)
                    )
                ),
                key=lambda settlement: (
                    settlement.reconciliation.settled_at,
                    settlement.settlement_id,
                ),
            )
            return [item.model_copy(deep=True) for item in settlements[:limit]]

    async def mark_settlement_event_published(
        self,
        *,
        settlement_id: str,
        event_id: str,
    ) -> BudgetSettlementRecord:
        settlement_id = require_clean_nonblank(settlement_id, "settlement_id")
        event_id = require_clean_nonblank(event_id, "event_id")
        async with self._lock:
            settlement = self._settlements.get(settlement_id)
            if settlement is None:
                raise KeyError(f"Budget settlement not found: {settlement_id}")
            if settlement.event.id != event_id:
                raise ValueError(
                    "Budget settlement event acknowledgement has conflicting identity."
                )
            if not settlement.event_published:
                settlement = settlement.model_copy(update={"event_published": True}, deep=True)
                self._settlements[settlement_id] = settlement
            return settlement.model_copy(deep=True)

    def _store_settlement_unlocked(self, settlement: BudgetSettlementRecord) -> None:
        existing = self._settlements.get(settlement.settlement_id)
        if existing is None:
            self._settlements[settlement.settlement_id] = settlement.model_copy(deep=True)
            return
        expected = existing.model_copy(update={"event_published": False}, deep=True)
        if expected != settlement:
            raise ValueError(
                f"Budget reservation has a conflicting settlement: {settlement.reservation_id}"
            )

    def _reap_expired_unlocked(
        self,
        now: datetime,
        *,
        limit: _EffectiveBudgetLimit,
    ) -> None:
        if self._reservation_ttl_seconds is None:
            return
        for reservation_id, record in self._records.items():
            if (
                record.status != "active"
                or record.dispatch_id is not None
                or not _reservation_matches_limit(record, limit)
                or not _reservation_is_expired(
                    record,
                    now=now,
                    ttl_seconds=self._reservation_ttl_seconds,
                )
            ):
                continue
            released = _released_record(
                record,
                reason=(
                    record.settlement_fallback.expiration_reason
                    or _expired_reservation_reason(self._reservation_ttl_seconds)
                ),
                updated_at=record.settlement_fallback.settled_at,
            )
            reconciliation = _reconciliation_from_record(
                released,
                settlement_kind="released",
            )
            self._store_settlement_unlocked(_budget_settlement_record(record, reconciliation))
            self._records[reservation_id] = released

    def _active_record(self, reservation_id: str) -> BudgetReservationRecord:
        record = self._records.get(reservation_id)
        if record is None:
            raise KeyError(f"Budget reservation not found: {reservation_id}")
        if record.status != "active":
            raise ValueError(f"Budget reservation is not active: {reservation_id}")
        return record

    def _releasable_record(self, reservation_id: str) -> BudgetReservationRecord:
        record = self._records.get(reservation_id)
        if record is None:
            raise KeyError(f"Budget reservation not found: {reservation_id}")
        if record.status == "active" and record.dispatch_id is not None:
            raise ValueError(f"Dispatched budget reservation cannot be released: {reservation_id}")
        if record.status in {"active", "released"}:
            return record
        raise ValueError(f"Budget reservation is not active: {reservation_id}")

    def _reconcilable_record(self, reservation_id: str) -> BudgetReservationRecord:
        record = self._records.get(reservation_id)
        if record is None:
            raise KeyError(f"Budget reservation not found: {reservation_id}")
        if record.status in {"active", "reconciled"}:
            return record
        raise ValueError(f"Budget reservation is not active: {reservation_id}")


def copy_budget_reservation(reservation: BudgetReservation) -> BudgetReservation:
    if type(reservation) is not BudgetReservation:
        raise TypeError("reservation must be a BudgetReservation.")
    return BudgetReservation(
        max_input_tokens=reservation.max_input_tokens,
        max_output_tokens=reservation.max_output_tokens,
        max_cache_read_input_tokens=reservation.max_cache_read_input_tokens,
        max_cache_write_input_tokens=reservation.max_cache_write_input_tokens,
    )


def copy_budget_window(value: BudgetWindow | Mapping[str, Any] | str | None) -> BudgetWindow:
    if value is None:
        return BudgetWindow.all_time()
    if type(value) is BudgetWindow:
        return BudgetWindow.model_validate(value.model_dump(mode="python"))
    if isinstance(value, str):
        return _budget_window_from_string(value)
    if isinstance(value, Mapping):
        return BudgetWindow.model_validate(dict(value))
    raise TypeError("Budget window must be a BudgetWindow, mapping, string, or None.")


def events_for_budget_window(
    events: Iterable[Event],
    window: BudgetWindow | Mapping[str, Any] | str | None,
    *,
    now: datetime | None = None,
) -> list[Event]:
    window = copy_budget_window(window)
    since, until = window.bounds(now=now)
    copied = [copy_event(event) for event in events]
    if since is None and until is None:
        return copied
    return [event for event in copied if _event_in_window(event, since=since, until=until)]


def copy_budget_limit(limit: BudgetLimit) -> BudgetLimit:
    return _copy_budget_limit(limit)


def copy_budget_policy(policy: BudgetPolicy | None) -> BudgetPolicy | None:
    if policy is None:
        return None
    if type(policy) is not BudgetPolicy:
        raise TypeError("Budget policy must be a BudgetPolicy instance.")
    return BudgetPolicy(limits=tuple(_copy_budget_limit(limit) for limit in policy.limits))


def copy_budget_limits(
    limits: Iterable[BudgetLimit | Mapping[str, Any]] | None,
    *,
    field_name: str = "budget_limits",
) -> tuple[BudgetLimit, ...]:
    if limits is None:
        return ()
    if isinstance(limits, str | bytes):
        raise ValueError(f"{field_name} must be an iterable of BudgetLimit values.")
    return tuple(_coerce_budget_limit(limit) for limit in limits)


def copy_request_budget_limits(
    limits: Iterable[BudgetLimit | Mapping[str, Any]] | None,
) -> tuple[BudgetLimit, ...]:
    """Copy per-request budget limits, allowing reservations on shared scopes.

    Request limits scoped ``app``/``agent``/``causal`` may configure a
    reservation: those budgets are shared across sessions, so the runtime
    routes them through the atomic budget ledger before each model step,
    keeping concurrent sessions from jointly overshooting the limit.

    ``session``/``run`` scoped limits must not reserve. They are accounted
    from a single session's own event stream, and a session executes its
    model steps sequentially, so there is no cross-session race to close.
    Without a reservation the residual race for those scopes is only the
    usage of the one model step that is in flight when the read-then-act
    check passes.
    """
    copied = copy_budget_limits(limits, field_name="budget_limits")
    for limit in copied:
        if limit.reservation is not None and limit.scope in ("session", "run"):
            raise ValueError(
                "Request budget limits must not use reservations for "
                f"{limit.scope!r} scope; reservations require a shared "
                "budget scope (app, agent, or causal)."
            )
    return copied


def budget_limits_for_session(
    *,
    policy: BudgetPolicy | None,
    agent_name: str,
    causal_budget_id: str,
) -> tuple[_EffectiveBudgetLimit, ...]:
    policy = copy_budget_policy(policy)
    if policy is None:
        return ()
    agent_name = require_clean_nonblank(agent_name, "agent_name")
    causal_budget_id = require_clean_nonblank(causal_budget_id, "causal_budget_id")
    matched: list[_EffectiveBudgetLimit] = []
    effective_limits = _effective_budget_limits(
        policy.limits,
        identity_namespace="app_policy",
    )
    for limit in effective_limits:
        if (
            limit.scope == "app"
            or (limit.scope == "agent" and limit.key == agent_name)
            or (limit.scope == "causal" and limit.key == causal_budget_id)
        ):
            matched.append(_copy_effective_budget_limit(limit))
    return tuple(matched)


def request_budget_limits_for_session(
    *,
    limits: Iterable[BudgetLimit | Mapping[str, Any]] | None,
    agent_name: str,
    causal_budget_id: str,
) -> tuple[_EffectiveBudgetLimit, ...]:
    copied = copy_request_budget_limits(limits)
    agent_name = require_clean_nonblank(agent_name, "agent_name")
    causal_budget_id = require_clean_nonblank(causal_budget_id, "causal_budget_id")
    effective_limits = _effective_budget_limits(
        copied,
        identity_namespace="request",
    )
    _validate_budget_limit_session_keys(
        effective_limits,
        agent_name=agent_name,
        causal_budget_id=causal_budget_id,
    )
    return effective_limits


def _operation_budget_limits_for_session(
    *,
    limits: Iterable[BudgetLimit | Mapping[str, Any]] | None,
    agent_name: str,
    causal_budget_id: str,
) -> tuple[_EffectiveBudgetLimit, ...]:
    """Verify resolved identities while assigning raw internal operation limits."""

    copied = copy_request_budget_limits(limits)
    agent_name = require_clean_nonblank(agent_name, "agent_name")
    causal_budget_id = require_clean_nonblank(causal_budget_id, "causal_budget_id")
    effective_limits = _effective_budget_limits(
        copied,
        identity_namespace="operation",
        preserve_effective=True,
    )
    _validate_budget_limit_session_keys(
        effective_limits,
        agent_name=agent_name,
        causal_budget_id=causal_budget_id,
    )
    return effective_limits


def _validate_budget_limit_session_keys(
    limits: Iterable[_EffectiveBudgetLimit],
    *,
    agent_name: str,
    causal_budget_id: str,
) -> None:
    """Ensure keyed limits cannot be applied to a different runtime scope."""

    agent_name = require_clean_nonblank(agent_name, "agent_name")
    causal_budget_id = require_clean_nonblank(causal_budget_id, "causal_budget_id")
    for limit in limits:
        if limit.scope == "agent" and limit.key != agent_name:
            raise ValueError(
                f"Request agent budget limit key {limit.key!r} does not match "
                f"session agent {agent_name!r}."
            )
        if limit.scope == "causal" and limit.key != causal_budget_id:
            raise ValueError(
                f"Request causal budget limit key {limit.key!r} does not match "
                f"session causal_budget_id {causal_budget_id!r}."
            )


def budget_check_from_events(
    *,
    limit: BudgetLimit,
    events: list[Event],
    provider_name: str | None = None,
    model: str | None = None,
    billing_identity_state: BillingIdentityState = UNRESOLVED_BILLING_IDENTITY,
    effective_at: datetime | None = None,
) -> BudgetCheck:
    if type(limit) not in {BudgetLimit, _EffectiveBudgetLimit}:
        raise TypeError("limit must be a BudgetLimit.")
    limit = _ensure_effective_budget_limit(
        limit,
        identity_namespace="app_policy",
    )
    summary = estimate_session_cost(
        session_id=_budget_summary_id(limit),
        events=events,
        pricing=limit.pricing,
        currency=limit.currency,
    )
    limit_reached = False
    if summary.unpriced_model_steps > 0 and not limit.allow_unpriced:
        limit_reached = True
        message = (
            "Budget cannot be verified because "
            f"{summary.unpriced_model_steps} model step(s) have no matching pricing."
        )
    elif summary.total_cost >= limit.max_estimated_cost:
        limit_reached = True
        message = (
            f"Budget reached: {summary.total_cost} >= {limit.max_estimated_cost} {limit.currency}."
        )
    elif (
        not limit.allow_unpriced
        and (
            preflight_error := _budget_preflight_error(
                limit,
                provider_name,
                model,
                billing_identity_state=billing_identity_state,
                effective_at=effective_at,
            )
        )
        is not None
    ):
        limit_reached = True
        message = preflight_error
    else:
        message = (
            f"Budget checked: {summary.total_cost} < {limit.max_estimated_cost} {limit.currency}."
        )
    return BudgetCheck(
        budget_limit_id=limit.budget_limit_id,
        scope=limit.scope,
        key=limit.key,
        window=limit.window,
        currency=limit.currency,
        maximum=limit.max_estimated_cost,
        actual=summary.total_cost,
        action=limit.action,
        model_steps=summary.model_steps,
        unpriced_model_steps=summary.unpriced_model_steps,
        limit_reached=limit_reached,
        message=message,
        cost_summary=summary,
    )


def budget_check_payload(check: BudgetCheck) -> dict[str, Any]:
    if type(check) is not BudgetCheck:
        raise TypeError("check must be a BudgetCheck.")
    return {
        "budget_limit_id": check.budget_limit_id,
        "scope": check.scope,
        "key": check.key,
        "window": check.window.storage_key,
        "window_details": check.window.model_dump(mode="json"),
        "currency": check.currency,
        "maximum": str(check.maximum),
        "actual": str(check.actual),
        "action": check.action,
        "model_steps": check.model_steps,
        "unpriced_model_steps": check.unpriced_model_steps,
        "limit_reached": check.limit_reached,
        "message": check.message,
        "cost_summary": check.cost_summary.model_dump(mode="json"),
    }


def _copy_budget_limit(limit: BudgetLimit) -> BudgetLimit:
    if type(limit) is _EffectiveBudgetLimit:
        return _copy_effective_budget_limit(limit)
    if type(limit) is not BudgetLimit:
        raise TypeError("Budget limits must be BudgetLimit instances.")
    return BudgetLimit(
        scope=limit.scope,
        max_estimated_cost=limit.max_estimated_cost,
        pricing=limit.pricing.model_copy(deep=True),
        currency=limit.currency,
        window=limit.window,
        key=limit.key,
        allow_unpriced=limit.allow_unpriced,
        action=limit.action,
        reservation=(
            None if limit.reservation is None else copy_budget_reservation(limit.reservation)
        ),
    )


def _copy_effective_budget_limit(limit: _EffectiveBudgetLimit) -> _EffectiveBudgetLimit:
    if type(limit) is not _EffectiveBudgetLimit:
        raise TypeError("Effective budget limits must be runtime-owned limit instances.")
    copied = _copy_budget_limit_fields(limit)
    return _EffectiveBudgetLimit(
        **copied,
        budget_limit_id=limit.budget_limit_id,
        identity_namespace=limit.identity_namespace,
        identity_definition_sha256=limit.identity_definition_sha256,
        identity_occurrences=limit.identity_occurrences,
        identity_occurrence=limit.identity_occurrence,
    )


def _copy_budget_limit_fields(limit: BudgetLimit) -> dict[str, Any]:
    return {
        "scope": limit.scope,
        "max_estimated_cost": limit.max_estimated_cost,
        "pricing": limit.pricing.model_copy(deep=True),
        "currency": limit.currency,
        "window": limit.window,
        "key": limit.key,
        "allow_unpriced": limit.allow_unpriced,
        "action": limit.action,
        "reservation": (
            None if limit.reservation is None else copy_budget_reservation(limit.reservation)
        ),
    }


def _effective_budget_limits(
    limits: Iterable[BudgetLimit],
    *,
    identity_namespace: str,
    preserve_effective: bool = False,
) -> tuple[_EffectiveBudgetLimit, ...]:
    """Assign stable opaque identities while keeping identical entries distinct."""

    namespace = require_clean_nonblank(identity_namespace, "identity_namespace")
    prepared: list[tuple[BudgetLimit, str | None]] = []
    definition_counts: dict[str, int] = {}
    for limit in limits:
        if type(limit) is _EffectiveBudgetLimit and preserve_effective:
            prepared.append((_copy_effective_budget_limit(limit), None))
            continue
        copied_limit = (
            BudgetLimit(**_copy_budget_limit_fields(limit))
            if type(limit) is _EffectiveBudgetLimit
            else _copy_budget_limit(limit)
        )
        definition_digest = _budget_limit_definition_digest(copied_limit)
        definition_counts[definition_digest] = definition_counts.get(definition_digest, 0) + 1
        prepared.append((copied_limit, definition_digest))

    occurrences: dict[str, int] = {}
    effective: list[_EffectiveBudgetLimit] = []
    seen_ids: set[str] = set()
    for limit, definition_digest in prepared:
        if type(limit) is _EffectiveBudgetLimit:
            copied = _copy_effective_budget_limit(limit)
        else:
            if definition_digest is None:  # pragma: no cover - prepared above
                raise AssertionError("Configured budget limit is missing its definition digest.")
            occurrence = occurrences.get(definition_digest, 0)
            occurrences[definition_digest] = occurrence + 1
            definition_occurrences = definition_counts[definition_digest]
            budget_limit_id = _budget_limit_id_from_material(
                namespace=namespace,
                definition_digest=definition_digest,
                definition_occurrences=definition_occurrences,
                occurrence=occurrence,
            )
            copied = _EffectiveBudgetLimit(
                **_copy_budget_limit_fields(limit),
                budget_limit_id=budget_limit_id,
                identity_namespace=namespace,
                identity_definition_sha256=definition_digest,
                identity_occurrences=definition_occurrences,
                identity_occurrence=occurrence,
            )
        if copied.budget_limit_id in seen_ids:
            raise ValueError("Effective budget limits must have distinct identities.")
        seen_ids.add(copied.budget_limit_id)
        effective.append(copied)
    return tuple(effective)


def _budget_limit_definition_digest(limit: BudgetLimit) -> str:
    """Hash budget semantics independently of Decimal input spelling."""

    configured = BudgetLimit(**_copy_budget_limit_fields(limit))
    python_value = configured.model_dump(mode="python")
    json_value = configured.model_dump(mode="json")
    semantic_value = _semantic_budget_identity_value(python_value, json_value)
    return sha256(canonical_durable_json_bytes(semantic_value, "budget_limit")).hexdigest()


def _semantic_budget_identity_value(python_value: Any, json_value: Any) -> Any:
    if type(python_value) is Decimal:
        return {"decimal": _canonical_decimal_identity(python_value)}
    if type(python_value) is dict and type(json_value) is dict:
        if python_value.keys() != json_value.keys():
            raise RuntimeError("Budget identity serialization changed object fields.")
        return {
            key: _semantic_budget_identity_value(python_value[key], json_value[key])
            for key in python_value
        }
    if isinstance(python_value, list | tuple) and type(json_value) is list:
        if len(python_value) != len(json_value):
            raise RuntimeError("Budget identity serialization changed collection length.")
        return [
            _semantic_budget_identity_value(item, serialized)
            for item, serialized in zip(python_value, json_value, strict=True)
        ]
    return json_value


def _canonical_decimal_identity(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("Budget identity decimals must be finite.")
    sign, raw_digits, raw_exponent = value.as_tuple()
    if not isinstance(raw_exponent, int):
        raise ValueError("Budget identity decimal exponent must be finite.")
    digits = list(raw_digits)
    exponent = raw_exponent
    while digits and digits[-1] == 0:
        digits.pop()
        exponent += 1
    if not digits:
        return "0"
    coefficient = "".join(str(digit) for digit in digits)
    return f"{'-' if sign else ''}{coefficient}e{exponent}"


def _budget_limit_id_from_material(
    *,
    namespace: str,
    definition_digest: str,
    definition_occurrences: int,
    occurrence: int,
) -> str:
    identity_material = canonical_durable_json_bytes(
        {
            "namespace": namespace,
            "definition_sha256": definition_digest,
            # Membership is part of an indistinguishable duplicate set's
            # identity. Adding or removing one duplicate therefore creates a
            # new set instead of letting a survivor inherit a sibling's ledger.
            "definition_occurrences": definition_occurrences,
            "occurrence": occurrence,
        },
        "budget_limit_identity",
    )
    return f"blim_{sha256(identity_material).hexdigest()}"


def _ensure_effective_budget_limit(
    limit: BudgetLimit,
    *,
    identity_namespace: str,
) -> _EffectiveBudgetLimit:
    if type(limit) is _EffectiveBudgetLimit:
        return _copy_effective_budget_limit(limit)
    return _effective_budget_limits(
        (limit,),
        identity_namespace=identity_namespace,
    )[0]


def _effective_budget_limit_id(limit: BudgetLimit) -> str:
    if type(limit) is not _EffectiveBudgetLimit:
        raise TypeError("Budget limit has not been resolved to an effective identity.")
    return limit.budget_limit_id


def _coerce_budget_limit(limit: BudgetLimit | Mapping[str, Any]) -> BudgetLimit:
    if type(limit) is BudgetLimit:
        return _copy_budget_limit(limit)
    if type(limit) is _EffectiveBudgetLimit:
        return _copy_effective_budget_limit(limit)
    if isinstance(limit, Mapping):
        return BudgetLimit.model_validate(dict(limit))
    raise TypeError("Budget limits must be BudgetLimit instances or mappings.")


def _budget_summary_id(limit: BudgetLimit) -> str:
    if type(limit) is _EffectiveBudgetLimit:
        return f"budget:{limit.budget_limit_id}"
    key = "all" if limit.key is None else limit.key
    return f"budget:{limit.scope}:{limit.window.storage_key}:{key}"


def _budget_preflight_error(
    limit: BudgetLimit,
    provider_name: str | None,
    model: str | None,
    *,
    billing_identity_state: BillingIdentityState,
    effective_at: datetime | None,
) -> str | None:
    billing_identity = billing_identity_value(billing_identity_state)
    reference = (
        datetime.now(UTC) if effective_at is None else _utc_datetime(effective_at, "effective_at")
    )
    for pricing_context in _budget_pricing_contexts(billing_identity):
        resolution = _resolve_price_book(
            limit.pricing,
            provider_name=provider_name,
            model=model,
            input_tokens=0,
            effective_on=reference.astimezone(UTC).date(),
            billing_identity=billing_identity,
            pricing_context=pricing_context,
        )
        price = resolution.resolved
        if price is None:
            # Contextual dimensions are request-specific. The run loop repeats
            # preflight with the provider-resolved identity immediately before
            # reserving and dispatching.
            if not isinstance(
                billing_identity_state, ResolvedBillingIdentity
            ) and has_deferred_contextual_price(
                limit.pricing,
                provider_name=provider_name,
                model=model,
            ):
                return None
            reason = resolution.missing_reason or "no matching model pricing"
            return f"Budget cannot be verified because {provider_name}/{model}: {reason}."
        if price.currency.upper() != limit.currency.upper():
            return (
                "Budget cannot be verified because "
                f"{provider_name}/{model} pricing currency {price.currency} "
                f"does not match requested {limit.currency}."
            )
    return None


def has_deferred_contextual_price(
    pricing: PriceBook,
    *,
    provider_name: str | None,
    model: str | None,
) -> bool:
    if provider_name is None or model is None:
        return False
    # A directly matching context-free price belongs to the declared provider and
    # does not need deferred resolution. This also prevents an unrelated contextual
    # row with the same model identifier from shadowing it.
    if (
        pricing._resolve_match(
            provider_name=provider_name,
            model=model,
            billing_identity=None,
        )
        is not None
    ):
        return False
    pricing_model = next(
        (
            mapping.pricing_model
            for mapping in pricing.resource_mappings
            if _normalize_provider(mapping.provider_name) == _normalize_provider(provider_name)
            and mapping.resource_id == model
        ),
        model,
    )
    return any(
        _normalize_provider(price.provider_name) == _normalize_provider(provider_name)
        and price.model == pricing_model
        and price.pricing_context is not None
        for price in pricing.prices
    )


def budget_reservation_payload(result: BudgetReservationResult) -> dict[str, Any]:
    if type(result) is not BudgetReservationResult:
        raise TypeError("result must be a BudgetReservationResult.")
    payload: dict[str, Any] = {
        "accepted": result.accepted,
        "budget_limit_id": result.budget_limit_id,
        "model_step_id": result.model_step_id,
        "model_attempt_id": result.model_attempt_id,
        "scope": result.scope,
        "key": result.key,
        "window": result.window.storage_key,
        "window_details": result.window.model_dump(mode="json"),
        "currency": result.currency,
        "maximum": str(result.maximum),
        "action": result.action,
        "requested": str(result.requested),
        "actual": str(result.actual),
        "message": result.message,
    }
    if result.record is not None:
        payload["reservation_id"] = result.record.reservation_id
        payload["session_id"] = result.record.session_id
        payload["agent_name"] = result.record.agent_name
        payload["provider_name"] = result.record.provider_name
        payload["model"] = result.record.model
        if result.record.billing_identity is not None:
            payload["billing_identity"] = result.record.billing_identity.model_dump(mode="json")
    return payload


def budget_reconciliation_payload(reconciliation: BudgetReconciliation) -> dict[str, Any]:
    if type(reconciliation) is not BudgetReconciliation:
        raise TypeError("reconciliation must be a BudgetReconciliation.")
    pricing = None
    if reconciliation.pricing_provider_name is not None:
        pricing = {
            "provider_name": reconciliation.pricing_provider_name,
            "model": reconciliation.pricing_model,
            "match": reconciliation.pricing_match,
            "provenance": reconciliation.pricing_provenance.model_dump(mode="json")
            if reconciliation.pricing_provenance is not None
            else None,
            "effective_from": (
                None
                if reconciliation.pricing_effective_from is None
                else reconciliation.pricing_effective_from.isoformat()
            ),
            "effective_through": (
                None
                if reconciliation.pricing_effective_through is None
                else reconciliation.pricing_effective_through.isoformat()
            ),
            "tier_max_input_tokens": reconciliation.pricing_tier_max_input_tokens,
        }
    return {
        "reservation_id": reconciliation.reservation_id,
        "settlement_id": reconciliation.settlement_id,
        "settlement_kind": reconciliation.settlement_kind,
        "budget_limit_id": reconciliation.budget_limit_id,
        "model_step_id": reconciliation.model_step_id,
        "model_attempt_id": reconciliation.model_attempt_id,
        "status": reconciliation.status,
        "reserved_amount": str(reconciliation.reserved_amount),
        "actual_amount": (
            None if reconciliation.actual_amount is None else str(reconciliation.actual_amount)
        ),
        "released_amount": str(reconciliation.released_amount),
        "reason": reconciliation.reason,
        # This is runtime-owned accounting authority, not descriptive provider
        # text. Keep it numeric so value-based workload redaction cannot turn a
        # valid completion timestamp into an invalid durable reconciliation.
        "settled_at_unix_us": _budget_settlement_timestamp(reconciliation.settled_at),
        "pricing": pricing,
        "billing_identity": (
            None
            if reconciliation.billing_identity is None
            else reconciliation.billing_identity.model_dump(mode="json")
        ),
    }


def budget_reconciliation_from_payload(payload: object) -> BudgetReconciliation:
    """Parse the exact bounded payload emitted for one committed settlement."""

    if type(payload) is not dict:
        raise ValueError("Budget settlement evidence must be an object.")
    payload = cast("dict[str, Any]", payload)
    expected_keys = {
        "reservation_id",
        "settlement_id",
        "settlement_kind",
        "budget_limit_id",
        "model_step_id",
        "model_attempt_id",
        "status",
        "reserved_amount",
        "actual_amount",
        "released_amount",
        "reason",
        "settled_at_unix_us",
        "pricing",
        "billing_identity",
    }
    if set(payload) != expected_keys:
        raise ValueError("Budget settlement evidence has unexpected fields.")
    pricing_value = payload["pricing"]
    pricing = (
        None if pricing_value is None else BudgetReconciliationPricing.model_validate(pricing_value)
    )
    billing_value = payload["billing_identity"]
    billing_identity = (
        None if billing_value is None else BillingIdentity.model_validate(billing_value)
    )
    if pricing is not None:
        pricing = pricing.model_copy(
            update={"billing_identity": billing_identity},
            deep=True,
        )
    values: dict[str, Any] = {
        key: value
        for key, value in payload.items()
        if key not in {"pricing", "billing_identity", "settled_at_unix_us"}
    }
    values["settled_at"] = _budget_settlement_datetime(payload["settled_at_unix_us"])
    values["billing_identity"] = billing_identity
    if pricing is not None:
        values.update(
            {
                "pricing_provider_name": pricing.provider_name,
                "pricing_model": pricing.model,
                "pricing_match": pricing.match,
                "pricing_provenance": pricing.provenance,
                "pricing_effective_from": pricing.effective_from,
                "pricing_effective_through": pricing.effective_through,
                "pricing_tier_max_input_tokens": pricing.tier_max_input_tokens,
            }
        )
    return BudgetReconciliation.model_validate(values)


def model_completion_budget_settlements(
    event: Event,
    *,
    reservation_ids: Iterable[str],
) -> tuple[BudgetReconciliation, ...]:
    """Parse and bind one model completion's exact reservation settlements."""

    if type(event) is not Event or event.type != EventType.MODEL_COMPLETED:
        raise ValueError("Budget settlement recovery requires one model.completed event.")
    expected_ids = tuple(
        require_clean_nonblank(reservation_id, "reservation_id")
        for reservation_id in reservation_ids
    )
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Model completion stage contains duplicate reservation ids.")
    raw_settlements = event.payload.get(MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY)
    if type(raw_settlements) is not list:
        raise ValueError("Model completion has no durable budget settlement evidence.")
    if len(raw_settlements) != len(expected_ids):
        raise ValueError("Model completion budget settlement count conflicts with its stage.")
    parsed = tuple(budget_reconciliation_from_payload(value) for value in raw_settlements)
    parsed_ids = tuple(item.reservation_id for item in parsed)
    if (
        len(set(parsed_ids)) != len(parsed_ids)
        or set(parsed_ids) != set(expected_ids)
        or any(item.status != "reconciled" for item in parsed)
        or any(item.settlement_kind == "released" for item in parsed)
        or any(item.actual_amount is None for item in parsed)
        or any(
            item.model_step_id != event.payload.get("model_step_id")
            or item.model_attempt_id != event.payload.get("model_attempt_id")
            for item in parsed
        )
    ):
        raise ValueError("Model completion budget settlement identity is contradictory.")
    by_id = {item.reservation_id: item for item in parsed}
    return tuple(by_id[reservation_id] for reservation_id in expected_ids)


def budget_settlement_id(reservation_id: str) -> str:
    """Return the stable terminal-settlement identity for one reservation."""

    reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
    digest = sha256(
        canonical_durable_json_bytes(
            {
                "kind": "budget-settlement",
                "reservation_id": reservation_id,
                "schema_version": 1,
            },
            "budget_settlement_identity",
        )
    ).hexdigest()
    return f"bset_{digest}"


def new_budget_reservation_id() -> str:
    """Allocate one opaque runtime-owned reservation authority."""

    return f"bres_{uuid4().hex}"


def budget_settlement_event_id(settlement_id: str) -> str:
    """Return the stable session-event identity for one settlement outbox row."""

    settlement_id = require_clean_nonblank(settlement_id, "settlement_id")
    digest = sha256(
        canonical_durable_json_bytes(
            {
                "kind": "budget-settlement-event",
                "settlement_id": settlement_id,
                "schema_version": 1,
            },
            "budget_settlement_event_identity",
        )
    ).hexdigest()
    return f"budget-settlement:{digest}"


def budget_reconciliation_pricing(
    line_item: CostLineItem,
) -> BudgetReconciliationPricing:
    """Project a priced cost line into bounded durable settlement provenance."""

    if type(line_item) is not CostLineItem or not line_item.priced:
        raise TypeError("line_item must be a priced CostLineItem.")
    if (
        line_item.pricing_provider_name is None
        or line_item.pricing_model is None
        or line_item.pricing_match is None
        or line_item.pricing_provenance is None
    ):
        raise ValueError("Priced line item has incomplete pricing provenance.")
    return BudgetReconciliationPricing(
        provider_name=line_item.pricing_provider_name,
        model=line_item.pricing_model,
        match=line_item.pricing_match,
        provenance=line_item.pricing_provenance,
        effective_from=line_item.pricing_effective_from,
        effective_through=line_item.pricing_effective_through,
        tier_max_input_tokens=line_item.pricing_tier_max_input_tokens,
        billing_identity=line_item.billing_identity,
    )


class _BudgetActualCost(NamedTuple):
    amount: Decimal
    line_item: CostLineItem


def budget_actual_cost_for_event(*, limit: BudgetLimit, event: Event) -> _BudgetActualCost:
    if type(limit) not in {BudgetLimit, _EffectiveBudgetLimit}:
        raise TypeError("limit must be a BudgetLimit.")
    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    summary = estimate_session_cost(
        session_id=_budget_summary_id(limit),
        events=[event],
        pricing=limit.pricing,
        currency=limit.currency,
    )
    if summary.unpriced_model_steps > 0:
        raise ValueError("Cannot reconcile budget reservation from unpriced model usage.")
    if len(summary.line_items) != 1:
        raise ValueError("Budget reconciliation requires exactly one priced model step.")
    return _BudgetActualCost(summary.total_cost, summary.line_items[0])


def budget_reconciliation_with_pricing(
    reconciliation: BudgetReconciliation,
    line_item: CostLineItem,
) -> BudgetReconciliation:
    if type(reconciliation) is not BudgetReconciliation:
        raise TypeError("reconciliation must be a BudgetReconciliation.")
    if type(line_item) is not CostLineItem or not line_item.priced:
        raise TypeError("line_item must be a priced CostLineItem.")
    pricing = budget_reconciliation_pricing(line_item)
    return budget_reconciliation_with_pricing_evidence(
        reconciliation,
        pricing,
        billing_identity=line_item.billing_identity,
    )


def budget_reconciliation_with_pricing_evidence(
    reconciliation: BudgetReconciliation,
    pricing: BudgetReconciliationPricing,
    *,
    billing_identity: BillingIdentity | None = None,
) -> BudgetReconciliation:
    """Attach already-validated durable pricing evidence to a settlement."""

    if type(reconciliation) is not BudgetReconciliation:
        raise TypeError("reconciliation must be a BudgetReconciliation.")
    if type(pricing) is not BudgetReconciliationPricing:
        raise TypeError("pricing must be a BudgetReconciliationPricing.")
    return BudgetReconciliation.model_validate(
        {
            **reconciliation.model_dump(mode="python"),
            "pricing_provider_name": pricing.provider_name,
            "pricing_model": pricing.model,
            "pricing_match": pricing.match,
            "pricing_provenance": pricing.provenance.model_copy(deep=True),
            "pricing_effective_from": pricing.effective_from,
            "pricing_effective_through": pricing.effective_through,
            "pricing_tier_max_input_tokens": pricing.tier_max_input_tokens,
            "billing_identity": (
                billing_identity or pricing.billing_identity or reconciliation.billing_identity
            ),
        }
    )


def _budget_reservation_amount(
    *,
    limit: BudgetLimit,
    provider_name: str,
    model: str,
    effective_at: datetime,
    billing_identity: BillingIdentity | None = None,
) -> Decimal:
    if limit.reservation is None:
        raise ValueError("Budget limit does not define a reservation policy.")
    reservation = limit.reservation
    reserved_input_tokens = (
        reservation.max_input_tokens
        + reservation.max_cache_read_input_tokens
        + reservation.max_cache_write_input_tokens
    )
    amounts: list[Decimal] = []
    for pricing_context in _budget_pricing_contexts(billing_identity):
        price = budget_price(
            limit,
            provider_name=provider_name,
            model=model,
            input_tokens=reserved_input_tokens,
            effective_at=effective_at,
            billing_identity=billing_identity,
            pricing_context=pricing_context,
        )
        if price is None:
            raise ValueError(f"Budget reservation cannot be priced for {provider_name}/{model}.")
        if price.currency.upper() != limit.currency.upper():
            raise ValueError(
                f"Budget reservation currency {price.currency} does not match {limit.currency}."
            )
        cache_read_price = (
            price.input_per_million
            if price.cache_read_input_per_million is None
            else price.cache_read_input_per_million
        )
        if not price.requires_cache_write_ttls:
            cache_write_price = (
                price.cache_write_input_per_million
                if price.cache_write_input_per_million is not None
                else price.input_per_million
            )
        else:
            supported_rates = {
                "5m": price.cache_write_5m_per_million,
                "1h": price.cache_write_1h_per_million,
            }
            cache_write_rates = [supported_rates[ttl] for ttl in price.price.cache_write_ttls]
            if reservation.max_cache_write_input_tokens and (
                not cache_write_rates or any(rate is None for rate in cache_write_rates)
            ):
                raise ValueError("Budget reservation cannot price the declared cache-write TTLs.")
            cache_write_price = max(
                (rate for rate in cache_write_rates if rate is not None),
                default=Decimal("0"),
            )
        amounts.append(
            _token_cost(reservation.max_input_tokens, price.input_per_million)
            + _token_cost(reservation.max_output_tokens, price.output_per_million)
            + _token_cost(reservation.max_cache_read_input_tokens, cache_read_price)
            + _token_cost(reservation.max_cache_write_input_tokens, cache_write_price)
        )
    return max(amounts)


def _budget_pricing_contexts(
    billing_identity: BillingIdentity | None,
) -> tuple[PricingContext | None, ...]:
    """Return every pricing outcome established before provider dispatch."""

    if billing_identity is None:
        return (None,)
    return billing_identity.pricing_contexts or (None,)


def _token_cost(tokens: int, price_per_million: Decimal) -> Decimal:
    return Decimal(tokens) * price_per_million / _TOKENS_PER_MILLION


def budget_price(
    limit: BudgetLimit,
    *,
    provider_name: str | None,
    model: str | None,
    input_tokens: int = 0,
    effective_at: datetime | None = None,
    billing_identity: BillingIdentity | None = None,
    pricing_context: PricingContext | None = None,
) -> _ResolvedPrice | None:
    effective_at = (
        datetime.now(UTC) if effective_at is None else _utc_datetime(effective_at, "effective_at")
    )
    return resolve_price_book(
        limit.pricing,
        provider_name=provider_name,
        model=model,
        input_tokens=input_tokens,
        effective_on=effective_at.astimezone(UTC).date(),
        billing_identity=billing_identity,
        pricing_context=pricing_context,
    )


def _ledger_used_amount(
    records: Iterable[BudgetReservationRecord],
    *,
    limit: _EffectiveBudgetLimit,
    now: datetime | None = None,
) -> Decimal:
    total = Decimal("0")
    since, until = limit.window.bounds(now=now)
    for record in records:
        if not _reservation_matches_limit(record, limit):
            continue
        if record.status == "active":
            total += record.reserved_amount
        elif record.status == "reconciled":
            if since is not None and record.updated_at < since:
                continue
            if until is not None and record.updated_at >= until:
                continue
            total += record.actual_amount or Decimal("0")
    return total


def _reservation_matches_limit(record: BudgetReservationRecord, limit: BudgetLimit) -> bool:
    if type(limit) is not _EffectiveBudgetLimit:
        raise TypeError("Reservation matching requires an effective budget limit.")
    return record.budget_limit_id == limit.budget_limit_id


def _validate_reservation_ttl(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError("reservation_ttl_seconds must be a positive integer or None.")
    return value


def _validate_reservation_id_batch(values: tuple[str, ...]) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise TypeError("reservation_ids must be a tuple.")
    reservation_ids = tuple(require_clean_nonblank(value, "reservation_id") for value in values)
    if len(set(reservation_ids)) != len(reservation_ids):
        raise ValueError("reservation_ids must be distinct.")
    return reservation_ids


def _validate_settlement_page_limit(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 1000:
        raise ValueError("limit must be between 1 and 1000.")
    return value


def _copy_budget_settlement_cursor(
    value: BudgetSettlementCursor | None,
) -> BudgetSettlementCursor | None:
    if value is None:
        return None
    if type(value) is not BudgetSettlementCursor:
        raise TypeError("after must be a BudgetSettlementCursor.")
    return BudgetSettlementCursor.model_validate(value.model_dump(mode="python"))


def _reservation_is_expired(
    record: BudgetReservationRecord,
    *,
    now: datetime,
    ttl_seconds: int | None,
) -> bool:
    if ttl_seconds is None:
        return False
    cutoff = now - timedelta(seconds=ttl_seconds)
    return record.updated_at <= cutoff


_EXPIRED_RESERVATION_REASON_PREFIX = "Reservation expired:"


def _expired_reservation_reason(ttl_seconds: int) -> str:
    return f"{_EXPIRED_RESERVATION_REASON_PREFIX} not reconciled within {ttl_seconds}s."


def _is_expired_reservation_reason(reason: str | None) -> bool:
    """True when a released reservation was reaped by the TTL (vs. explicitly released)."""
    return reason is not None and reason.startswith(_EXPIRED_RESERVATION_REASON_PREFIX)


def _reservation_result(
    *,
    limit: _EffectiveBudgetLimit,
    model_attempt_identity: ModelAttemptIdentity,
    accepted: bool,
    requested: Decimal,
    actual: Decimal,
    message: str,
    record: BudgetReservationRecord | None = None,
) -> BudgetReservationResult:
    model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
    return BudgetReservationResult(
        accepted=accepted,
        budget_limit_id=limit.budget_limit_id,
        model_step_id=model_attempt_identity.model_step_id,
        model_attempt_id=model_attempt_identity.model_attempt_id,
        scope=limit.scope,
        key=limit.key,
        window=limit.window,
        currency=limit.currency,
        maximum=limit.max_estimated_cost,
        action=limit.action,
        requested=requested,
        actual=actual,
        message=message,
        record=record,
    )


def _reconciled_record(
    record: BudgetReservationRecord,
    *,
    actual_amount: Decimal,
    reason: str | None,
    updated_at: datetime | None = None,
    billing_identity: BillingIdentity | None = None,
) -> BudgetReservationRecord:
    resolved_identity = _reconciled_billing_identity(record.billing_identity, billing_identity)
    if record.status == "reconciled":
        if (
            record.actual_amount == actual_amount
            and record.reason == reason
            and record.billing_identity == resolved_identity
        ):
            return record
        raise ValueError(
            f"Budget reservation has a conflicting reconciliation: {record.reservation_id}"
        )
    if record.status != "active":
        raise ValueError(f"Budget reservation is not active: {record.reservation_id}")
    reconciled_at = (
        _utc_datetime(updated_at, "updated_at") if updated_at is not None else datetime.now(UTC)
    )
    return record.model_copy(
        update={
            "actual_amount": actual_amount,
            "status": "reconciled",
            "reason": reason,
            "updated_at": reconciled_at,
            "billing_identity": resolved_identity,
        },
        deep=True,
    )


def _released_record(
    record: BudgetReservationRecord,
    *,
    reason: str,
    updated_at: datetime | None = None,
) -> BudgetReservationRecord:
    if record.status == "released":
        if record.reason == reason or _is_expired_reservation_reason(record.reason):
            return record
        raise ValueError(f"Budget reservation has a conflicting release: {record.reservation_id}")
    if record.status != "active":
        raise ValueError(f"Budget reservation is not active: {record.reservation_id}")
    released_at = (
        _utc_datetime(updated_at, "updated_at") if updated_at is not None else datetime.now(UTC)
    )
    return record.model_copy(
        update={
            "status": "released",
            "reason": reason,
            "updated_at": released_at,
        },
        deep=True,
    )


def _reconciliation_from_record(
    record: BudgetReservationRecord,
    *,
    settlement_kind: BudgetSettlementKind,
    pricing: BudgetReconciliationPricing | None = None,
) -> BudgetReconciliation:
    if record.status == "active":
        raise ValueError("Cannot construct a settlement from an active reservation.")
    if settlement_kind == "released" and record.status != "released":
        raise ValueError("Released settlement kind requires a released reservation.")
    if settlement_kind != "released" and record.status != "reconciled":
        raise ValueError("Reconciliation settlement kind requires a reconciled reservation.")
    actual_amount = record.actual_amount
    released_amount = Decimal("0")
    if record.status == "released":
        released_amount = record.reserved_amount
    elif actual_amount is not None and record.reserved_amount > actual_amount:
        released_amount = record.reserved_amount - actual_amount
    reconciliation = BudgetReconciliation(
        reservation_id=record.reservation_id,
        settlement_id=budget_settlement_id(record.reservation_id),
        settlement_kind=settlement_kind,
        budget_limit_id=record.budget_limit_id,
        model_step_id=record.model_step_id,
        model_attempt_id=record.model_attempt_id,
        status=record.status,
        reserved_amount=record.reserved_amount,
        actual_amount=actual_amount,
        released_amount=released_amount,
        reason=record.reason,
        settled_at=record.updated_at,
        billing_identity=record.billing_identity,
    )
    if pricing is None:
        return reconciliation
    return budget_reconciliation_with_pricing_evidence(reconciliation, pricing)


def budget_reconciliation_preview(
    record: BudgetReservationRecord,
    *,
    actual_amount: Decimal,
    settlement_kind: Literal["completed", "conservative"],
    reason: str | None,
    occurred_at: datetime,
    billing_identity: BillingIdentity | None = None,
    pricing: BudgetReconciliationPricing | None = None,
) -> BudgetReconciliation:
    """Construct the exact reconciliation that a ledger commit must persist."""

    if type(record) is not BudgetReservationRecord:
        raise TypeError("record must be a BudgetReservationRecord.")
    reconciled = _reconciled_record(
        record,
        actual_amount=_validate_amount(actual_amount, "actual_amount"),
        reason=reason,
        updated_at=_utc_datetime(occurred_at, "occurred_at"),
        billing_identity=billing_identity,
    )
    return _reconciliation_from_record(
        reconciled,
        settlement_kind=settlement_kind,
        pricing=pricing,
    )


def budget_release_preview(
    record: BudgetReservationRecord,
    *,
    reason: str,
    occurred_at: datetime,
) -> BudgetReconciliation:
    """Construct the exact release that a ledger commit must persist."""

    if type(record) is not BudgetReservationRecord:
        raise TypeError("record must be a BudgetReservationRecord.")
    released = _released_record(
        record,
        reason=require_clean_nonblank(reason, "reason"),
        updated_at=_utc_datetime(occurred_at, "occurred_at"),
    )
    return _reconciliation_from_record(
        released,
        settlement_kind="released",
    )


def _budget_settlement_record(
    reservation: BudgetReservationRecord,
    reconciliation: BudgetReconciliation,
    *,
    event_published: bool = False,
) -> BudgetSettlementRecord:
    if type(reservation) is not BudgetReservationRecord:
        raise TypeError("reservation must be a BudgetReservationRecord.")
    if type(reconciliation) is not BudgetReconciliation:
        raise TypeError("reconciliation must be a BudgetReconciliation.")
    if reservation.reservation_id != reconciliation.reservation_id:
        raise ValueError("Budget settlement belongs to a different reservation.")
    payload = {
        **reservation.settlement_event_payload,
        **budget_reconciliation_payload(reconciliation),
    }
    event = Event(
        type=(
            EventType.BUDGET_RESERVATION_RELEASED
            if reconciliation.settlement_kind == "released"
            else EventType.BUDGET_RECONCILED
        ),
        id=budget_settlement_event_id(reconciliation.settlement_id),
        timestamp=reconciliation.settled_at,
        session_id=reservation.session_id,
        interaction_id=reservation.settlement_event_payload.get("interaction_id"),
        agent_name=reservation.agent_name,
        environment_name=reservation.environment_name,
        payload=payload,
    )
    return BudgetSettlementRecord(
        settlement_id=reconciliation.settlement_id,
        reservation_id=reservation.reservation_id,
        settlement_kind=reconciliation.settlement_kind,
        session_id=reservation.session_id,
        agent_name=reservation.agent_name,
        environment_name=reservation.environment_name,
        reconciliation=reconciliation,
        event=event,
        event_published=event_published,
    )


def _reconciled_billing_identity(
    requested: BillingIdentity | None,
    completed: BillingIdentity | None,
) -> BillingIdentity | None:
    if completed is None:
        return requested
    return completed_billing_identity(requested, completed)


def _budget_settlement_timestamp(value: datetime) -> int:
    """Encode one UTC accounting instant as exact portable Unix microseconds."""

    normalized = _utc_datetime(value, "settled_at")
    delta = normalized - _UNIX_EPOCH
    timestamp = (
        delta.days * 86_400 + delta.seconds
    ) * _MICROSECONDS_PER_SECOND + delta.microseconds
    if not MIN_DURABLE_JSON_INTEGER <= timestamp <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("settled_at exceeds the portable timestamp range.")
    return timestamp


def _budget_settlement_datetime(value: object) -> datetime:
    """Decode exact portable Unix microseconds without float conversion."""

    if type(value) is not int or not MIN_DURABLE_JSON_INTEGER <= value <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("settled_at_unix_us must be a signed 64-bit integer.")
    try:
        return _UNIX_EPOCH + timedelta(microseconds=value)
    except OverflowError:
        raise ValueError("settled_at_unix_us is outside the supported datetime range.") from None


def _validate_amount(value: Decimal, field_name: str) -> Decimal:
    if type(value) is not Decimal:
        value = Decimal(str(value))
    if not value.is_finite() or value < 0:
        raise ValueError(f"{field_name} must be a finite non-negative Decimal.")
    return value


def _budget_window_from_string(value: str) -> BudgetWindow:
    text = require_clean_nonblank(value, "window")
    if text == _ALL_TIME_WINDOW:
        return BudgetWindow.all_time()
    if text.startswith(_ROLLING_PREFIX) and text.endswith(_ROLLING_SUFFIX):
        raw_seconds = text[len(_ROLLING_PREFIX) : -len(_ROLLING_SUFFIX)]
        try:
            seconds = int(raw_seconds)
        except ValueError as exc:
            raise ValueError(f"Invalid rolling budget window: {text}") from exc
        return BudgetWindow.rolling(seconds=seconds)
    if text.startswith(_CALENDAR_PREFIX):
        parts = text.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid calendar budget window: {text}")
        return BudgetWindow.calendar(
            period=_calendar_period(parts[1]),
            timezone=parts[2],
        )
    raise ValueError(f"Unsupported budget window: {text}")


def _utc_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


def _clock_or_utc_now(clock: Callable[[], datetime] | None) -> Callable[[], datetime]:
    if clock is None:
        return lambda: datetime.now(UTC)
    if not callable(clock):
        raise TypeError("clock must be callable.")

    def _checked_clock() -> datetime:
        return _utc_datetime(clock(), "clock()")

    return _checked_clock


def _event_in_window(
    event: Event,
    *,
    since: datetime | None,
    until: datetime | None,
) -> bool:
    timestamp = _utc_datetime(event.timestamp, "event.timestamp")
    if since is not None and timestamp < since:
        return False
    return not (until is not None and timestamp >= until)


def _calendar_period(value: str) -> BudgetCalendarPeriod:
    if value in {"day", "week", "month"}:
        return cast("BudgetCalendarPeriod", value)
    raise ValueError(f"Unsupported calendar budget period: {value}")


def _calendar_window_bounds(
    now: datetime,
    *,
    period: BudgetCalendarPeriod,
    timezone: str,
) -> tuple[datetime, datetime]:
    zone = ZoneInfo(timezone)
    local_now = now.astimezone(zone)
    if period == "day":
        local_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        local_end = local_start + timedelta(days=1)
    elif period == "week":
        local_day_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        local_start = local_day_start - timedelta(days=local_now.weekday())
        local_end = local_start + timedelta(days=7)
    elif period == "month":
        local_start = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if local_start.month == 12:
            local_end = local_start.replace(year=local_start.year + 1, month=1)
        else:
            local_end = local_start.replace(month=local_start.month + 1)
    else:
        raise ValueError(f"Unsupported calendar budget period: {period}")
    return local_start.astimezone(UTC), local_end.astimezone(UTC)
