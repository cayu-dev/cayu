"""Bounded inventory inputs for registered-owner accounting reconciliation."""

from cayu._validation import MAX_DURABLE_JSON_INTEGER, require_clean_nonblank


def reservation_scan_bounds(after: str | None, limit: int) -> tuple[str | None, int]:
    if after is not None:
        if type(after) is not str:
            raise TypeError("Reservation scan cursor must be text.")
        after = require_clean_nonblank(after, "reservation scan cursor")
    if type(limit) is not int or not 1 <= limit <= 128:
        raise ValueError("Reservation scan page size must be between 1 and 128.")
    return after, limit


def binding_read_expectation(
    binding_id: object, authority_digest: object, allowance: object
) -> tuple[str, str, int]:
    if type(binding_id) is not str or type(authority_digest) is not str:
        raise TypeError("Budget binding expectation requires text identities.")
    binding_id = require_clean_nonblank(binding_id, "binding_id")
    if len(authority_digest) != 64 or any(c not in "0123456789abcdef" for c in authority_digest):
        raise ValueError("Budget binding expectation requires a SHA-256 digest.")
    if type(allowance) is not int or not 0 < allowance <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Budget binding expectation requires a bounded allowance.")
    return binding_id, authority_digest, allowance


def require_binding_readback(actual, expected):
    from cayu.budgets.base import BudgetBindingRegistrationConflict

    if actual is None:
        raise LookupError("Original budget binding registration is unavailable.")
    values = tuple(actual)
    if (
        len(values) != 2
        or type(values[0]) is not str
        or type(values[1]) is not int
        or values != expected
    ):
        raise BudgetBindingRegistrationConflict("Original budget binding registration conflicts.")
