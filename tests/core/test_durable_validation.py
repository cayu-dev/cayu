from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    MAX_DURABLE_JSON_NESTING,
    MIN_DURABLE_JSON_INTEGER,
    DurableValueError,
    JsonUtf8SizeCounter,
    canonical_durable_json_bytes,
    copy_durable_json_value,
    json_utf8_size_within_limit,
)

_SCALAR_TEXT = st.text(
    st.characters(exclude_categories=("Cs",), exclude_characters="\x00"),
    max_size=24,
)
_DURABLE_FLOATS = st.floats(allow_nan=False, allow_infinity=False, width=64).filter(
    lambda value: (
        not value.is_integer() or MIN_DURABLE_JSON_INTEGER <= value <= MAX_DURABLE_JSON_INTEGER
    )
)
_DURABLE_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(MIN_DURABLE_JSON_INTEGER, MAX_DURABLE_JSON_INTEGER),
    _DURABLE_FLOATS,
    _SCALAR_TEXT,
)
_DURABLE_VALUES = st.recursive(
    _DURABLE_SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(_SCALAR_TEXT, children, max_size=5),
    ),
    max_leaves=30,
)


def test_json_utf8_size_counter_supports_dates() -> None:
    value = date(2026, 7, 1)

    assert json_utf8_size_within_limit(value, 12)
    assert not json_utf8_size_within_limit(value, 11)


def test_json_utf8_size_counter_supports_pydantic_decimal_serialization() -> None:
    value = Decimal("12.50")

    assert json_utf8_size_within_limit(value, 7)
    assert not json_utf8_size_within_limit(value, 6)


def test_json_utf8_size_counter_distinguishes_overflow_from_unsupported_values() -> None:
    overflow = JsonUtf8SizeCounter(1)
    assert overflow.value("value") is False
    assert overflow.exceeded_limit is True
    assert overflow.encountered_unsupported_value is False

    unsupported = JsonUtf8SizeCounter(1024)
    assert unsupported.value(object()) is False
    assert unsupported.exceeded_limit is False
    assert unsupported.encountered_unsupported_value is True


@settings(max_examples=250, deadline=None)
@given(_DURABLE_VALUES)
def test_durable_values_round_trip_portably_and_are_defensively_copied(value: Any) -> None:
    source = deepcopy(value)

    copied = copy_durable_json_value(source, "payload")
    encoded = json.dumps(
        copied,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    assert json.loads(encoded) == copied
    assert json.loads(canonical_durable_json_bytes(source, "payload")) == copied
    if type(source) is list:
        source.append("mutated")
        assert copied != source
    elif type(source) is dict:
        source["mutation_probe"] = "mutated"
        assert copied != source


@settings(max_examples=80, deadline=None)
@given(
    key=_SCALAR_TEXT.filter(lambda value: value != "stable"),
    valid_prefix=st.lists(_DURABLE_SCALARS, max_size=5),
    invalid=st.sampled_from(
        (
            ("nul_character", "workload-secret\x00value"),
            ("unicode_surrogate", "workload-secret\ud800value"),
            ("non_finite_number", float("nan")),
            ("non_finite_number", float("inf")),
            ("integer_out_of_range", MAX_DURABLE_JSON_INTEGER + 1),
            ("integer_out_of_range", MIN_DURABLE_JSON_INTEGER - 1),
            ("invalid_json_type", b"workload-secret"),
        )
    ),
)
def test_nested_nonportable_values_fail_with_stable_code_and_position(
    key: str,
    valid_prefix: list[Any],
    invalid: tuple[str, Any],
) -> None:
    expected_code, rejected = invalid
    payload = {"stable": valid_prefix, key: [rejected]}

    with pytest.raises(DurableValueError) as raised:
        copy_durable_json_value(payload, "payload")

    assert raised.value.code == expected_code
    assert raised.value.path == "$/#1/0"
    assert "workload-secret" not in str(raised.value)


@settings(max_examples=80, deadline=None)
@given(
    prefix=_SCALAR_TEXT,
    suffix=_SCALAR_TEXT,
    marker=st.sampled_from((("nul_character", "\x00"), ("unicode_surrogate", "\ud800"))),
)
def test_nonportable_object_keys_fail_without_echoing_the_key(
    prefix: str,
    suffix: str,
    marker: tuple[str, str],
) -> None:
    expected_code, invalid_character = marker
    rejected_key = f"{prefix}workload-secret{invalid_character}{suffix}"

    with pytest.raises(DurableValueError) as raised:
        copy_durable_json_value({rejected_key: "value"}, "payload")

    assert raised.value.code == expected_code
    assert raised.value.path == "$/#0/key"
    assert "workload-secret" not in str(raised.value)


def test_durable_number_and_nesting_boundaries_are_exact() -> None:
    largest_integral_float = math.nextafter(float(2**63), 0.0)
    smallest_integral_float = float(MIN_DURABLE_JSON_INTEGER)
    accepted = [
        MIN_DURABLE_JSON_INTEGER,
        MAX_DURABLE_JSON_INTEGER,
        smallest_integral_float,
        largest_integral_float,
    ]
    assert copy_durable_json_value(accepted, "payload") == [
        MIN_DURABLE_JSON_INTEGER,
        MAX_DURABLE_JSON_INTEGER,
        int(smallest_integral_float),
        int(largest_integral_float),
    ]

    rejected_numbers = (
        (MIN_DURABLE_JSON_INTEGER - 1, "integer_out_of_range"),
        (MAX_DURABLE_JSON_INTEGER + 1, "integer_out_of_range"),
        (math.nextafter(float(MIN_DURABLE_JSON_INTEGER), -math.inf), "integral_float_out_of_range"),
        (float(2**63), "integral_float_out_of_range"),
    )
    for value, expected_code in rejected_numbers:
        with pytest.raises(DurableValueError) as raised:
            copy_durable_json_value(value, "payload")
        assert raised.value.code == expected_code

    within_limit: Any = "leaf"
    for _ in range(MAX_DURABLE_JSON_NESTING):
        within_limit = [within_limit]
    assert copy_durable_json_value(within_limit, "payload") == within_limit

    beyond_limit = [within_limit]
    with pytest.raises(DurableValueError) as raised:
        copy_durable_json_value(beyond_limit, "payload")
    assert raised.value.code == "nesting_too_deep"
