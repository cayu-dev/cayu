from __future__ import annotations

import json
import math
import pickle
import re
from copy import deepcopy
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, Field

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    MAX_DURABLE_JSON_NESTING,
    MIN_DURABLE_JSON_INTEGER,
    DurableValueError,
    JsonUtf8SizeCounter,
    canonical_durable_json_bytes,
    copy_bounded_durable_json_value,
    copy_durable_json_value,
    inspect_bounded_durable_json,
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


@pytest.mark.parametrize(
    "suffix",
    ["plain", "é😀", '\n"\\\x01', "\x00", "\ud800", "x" * 256],
    ids=["plain", "unicode", "escaped", "nul", "surrogate", "uncached-length"],
)
def test_repeated_json_strings_preserve_exact_failure_evidence(suffix):
    repeated = ["a" + suffix, {"a" + suffix: "a" + suffix}]
    distinct = ["a" + suffix, {"b" + suffix: "c" + suffix}]
    # Distinct strings exercise the uncached path with identical byte/node
    # consumption. Include failures at every byte and after successful reuse.
    size = len(json.dumps(repeated, ensure_ascii=True).encode())

    def outcome(value, limit):
        try:
            inspect_bounded_durable_json(value, "record", max_bytes=limit, max_nodes=20)
        except DurableValueError as error:
            return error.code, error.path, error.limit, error.observed_lower_bound
        return None

    for limit in range(size + 2):
        assert outcome(repeated, limit) == outcome(distinct, limit)


def test_repeated_json_string_sizing_is_local_and_bounded(monkeypatch):
    from cayu import _validation

    search = _validation.re.search
    scanned = []

    def record(pattern, text, *args, **kwargs):
        scanned.append(text)
        return search(pattern, text, *args, **kwargs)

    monkeypatch.setattr(_validation.re, "search", record)
    for _ in range(2):
        scanned.clear()
        inspect_bounded_durable_json(["repeat"] * 20, "record", max_bytes=1024, max_nodes=100)
        assert scanned.count("repeat") == 1
    # Filling the cache never drops validation of later strings.
    scanned.clear()
    strings = [str(index) for index in range(129)] + ["128", "x" * 257, "x" * 257]
    inspect_bounded_durable_json(strings, "record", max_bytes=4096, max_nodes=256)
    assert scanned.count("128") == 2
    assert scanned.count("x" * 257) == 2


@pytest.mark.parametrize(
    "text",
    [
        '\n"\\\x01é😀',
        "\n" * 4097,
        '"' * 4095 + "\x00",
        '"' * 4095 + "\ud800",
        "\x01" * 4096 + "é😀",
    ],
    ids=["mixed", "newline-chunk", "nul-boundary", "surrogate-boundary", "unicode-tail"],
)
def test_escaped_chunk_sizing_preserves_scalar_failure_order(monkeypatch, text):
    from cayu import _validation

    def outcome(limit):
        try:
            return copy_bounded_durable_json_value(text, "record", max_bytes=limit, max_nodes=1)
        except DurableValueError as error:
            return error.code, error.path, error.limit, error.observed_lower_bound

    size = len(json.dumps(text, ensure_ascii=False).encode("utf-8", "surrogatepass"))
    limits = sorted({0, 1, 2, 3, 17, 4096, 8192, size - 1, size, size + 1})
    fast = [outcome(limit) for limit in limits]
    # Force every special chunk down the original ordered scalar path.
    monkeypatch.setattr(_validation, "_NONPORTABLE_STRING_RE", re.compile(""))
    assert [outcome(limit) for limit in limits] == fast


@pytest.mark.parametrize("suffix", ["\x00", "\ud800"])
@pytest.mark.parametrize("as_key", [False, True])
def test_bounded_json_stops_before_invalid_text_beyond_byte_limit(
    suffix: str, as_key: bool
) -> None:
    text = "x" * 1000 + suffix
    value = {text: None} if as_key else text
    with pytest.raises(DurableValueError) as caught:
        copy_bounded_durable_json_value(value, "record", max_bytes=32, max_nodes=10)
    assert caught.value.code == "json_value_too_large"


@pytest.mark.parametrize(
    ("text", "code"), [("\x00", "nul_character"), ("\ud800", "unicode_surrogate")]
)
def test_bounded_json_rejects_invalid_text_within_byte_limit(text: str, code: str) -> None:
    with pytest.raises(DurableValueError) as caught:
        copy_bounded_durable_json_value(text, "record", max_bytes=32, max_nodes=10)
    assert caught.value.code == code


@pytest.mark.parametrize("text", ["ascii", "é", "€", "😀", "\n", '"', "\\"])
def test_bounded_json_text_exact_utf8_limit(text: str) -> None:
    size = len(json.dumps(text, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    assert copy_bounded_durable_json_value(text, "record", max_bytes=size, max_nodes=1) == text
    with pytest.raises(DurableValueError) as caught:
        copy_bounded_durable_json_value(text, "record", max_bytes=size - 1, max_nodes=1)
    assert caught.value.code == "json_value_too_large"


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


def test_json_utf8_size_counter_honors_pydantic_serialization_exclusions() -> None:
    class Projection(BaseModel):
        retained: str
        excluded: str = Field(exclude=True)
        conditional: str | None = Field(default=None, exclude_if=lambda value: value is None)

    value = Projection(retained="visible", excluded="private")
    encoded = json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    assert json_utf8_size_within_limit(value, len(encoded))
    assert not json_utf8_size_within_limit(value, len(encoded) - 1)


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


@settings(max_examples=250, deadline=None)
@given(_DURABLE_VALUES)
def test_bounded_copy_matches_canonical_copy_and_detaches(value: Any) -> None:
    expected = copy_durable_json_value(value, "record")
    size = len(canonical_durable_json_bytes(expected, "record"))
    copied = copy_bounded_durable_json_value(value, "record", max_bytes=size, max_nodes=1000)
    assert copied == expected
    inspect_bounded_durable_json(value, "record", max_bytes=size, max_nodes=1000)
    if type(value) is list:
        value.append("mutation")
        assert copied == expected
    elif type(value) is dict:
        value["mutation"] = True
        assert copied == expected
    with pytest.raises(DurableValueError) as caught:
        copy_bounded_durable_json_value(expected, "record", max_bytes=size - 1, max_nodes=1000)
    assert caught.value.dimension == "bytes"
    assert caught.value.limit == size - 1
    assert caught.value.observed_lower_bound >= size


@pytest.mark.parametrize("is_object", [False, True])
def test_bounded_container_cardinality_reports_content_free_limits(is_object: bool) -> None:
    value = {"private-key": None, "other": None} if is_object else [None, None]
    limits = {"max_object_entries": 1} if is_object else {"max_array_entries": 1}
    with pytest.raises(DurableValueError) as caught:
        copy_bounded_durable_json_value(value, "record", max_bytes=100, max_nodes=10, **limits)
    error = pickle.loads(pickle.dumps(caught.value))
    assert error.dimension == "entries"
    assert error.limit == 1
    assert error.observed_lower_bound == 2
    assert "private-key" not in str(error)


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


@given(value=st.text(), limit=st.integers(min_value=3, max_value=512))
def test_bounded_diagnostic_label_preserves_printable_ascii_projection(value, limit):
    from cayu._validation import _bounded_ascii_label

    safe = "".join(char if 0x20 <= ord(char) <= 0x7E else "?" for char in value)
    expected = (
        "fallback" if not value else safe if len(safe) <= limit else safe[: limit - 3] + "..."
    )
    assert _bounded_ascii_label(value, limit=limit, fallback="fallback") == expected


def test_generated_paths_do_not_require_python_character_sanitization(monkeypatch):
    import cayu._validation as validation

    def unexpected(value):
        raise AssertionError("Printable ASCII diagnostic paths need no per-character Python work")

    monkeypatch.setattr(validation, "ord", unexpected, raising=False)
    path = "$"
    for i in range(64):
        path = validation._durable_child_path(path, i, object_value=bool(i % 2))
    assert path.isascii() and path.isprintable()
    assert len(path) <= validation._MAX_DURABLE_ERROR_PATH_CHARS


def test_diagnostic_label_does_not_scan_omitted_suffix(monkeypatch):
    import cayu._validation as validation

    calls = []
    original_ord = ord

    def counted(value):
        calls.append(value)
        return original_ord(value)

    monkeypatch.setattr(validation, "ord", counted, raising=False)
    assert (
        validation._bounded_ascii_label("é" * 100_000, limit=12, fallback="value")
        == "?" * 9 + "..."
    )
    assert len(calls) == 9


def _reference_json_string_body_size(text: str, *, ensure_ascii: bool) -> int:
    size = 0
    for character in text:
        codepoint = ord(character)
        if character in {'"', "\\"} or character in "\b\f\n\r\t":
            size += 2
        elif codepoint < 0x20 or (ensure_ascii and 0x7F <= codepoint < 0x10000):
            size += 6
        elif ensure_ascii and codepoint >= 0x10000:
            size += 12
        else:
            size += len(character.encode("utf-8", "surrogatepass"))
    return size


@settings(max_examples=300, deadline=None)
@given(
    text=st.text(
        alphabet=st.one_of(
            st.characters(),
            st.sampled_from(list('"\\\b\f\n\r\t\x00\x01\x1f\x7f\x80߿ࠀ\ud800\U0001f600')),
        ),
        max_size=64,
    ),
    ensure_ascii=st.booleans(),
)
def test_escaped_string_size_matches_per_character_accounting(
    text: str, ensure_ascii: bool
) -> None:
    from cayu._validation import _json_string_body_size

    assert _json_string_body_size(text, ensure_ascii=ensure_ascii) == (
        _reference_json_string_body_size(text, ensure_ascii=ensure_ascii)
    )


def test_escape_heavy_strings_keep_exact_byte_bound_errors() -> None:
    text = 'line "one"\n' * 400
    expected = copy_durable_json_value(text, "record")
    size = len(canonical_durable_json_bytes(expected, "record"))
    assert copy_bounded_durable_json_value(text, "record", max_bytes=size, max_nodes=1) == text
    with pytest.raises(DurableValueError) as caught:
        copy_bounded_durable_json_value(text, "record", max_bytes=size - 1, max_nodes=1)
    assert caught.value.code == "json_value_too_large"
    assert caught.value.observed_lower_bound == size


@pytest.mark.parametrize("character", ["\x01", "é", "😀"])
@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_json_size_counter_bounds_temporary_memory_on_oversized_strings(
    character: str, ensure_ascii: bool
) -> None:
    import tracemalloc

    text = character * 1_000_000
    counter = JsonUtf8SizeCounter(64, ensure_ascii=ensure_ascii)
    tracemalloc.start()
    try:
        assert counter.value(text) is False
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert counter.exceeded_limit
    assert peak < 1_000_000


@pytest.mark.parametrize("ensure_ascii", [False, True])
@pytest.mark.parametrize("length", [4095, 4096, 4097, 8193])
def test_json_size_counter_preserves_accounting_across_chunks(
    ensure_ascii: bool, length: int
) -> None:
    text = ('a"\n\x01é😀\ud800' * length)[:length]
    body_size = _reference_json_string_body_size(text, ensure_ascii=ensure_ascii)
    for limit in (2, 64, 4096, body_size + 1, body_size + 2):
        remaining = limit - 2
        for character in text:
            remaining -= _reference_json_string_body_size(character, ensure_ascii=ensure_ascii)
            if remaining < 0:
                break
        counter = JsonUtf8SizeCounter(limit, ensure_ascii=ensure_ascii)
        assert counter.value(text) is (remaining >= 0)
        assert counter.remaining == remaining
        assert counter.exceeded_limit is (remaining < 0)
