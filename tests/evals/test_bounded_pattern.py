"""Linear-time patterns for portable structured-output `matches` assertions."""

from __future__ import annotations

import re
import time

import pytest

from cayu.evals._bounded_pattern import (
    BOUNDED_PATTERN_MAX_CHARS,
    BOUNDED_PATTERN_MAX_INSTRUCTIONS,
    BoundedPattern,
)


@pytest.mark.parametrize(
    ("pattern", "text"),
    (
        (r"^CVE-\d{4}-\d+$", "CVE-2026-1234"),
        (r"^CVE-\d{4}-\d+$", "CVE-26-1234"),
        (r"^CVE-\d{4}-\d+$", "CVE-2026-1234\n"),
        (r"^CVE-\d{4}-\d+$", "CVE-2026-1234\n\n"),
        (r"confirm", "unconfirmed"),
        (r"^(confirmed|refuted)$", "refuted"),
        (r"^(?:confirmed|refuted)$", "undecided"),
        (r"[^a-z]", "abc"),
        (r"[]a]+$", "x]a]"),
        (r"[a-]x", "-x"),
        (r"\w+@\w+\.com", "ops@cayu.com"),
        (r"\s\S\W\D", " a!x"),
        (r"a{2,3}b", "aab"),
        (r"a{2,3}b", "ab"),
        (r"a{2,}?b", "aaaab"),
        (r"x*", ""),
        (r"^$", ""),
        (r".", "\n"),
        (r"\.\*\(\)", ".*()"),
        (r"[\d_]{3}", "1_2"),
        (r"\t|\n", "tab\there"),
        (r"(a|ab)(c|bcd)(d*)", "abcd"),
        (r"٣", "x٣"),
        (r"\d", "٣"),
    ),
)
def test_bounded_pattern_search_agrees_with_python_re(pattern, text):
    assert BoundedPattern(pattern).search(text) is bool(re.search(pattern, text))


@pytest.mark.parametrize(
    ("pattern", "message"),
    (
        ("", "must not be empty"),
        ("a" * (BOUNDED_PATTERN_MAX_CHARS + 1), "at most 256 characters"),
        (r"(a)\1", r"escape \1 is not supported"),
        (r"\bword", r"escape \b is not supported"),
        ("(?=a)", "support only"),
        ("(?i)a", "support only"),
        ("a*+", "repeated or possessive"),
        ("a**", "repeated or possessive"),
        ("*a", "nothing to repeat"),
        ("^*", "anchors cannot be repeated"),
        ("(a", "unterminated group"),
        ("a)", "unmatched"),
        ("[a", "unterminated character class"),
        ("[z-a]", "out of order"),
        (r"[\d-z]", "shorthand"),
        ("a{2,1}", "below its minimum"),
        ("a{,3}", "{m}, {m,}, or {m,n}"),
        ("a{1000}", "at most 3 digits"),
        ("(" * 33 + "a" + ")" * 33, "nest deeper"),
        ("(a{20}){30}", f"{BOUNDED_PATTERN_MAX_INSTRUCTIONS}-instruction"),
        ("a\\", "dangling backslash"),
    ),
)
def test_bounded_pattern_rejects_constructs_outside_the_linear_subset(pattern, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        BoundedPattern(pattern)


def test_bounded_pattern_search_stays_linear_on_backtracking_patterns():
    # Python's backtracking engine needs exponential time for these inputs.
    assert BoundedPattern("(a+)+$").search("a" * 4_096 + "b") is False
    assert BoundedPattern(r"(\w+\s?)+$").search("word " * 800 + "!") is False
    assert BoundedPattern("(a|a)*c").search("a" * 4_096) is False


@pytest.mark.parametrize("pattern", ["(((){999}){999}){999}", "((a{0}){999}){999}", "(()*){999}"])
def test_repeating_an_empty_body_compiles_at_once_and_matches_the_empty_string(
    pattern: str,
) -> None:
    started = time.monotonic()
    compiled = BoundedPattern(pattern)

    assert time.monotonic() - started < 1.0
    assert compiled.search("") is True
    assert compiled.search("x") is True


@pytest.mark.parametrize(
    ("pattern", "subject"),
    [("^[]-a]$", "_"), ("^[]-a]$", "-"), ("^[]-a]$", "]"), ("^[]]$", "]"), ("^[^]-a]$", "-")],
)
def test_a_leading_bracket_in_a_class_can_start_a_range(pattern: str, subject: str) -> None:
    assert BoundedPattern(pattern).search(subject) is (re.search(pattern, subject) is not None)


def test_repeated_character_classes_charge_each_range_to_the_matching_budget():
    ranges = "".join(f"{chr(256 + i)}-{chr(256 + i)}" for i in range(77))
    with pytest.raises(ValueError, match="matching limit"):
        BoundedPattern("[" + ranges + "]{510}x")
    assert BoundedPattern("[" + ranges + "]{3}x").search(chr(332) * 3 + "x")
