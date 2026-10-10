"""The guest's anti-reflection matcher answers exactly like ``value in text``."""

from __future__ import annotations

import random
import time

import pytest

from cayu.tools import _browser_guest

# Keep timing assertions tied to a fixed workload, not the transport envelope.
_BENCHMARK_SNAPSHOT_BYTES = 256 * 1024


def _naive(values: tuple[str, ...], texts: tuple[str, ...]) -> bool:
    return any(value in text for value in values for text in texts)


def _random_text(rng: random.Random, alphabet: str, size: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(size))


@pytest.mark.parametrize("plain_scan_budget", [0, _browser_guest._PLAIN_SCAN_BUDGET])
def test_matcher_agrees_with_the_plain_scan_on_random_inputs(
    monkeypatch: pytest.MonkeyPatch, plain_scan_budget: int
) -> None:
    # Budget 0 forces the indexed path even for small inputs.
    monkeypatch.setattr(_browser_guest, "_PLAIN_SCAN_BUDGET", plain_scan_budget)
    rng = random.Random(1973)
    for _case in range(400):
        alphabet = rng.choice(["ab", "abc", "abcdefgh", "a\x00b"])
        texts = tuple(
            _random_text(rng, alphabet, rng.randint(0, 600)) for _ in range(rng.randint(0, 4))
        )
        values: set[str] = set()
        for _ in range(rng.randint(0, 60)):
            kind = rng.random()
            if kind < 0.2 and texts and any(texts):
                text = rng.choice([text for text in texts if text])
                start = rng.randrange(len(text))
                values.add(text[start : start + rng.randint(1, 40)])
            elif kind < 0.25:
                values.add("")
            else:
                values.add(_random_text(rng, alphabet, rng.randint(1, 40)))
        ordered = tuple(sorted(values, key=lambda item: (-len(item), item)))
        matcher = _browser_guest._ProtectedValueMatcher(ordered)
        assert matcher.found_in(texts) is _naive(ordered, texts), (ordered, texts)


def test_matcher_never_matches_across_joined_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_browser_guest, "_PLAIN_SCAN_BUDGET", 0)
    matcher = _browser_guest._ProtectedValueMatcher(("abcdefghij", "xy"))
    assert matcher.found_in(("abcde", "fghij", "x", "y")) is False
    assert matcher.found_in(("zzabcdefghijzz",)) is True
    assert _browser_guest._ProtectedValueMatcher(("a\x00b",)).found_in(("a", "b")) is False


def _visible(snapshot: str) -> tuple[str, ...]:
    return ("https://app.example.test/", "Title", snapshot, *(["button", "Submit"] * 1024))


def _timed(values: list[str], texts: tuple[str, ...]) -> tuple[bool, float]:
    ordered = tuple(sorted(set(values), key=lambda item: (-len(item), item)))
    start = time.perf_counter()
    matcher = _browser_guest._ProtectedValueMatcher(ordered)
    found = matcher.found_in(texts)
    return found, time.perf_counter() - start


def test_adversarial_repeated_page_and_near_miss_cookies_are_bounded() -> None:
    # An all-'a' page and 24k distinct cookies of a run of 'a' then 'b': every
    # plain scan walks the whole page; the old nested loop took tens of seconds.
    values = [
        "a" * (1 + index % 200) + "b" + str(index)
        for index in range(_browser_guest._INTERACTIVE_MAX_PROFILE_PRIVATE_VALUES)
    ]
    found, elapsed = _timed(values, _visible("a" * _BENCHMARK_SNAPSHOT_BYTES))
    assert found is False
    assert elapsed < 2.0


def test_adversarial_short_values_are_bounded() -> None:
    rng = random.Random(7)
    values = {
        _random_text(rng, "abcdefgh", rng.randint(1, 6)) + "z"
        for _ in range(_browser_guest._INTERACTIVE_MAX_PROFILE_PRIVATE_VALUES)
    }
    found, elapsed = _timed(sorted(values), _visible("a" * _BENCHMARK_SNAPSHOT_BYTES))
    assert found is False
    assert elapsed < 2.0


def test_adversarial_runs_around_one_foreign_character_are_bounded() -> None:
    # 2,048 values like 'aaaaaaaabaaaaaaaa' against an all-'a' page: each plain
    # scan walks the page, which took 0.7-1.8 s on the plain path.
    values = ["a" * (8 + index % 32) + "b" + "a" * (8 + index // 32) for index in range(2048)]
    found, elapsed = _timed(values, _visible("a" * _BENCHMARK_SNAPSHOT_BYTES))
    assert found is False
    assert elapsed < 1.0


def test_small_profiles_keep_the_plain_scan() -> None:
    rng = random.Random(11)
    values = tuple(_random_text(rng, "abcdefghijklmnop", 40) for _ in range(20))
    matcher = _browser_guest._ProtectedValueMatcher(values)
    assert matcher.found_in(_visible(_random_text(rng, "abcdefghijklmnop ", 256 * 1024))) is False
    assert matcher._index_built is False
