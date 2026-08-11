from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from cayu.workflows._step_identity import (
    GATED_LOOP_STEP_ID_VERSION,
    gated_loop_step_id,
    upgraded_legacy_gated_loop_step_id,
    validated_modern_gated_loop_step_id,
)

_DURABLE_TEXT = st.text(
    alphabet=st.characters(blacklist_characters="\x00", blacklist_categories=("Cs",)),
    max_size=80,
)


@settings(max_examples=250)
@given(
    left=st.tuples(_DURABLE_TEXT, _DURABLE_TEXT),
    right=st.tuples(_DURABLE_TEXT, _DURABLE_TEXT),
)
def test_gated_loop_v2_identity_separates_distinct_source_tuples(
    left: tuple[str, str],
    right: tuple[str, str],
) -> None:
    assume(left != right)

    assert gated_loop_step_id(*left) != gated_loop_step_id(*right)


def test_gated_loop_v2_identity_is_bounded_for_long_unicode_sources() -> None:
    identities = {
        gated_loop_step_id("", ""),
        gated_loop_step_id("a", "b:c"),
        gated_loop_step_id("a:b", "c"),
        gated_loop_step_id("循环:🚀", "键:值"),
        gated_loop_step_id("x" * 100_000, "y" * 100_000),
    }

    assert len(identities) == 5
    assert all(identity.startswith("gated-loop:v2:") for identity in identities)
    assert {len(identity.encode("utf-8")) for identity in identities} == {78}


def test_legacy_gated_loop_identity_upgrades_canonical_historical_tuple() -> None:
    assert upgraded_legacy_gated_loop_step_id(
        "gated-loop:a:b:c",
        "b:c",
        kind="gated_loop",
    ) == gated_loop_step_id("a", "b:c")


@pytest.mark.parametrize("kind", [None, True, "", "ordinary"])
def test_legacy_gated_loop_identity_requires_exact_historical_kind(kind: object) -> None:
    assert (
        upgraded_legacy_gated_loop_step_id(
            "gated-loop:loop:item",
            "item",
            kind=kind,
        )
        is None
    )


@pytest.mark.parametrize(
    ("step_id", "item_key"),
    [
        (None, "item"),
        ("gated-loop:loop:item", None),
        ("gated-loop:loop:item", True),
        ("gated-loop:loop:item", 1),
        ("gated-loop:loop:item", []),
        ("gated-loop:loop:item", {}),
        ("gated-loop:loop:", ""),
        ("gated-loop:loop: ", " "),
        ("gated-loop:loop: item", " item"),
        ("gated-loop:loop:item ", "item "),
        ("gated-loop: loop:item", "item"),
        ("gated-loop:loop :item", "item"),
        ("gated-loop:loop:other", "item"),
        ("gated-loop:loop:\x00", "\x00"),
        ("gated-loop:loop:\ud800", "\ud800"),
    ],
)
def test_legacy_gated_loop_identity_rejects_noncanonical_historical_tuple(
    step_id: object,
    item_key: object,
) -> None:
    assert (
        upgraded_legacy_gated_loop_step_id(
            step_id,
            item_key,
            kind="gated_loop",
        )
        is None
    )


def test_modern_gated_loop_identity_accepts_coherent_current_evidence() -> None:
    step_id = gated_loop_step_id("loop", "item")

    assert (
        validated_modern_gated_loop_step_id(
            step_id,
            "item",
            kind="gated_loop",
            loop_name="loop",
            step_id_version=GATED_LOOP_STEP_ID_VERSION,
        )
        == step_id
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"step_id": "gated-loop:v2:not-the-canonical-hash"},
        {"item_key": "other"},
        {"kind": None},
        {"kind": True},
        {"kind": "ordinary"},
        {"loop_name": None},
        {"loop_name": "other"},
        {"step_id_version": None},
        {"step_id_version": True},
        {"step_id_version": GATED_LOOP_STEP_ID_VERSION + 1},
    ],
)
def test_modern_gated_loop_identity_rejects_incoherent_evidence(
    overrides: dict[str, object],
) -> None:
    evidence: dict[str, object] = {
        "step_id": gated_loop_step_id("loop", "item"),
        "item_key": "item",
        "kind": "gated_loop",
        "loop_name": "loop",
        "step_id_version": GATED_LOOP_STEP_ID_VERSION,
    }
    evidence.update(overrides)

    assert (
        validated_modern_gated_loop_step_id(
            evidence["step_id"],
            evidence["item_key"],
            kind=evidence["kind"],
            loop_name=evidence["loop_name"],
            step_id_version=evidence["step_id_version"],
        )
        is None
    )
