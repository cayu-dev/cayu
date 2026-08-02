from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from cayu.workflows._step_identity import gated_loop_step_id

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
