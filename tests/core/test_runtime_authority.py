from __future__ import annotations

import pytest

from cayu.core.runtime_authority import (
    CheckpointValueAuthority,
    checkpoint_value_authority,
)


def test_checkpoint_value_authority_is_canonical_and_content_bound() -> None:
    first = checkpoint_value_authority(
        {"profile": {"fingerprint": "a"}, "run_epoch": 7},
        "active_invocation_profile",
    )
    reordered = checkpoint_value_authority(
        {"run_epoch": 7, "profile": {"fingerprint": "a"}},
        "active_invocation_profile",
    )
    changed = checkpoint_value_authority(
        {"profile": {"fingerprint": "a"}, "run_epoch": 8},
        "active_invocation_profile",
    )

    assert first == reordered
    assert first != changed


@pytest.mark.parametrize("digest", ["", "A" * 64, "0" * 63, "g" * 64])
def test_checkpoint_value_authority_rejects_noncanonical_digest(digest: str) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        CheckpointValueAuthority(sha256=digest)
