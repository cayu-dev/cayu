from __future__ import annotations

import copy
import sqlite3
import stat
from pathlib import Path

import pytest
from examples.execution_snapshot_probe import (
    FixtureAcknowledgementLost,
    continuation,
    external_fixture,
    load,
    tree_hashes,
    verify_artifact,
    verify_compatibility,
    verify_restore,
    write_private,
)


@pytest.fixture
def restore_point() -> dict[str, object]:
    return {
        "token": "memory-only-fixture",
        "counter": 17,
        "quiescent": True,
        "digests": {"payload.bin": "digest", "action.json": "receipt", "counter.json": "counter"},
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("token", "restarted-process", "memory-only"),
        ("counter", 18, "restore boundary"),
        ("quiescent", False, "quiescent"),
    ],
)
def test_changed_process_state_cannot_become_a_verified_restore(
    restore_point: dict[str, object], field: str, value: object, message: str
) -> None:
    observed = copy.deepcopy(restore_point)
    observed[field] = value
    with pytest.raises(RuntimeError, match=message):
        verify_restore(restore_point, observed, restore_point["digests"], 17)


def test_mixed_age_filesystem_and_memory_are_rejected(restore_point: dict[str, object]) -> None:
    with pytest.raises(RuntimeError, match="restore boundary"):
        verify_restore(restore_point, restore_point, restore_point["digests"], 16)


def test_corrupt_files_are_rejected_even_when_process_state_matches(
    restore_point: dict[str, object],
) -> None:
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        verify_restore(restore_point, restore_point, {"payload.bin": "changed"}, 17)


def test_unknown_external_effect_stays_a_reconciliation_barrier() -> None:
    assert continuation({"call": "fixture-write", "count": 1}, {"state": "outcome_unknown"}) == {
        "fixture-write": "reuse_recorded_result",
        "external-mutation": "reconcile",
    }
    with pytest.raises(RuntimeError, match="requires reconciliation"):
        continuation({"call": "fixture-write", "count": 2}, {"state": "outcome_unknown"})


def test_external_fixture_commits_before_losing_the_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(FixtureAcknowledgementLost):
        external_fixture(tmp_path)
    with sqlite3.connect(tmp_path / "external.db") as database:
        assert database.execute("SELECT count FROM effects").fetchall() == [(1,)]
    assert (
        continuation({"call": "fixture-write", "count": 1}, {"state": "outcome_unknown"})[
            "external-mutation"
        ]
        == "reconcile"
    )


def test_private_journal_replacement_preserves_incomplete_operation(tmp_path: Path) -> None:
    path = tmp_path / "capture.json"
    write_private(path, {"state": "intent", "active": False})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load(path) == {"state": "intent", "active": False}
    assert not path.with_suffix(".tmp").exists()
    write_private(path, {"state": "verifying", "active": False, "sandbox": "private-handle"})
    assert load(path)["active"] is False
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_corrupted_checkpoint_is_rejected_before_restoration(tmp_path: Path) -> None:
    artifact = tmp_path / "snapshot"
    artifact.mkdir()
    content = artifact / "inventory.img"
    content.write_bytes(b"valid checkpoint")
    expected = tree_hashes(artifact)
    content.write_bytes(b"corrupted checkpoint")
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        verify_artifact(artifact, expected)


def test_deleted_checkpoint_is_distinct_from_a_verified_restore(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unavailable"):
        verify_artifact(tmp_path / "deleted", {"inventory.img": "digest"})


def test_partial_checkpoint_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "snapshot"
    artifact.mkdir()
    (artifact / "inventory.img").write_bytes(b"inventory")
    expected = tree_hashes(artifact)
    expected["pages.img"] = "missing memory pages"
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        verify_artifact(artifact, expected)


@pytest.mark.parametrize("dimension", ["image", "kernel", "architecture", "criu"])
def test_incompatible_process_checkpoint_is_rejected(dimension: str) -> None:
    expected = dict.fromkeys(("image", "kernel", "architecture", "criu"), "original")
    observed = {**expected, dimension: "changed"}
    with pytest.raises(RuntimeError, match="compatibility mismatch"):
        verify_compatibility(expected, observed)
