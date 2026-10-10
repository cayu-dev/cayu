from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from examples.aws import lambda_microvm_dmtcp_snapshot_probe as probe
from examples.execution_snapshot_probe import tree_hashes, write_private


@pytest.mark.parametrize(
    ("images", "temporary", "status", "expected"),
    [
        ([], [], "NUM_PEERS=1\nRUNNING=yes", False),
        (["process.dmtcp"], ["process.dmtcp.temp"], "NUM_PEERS=1\nRUNNING=yes", False),
        (["process.dmtcp"], [], "NUM_PEERS=1\nRUNNING=no", False),
        (["process.dmtcp"], [], "NUM_PEERS=0\nRUNNING=yes", False),
        (["one.dmtcp", "two.dmtcp"], [], "NUM_PEERS=2\nRUNNING=yes", False),
        (["process.dmtcp"], [], "  NUM_PEERS=1\n  RUNNING=yes", True),
    ],
)
def test_checkpoint_acknowledgement_requires_complete_artifacts_and_coordinator(
    images: list[str], temporary: list[str], status: str, expected: bool
) -> None:
    assert probe.checkpoint_ready({"images": images, "temporary": temporary}, status) is expected


@pytest.mark.parametrize("damage", ["corrupt", "deleted", "missing_support_file"])
def test_invalid_checkpoint_cannot_submit_an_aws_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    image = artifact / "process.dmtcp"
    image.write_bytes(b"memory")
    support = artifact / "cache"
    support.write_bytes(b"required mapped file")
    write_private(tmp_path / "state.json", {"hashes": tree_hashes(artifact)})
    if damage == "corrupt":
        image.write_bytes(b"changed")
    elif damage == "deleted":
        shutil.rmtree(artifact)
    else:
        support.unlink()

    def forbidden_client(*_args: object) -> None:
        pytest.fail("invalid artifact reached AWS allocation submission")

    monkeypatch.setattr(probe, "client", forbidden_client)
    with pytest.raises(RuntimeError, match="artifact"):
        asyncio.run(probe.worker(tmp_path, "restore"))
    assert not (tmp_path / "restore").exists()
