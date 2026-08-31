from __future__ import annotations

import asyncio
import json
from hashlib import sha256
from pathlib import Path

import pytest
from tests.core.test_agent_bundle_containers import _export_directory
from tests.core.test_agent_bundles import _portable_fixture

from cayu import (
    AgentSnapshotAccess,
    AgentSnapshotCaptureRequest,
    AgentSnapshotComponentSelector,
    AgentSnapshotCoordinator,
    SQLiteAgentSnapshotStore,
)
from cayu.cli import main


def test_agent_bundle_cli_help_exposes_single_file_workflow(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["agent", "bundle", "--help"])
    assert stopped.value.code == 0
    output = capsys.readouterr().out
    for command in ("export", "inspect", "import", "unpack", "pack"):
        assert command in output
    assert ".cayu" in output


def test_agent_bundle_cli_packs_inspects_and_unpacks(tmp_path: Path, capsys) -> None:
    *_, directory, export = asyncio.run(_export_directory(tmp_path))
    container = (tmp_path / "agent.cayu").resolve()
    unpacked = (tmp_path / "agent.cayu.d").resolve()

    assert main(["agent", "bundle", "pack", str(directory), "--output", str(container)]) == 0
    packed = json.loads(capsys.readouterr().out)
    assert packed["bundle_id"] == export.bundle.bundle_id
    assert packed["mode"] == "full"

    assert main(["agent", "bundle", "inspect", str(container), "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected == packed

    assert (
        main(
            [
                "agent",
                "bundle",
                "unpack",
                str(container),
                "--destination",
                str(unpacked),
            ]
        )
        == 0
    )
    unpacked_report = json.loads(capsys.readouterr().out)
    assert unpacked_report == packed
    assert (unpacked / "index.json").read_bytes() == (directory / "index.json").read_bytes()


def test_agent_bundle_cli_governed_export_and_import_persist_and_pin_snapshot(
    tmp_path: Path,
    capsys,
) -> None:
    providers, object_store, subject, source_scope = _portable_fixture(tmp_path / "source-objects")
    source_store_path = (tmp_path / "source-snapshots.sqlite3").resolve()
    source_store = SQLiteAgentSnapshotStore(source_store_path)
    snapshot = asyncio.run(
        AgentSnapshotCoordinator(providers, store=source_store).capture(
            AgentSnapshotCaptureRequest(
                capture_request_id="cli-governed-container-capture",
                subject=subject,
                authority_scope_fingerprint=source_scope,
                components=tuple(
                    AgentSnapshotComponentSelector(kind=kind)
                    for kind in sorted((provider.kind for provider in providers), key=str)
                ),
            )
        )
    )
    container = (tmp_path / "governed-agent.cayu").resolve()
    assert (
        main(
            [
                "agent",
                "bundle",
                "export",
                "--snapshot-store",
                str(source_store_path),
                "--object-store",
                str(object_store.root),
                "--snapshot-root",
                snapshot.snapshot_root,
                "--binding-id",
                snapshot.identity_binding.binding_id,
                "--authority-scope-fingerprint",
                source_scope,
                "--output",
                str(container),
            ]
        )
        == 0
    )
    exported = json.loads(capsys.readouterr().out)
    assert exported["snapshot_root"] == snapshot.snapshot_root
    assert container.is_file()

    destination_subject = subject.model_copy(update={"agent_id": "cli-imported-agent"})
    subject_path = (tmp_path / "destination-subject.json").resolve()
    subject_path.write_text(
        json.dumps(destination_subject.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    destination_scope = sha256(b"cli-governed-destination-scope").hexdigest()
    destination_store_path = (tmp_path / "destination-snapshots.sqlite3").resolve()
    destination_objects = (tmp_path / "destination-objects").resolve()
    assert (
        main(
            [
                "agent",
                "bundle",
                "import",
                str(container),
                "--snapshot-store",
                str(destination_store_path),
                "--object-store",
                str(destination_objects),
                "--subject",
                str(subject_path),
                "--authority-scope-fingerprint",
                destination_scope,
                "--owner",
                "cli-test-owner",
            ]
        )
        == 0
    )
    imported = json.loads(capsys.readouterr().out)
    assert imported["snapshot_ref"]["snapshot_root"] == snapshot.snapshot_root
    assert imported["pin"]["binding_id"] == imported["binding_id"]
    persisted = asyncio.run(
        SQLiteAgentSnapshotStore(destination_store_path).get_snapshot(
            AgentSnapshotAccess(
                snapshot=snapshot.ref,
                binding_id=imported["binding_id"],
                authority_scope_fingerprint=destination_scope,
            )
        )
    )
    assert persisted.snapshot_root == snapshot.snapshot_root
    assert persisted.subject.agent_id == "cli-imported-agent"


def test_agent_bundle_cli_rejects_extension_only_input(tmp_path: Path, capsys) -> None:
    fake = tmp_path / "not-a-bundle.cayu"
    fake.write_bytes(b"not a zip")
    assert main(["agent", "bundle", "inspect", str(fake), "--json"]) == 2
    error = json.loads(capsys.readouterr().out)
    assert "container_" in error["error"]
