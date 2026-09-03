from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import cayu.cli._cloud_private_state as cloud_private_state
import cayu.cli._guarded_tree_publication as guarded_publication
from cayu.support_bundles import (
    SupportBundleOutcome,
    _open_private_windows_file,
    encode_support_bundle,
    minimal_support_bundle_report,
    validate_support_bundle_archive,
    write_support_bundle_atomic,
)


def _archive() -> bytes:
    return encode_support_bundle(
        minimal_support_bundle_report(
            outcome=SupportBundleOutcome.BOOT_FAILED,
            reason_code="windows_publication_test",
        )
    )


@pytest.mark.skipif(os.name != "nt", reason="requires Windows DACL inspection")
def test_windows_support_bundle_publication_uses_protected_private_dacl(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "support.zip"
    destination.write_bytes(b"replace-me")

    write_support_bundle_atomic(destination, _archive())

    validate_support_bundle_archive(destination.read_bytes())
    dacl_present, dacl_protected = guarded_publication._windows_directory_dacl_state(destination)
    assert dacl_present is True
    assert dacl_protected is True
    assert list(tmp_path.glob(".support.zip.cayu-doctor-*.tmp")) == []


@pytest.mark.skipif(os.name != "nt", reason="requires a real Windows directory junction")
def test_windows_support_bundle_publication_refuses_junction_parent(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    junction = tmp_path / "junction"
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/j", str(junction), str(target)],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"directory junctions are unavailable: {result.stderr.strip()}")
    try:
        with pytest.raises(OSError, match="secure Windows support bundle publication failed"):
            write_support_bundle_atomic(junction / "support.zip", _archive())
        assert not (target / "support.zip").exists()
    finally:
        junction.rmdir()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows file identity semantics")
def test_windows_support_bundle_publication_rejects_staging_substitution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = _archive()
    destination = tmp_path / "support.zip"
    displaced = tmp_path / ".displaced-support-bundle"
    original_move = cloud_private_state._move_file_ex_windows

    def substitute_staging(source: Path, target: Path, *, flags: int) -> None:
        original_move(source, displaced, flags=flags)
        replacement = _open_private_windows_file(source)
        try:
            garbage = b"x" * len(payload)
            offset = 0
            while offset < len(garbage):
                offset += os.write(replacement, garbage[offset:])
            os.fsync(replacement)
        finally:
            os.close(replacement)
        original_move(source, target, flags=flags)

    monkeypatch.setattr(cloud_private_state, "_move_file_ex_windows", substitute_staging)

    try:
        with pytest.raises(OSError, match="identity"):
            write_support_bundle_atomic(destination, payload)
    finally:
        displaced.unlink(missing_ok=True)

    assert destination.read_bytes() != payload
