"""`cayu check` notes the unedited membership copies older releases generated, and only those."""

from __future__ import annotations

import hashlib
from pathlib import Path

from cayu.cli._membership_copies import GENERATED_COPIES, membership_copy_findings
from cayu.cli.scaffold import project_files
from cayu.runtime.checks import DiagnosticSeverity

RELEASED = Path(__file__).resolve().parents[1] / "fixtures" / "memberships"
COPIES = {
    "policies/access.py": RELEASED / "policies_access.py.fixture",
    "database/memberships.py": RELEASED / "database_memberships.py.fixture",
}


def _project(tmp_path: Path, files: dict[str, bytes]) -> Path:
    for path, content in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return tmp_path


def test_the_known_copies_are_the_released_files() -> None:
    for path, source in COPIES.items():
        assert hashlib.sha256(source.read_bytes()).hexdigest() in GENERATED_COPIES[path]


def test_an_unedited_generated_copy_is_noted_with_the_way_to_move(tmp_path: Path) -> None:
    root = _project(tmp_path, {path: source.read_bytes() for path, source in COPIES.items()})

    findings = membership_copy_findings(root)

    assert [item.path for item in findings] == ["policies/access.py", "database/memberships.py"]
    for item in findings:
        assert item.code == "MEMBERSHIPS_GENERATED_COPY"
        assert item.severity is DiagnosticSeverity.INFO
        assert "0.16.1 to 0.20.0" in item.message
        assert item.documentation_anchor == "cayu guide b2b#move-to-cayumemberships"


def test_an_edited_copy_and_todays_generated_files_are_left_alone(tmp_path: Path) -> None:
    edited = (
        COPIES["policies/access.py"]
        .read_bytes()
        .replace(b"MAX_MEMBERSHIPS = 16", b"MAX_MEMBERSHIPS = 8")
    )
    current = project_files("acme-desk", preset="b2b")
    root = _project(
        tmp_path,
        {
            "policies/access.py": edited,
            "database/memberships.py": current["database/memberships.py"].encode(),
        },
    )
    assert membership_copy_findings(root) == ()
    # A project without organizations has neither file.
    assert membership_copy_findings(tmp_path / "missing") == ()


def test_the_move_section_the_note_points_to_exists(capsys) -> None:
    from cayu.cli import main

    for topic in ("b2b#move-to-cayumemberships", "applications#move-to-cayumemberships"):
        assert main(["guide", topic]) == 0
        assert "no database migration" in " ".join(capsys.readouterr().out.split())
