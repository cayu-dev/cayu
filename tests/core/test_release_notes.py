from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).parents[2]
_EXTRACTOR = _REPOSITORY_ROOT / "scripts" / "extract_release_notes.py"


def test_release_note_extractor_writes_only_the_exact_tag_section(tmp_path: Path) -> None:
    notes = tmp_path / "release-notes.md"
    output = tmp_path / "selected.md"
    notes.write_text(
        "# Release notes\n\n"
        "## v0.2.0\n\n"
        "Selected release.\n\n"
        "### Upgrade\n\n"
        "Use the new version.\n\n"
        "## v0.1.0\n\n"
        "Older release.\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(_EXTRACTOR),
            "--notes",
            str(notes),
            "--version",
            "v0.2.0",
            "--output",
            str(output),
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text() == ("Selected release.\n\n### Upgrade\n\nUse the new version.\n")


@pytest.mark.parametrize(
    ("document", "expected_error"),
    [
        ("# Release notes\n\n## v0.1.0\n\nOlder.\n", "do not contain"),
        (
            "# Release notes\n\n## v0.2.0\n\nFirst.\n\n## v0.2.0\n\nSecond.\n",
            "contain duplicate",
        ),
    ],
)
def test_release_note_extractor_fails_closed_without_one_exact_tag_section(
    tmp_path: Path,
    document: str,
    expected_error: str,
) -> None:
    notes = tmp_path / "release-notes.md"
    output = tmp_path / "selected.md"
    notes.write_text(document)

    result = subprocess.run(
        [
            sys.executable,
            str(_EXTRACTOR),
            "--notes",
            str(notes),
            "--version",
            "v0.2.0",
            "--output",
            str(output),
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr
    assert not output.exists()


def test_release_note_extractor_rejects_an_empty_tag_section(tmp_path: Path) -> None:
    notes = tmp_path / "release-notes.md"
    output = tmp_path / "selected.md"
    notes.write_text("# Release notes\n\n## v0.2.0\n\n## v0.1.0\n\nOlder.\n")

    result = subprocess.run(
        [
            sys.executable,
            str(_EXTRACTOR),
            "--notes",
            str(notes),
            "--version",
            "v0.2.0",
            "--output",
            str(output),
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "is empty" in result.stderr
    assert not output.exists()
