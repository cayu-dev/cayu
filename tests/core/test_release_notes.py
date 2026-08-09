from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).parents[2]
_EXTRACTOR = _REPOSITORY_ROOT / "scripts" / "extract_release_notes.py"
_VERIFIER = _REPOSITORY_ROOT / "scripts" / "verify_release_state.py"


def _git(repository: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _initialize_tagged_notes(repository: Path) -> Path:
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "release-tests@example.invalid")
    _git(repository, "config", "user.name", "Release Tests")
    notes = repository / "docs" / "release-notes.md"
    notes.parent.mkdir()
    notes.write_text(
        "# Release notes\n\n## Unreleased\n\nWork in progress.\n\n## v0.1.0\n\nPublished release.\n"
    )
    (repository / "pyproject.toml").write_text('[project]\nname = "example"\nversion = "0.1.0"\n')
    _git(repository, "add", "docs/release-notes.md", "pyproject.toml")
    _git(repository, "commit", "--quiet", "-m", "release")
    _git(repository, "tag", "v0.1.0")
    return notes


def _verify(repository: Path, notes: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_VERIFIER),
            "--repository",
            str(repository),
            "--notes",
            str(notes),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


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
    assert output.read_bytes() == (
        b"\nSelected release.\n\n### Upgrade\n\nUse the new version.\n\n"
    )


def test_release_note_extractor_preserves_body_bytes_verbatim(tmp_path: Path) -> None:
    notes = tmp_path / "release-notes.md"
    output = tmp_path / "selected.md"
    notes.write_bytes(
        b"# Release notes\r\n\r\n"
        b"## v0.2.0\r\n\r\n"
        b"  Selected release.  \r\n\r\n"
        b"## v0.1.0\r\n\r\nOlder release.\r\n"
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
    assert output.read_bytes() == b"\r\n  Selected release.  \r\n\r\n"


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


def test_release_note_verifier_rejects_changes_to_a_tagged_section(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    notes.write_text(notes.read_text().replace("Published release.", "Rewritten release."))

    result = _verify(repository, notes)

    assert result.returncode != 0
    assert "v0.1.0" in result.stderr
    assert "differs from its immutable tagged section" in result.stderr


def test_release_note_verifier_rejects_whitespace_changes_to_a_tagged_section(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    notes.write_text(notes.read_text() + "\n")

    result = _verify(repository, notes)

    assert result.returncode != 0
    assert "differs from its immutable tagged section" in result.stderr


def test_release_note_verifier_rejects_line_ending_changes_to_a_tagged_section(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    notes.write_bytes(notes.read_bytes().replace(b"\n", b"\r\n"))

    result = _verify(repository, notes)

    assert result.returncode != 0
    assert "differs from its immutable tagged section" in result.stderr


def test_release_note_verifier_allows_unreleased_changes(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    notes.write_text(notes.read_text().replace("Work in progress.", "More work in progress."))

    result = _verify(repository, notes)

    assert result.returncode == 0, result.stderr


def test_release_note_verifier_rejects_retroactive_notes_for_an_existing_tag(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    _git(repository, "tag", "v0.1.0a1")
    notes.write_text(
        notes.read_text().replace(
            "## v0.1.0\n",
            "## v0.1.0a1\n\nFabricated historical notes.\n\n## v0.1.0\n",
        )
    )

    result = _verify(repository, notes)

    assert result.returncode != 0
    assert "v0.1.0a1" in result.stderr
    assert "did not publish a matching release-note section" in result.stderr


def test_release_note_verifier_rejects_reusing_a_published_version(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    (repository / "post-release.txt").write_text("new behavior\n")
    _git(repository, "add", "post-release.txt")
    _git(repository, "commit", "--quiet", "-m", "post-release work")

    result = _verify(repository, notes)

    assert result.returncode != 0
    assert "source version 0.1.0 is already published by v0.1.0" in result.stderr
    assert "distinct development version" in result.stderr


def test_release_note_verifier_allows_a_distinct_development_version(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    (repository / "pyproject.toml").write_text(
        '[project]\nname = "example"\nversion = "0.1.1.dev0"\n'
    )
    _git(repository, "add", "pyproject.toml")
    _git(repository, "commit", "--quiet", "-m", "start post-release development")

    result = _verify(repository, notes)

    assert result.returncode == 0, result.stderr
