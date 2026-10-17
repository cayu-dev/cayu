from __future__ import annotations

import importlib.util
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
        "# Release notes\n\n## Unreleased\n\n"
        "Upgrade: no action needed. Work in progress.\n\n"
        "### Breaking changes\n\nNone.\n\n"
        "## v0.1.0\n\nPublished release.\n"
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


def _replace_unreleased(notes: Path, body: str) -> None:
    document = notes.read_text()
    start = document.index("## Unreleased\n")
    end = document.index("## v0.1.0\n")
    notes.write_text(document[:start] + "## Unreleased\n\n" + body + "\n" + document[end:])


def test_release_note_verifier_accepts_breaking_changes_with_entries(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    _replace_unreleased(
        notes,
        "Upgrade: run `cayu storage migrate` before starting new writers.\n\n"
        "### Breaking changes\n\n- Storage revision 2 is a breaking writer boundary.\n\n"
        "### Changes\n\n- Add a feature.\n",
    )

    result = _verify(repository, notes)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("body", "expected_error"),
    [
        (
            "Cayu adds a feature.\n\n### Breaking changes\n\nNone.\n",
            "'## Unreleased' must start with an upgrade summary beginning with 'Upgrade:'",
        ),
        (
            "Upgrade:\n\n### Breaking changes\n\nNone.\n",
            "'## Unreleased' must start with an upgrade summary beginning with 'Upgrade:'",
        ),
        (
            "### Breaking changes\n\nNone.\n\nUpgrade: too late.\n",
            "'## Unreleased' must start with an upgrade summary beginning with 'Upgrade:'",
        ),
        (
            "Upgrade: no action needed.\n\n- Fix a bug.\n",
            "'## Unreleased' must contain exactly one '### Breaking changes' heading",
        ),
        (
            "Upgrade: no action needed.\n\n### Breaking changes\n\nNone.\n\n"
            "### Breaking changes\n\nNone.\n",
            "'## Unreleased' must contain exactly one '### Breaking changes' heading",
        ),
        (
            "Upgrade: no action needed.\n\n### Breaking changes\n\n### Changes\n\n- Fix.\n",
            "'## Unreleased' has an empty '### Breaking changes' heading; write 'None.'",
        ),
        (
            "Upgrade: no action needed.\n\n### Breaking changes\n",
            "'## Unreleased' has an empty '### Breaking changes' heading; write 'None.'",
        ),
    ],
)
def test_release_note_verifier_rejects_new_sections_without_the_format(
    tmp_path: Path,
    body: str,
    expected_error: str,
) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    _replace_unreleased(notes, body)

    result = _verify(repository, notes)

    assert result.returncode != 0
    assert expected_error in result.stderr


_FENCED_EXAMPLE = (
    "```markdown\n## v9.0.0\n\nExample release.\n\n### Breaking changes\n\nNone.\n```\n"
)


def test_release_note_verifier_ignores_headings_inside_code_examples(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    _replace_unreleased(
        notes,
        "Upgrade: no action needed.\n\n### Breaking changes\n\n"
        "~~~bash\n# Rename the setting before upgrading.\ncayu config migrate\n~~~\n\n"
        "### Changes\n\n- Document the format:\n\n" + _FENCED_EXAMPLE,
    )

    result = _verify(repository, notes)

    assert result.returncode == 0, result.stderr


def test_release_note_verifier_treats_one_line_triple_backticks_as_inline_code(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    _replace_unreleased(
        notes,
        "Upgrade: no action needed.\n\nRun ```cayu version``` to check.\n\n"
        "```cayu version```\n\n### Breaking changes\n\nNone.\n",
    )

    result = _verify(repository, notes)

    assert result.returncode == 0, result.stderr


def test_release_note_verifier_rejects_a_breaking_changes_section_with_only_a_comment(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    _replace_unreleased(
        notes,
        "Upgrade: no action needed.\n\n### Breaking changes\n\n"
        "<!-- Fill this in\nbefore publishing. -->\n\n### Changes\n\n- Fix a typo.\n",
    )

    result = _verify(repository, notes)

    assert result.returncode != 0
    assert "has an empty '### Breaking changes' heading" in result.stderr


@pytest.mark.parametrize(
    "body",
    [
        "Upgrade: no action needed; the literal `<!--` marker is unchanged.\n\n"
        "### Breaking changes\n\nNone.\n",
        "Upgrade: no action needed.\n\n### Breaking changes\n\n"
        "```markdown\n<!-- This comment is displayed as code. -->\n```\n",
        "Upgrade: no action needed; the literal `<!--\nmarker` is unchanged.\n\n"
        "### Breaking changes\n\nNone.\n",
        "Upgrade: run `cayu storage migrate\n--acknowledge-breaking 115` first.\n\n"
        "### Breaking changes\n\nNone.\n",
        "Upgrade: no action needed; ``<!--` example`` stays literal.\n\n"
        "### Breaking changes\n\nNone.\n",
        "Upgrade: no action needed.\n\n<!--\n### Breaking changes\nHidden.\n-->\n\n"
        "### Breaking changes\n\nNone.\n",
        "Upgrade: no action needed.\n\n### Breaking changes\n\nNone.\n\n"
        "<!--\n## v0.12.0\nTODO\n-->\n",
    ],
    ids=[
        "comment-opener-in-inline-code",
        "comment-in-a-code-fence",
        "comment-opener-in-wrapped-inline-code",
        "wrapped-inline-code",
        "double-backtick-span",
        "commented-out-heading",
        "commented-out-release",
    ],
)
def test_release_note_verifier_keeps_comment_markers_inside_code(tmp_path: Path, body: str) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    _replace_unreleased(notes, body)

    result = _verify(repository, notes)

    assert result.returncode == 0, result.stderr


def test_release_note_verifier_ignores_a_code_fence_inside_a_comment(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    _replace_unreleased(
        notes,
        "Upgrade: no action needed.\n\n### Breaking changes\n\n"
        "<!-- Reviewer template:\n```markdown\nNone.\n```\n-->\n\n"
        "### Changes\n\n- Fix a typo.\n",
    )

    result = _verify(repository, notes)

    assert result.returncode != 0
    assert "has an empty '### Breaking changes' heading" in result.stderr


def test_release_note_verifier_does_not_count_a_heading_inside_a_code_example(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    notes = _initialize_tagged_notes(repository)
    _replace_unreleased(
        notes,
        "Upgrade: no action needed.\n\n```markdown\n### Breaking changes\n\nNone.\n```\n",
    )

    result = _verify(repository, notes)

    assert result.returncode != 0
    assert (
        "'## Unreleased' must contain exactly one '### Breaking changes' heading" in result.stderr
    )


_FORMATTED_BODY = "Upgrade: no action needed.\n\n### Breaking changes\n\nNone.\n"
_LEGACY_SECTIONS = (
    "## v0.11.4\n\nLast release before the format policy.\n\n"
    "## v0.5.0\n\nOlder release.\n\n"
    "## v0.1.0rc4\n\nOut-of-order pre-release.\n\n"
    "## v0.2.0.dev0\n\nDevelopment release.\n"
)


def _commit_release(repository: Path, *, version: str, notes_text: str, tag: bool) -> Path:
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "release-tests@example.invalid")
    _git(repository, "config", "user.name", "Release Tests")
    notes = repository / "docs" / "release-notes.md"
    notes.parent.mkdir()
    notes.write_text(notes_text)
    (repository / "pyproject.toml").write_text(
        f'[project]\nname = "example"\nversion = "{version}"\n'
    )
    _git(repository, "add", "docs/release-notes.md", "pyproject.toml")
    _git(repository, "commit", "--quiet", "-m", "release")
    if tag:
        _git(repository, "tag", f"v{version}")
    return notes


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_release_state", _VERIFIER)
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(_VERIFIER.parent))
    try:
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
    finally:
        sys.path.remove(str(_VERIFIER.parent))
    return verifier


def test_release_note_verifier_checks_the_release_tagged_at_head(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    notes = _commit_release(
        repository,
        version="0.12.0",
        notes_text="# Release notes\n\n## v0.12.0\n\nCayu adds a feature.\n\n" + _LEGACY_SECTIONS,
        tag=False,
    )
    before_tag = _verify(repository, notes)
    _git(repository, "tag", "v0.12.0")

    after_tag = _verify(repository, notes)

    for result in (before_tag, after_tag):
        assert result.returncode != 0
        assert "'## v0.12.0' must start with an upgrade summary" in result.stderr
        assert "'## v0.12.0' must contain exactly one '### Breaking changes'" in result.stderr


def test_release_note_verifier_passes_a_shallow_checkout_with_only_the_current_tag(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    notes = _commit_release(
        repository,
        version="0.12.0",
        notes_text="# Release notes\n\n## v0.12.0\n\n" + _FORMATTED_BODY + "\n" + _LEGACY_SECTIONS,
        tag=True,
    )

    result = _verify(repository, notes)

    assert result.returncode == 0, result.stderr


def test_release_note_verifier_exempts_older_sections_without_any_tags(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    notes = _commit_release(
        repository,
        version="0.12.0.dev0",
        notes_text="# Release notes\n\n## Unreleased\n\n"
        + _FORMATTED_BODY
        + "\n"
        + _LEGACY_SECTIONS,
        tag=False,
    )

    result = _verify(repository, notes)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("title", "requires_format"),
    [
        ("Unreleased", True),
        ("v0.11.4", False),
        ("v0.11.3", False),
        ("v0.1.0rc5", False),
        ("v0.2.0.dev0", False),
        ("v0.11.5", True),
        ("v0.11.10", True),
        ("v0.12.0.dev0", True),
        ("v0.12.0rc1", True),
        ("v1.0.0", True),
    ],
)
def test_release_note_format_applies_after_the_cutoff_version(
    title: str, requires_format: bool
) -> None:
    verifier = _load_verifier()

    assert verifier.LAST_RELEASE_BEFORE_FORMAT_POLICY == "v0.11.4"
    assert verifier.release_section_requires_format(title) is requires_format


def test_repository_release_notes_follow_the_format() -> None:
    document = (_REPOSITORY_ROOT / "docs" / "release-notes.md").read_text(encoding="utf-8")

    _load_verifier().verify_release_note_format(document)
