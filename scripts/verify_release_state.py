"""Verify release-note and package-version identity against local tags."""

from __future__ import annotations

import argparse
import subprocess
import tomllib
from pathlib import Path

from extract_release_notes import extract_release_notes, extract_release_section


def _git_bytes(repository: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _git(repository: Path, *args: str) -> str:
    return _git_bytes(repository, *args).decode("utf-8")


def _release_tags(repository: Path) -> tuple[str, ...]:
    return tuple(sorted(_git(repository, "tag", "--list", "v*").splitlines()))


def verify_tagged_release_notes(
    document: str,
    *,
    repository: Path,
    notes_path: Path,
) -> None:
    """Reject changes to sections that were already published from local tags."""

    repository = repository.resolve()
    try:
        relative_notes = notes_path.resolve().relative_to(repository).as_posix()
    except ValueError as exc:
        raise ValueError("release notes must be inside the Git repository") from exc

    for tag in _release_tags(repository):
        tagged_paths = _git(
            repository,
            "ls-tree",
            "-r",
            "--name-only",
            tag,
            "--",
            relative_notes,
        ).splitlines()
        if relative_notes not in tagged_paths:
            continue
        tagged_document = _git_bytes(
            repository,
            "show",
            f"{tag}:{relative_notes}",
        ).decode("utf-8")
        heading = f"## {tag}"
        tagged_has_section = heading in tagged_document.splitlines()
        current_has_section = heading in document.splitlines()
        if not tagged_has_section:
            if current_has_section:
                raise ValueError(
                    f"immutable tag {tag} did not publish a matching release-note section; "
                    "do not add retroactive release notes"
                )
            continue
        try:
            extract_release_notes(tagged_document, version=tag)
            extract_release_notes(document, version=tag)
        except ValueError as exc:
            raise ValueError(f"cannot verify immutable {tag} release notes: {exc}") from exc
        if extract_release_section(document, version=tag) != extract_release_section(
            tagged_document,
            version=tag,
        ):
            raise ValueError(
                f"release notes {tag} section differs from its immutable tagged section"
            )


def verify_source_version(*, repository: Path, pyproject_path: Path) -> None:
    """Reject source trees that reuse an existing tag's package version."""

    repository = repository.resolve()
    try:
        pyproject_path.resolve().relative_to(repository)
    except ValueError as exc:
        raise ValueError("pyproject.toml must be inside the Git repository") from exc
    with pyproject_path.open("rb") as pyproject:
        version = tomllib.load(pyproject).get("project", {}).get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("pyproject.toml must declare a non-empty project.version")

    tag = f"v{version}"
    if tag not in _release_tags(repository):
        return
    head_commit = _git(repository, "rev-parse", "HEAD").strip()
    tagged_commit = _git(repository, "rev-list", "-n", "1", tag).strip()
    if head_commit != tagged_commit:
        raise ValueError(
            f"source version {version} is already published by {tag}; "
            "use a distinct development version"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--pyproject", type=Path)
    args = parser.parse_args(argv)
    pyproject = args.pyproject or args.repository / "pyproject.toml"

    try:
        verify_tagged_release_notes(
            args.notes.read_bytes().decode("utf-8"),
            repository=args.repository,
            notes_path=args.notes,
        )
        verify_source_version(repository=args.repository, pyproject_path=pyproject)
    except (subprocess.CalledProcessError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
