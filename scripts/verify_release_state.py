"""Verify release-note and package-version identity against local tags."""

from __future__ import annotations

import argparse
import re
import subprocess
import tomllib
from pathlib import Path

from extract_release_notes import extract_release_notes, extract_release_section

#: Every new release section starts with a paragraph that begins with this label.
UPGRADE_SUMMARY_PREFIX = "Upgrade:"
#: Every new release section has this heading; its body is "None." when nothing breaks.
BREAKING_CHANGES_HEADING = "### Breaking changes"


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


_FENCE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})")
_ATX_HEADING = re.compile(r"^ {0,3}#{1,6}(?:[ \t]|$)")
_COMMENT_BLOCK = re.compile(r"^ {0,3}<!--")
_INLINE_MARK = re.compile(r"`+|<!--")
_BACKTICKS = re.compile(r"`+")


def _opening_fence(line: str) -> str | None:
    match = _FENCE.match(line)
    if match is None:
        return None
    fence = match.group("fence")
    # A backtick fence's info string cannot contain a backtick, so a line such
    # as ```code``` is an inline code span, not an opening fence.
    if fence[0] == "`" and "`" in line[match.end() :]:
        return None
    return fence


def _closes_fence(line: str, fence: str) -> bool:
    match = _FENCE.match(line)
    return (
        match is not None
        and match.group("fence")[0] == fence[0]
        and len(match.group("fence")) >= len(fence)
        and not line[match.end() :].strip()
    )


def _rendered_lines(lines: list[str]) -> tuple[list[str], list[bool]]:
    """Read lines the way Markdown renders them, for the format checks.

    Returns each line's visible text, with HTML comments removed and line
    positions kept, and whether the line is a heading. Blocks are found first:
    code fences, comment blocks, ATX headings and paragraphs. Code spans and
    inline comments are then resolved across each paragraph's lines, so a
    ``<!--`` inside code stays literal, wrapped code spans stay one span, and a
    fence inside a comment stays hidden.
    """

    visible: list[str] = []
    headings: list[bool] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        fence = _opening_fence(line)
        if fence is not None:
            # The fence and its content render as a code block.
            visible.append(line)
            headings.append(False)
            index += 1
            while index < len(lines):
                visible.append(lines[index])
                headings.append(False)
                index += 1
                if _closes_fence(lines[index - 1], fence):
                    break
            continue
        if _COMMENT_BLOCK.match(line):
            # A comment block runs to the line that closes it, across blank lines.
            first = index
            while index < len(lines):
                search_from = lines[index].index("<!--") + 4 if index == first else 0
                closing = lines[index].find("-->", search_from)
                if closing >= 0:
                    rest = _inline_visible(lines[index][closing + 3 :])
                    visible.append(rest)
                    headings.append(False)
                    index += 1
                    break
                visible.append("")
                headings.append(False)
                index += 1
            continue
        if not line.strip() or _ATX_HEADING.match(line):
            text = _inline_visible(line)
            visible.append(text)
            headings.append(bool(line.strip()))
            index += 1
            continue
        paragraph = [line]
        index += 1
        while index < len(lines):
            following = lines[index]
            if (
                not following.strip()
                or _ATX_HEADING.match(following)
                or _COMMENT_BLOCK.match(following)
                or _opening_fence(following) is not None
            ):
                break
            paragraph.append(following)
            index += 1
        text = _inline_visible("\n".join(paragraph))
        visible.extend(text.split("\n"))
        headings.extend(False for _ in paragraph)
    return visible, headings


def _inline_visible(text: str) -> str:
    """Remove inline HTML comments outside code spans; keep every newline.

    A backtick run opens a code span only when the same length run closes it
    within ``text``, and a comment is removed only when it closes there too;
    otherwise the marker is literal text, as in CommonMark.
    """

    kept: list[str] = []
    position = 0
    for mark in _INLINE_MARK.finditer(text):
        if mark.start() < position:
            continue
        kept.append(text[position : mark.start()])
        token = mark.group()
        if token == "<!--":
            closing = text.find("-->", mark.end())
            if closing < 0:
                kept.append(token)
                position = mark.end()
            else:
                kept.append("\n" * text.count("\n", mark.start(), closing))
                position = closing + 3
            continue
        closing_end = next(
            (
                later.end()
                for later in _BACKTICKS.finditer(text[mark.end() :])
                if len(later.group()) == len(token)
            ),
            None,
        )
        if closing_end is None:
            kept.append(token)
            position = mark.end()
        else:
            position = mark.end() + closing_end
            kept.append(text[mark.start() : position])
    kept.append(text[position:])
    return "".join(kept)


def _release_sections(document: str) -> list[tuple[str, list[str]]]:
    lines = document.splitlines()
    _, headings = _rendered_lines(lines)
    starts = [
        index for index, line in enumerate(lines) if headings[index] and line.startswith("## ")
    ]
    return [
        (
            lines[start][3:].strip(),
            lines[start + 1 : starts[position + 1] if position + 1 < len(starts) else len(lines)],
        )
        for position, start in enumerate(starts)
    ]


def release_section_format_errors(body: list[str]) -> list[str]:
    """Return the format problems of one release section body, or an empty list."""

    errors: list[str] = []
    body, rendered = _rendered_lines(body)
    content = [line for line in body if line.strip()]
    first = content[0] if content else ""
    if (
        not first.startswith(UPGRADE_SUMMARY_PREFIX)
        or not first[len(UPGRADE_SUMMARY_PREFIX) :].strip()
    ):
        errors.append(
            f"must start with an upgrade summary beginning with {UPGRADE_SUMMARY_PREFIX!r}"
        )
    headings = [
        index
        for index, line in enumerate(body)
        if rendered[index] and line.rstrip() == BREAKING_CHANGES_HEADING
    ]
    if len(headings) != 1:
        errors.append(f"must contain exactly one {BREAKING_CHANGES_HEADING!r} heading")
        return errors
    following = body[headings[0] + 1 :]
    following_rendered = rendered[headings[0] + 1 :]
    end = next(
        (
            index
            for index, line in enumerate(following)
            if following_rendered[index] and line.startswith("#")
        ),
        len(following),
    )
    if not any(line.strip() for line in following[:end]):
        errors.append(
            f"has an empty {BREAKING_CHANGES_HEADING!r} heading; write 'None.' when nothing breaks"
        )
    return errors


#: The last release published before the release-note format policy took effect.
#: The format applies from the next release on; sections for this version and
#: older ones were published in their original format and are exempt.
LAST_RELEASE_BEFORE_FORMAT_POLICY = "v0.11.4"

_RELEASE_VERSION = re.compile(
    r"^v(?P<release>[0-9]+(?:\.[0-9]+)*)"
    r"(?:(?P<pre>a|b|rc)(?P<pre_number>[0-9]+))?"
    r"(?:\.dev(?P<dev>[0-9]+))?$"
)
_PRE_RELEASE_ORDER = {"a": 0, "b": 1, "rc": 2}


def _release_version_key(title: str) -> tuple[object, ...] | None:
    """Return a PEP 440 ordering key for a ``vX.Y.Z[aN|bN|rcN][.devN]`` heading."""

    match = _RELEASE_VERSION.fullmatch(title)
    if match is None:
        return None
    release = tuple(int(part) for part in match["release"].split("."))
    release += (0,) * (3 - len(release))
    if match["pre"] is not None:
        phase: tuple[int, int] = (_PRE_RELEASE_ORDER[match["pre"]], int(match["pre_number"]))
    elif match["dev"] is not None:
        # A development release without a pre-release sorts before every pre-release.
        phase = (-1, 0)
    else:
        phase = (3, 0)
    dev = (0, int(match["dev"])) if match["dev"] is not None else (1, 0)
    return (release, phase, dev)


_FORMAT_POLICY_CUTOFF = _release_version_key(LAST_RELEASE_BEFORE_FORMAT_POLICY)


def release_section_requires_format(title: str) -> bool:
    """Whether a ``## title`` section must follow the release-note format.

    The decision depends only on the version in the heading, never on which tags
    a checkout happens to have, so the release being published is checked and a
    shallow checkout reaches the same result as a full clone.
    """

    key = _release_version_key(title)
    return key is None or key > _FORMAT_POLICY_CUTOFF


def verify_release_note_format(document: str) -> None:
    """Require the upgrade summary and breaking-changes heading in new sections.

    ``## Unreleased`` and every release newer than
    :data:`LAST_RELEASE_BEFORE_FORMAT_POLICY` must follow the format.
    """

    problems = [
        f"release notes section '## {title}' {error}"
        for title, body in _release_sections(document)
        if release_section_requires_format(title)
        for error in release_section_format_errors(body)
    ]
    if problems:
        raise ValueError("; ".join(problems))


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
        document = args.notes.read_bytes().decode("utf-8")
        verify_tagged_release_notes(
            document,
            repository=args.repository,
            notes_path=args.notes,
        )
        verify_release_note_format(document)
        verify_source_version(repository=args.repository, pyproject_path=pyproject)
    except (subprocess.CalledProcessError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
