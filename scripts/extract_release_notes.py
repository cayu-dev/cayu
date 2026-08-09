"""Extract one tagged release section from the curated release notes."""

from __future__ import annotations

import argparse
from pathlib import Path


def extract_release_section(document: str, *, version: str) -> str:
    """Return the exact heading and body of one ``## VERSION`` section."""

    lines = document.splitlines(keepends=True)
    heading = f"## {version}"
    matches = [index for index, line in enumerate(lines) if line.rstrip("\r\n") == heading]
    if not matches:
        raise ValueError(f"release notes do not contain an exact {heading!r} section")
    if len(matches) > 1:
        raise ValueError(f"release notes contain duplicate {heading!r} sections")
    start = matches[0]
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].rstrip("\r\n").startswith("## ")
        ),
        len(lines),
    )
    return "".join(lines[start:end])


def extract_release_notes(document: str, *, version: str) -> str:
    """Return the exact body bytes-as-text of one ``## VERSION`` section."""

    heading = f"## {version}"
    section = extract_release_section(document, version=version)
    selected = "".join(section.splitlines(keepends=True)[1:])
    if not selected.strip():
        raise ValueError(f"release notes section {heading!r} is empty")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        selected = extract_release_notes(
            args.notes.read_bytes().decode("utf-8"),
            version=args.version,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.output.write_bytes(selected.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
