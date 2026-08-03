"""Extract one tagged release section from the curated release notes."""

from __future__ import annotations

import argparse
from pathlib import Path


def extract_release_notes(document: str, *, version: str) -> str:
    """Return the body of the exact ``## VERSION`` section."""

    lines = document.splitlines()
    heading = f"## {version}"
    matches = [index for index, line in enumerate(lines) if line == heading]
    if not matches:
        raise ValueError(f"release notes do not contain an exact {heading!r} section")
    if len(matches) > 1:
        raise ValueError(f"release notes contain duplicate {heading!r} sections")
    start = matches[0] + 1
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    selected = "\n".join(lines[start:end]).strip()
    if not selected:
        raise ValueError(f"release notes section {heading!r} is empty")
    return selected + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        selected = extract_release_notes(
            args.notes.read_text(encoding="utf-8"),
            version=args.version,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.output.write_text(selected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
