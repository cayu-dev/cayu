from __future__ import annotations

import argparse
import contextlib
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TextIO


def add_output_options(
    parser: argparse.ArgumentParser,
    *,
    formats: Sequence[str] = ("json", "table"),
    default: str = "json",
) -> None:
    """Add the shared format/destination contract for structured CLI output."""

    choices = tuple(formats)
    if default not in choices:
        raise ValueError("default output format must be one of formats")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--format",
        dest="output_format",
        choices=choices,
        default=default,
        help=f"Output format (default: {default}).",
    )
    if "json" in choices:
        group.add_argument(
            "--json",
            dest="output_format",
            action="store_const",
            const="json",
            help="Emit JSON (shorthand for `--format json`).",
        )
    if "table" in choices:
        group.add_argument(
            "--table",
            dest="output_format",
            action="store_const",
            const="table",
            help="Emit human-readable output (shorthand for `--format table`).",
        )
    if "html" in choices:
        group.add_argument(
            "--html",
            dest="output_format",
            action="store_const",
            const="html",
            help="Emit HTML (shorthand for `--format html`).",
        )
    if "jsonl" in choices:
        group.add_argument(
            "--jsonl",
            dest="output_format",
            action="store_const",
            const="jsonl",
            help="Emit JSON Lines (shorthand for `--format jsonl`).",
        )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Write output to FILE instead of stdout.",
    )


@contextlib.contextmanager
def output_destination(path: str | None) -> Iterator[TextIO]:
    """Redirect stdout to an explicit destination while leaving stderr alone."""

    if path is None:
        yield sys.stdout
        return
    with (
        Path(path).open("w", encoding="utf-8") as stream,
        contextlib.redirect_stdout(stream),
    ):
        yield stream


# C0 and C1 control characters, which a terminal may act on (escape sequences, OSC 52).
_TERMINAL_CONTROLS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_TERMINAL_CONTROLS_EXCEPT_LINES = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def terminal_safe(text: str, *, keep_lines: bool = False) -> str:
    """``text`` with control characters shown as ``\\xNN``, for human-readable output.

    Stored or remote text (task titles, error messages, knowledge entries) can carry
    escape sequences that would rewrite the terminal or its clipboard. ``keep_lines``
    keeps newlines and tabs for multi-line text.
    """

    pattern = _TERMINAL_CONTROLS_EXCEPT_LINES if keep_lines else _TERMINAL_CONTROLS
    return pattern.sub(lambda match: f"\\x{ord(match.group()):02x}", text)


def print_table(headers: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    """Print ``rows`` as aligned columns, ``-`` for a missing value, controls escaped."""

    if not rows:
        return
    rendered = [
        [
            "-" if row.get(header) is None else terminal_safe(str(row.get(header)))
            for header in headers
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    for row in rendered:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
