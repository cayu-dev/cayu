from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Iterator, Sequence
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
