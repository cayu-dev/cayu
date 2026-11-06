"""`--cursor VALUE` for a cursor that starts with a dash."""

from __future__ import annotations

import argparse

from cayu.cli._cursor_arguments import attach_cursor_values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    listing = commands.add_parser("list")
    listing.add_argument("--cursor")
    listing.add_argument("--json", action="store_true")
    listing.add_argument("rest", nargs="*")
    return parser


def test_a_dash_leading_cursor_is_the_options_value() -> None:
    parser = _parser()
    arguments = attach_cursor_values(parser, ["list", "--cursor", "-Jx4_a", "--json"])

    assert arguments == ["list", "--cursor=-Jx4_a", "--json"]
    parsed = parser.parse_args(arguments)
    assert parsed.cursor == "-Jx4_a" and parsed.json is True


def test_an_option_after_cursor_and_tokens_after_a_separator_stay_as_they_are() -> None:
    parser = _parser()

    # A missing value still fails in argparse instead of becoming the cursor.
    assert attach_cursor_values(parser, ["list", "--cursor", "--json"]) == [
        "list",
        "--cursor",
        "--json",
    ]
    assert attach_cursor_values(parser, ["list", "--", "--cursor", "-x"]) == [
        "list",
        "--",
        "--cursor",
        "-x",
    ]
    assert attach_cursor_values(parser, ["list", "--cursor", "plain"]) == [
        "list",
        "--cursor",
        "plain",
    ]
