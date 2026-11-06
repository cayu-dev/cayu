"""Read ``--cursor VALUE`` when the cursor starts with a dash.

argparse takes any token that starts with ``-`` for an option, so a cursor such as
``-Jx4…`` (base64url text can start with ``-``) worked only as ``--cursor=-Jx4…``;
``--cursor -Jx4…`` failed with "expected one argument". A token after ``--cursor``
that names no option of the command line is its value.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def attach_cursor_values(parser: argparse.ArgumentParser, arguments: Sequence[str]) -> list[str]:
    """``arguments`` with each ``--cursor -VALUE`` written as ``--cursor=-VALUE``."""

    options = _option_strings(parser)
    attached: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == "--":
            attached.extend(arguments[index:])
            break
        value = arguments[index + 1] if index + 1 < len(arguments) else None
        if (
            token == "--cursor"
            and value is not None
            and value.startswith("-")
            and value != "--"
            and value not in options
        ):
            attached.append(f"--cursor={value}")
            index += 2
            continue
        attached.append(token)
        index += 1
    return attached


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    found: set[str] = set()
    for action in parser._actions:
        found.update(action.option_strings)
        if isinstance(action, argparse._SubParsersAction):
            for command in action.choices.values():
                found |= _option_strings(command)
    return found
