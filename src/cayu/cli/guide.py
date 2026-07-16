from __future__ import annotations

import argparse
from importlib.resources import files
from typing import Any

_GUIDES = {
    "anatomy": "application-anatomy.md",
    "authoring": "authoring.md",
    "diagnostics": "diagnostics.md",
    "tool-effects": "tool-effects.md",
}
_INCLUDES = {
    "<!-- cayu-guide-include:pytest-selector -->": (
        "command_selectors.py",
        "# cayu-guide-include:pytest-selector:start",
        "# cayu-guide-include:pytest-selector:end",
    ),
}


def _render_includes(content: str) -> str:
    for placeholder, (resource_name, start_marker, end_marker) in _INCLUDES.items():
        count = content.count(placeholder)
        if count == 0:
            continue
        if count != 1:
            raise RuntimeError(f"guide include must appear exactly once: {placeholder}")
        source = files("cayu.guides").joinpath(resource_name).read_text(encoding="utf-8")
        try:
            recipe = source.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
        except IndexError as error:
            raise RuntimeError(f"guide include markers are malformed: {resource_name}") from error
        content = content.replace(placeholder, f"```python\n{recipe}\n```")
    return content


def add_guide_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "guide",
        help="Print package-shipped Cayu application guidance.",
    )
    parser.add_argument("name", choices=tuple(_GUIDES))


def run_guide(args: argparse.Namespace) -> int:
    guide = files("cayu.guides").joinpath(_GUIDES[args.name])
    print(_render_includes(guide.read_text(encoding="utf-8")), end="")
    return 0
