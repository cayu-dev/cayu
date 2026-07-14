from __future__ import annotations

import argparse
from importlib.resources import files
from typing import Any

_GUIDES = {
    "authoring": "authoring.md",
    "diagnostics": "diagnostics.md",
}


def add_guide_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "guide",
        help="Print package-shipped Cayu application guidance.",
    )
    parser.add_argument("name", choices=tuple(_GUIDES))


def run_guide(args: argparse.Namespace) -> int:
    guide = files("cayu.guides").joinpath(_GUIDES[args.name])
    print(guide.read_text(encoding="utf-8"), end="")
    return 0
