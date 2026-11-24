"""The data moves a project declares, by name, in `[tool.cayu.data_moves]`.

    [tool.cayu.data_moves]
    invoices = "moves.invoices:INVOICES"

Each value is the `module:name` of a `DataMove`, or of a function without arguments that
returns one. `cayu data-move run invoices` and the server's data-move routes (which `cayu
cloud data-move` calls on a deployed Agent) find moves by these names. Nothing is imported
until a move is used.
"""

from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from cayu.data_moves import DataMove

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class DataMoveDeclarationError(ValueError):
    """`[tool.cayu.data_moves]` is malformed, or a declared move can't be loaded."""


def declared_data_moves(start: Path | None = None) -> dict[str, str]:
    """Name to `module:name`, from the nearest `[tool.cayu]` that has `data_moves`."""

    return declared_data_move_project(start)[1]


def declared_data_move_project(start: Path | None = None) -> tuple[Path | None, dict[str, str]]:
    """The declaring project's root, whose modules the targets name, and its declarations."""

    from cayu.cli.project import discover_cayu_project_configuration

    discovered = discover_cayu_project_configuration(discovery_keys=("data_moves",), start=start)
    if discovered is None:
        return None, {}
    table = discovered.config["data_moves"]
    if not isinstance(table, dict):
        raise DataMoveDeclarationError(
            f"{discovered.pyproject}: [tool.cayu].data_moves must be a table of names."
        )
    declared: dict[str, str] = {}
    for name, target in table.items():
        if not _NAME.fullmatch(name) or not isinstance(target, str) or ":" not in target:
            raise DataMoveDeclarationError(
                f"{discovered.pyproject}: [tool.cayu.data_moves] {name!r} must name a "
                '"module:name" target, and names are letters, digits, "_", "." and "-".'
            )
        declared[name] = target
    return discovered.root, declared


def load_declared_data_move(name: str, target: str, *, project: Path | None) -> DataMove[Any]:
    """The move `target` names, imported from `project`, under its declared `name`.

    Run records and the routes then know it by one name, whatever its `DataMove.name` is.
    """

    move = load_data_move(target, project=project)
    return move if move.name == name else replace(move, name=name)


def load_data_move(target: str, *, project: Path | None = None) -> DataMove[Any]:
    """The `DataMove` that `target` (`module:name`) is, or that the function it names returns."""

    from cayu.cli._targets import TargetResolutionError, load_target

    root = str(Path.cwd() if project is None else project)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        value = load_target(target, label="Data move", normalize_errors=True)
    except TargetResolutionError as exc:
        raise DataMoveDeclarationError(str(exc)) from None
    move = value if isinstance(value, DataMove) else value() if callable(value) else None
    if not isinstance(move, DataMove):
        raise DataMoveDeclarationError(
            f"{target} is not a DataMove or a function that returns one."
        )
    return move
