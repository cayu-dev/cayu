"""`cayu objectives`: validate a project's improvement declaration and objectives file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from cayu.cli._output import add_output_options, output_destination
from cayu.evals.objectives import (
    PROJECT_IMPROVEMENT_DOCUMENT_SCHEMA_VERSION,
    ProjectImprovementDeclaration,
    ProjectImprovementError,
    discover_project_improvement,
    project_improvement_document,
    validate_project_improvement,
)

_COMMAND_FAILED = "OBJECTIVES_COMMAND_FAILED"


def add_objectives_parser(subparsers: Any) -> None:
    group = subparsers.add_parser(
        "objectives",
        help="Validate the project's objectives and improvement evals.",
        description=(
            "Read [tool.cayu.improvement] and the objectives file it names. Never imports "
            "project code or runs an eval."
        ),
    )
    commands = group.add_subparsers(dest="objectives_command", required=True)
    validate = commands.add_parser(
        "validate",
        help="Validate the declaration against the checkout and print its fingerprint.",
        description=(
            "Find the nearest Cayu project, load its objectives and improvement evals, and "
            "check every declared file. Exits 0 when valid and 2 otherwise."
        ),
    )
    validate.add_argument(
        "project",
        nargs="?",
        default=None,
        metavar="PROJECT_DIR",
        help="Directory to search upward from (default: the working directory).",
    )
    add_output_options(validate, formats=("json", "table"), default="table")


def run_objectives(args: argparse.Namespace) -> int:
    start = Path.cwd() if args.project is None else Path(args.project)
    declaration: ProjectImprovementDeclaration | None = None
    try:
        # Discovery walks upward, so a mistyped directory would silently validate
        # whichever project encloses it.
        if args.project is not None and not start.is_dir():
            raise ProjectImprovementError(f"{args.project} is not a directory.")
        declaration = discover_project_improvement(start)
        if declaration is None:
            raise ProjectImprovementError(
                "No Cayu project with [tool.cayu.improvement] found. Add "
                'objectives = "cayu-objectives.toml" and the improvement evals under '
                "[tool.cayu.improvement]."
            )
        problems = validate_project_improvement(declaration)
    except (OSError, ValueError) as exc:
        if args.output_format == "table":
            print(f"error: {exc}", file=sys.stderr)
        report = _error_report(args, str(exc))
        status = 2
    else:
        report = _report(args, declaration, problems)
        status = 2 if problems else 0
    # Everything is read before the report is written, and the report never replaces
    # a file the declaration depends on.
    if (overwritten := _overwritten_input(args.output, start, declaration)) is not None:
        message = f"--output must not overwrite {overwritten}."
        if args.output_format == "json":
            print(_error_report(args, message), end="")
        else:
            print(f"error: {message}", file=sys.stderr)
        return 2
    # Written on every run, like `cayu check`, so a report file never keeps the
    # result of an earlier run.
    try:
        with output_destination(args.output) as stream:
            stream.write(report)
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 2
    return status


def _report(
    args: argparse.Namespace,
    declaration: ProjectImprovementDeclaration,
    problems: tuple[str, ...],
) -> str:
    if args.output_format == "json":
        return _json(project_improvement_document(declaration, problems))
    if problems:
        return f"Invalid improvement declaration in {declaration.pyproject}:\n" + "".join(
            f"- {problem}\n" for problem in problems
        )
    return (
        f"Valid improvement declaration {declaration.declaration_fingerprint}\n"
        f"Objectives {declaration.objectives_fingerprint} ({declaration.config.objectives})\n"
    )


def _error_report(args: argparse.Namespace, message: str) -> str:
    if args.output_format != "json":
        return ""
    return _json(
        {
            "schema_version": PROJECT_IMPROVEMENT_DOCUMENT_SCHEMA_VERSION,
            "error": {"code": _COMMAND_FAILED, "message": message},
        }
    )


def _overwritten_input(
    output: str | None, start: Path, declaration: ProjectImprovementDeclaration | None
) -> Path | None:
    """The input an ``--output`` path would replace: a ``pyproject.toml`` discovery may
    read, or a path the declaration protects, including anything inside a protected
    directory."""

    if output is None:
        return None
    destination = Path(output).resolve()
    directory = start.resolve()
    inputs = [candidate / "pyproject.toml" for candidate in (directory, *directory.parents)]
    if declaration is not None:
        inputs.extend(declaration.root / path for path in declaration.protected_paths)
    for path in inputs:
        if destination == path or (path.is_dir() and destination.is_relative_to(path)):
            return path
        try:
            if destination.exists() and path.exists() and os.path.samefile(destination, path):
                return path
        except OSError:
            continue
    return None


def _json(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"
