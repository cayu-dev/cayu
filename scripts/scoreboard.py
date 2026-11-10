#!/usr/bin/env python3
"""Print the structural scoreboard for one commit as a JSON object.

The numbers come from committed files only; nothing is imported or run.
Field definitions live in docs/refactoring-baseline.md.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any

from size_report import (
    _assigned_value,
    _physical_lines,
    _resolve_commit,
    _tracked_blobs,
    public_name_count,
)

_OVERHEAD_BUDGETS = "tests/test_overhead_budgets.py"
_MIGRATIONS = "src/cayu/storage/migrations.py"


def _schema_revisions(source: str) -> tuple[int, int]:
    revisions = _assigned_value(ast.parse(source, filename=_MIGRATIONS), "REVISIONS")
    if not isinstance(revisions, ast.Tuple):
        raise ValueError(f"{_MIGRATIONS}: REVISIONS must be a literal tuple")
    kinds = []
    for entry in revisions.elts:
        kind = next(
            (keyword.value for keyword in getattr(entry, "keywords", ()) if keyword.arg == "kind"),
            None,
        )
        if not isinstance(kind, ast.Attribute) or kind.attr not in {"ADDITIVE", "BREAKING"}:
            raise ValueError(f"{_MIGRATIONS}:{entry.lineno}: revision kind is not literal")
        kinds.append(kind.attr)
    return len(kinds), kinds.count("BREAKING")


def build_scoreboard(repo: Path, revision: str) -> dict[str, Any]:
    commit = _resolve_commit(repo, revision)
    source_lines = test_lines = public_names = 0
    store_calls = schema = None
    with closing(_tracked_blobs(repo, commit)) as blobs:
        for path, data in blobs:
            if not path.endswith(".py"):
                continue
            source = data.decode("utf-8")
            if path.startswith("src/cayu/"):
                source_lines += _physical_lines(source)
            else:
                test_lines += _physical_lines(source)
            if path.startswith("src/cayu/") and path.endswith("/_exports.py"):
                public_names += public_name_count(source, path)
            elif path == _OVERHEAD_BUDGETS:
                budgets = ast.literal_eval(
                    _assigned_value(ast.parse(source, filename=path), "BUDGETS")
                )
                store_calls = {
                    step: counts["store_calls"]
                    for step, counts in budgets.items()
                    if "store_calls" in counts
                }
            elif path == _MIGRATIONS:
                schema = _schema_revisions(source)
    if store_calls is None or schema is None:
        raise ValueError(f"{_OVERHEAD_BUDGETS} or {_MIGRATIONS} is absent at {commit}")
    return {
        "commit": commit,
        "source_python_lines": source_lines,
        "test_python_lines": test_lines,
        "public_names": public_names,
        "store_call_budgets": store_calls,
        "schema_revisions": schema[0],
        "breaking_schema_revisions": schema[1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--revision", default="HEAD", help="Commit or ref to measure")
    args = parser.parse_args()
    try:
        scoreboard = build_scoreboard(args.repo, args.revision)
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"scoreboard: {error}\n")
    print(json.dumps(scoreboard, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
