from __future__ import annotations

import ast
from pathlib import Path


def test_workspace_package_does_not_import_runtime() -> None:
    workspace_package = Path(__file__).parents[2] / "src" / "cayu" / "workspaces"
    offenders: list[str] = []
    for source_path in workspace_package.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            imports_runtime = (
                isinstance(node, ast.ImportFrom)
                and (
                    (node.level == 0 and (node.module or "").startswith("cayu.runtime"))
                    or (node.level >= 2 and (node.module or "").startswith("runtime"))
                    or (
                        node.level == 0
                        and node.module == "cayu"
                        and any(alias.name == "runtime" for alias in node.names)
                    )
                )
            ) or (
                isinstance(node, ast.Import)
                and any(alias.name.startswith("cayu.runtime") for alias in node.names)
            )
            if imports_runtime:
                offenders.append(str(source_path.relative_to(workspace_package)))
    assert offenders == []
