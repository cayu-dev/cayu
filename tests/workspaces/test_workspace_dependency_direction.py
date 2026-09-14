from __future__ import annotations

import ast
from pathlib import Path

# Lifecycle implementations now live with their public workspace concepts. Keep
# their existing coordination dependencies explicit; adapters must not acquire
# new dependencies on runtime orchestration. Authority is a dependency-neutral
# value module formerly owned by core.
_ALLOWED_RUNTIME_IMPORTS = {
    ("_local_branch.py", "cayu.runtime.authority", "SessionRunFenced"),
    ("branch_lifecycle.py", "cayu.runtime.service_manifest", "RuntimeStoreDurability"),
    ("checkpoint_lifecycle.py", "cayu.runtime._runtime_records", "RegisteredEnvironment"),
    ("checkpoint_lifecycle.py", "cayu.runtime._tool_round_executor", "_workspace_writer_isolation"),
    ("observation_recovery.py", "cayu.runtime._event_writer", "RuntimeEventWriter"),
    ("observation_recovery.py", "cayu.runtime.public_authority", "PublicAuthorityAliasCodec"),
    (
        "observation_recovery.py",
        "cayu.runtime.public_authority",
        "public_authority_alias_is_reserved",
    ),
}


def test_workspace_runtime_dependencies_stay_with_their_existing_owners() -> None:
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
                relative_path = str(source_path.relative_to(workspace_package))
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    for alias in node.names:
                        if (relative_path, node.module, alias.name) not in _ALLOWED_RUNTIME_IMPORTS:
                            offenders.append(f"{relative_path}: {node.module}.{alias.name}")
                else:
                    offenders.append(relative_path)
    assert offenders == []
