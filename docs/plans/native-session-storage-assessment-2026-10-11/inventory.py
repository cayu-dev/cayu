"""Report native session ownership and physical source sizes from a checkout."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

BASE = "7fbcd127e8ffecef6fefeb6add32826962ddf040"
ROOT = Path(__file__).resolve().parents[3]


def inventory():
    result = {"base": BASE, "backends": {}, "new_production_modules": 2}
    for backend in ("sqlite", "postgres"):
        path = Path(f"src/cayu/storage/{backend}.py")
        source = (ROOT / path).read_text()
        parent = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT, text=True)
        owner_path = path.with_name(f"_{backend}_publication.py")
        owner = (ROOT / owner_path).read_text()
        tree = ast.parse(source)
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name.lower() == backend + "sessionstore"
        )
        methods = [
            node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        owners = {}
        for candidate in sorted((ROOT / "src/cayu/storage").glob(f"_{backend}_*.py")):
            owners[str(candidate.relative_to(ROOT))] = len(candidate.read_text().splitlines())
        before, after, receiving = map(lambda text: len(text.splitlines()), (parent, source, owner))
        result["backends"][backend] = {
            "adapter_before": before,
            "adapter_after": after,
            "publication_owner": receiving,
            "net_production_lines": after + receiving - before,
            "adapter_below_10000": after < 10000,
            "receiving_owner_below_10000": receiving < 10000,
            "native_modules": owners,
            "retained_methods": [
                {"name": node.name, "lines": node.end_lineno - node.lineno + 1} for node in methods
            ],
        }
    return result


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))
