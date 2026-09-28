#!/usr/bin/env python3
"""Measure tracked Cayu source at one commit without importing that source.

This is a structural inventory, not a complexity score or performance benchmark.
Usage and metric definitions live in docs/refactoring-baseline.md.
No worktree files are included in measurements.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Generator
from contextlib import closing
from importlib.util import resolve_name
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
_WATCHED_CALLS = frozenset(
    {
        "pending_tool_round_from_checkpoint",
        "prepare_tool_round_publication",
        "publish_tool_round_with_exact_replay",
        "_ToolRoundPublicationCoordinator",
    }
)
_PROVIDER_NAMES = frozenset(
    {"cayu.ModelProvider", "cayu.providers.ModelProvider", "cayu.providers.base.ModelProvider"}
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _module(path: str) -> str:
    name = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
    return name.removesuffix(".__init__")


def _tracked_blobs(repo: Path, revision: str) -> Generator[tuple[str, bytes], None, None]:
    entries = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "-z", revision]
    ).split(b"\0")
    process = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for entry in entries:
            if not entry:
                continue
            header, encoded_path = entry.split(b"\t", 1)
            mode, kind, object_id = header.split()
            path = encoded_path.decode("utf-8")
            selected = path.startswith(("src/cayu/", "tests/")) and path.endswith((".py", ".pyi"))
            if mode not in (b"100644", b"100755") or not (selected or path == ".test_durations"):
                continue
            if kind != b"blob":
                raise ValueError(f"Expected a regular tracked blob: {path}")
            process.stdin.write(object_id + b"\n")
            process.stdin.flush()
            returned_id, returned_kind, size = process.stdout.readline().split()
            if returned_id != object_id or returned_kind != b"blob":
                raise ValueError(f"Unexpected Git object response for {path}")
            data = process.stdout.read(int(size))
            if len(data) != int(size) or process.stdout.read(1) != b"\n":
                raise ValueError(f"Incomplete Git object response for {path}")
            yield path, data
        process.stdin.close()
        if process.wait() != 0:
            raise ValueError("git cat-file failed; no complete report was produced")
    finally:
        process.stdin.close()
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _qualified(node: ast.AST, bindings: dict[str, str]) -> str:
    parts = _dotted(node).split(".")
    return ".".join([bindings.get(parts[0], parts[0]), *parts[1:]])


def _imports(
    tree: ast.Module, path: str, modules: set[str]
) -> tuple[set[str], dict[str, str], int]:
    dependencies: set[str] = set()
    bindings: dict[str, str] = {}
    private_statements = 0
    package = _module(path) if path.endswith("/__init__.py") else _module(path).rpartition(".")[0]
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.append(alias.name)
                bindings[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = resolve_name("." * node.level + base, package)
            for alias in node.names:
                qualified = f"{base}.{alias.name}"
                targets.append(qualified if qualified in modules else base)
                bindings[alias.asname or alias.name] = qualified
        dependencies.update(targets)
        if any(
            target.startswith("cayu.")
            and any(part.startswith("_") for part in target.split(".")[1:])
            for target in targets
        ):
            private_statements += 1
    return dependencies, bindings, private_statements


def _literal_names(tree: ast.Module, name: str) -> list[str]:
    declarations = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    unsupported = f"{name} must have a single standalone literal assignment without other uses"
    if len(declarations) != 1 or len(declarations[0].targets) != 1:
        raise ValueError(unsupported)
    declaration = declarations[0]
    # Other references may mutate the list directly or pass it to an alias/call.
    # Reject them conservatively instead of trying to execute or track Python state.
    for node in ast.walk(tree):
        if node is declaration.targets[0]:
            continue
        if (
            (isinstance(node, ast.Name) and node.id == name)
            or (
                isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        ast.ExceptHandler,
                        ast.MatchAs,
                        ast.MatchStar,
                    ),
                )
                and node.name == name
            )
            or (isinstance(node, ast.MatchMapping) and node.rest == name)
            or (isinstance(node, ast.alias) and (node.asname or node.name.split(".")[0]) == name)
        ):
            raise ValueError(unsupported)
    value = ast.literal_eval(declaration.value)
    if not isinstance(value, (list, tuple)) or any(type(item) is not str for item in value):
        raise ValueError(f"{name} must be a literal sequence of strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicate names")
    return list(value)


def _duration_snapshot(data: bytes) -> dict[str, int | float]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate test node ID in duration snapshot")
            result[key] = value
        return result

    values = json.loads(data, object_pairs_hook=unique_pairs)
    if not isinstance(values, dict) or any(
        not key or type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for key, value in values.items()
    ):
        raise ValueError(
            "Duration snapshot must map nonempty node IDs to finite nonnegative seconds"
        )
    total = math.fsum(values.values())
    return {"entries": len(values), "seconds": round(total, 6)}


class SourceInventory:
    def __init__(self, modules: set[str]) -> None:
        self.modules = modules
        self.metrics: Counter[str] = Counter(
            dict.fromkeys(
                (
                    "source_python_files",
                    "source_python_lines",
                    "test_python_files",
                    "test_python_lines",
                    "export_declaration_lines",
                    "test_private_module_import_statements",
                    "test_direct_provider_subclasses_outside_support",
                    "pending_round_reader_text_occurrences",
                    "app_private_engine_sole_call_methods",
                ),
                0,
            )
        )
        self.dependencies: dict[str, set[str]] = defaultdict(set)
        self.calls: dict[str, list[str]] = {name: [] for name in sorted(_WATCHED_CALLS)}
        self.forwarders: list[str] = []
        self.provider_subclasses: list[str] = []
        self.module_lines: dict[str, int] = {}
        self.large_test_modules: dict[str, int] = {}

    def observe(self, path: str, data: bytes) -> None:
        source = data.decode("utf-8")
        # Python physical lines end at LF, CRLF or CR, not Unicode string separators.
        lines = source.count("\n") + source.count("\r") - source.count("\r\n")
        if source and not source.endswith(("\n", "\r")):
            lines += 1
        if path.startswith("src/cayu/") and (
            path.endswith("/_exports.py") or path.endswith("/__init__.pyi")
        ):
            self.metrics["export_declaration_lines"] += lines
        if not path.endswith(".py"):
            return
        family = "source" if path.startswith("src/cayu/") else "test"
        self.metrics[f"{family}_python_files"] += 1
        self.metrics[f"{family}_python_lines"] += lines
        tree = ast.parse(source, filename=path)
        dependencies, bindings, private_count = _imports(tree, path, self.modules)
        if family == "test":
            self.metrics["test_private_module_import_statements"] += private_count
            if lines >= 10_000:
                self.large_test_modules[path] = lines
        else:
            self.module_lines[path] = lines
            self.metrics["pending_round_reader_text_occurrences"] += source.count(
                "pending_tool_round_from_checkpoint"
            )
            if path.startswith(("src/cayu/storage/", "src/cayu/sessions/")):
                self.dependencies[path].update(
                    module
                    for module in dependencies
                    if module == "cayu.runtime" or module.startswith("cayu.runtime.")
                )
        if path == "src/cayu/_exports.py":
            self.metrics["root_public_names"] = len(_literal_names(tree, "PUBLIC_NAMES"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and family == "source":
                name = _qualified(node.func, bindings).rsplit(".", 1)[-1]
                if name in self.calls:
                    self.calls[name].append(f"{path}:{node.lineno}")
            if not isinstance(node, ast.ClassDef):
                continue
            methods = [
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if (
                family == "test"
                and not path.startswith("tests/support/")
                and any(_qualified(base, bindings) in _PROVIDER_NAMES for base in node.bases)
            ):
                self.provider_subclasses.append(f"{path}:{node.lineno}:{node.name}")
            if path == "src/cayu/sessions/base.py" and node.name == "SessionStore":
                self.metrics["session_store_public_methods"] = len(
                    {method.name for method in methods if not method.name.startswith("_")}
                )
            if (
                path == "src/cayu/runtime/_recovery_coordinator.py"
                and node.name == "RecoveryCoordinator"
            ):
                constructor = next(method for method in methods if method.name == "__init__")
                self.metrics["recovery_constructor_named_parameters"] = sum(
                    argument.arg != "self"
                    for argument in (
                        constructor.args.posonlyargs
                        + constructor.args.args
                        + constructor.args.kwonlyargs
                    )
                )
            if path == "src/cayu/applications.py" and node.name == "CayuApp":
                for method in methods:
                    calls = [child for child in ast.walk(method) if isinstance(child, ast.Call)]
                    if (
                        method.name.startswith("_")
                        and len(calls) == 1
                        and _dotted(calls[0].func).startswith("self._session_engine.")
                    ):
                        self.forwarders.append(f"{path}:{method.lineno}:{method.name}")

    def report(self) -> dict[str, Any]:
        self.metrics["app_private_engine_sole_call_methods"] = len(self.forwarders)
        self.metrics["test_direct_provider_subclasses_outside_support"] = len(
            self.provider_subclasses
        )
        for family in ("storage", "sessions"):
            self.metrics[f"{family}_runtime_modules"] = len(
                set().union(
                    *(
                        modules
                        for path, modules in self.dependencies.items()
                        if path.startswith(f"src/cayu/{family}/")
                    )
                )
            )
        for name, sites in self.calls.items():
            self.metrics[f"calls:{name}"] = len(sites)
        return {
            "metrics": dict(sorted(self.metrics.items())),
            "ownership_call_sites": {name: sorted(sites) for name, sites in self.calls.items()},
            "runtime_dependencies": {
                path: sorted(modules)
                for path, modules in sorted(self.dependencies.items())
                if modules
            },
            "app_private_engine_sole_call_methods": sorted(self.forwarders),
            "source_module_lines": dict(sorted(self.module_lines.items())),
            "test_modules_at_least_10000_lines": dict(sorted(self.large_test_modules.items())),
        }


def build_report(repo: Path, revision: str) -> dict[str, Any]:
    commit = _git(repo, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}")
    # Subsequent Git commands receive this object ID, never a caller-controlled option.
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
        raise ValueError("Revision did not resolve to a commit object ID")
    paths = _git(repo, "ls-tree", "-r", "--name-only", "-z", commit).split("\0")
    inventory = SourceInventory({_module(path) for path in paths if path.endswith(".py")})
    durations = None
    with closing(_tracked_blobs(repo, commit)) as blobs:
        for path, data in blobs:
            if path == ".test_durations":
                durations = _duration_snapshot(data)
            else:
                inventory.observe(path, data)
    required = {
        "root_public_names",
        "session_store_public_methods",
        "recovery_constructor_named_parameters",
    }
    if not required <= inventory.metrics.keys() or durations is None:
        raise ValueError(
            "Required baseline owners or duration snapshot are absent at this revision"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": commit,
        **inventory.report(),
        "committed_duration_snapshot": durations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--revision", default="HEAD", help="Commit or ref to measure (default: HEAD)"
    )
    selection.add_argument(
        "--check", type=Path, help="Verify a saved report at its recorded revision"
    )
    args = parser.parse_args()
    try:
        expected = json.loads(args.check.read_text()) if args.check else None
        if args.check and (
            not isinstance(expected, dict)
            or type(expected.get("schema_version")) is not int
            or expected.get("schema_version") != SCHEMA_VERSION
            or not isinstance(expected.get("revision"), str)
        ):
            raise ValueError("Saved report has an unsupported schema or missing revision")
        revision = expected["revision"] if expected is not None else args.revision
        report = build_report(args.repo, revision)
        if expected is not None:
            if report != expected:
                raise ValueError("Saved report differs from the measured commit")
            print(f"Baseline reproduced at {report['revision']}")
        else:
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, KeyError, SyntaxError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"size_report: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
