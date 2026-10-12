"""Reproduce committed memory ownership sizes and own-parent delivery growth."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REVISION = "5b50132aefd95def802faa92a4756f89e990e94c"
DELIVERIES = (
    "e50bc3527",
    "8dd42c6f4",
    "9b1e61af3",
    "e45d2488f",
    "bc9479c92",
    "dad0fadf2",
    "9f3437c2b",
    "f763477fc",
    "42b30e7e5",
    "36409368a",
    "d027937d7",
    "0bbd1bec6",
    "1c0e9bfb0",
)
CORRECTIONS = ("5b50132aefd95def802faa92a4756f89e990e94c",)
MEMORY = "src/cayu/sessions/memory.py"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def source(revision, path):
    return git("show", f"{revision}:{path}")


def owned_path(path):
    return path == MEMORY or path.startswith("src/cayu/sessions/_memory_")


def inventory(revision):
    revision = git("rev-parse", revision).strip()
    baseline = git("rev-parse", f"{DELIVERIES[0]}^").strip()
    paths = git("ls-tree", "-r", "--name-only", revision, "src/cayu").splitlines()
    owners = {}
    for path in sorted(p for p in paths if owned_path(p) and p != MEMORY):
        text = source(revision, path)
        tree = ast.parse(text)
        state_fields = {
            node.name: len([field for field in node.body if isinstance(field, ast.AnnAssign)])
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }
        dependencies = sorted(
            {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("cayu.")
                and any(
                    part in node.module
                    for part in (
                        ".memory",
                        "._memory_",
                        ".store",
                        ".sqlite",
                        ".postgres",
                        "._creation_fence",
                    )
                )
            }
            | {
                f"{node.module}.{alias.name}"
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module == "cayu.sessions"
                for alias in node.names
                if alias.name.startswith("_memory_")
            }
        )
        owners[path] = {
            "physical_lines": len(text.splitlines()),
            "async_operations": len([n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)]),
            "state_and_protocol_fields": state_fields,
            "storage_imports": dependencies,
        }
    memory = source(revision, MEMORY)
    cls = next(
        n
        for n in ast.parse(memory).body
        if isinstance(n, ast.ClassDef) and n.name == "InMemorySessionStore"
    )
    methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    largest = sorted(methods, key=lambda n: n.end_lineno - n.lineno, reverse=True)
    deliveries = []
    corrections = []
    for recorded_commit in (*DELIVERIES, *CORRECTIONS):
        commit = git("rev-parse", recorded_commit).strip()
        # Reject an accidentally unrelated target revision.
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, revision], cwd=ROOT, check=True
        )
        counts = git("diff", "--numstat", f"{commit}^", commit, "--", "src/cayu").splitlines()
        changes = [line.split("\t") for line in counts]
        assert all(owned_path(path) for _, _, path in changes), changes
        destination = corrections if recorded_commit in CORRECTIONS else deliveries
        destination.append(
            {
                "commit": commit,
                "subject": git("show", "-s", "--format=%s", commit).strip(),
                "production_added": sum(int(a) for a, _, _ in changes),
                "production_removed": sum(int(d) for _, d, _ in changes),
                "net_production_lines": sum(int(a) - int(d) for a, d, _ in changes),
                "adapter_delta": sum(int(a) - int(d) for a, d, p in changes if p == MEMORY),
            }
        )
    baseline_paths = git("ls-tree", "-r", "--name-only", baseline, "src/cayu").splitlines()
    before = sum(len(source(baseline, p).splitlines()) for p in baseline_paths if owned_path(p))
    after = len(memory.splitlines()) + sum(o["physical_lines"] for o in owners.values())
    delivery_growth = sum(d["net_production_lines"] for d in deliveries)
    correction_growth = sum(d["net_production_lines"] for d in corrections)
    return {
        "revision": revision,
        "baseline_before_first_memory_extraction": baseline,
        "scope": "Memory session adapter and _memory_ operation owners; physical lines include comments and blanks.",
        "adapter_lines_before": len(source(baseline, MEMORY).splitlines()),
        "adapter_lines_after": len(memory.splitlines()),
        "owned_source_lines_before": before,
        "owned_source_lines_after": after,
        "own_parent_refactoring_growth": delivery_growth,
        "own_parent_correction_growth": correction_growth,
        "total_own_parent_growth": delivery_growth + correction_growth,
        "interleaved_growth_in_same_files": after - before - delivery_growth - correction_growth,
        "memory_owners": owners,
        "retained_base_classes": [ast.unparse(base) for base in cls.bases],
        "retained_method_count": len(methods),
        "retained_methods_at_least_80_lines": [
            {
                "name": n.name,
                "physical_lines": n.end_lineno - n.lineno + 1,
                "await_expressions": sum(isinstance(child, ast.Await) for child in ast.walk(n)),
            }
            for n in largest
            if n.end_lineno - n.lineno + 1 >= 80
        ],
        "native_adapter_lines": {
            backend: len(source(revision, f"src/cayu/storage/{backend}.py").splitlines())
            for backend in ("sqlite", "postgres")
        },
        "deliveries": deliveries,
        "corrections": corrections,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=REVISION)
    print(json.dumps(inventory(parser.parse_args().revision), indent=2))
