"""Fresh-process smoke check; run with the installed wheel on the import path."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-layout", action="store_true")
    parser.add_argument("--all-exports", action="store_true", help="Requires cayu[all].")
    args = parser.parse_args()
    started = time.perf_counter()
    from cayu.events import Event

    elapsed = time.perf_counter() - started
    for module_name in (
        "cayu.evals.browser_acceptance",
        "cayu.evals.browser_acceptance_fixture",
        "cayu.evals.browser_acceptance_manifests",
        "cayu.evals.causal_memory_campaign",
    ):
        assert module_name not in sys.modules, module_name
    print(
        f"core import: {elapsed:.3f}s; Cayu modules: {sum(n.startswith('cayu') for n in sys.modules)}"
    )

    from cayu import CayuApp, WorkflowEvalTarget
    from cayu.evals import (
        BrowserAcceptanceFixtureV1,
        deterministic_browser_acceptance_manifest,
        run_causal_memory_reference_campaign,
    )

    assert Event.__name__ == "Event"
    assert CayuApp.__name__ == "CayuApp"
    assert WorkflowEvalTarget.__name__ == "WorkflowEvalTarget"
    assert BrowserAcceptanceFixtureV1.__name__ == "BrowserAcceptanceFixtureV1"
    assert callable(deterministic_browser_acceptance_manifest)
    assert callable(run_causal_memory_reference_campaign)
    print("supported lazy public imports passed")
    if not (args.installed_layout or args.all_exports):
        return
    import cayu

    installed = Path(cayu.__file__).resolve().parent
    checkout = Path(__file__).resolve().parents[1]
    assert installed != checkout / "src/cayu", "smoke must exercise the installed wheel"
    moves = json.loads((checkout / "docs/public-api-migration.json").read_text())
    packages = json.loads((checkout / "docs/public-api-packages.json").read_text())
    # Resolve each package in its own interpreter: loading implementations first
    # can hide broken lazy exports by populating package attributes as a side effect.
    for package_name in packages if args.all_exports else ():
        script = f"""
import importlib
package = importlib.import_module({package_name!r})
exports = importlib.import_module({package_name!r} + '._exports').EXPORTS
for name, (module_name, symbol) in exports.items():
    assert module_name != package.__name__, name
    actual = getattr(package, name)
    assert actual is getattr(importlib.import_module(module_name), symbol), name
"""
        subprocess.run([sys.executable, "-I", "-c", script], check=True, timeout=60)
    if args.all_exports:
        subprocess.run(
            [sys.executable, "-I", "-c", "from cayu.storage import postgres; assert postgres"],
            check=True,
            timeout=60,
        )
    for original, canonical in moves.items():
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                f"import importlib; importlib.import_module({canonical!r})",
            ],
            check=True,
            timeout=60,
        )
        module = importlib.import_module(canonical)
        assert module.__name__ == canonical
        assert Path(module.__file__).resolve().is_relative_to(installed)
        for suffix in (".py", ".pyi"):
            removed = installed.parent / (original.replace(".", "/") + suffix)
            assert not removed.exists(), removed
    assert not (installed / "core").exists()
    assert (installed / "py.typed").is_file()
    for package_name in packages:
        package = importlib.import_module(package_name)
        exports = importlib.import_module(package_name + "._exports")
        stub = ast.parse(Path(package.__file__).with_suffix(".pyi").read_text())
        declared = {
            alias.asname or alias.name
            for node in stub.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert declared == set(exports.EXPORTS), package_name
        assert set(package.__all__) <= declared, package_name
    print("complete installed concept layout and lazy public imports passed")


if __name__ == "__main__":
    main()
