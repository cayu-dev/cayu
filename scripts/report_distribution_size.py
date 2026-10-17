#!/usr/bin/env python3
"""Report the size of the Cayu distribution and its core import closure.

This is a report, not a gate: it prints measurements and never fails on size.
It measures a built wheel (``--wheel``) or the source tree (default) and,
unless ``--no-closure`` is given, the Cayu modules that a minimal application
actually loads. Usage and definitions live in docs/distribution-size.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
ROOT_MODULES = "<root modules>"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Runs in a separate, isolated interpreter so the measuring process does not
# pollute sys.modules. The stub provider and temporary SQLite store need no
# network access or credentials.
_CLOSURE_PROBE = r"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path


def snapshot(name):
    modules = {}
    for module_name, module in sorted(sys.modules.items()):
        if module_name == "cayu" or module_name.startswith("cayu."):
            modules[module_name] = getattr(module, "__file__", None)
    return {"name": name, "modules": modules}


stages = []
import cayu

stages.append(snapshot("import cayu"))

from cayu import AgentSpec, CayuApp, Message, RunRequest, run_to_completion
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.storage import SQLiteSessionStore


class StubProvider(ModelProvider):
    name = "stub"

    async def stream(self, request):
        yield ModelStreamEvent.text_delta("ok")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


async def run_one_turn(app):
    outcome = await run_to_completion(
        app,
        RunRequest(agent_name="assistant", messages=[Message.text("user", "Hello.")]),
    )
    if not outcome.ok:
        raise RuntimeError(f"stub turn did not complete: {outcome.status}: {outcome.error}")


async def close(app, store):
    await app.aclose()
    await store.close()


with tempfile.TemporaryDirectory() as directory:
    store = SQLiteSessionStore(Path(directory) / "sessions.db")
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(StubProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="stub-model"))
    stages.append(snapshot("minimal app"))
    asyncio.run(run_one_turn(app))
    stages.append(snapshot("one stub turn"))
    asyncio.run(close(app, store))

third_party = sorted(
    {
        name.partition(".")[0]
        for name in sys.modules
        if not name.startswith(("cayu.", "_"))
        and name != "cayu"
        and name.partition(".")[0] not in sys.stdlib_module_names
    }
)
Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "package_root": str(Path(cayu.__file__).parent),
            "stages": stages,
            "third_party_top_level_modules": third_party,
        }
    ),
    encoding="utf-8",
)
"""

CLOSURE_STAGE_DESCRIPTIONS = {
    "import cayu": "`import cayu` only",
    "minimal app": (
        "import the core API, construct CayuApp with a SQLite session store, "
        "register a stub provider and one agent"
    ),
    "one stub turn": "run one turn of that app to completion against the stub provider",
}


def physical_lines(data: bytes) -> int:
    """Count lines the way Python and ``wc -l`` do for newline-terminated files."""

    lines = data.count(b"\n") + data.count(b"\r") - data.count(b"\r\n")
    if data and not data.endswith((b"\n", b"\r")):
        lines += 1
    return lines


def subpackage_of(relative: PurePosixPath) -> str:
    """Name the top-level ``cayu`` subpackage that owns a package-relative path."""

    return relative.parts[0] if len(relative.parts) > 1 else ROOT_MODULES


@dataclass
class SubpackageSize:
    files: int = 0
    bytes: int = 0
    python_files: int = 0
    python_lines: int = 0


@dataclass
class Inventory:
    source: str
    archive_bytes: int | None = None
    subpackages: dict[str, SubpackageSize] = field(default_factory=dict)

    def add(self, relative: PurePosixPath, data: bytes) -> None:
        size = self.subpackages.setdefault(subpackage_of(relative), SubpackageSize())
        size.files += 1
        size.bytes += len(data)
        if relative.suffix == ".py":
            size.python_files += 1
            size.python_lines += physical_lines(data)

    def totals(self) -> SubpackageSize:
        total = SubpackageSize()
        for size in self.subpackages.values():
            total.files += size.files
            total.bytes += size.bytes
            total.python_files += size.python_files
            total.python_lines += size.python_lines
        return total


def _wheel_files(wheel: Path) -> Iterator[tuple[PurePosixPath, bytes]]:
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or path.parts[0] != "cayu":
                continue
            yield path.relative_to("cayu"), archive.read(info)


def _source_files(package: Path) -> Iterator[tuple[PurePosixPath, bytes]]:
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if not path.is_file() or "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        yield PurePosixPath(relative.as_posix()), path.read_bytes()


def inventory_wheel(wheel: Path) -> Inventory:
    inventory = Inventory(source=f"wheel {wheel.name}", archive_bytes=wheel.stat().st_size)
    _fill(inventory, _wheel_files(wheel))
    return inventory


def inventory_source(package: Path) -> Inventory:
    if not (package / "__init__.py").is_file():
        raise ValueError(f"{package} is not a Python package directory")
    inventory = Inventory(source=f"source tree {package}")
    _fill(inventory, _source_files(package))
    return inventory


def _fill(inventory: Inventory, files: Iterable[tuple[PurePosixPath, bytes]]) -> None:
    for relative, data in files:
        inventory.add(relative, data)
    if not inventory.subpackages:
        raise ValueError(f"{inventory.source} contains no cayu package files")


def run_closure_probe(python: str) -> dict[str, Any]:
    """Run the closure probe in an isolated interpreter and return its raw result."""

    # Configuration such as CAYU_REQUIRE_POSTGRES would change what the minimal
    # application loads, so the probe always runs with Cayu defaults.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CAYU_")}
    # The probe runs in a temporary directory, so resolve a relative interpreter first.
    executable = shutil.which(python)
    if executable is None:
        raise ValueError(f"Python interpreter not found: {python}")
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "closure.json"
        completed = subprocess.run(
            [os.path.abspath(executable), "-I", "-c", _CLOSURE_PROBE, str(output)],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"closure probe failed with exit code {completed.returncode}:\n"
                f"{completed.stderr.strip()}"
            )
        return json.loads(output.read_text(encoding="utf-8"))


def summarize_closure(probe: dict[str, Any]) -> dict[str, Any]:
    """Count modules and source lines per probe stage from the loaded module files."""

    package_root = Path(probe["package_root"])
    line_cache: dict[str, int] = {}
    stages = []
    for stage in probe["stages"]:
        per_subpackage: dict[str, dict[str, int]] = {}
        modules = lines = 0
        for filename in stage["modules"].values():
            if not filename or not filename.endswith(".py"):
                continue
            if filename not in line_cache:
                line_cache[filename] = physical_lines(Path(filename).read_bytes())
            relative = PurePosixPath(Path(filename).relative_to(package_root).as_posix())
            entry = per_subpackage.setdefault(subpackage_of(relative), {"modules": 0, "lines": 0})
            entry["modules"] += 1
            entry["lines"] += line_cache[filename]
            modules += 1
            lines += line_cache[filename]
        stages.append(
            {
                "name": stage["name"],
                "description": CLOSURE_STAGE_DESCRIPTIONS.get(stage["name"], ""),
                "modules": modules,
                "lines": lines,
                "subpackages": dict(sorted(per_subpackage.items())),
            }
        )
    return {
        "package_root": str(package_root),
        "stages": stages,
        "third_party_top_level_modules": probe["third_party_top_level_modules"],
    }


def build_report(inventory: Inventory, closure: dict[str, Any] | None) -> dict[str, Any]:
    totals = inventory.totals()
    return {
        "schema_version": SCHEMA_VERSION,
        "source": inventory.source,
        "archive_bytes": inventory.archive_bytes,
        "totals": vars(totals),
        "subpackages": {
            name: vars(size)
            for name, size in sorted(
                inventory.subpackages.items(), key=lambda item: (-item[1].python_lines, item[0])
            )
        },
        "closure": closure,
    }


def _megabytes(value: int) -> str:
    return f"{value / 1_000_000:.1f} MB"


def render_text(report: dict[str, Any]) -> str:
    totals = report["totals"]
    lines = [f"Cayu distribution size ({report['source']})", ""]
    if report["archive_bytes"] is not None:
        lines.append(f"Wheel size:          {_megabytes(report['archive_bytes'])}")
    lines.extend(
        (
            f"Unpacked cayu files: {totals['files']:,} files, {_megabytes(totals['bytes'])}",
            f"Python source:       {totals['python_files']:,} files, "
            f"{totals['python_lines']:,} lines",
            "",
        )
    )
    closure = report["closure"]
    final_stage = closure["stages"][-1] if closure else None
    header = f"{'Subpackage':<26}{'Py files':>10}{'Py lines':>12}{'Share':>8}{'Size':>10}"
    if final_stage:
        header += f"{'Loaded lines':>15}"
    lines.extend((header, "-" * len(header)))
    for name, size in report["subpackages"].items():
        share = size["python_lines"] / totals["python_lines"] if totals["python_lines"] else 0
        row = (
            f"{name:<26}{size['python_files']:>10,}{size['python_lines']:>12,}"
            f"{share:>8.1%}{_megabytes(size['bytes']):>10}"
        )
        if final_stage:
            loaded = final_stage["subpackages"].get(name, {"lines": 0})["lines"]
            row += f"{loaded:>15,}"
        lines.append(row)
    if closure:
        lines.extend(("", f"Core import closure (loaded from {closure['package_root']})"))
        for stage in closure["stages"]:
            share = stage["lines"] / totals["python_lines"] if totals["python_lines"] else 0
            lines.append(
                f"  {stage['name']:<14} {stage['modules']:>5,} modules "
                f"{stage['lines']:>10,} lines ({share:.0%} of shipped Python)"
            )
        lines.append("")
        lines.extend(
            f"  {stage['name']}: {stage['description']}"
            for stage in closure["stages"]
            if stage["description"]
        )
        third_party = closure["third_party_top_level_modules"]
        lines.append(
            f"  Third-party top-level modules loaded ({len(third_party)}): {', '.join(third_party)}"
        )
        if final_stage:
            lines.append(f"  The 'Loaded lines' column shows the '{final_stage['name']}' stage.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--wheel", type=Path, help="Built Cayu wheel to measure")
    target.add_argument(
        "--source",
        type=Path,
        default=_REPOSITORY_ROOT / "src" / "cayu",
        help="Cayu package directory to measure (default: this checkout's src/cayu)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter with Cayu installed for the closure probe (default: this one)",
    )
    parser.add_argument(
        "--no-closure", action="store_true", help="Skip the import closure measurement"
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = parser.parse_args(argv)
    try:
        inventory = (
            inventory_wheel(args.wheel) if args.wheel is not None else inventory_source(args.source)
        )
        closure = None if args.no_closure else summarize_closure(run_closure_probe(args.python))
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
    ) as error:
        parser.exit(1, f"report_distribution_size: {error}\n")
    report = build_report(inventory, closure)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
