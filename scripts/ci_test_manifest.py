"""Collect CI assignments once and validate scoped collection in each worker.

Loaded explicitly with ``pytest -p scripts.ci_test_manifest``. Normal pytest
invocations do not load this plugin or require a manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import pytest
from _pytest.mark.expression import Expression
from pytest_split.algorithms import LeastDurationAlgorithm
from scripts.run_ci import _GENERAL_MARKER, _GENERAL_SHARDS, _SPECIALIST_LANES

_MANIFEST = pytest.StashKey[dict]()
_STARTED = pytest.StashKey[float]()
_COLLECTION_PATHS = pytest.StashKey[set[Path]]()
_ROUTING_MARKERS = frozenset(
    {"stress", "qualification", "postgres", "browser_docker", "host_journey"}
)
_SOURCE_PATHS = (
    "src",
    "tests",
    "scripts",
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "conftest.py",
    "uv.lock",
    ".test_durations",
)


def lane_specs() -> dict[str, tuple[str, int, int]]:
    return {
        **{
            f"general-{group}": (_GENERAL_MARKER, _GENERAL_SHARDS, group)
            for group in range(1, _GENERAL_SHARDS + 1)
        },
        **_SPECIALIST_LANES,
    }


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("CI collection")
    group.addoption("--ci-write-manifest", type=Path)
    group.addoption("--ci-manifest", type=Path)
    group.addoption("--ci-lane")


def _source_identity(root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(("git", *args), cwd=root, text=True).strip()

    if git("status", "--porcelain", "--untracked-files=all", "--", *_SOURCE_PATHS):
        raise pytest.UsageError("CI manifests require a clean source and test checkout")
    return {
        "revision": git("rev-parse", "HEAD"),
        "python": list(sys.version_info[:3]),
        "platform": [platform.system(), platform.machine()],
        "pytest": version("pytest"),
        "pytest_split": version("pytest-split"),
        "durations_sha256": hashlib.sha256((root / ".test_durations").read_bytes()).hexdigest(),
    }


def _file(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def _matches(expression: Expression, marks: list[str]) -> bool:
    # Canonical lanes use Boolean combinations of marker names. Let pytest parse
    # those expressions rather than implementing a second expression language.
    return expression.evaluate(lambda name, **_kwargs: name in marks)


def _validate_manifest(data: dict, identity: dict) -> None:
    if data["schema_version"] != 1 or data["source"] != identity:
        raise ValueError("manifest source, environment or duration snapshot differs")
    inventory = data["inventory"]
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("manifest inventory must be nonempty")
    for nodeid, marks in inventory.items():
        relative = Path(_file(nodeid))
        if (
            not isinstance(nodeid, str)
            or "::" not in nodeid
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "tests"
            or not isinstance(marks, list)
            or any(mark not in _ROUTING_MARKERS for mark in marks)
        ):
            raise ValueError("invalid inventory entry")
    specs = lane_specs()
    lanes = data["lanes"]
    if not isinstance(lanes, dict) or lanes.keys() != specs.keys():
        raise ValueError("manifest lanes differ from canonical CI lanes")
    expressions = {marker for marker, _, _ in specs.values()}
    for marker in expressions:
        expression = Expression.compile(marker)
        expected = [nodeid for nodeid, marks in inventory.items() if _matches(expression, marks)]
        assigned: list[str] = []
        for lane, (lane_marker, _, _) in specs.items():
            if lane_marker != marker:
                continue
            nodes = lanes[lane]
            if not isinstance(nodes, list) or any(not isinstance(node, str) for node in nodes):
                raise ValueError(f"invalid assignment for {lane}")
            wanted = set(nodes)
            if nodes != [node for node in expected if node in wanted]:
                raise ValueError(f"invalid assignment order or membership for {lane}")
            assigned.extend(nodes)
        if Counter(assigned) != Counter(expected):
            raise ValueError(f"manifest has missing or duplicate assignments for {marker}")
    if Counter(node for nodes in lanes.values() for node in nodes) != Counter(inventory.keys()):
        raise ValueError("canonical CI lanes must cover every collected node exactly once")


def pytest_configure(config: pytest.Config) -> None:
    output = config.getoption("ci_write_manifest")
    manifest = config.getoption("ci_manifest")
    lane = config.getoption("ci_lane")
    if not output and not manifest and not lane:
        return
    if bool(manifest) != bool(lane) or (output and manifest):
        raise pytest.UsageError("use --ci-write-manifest or --ci-manifest with --ci-lane")
    if any(
        config.getoption(option, None)
        for option in (
            "markexpr",
            "keyword",
            "splits",
            "ignore",
            "ignore_glob",
            "deselect",
            "lf",
            "stepwise",
        )
    ):
        raise pytest.UsageError("CI manifests cannot be combined with additional test selection")
    try:
        identity = _source_identity(config.rootpath)
        if output:
            if (
                not config.option.collectonly
                or config.getoption("numprocesses", 0)
                or [Path(arg).resolve() for arg in config.args] != [config.rootpath / "tests"]
                or os.environ.get("CAYU_REQUIRE_CURRENT_TEST_DURATIONS", "").lower()
                not in {"1", "true", "yes", "on"}
            ):
                raise ValueError(
                    "manifest generation requires unfiltered serial collection and the duration gate"
                )
            output.unlink(missing_ok=True)
            config.stash[_MANIFEST] = {"schema_version": 1, "source": identity}
        else:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            _validate_manifest(data, identity)
            if lane not in data["lanes"] or not data["lanes"][lane]:
                raise ValueError(f"unknown or empty CI lane: {lane}")
            if not hasattr(config, "workerinput") and [
                Path(arg).resolve() for arg in config.args
            ] != [config.rootpath / "tests"]:
                raise ValueError("CI manifest consumers cannot also select test paths")
            config.stash[_MANIFEST] = data
            paths: set[Path] = set()
            for node in data["lanes"][lane]:
                relative = Path(_file(node))
                paths.add(config.rootpath / relative)
                paths.update(config.rootpath / parent for parent in relative.parents)
            config.stash[_COLLECTION_PATHS] = paths
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        raise pytest.UsageError(f"invalid CI manifest: {exc}") from exc


def pytest_sessionstart(session: pytest.Session) -> None:
    if _MANIFEST in session.config.stash:
        session.config.stash[_STARTED] = time.perf_counter()


@pytest.hookimpl(tryfirst=True)
def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    # Traverse tests once. Hundreds of explicit file arguments make pytest
    # repeatedly search the same package collectors before selecting each file.
    paths = config.stash.get(_COLLECTION_PATHS, None)
    if paths is not None and collection_path not in paths:
        return True
    return None


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    original = list(items) if config.getoption("ci_write_manifest") else None
    yield
    if original is not None and Counter(items) != Counter(original):
        raise pytest.UsageError("CI manifest generation requires the full unfiltered inventory")
    if not config.getoption("ci_manifest"):
        return
    data = config.stash[_MANIFEST]
    nodes = data["lanes"][config.getoption("ci_lane")]
    files = {_file(node) for node in nodes}
    expected = [node for node in data["inventory"] if _file(node) in files]
    if Counter(item.nodeid for item in items) != Counter(expected):
        raise pytest.UsageError("scoped collection differs from the preflight inventory")
    for item in items:
        marks = sorted({mark.name for mark in item.iter_markers()} & _ROUTING_MARKERS)
        if marks != data["inventory"][item.nodeid]:
            raise pytest.UsageError(f"CI routing markers changed for {item.nodeid}")
    wanted = set(nodes)
    by_id = {item.nodeid: item for item in items}
    # Fixture parametrization can reorder a smaller collection differently.
    # Restore the exact order selected from the full preflight inventory.
    selected = [by_id[node] for node in nodes]
    config.hook.pytest_deselected(items=[item for item in items if item.nodeid not in wanted])
    items[:] = selected


def pytest_collection_finish(session: pytest.Session) -> None:
    config = session.config
    if _MANIFEST not in config.stash:
        return
    elapsed = time.perf_counter() - config.stash[_STARTED]
    if hasattr(config, "workeroutput"):
        config.workeroutput["ci_collection_seconds"] = elapsed
    print(f"\nCI collection: {elapsed:.3f}s; {len(session.items)} selected tests", flush=True)
    if not config.getoption("ci_write_manifest") or session.testsfailed:
        return
    data = config.stash[_MANIFEST]
    inventory = {
        item.nodeid: sorted({mark.name for mark in item.iter_markers()} & _ROUTING_MARKERS)
        for item in session.items
    }
    if len(inventory) != len(session.items):
        raise pytest.UsageError("duplicate node IDs in CI collection")
    durations = json.loads((config.rootpath / ".test_durations").read_text(encoding="utf-8"))
    partitions = {}
    lanes = {}
    for lane, (marker, splits, group) in lane_specs().items():
        key = (marker, splits)
        if key not in partitions:
            expression = Expression.compile(marker)
            items = [item for item in session.items if _matches(expression, inventory[item.nodeid])]
            partitions[key] = LeastDurationAlgorithm()(splits, items, durations)
        lanes[lane] = [item.nodeid for item in partitions[key][group - 1].selected]
    data.update(inventory=inventory, lanes=lanes)
    try:
        _validate_manifest(data, data["source"])
    except ValueError as exc:
        raise pytest.UsageError(f"invalid generated CI manifest: {exc}") from exc


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    output = session.config.getoption("ci_write_manifest")
    if output and exitstatus == pytest.ExitCode.OK:
        data = session.config.stash[_MANIFEST]
        if "lanes" not in data:
            raise pytest.UsageError("CI collection did not produce an inventory")
        if _source_identity(session.config.rootpath) != data["source"]:
            raise pytest.UsageError("source or duration snapshot changed during collection")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(data) + "\n", encoding="utf-8")
        temporary.replace(output)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error) -> None:
    del error
    elapsed = getattr(node, "workeroutput", {}).get("ci_collection_seconds")
    if elapsed is not None:
        print(f"\nCI worker {node.gateway.id} collection: {elapsed:.3f}s", flush=True)
