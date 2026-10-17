"""Run one planned CI shard, or gate full collection against the shard plan.

Loaded explicitly with ``pytest -p scripts.ci_test_shards``. Normal pytest
invocations do not load this plugin.

``--ci-shard LANE`` collects only the lane's files, keeps the lane's piece of
any split file and applies the lane's marker expression. ``--ci-check-plan``
collects everything serially and fails when a collected test belongs to a
lane family whose shards would not collect its file. With ``--ci-lanes PATH``
it also writes the family that runs each collected test, which
``merge_test_durations.py`` turns into ``.test_lanes`` corrections.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import pytest
from _pytest.mark.expression import Expression
from scripts.ci_shard_plan import FAMILIES, Plan, build_plan, store_mode

_PLAN = pytest.StashKey[Plan]()
_LANE = pytest.StashKey[str]()
_STARTED = pytest.StashKey[float]()
_COLLECTION_PATHS = pytest.StashKey[set[Path]]()
# Matches pytest-split: longer setup or teardown is a stale clock, not a cost.
_MAX_PHASE_SECONDS = 600
_SELECTION_OPTIONS = (
    "markexpr",
    "keyword",
    "splits",
    "ignore",
    "ignore_glob",
    "deselect",
    "lf",
    "stepwise",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("CI shards")
    group.addoption("--ci-shard", help="run one CI lane from the shard plan")
    group.addoption(
        "--ci-check-plan",
        action="store_true",
        help="with --collect-only: fail if the shard plan would miss a collected test",
    )
    group.addoption("--ci-collected", type=Path, help="write collected node IDs to this file")
    group.addoption("--ci-durations", type=Path, help="write measured test durations here")
    group.addoption(
        "--ci-lanes", type=Path, help="with --ci-check-plan: write each test's lane family here"
    )


def _matches(expression: Expression, names: set[str]) -> bool:
    return expression.evaluate(lambda name, **_kwargs: name in names)


def _marks(item: pytest.Item) -> set[str]:
    return {mark.name for mark in item.iter_markers()}


def _whole_suite(config: pytest.Config) -> bool:
    return [Path(arg).resolve() for arg in config.args] == [config.rootpath / "tests"]


def pytest_configure(config: pytest.Config) -> None:
    lane = config.getoption("ci_shard")
    check = config.getoption("ci_check_plan")
    durations = config.getoption("ci_durations")
    if durations and not hasattr(config, "workerinput"):
        config.pluginmanager.register(_DurationRecorder(durations), "ci-test-durations")
    if not lane and not check:
        return
    if lane and check:
        raise pytest.UsageError("use --ci-shard or --ci-check-plan, not both")
    if any(config.getoption(option, None) for option in _SELECTION_OPTIONS):
        raise pytest.UsageError("CI shards cannot be combined with additional test selection")
    if not hasattr(config, "workerinput") and not _whole_suite(config):
        raise pytest.UsageError("CI shards cannot also select test paths")
    if check and (not config.option.collectonly or config.getoption("numprocesses", 0)):
        raise pytest.UsageError("--ci-check-plan requires serial --collect-only")
    try:
        plan = build_plan(config.rootpath, store_mode())
        if lane:
            selected = plan.lane(lane)
            paths: set[Path] = set()
            for path in selected.paths:
                relative = Path(path)
                paths.add(config.rootpath / relative)
                paths.update(config.rootpath / parent for parent in relative.parents)
            config.stash[_COLLECTION_PATHS] = paths
            config.stash[_LANE] = lane
    except (OSError, ValueError) as exc:
        raise pytest.UsageError(f"invalid CI shard plan: {exc}") from exc
    config.stash[_PLAN] = plan


def pytest_sessionstart(session: pytest.Session) -> None:
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
    # Run after the conftest hooks that add routing markers and select stores.
    yield
    plan = config.stash.get(_PLAN, None)
    if plan is None:
        return
    lane = config.stash.get(_LANE, None)
    if lane is None:
        _check_plan(plan, items, config.getoption("ci_lanes"))
        return
    expression = Expression.compile(plan.lane(lane).marker)
    selected = [
        item
        for item in items
        if plan.selects(lane, item.nodeid) and _matches(expression, _marks(item))
    ]
    wanted = set(map(id, selected))
    config.hook.pytest_deselected(items=[item for item in items if id(item) not in wanted])
    items[:] = selected


def _check_plan(plan: Plan, items: list[pytest.Item], lanes_output: Path | None) -> None:
    expressions = [(family.name, Expression.compile(family.marker)) for family in FAMILIES]
    counts: Counter[str] = Counter()
    missing: list[str] = []
    lanes: dict[str, str] = {}
    for item in items:
        path = item.nodeid.split("::", 1)[0]
        marks = _marks(item)
        for name, expression in expressions:
            if _matches(expression, marks):
                counts[name] += 1
                lanes.setdefault(item.nodeid, name)
                if path not in plan.members[name]:
                    missing.append(f"{item.nodeid} ({name})")
    summary = ", ".join(f"{family.name}={counts[family.name]}" for family in FAMILIES)
    print(f"\nCI shard plan ({plan.mode} stores): {summary}", flush=True)
    if missing:
        raise pytest.UsageError(
            f"{len(missing)} collected tests are outside their lane family's shard files; "
            "mention the marker or fixture in the test module (or a tests module it imports). "
            "First: " + "; ".join(missing[:10])
        )
    if lanes_output is not None:
        lanes_output.parent.mkdir(parents=True, exist_ok=True)
        lanes_output.write_text(
            json.dumps({"mode": plan.mode, "lanes": lanes}, sort_keys=True, indent=1) + "\n",
            encoding="utf-8",
        )


def pytest_collection_finish(session: pytest.Session) -> None:
    config = session.config
    if _PLAN not in config.stash and not config.getoption("ci_collected"):
        return
    elapsed = time.perf_counter() - config.stash[_STARTED]
    if hasattr(config, "workeroutput"):
        config.workeroutput["ci_collection_seconds"] = elapsed
    print(f"\nCI collection: {elapsed:.3f}s; {len(session.items)} selected tests", flush=True)
    output = config.getoption("ci_collected")
    if output and not hasattr(config, "workerinput") and not session.testsfailed:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(f"{item.nodeid}\n" for item in session.items), encoding="utf-8")


class _DurationRecorder:
    """Record call, setup and teardown time per node on the controller only."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.durations: dict[str, float] = {}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.duration < 0:
            return
        if report.when in {"setup", "teardown"} and report.duration > _MAX_PHASE_SECONDS:
            return
        self.durations[report.nodeid] = self.durations.get(report.nodeid, 0.0) + report.duration

    def pytest_sessionfinish(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(
            json.dumps(self.durations, sort_keys=True, indent=1) + "\n", encoding="utf-8"
        )


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _LANE in session.config.stash and exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        # Plan membership is conservative; a file can hold no test for this lane.
        print("\nCI shard: no tests match this lane in the current selection", flush=True)
        session.exitstatus = pytest.ExitCode.OK


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error) -> None:
    del error
    elapsed = getattr(node, "workeroutput", {}).get("ci_collection_seconds")
    if elapsed is not None:
        print(f"\nCI worker {node.gateway.id} collection: {elapsed:.3f}s", flush=True)
