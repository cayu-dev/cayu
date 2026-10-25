#!/usr/bin/env python3
"""Plan CI test shards from the repository and its duration snapshot.

Every shard derives the same plan from the checked-out files and
``.test_durations`` without collecting the suite first. Whole test files are
bin-packed by their summed durations; a file is split into deterministic
pieces only when one lane would spend more than its family's split threshold
(``SPLIT_SECONDS`` by default) on it.

Which lane family runs a test depends on markers that only collection
resolves, so the planner estimates it from the node ID and the file's source.
``.test_lanes`` records, per store mode, the tests whose collected family (or
absence) differs from that estimate. The collection job writes the observed
families and ``merge_test_durations.py`` refreshes the corrections.

The store switch lives here too so pytest collection, the collection gate and
the planner agree on which store variants a pull request runs.

Keep this module stdlib-only and Python 3.9 compatible: ``run_ci.py`` imports
it under the runner's system interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

STORES_ENV = "CAYU_TEST_STORES"
STORE_MODES = ("sqlite", "all")
# Developers keep every store locally; CI sets the mode for each lane.
DEFAULT_STORE_MODE = "all"
_STORES = frozenset({"memory", "sqlite", "postgres"})
_KEPT_STORE = "sqlite"
_MASK = "<store>"

GENERAL_MARKER = "not (stress or qualification or postgres or browser_docker or host_journey)"
POSTGRES_MARKER = "postgres and not (stress or qualification or host_journey)"

# Unit and shard sizes are in seconds of single-worker test time.
SPLIT_SECONDS = 8 * 60
DEFAULT_FILE_SECONDS = 5.0
# Each xdist worker imports every file of its shard before running tests.
COLLECT_SECONDS_PER_FILE = 0.3
DEFAULT_WORKERS = 4
LANES_FILE = ".test_lanes"
TARGET_SHARD_MINUTES = 8.0


@dataclass(frozen=True)
class Family:
    name: str
    marker: str
    shards: Mapping[str, int]
    workers: int | None = None
    pattern: str | None = None
    qualification_paths: bool = False
    # Work stealing spreads a large file across workers. "loadfile" keeps each
    # file on one worker, for module fixtures that must not run twice at once.
    distribution: str = "worksteal"
    # Single-worker seconds of one file a lane may hold before the file splits.
    # A single-worker family needs a threshold below its target lane time.
    split_seconds: float = SPLIT_SECONDS
    target_minutes: float = TARGET_SHARD_MINUTES

    def lane(self, shard: int) -> str:
        if max(self.shards.values()) == 1:
            return self.name
        return f"{self.name}-{shard}"

    def lanes(self, mode: str) -> tuple[str, ...]:
        return tuple(self.lane(shard) for shard in range(1, self.shards[mode] + 1))


FAMILIES = (
    Family("general", GENERAL_MARKER, {"sqlite": 43, "all": 32}),
    Family(
        "postgres-conformance",
        POSTGRES_MARKER,
        {"sqlite": 6, "all": 12},
        pattern=r"[Pp]ostgres",
    ),
    Family(
        "host-journey",
        "host_journey and not (stress or qualification)",
        {"sqlite": 2, "all": 2},
        1,
        r"\bhost_journey\b",
    ),
    # Its module fixture builds one Docker image tag; concurrent builds collide.
    Family(
        "browser-docker",
        "browser_docker",
        {"sqlite": 1, "all": 1},
        None,
        r"\bbrowser_docker\b",
        distribution="loadfile",
    ),
    # One worker per lane keeps internally concurrent capacity scenarios
    # isolated, so lane wall time is the lane's summed test time.
    Family(
        "qualification",
        "stress or qualification",
        {"sqlite": 12, "all": 12},
        1,
        r"mark\.(?:stress|qualification)\b",
        qualification_paths=True,
        split_seconds=4 * 60,
        target_minutes=6.0,
    ),
)
# Pull request and release CI run these; qualification has its own workflow.
CI_FAMILIES = ("general", "postgres-conformance", "host-journey", "browser-docker")


def family(name: str) -> Family:
    for candidate in FAMILIES:
        if candidate.name == name:
            return candidate
    raise ValueError(f"unknown test family: {name}")


def lane_family(lane: str) -> tuple[Family, int]:
    for candidate in FAMILIES:
        for mode in STORE_MODES:
            for shard in range(1, candidate.shards[mode] + 1):
                if candidate.lane(shard) == lane:
                    return candidate, shard
    raise ValueError(f"unknown CI lane: {lane}")


def all_lanes() -> tuple[str, ...]:
    names: list[str] = []
    for candidate in FAMILIES:
        for shard in range(1, max(candidate.shards.values()) + 1):
            names.append(candidate.lane(shard))
    return tuple(names)


def store_mode(environ: Mapping[str, str] | None = None) -> str:
    environ = os.environ if environ is None else environ
    value = environ.get(STORES_ENV, "").strip().lower() or DEFAULT_STORE_MODE
    if value not in STORE_MODES:
        raise ValueError(f"{STORES_ENV} must be one of {', '.join(STORE_MODES)}; got {value!r}")
    return value


def store_slot(values: Sequence[Hashable]) -> tuple[str | None, tuple[Hashable, ...]]:
    """Return the one store named by parameter values, with that store masked.

    A plain string parameter or one element of a tuple parameter can name the
    store. Values naming no store, or more than one (a migration from one store
    to another), are not store variants.
    """

    found: list[str] = []
    masked: list[Hashable] = []
    for value in values:
        if isinstance(value, str) and value in _STORES:
            found.append(value)
            masked.append(_MASK)
        elif isinstance(value, tuple) and any(
            isinstance(element, str) and element in _STORES for element in value
        ):
            inner = []
            for element in value:
                if isinstance(element, str) and element in _STORES:
                    found.append(element)
                    inner.append(_MASK)
                else:
                    inner.append(element)
            masked.append(tuple(inner))
        else:
            masked.append(value)
    if len(found) != 1:
        return None, tuple(values)
    return found[0], tuple(masked)


@dataclass(frozen=True)
class StoreVariant:
    key: Hashable
    group: Hashable
    store: str | None
    protected: bool = False


def deselected_store_variants(variants: Iterable[StoreVariant], mode: str) -> set[Hashable]:
    """Select the memory and Postgres variants a mode skips.

    Under ``sqlite`` a variant is skipped only when the same test, with every
    other parameter identical, also has a SQLite variant. ``protected``
    variants (``store_semantics``) always run.
    """

    if mode not in STORE_MODES:
        raise ValueError(f"unknown store mode: {mode}")
    if mode == "all":
        return set()
    variants = list(variants)
    covered = {variant.group for variant in variants if variant.store == _KEPT_STORE}
    return {
        variant.key
        for variant in variants
        if variant.store in _STORES - {_KEPT_STORE}
        and not variant.protected
        and variant.group in covered
    }


# Storage modules, and tests about races, concurrency, locking or clocks, keep
# every store on pull requests along with tests marked store_semantics.
_EVERY_STORE_FILES = re.compile(
    r"(?:^|_)(?:conformance|stores?|storage|repository|migrations?|ledger|races?|"
    r"concurrency|contention)(?:_|\.py$)"
)
_EVERY_STORE_TESTS = re.compile(
    r"race|racing|concurren|contention|contend|competing|compete|simultan|interleav|"
    r"serializ|two_connection|(?:^|_)lock(?:s|ed|ing)?(?:_|$)|clock|"
    # Elections and winners between owners, workers, stores or processes.
    r"(?:^|_)(?:elect|elects|elected|election|winners?|losers?|losing|takeover|converge)"
    r"(?:_|$)|independent_(?:applications|apps|owners?|workers|hosts|store_instances|"
    r"lifecycle_owner|heartbeat)|two_(?:store|instance|process)|across_two_processes|"
    r"another_(?:app|instance|control|owner|process)",
    re.IGNORECASE,
)


def keeps_every_store(nodeid: str) -> bool:
    """Whether a test's store variants all run under every store mode."""

    path, _, rest = nodeid.partition("::")
    name = rest.split("[", 1)[0].rsplit("::", 1)[-1]
    return bool(
        _EVERY_STORE_FILES.search(path.rsplit("/", 1)[-1]) or _EVERY_STORE_TESTS.search(name)
    )


def nodeid_variant(nodeid: str, *, protected: bool = False) -> StoreVariant:
    """Approximate a store variant from a node id (used for planning only)."""

    function, bracket, rest = nodeid.partition("[")
    if not bracket or not rest.endswith("]"):
        return StoreVariant(nodeid, (nodeid, ()), None, protected)
    store, masked = store_slot(tuple(rest[:-1].split("-")))
    return StoreVariant(nodeid, (function, masked), store, protected)


_IMPORT_FROM = re.compile(r"^\s*from\s+(tests(?:\.\w+)*)\s+import\s+(\([^)]*\)|[^\n]+)", re.M)
_IMPORT = re.compile(r"^\s*import\s+(tests(?:\.\w+)+)", re.M)


def is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def test_files(root: Path) -> tuple[str, ...]:
    tests = root / "tests"
    found = []
    for path in tests.rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" not in relative and is_test_file(relative):
            found.append(relative)
    return tuple(sorted(found))


class _Sources:
    """Read test modules and follow their ``tests.*`` imports."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._text: dict[str, str] = {}
        self._imports: dict[str, tuple[str, ...]] = {}
        self._closures: dict[str, tuple[str, ...]] = {}
        self._matched: dict[tuple[str, str], bool] = {}

    def text(self, relative: str) -> str:
        if relative not in self._text:
            try:
                self._text[relative] = (self.root / relative).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                self._text[relative] = ""
        return self._text[relative]

    def _module_path(self, module: str) -> str | None:
        base = module.replace(".", "/")
        for candidate in (f"{base}.py", f"{base}/__init__.py"):
            if (self.root / candidate).is_file():
                return candidate
        return None

    def imports(self, relative: str) -> tuple[str, ...]:
        if relative in self._imports:
            return self._imports[relative]
        found: list[str] = []
        text = self.text(relative)
        for module, names in _IMPORT_FROM.findall(text):
            path = self._module_path(module)
            if path:
                found.append(path)
            for name in re.findall(r"\b(\w+)\b(?:\s+as\s+\w+)?", names.strip("()")):
                path = self._module_path(f"{module}.{name}")
                if path:
                    found.append(path)
        for module in _IMPORT.findall(text):
            path = self._module_path(module)
            if path:
                found.append(path)
        parts = relative.split("/")
        # The root conftest serves fixtures by name only; deeper conftests may
        # import parametrized fixtures that a test module never names.
        for depth in range(2, len(parts)):
            conftest = "/".join(parts[:depth]) + "/conftest.py"
            if (self.root / conftest).is_file():
                found.append(conftest)
        self._imports[relative] = tuple(dict.fromkeys(found))
        return self._imports[relative]

    def closure(self, relative: str) -> tuple[str, ...]:
        if relative in self._closures:
            return self._closures[relative]
        seen = {relative: None}
        pending = [relative]
        while pending:
            for imported in self.imports(pending.pop()):
                if imported not in seen:
                    seen[imported] = None
                    pending.append(imported)
        self._closures[relative] = tuple(seen)
        return self._closures[relative]

    def _matches(self, relative: str, pattern: re.Pattern[str]) -> bool:
        key = (relative, pattern.pattern)
        if key not in self._matched:
            self._matched[key] = pattern.search(self.text(relative)) is not None
        return self._matched[key]

    def mentions(self, relative: str, pattern: re.Pattern[str]) -> bool:
        return any(self._matches(path, pattern) for path in self.closure(relative))


def _is_qualification_path(path: str) -> bool:
    return path.startswith("tests/qualification/")


def family_members(root: Path, files: Sequence[str] | None = None) -> dict[str, frozenset[str]]:
    """Return the test files each family's shards collect.

    General shards collect every file outside ``tests/qualification``. The
    others collect files whose source or imported test modules mention their
    marker. The collection gate fails if a collected test reaches a family
    whose shards would not collect its file.
    """

    files = test_files(root) if files is None else tuple(files)
    sources = _Sources(root)
    members: dict[str, frozenset[str]] = {}
    for candidate in FAMILIES:
        if candidate.pattern is None:
            members[candidate.name] = frozenset(
                path for path in files if not _is_qualification_path(path)
            )
            continue
        pattern = re.compile(candidate.pattern)
        selected = set()
        for path in files:
            if _is_qualification_path(path):
                # The root conftest marks this whole tree as qualification.
                if candidate.qualification_paths:
                    selected.add(path)
            elif sources.mentions(path, pattern):
                selected.add(path)
        members[candidate.name] = frozenset(selected)
    return members


def _estimated_family(nodeid: str, path: str, marked: Mapping[str, frozenset[str]]) -> str:
    if _is_qualification_path(path):
        return "qualification"
    for name in ("host-journey", "browser-docker", "qualification"):
        if path in marked[name]:
            return name
    store = nodeid_variant(nodeid).store
    if store == "postgres" or "postgres" in path.rsplit("/", 1)[-1]:
        return "postgres-conformance"
    return "general"


@dataclass(frozen=True)
class Unit:
    path: str
    piece: int
    pieces: int
    seconds: float
    # The slowest single test: work stealing spreads a unit across workers.
    longest: float = 0.0

    @property
    def name(self) -> str:
        return self.path if self.pieces == 1 else f"{self.path}#{self.piece + 1}/{self.pieces}"


@dataclass
class Lane:
    name: str
    family: str
    marker: str
    workers: int | None
    units: list[Unit] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return sum(unit.seconds for unit in self.units)

    def wall_seconds(self, default_workers: int = DEFAULT_WORKERS) -> float:
        workers = self.workers or default_workers
        longest = max((unit.longest for unit in self.units), default=0.0)
        return max(self.seconds / workers, longest)

    @property
    def paths(self) -> frozenset[str]:
        return frozenset(unit.path for unit in self.units)


@dataclass
class Plan:
    mode: str
    lanes: dict[str, Lane]
    members: dict[str, frozenset[str]]
    pieces: dict[tuple[str, str], tuple[int, dict[str, int]]]

    def lane(self, name: str) -> Lane:
        try:
            return self.lanes[name]
        except KeyError:
            raise ValueError(f"lane {name} is not part of the {self.mode} plan") from None

    def selects(self, lane: str, nodeid: str) -> bool:
        """Whether a lane runs this node, before its marker expression."""

        current = self.lane(lane)
        path = nodeid.split("::", 1)[0]
        units = [unit for unit in current.units if unit.path == path]
        if not units:
            return False
        if units[0].pieces == 1:
            return True
        pieces, assigned = self.pieces[(current.family, path)]
        return piece_of(nodeid, pieces, assigned) in {unit.piece for unit in units}


def piece_of(nodeid: str, pieces: int, assigned: Mapping[str, int]) -> int:
    if nodeid in assigned:
        return assigned[nodeid]
    digest = hashlib.sha256(nodeid.encode()).digest()
    return int.from_bytes(digest[:8], "big") % pieces


def _least_loaded(loads: Sequence[float]) -> int:
    return min(range(len(loads)), key=lambda index: (loads[index], index))


def _split(ids: Mapping[str, float], pieces: int) -> dict[str, int]:
    loads = [0.0] * pieces
    assigned = {}
    for nodeid, seconds in sorted(ids.items(), key=lambda item: (-item[1], item[0])):
        index = _least_loaded(loads)
        assigned[nodeid] = index
        loads[index] += seconds
    return assigned


def estimated_family(
    nodeid: str, members: Mapping[str, frozenset[str]], skipped: Collection[Hashable]
) -> str | None:
    """The family the planner assumes for a node, or None when a mode skips it."""

    if nodeid in skipped:
        return None
    return _estimated_family(nodeid, nodeid.split("::", 1)[0], members)


def lane_corrections(
    nodeids: Iterable[str],
    observed: Mapping[str, str],
    mode: str,
    members: Mapping[str, frozenset[str]],
) -> dict[str, str | None]:
    """Return the nodes whose collected family differs from the estimate.

    ``nodeids`` are the planned (timed) nodes and ``observed`` maps every node
    collected under ``mode`` to its family. A planned node that was not
    collected is recorded as None.
    """

    nodeids = list(nodeids)
    skipped = deselected_store_variants(
        (nodeid_variant(nodeid, protected=keeps_every_store(nodeid)) for nodeid in nodeids), mode
    )
    corrections: dict[str, str | None] = {}
    for nodeid in nodeids:
        actual = observed.get(nodeid)
        if actual != estimated_family(nodeid, members, skipped):
            corrections[nodeid] = actual
    return corrections


def build_plan(
    root: Path,
    mode: str,
    *,
    durations: Mapping[str, float] | None = None,
    files: Sequence[str] | None = None,
    shards: Mapping[str, int] | None = None,
    lanes: Mapping[str, str | None] | None = None,
) -> Plan:
    if mode not in STORE_MODES:
        raise ValueError(f"unknown store mode: {mode}")
    if durations is None:
        durations = load_durations(root)
    if lanes is None:
        lanes = load_lanes(root, mode)
    files = test_files(root) if files is None else tuple(sorted(files))
    members = family_members(root, files)
    skipped = deselected_store_variants(
        (nodeid_variant(nodeid, protected=keeps_every_store(nodeid)) for nodeid in durations),
        mode,
    )
    measured = {nodeid.split("::", 1)[0] for nodeid in durations}
    weights: dict[str, dict[str, dict[str, float]]] = {candidate.name: {} for candidate in FAMILIES}
    for nodeid, seconds in durations.items():
        path = nodeid.split("::", 1)[0]
        # Recorded corrections win over the estimate, including None (not run).
        name = lanes[nodeid] if nodeid in lanes else estimated_family(nodeid, members, skipped)
        if name is not None and path in members[name]:
            weights[name].setdefault(path, {})[nodeid] = float(seconds)

    lanes: dict[str, Lane] = {}
    pieces: dict[tuple[str, str], tuple[int, dict[str, int]]] = {}
    for candidate in FAMILIES:
        count = (shards or {}).get(candidate.name, candidate.shards[mode])
        family_lanes = [
            Lane(candidate.lane(shard), candidate.name, candidate.marker, candidate.workers)
            for shard in range(1, count + 1)
        ]
        units: list[Unit] = []
        for path in sorted(members[candidate.name]):
            ids = weights[candidate.name].get(path, {})
            if ids:
                seconds = sum(ids.values())
            elif candidate.pattern is None and path not in measured:
                seconds = DEFAULT_FILE_SECONDS
            else:
                seconds = 0.0
            seconds += COLLECT_SECONDS_PER_FILE
            count_pieces = max(1, min(count, math.ceil(seconds / candidate.split_seconds)))
            # Without work stealing a whole file (piece) runs on one worker.
            whole_files = candidate.distribution == "loadfile"
            if count_pieces == 1:
                longest = seconds if whole_files else max(ids.values(), default=seconds)
                units.append(Unit(path, 0, 1, seconds, longest))
                continue
            assigned = _split(ids, count_pieces)
            pieces[(candidate.name, path)] = (count_pieces, assigned)
            piece_seconds = [COLLECT_SECONDS_PER_FILE] * count_pieces
            piece_longest = [0.0] * count_pieces
            for nodeid, index in assigned.items():
                piece_seconds[index] += ids[nodeid]
                piece_longest[index] = max(piece_longest[index], ids[nodeid])
            units.extend(
                Unit(
                    path,
                    index,
                    count_pieces,
                    piece_seconds[index],
                    piece_seconds[index] if whole_files else piece_longest[index],
                )
                for index in range(count_pieces)
            )
        loads = [0.0] * count
        for unit in sorted(units, key=lambda unit: (-unit.seconds, unit.path, unit.piece)):
            index = _least_loaded(loads)
            family_lanes[index].units.append(unit)
            loads[index] += unit.seconds
        for lane in family_lanes:
            lane.units.sort(key=lambda unit: (unit.path, unit.piece))
            lanes[lane.name] = lane
    return Plan(mode, lanes, members, pieces)


def load_durations(root: Path) -> dict[str, float]:
    try:
        data = json.loads((root / ".test_durations").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise ValueError(".test_durations must be a JSON object")
    return {str(key): float(value) for key, value in data.items()}


def load_lanes(root: Path, mode: str) -> dict[str, str | None]:
    """Return the recorded family corrections for one store mode."""

    try:
        data = json.loads((root / LANES_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    names = {candidate.name for candidate in FAMILIES}
    corrections = data.get(mode, {}) if isinstance(data, dict) else None
    if not isinstance(corrections, dict) or not all(
        isinstance(nodeid, str) and (name is None or name in names)
        for nodeid, name in corrections.items()
    ):
        raise ValueError(f"{LANES_FILE} must map store modes to node IDs and family names")
    return dict(corrections)


def recommended_shards(plan: Plan, name: str) -> int:
    candidate = family(name)
    workers = candidate.workers or DEFAULT_WORKERS
    total = sum(lane.seconds for lane in plan.lanes.values() if lane.family == name)
    return max(1, math.ceil(total / (workers * candidate.target_minutes * 60)))


def describe(plan: Plan, families: Sequence[str]) -> list[str]:
    lines = [f"Store mode: {plan.mode} ({DEFAULT_WORKERS} workers unless noted)"]
    runners = 0
    for name in families:
        lanes = [lane for lane in plan.lanes.values() if lane.family == name]
        walls = [lane.wall_seconds() / 60 for lane in lanes]
        files = sum(len(lane.paths) for lane in lanes)
        runners += len(lanes)
        lines.append(
            f"{name}: {len(lanes)} shard(s), workers {lanes[0].workers or DEFAULT_WORKERS}, "
            f"predicted wall max {max(walls):.1f} min / min {min(walls):.1f} min, "
            f"{sum(lane.seconds for lane in lanes) / 3600:.2f} h of test time, "
            f"{files / len(lanes):.0f} files per shard, "
            f"recommended {recommended_shards(plan, name)} shard(s)"
        )
    lines.append(f"Runners: {runners}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stores", choices=STORE_MODES, default="sqlite")
    parser.add_argument(
        "--family",
        action="append",
        choices=[candidate.name for candidate in FAMILIES],
        help="families to describe (default: the CI families)",
    )
    parser.add_argument("--lane", help="list the files and pieces assigned to one lane")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    plan = build_plan(root, args.stores)
    if args.lane:
        lane = plan.lane(args.lane)
        for unit in lane.units:
            print(f"{unit.seconds / 60:7.1f} min  {unit.name}")
        print(f"predicted wall {lane.wall_seconds() / 60:.1f} min")
        return 0
    print("\n".join(describe(plan, args.family or CI_FAMILIES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
