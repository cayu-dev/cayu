"""Optional pytest process measurements, independent of CI test selection."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _counters(path: Path) -> dict[str, int] | None:
    text = _read(path)
    if text is None:
        return None
    try:
        return {key.rstrip(":"): int(value) for key, value in map(str.split, text.splitlines())}
    except ValueError:
        return None


def _pressure(path: Path) -> dict[str, int] | None:
    text = _read(path)
    if text is None:
        return None
    try:
        return {
            fields[0]: int(dict(part.split("=") for part in fields[1:])["total"])
            for fields in map(str.split, text.splitlines())
        }
    except (ValueError, KeyError, IndexError):
        return None


def _cgroup() -> tuple[Path, Path] | None:
    # Resolve the unified hierarchy through its mount root, including containers
    # whose cgroup namespace exposes "/" rather than the host's membership path.
    membership = _read(Path("/proc/self/cgroup"))
    mounts = _read(Path("/proc/self/mountinfo"))
    if membership is None or mounts is None:
        return None
    group = next((line[3:] for line in membership.splitlines() if line.startswith("0::")), None)
    if group is None or ".." in Path(group).parts:
        return None
    for line in mounts.splitlines():
        left, separator, right = line.partition(" - ")
        if not separator or right.split()[0] != "cgroup2":
            continue
        fields = left.split()
        mount_root, mount_path = (Path(value.replace(r"\040", " ")) for value in fields[3:5])
        try:
            relative = Path(group).relative_to(mount_root)
        except ValueError:
            continue
        candidate = mount_path / relative
        if candidate.is_dir():
            return candidate, mount_path
    return None


def _host_snapshot() -> dict:
    membership = _cgroup()
    group = membership[0] if membership else None
    limits = []
    if membership:
        current, mount = membership
        while True:
            limits.append(
                {
                    "cpu_max": _read(current / "cpu.max"),
                    "memory_max": _read(current / "memory.max"),
                }
            )
            if current == mount:
                break
            current = current.parent
    memory = _read(Path("/proc/meminfo"))
    memory_fields = {}
    if memory:
        for line in memory.splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory_fields[key] = int(value.split()[0]) * 1024
    return {
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "load_average": list(os.getloadavg()),
        "memory_bytes": memory_fields or None,
        "pressure_total_microseconds": {
            kind: _pressure(Path("/proc/pressure") / kind) for kind in ("cpu", "memory", "io")
        },
        "cgroup_v2": None
        if group is None
        else {
            "visible_ancestor_limits": limits,
            "cpu_stat": _counters(group / "cpu.stat"),
            "memory_current": _read(group / "memory.current"),
            "memory_peak": _read(group / "memory.peak"),
            "memory_events": _counters(group / "memory.events"),
        },
    }


def _usage(who: int) -> dict:
    usage = resource.getrusage(who)
    return {
        "user_seconds": usage.ru_utime,
        "system_seconds": usage.ru_stime,
        "peak_rss_bytes": int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
        "major_faults": usage.ru_majflt,
        "input_blocks": usage.ru_inblock,
        "output_blocks": usage.ru_oublock,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def _snapshot() -> dict:
    return {
        "monotonic": time.perf_counter(),
        "self": _usage(resource.RUSAGE_SELF),
        "waited_children": _usage(resource.RUSAGE_CHILDREN),
        "io_self_and_waited_children": _counters(Path("/proc/self/io")),
    }


def _delta(start: dict | None, end: dict | None) -> dict | None:
    if start is None or end is None:
        return None
    return {
        key: end[key] - value if end.get(key, -1) >= value else None
        for key, value in start.items()
        if key != "peak_rss_bytes"
    }


def _phase(start: dict, end: dict) -> dict:
    return {
        "wall_seconds": end["monotonic"] - start["monotonic"],
        "self": _delta(start["self"], end["self"]),
        "waited_children": _delta(start["waited_children"], end["waited_children"]),
        "io_self_and_waited_children": _delta(
            start["io_self_and_waited_children"], end["io_self_and_waited_children"]
        ),
    }


def _digest(nodes) -> str:
    return hashlib.sha256(json.dumps(sorted(nodes)).encode()).hexdigest()


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.getgroup("CI profiling").addoption("--ci-profile", type=Path)


def pytest_configure(config: pytest.Config) -> None:
    output = config.getoption("ci_profile")
    if output is not None:
        if not hasattr(config, "workerinput"):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.unlink(missing_ok=True)
        config.pluginmanager.register(_Profile(config, output), "cayu-ci-profile")


class _Profile:
    def __init__(self, config: pytest.Config, output: Path) -> None:
        self.config = config
        self.output = output
        self.is_worker = hasattr(config, "workerinput")
        self.start: dict = {}
        self.collected: dict | None = None
        self.host_start: dict = {}
        self.workers: dict[str, dict] = {}
        self.collections: dict[str, dict] = {}
        self.outcomes: Counter[str] = Counter()
        self.tests: dict[str, float] = defaultdict(float)

    def pytest_sessionstart(self) -> None:
        self.start = _snapshot()
        if not self.is_worker:
            self.host_start = _host_snapshot()

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.collected = _snapshot()
        self.collections["local"] = self._inventory([item.nodeid for item in session.items])

    @staticmethod
    def _inventory(nodes) -> dict:
        return {"count": len(nodes), "nodeids_sha256": _digest(nodes)}

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids) -> None:
        self.collections[node.gateway.id] = self._inventory(ids)

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error) -> None:
        profile = getattr(node, "workeroutput", {}).get("ci_profile")
        self.workers[node.gateway.id] = {
            "completed": error is None and profile is not None,
            "measurements": profile,
        }

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if not self.is_worker:
            self.outcomes[f"{report.when}.{report.outcome}"] += 1
            self.tests[report.nodeid] += report.duration

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if not self.start:
            return
        end = _snapshot()
        process = {
            "session": _phase(self.start, end),
            "collection": _phase(self.start, self.collected) if self.collected else None,
            "execution": _phase(self.collected, end) if self.collected else None,
            "lifetime_peak_rss_bytes": {
                "self": end["self"]["peak_rss_bytes"],
                "waited_children": end["waited_children"]["peak_rss_bytes"],
            },
        }
        if self.is_worker:
            assert hasattr(self.config, "workeroutput")
            self.config.workeroutput["ci_profile"] = process
            return
        try:
            revision = subprocess.check_output(
                ("git", "rev-parse", "HEAD"),
                cwd=self.config.rootpath,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            source_clean = not subprocess.check_output(
                ("git", "status", "--porcelain", "--untracked-files=all"),
                cwd=self.config.rootpath,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
            source_clean = None
        files: dict[str, float] = defaultdict(float)
        for nodeid, seconds in self.tests.items():
            files[nodeid.split("::", 1)[0]] += seconds
        report = {
            "schema_version": 1,
            "revision": revision,
            "source_clean": source_clean,
            "python": platform.python_version(),
            "pytest": pytest.__version__,
            "pytest_xdist": _version("pytest-xdist"),
            "pytest_split": _version("pytest-split"),
            "platform": [platform.system(), platform.machine()],
            "exitstatus": int(exitstatus),
            "requested_workers": self.config.getoption("numprocesses", 0),
            "distribution": self.config.getoption("dist", "no"),
            "collections": self.collections,
            "executed": self._inventory(self.tests),
            "outcomes": dict(self.outcomes),
            "slowest_tests_seconds": sorted(
                self.tests.items(), key=lambda item: (-item[1], item[0])
            )[:20],
            "slowest_files_seconds": sorted(files.items(), key=lambda item: (-item[1], item[0]))[
                :20
            ],
            "controller" if self.workers else "process": process,
            "workers": self.workers,
            "shared_host_start": self.host_start,
            "shared_host_end": _host_snapshot(),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.output.parent, delete=False
        ) as temporary:
            json.dump(report, temporary, indent=2)
            temporary.write("\n")
        try:
            Path(temporary.name).replace(self.output)
        finally:
            Path(temporary.name).unlink(missing_ok=True)
        print(f"\nCI execution profile: {json.dumps(report, separators=(',', ':'))}", flush=True)
