"""How many processes `cayu worker` starts, and how it counts the CPUs it may use."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli import worker as worker_cli
from cayu.runtime import _process_workers
from cayu.runtime._process_workers import ProcessWorkerOutcome, available_cpu_count

_PROJECT = """[tool.cayu]
factory = "processes_project:build_app"

[tool.cayu.workers]
once = "processes_project:run_once"
other = "processes_project:run_once"
"""

_MODULE = """from pathlib import Path

from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)


async def run_once(app, stop):
    Path("ran-in-process.txt").write_text("yes", encoding="utf-8")
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(_PROJECT, encoding="utf-8")
    (root / "processes_project.py").write_text(_MODULE, encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.delenv("CAYU_WORKER_PROCESSES", raising=False)
    sys.modules.pop("processes_project", None)
    return root


@pytest.fixture
def supervised(monkeypatch: pytest.MonkeyPatch) -> list[list[tuple[str, ...]]]:
    """The child argv lists of each supervised start; nothing is started."""

    starts: list[list[tuple[str, ...]]] = []

    async def supervise(commands, **_kwargs):  # type: ignore[no-untyped-def]
        starts.append([command.argv for command in commands])
        return ProcessWorkerOutcome(
            exit_code=0,
            child_exit_codes=tuple(0 for _ in commands),
            pids=(),
            forced_shutdown=False,
        )

    monkeypatch.setattr(worker_cli, "supervise_process_workers", supervise)
    return starts


def _declare(root: Path, table: str) -> None:
    with (root / "pyproject.toml").open("a", encoding="utf-8") as pyproject:
        pyproject.write("\n[tool.cayu.worker_processes]\n" + table)


def test_a_worker_starts_one_process_per_cpu_by_default(
    project: Path, supervised, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(worker_cli, "available_cpu_count", lambda: 3)

    assert main(["worker", "once"]) == 0

    assert len(supervised) == 1
    assert len(supervised[0]) == 3
    # Each child runs the worker in one process of its own.
    for argv in supervised[0]:
        assert argv[argv.index("--processes") + 1] == "1"
    assert "starting 3 processes (one per CPU)" in capsys.readouterr().err
    assert not (project / "ran-in-process.txt").exists()


def test_one_cpu_runs_the_worker_in_this_process(project: Path, supervised, monkeypatch) -> None:
    monkeypatch.setattr(worker_cli, "available_cpu_count", lambda: 1)

    assert main(["worker", "once"]) == 0

    assert supervised == []
    assert (project / "ran-in-process.txt").read_text(encoding="utf-8") == "yes"


@pytest.mark.parametrize(
    ("flag", "environment", "declared", "expected"),
    [
        # --processes wins over the environment and pyproject.
        (["--processes", "2"], "4", "once = 5\n", 2),
        (["--processes", "auto"], "4", "once = 5\n", 6),
        # The environment wins over pyproject.
        ([], "4", "once = 5\n", 4),
        ([], "auto", "once = 5\n", 6),
        # pyproject applies to the worker it names.
        ([], None, "once = 5\n", 5),
        ([], None, 'once = "auto"\n', 6),
        ([], None, "other = 5\n", 6),
    ],
)
def test_the_process_count_comes_from_the_flag_then_environment_then_pyproject(
    project: Path,
    supervised,
    monkeypatch,
    flag: list[str],
    environment: str | None,
    declared: str,
    expected: int,
) -> None:
    monkeypatch.setattr(worker_cli, "available_cpu_count", lambda: 6)
    if environment is not None:
        monkeypatch.setenv("CAYU_WORKER_PROCESSES", environment)
    _declare(project, declared)

    assert main(["worker", "once", *flag]) == 0

    assert [len(start) for start in supervised] == [expected]


@pytest.mark.parametrize(
    "selection",
    [
        (["--processes", "1"], None, None),
        ([], "1", None),
        ([], None, "once = 1\n"),
    ],
)
def test_one_process_can_be_asked_for_anywhere(
    project: Path, supervised, monkeypatch, selection
) -> None:
    flag, environment, declared = selection
    monkeypatch.setattr(worker_cli, "available_cpu_count", lambda: 6)
    if environment is not None:
        monkeypatch.setenv("CAYU_WORKER_PROCESSES", environment)
    if declared is not None:
        _declare(project, declared)

    assert main(["worker", "once", *flag]) == 0

    assert supervised == []
    assert (project / "ran-in-process.txt").exists()


def test_a_declared_single_worker_runs_in_one_process_unless_more_were_asked_for(
    project: Path, supervised, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(worker_cli, "available_cpu_count", lambda: 6)
    # The flag sets this variable for the process; an empty value declares nothing.
    monkeypatch.setenv("CAYU_WORKER_RECONCILE_ABANDONED_AFTER_S", "")

    assert main(["worker", "once", "--reconcile-abandoned-after-seconds", "60"]) == 0
    assert supervised == []

    monkeypatch.setenv("CAYU_WORKER_RECONCILE_ABANDONED_AFTER_S", "")
    monkeypatch.setenv("CAYU_WORKER_PROCESSES", "2")
    assert main(["worker", "once", "--reconcile-abandoned-after-seconds", "60"]) == 1
    assert "can't be combined with CAYU_WORKER_PROCESSES" in capsys.readouterr().err
    assert supervised == []


def test_a_run_until_idle_uses_one_process_whatever_the_environment_says(
    project: Path, supervised, monkeypatch, capsys
) -> None:
    (project / "processes_project.py").write_text(
        _MODULE
        + """

from cayu.tasks import InMemoryTaskStore, run_task_worker_until_idle


async def handle(app, task, worker_id):
    return None


async def run_idle(app, stop):
    return await run_task_worker_until_idle(app, InMemoryTaskStore(), handle, worker_id="idle")
""",
        encoding="utf-8",
    )
    (project / "pyproject.toml").write_text(
        _PROJECT.replace(
            'other = "processes_project:run_once"', 'idle = "processes_project:run_idle"'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(worker_cli, "available_cpu_count", lambda: 6)
    monkeypatch.setenv("CAYU_WORKER_PROCESSES", "4")

    assert main(["worker", "idle", "--until-idle"]) == 0
    assert supervised == []
    assert '"handled":0' in capsys.readouterr().out

    assert main(["worker", "idle", "--until-idle", "--processes", "2"]) == 1
    assert "--until-idle can't be combined with --processes" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        ("once = 0\n", "[tool.cayu.worker_processes].once must be a number from 1 to 256"),
        ("once = true\n", "[tool.cayu.worker_processes].once must be a number from 1 to 256"),
        ('once = "many"\n', "[tool.cayu.worker_processes].once must be a number from 1 to 256"),
        ("missing = 2\n", "names 'missing', which [tool.cayu.workers] doesn't declare"),
    ],
)
def test_pyproject_process_counts_are_checked(
    project: Path, supervised, capsys, declared: str, message: str
) -> None:
    _declare(project, declared)

    assert main(["worker", "once"]) == 1

    assert message in capsys.readouterr().err
    assert supervised == []


def test_an_invalid_environment_count_is_refused(project: Path, supervised, monkeypatch, capsys):
    monkeypatch.setenv("CAYU_WORKER_PROCESSES", "0")

    assert main(["worker", "once"]) == 1

    assert "CAYU_WORKER_PROCESSES: process count must be auto" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "257", "two"])
def test_the_flag_takes_auto_or_a_count_from_1_to_256(value: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["worker", "configured", "--processes", value])

    assert excinfo.value.code == 2


def _host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    cgroup: str,
    mounts: list[str],
    files: dict[str, str],
) -> Path:
    """A fake /proc/self and cgroup filesystem under ``tmp_path``; returns its root."""

    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "cgroup").write_text(cgroup, encoding="utf-8")
    fs = tmp_path / "fs"
    (proc / "mountinfo").write_text(
        "".join(line.replace("FS", str(fs)) + "\n" for line in mounts), encoding="utf-8"
    )
    for relative, content in files.items():
        path = fs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="ascii")
    monkeypatch.setattr(_process_workers, "_PROC_SELF", proc)
    monkeypatch.setattr(_process_workers, "_schedulable_cpu_count", lambda: 8)
    return fs


_V2_MOUNT = "30 23 0:26 / FS/cgroup rw,nosuid - cgroup2 cgroup2 rw,nsdelegate"
_V1_CPU_MOUNT = "33 25 0:29 / FS/cgroup/cpu,cpuacct rw - cgroup cgroup rw,cpu,cpuacct"
_V1_MEMORY_MOUNT = "34 25 0:30 / FS/cgroup/memory rw - cgroup cgroup rw,memory"


@pytest.mark.parametrize(
    ("cgroup", "mounts", "files", "expected"),
    [
        # No cgroup limit: the CPUs the process may be scheduled on.
        ("0::/\n", [_V2_MOUNT], {}, 8),
        ("0::/\n", [_V2_MOUNT], {"cgroup/cpu.max": "max 100000\n"}, 8),
        # A container's own cgroup v2 quota, rounded down, as Cayu Cloud's cpu_millis sets it.
        ("0::/\n", [_V2_MOUNT], {"cgroup/cpu.max": "200000 100000\n"}, 2),
        ("0::/\n", [_V2_MOUNT], {"cgroup/cpu.max": "150000 100000\n"}, 1),
        ("0::/\n", [_V2_MOUNT], {"cgroup/cpu.max": "50000 100000\n"}, 1),
        # A service's cgroup below the mount, such as systemd's CPUQuota=200%.
        (
            "0::/system.slice/worker.service\n",
            [_V2_MOUNT],
            {
                "cgroup/cpu.max": "max 100000\n",
                "cgroup/system.slice/worker.service/cpu.max": "200000 100000\n",
            },
            2,
        ),
        # An ancestor's quota limits its descendants.
        (
            "0::/agents/one\n",
            [_V2_MOUNT],
            {
                "cgroup/agents/cpu.max": "300000 100000\n",
                "cgroup/agents/one/cpu.max": "max 100000\n",
            },
            3,
        ),
        # A mount of part of the hierarchy, as inside a container without a cgroup namespace.
        (
            "0::/kubepods/pod1/abc\n",
            ["30 23 0:26 /kubepods/pod1/abc FS/cgroup rw - cgroup2 cgroup2 rw"],
            {"cgroup/cpu.max": "400000 100000\n"},
            4,
        ),
        # cgroup v1, found through the cpu controller's mount.
        (
            "5:memory:/docker/abc\n4:cpu,cpuacct:/docker/abc\n",
            [_V1_MEMORY_MOUNT, _V1_CPU_MOUNT],
            {
                "cgroup/cpu,cpuacct/docker/abc/cpu.cfs_quota_us": "400000\n",
                "cgroup/cpu,cpuacct/docker/abc/cpu.cfs_period_us": "100000\n",
            },
            4,
        ),
        (
            "4:cpu,cpuacct:/\n",
            [_V1_CPU_MOUNT],
            {
                "cgroup/cpu,cpuacct/cpu.cfs_quota_us": "-1\n",
                "cgroup/cpu,cpuacct/cpu.cfs_period_us": "100000\n",
            },
            8,
        ),
        # Unreadable values don't limit it.
        ("0::/\n", [_V2_MOUNT], {"cgroup/cpu.max": "garbage\n"}, 8),
        ("garbage\n", ["garbage"], {}, 8),
    ],
)
def test_available_cpu_count_respects_the_process_cgroup_cpu_limit(
    monkeypatch, tmp_path: Path, cgroup: str, mounts: list[str], files: dict[str, str], expected
) -> None:
    _host(monkeypatch, tmp_path, cgroup=cgroup, mounts=mounts, files=files)

    assert available_cpu_count() == expected


def test_available_cpu_count_without_cgroup_information(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_process_workers, "_PROC_SELF", tmp_path / "missing")
    monkeypatch.setattr(_process_workers, "_schedulable_cpu_count", lambda: 5)

    assert available_cpu_count() == 5


def test_available_cpu_count_is_a_public_runtime_part() -> None:
    from cayu import runtime

    assert runtime.available_cpu_count is available_cpu_count
    assert available_cpu_count() >= 1
