from __future__ import annotations

import json
from contextlib import suppress
from types import SimpleNamespace

import pytest

from tests.qualification import report_plugin


def test_report_excludes_payloads_exceptions_paths_and_untrusted_properties():
    report_plugin._records.clear()
    report_plugin._records["cases"] = {}
    report_plugin.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="secret-prompt-private-path",
            when="call",
            outcome="failed",
            longrepr="secret-exception",
            capstdout="secret-tool-result",
            user_properties=[
                ("arbitrary-secret-key", "secret"),
                ("tool_calls", "secret"),
                ("active_claims", -1),
                ("tool_calls", 12),
            ],
        )
    )
    serialized = json.dumps(report_plugin._records)
    assert "secret" not in serialized
    [case] = report_plugin._records["cases"].values()
    assert case == {"phases": {"call": "failed"}, "counters": {"tool_calls": 12}}


def test_registry_names_are_unique_and_every_scenario_has_a_boundary():
    from tests.qualification.registry import DOCKER_SCENARIOS, POSTGRES_SCENARIOS, SCENARIOS

    scenarios = SCENARIOS + POSTGRES_SCENARIOS + DOCKER_SCENARIOS
    assert len({s.name for s in scenarios}) == len(scenarios)
    assert all(s.invariant and s.boundary and s.selectors for s in scenarios)


def test_maintenance_partition_selects_every_collected_case_exactly_once():
    import subprocess
    import sys
    from collections import Counter
    from pathlib import Path

    from tests.qualification.registry import MAINTENANCE_SCENARIOS

    root = Path(__file__).resolve().parents[2]
    modules = sorted(
        str(path.relative_to(root))
        for path in (root / "tests/qualification").glob("test_repository_maintenance_*.py")
    )
    modules += [
        "tests/recovery/test_worker_harness_ownership.py",
        "tests/core/test_subagent_registry_drain.py",
    ]
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *modules],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    nodes = [
        line for line in collected.stdout.splitlines() if line.startswith("tests/") and "::" in line
    ]
    assert nodes and len(nodes) == len(set(nodes))
    coverage = Counter()
    for scenario in MAINTENANCE_SCENARIOS:
        for selector in scenario.selectors:
            matches = [
                node
                for node in nodes
                if node == selector or node.startswith((selector + "::", selector + "["))
            ]
            assert matches, f"Empty or obsolete qualification selector: {selector}"
            coverage.update(matches)
    assert set(coverage) == set(nodes), f"Unregistered cases: {set(nodes) - set(coverage)}"
    assert all(count == 1 for count in coverage.values()), {
        node: count for node, count in coverage.items() if count != 1
    }
    journeys = [scenario for scenario in MAINTENANCE_SCENARIOS if "-journey-" in scenario.name]
    assert len(journeys) == 18 and all(len(scenario.selectors) == 1 for scenario in journeys)


def test_retained_subprocess_fails_report_and_is_reaped(tmp_path, monkeypatch):
    import subprocess
    import sys

    child = report_plugin._TrackedProcess(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    destination = tmp_path / "result.json"
    monkeypatch.setenv("CAYU_QUALIFICATION_RESULT", str(destination))
    session = SimpleNamespace(exitstatus=0)
    try:
        report_plugin.pytest_sessionfinish(session, 0)
        assert session.exitstatus == 1
        assert child.poll() is not None
        assert json.loads(destination.read_text())["resources"]["subprocesses_retained"] == 1
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        report_plugin._children.clear()


def test_scenario_timeout_returns_failure_and_drains_process_group(tmp_path):
    import os
    import sys

    from scripts.run_runtime_qualification import run_bounded

    assert (
        run_bounded(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout=0.1,
        )
        == 124
    )


def test_report_write_is_atomic_and_does_not_leave_temporary_files(tmp_path):
    from scripts.run_runtime_qualification import write_report

    destination = tmp_path / "report.json"
    write_report(destination, {"status": "running"})
    write_report(destination, {"status": "failed"})
    assert json.loads(destination.read_text()) == {"status": "failed"}
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("group_options", [{"start_new_session": True}, {"process_group": 0}])
def test_exited_parent_with_descendant_fails_qualification_and_drains_group(
    tmp_path, group_options
):
    import os
    import signal
    import sys
    from pathlib import Path

    from scripts.run_runtime_qualification import run_bounded

    import cayu
    from tests.qualification.process_cleanup import process_group_exists

    group_file = tmp_path / "group"
    child_ready = tmp_path / "child-ready"
    result = tmp_path / "result.json"
    child_code = (
        f"from pathlib import Path; import time; Path({str(child_ready)!r}).touch(); time.sleep(60)"
    )
    parent_code = (
        "import subprocess, sys, time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"ready = Path({str(child_ready)!r}); "
        "deadline = time.monotonic() + 10\n"
        "while not ready.exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
        "assert ready.exists()\n"
    )
    fixture = tmp_path / "test_descendant.py"
    fixture.write_text(
        "import subprocess, sys\nfrom pathlib import Path\n"
        "def test_parent_exits_successfully():\n"
        f"    parent = subprocess.Popen([sys.executable, '-c', {parent_code!r}], "
        f"**{group_options!r})\n"
        f"    Path({str(group_file)!r}).write_text(str(parent.pid))\n"
        "    assert parent.wait(timeout=15) == 0\n"
    )
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join((str(root), str(root / "src"))),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        CAYU_QUALIFICATION_PACKAGE=str(Path(cayu.__file__).resolve()),
        CAYU_QUALIFICATION_RESULT=str(result),
    )
    env.pop("CAYU_QUALIFICATION_POSTGRES", None)
    env.pop("PYTEST_ADDOPTS", None)
    try:
        code = run_bounded(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "tests.qualification.report_plugin",
                str(fixture),
            ],
            cwd=tmp_path,
            env=env,
            timeout=30,
        )
        assert code == 1
        report = json.loads(result.read_text())
        assert all(
            phase == "passed"
            for case in report["cases"].values()
            for phase in case["phases"].values()
        )
        assert report["resources"]["subprocesses_retained"] == 1
        assert report["resources"]["subprocess_groups_remaining"] == 0
        assert report["resources"]["subprocesses_remaining"] == 0
        assert not process_group_exists(int(group_file.read_text()))
    finally:
        if group_file.exists():
            with suppress(ProcessLookupError):
                os.killpg(int(group_file.read_text()), signal.SIGKILL)


def test_outer_runner_fails_when_exited_leader_leaves_its_group(tmp_path):
    import os
    import signal
    import sys

    from scripts.run_runtime_qualification import run_bounded

    from tests.qualification.process_cleanup import process_group_exists

    group_file = tmp_path / "group"
    code = (
        "import os, subprocess, sys; from pathlib import Path; "
        f"Path({str(group_file)!r}).write_text(str(os.getpid())); "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])"
    )
    try:
        assert (
            run_bounded(
                [sys.executable, "-c", code],
                cwd=tmp_path,
                env=os.environ.copy(),
                timeout=10,
            )
            == 1
        )
        assert not process_group_exists(int(group_file.read_text()))
    finally:
        if group_file.exists():
            with suppress(ProcessLookupError):
                os.killpg(int(group_file.read_text()), signal.SIGKILL)


def test_outer_runner_accepts_a_clean_process_exit(tmp_path):
    import os
    import sys

    from scripts.run_runtime_qualification import run_bounded

    assert (
        run_bounded(
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout=10,
        )
        == 0
    )


@pytest.mark.parametrize("shutdown", ["timeout", "abrupt-exit"])
@pytest.mark.parametrize("group_options", [{"start_new_session": True}, {"process_group": 0}])
def test_outer_runner_drains_registered_workers_without_pytest_shutdown(
    tmp_path,
    shutdown,
    group_options,
):
    import os
    import signal
    import sys
    from pathlib import Path

    from scripts.run_runtime_qualification import run_bounded

    import cayu
    from tests.qualification.process_cleanup import process_group_exists

    group_file = tmp_path / "worker-group"
    result = tmp_path / "result.json"
    fixture = tmp_path / "test_forced_shutdown.py"
    fixture.write_text(
        "import os, signal, subprocess, sys, time\nfrom pathlib import Path\n"
        "def test_worker_outlives_pytest():\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
        f"**{group_options!r})\n"
        f"    Path({str(group_file)!r}).write_text(str(worker.pid))\n"
        + ("    time.sleep(60)\n" if shutdown == "timeout" else "    os._exit(0)\n")
    )
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join((str(root), str(root / "src"))),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        CAYU_QUALIFICATION_PACKAGE=str(Path(cayu.__file__).resolve()),
        CAYU_QUALIFICATION_RESULT=str(result),
    )
    env.pop("CAYU_QUALIFICATION_POSTGRES", None)
    env.pop("PYTEST_ADDOPTS", None)
    try:
        code = run_bounded(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "tests.qualification.report_plugin",
                str(fixture),
            ],
            cwd=tmp_path,
            env=env,
            timeout=10,
        )
        assert group_file.exists(), "The fixture must reach worker dispatch before termination"
        assert code == (124 if shutdown == "timeout" else 1)
        assert not result.exists(), "Pytest must not run its shutdown/report hook"
        assert not process_group_exists(int(group_file.read_text()))
    finally:
        if group_file.exists():
            with suppress(ProcessLookupError):
                os.killpg(int(group_file.read_text()), signal.SIGKILL)


def test_failed_group_registration_terminates_the_launch(monkeypatch):
    import sys

    from tests.qualification.process_cleanup import process_group_exists

    groups = []

    def reject_registration(group_id):
        groups.append(group_id)
        raise OSError("journal unavailable")

    monkeypatch.setattr(report_plugin, "register_process_group", reject_registration)
    before = len(report_plugin._children)
    with pytest.raises(OSError, match="journal unavailable"):
        report_plugin._TrackedProcess(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
    assert len(groups) == 1
    assert not process_group_exists(groups[0])
    assert len(report_plugin._children) == before
