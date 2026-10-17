"""Exercise planned CI shards through real pytest collection and xdist workers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from scripts.ci_shard_plan import build_plan, family

_ROOT = Path(__file__).resolve().parents[2]


def _pytest(root: Path, *args: str, env: dict[str, str] | None = None, plugin: bool = True):
    process_env = os.environ.copy()
    for name in ("CAYU_CI_REPORT_OWNER_PID", "CAYU_TEST_STORES", "GITHUB_ACTIONS"):
        process_env.pop(name, None)
    process_env.update(
        PYTHONPATH=os.pathsep.join((str(_ROOT), str(_ROOT / "src"), str(root))),
        CAYU_REQUIRE_CURRENT_TEST_DURATIONS="1",
    )
    process_env.update(env or {})
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--rootdir",
            str(root),
            "-c",
            str(root / "pytest.ini"),
            *(("-p", "scripts.ci_test_shards") if plugin else ()),
            *args,
        ],
        cwd=root,
        env=process_env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _nodes(result) -> list[str]:
    assert result.returncode == 0, result.stdout + result.stderr
    return [
        line for line in result.stdout.splitlines() if line.startswith("tests/") and "::" in line
    ]


def _lanes(root: Path, mode: str, name: str) -> list[str]:
    plan = build_plan(root, mode)
    return [lane.name for lane in plan.lanes.values() if lane.family == name and lane.units]


_BODY = """import json, os
from pathlib import Path
import pytest

@pytest.fixture(scope="module")
def resource():
    events = os.environ.get("CI_SAMPLE_EVENTS")
    target = Path(events) / (Path(__file__).stem + ".json") if events else None
    if target:
        assert not target.exists(), "module fixture ran on multiple workers"
        target.write_text(json.dumps({"pid": os.getpid(), "closed": False}))
    yield
    if target:
        target.write_text(json.dumps({"pid": os.getpid(), "closed": True}))

@pytest.mark.parametrize("case", range(40))
def test_general(case, resource):
    if os.environ.get("CI_SAMPLE_CRASH"): os._exit(42)
    assert not os.environ.get("CI_SAMPLE_FAIL"), "shard-immediate-failure"

@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("case", range(4))
def test_backend(backend, case, resource): pass

@pytest.mark.host_journey
@pytest.mark.parametrize("case", range(3))
def test_host(case, resource): pass
"""


@pytest.fixture(scope="module")
def shard_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("ci-shards")
    (root / "tests/qualification").mkdir(parents=True)
    (root / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests\nmarkers =\n    postgres\n    browser_docker\n"
        "    host_journey\n    qualification\n    stress\n    store_semantics\n"
    )
    # Reuse the real dynamic-marker, store-switch and duration hooks.
    (root / "tests/conftest.py").write_text(
        "from tests.conftest import (pytest_collection_modifyitems, pytest_configure, "
        "pytest_runtest_logstart, pytest_runtest_logfinish, pytest_runtest_logreport)\n"
    )
    for name in ("alpha", "beta"):
        (root / f"tests/test_{name}.py").write_text(_BODY)
    (root / "tests/test_semantics.py").write_text(
        "import pytest\npytestmark = pytest.mark.store_semantics\n"
        "@pytest.mark.parametrize('backend', ['memory', 'sqlite', 'postgres'])\n"
        "def test_race(backend): pass\n"
        "@pytest.mark.parametrize('store', [('memory', 1), ('postgres', 1)])\n"
        "def test_without_sqlite_twin(store): pass\n"
    )
    # A non-string value selects the backend; only the parameter IDs name it.
    (root / "tests/test_durable.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('durable', [False, True], ids=['memory', 'sqlite'])\n"
        "def test_profile(durable): pass\n"
        # Distinct cases whose IDs merely mention a store are not twins.
        "@pytest.mark.parametrize(('statement', 'allowed'), [\n"
        "    pytest.param('import a.sqlite', ['a'], id='sqlite'),\n"
        "    pytest.param('import a.postgres', ['b'], id='postgres'),\n"
        "])\n"
        "def test_imports(statement, allowed): pass\n"
    )
    (root / "tests/qualification/test_capacity.py").write_text(
        "import os\nif os.environ.get('CI_SAMPLE_FORBID_QUALIFICATION'):\n"
        "    raise RuntimeError('unassigned module was imported')\n"
        "import pytest\n@pytest.mark.parametrize('case', range(16))\n"
        "def test_capacity(case): pass\n"
    )
    (root / "tests/test_browser.py").write_text(
        "import pytest\n@pytest.mark.browser_docker\n"
        "@pytest.mark.parametrize('case', range(2))\ndef test_browser(case): pass\n"
    )
    result = _pytest(
        root,
        "--collect-only",
        "-q",
        "-n",
        "0",
        plugin=False,
        env={"CAYU_REQUIRE_CURRENT_TEST_DURATIONS": "0"},
    )
    # test_alpha's general tests exceed the split threshold; the rest stay whole.
    durations = {
        node: 30.0 if node.startswith("tests/test_alpha.py::test_general") else 0.5
        for node in _nodes(result)
        if not node.startswith("tests/test_browser.py")
    }
    (root / ".test_durations").write_text(json.dumps(durations))
    return root


@pytest.mark.parametrize(("mode", "name"), [("sqlite", "general"), ("all", "postgres-conformance")])
def test_lanes_partition_the_stock_marker_selection(shard_project, mode, name):
    env = {"CAYU_TEST_STORES": mode}
    stock = _pytest(
        shard_project,
        "--collect-only",
        "-q",
        "-n",
        "0",
        "-m",
        family(name).marker,
        plugin=False,
        env=env,
    )
    lanes = _lanes(shard_project, mode, name)
    assert len(lanes) > 1
    assigned = Counter()
    for lane in lanes:
        assigned.update(
            _nodes(_pytest(shard_project, "--collect-only", "-q", "--ci-shard", lane, env=env))
        )
    assert assigned == Counter(_nodes(stock))
    assert all(count == 1 for count in assigned.values())


def test_pull_request_stores_keep_sqlite_twins_and_store_semantics(shard_project):
    nodes = _nodes(
        _pytest(
            shard_project,
            "--collect-only",
            "-q",
            "-n",
            "0",
            plugin=False,
            env={"CAYU_TEST_STORES": "sqlite"},
        )
    )
    backend = [node for node in nodes if node.startswith("tests/test_alpha.py::test_backend")]
    assert sorted(backend) == [
        f"tests/test_alpha.py::test_backend[{case}-sqlite]" for case in range(4)
    ]
    semantics = [node for node in nodes if node.startswith("tests/test_semantics.py")]
    assert len(semantics) == 5
    durable = [node for node in nodes if node.startswith("tests/test_durable.py")]
    assert durable == [
        "tests/test_durable.py::test_profile[sqlite]",
        "tests/test_durable.py::test_imports[sqlite]",
        "tests/test_durable.py::test_imports[postgres]",
    ]
    every = _nodes(_pytest(shard_project, "--collect-only", "-q", "-n", "0", plugin=False))
    assert len([node for node in every if "::test_backend[" in node]) == 24


def test_unknown_store_mode_is_rejected(shard_project):
    result = _pytest(
        shard_project, "--collect-only", "-q", plugin=False, env={"CAYU_TEST_STORES": "memory"}
    )
    assert result.returncode == 4
    assert "CAYU_TEST_STORES must be one of" in result.stderr


def test_collection_gate_accepts_the_plan_and_writes_collected_ids(shard_project, tmp_path):
    collected = tmp_path / "ids.txt"
    lanes = tmp_path / "lanes/test-lanes.json"
    result = _pytest(
        shard_project,
        "--collect-only",
        "-q",
        "-n",
        "0",
        "--ci-check-plan",
        "--ci-collected",
        str(collected),
        "--ci-lanes",
        str(lanes),
        env={"CAYU_TEST_STORES": "sqlite"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CI shard plan (sqlite stores): general=" in result.stdout
    assert collected.read_text().splitlines() == _nodes(result)
    observed = json.loads(lanes.read_text())
    assert observed["mode"] == "sqlite"
    assert set(observed["lanes"]) == set(_nodes(result))
    assert observed["lanes"]["tests/test_alpha.py::test_host[0]"] == "host-journey"
    assert observed["lanes"]["tests/test_semantics.py::test_race[postgres]"] == (
        "postgres-conformance"
    )
    assert observed["lanes"]["tests/qualification/test_capacity.py::test_capacity[0]"] == (
        "qualification"
    )


def test_collection_gate_rejects_a_marker_its_family_would_not_collect(shard_project, tmp_path):
    # The marker arrives from a helper outside tests/, so the test module never names it.
    project = tmp_path / "project"
    subprocess.run(["cp", "-R", str(shard_project), str(project)], check=True)
    (project / "sample_helpers.py").write_text(
        "import pytest\njourney = getattr(pytest.mark, 'host' + '_journey')\n"
    )
    (project / "tests/test_hidden.py").write_text(
        "from sample_helpers import journey\n@journey\ndef test_hidden(): pass\n"
    )
    result = _pytest(project, "--collect-only", "-q", "-n", "0", "--ci-check-plan")
    assert result.returncode != 0
    assert "tests/test_hidden.py::test_hidden (host-journey)" in result.stdout + result.stderr


def test_work_stealing_shard_records_controller_durations(shard_project, tmp_path):
    durations = tmp_path / "durations/general.json"
    lane = next(
        lane.name
        for lane in build_plan(shard_project, "sqlite").lanes.values()
        if lane.family == "general" and "tests/test_beta.py" in lane.paths
    )
    result = _pytest(
        shard_project,
        "-q",
        "-n",
        "2",
        "--dist",
        "worksteal",
        "--ci-shard",
        lane,
        "--ci-durations",
        str(durations),
        env={"CAYU_TEST_STORES": "sqlite"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("CI worker gw") == 2
    measured = json.loads(durations.read_text())
    assert "tests/test_beta.py::test_general[0]" in measured
    assert not any("test_backend" in node and "memory" in node for node in measured)


def _alpha_lane(root: Path) -> str:
    plan = build_plan(root, "sqlite")
    return next(
        lane.name
        for lane in plan.lanes.values()
        if lane.family == "general" and "tests/test_alpha.py" in lane.paths
    )


def test_shard_preserves_immediate_ci_failure_reporting(shard_project):
    result = _pytest(
        shard_project,
        "-q",
        "-n",
        "2",
        "--dist",
        "worksteal",
        "--ci-shard",
        _alpha_lane(shard_project),
        env={"GITHUB_ACTIONS": "true", "CI_SAMPLE_FAIL": "1", "CAYU_TEST_STORES": "sqlite"},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert "CI test started:" in output
    assert "CI test finished:" in output
    assert "CI failure:" in output
    assert "shard-immediate-failure" in output


def test_worker_crash_is_reported_without_internal_error(shard_project):
    result = _pytest(
        shard_project,
        "-q",
        "-n",
        "2",
        "--max-worker-restart",
        "0",
        "--dist",
        "worksteal",
        "--ci-shard",
        _alpha_lane(shard_project),
        env={"CI_SAMPLE_CRASH": "1", "CAYU_TEST_STORES": "sqlite"},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "INTERNALERROR" not in result.stdout + result.stderr
    assert "crashed" in result.stdout


def test_unassigned_modules_are_omitted_before_import(shard_project):
    result = _pytest(
        shard_project,
        "-q",
        "--ci-shard",
        _alpha_lane(shard_project),
        env={"CI_SAMPLE_FORBID_QUALIFICATION": "1", "CAYU_TEST_STORES": "sqlite"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_shard_without_matching_tests_succeeds(shard_project):
    # Every sample file is small, so most PR shards hold nothing.
    plan = build_plan(shard_project, "sqlite")
    empty = next(
        lane.name for lane in plan.lanes.values() if lane.family == "general" and not lane.units
    )
    result = _pytest(shard_project, "-q", "--ci-shard", empty, env={"CAYU_TEST_STORES": "sqlite"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no tests match this lane" in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ("-m", "postgres"),
        ("-k", "backend"),
        ("--splits", "2", "--group", "1"),
        ("--ignore", "tests/test_alpha.py"),
        ("--deselect", "tests/test_alpha.py::test_general[0]"),
        ("--lf",),
        ("tests/test_alpha.py",),
    ],
)
def test_additional_selection_cannot_shrink_a_shard(shard_project, args):
    result = _pytest(shard_project, "--collect-only", "-q", "--ci-shard", "general-1", *args)
    assert result.returncode == 4
    assert "CI shards cannot" in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ("--ci-check-plan",),
        ("--collect-only", "-n", "2", "--ci-check-plan"),
        ("--collect-only", "--ci-shard", "general-99"),
    ],
)
def test_invalid_plan_requests_fail_before_tests(shard_project, args):
    result = _pytest(shard_project, "-q", *args)
    assert result.returncode == 4, result.stdout + result.stderr
