from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import ci_test_profile as profile
from tests.core.test_ci_test_manifest import _pytest as manifest_pytest
from tests.core.test_ci_test_manifest import manifest_project as manifest_project

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_profile_composes_with_manifest_selection_without_changing_assignments(
    manifest_project, tmp_path, workers
):
    root, manifest = manifest_project
    output = tmp_path / "profile.json"
    result = manifest_pytest(
        root,
        "-q",
        "-n",
        str(workers),
        "--dist",
        "loadfile",
        "--ci-manifest",
        str(manifest),
        "--ci-lane",
        "general-1",
        "-p",
        "scripts.ci_test_profile",
        "--ci-profile",
        str(output),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    nodes = json.loads(manifest.read_text())["lanes"]["general-1"]
    report = json.loads(output.read_text())
    expected = {"count": len(nodes), "nodeids_sha256": profile._digest(nodes)}
    assert report["executed"] == expected
    assert all(value == expected for value in report["collections"].values())
    assert report["outcomes"]["call.passed"] == len(nodes)


@pytest.fixture
def profile_project(tmp_path):
    root = tmp_path / "project"
    (root / "tests").mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\ntestpaths=tests\n")
    body = """import json, os, subprocess, sys
from pathlib import Path
import pytest

@pytest.fixture(scope="module")
def owner():
    path = Path(os.environ["PROFILE_EVENTS"]) / (Path(__file__).stem + ".json")
    assert not path.exists(), "module fixture ran on multiple workers"
    path.write_text(json.dumps({"closed": False}))
    yield path
    path.write_text(json.dumps({"closed": True}))

@pytest.mark.parametrize("case", range(8))
def test_work(case, owner):
    if Path(__file__).stem == "test_alpha" and case == 0:
        if os.environ.get("PROFILE_CRASH"): os._exit(42)
        assert not os.environ.get("PROFILE_FAIL"), "profile-failure"
    if case == 0:
        subprocess.run([sys.executable, "-c", "sum(i*i for i in range(100000))"], check=True)
    assert sum(i*i for i in range(1000)) > 0
"""
    for name in ("alpha", "beta", "gamma", "delta"):
        (root / "tests" / f"test_{name}.py").write_text(body)
    return root


def _run(root, tmp_path, workers, *, enabled=True, extra=(), env=None):
    output = tmp_path / "profile.json"
    events = tmp_path / "events"
    events.mkdir()
    output.write_text("stale")
    args = ["-p", "scripts.ci_test_profile", "--ci-profile", str(output)] if enabled else []
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--rootdir",
            str(root),
            "-c",
            str(root / "pytest.ini"),
            "-q",
            "-n",
            str(workers),
            "--dist",
            "loadfile",
            *args,
            *extra,
        ],
        cwd=root,
        env=os.environ
        | {
            "PYTHONPATH": str(_ROOT),
            "PROFILE_EVENTS": str(events),
        }
        | (env or {}),
        text=True,
        capture_output=True,
        timeout=90,
    )
    return result, output, events


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_profile_preserves_inventory_results_and_file_fixture_ownership(
    profile_project, tmp_path, workers
):
    result, output, events = _run(profile_project, tmp_path, workers)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    expected = [
        f"tests/test_{name}.py::test_work[{case}]"
        for name in ("alpha", "beta", "gamma", "delta")
        for case in range(8)
    ]
    assert report["executed"] == {"count": 32, "nodeids_sha256": profile._digest(expected)}
    assert report["outcomes"] == {
        "setup.passed": 32,
        "call.passed": 32,
        "teardown.passed": 32,
    }
    assert report["exitstatus"] == 0
    assert report["requested_workers"] == workers
    assert len(report["workers"]) == workers
    assert len(report["collections"]) == workers
    assert all(value == report["executed"] for value in report["collections"].values())
    assert len(list(events.iterdir())) == 4
    assert all(json.loads(path.read_text())["closed"] for path in events.iterdir())
    for worker in report["workers"].values():
        assert worker["completed"]
        measured = worker["measurements"]
        assert measured["collection"]["wall_seconds"] >= 0
        assert measured["execution"]["wall_seconds"] >= 0
        assert measured["session"]["self"]["user_seconds"] >= 0
        assert measured["lifetime_peak_rss_bytes"]["self"] > 0
    assert (
        sum(
            worker["measurements"]["execution"]["waited_children"]["user_seconds"]
            for worker in report["workers"].values()
        )
        > 0
    )
    assert len(report["slowest_tests_seconds"]) == 20
    assert len(report["slowest_files_seconds"]) == 4
    assert str(profile_project) not in output.read_text()
    assert "CI execution profile:" in result.stdout


def test_serial_profile_has_no_fabricated_worker_measurements(profile_project, tmp_path):
    result, output, _ = _run(profile_project, tmp_path, 0)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert not report["workers"]
    assert report["executed"]["count"] == 32
    assert "process" in report and "controller" not in report
    assert report["process"]["execution"]["self"]["user_seconds"] > 0


def test_profiling_is_optional(profile_project, tmp_path):
    result, output, _ = _run(profile_project, tmp_path, 2, enabled=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text() == "stale"
    assert "CI execution profile:" not in result.stdout


@pytest.mark.parametrize("fault", ["PROFILE_FAIL", "PROFILE_CRASH"])
def test_failed_tests_and_worker_crashes_remain_failed_with_partial_evidence(
    profile_project, tmp_path, fault
):
    result, output, _ = _run(
        profile_project,
        tmp_path,
        2,
        extra=("--max-worker-restart", "0"),
        env={fault: "1"},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "INTERNALERROR" not in result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["exitstatus"] == 1
    if fault == "PROFILE_CRASH":
        assert any(not worker["completed"] for worker in report["workers"].values())
    else:
        assert report["outcomes"]["call.failed"] == 1
        assert report["executed"]["count"] == 32


def test_collection_errors_replace_stale_profiles_with_failure_evidence(profile_project, tmp_path):
    (profile_project / "tests/test_broken.py").write_text("raise RuntimeError('broken-collection')")
    result, output, _ = _run(profile_project, tmp_path, 0)
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["exitstatus"] == 2
    assert report["executed"]["count"] == 0
    assert report["outcomes"] == {}


def test_resource_deltas_keep_unknown_and_reset_counters_unavailable():
    assert profile._delta(None, {"x": 1}) is None
    assert profile._delta({"x": 1}, None) is None
    assert profile._delta(
        {"user_seconds": 3, "input_blocks": 2, "missing": 4, "peak_rss_bytes": 100},
        {"user_seconds": 5, "input_blocks": 1, "peak_rss_bytes": 150},
    ) == {"user_seconds": 2, "input_blocks": None, "missing": None}


@pytest.mark.parametrize(("system", "factor"), [("darwin", 1), ("linux", 1024)])
def test_peak_rss_uses_platform_units_and_is_never_a_phase_delta(monkeypatch, system, factor):
    usage = SimpleNamespace(
        **{
            field: 7
            for field in (
                "ru_utime",
                "ru_stime",
                "ru_maxrss",
                "ru_majflt",
                "ru_inblock",
                "ru_oublock",
                "ru_nvcsw",
                "ru_nivcsw",
            )
        }
    )
    monkeypatch.setattr(profile.sys, "platform", system)
    monkeypatch.setattr(profile.resource, "getrusage", lambda _who: usage)
    assert profile._usage(0)["peak_rss_bytes"] == 7 * factor
    assert "peak_rss_bytes" not in profile._delta(profile._usage(0), profile._usage(0))


def test_linux_readers_preserve_missing_data_and_parse_pressure_counters(tmp_path):
    missing = tmp_path / "missing"
    assert profile._counters(missing) is None
    assert profile._pressure(missing) is None
    path = tmp_path / "metrics"
    path.write_text("read_bytes: 42\nwrite_bytes: 100\n")
    assert profile._counters(path) == {"read_bytes": 42, "write_bytes": 100}
    path.write_text("some avg10=0.00 avg60=0.00 avg300=0.00 total=42\nfull total=7\n")
    assert profile._pressure(path) == {"some": 42, "full": 7}
    path.write_text("bad data")
    assert profile._counters(path) is None
    assert profile._pressure(path) is None


@pytest.mark.parametrize("mount_root", ["/", "/host/parent"])
def test_cgroup_mapping_resolves_mount_root_without_exposing_host_paths(
    tmp_path, monkeypatch, mount_root
):
    mount = tmp_path / "cgroup"
    (mount / "worker").mkdir(parents=True)
    paths = {
        "/proc/self/cgroup": f"0::{mount_root.rstrip('/')}/worker\n",
        "/proc/self/mountinfo": f"1 2 0:1 {mount_root} {mount} rw - cgroup2 cgroup rw\n",
    }
    read = profile._read
    monkeypatch.setattr(profile, "_read", lambda path: paths.get(str(path), read(path)))
    assert profile._cgroup() == (mount / "worker", mount)
    (mount / "cpu.max").write_text("200000 100000")
    (mount / "worker/cpu.max").write_text("max 100000")
    (mount / "memory.max").write_text("4294967296")
    snapshot = profile._host_snapshot()
    limits = snapshot["cgroup_v2"]["visible_ancestor_limits"]
    assert limits == [
        {"cpu_max": "max 100000", "memory_max": None},
        {"cpu_max": "200000 100000", "memory_max": "4294967296"},
    ]
    assert str(mount) not in json.dumps(snapshot)


def test_missing_cgroup_membership_is_unavailable(monkeypatch):
    monkeypatch.setattr(profile, "_read", lambda _path: None)
    assert profile._cgroup() is None
    assert profile._host_snapshot()["cgroup_v2"] is None


def _workflow_script(workflow, name):
    [step] = re.findall(
        rf"      - name: {re.escape(name)}\n(.*?)(?=      - |\Z)",
        workflow,
        re.DOTALL,
    )
    assert "shell: bash" in step
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


def _workflow_shell(script, env):
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        env=os.environ | env,
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.fixture
def manual_profile_workflow(tmp_path):
    workflow = (_ROOT / ".github/workflows/ci-profiling.yml").read_text()
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['CALLS']).open('a') as out: out.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "print('retained run output', flush=True)\n"
        "sys.exit(13 if os.environ.get('SAMPLE_FAIL') else 0)\n"
    )
    python.chmod(0o755)
    return workflow, {
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        "REVISION": "a" * 40,
        "CALLS": str(tmp_path / "calls.jsonl"),
        "SAMPLE_FAIL": "",
    }


@pytest.mark.parametrize("repetitions", ["1", "2"])
def test_manual_workflow_profiles_each_requested_configuration_once(
    manual_profile_workflow, repetitions
):
    workflow, env = manual_profile_workflow
    assert "workflow_dispatch:" in workflow and "pull_request:" not in workflow
    assert "push:" not in workflow and "continue-on-error:" not in workflow
    for reference in re.findall(r"uses: ([^\s#]+)", workflow):
        assert reference.startswith("./") or re.fullmatch(r"[^@]+@[a-f0-9]{40}", reference)
    result = _workflow_shell(
        _workflow_script(workflow, "Validate comparison inputs"),
        env | {"REPETITIONS": repetitions},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    key, value = Path(env["GITHUB_OUTPUT"]).read_text().strip().split("=", 1)
    assert key == "repetitions"
    rounds = json.loads(value)
    assert rounds == list(range(1, int(repetitions) + 1))
    assert "repetitions: ${{ steps.inputs.outputs.repetitions }}" in workflow
    assert "repetition: ${{ fromJSON(needs.collection.outputs.repetitions) }}" in workflow
    [lanes] = re.findall(r"        lane: (\[.*\])", workflow)
    [workers] = re.findall(r"        workers: (\[.*\])", workflow)
    assert json.loads(lanes) == ["general-1", "postgres-conformance-1"]
    assert json.loads(workers) == [2, 4, 8]
    assert "fail-fast: false" in workflow and "max-parallel: 2" in workflow
    benchmark = workflow.split("  benchmark:\n", 1)[1]
    [job_timeout] = re.findall(r"^    timeout-minutes: (\d+)$", benchmark, re.MULTILINE)
    step_timeouts = [
        int(value)
        for value in re.findall(r"^        timeout-minutes: (\d+)$", benchmark, re.MULTILINE)
    ]
    assert int(job_timeout) <= 55
    assert int(job_timeout) >= sum(step_timeouts)
    assert "timeout-minutes: 45" in benchmark
    upload = workflow.split("      - name: Retain", 1)[1]
    assert "if: always()" in upload and "timeout-minutes: 5" in upload
    assert "overwrite:" not in upload and "if-no-files-found: error" in upload
    [artifact_name] = re.findall(r"          name: (execution-profile-.*)", upload)
    artifact_names = set()
    commands_expected = []
    script = _workflow_script(workflow, "Profile one worker count on the assigned tests")
    for lane in json.loads(lanes):
        for count in json.loads(workers):
            for repetition in rounds:
                result = _workflow_shell(script, env | {"LANE": lane, "WORKERS": str(count)})
                assert result.returncode == 0, result.stdout + result.stderr
                selection = (
                    ["--lane", "general", "--shard", "1"]
                    if lane == "general-1"
                    else ["--lane", "specialist", "--specialist-lane", lane]
                )
                commands_expected.append(
                    [
                        "scripts/run_ci.py",
                        *selection,
                        "--test-manifest",
                        str(Path(env["RUNNER_TEMP"]) / "profile-manifest.json"),
                        "--test-workers",
                        str(count),
                        "--profile-tests",
                        str(Path(env["RUNNER_TEMP"]) / "ci-profiles"),
                    ]
                )
                for attempt in (1, 2):
                    name = artifact_name
                    for field, value in {
                        "matrix.lane": lane,
                        "matrix.workers": count,
                        "matrix.repetition": repetition,
                        "github.run_attempt": attempt,
                    }.items():
                        name = name.replace("${{ " + field + " }}", str(value))
                    assert "${{" not in name and name not in artifact_names
                    artifact_names.add(name)
    commands = [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]
    assert commands == commands_expected
    assert len(commands) == 6 * int(repetitions)


@pytest.mark.parametrize("lane", ["general-1", "postgres-conformance-1"])
def test_manual_workflow_preserves_test_failure_and_log(manual_profile_workflow, lane):
    workflow, env = manual_profile_workflow
    result = _workflow_shell(
        _workflow_script(workflow, "Profile one worker count on the assigned tests"),
        env | {"LANE": lane, "WORKERS": "4", "SAMPLE_FAIL": "1"},
    )
    assert result.returncode == 13, result.stdout + result.stderr
    assert len(Path(env["CALLS"]).read_text().splitlines()) == 1
    assert (Path(env["RUNNER_TEMP"]) / "ci-profiles/run.log").read_text() == "retained run output\n"


@pytest.mark.parametrize(
    ("revision", "repetitions"),
    [("main", "2"), ("a" * 39, "2"), ("a" * 40, "0"), ("a" * 40, "3")],
)
def test_manual_workflow_rejects_invalid_inputs(manual_profile_workflow, revision, repetitions):
    workflow, env = manual_profile_workflow
    result = _workflow_shell(
        _workflow_script(workflow, "Validate comparison inputs"),
        env | {"REVISION": revision, "REPETITIONS": repetitions},
    )
    assert result.returncode != 0
    assert not Path(env["GITHUB_OUTPUT"]).exists()
    assert not Path(env["CALLS"]).exists()
