"""Exercise the manifest through real pytest collection and xdist workers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from scripts.ci_test_manifest import _validate_manifest, lane_specs

_ROOT = Path(__file__).resolve().parents[2]


def _pytest(root: Path, *args: str, env: dict[str, str] | None = None):
    process_env = os.environ.copy()
    process_env.pop("CAYU_CI_REPORT_OWNER_PID", None)
    process_env.update(
        PYTHONPATH=os.pathsep.join((str(_ROOT), str(_ROOT / "src"))),
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
            "-p",
            "scripts.ci_test_manifest",
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


@pytest.fixture(scope="module")
def manifest_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("ci-manifest")
    (root / "tests/qualification").mkdir(parents=True)
    (root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    (root / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests\nmarkers =\n"
        "    postgres\n    browser_docker\n    host_journey\n    qualification\n    stress\n"
    )
    # Reuse the real dynamic-marker and duration hooks, including the worker exemption.
    (root / "tests/conftest.py").write_text(
        "from tests.conftest import (pytest_collection_modifyitems, pytest_configure, "
        "pytest_runtest_logstart, pytest_runtest_logfinish, pytest_runtest_logreport)\n"
        "import pytest\n@pytest.fixture(scope='session', params=['first', 'second'])\n"
        "def shared_param(request): return request.param\n"
    )
    body = """import json, os
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
    assert not os.environ.get("CI_SAMPLE_FAIL"), "manifest-immediate-failure"

def test_session_parameter_order(shared_param): pass

@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("case", range(12))
def test_backend(backend, case, resource): pass

@pytest.mark.host_journey
@pytest.mark.parametrize("case", range(3))
def test_host(case, resource): pass

"""
    for name in ("alpha", "beta"):
        (root / f"tests/test_{name}.py").write_text(body)
    (root / "tests/qualification/test_capacity.py").write_text(
        "import os\nif os.environ.get('CI_SAMPLE_FORBID_QUALIFICATION'):\n"
        "    raise RuntimeError('unassigned module was imported')\n"
        "import pytest\n@pytest.mark.parametrize('case', range(16))\n"
        "def test_capacity(case): pass\n"
    )
    (root / "tests/test_browser.py").write_text(
        "import pytest\n@pytest.mark.browser_docker\n@pytest.mark.parametrize('case', range(2))\ndef test_browser(case): pass\n"
    )
    result = _pytest(
        root, "--collect-only", "-q", "-n", "0", env={"CAYU_REQUIRE_CURRENT_TEST_DURATIONS": "0"}
    )
    nodes = _nodes(result)
    (root / ".test_durations").write_text(
        json.dumps(
            {
                node: (i % 7 + 1) / 10
                for i, node in enumerate(nodes)
                if not node.startswith("tests/test_browser.py")
            }
        )
    )
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=CI test",
            "-c",
            "user.email=ci@example.invalid",
            "commit",
            "-qm",
            "Sample test inventory",
        ],
    ):
        subprocess.run(command, cwd=root, check=True, capture_output=True)
    manifest = root / ".pytest_cache/manifest.json"
    result = _pytest(root, "--collect-only", "-q", "-n", "0", "--ci-write-manifest", str(manifest))
    assert result.returncode == 0, result.stdout + result.stderr
    return root, manifest


@pytest.mark.parametrize(
    "lane",
    ["general-1", "postgres-conformance-2", "host-journey-3", "browser-docker", "qualification-2"],
)
def test_scoped_collection_matches_stock_marker_and_duration_selection(manifest_project, lane):
    root, manifest = manifest_project
    marker, splits, group = lane_specs()[lane]
    stock = _pytest(
        root,
        "--collect-only",
        "-q",
        "-n",
        "0",
        "-m",
        marker,
        "--splits",
        str(splits),
        "--group",
        str(group),
        "--splitting-algorithm",
        "least_duration",
    )
    scoped = _pytest(
        root, "--collect-only", "-q", "-n", "0", "--ci-manifest", str(manifest), "--ci-lane", lane
    )
    assert _nodes(stock) == _nodes(scoped) == json.loads(manifest.read_text())["lanes"][lane]


def test_partitions_cover_each_required_node_once_in_original_order(manifest_project):
    _, manifest = manifest_project
    data = json.loads(manifest.read_text())
    assigned = [node for group in data["lanes"].values() for node in group]
    assert Counter(assigned) == Counter({node: 1 for node in data["inventory"]})
    positions = {node: index for index, node in enumerate(data["inventory"])}
    for group in data["lanes"].values():
        assert group == sorted(group, key=positions.__getitem__)
    assert all("postgres" in node for node in data["lanes"]["postgres-conformance-1"])
    assert all(node.startswith("tests/qualification/") for node in data["lanes"]["qualification-1"])


def test_xdist_keeps_file_fixture_ownership_and_teardown(manifest_project, tmp_path):
    root, manifest = manifest_project
    events = tmp_path / "events"
    events.mkdir()
    result = _pytest(
        root,
        "-q",
        "-n",
        "2",
        "--dist",
        "loadfile",
        "--ci-manifest",
        str(manifest),
        "--ci-lane",
        "general-1",
        env={"CI_SAMPLE_EVENTS": str(events)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(manifest.read_text())
    expected = {Path(node.split("::")[0]).stem for node in data["lanes"]["general-1"]}
    assert {p.stem for p in events.iterdir()} == expected
    assert all(json.loads(p.read_text())["closed"] for p in events.iterdir())
    assert result.stdout.count("CI worker gw") == 2


def test_manifest_preserves_immediate_ci_failure_reporting(manifest_project):
    root, manifest = manifest_project
    result = _pytest(
        root,
        "-q",
        "-n",
        "2",
        "--dist",
        "loadfile",
        "--ci-manifest",
        str(manifest),
        "--ci-lane",
        "general-1",
        env={"GITHUB_ACTIONS": "true", "CI_SAMPLE_FAIL": "1"},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert "CI test started:" in output
    assert "CI test finished:" in output
    assert "CI failure:" in output
    assert "manifest-immediate-failure" in output


def test_worker_crash_does_not_mask_failure_with_missing_timing_output(manifest_project):
    root, manifest = manifest_project
    result = _pytest(
        root,
        "-q",
        "-n",
        "2",
        "--max-worker-restart",
        "0",
        "--dist",
        "loadfile",
        "--ci-manifest",
        str(manifest),
        "--ci-lane",
        "general-1",
        env={"CI_SAMPLE_CRASH": "1"},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "INTERNALERROR" not in result.stdout + result.stderr
    assert "crashed" in result.stdout


def test_unassigned_modules_are_omitted_before_import(manifest_project):
    root, manifest = manifest_project
    result = _pytest(
        root,
        "-q",
        "-n",
        "2",
        "--dist",
        "loadfile",
        "--ci-manifest",
        str(manifest),
        "--ci-lane",
        "general-1",
        env={"CI_SAMPLE_FORBID_QUALIFICATION": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "duplicate",
        "unknown",
        "order",
        "routing",
        "revision",
        "durations",
        "python",
        "version",
    ],
)
def test_invalid_manifest_rejected_before_tests(manifest_project, tmp_path, corruption):
    root, manifest = manifest_project
    data = json.loads(manifest.read_text())
    lane = data["lanes"]["general-1"]
    if corruption == "missing":
        lane.pop()
    elif corruption == "duplicate":
        lane.append(lane[0])
    elif corruption == "unknown":
        lane.append("tests/test_missing.py::test_missing")
    elif corruption == "order":
        lane.reverse()
    elif corruption == "routing":
        data["inventory"][lane[0]] = ["postgres"]
    elif corruption == "revision":
        data["source"]["revision"] = "0" * 40
    elif corruption == "durations":
        data["source"]["durations_sha256"] = "0" * 64
    elif corruption == "python":
        data["source"]["python"] = [0, 0]
    elif corruption == "version":
        data["schema_version"] = 100
    damaged = tmp_path / "manifest.json"
    damaged.write_text(json.dumps(data))
    result = _pytest(root, "-q", "--ci-manifest", str(damaged), "--ci-lane", "general-1")
    assert result.returncode == 4, result.stdout + result.stderr
    assert "invalid CI manifest" in result.stderr


@pytest.mark.parametrize("change", ["tracked", "untracked", "duration"])
def test_source_changes_reject_stale_manifest(manifest_project, tmp_path, change):
    root, manifest = manifest_project
    checkout = tmp_path / "checkout"
    shutil.copytree(root, checkout)
    if change == "tracked":
        with (checkout / "tests/test_alpha.py").open("a") as out:
            out.write("\ndef test_added(): pass\n")
    elif change == "untracked":
        (checkout / "tests/test_added.py").write_text("def test_added(): pass\n")
    else:
        (checkout / ".test_durations").write_text("{}\n")
    result = _pytest(checkout, "-q", "--ci-manifest", str(manifest), "--ci-lane", "general-1")
    assert result.returncode == 4, result.stdout + result.stderr
    assert "clean source and test checkout" in result.stderr


def test_missing_collected_node_cannot_silently_drop_a_required_test(manifest_project, tmp_path):
    root, manifest = manifest_project
    data = json.loads(manifest.read_text())
    # A structurally valid assignment still has to agree with actual collection.
    old = data["lanes"]["general-1"][0]
    changed = old.replace("test_", "test_nonexistent_", 1)
    data["inventory"] = {
        changed if node == old else node: marks for node, marks in data["inventory"].items()
    }
    data["lanes"]["general-1"][0] = changed
    _validate_manifest(data, data["source"])
    damaged = tmp_path / "manifest.json"
    damaged.write_text(json.dumps(data))
    result = _pytest(
        root, "--collect-only", "-q", "--ci-manifest", str(damaged), "--ci-lane", "general-1"
    )
    assert result.returncode != 0


@pytest.mark.parametrize(
    "args",
    [
        ("-m", "postgres"),
        ("-k", "backend"),
        ("--splits", "2", "--group", "1"),
        ("--ignore", "tests/test_alpha.py"),
        ("--deselect", "tests/test_alpha.py::test_general[0]"),
        ("--lf",),
    ],
)
def test_additional_selectors_cannot_shrink_manifest_coverage(manifest_project, args):
    root, manifest = manifest_project
    result = _pytest(
        root,
        "--collect-only",
        "-q",
        "--ci-manifest",
        str(manifest),
        "--ci-lane",
        "general-1",
        *args,
    )
    assert result.returncode == 4
    assert "additional test selection" in result.stderr


def test_explicit_paths_cannot_override_manifest_scope(manifest_project):
    root, manifest = manifest_project
    result = _pytest(
        root,
        "--collect-only",
        "-q",
        "--ci-manifest",
        str(manifest),
        "--ci-lane",
        "general-1",
        "tests/test_alpha.py",
    )
    assert result.returncode == 4
    assert "cannot also select test paths" in result.stderr


def test_missing_manifest_fails_without_full_collection_fallback(manifest_project, tmp_path):
    root, _ = manifest_project
    result = _pytest(
        root, "-q", "--ci-manifest", str(tmp_path / "absent.json"), "--ci-lane", "general-1"
    )
    assert result.returncode == 4
    assert "invalid CI manifest" in result.stderr


@pytest.mark.parametrize("args", [("-n", "2"), ("tests/test_alpha.py",)])
def test_manifest_writer_requires_full_serial_collection(manifest_project, tmp_path, args):
    root, _ = manifest_project
    output = tmp_path / "manifest.json"
    result = _pytest(root, "--collect-only", "-q", "--ci-write-manifest", str(output), *args)
    assert result.returncode == 4
    assert not output.exists()


def test_collection_errors_cannot_publish_a_manifest(manifest_project, tmp_path):
    root, _ = manifest_project
    checkout = tmp_path / "checkout"
    shutil.copytree(root, checkout)
    (checkout / "tests/test_broken.py").write_text("raise RuntimeError('broken-collection')\n")
    subprocess.run(["git", "add", "tests/test_broken.py"], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=CI test",
            "-c",
            "user.email=ci@example.invalid",
            "commit",
            "-qm",
            "Broken import fixture",
        ],
        cwd=checkout,
        check=True,
    )
    output = tmp_path / "manifest.json"
    output.write_text("stale manifest")
    result = _pytest(
        checkout, "--collect-only", "-q", "-n", "0", "--ci-write-manifest", str(output)
    )
    assert result.returncode != 0
    assert "broken-collection" in result.stdout
    assert not output.exists()
