from __future__ import annotations

import json
import runpy
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.qualification.registry import MAINTENANCE_SCENARIOS

_CONFTEST = runpy.run_path(str(Path(__file__).parents[1] / "conftest.py"))
_requests_postgres = _CONFTEST["_requests_postgres"]
_require_current_test_durations = _CONFTEST["_require_current_test_durations"]


# Half of the qualification runner's 300-second per-scenario bound, leaving
# headroom for loaded runners and per-process startup.
_MAINTENANCE_SCENARIO_RECORDED_SECONDS = 150


def test_maintenance_partition_selects_every_collected_case_exactly_once() -> None:
    root = Path(__file__).resolve().parents[2]
    modules = {
        str(path.relative_to(root))
        for path in (root / "tests/qualification").glob("test_repository_maintenance_*.py")
    }
    modules.update(
        selector.split("::", 1)[0]
        for scenario in MAINTENANCE_SCENARIOS
        for selector in scenario.selectors
    )
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *sorted(modules)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert collected.returncode == 0, collected.stdout + collected.stderr
    nodes = [
        line for line in collected.stdout.splitlines() if line.startswith("tests/") and "::" in line
    ]
    assert nodes and len(nodes) == len(set(nodes))
    durations = json.loads((root / ".test_durations").read_text(encoding="utf-8"))
    coverage = Counter()
    for scenario in MAINTENANCE_SCENARIOS:
        selected = []
        for selector in scenario.selectors:
            matches = [
                node
                for node in nodes
                if node == selector or node.startswith((selector + "::", selector + "["))
            ]
            assert matches, f"Empty or obsolete qualification selector: {selector}"
            selected += matches
        coverage.update(selected)
        recorded = sum(durations.get(node, 0) for node in selected)
        assert recorded <= _MAINTENANCE_SCENARIO_RECORDED_SECONDS, (scenario.name, recorded)
    assert set(coverage) == set(nodes), f"Unregistered cases: {set(nodes) - set(coverage)}"
    assert all(count == 1 for count in coverage.values()), {
        node: count for node, count in coverage.items() if count != 1
    }


def test_dynamic_postgres_parameter_is_routed_to_the_postgres_lane() -> None:
    item = SimpleNamespace(
        fixturenames=(),
        callspec=SimpleNamespace(params={"knowledge_store_case": "postgres"}),
    )

    assert _requests_postgres(item)


def test_duration_snapshot_rejects_more_than_five_percent_unknown_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    known = {f"tests/test_example.py::test_{index}": 0.1 for index in range(19)}
    (tmp_path / ".test_durations").write_text(json.dumps(known), encoding="utf-8")
    items = [SimpleNamespace(nodeid=f"tests/test_example.py::test_{index}") for index in range(21)]
    monkeypatch.setenv("CAYU_REQUIRE_CURRENT_TEST_DURATIONS", "1")

    with pytest.raises(pytest.UsageError, match="2 of 21 collected tests lack timings"):
        _require_current_test_durations(
            SimpleNamespace(rootpath=tmp_path),
            items,
        )


def test_duration_snapshot_allows_a_small_new_test_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    known = {f"tests/test_example.py::test_{index}": 0.1 for index in range(19)}
    (tmp_path / ".test_durations").write_text(json.dumps(known), encoding="utf-8")
    items = [SimpleNamespace(nodeid=f"tests/test_example.py::test_{index}") for index in range(20)]
    monkeypatch.setenv("CAYU_REQUIRE_CURRENT_TEST_DURATIONS", "1")

    _require_current_test_durations(
        SimpleNamespace(rootpath=tmp_path),
        items,
    )
