from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

_CONFTEST = runpy.run_path(str(Path(__file__).parents[1] / "conftest.py"))
_requests_postgres = _CONFTEST["_requests_postgres"]
_require_current_test_durations = _CONFTEST["_require_current_test_durations"]


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
