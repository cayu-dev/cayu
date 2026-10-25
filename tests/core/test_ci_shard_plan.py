"""Unit-test the CI store switch, shard planner and duration merge."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest
from scripts import ci_shard_plan as plan_module
from scripts.ci_shard_plan import (
    FAMILIES,
    GENERAL_MARKER,
    STORE_MODES,
    StoreVariant,
    build_plan,
    deselected_store_variants,
    family,
    lane_family,
    nodeid_variant,
    store_mode,
    store_slot,
)
from scripts.merge_test_durations import main as merge_main
from scripts.merge_test_durations import merge_durations

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("values", "store", "masked"),
    [
        (("asyncio", "memory", "cancel"), "memory", ("asyncio", "<store>", "cancel")),
        ((("postgres", True), 3), "postgres", (("<store>", True), 3)),
        (("asyncio", True), None, ("asyncio", True)),
        (("memory", "sqlite"), None, ("memory", "sqlite")),
        ((("index", 2),), None, (("index", 2),)),
    ],
)
def test_store_slot_finds_exactly_one_store(values, store, masked):
    assert store_slot(values) == (store, masked)


def _variants(*rows):
    return [StoreVariant(key, group, store, protected) for key, group, store, protected in rows]


def test_sqlite_mode_skips_only_twins_of_a_sqlite_variant():
    variants = _variants(
        ("memory-a", "a", "memory", False),
        ("sqlite-a", "a", "sqlite", False),
        ("postgres-a", "a", "postgres", False),
        ("memory-b", "b", "memory", False),
        ("postgres-b", "b", "postgres", False),
        ("memory-c", "c", "memory", True),
        ("sqlite-c", "c", "sqlite", True),
        ("plain", "d", None, False),
    )
    assert deselected_store_variants(variants, "sqlite") == {"memory-a", "postgres-a"}
    assert deselected_store_variants(variants, "all") == set()
    with pytest.raises(ValueError):
        deselected_store_variants(variants, "memory")


def test_node_ids_group_variants_by_every_other_parameter():
    ids = [
        "tests/t.py::test_x[asyncio-memory-cancel]",
        "tests/t.py::test_x[asyncio-sqlite-cancel]",
        "tests/t.py::test_x[asyncio-postgres-restart]",
        "tests/t.py::test_x[asyncio-sqlite-restart]",
        "tests/t.py::test_x[asyncio-memory-fork]",
        "tests/t.py::test_y",
    ]
    skipped = deselected_store_variants(map(nodeid_variant, ids), "sqlite")
    assert skipped == {ids[0], ids[2]}
    assert nodeid_variant(ids[-1]).store is None


@pytest.mark.parametrize(
    ("nodeid", "kept"),
    [
        ("tests/core/test_tool_effect_store_conformance.py::test_x[sqlite]", True),
        ("tests/core/test_task_schedule_races.py::test_x[postgres]", True),
        ("tests/core/test_context_selection_migrations.py::test_x[memory]", True),
        ("tests/core/test_a.py::test_persistent_two_connection_race[postgres]", True),
        ("tests/core/test_a.py::test_concurrent_claims_finalize_once[postgres]", True),
        ("tests/core/test_a.py::test_owner_time_is_sampled_after_lock_acquisition[memory]", True),
        ("tests/core/test_a.py::test_task_ownership_uses_the_store_clock[postgres]", True),
        ("tests/core/test_a.py::TestGroup::test_competing_owners[postgres]", True),
        ("tests/core/test_a.py::test_native_reply_and_cancellation_workers_elect_once[pg]", True),
        ("tests/core/test_a.py::test_independent_applications_accept_once[postgres]", True),
        ("tests/core/test_a.py::test_exclusion_and_admission_have_one_winner[postgres]", True),
        ("tests/core/test_a.py::test_idle_loser_and_finalizer_release_together[postgres]", True),
        ("tests/core/test_a.py::test_selection_permit_orders_two_instance_disablement[pg]", True),
        # "selected" is not an election, and "block" is not locking. The storage
        # words must name the module.
        ("tests/core/test_a.py::test_selected_fallback_survives_fork[postgres]", False),
        ("tests/core/test_a.py::test_pending_capture_blocks_delete[postgres]", False),
        ("tests/core/test_restore_flow.py::test_plain[postgres]", False),
        ("tests/core/test_a.py::test_public_question_flow[postgres]", False),
    ],
)
def test_storage_modules_and_concurrency_tests_keep_every_store(nodeid, kept):
    assert plan_module.keeps_every_store(nodeid) is kept
    twin = re.sub(r"\[(memory|postgres)\]$", "[sqlite]", nodeid)
    if twin != nodeid:
        variants = [
            nodeid_variant(twin),
            nodeid_variant(nodeid, protected=plan_module.keeps_every_store(nodeid)),
        ]
        assert (nodeid in deselected_store_variants(variants, "sqlite")) is not kept


def test_store_mode_defaults_to_every_store_and_rejects_unknown_values():
    assert store_mode({}) == "all"
    assert store_mode({"CAYU_TEST_STORES": " SQLite "}) == "sqlite"
    with pytest.raises(ValueError, match="CAYU_TEST_STORES"):
        store_mode({"CAYU_TEST_STORES": "memory"})


def _project(tmp_path: Path, durations: dict[str, float]) -> Path:
    tests = tmp_path / "tests"
    (tests / "core/conformance").mkdir(parents=True)
    (tests / "qualification").mkdir()
    (tests / "__init__.py").write_text("")
    (tests / "conftest.py").write_text("postgres_dsn = None\n")
    (tests / "core/support.py").write_text("STORES = ('memory', 'sqlite', 'postgres')\n")
    (tests / "core/conformance/conftest.py").write_text("from tests.core import support\n")
    (tests / "core/test_big.py").write_text("def test_a(): pass\n")
    (tests / "core/test_imports.py").write_text("from tests.core.support import STORES\n")
    (tests / "core/test_plain.py").write_text("def test_plain(): pass\n")
    (tests / "core/test_new.py").write_text("def test_new(): pass\n")
    (tests / "core/conformance/test_case.py").write_text("def test_case(): pass\n")
    (tests / "core/test_journey.py").write_text("import pytest\npytest.mark.host_journey\n")
    (tests / "qualification/test_scale.py").write_text("def test_scale(): pass\n")
    (tmp_path / ".test_durations").write_text(json.dumps(durations))
    return tmp_path


def _durations() -> dict[str, float]:
    durations = {f"tests/core/test_big.py::test_a[{i}]": 60.0 + i for i in range(40)}
    durations |= {
        f"tests/core/test_imports.py::test_s[{store}-{i}]": 2.0
        for store in ("memory", "sqlite", "postgres")
        for i in range(5)
    }
    durations["tests/core/test_plain.py::test_plain"] = 1.0
    durations["tests/qualification/test_scale.py::test_scale"] = 100.0
    return durations


def test_family_members_follow_markers_imports_and_nested_conftests(tmp_path):
    root = _project(tmp_path, _durations())
    members = plan_module.family_members(root)
    assert "tests/qualification/test_scale.py" not in members["general"]
    assert members["qualification"] == {"tests/qualification/test_scale.py"}
    # The root conftest names Postgres fixtures but does not route every module.
    assert members["postgres-conformance"] == {
        "tests/core/test_imports.py",
        "tests/core/conformance/test_case.py",
    }
    assert members["host-journey"] == {"tests/core/test_journey.py"}


@pytest.mark.parametrize("mode", STORE_MODES)
def test_plan_puts_every_file_in_exactly_one_shard_per_family(tmp_path, mode):
    durations = _durations()
    root = _project(tmp_path, durations)
    plan = build_plan(root, mode, shards={"general": 4})
    for candidate in FAMILIES:
        lanes = [lane for lane in plan.lanes.values() if lane.family == candidate.name]
        assert len(lanes) == (4 if candidate.name == "general" else candidate.shards[mode])
        files = Counter(path for lane in lanes for path in lane.paths)
        assert set(files) == plan.members[candidate.name]
        split = {path for path, count in files.items() if count > 1}
        assert split <= {"tests/core/test_big.py"}
    general = [lane for lane in plan.lanes.values() if lane.family == "general"]
    # 40 minutes in one file exceeds the split threshold, so its pieces spread.
    assert sum("tests/core/test_big.py" in lane.paths for lane in general) == 4
    new_ids = [f"tests/core/test_big.py::test_new[{i}]" for i in range(50)]
    for nodeid in [*durations, *new_ids]:
        if nodeid.startswith("tests/core/test_big.py"):
            assert sum(plan.selects(lane.name, nodeid) for lane in general) == 1
    assert build_plan(root, mode, shards={"general": 4}).lanes == plan.lanes


def test_single_worker_family_splits_a_file_below_its_target_lane_time(tmp_path):
    durations = _durations()
    scale = [f"tests/qualification/test_scale.py::test_scale[{i}]" for i in range(6)]
    del durations["tests/qualification/test_scale.py::test_scale"]
    durations |= dict.fromkeys(scale, 100.0)
    root = _project(tmp_path, durations)
    qualification = family("qualification")
    assert qualification.workers == 1
    assert qualification.split_seconds < qualification.target_minutes * 60
    plan = build_plan(root, "all")
    lanes = [lane for lane in plan.lanes.values() if lane.family == "qualification"]
    # The general eight-minute threshold would split this 600-second file in
    # two; the four-minute qualification threshold spreads it over three lanes.
    assert sum("tests/qualification/test_scale.py" in lane.paths for lane in lanes) == 3
    assert max(lane.wall_seconds() for lane in lanes) <= qualification.split_seconds
    for nodeid in [*scale, "tests/qualification/test_scale.py::test_scale[new]"]:
        assert sum(plan.selects(lane.name, nodeid) for lane in lanes) == 1


def test_unmeasured_files_get_a_default_weight(tmp_path):
    root = _project(tmp_path, _durations())
    plan = build_plan(root, "sqlite", shards={"general": 4})
    units = {unit.path: unit for lane in plan.lanes.values() for unit in lane.units}
    assert units["tests/core/test_new.py"].seconds == pytest.approx(
        plan_module.DEFAULT_FILE_SECONDS + plan_module.COLLECT_SECONDS_PER_FILE
    )


def test_pull_request_plan_weighs_only_selected_store_variants(tmp_path):
    root = _project(tmp_path, _durations())
    weights = {
        mode: sum(
            lane.seconds
            for lane in build_plan(root, mode).lanes.values()
            if lane.family == "general" and "tests/core/test_imports.py" in lane.paths
        )
        for mode in STORE_MODES
    }
    assert weights["all"] - weights["sqlite"] == pytest.approx(10.0)


def test_lane_names_match_run_ci_and_workflow_matrices():
    workflow = (_ROOT / ".github/workflows/python-tests.yml").read_text()
    [shards] = re.findall(r"shard: \$\{\{ fromJSON\((.*)\) \}\}", workflow)
    [lanes] = re.findall(r"lane: \$\{\{ fromJSON\((.*)\) \}\}", workflow)
    expressions = {}
    for name, expression in (("shard", shards), ("lane", lanes)):
        sqlite, every = re.fullmatch(
            r"inputs.stores == 'sqlite' && '(.*)' \|\| '(.*)'", expression
        ).groups()
        expressions[name] = {"sqlite": json.loads(sqlite), "all": json.loads(every)}
    specialists = ("browser-docker", "postgres-conformance", "host-journey")
    for mode in STORE_MODES:
        assert expressions["shard"][mode] == list(range(1, family("general").shards[mode] + 1))
        expected = [lane for name in specialists for lane in family(name).lanes(mode)]
        assert sorted(expressions["lane"][mode]) == sorted(expected)
    qualification = (_ROOT / ".github/workflows/qualification.yml").read_text()
    count = family("qualification").shards["all"]
    assert f"shard: {list(range(1, count + 1))}" in qualification
    assert lane_family("host-journey-2") == (family("host-journey"), 2)
    assert family("host-journey").marker == "host_journey and not (stress or qualification)"


def test_lane_wall_is_bounded_by_its_slowest_test_not_its_largest_file():
    lane = plan_module.Lane("general-1", "general", GENERAL_MARKER, None)
    lane.units = [
        plan_module.Unit("tests/a.py", 0, 1, 100.0, 10.0),
        plan_module.Unit("tests/b.py", 0, 1, 20.0, 20.0),
    ]
    assert lane.wall_seconds(4) == 30.0
    lane.units.append(plan_module.Unit("tests/c.py", 0, 1, 45.0, 45.0))
    assert lane.wall_seconds(4) == 45.0


def _sqlite_lanes(root: Path) -> dict[str, str]:
    # What collection reports: the Postgres fixture routes this module's SQLite
    # variants, one Postgres variant is store_semantics and test_plain is gone.
    lanes = {f"tests/core/test_big.py::test_a[{i}]": "general" for i in range(40)}
    lanes |= {f"tests/core/test_imports.py::test_s[sqlite-{i}]": "general" for i in range(5)}
    lanes["tests/core/test_imports.py::test_s[sqlite-0]"] = "postgres-conformance"
    lanes["tests/core/test_imports.py::test_s[postgres-0]"] = "postgres-conformance"
    lanes["tests/qualification/test_scale.py::test_scale"] = "qualification"
    return lanes


def test_lane_corrections_record_only_where_collection_disagrees(tmp_path):
    durations = _durations()
    root = _project(tmp_path, durations)
    members = plan_module.family_members(root)
    corrections = plan_module.lane_corrections(durations, _sqlite_lanes(root), "sqlite", members)
    assert corrections == {
        "tests/core/test_imports.py::test_s[sqlite-0]": "postgres-conformance",
        "tests/core/test_imports.py::test_s[postgres-0]": "postgres-conformance",
        "tests/core/test_plain.py::test_plain": None,
    }

    plan = build_plan(root, "sqlite", lanes=corrections)
    postgres = sum(
        lane.seconds for lane in plan.lanes.values() if lane.family == "postgres-conformance"
    )
    estimated = build_plan(root, "sqlite", lanes={})
    assert postgres == pytest.approx(
        4.0
        + sum(
            lane.seconds
            for lane in estimated.lanes.values()
            if lane.family == "postgres-conformance"
        )
    )
    plain = [unit for lane in plan.lanes.values() for unit in lane.units]
    assert next(u for u in plain if u.path == "tests/core/test_plain.py").seconds == pytest.approx(
        plan_module.COLLECT_SECONDS_PER_FILE
    )


def test_merge_refreshes_lane_corrections_without_new_timings(tmp_path):
    root = _project(tmp_path / "project", _durations())
    before = (root / ".test_durations").read_text()
    (root / plan_module.LANES_FILE).write_text(json.dumps({"all": {"kept": None}}))
    artifact = tmp_path / "artifacts/test-lanes-sqlite"
    artifact.mkdir(parents=True)
    (artifact / "test-lanes.json").write_text(
        json.dumps({"mode": "sqlite", "lanes": _sqlite_lanes(root)})
    )
    assert (
        merge_main(
            ["--input", str(tmp_path / "artifacts"), "--durations", str(root / ".test_durations")]
        )
        == 0
    )
    assert (root / ".test_durations").read_text() == before
    recorded = json.loads((root / plan_module.LANES_FILE).read_text())
    assert recorded["all"] == {"kept": None}
    assert recorded["sqlite"]["tests/core/test_plain.py::test_plain"] is None
    assert plan_module.load_lanes(root, "sqlite") == recorded["sqlite"]


def test_merge_updates_adds_and_drops_stale_timings():
    current = {"a": 1.0, "b": 2.0, "gone": 3.0}
    merged = merge_durations(current, [{"a": 4.0, "new": 5.0}, {"a": 6.0}], ["a", "b", "new"])
    assert merged == {"a": 6.0, "b": 2.0, "new": 5.0}
    assert merge_durations(current, [{"b": 1.0}]) == {"a": 1.0, "b": 1.0, "gone": 3.0}
