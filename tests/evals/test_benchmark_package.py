from __future__ import annotations

import json
from dataclasses import replace

import pytest

from cayu.evals.benchmark_package import (
    BENCHMARK_PACKAGE_MAX_BYTES,
    BenchmarkFileV1,
    BenchmarkPackageV1,
    benchmark_package_from_json,
    benchmark_package_scenarios,
    benchmark_package_to_json,
    benchmark_suite_selection,
    load_benchmark_package,
)
from cayu.evals.benchmark_synthetic import (
    synthetic_benchmark_package,
    write_synthetic_benchmark_package,
)
from cayu.evals.corpus import _content_revision
from cayu.evals.store import validate_authored_suite_scenario
from cayu.evals.suite_authoring import EvalScenarioStimulusV1


def _changed(package, **changes):
    material = {**package.model_dump(mode="json", exclude={"revision"}), **changes}
    return BenchmarkPackageV1(revision=_content_revision(material, "benchmark package"), **material)


def test_package_roundtrip_selection_and_identity(tmp_path):
    manifest = write_synthetic_benchmark_package(tmp_path / "package")
    loaded = load_benchmark_package(manifest)
    package = loaded.package
    assert package == benchmark_package_from_json(benchmark_package_to_json(package))
    suite, selection = benchmark_suite_selection(package, ["echo", "attachment"])
    assert [case.id for case in selection.cases] == ["attachment", "echo"]
    assert selection == benchmark_suite_selection(package, ["attachment", "echo"])[1]
    assert selection.revision != benchmark_suite_selection(package)[1].revision
    assert all(case.source.evidence_revision == package.revision for case in suite.cases)
    assert len(loaded.file_contents) == 1
    assert loaded.file_contents[0][1].startswith(b"\x89PNG")
    # A validated input snapshot stays fixed after the mutable source changes.
    (loaded.root / "inputs/pixel.png").write_bytes(b"changed")
    assert loaded.file_contents[0][1].startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="immutable requirement"):
        load_benchmark_package(manifest)


@pytest.mark.parametrize("dimension", ["version", "scorer_version"])
def test_package_versions_change_existing_comparable_case_identities(dimension):
    package = synthetic_benchmark_package()
    changed = _changed(package, **{dimension: "next"})
    before, _ = benchmark_suite_selection(package)
    after, _ = benchmark_suite_selection(changed)
    assert package.revision != changed.revision
    assert all(a.revision != b.revision for a, b in zip(before.cases, after.cases, strict=True))


@pytest.mark.parametrize("case_ids", [[], ["missing"], ["echo", "echo"]])
def test_selection_rejects_ambiguous_or_unknown_cases(case_ids):
    with pytest.raises((ValueError, KeyError)):
        benchmark_suite_selection(synthetic_benchmark_package(), case_ids)


@pytest.mark.parametrize(
    "path", ["../truth.json", "/etc/passwd", "a/../b", "a//b", "a\\b", "C:foo"]
)
def test_file_paths_are_package_relative(path):
    with pytest.raises(ValueError, match="relative POSIX"):
        BenchmarkFileV1(scenario_revision="sha256:" + "a" * 64, requirement_id="file", path=path)


def test_missing_file_and_symlink_are_rejected(tmp_path):
    manifest = write_synthetic_benchmark_package(tmp_path / "package")
    image = manifest.parent / "inputs/pixel.png"
    image.unlink()
    with pytest.raises(ValueError, match="missing"):
        load_benchmark_package(manifest)
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    image.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic links"):
        load_benchmark_package(manifest)


@pytest.mark.parametrize("field", ["scenarios", "files"])
def test_package_requires_all_case_material(field):
    with pytest.raises(ValueError, match="missing|cover exactly"):
        _changed(synthetic_benchmark_package(), **{field: []})


def test_strict_manifest_and_bounded_reader(tmp_path):
    source = benchmark_package_to_json(synthetic_benchmark_package())
    with pytest.raises(ValueError):
        benchmark_package_from_json(source[:-1] + ',"schema_version":1}')
    for value in (True, "1", 1.0):
        document = json.loads(source)
        document["schema_version"] = value
        with pytest.raises(ValueError):
            benchmark_package_from_json(json.dumps(document))
    manifest = tmp_path / "benchmark.json"
    manifest.write_bytes(b" " * (BENCHMARK_PACKAGE_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="byte limit"):
        load_benchmark_package(manifest)


def test_scorer_truth_does_not_enter_loaded_agent_files(tmp_path):
    loaded = load_benchmark_package(write_synthetic_benchmark_package(tmp_path / "package"))
    assert "CORRECT" in benchmark_package_to_json(loaded.package)
    assert all(b"CORRECT" not in content for _, content in loaded.file_contents)
    assert replace(loaded, root=tmp_path).package.revision == loaded.package.revision


def test_existing_package_directory_is_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        write_synthetic_benchmark_package(tmp_path)


def test_resolved_scenarios_preserve_native_catalog_source_contract():
    package = synthetic_benchmark_package()
    suite, _ = benchmark_suite_selection(package)
    scenarios = {item.revision: item for item in benchmark_package_scenarios(package)}
    for case in suite.cases:
        if type(case.stimulus) is EvalScenarioStimulusV1:
            scenario = scenarios[case.stimulus.scenario_revision]
            validate_authored_suite_scenario(suite, case, scenario)
            assert scenario.source == case.source
            assert scenario.source.evidence_revision == package.revision
