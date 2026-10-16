from __future__ import annotations

import os
import shutil
import unicodedata
from pathlib import Path

import pytest

from cayu.evals import (
    PROJECT_OBJECTIVES_MAX_BYTES,
    MandatoryControlV1,
    ProjectImprovementConfigV1,
    ProjectImprovementError,
    ProjectObjectivesV1,
    discover_project_improvement,
    load_project_improvement,
    load_project_objectives,
    project_improvement_fingerprint,
    project_objectives_fingerprint,
    project_objectives_from_toml,
    validate_project_improvement,
)

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "improvement_objectives"

_MINIMAL = """
schema_version = 1
sampled_review_rate_bps = 500

[[correctness_floor]]
suite_id = "refund-regressions"
"""


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    shutil.copytree(_EXAMPLE, root)
    return root


def _rewrite(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_example_project_declares_objectives_and_improvement_evals_that_validate() -> None:
    declaration = load_project_improvement(_EXAMPLE)

    assert declaration.root == _EXAMPLE
    assert declaration.config.corpora == ("evals/improvement.json",)
    assert declaration.config.evaluator_only_paths == ("evals/reference",)
    assert declaration.objectives.sampled_review_rate_bps == 500
    assert declaration.objectives.mandatory_controls == (
        MandatoryControlV1(tool_name="close_account"),
        MandatoryControlV1(tool_name="issue_refund", approval_gate="refund-over-limit"),
    )
    assert declaration.objectives.mutable_surfaces.body_paths == ("prompts",)
    assert declaration.objectives_fingerprint == project_objectives_fingerprint(
        load_project_objectives(_EXAMPLE / "cayu-objectives.toml")
    )
    assert declaration.objectives_fingerprint.startswith("sha256:")
    assert declaration.protected_paths == (
        "pyproject.toml",
        "cayu-objectives.toml",
        "evals/improvement.json",
        "evals/reference",
    )
    assert validate_project_improvement(declaration) == ()


def test_discovery_walks_up_from_a_nested_directory(tmp_path: Path) -> None:
    root = _project(tmp_path)

    declaration = discover_project_improvement(root / "evals" / "reference")

    assert declaration is not None
    assert declaration.root == root.resolve()
    assert (
        declaration.objectives_fingerprint == load_project_improvement(root).objectives_fingerprint
    )


def test_discovery_stops_at_the_nearest_cayu_project(tmp_path: Path) -> None:
    root = _project(tmp_path)
    nested = root / "agents" / "other"
    nested.mkdir(parents=True)
    (nested / "pyproject.toml").write_text(
        '[tool.cayu]\neval_target = "evals:build"\n', encoding="utf-8"
    )

    assert discover_project_improvement(nested) is None
    assert discover_project_improvement(tmp_path) is None
    with pytest.raises(ProjectImprovementError, match="not configured"):
        load_project_improvement(nested)


def test_fingerprint_ignores_formatting_order_and_defaults_but_not_content() -> None:
    first = project_objectives_from_toml(
        _MINIMAL
        + """
[[correctness_floor]]
suite_id = "audit"
metrics = ["reliability", "status"]

[[mandatory_controls]]
tool_name = "issue_refund"

[[mandatory_controls]]
tool_name = "close_account"
approval_gate = "always"
"""
    )
    second = project_objectives_from_toml(
        """
# Same objectives, different spelling.
sampled_review_rate_bps = 500
schema_version = 1
mandatory_controls = [
  { approval_gate = "always", tool_name = "close_account" },
  { tool_name = "issue_refund" },
]
correctness_floor = [
  { suite_id = "audit", metrics = ["status", "reliability"] },
  { suite_id = "refund-regressions", metrics = ["status", "score", "reliability"] },
]

[approval_rules]
autonomy_grant = "person"

[mutable_surfaces]
execution_profile = false
"""
    )
    changed = first.model_copy(update={"sampled_review_rate_bps": 501})

    assert project_objectives_fingerprint(first) == project_objectives_fingerprint(second)
    assert project_objectives_fingerprint(changed) != project_objectives_fingerprint(first)
    assert [entry.suite_id for entry in first.correctness_floor] == ["audit", "refund-regressions"]


def test_floor_metrics_default_to_the_schema_version_one_dimensions() -> None:
    (floor,) = project_objectives_from_toml(_MINIMAL).correctness_floor

    assert floor.metrics == ("status", "score", "reliability")


def test_declaration_fingerprint_covers_the_improvement_table_but_not_its_order(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    pyproject = root / "pyproject.toml"
    baseline = load_project_improvement(root)

    assert baseline.declaration_fingerprint == project_improvement_fingerprint(
        baseline.config, baseline.objectives
    )
    assert baseline.declaration_fingerprint.startswith("sha256:")
    assert baseline.declaration_fingerprint != baseline.objectives_fingerprint

    _rewrite(pyproject, '["evals/reference"]', '["evals/reference", "evals/expected"]')
    widened = load_project_improvement(root)
    assert widened.objectives_fingerprint == baseline.objectives_fingerprint
    assert widened.declaration_fingerprint != baseline.declaration_fingerprint

    _rewrite(
        pyproject, '["evals/reference", "evals/expected"]', '["evals/expected", "evals/reference"]'
    )
    reordered = load_project_improvement(root)
    assert reordered.config.evaluator_only_paths == ("evals/expected", "evals/reference")
    assert reordered.declaration_fingerprint == widened.declaration_fingerprint


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("sampled_review_rate_bps = 0.5\n", "valid integer"),
        ("sampled_review_rate_bps = 10001\n", "less than or equal to 10000"),
        ("unknown = true\n", "Extra inputs are not permitted"),
        ('[[correctness_floor]]\nsuite_id = "refund-regressions"\n', "each suite once"),
        ('[[correctness_floor]]\nsuite_id = "x"\nmetrics = []\n', "non-empty set"),
        ('[[correctness_floor]]\nsuite_id = "x"\nmetrics = ["latency"]\n', "metrics"),
        (
            '[[mandatory_controls]]\ntool_name = "t"\n'
            '[[mandatory_controls]]\ntool_name = "t"\napproval_gate = "g"\n',
            "redundant",
        ),
        (
            '[[mandatory_controls]]\ntool_name = "t"\n[[mandatory_controls]]\ntool_name = "t"\n',
            "repeat",
        ),
        ('[mutable_surfaces]\nbody_paths = ["../outside"]\n', "project-relative"),
        ('[mutable_surfaces]\nbody_paths = ["/abs"]\n', "project-relative"),
        ('[approval_rules]\nbody_paths = "none"\n', "person"),
        ('[targets]\ncost_per_verified_completion = { amount = "0.10" }\n', "canonical decimal"),
        ("[targets]\nseconds_per_verified_completion = 0\n", "greater than or equal to 1"),
        ("when = 1979-05-27\n", "JSON-compatible"),
    ],
)
def test_objectives_schema_rejects_invalid_documents(extra: str, message: str) -> None:
    source = _MINIMAL
    if extra.startswith("sampled_review_rate_bps"):
        source = source.replace("sampled_review_rate_bps = 500\n", "")
    # TOML keys before the first table header belong to the document root.
    source = source + extra if extra.startswith("[") else extra + source
    with pytest.raises(ValueError, match=message):
        project_objectives_from_toml(source)


def test_objectives_require_schema_version_floor_and_review_rate() -> None:
    with pytest.raises(ValueError, match="schema_version must be the integer 1"):
        project_objectives_from_toml(_MINIMAL.replace("schema_version = 1", 'schema_version = "1"'))
    with pytest.raises(ValueError, match="schema_version"):
        project_objectives_from_toml(_MINIMAL.replace("schema_version = 1", "schema_version = 2"))
    with pytest.raises(ValueError, match="correctness_floor"):
        project_objectives_from_toml("schema_version = 1\nsampled_review_rate_bps = 0\n")
    with pytest.raises(ValueError, match="sampled_review_rate_bps"):
        project_objectives_from_toml(_MINIMAL.replace("sampled_review_rate_bps = 500\n", ""))
    with pytest.raises(ValueError, match="not valid TOML"):
        project_objectives_from_toml("schema_version = ")


def test_objectives_defaults_require_a_person_for_every_change() -> None:
    objectives = project_objectives_from_toml(_MINIMAL)

    assert objectives.approval_rules.model_dump() == {
        "body_paths": "person",
        "execution_profile": "person",
        "model_per_step": "person",
        "autonomy_grant": "person",
    }
    assert objectives.mutable_surfaces.body_paths == ()
    assert not objectives.mutable_surfaces.execution_profile
    assert objectives.mandatory_controls == ()
    assert type(objectives) is ProjectObjectivesV1


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ({"objectives": "o.toml"}, "at least one of eval_targets"),
        ({"objectives": "o.toml", "eval_targets": ["evals"]}, "module:attribute"),
        ({"objectives": "o.toml", "eval_targets": ["a:b", "a:b"]}, "repeat"),
        ({"objectives": "o.toml", "corpora": ["c.json", "C.json"]}, "ignoring case"),
        ({"objectives": "/o.toml", "corpora": ["c.json"]}, "project-relative"),
        ({"objectives": "o.toml", "corpora": "c.json"}, "ordered array"),
        ({"objectives": "o.toml", "corpora": ["c.json"], "other": 1}, "Extra inputs"),
    ],
)
def test_improvement_table_rejects_invalid_declarations(table: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ProjectImprovementConfigV1.model_validate(table)


def test_validator_reports_every_file_level_problem(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _rewrite(
        root / "pyproject.toml",
        'corpora = ["evals/improvement.json"]',
        'corpora = ["evals/improvement.json", "evals/missing.json", "prompts/system.md", '
        '"prompts"]',
    )
    _rewrite(root / "pyproject.toml", '["evals/reference"]', '["evals/reference", "evals/gone"]')
    _rewrite(root / "cayu-objectives.toml", 'body_paths = ["prompts"]', 'body_paths = ["evals"]')
    _rewrite(root / "cayu-objectives.toml", '"refund-regressions"', '"unknown-suite"')

    problems = validate_project_improvement(load_project_improvement(root))

    assert "corpus evals/missing.json does not exist." in problems
    assert "corpus prompts is not a file." in problems
    assert any(problem.startswith("corpus prompts/system.md is invalid:") for problem in problems)
    assert "evaluator-only path evals/gone does not exist." in problems
    assert "body path evals overlaps protected path evals/reference." in problems
    assert "body path evals overlaps protected path evals/improvement.json." in problems
    assert (
        "correctness floor suite unknown-suite is not in any declared corpus or benchmark package."
        in problems
    )


def test_body_paths_that_differ_from_a_protected_path_only_in_case_overlap(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    _rewrite(root / "cayu-objectives.toml", 'body_paths = ["prompts"]', 'body_paths = ["Evals"]')

    problems = validate_project_improvement(load_project_improvement(root))

    assert "body path Evals overlaps protected path evals/reference." in problems


def test_eval_targets_without_a_module_in_the_checkout_are_reported(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _rewrite(
        root / "pyproject.toml",
        "corpora =",
        'eval_targets = ["installed_agent.evals:build_eval"]\ncorpora =',
    )

    problems = validate_project_improvement(load_project_improvement(root))

    assert problems == (
        "eval target installed_agent.evals:build_eval cannot be found, so it cannot be protected.",
    )


def test_eval_target_modules_resolve_to_their_file_at_the_root_or_under_src(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    (root / "src" / "agent" / "evals").mkdir(parents=True)
    (root / "src" / "agent" / "evals" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "agent" / "scoring.py").write_text("def build(): ...\n", encoding="utf-8")
    _rewrite(
        root / "pyproject.toml",
        "corpora =",
        'eval_targets = ["agent.evals:build", "agent.scoring:build"]\ncorpora =',
    )
    _rewrite(
        root / "cayu-objectives.toml",
        'body_paths = ["prompts"]',
        'body_paths = ["prompts", "src/agent/prompts"]',
    )
    (root / "src" / "agent" / "prompts").mkdir()

    declaration = load_project_improvement(root)

    assert declaration.protected_paths[-2:] == (
        "src/agent/evals/__init__.py",
        "src/agent/scoring.py",
    )
    assert validate_project_improvement(declaration) == ()


def test_benchmark_packages_protect_their_directory_and_reject_a_symlinked_manifest(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    package = root / "benchmarks" / "refunds"
    package.mkdir(parents=True)
    (package / "benchmark.json").write_text("{}", encoding="utf-8")
    _rewrite(
        root / "pyproject.toml",
        "corpora =",
        'benchmark_packages = ["benchmarks/refunds/benchmark.json"]\ncorpora =',
    )

    assert "benchmarks/refunds" in load_project_improvement(root).protected_paths

    outside = tmp_path / "benchmark.json"
    outside.write_text("{}", encoding="utf-8")
    (package / "benchmark.json").unlink()
    try:
        (package / "benchmark.json").symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are unavailable")
    _rewrite(root / "pyproject.toml", '"benchmarks/refunds/benchmark.json"', '"benchmarks/refunds"')

    problems = validate_project_improvement(load_project_improvement(root))

    assert any("symbolic links" in problem for problem in problems)


def test_benchmark_packages_resolve_like_the_loader(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "pkg").mkdir()
    (root / "pkg" / "benchmark.json").write_text("{}", encoding="utf-8")
    (root / "pkg" / "custom.json").write_text("{}", encoding="utf-8")
    (root / "benchmark.json").write_text("{}", encoding="utf-8")
    (root / "later").mkdir()
    _rewrite(
        root / "pyproject.toml",
        "corpora =",
        "benchmark_packages = "
        '["benchmark.json", "pkg", "pkg/benchmark.json", "pkg/custom.json", "later/pkg"]'
        "\ncorpora =",
    )

    declaration = load_project_improvement(root)
    problems = validate_project_improvement(declaration)

    # A manifest names its directory, a missing path protects only itself, and a
    # root manifest never protects the whole project.
    assert [path for path in declaration.protected_paths if "pkg" in path or "bench" in path] == [
        "benchmark.json",
        "later/pkg",
        "pkg",
    ]
    assert not any("custom.json/benchmark.json" in problem for problem in problems)
    assert not any(problem.startswith("body path") for problem in problems)
    assert (
        "benchmark package benchmark.json is at the project root; give it its own directory "
        "so its inputs can be protected." in problems
    )
    # Two spellings of one manifest are one package; another manifest beside it is not.
    assert "benchmark packages pkg and pkg/benchmark.json are the same package." in problems
    assert not any("same package" in problem and "custom" in problem for problem in problems)


def test_eval_target_packages_shadow_modules_and_symlinked_modules_are_rejected(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    (root / "evals" / "assistant").mkdir()
    (root / "evals" / "assistant" / "__init__.py").write_text("", encoding="utf-8")
    (root / "evals" / "assistant.py").write_text("", encoding="utf-8")
    (root / "linked").mkdir()
    _rewrite(
        root / "pyproject.toml",
        "corpora =",
        'eval_targets = ["evals.assistant:build", "linked.mod:build"]\ncorpora =',
    )
    try:
        (root / "linked" / "mod.py").symlink_to(root / "prompts" / "system.md")
    except OSError:
        pytest.skip("symbolic links are unavailable")

    declaration = load_project_improvement(root)
    problems = validate_project_improvement(declaration)

    assert "evals/assistant/__init__.py" in declaration.protected_paths
    assert "evals/assistant.py" not in declaration.protected_paths
    assert any(
        problem.startswith("eval target linked.mod:build module") and "symbolic links" in problem
        for problem in problems
    )


def test_eval_targets_protect_the_package_init_files_an_import_runs(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "evals" / "__init__.py").write_text("", encoding="utf-8")
    (root / "evals" / "improvement.py").write_text("def a(): ...\n", encoding="utf-8")
    _rewrite(
        root / "pyproject.toml",
        "corpora =",
        'eval_targets = ["evals.improvement:a", "evals.improvement:b"]\ncorpora =',
    )
    _rewrite(
        root / "cayu-objectives.toml",
        'body_paths = ["prompts"]',
        'body_paths = ["prompts", "evals/__init__.py"]',
    )

    declaration = load_project_improvement(root)

    assert declaration.protected_paths[-2:] == ("evals/__init__.py", "evals/improvement.py")
    assert len(set(declaration.protected_paths)) == len(declaration.protected_paths)
    assert validate_project_improvement(declaration) == (
        "body path evals/__init__.py overlaps protected path evals/__init__.py.",
    )


def test_paths_that_differ_only_in_unicode_normalization_overlap(tmp_path: Path) -> None:
    root = _project(tmp_path)
    composed = unicodedata.normalize("NFC", "evals/référence")
    decomposed = unicodedata.normalize("NFD", composed)
    _rewrite(root / "pyproject.toml", '["evals/reference"]', f'["evals/reference", "{composed}"]')
    _rewrite(
        root / "cayu-objectives.toml",
        'body_paths = ["prompts"]',
        f'body_paths = ["prompts", "{decomposed}"]',
    )

    problems = validate_project_improvement(load_project_improvement(root))

    assert f"body path {decomposed} overlaps protected path {composed}." in problems
    with pytest.raises(ValueError, match="ignoring case"):
        ProjectImprovementConfigV1.model_validate(
            {"objectives": "o.toml", "corpora": [composed, decomposed]}
        )


def test_eval_targets_resolve_like_python_imports_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    # The project's evals/ has no __init__.py, so a regular package under src/ wins.
    (root / "evals" / "scoring.py").write_text("", encoding="utf-8")
    (root / "src" / "evals").mkdir(parents=True)
    (root / "src" / "evals" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "evals" / "scoring.py").write_text("", encoding="utf-8")
    outside = tmp_path / "site"
    (outside / "vendored_evals").mkdir(parents=True)
    (outside / "vendored_evals" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(outside))
    _rewrite(
        root / "pyproject.toml",
        "corpora =",
        'eval_targets = ["evals.scoring:build", "vendored_evals:build"]\ncorpora =',
    )

    declaration = load_project_improvement(root)
    problems = validate_project_improvement(declaration)

    assert declaration.protected_paths[-2:] == ("src/evals/__init__.py", "src/evals/scoring.py")
    assert "evals/scoring.py" not in declaration.protected_paths
    assert problems == (
        f"eval target vendored_evals:build imports {outside / 'vendored_evals' / '__init__.py'}, "
        "which is outside the project, so it cannot be protected.",
    )


def test_symbolic_links_inside_body_and_protected_directories_are_reported(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    try:
        (root / "prompts" / "ref").symlink_to(root / "evals" / "reference")
        (root / "evals" / "reference" / "notes.md").symlink_to(root / "prompts" / "system.md")
    except OSError:
        pytest.skip("symbolic links are unavailable")

    problems = validate_project_improvement(load_project_improvement(root))

    assert problems == (
        "body path prompts contains a symbolic link at prompts/ref.",
        "protected path evals/reference contains a symbolic link at evals/reference/notes.md.",
    )


def test_a_symlinked_pyproject_is_reported(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "pyproject.toml").rename(root / "prompts" / "pyproject.toml")
    try:
        (root / "pyproject.toml").symlink_to(root / "prompts" / "pyproject.toml")
    except OSError:
        pytest.skip("symbolic links are unavailable")

    problems = validate_project_improvement(load_project_improvement(root))

    assert any(problem.startswith("project file pyproject.toml") for problem in problems)


def test_body_paths_with_trailing_dots_or_spaces_overlap_like_on_windows(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _rewrite(root / "cayu-objectives.toml", 'body_paths = ["prompts"]', 'body_paths = ["evals."]')

    problems = validate_project_improvement(load_project_improvement(root))

    assert "body path evals. overlaps protected path evals/reference." in problems


def test_discovery_reports_a_deeply_nested_pyproject_as_a_project_error(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "a = " + "[" * 50_000 + "]" * 50_000 + "\n", encoding="utf-8"
    )

    with pytest.raises(ProjectImprovementError, match="supported TOML nesting"):
        discover_project_improvement(tmp_path)


def test_python_eval_targets_cover_unresolved_floor_suites_and_protect_their_module(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    (root / "evals" / "improvement.py").write_text("def build_eval(): ...\n", encoding="utf-8")
    _rewrite(
        root / "pyproject.toml",
        "corpora =",
        'eval_targets = ["evals.improvement:build_eval"]\ncorpora =',
    )
    _rewrite(root / "cayu-objectives.toml", '"refund-regressions"', '"from-python-target"')

    declaration = load_project_improvement(root)

    assert "evals/improvement.py" in declaration.protected_paths
    assert validate_project_improvement(declaration) == ()


def test_deeply_nested_objectives_are_rejected_as_invalid() -> None:
    nested = "a = " + "[" * 50_000 + "]" * 50_000 + "\n"

    with pytest.raises(ValueError, match="supported TOML nesting"):
        project_objectives_from_toml(_MINIMAL + nested)


def test_objectives_text_is_bounded_in_utf8_bytes() -> None:
    # Under the limit in characters, over it once encoded.
    padding = "#" + "é" * (PROJECT_OBJECTIVES_MAX_BYTES // 2) + "\n"

    with pytest.raises(ValueError, match="bounded TOML text"):
        project_objectives_from_toml(_MINIMAL + padding)


def test_loader_reports_an_unreadable_objectives_file_as_a_project_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)

    def unreadable(path: Path) -> ProjectObjectivesV1:
        raise PermissionError(f"Permission denied: {path}")

    monkeypatch.setattr("cayu.evals.objectives.load_project_objectives", unreadable)
    with pytest.raises(ProjectImprovementError, match="Permission denied"):
        load_project_improvement(root)


def test_loader_rejects_missing_invalid_and_symlinked_objectives(tmp_path: Path) -> None:
    root = _project(tmp_path)
    objectives = root / "cayu-objectives.toml"

    _rewrite(objectives, "sampled_review_rate_bps = 500", "sampled_review_rate_bps = 20000")
    with pytest.raises(ProjectImprovementError, match="cayu-objectives.toml"):
        load_project_improvement(root)

    outside = tmp_path / "outside.toml"
    outside.write_text(_MINIMAL, encoding="utf-8")
    objectives.unlink()
    with pytest.raises(ProjectImprovementError, match="does not exist"):
        load_project_improvement(root)
    if hasattr(os, "symlink"):
        try:
            objectives.symlink_to(outside)
        except OSError:
            pytest.skip("symbolic links are unavailable")
        with pytest.raises(ProjectImprovementError, match="symbolic links"):
            load_project_improvement(root)
