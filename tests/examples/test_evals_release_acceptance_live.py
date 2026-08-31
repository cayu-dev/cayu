from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import cayu

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _REPOSITORY_ROOT / "examples" / "evals_release_acceptance_live.py"


def _load_example(monkeypatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
    spec = importlib.util.spec_from_file_location("evals_release_acceptance_live", _EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_server_uses_exact_loaded_cayu_package(monkeypatch, tmp_path: Path) -> None:
    live = _load_example(monkeypatch)
    project_root = tmp_path / "generated-project"
    project_root.mkdir()

    pythonpath = live._generated_server_pythonpath(project_root)

    assert cayu.__file__ is not None
    assert pythonpath.split(os.pathsep) == [
        str(Path(cayu.__file__).resolve().parent.parent),
        str(project_root.resolve()),
    ]


def test_browser_abort_filter_requires_a_proven_safe_boundary(monkeypatch) -> None:
    live = _load_example(monkeypatch)

    assert live._is_expected_browser_abort(
        method="GET",
        path="/api/evals/runs",
        failure="net::ERR_ABORTED",
        mutation_id=None,
    )
    assert live._is_deferred_source_run_abort(
        method="POST",
        path="/api/run",
        failure="net::ERR_ABORTED",
    )
    assert not live._is_deferred_source_run_abort(
        method="POST",
        path="/api/run",
        failure="net::ERR_FAILED",
    )
    assert not live._is_deferred_source_run_abort(
        method="POST",
        path="/api/evals/runs",
        failure="net::ERR_ABORTED",
    )


def test_browser_failure_gate_accepts_only_one_terminal_source_observer_abort(monkeypatch) -> None:
    live = _load_example(monkeypatch)
    failures = {"console": [], "page": [], "request": [], "api": []}

    live._require_no_browser_failures(
        failures,
        deferred_source_run_aborts=["POST /api/run: net::ERR_ABORTED"],
        source_session_terminal=True,
    )

    try:
        live._require_no_browser_failures(
            failures,
            deferred_source_run_aborts=["POST /api/run: net::ERR_ABORTED"],
            source_session_terminal=False,
        )
    except RuntimeError as error:
        assert "without a proven terminal" in str(error)
    else:
        raise AssertionError("a pre-terminal source observer abort must remain fatal")

    try:
        live._require_no_browser_failures(
            failures,
            deferred_source_run_aborts=["first", "second"],
            source_session_terminal=True,
        )
    except RuntimeError as error:
        assert "more than one" in str(error)
    else:
        raise AssertionError("multiple source observer aborts must remain fatal")
