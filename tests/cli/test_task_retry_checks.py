from __future__ import annotations

from pathlib import Path

import pytest

from cayu.cli.scaffold_check import check_declared_scaffold_source

_BASE = """
from cayu.tasks import TaskCreate, TaskQuery, run_task_worker
TIMEOUT = 30
async def worker(app):
    await run_task_worker(app, app.task_store, handler, worker_id="w",
                          query=TaskQuery(type="event"), execution_timeout_s=TIMEOUT)
"""


def _check(root: Path, creation: str, *, worker: str = _BASE):
    (root / "worker.py").write_text(worker)
    (root / "submit.py").write_text(
        "from cayu.tasks import TaskCreate\nasync def submit(app):\n    await app.create_task("
        + creation
        + ")\n"
    )
    return check_declared_scaffold_source(root)


def test_check_warns_about_event_resubmission_without_changing_runtime_policy(tmp_path):
    findings = _check(tmp_path, 'TaskCreate(type="event")')
    assert len(findings) == 1
    finding = findings[0]
    assert finding.code == "TASK_EVENT_RETRY_POLICY_UNDECLARED"
    assert finding.severity.value == "warning"
    assert finding.path == "submit.py:3"
    assert "unlimited()" in finding.hint
    assert "Independent roots get separate budgets" in finding.message
    assert check_declared_scaffold_source(tmp_path, deploy_only=True) == ()


@pytest.mark.parametrize(
    "creation",
    [
        'TaskCreate(type="event", retry_key=event_id)',
        'TaskCreate(type="event", event_id=event_id)',
        'TaskCreate(type="event", input={"event_id": event_id})',
        'TaskCreate(type="event", retry_budget=budget)',
        'TaskCreate(type="event", retry_policy=policy)',
        'TaskCreate(type="different")',
        "TaskCreate(type=dynamic_type)",
        'TaskCreate(type="event", **options)',
    ],
)
def test_check_does_not_guess_dynamic_or_declared_retry_policy(tmp_path, creation):
    assert _check(tmp_path, creation) == ()


@pytest.mark.parametrize("timeout", ["None", "0", "dynamic_timeout"])
def test_unbounded_or_dynamic_workers_do_not_require_an_event_budget(tmp_path, timeout):
    assert (
        _check(
            tmp_path,
            'TaskCreate(type="event")',
            worker=_BASE.replace("execution_timeout_s=TIMEOUT", f"execution_timeout_s={timeout}"),
        )
        == ()
    )


def test_unrelated_functions_with_the_same_name_are_not_cayu_workers(tmp_path):
    assert (
        _check(
            tmp_path,
            'TaskCreate(type="event")',
            worker=_BASE.replace("from cayu.tasks import", "from another_library import"),
        )
        == ()
    )


def test_import_aliases_are_resolved(tmp_path):
    worker = _BASE.replace("run_task_worker\n", "run_task_worker as run\n").replace(
        "await run_task_worker(", "await run("
    )
    assert len(_check(tmp_path, 'TaskCreate(type="event", retry_key=None)', worker=worker)) == 1


def test_shadowed_worker_import_is_not_treated_as_cayu(tmp_path):
    worker = _BASE.replace("async def worker(app):", "async def worker(app, run_task_worker):")
    assert _check(tmp_path, 'TaskCreate(type="event")', worker=worker) == ()


def test_shadowed_timeout_constant_is_not_treated_as_bounded(tmp_path):
    worker = _BASE.replace("async def worker(app):", "async def worker(app, TIMEOUT=None):")
    assert _check(tmp_path, 'TaskCreate(type="event")', worker=worker) == ()


def test_check_json_includes_event_warning_for_loadable_app(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from cayu.cli import main

    _check(tmp_path, 'TaskCreate(type="event")')
    (tmp_path / "retry_check_app.py").write_text(
        "from cayu import CayuApp\ndef build_app():\n    return CayuApp(enable_logging=False)\n"
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("retry_check_app", None)
    try:
        main(["check", "retry_check_app:build_app", "--json"])
        report = json.loads(capsys.readouterr().out)
        codes = [item["code"] for item in report["diagnostics"]]
        assert codes.count("TASK_EVENT_RETRY_POLICY_UNDECLARED") == 1
    finally:
        sys.modules.pop("retry_check_app", None)
