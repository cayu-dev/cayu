"""Durable run accounting composes independently of live runtime owners."""

import os
import pickle
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import get_type_hints

import cayu
from cayu.budgets import _run_limit_accounting as accounting
from cayu.budgets import run_limits
from cayu.budgets.usage import SessionUsageSummary


def test_run_accounting_composes_without_clock_execution_or_store_owners():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from datetime import UTC, datetime, timedelta

blocked = (
    "cayu.runtime.stop_policy", "cayu.runtime._run_limit_accounting",
    "cayu.runtime._run_limits", "cayu.runtime._session_engine",
    "cayu.runtime._recovery_coordinator", "cayu.runtime._runtime_records",
    "cayu.sessions.base", "cayu.storage", "cayu.budgets.base",
)
class BlockOwners:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockOwners())

from cayu.budgets.run_limits import RunLimits
assert "cayu.budgets.base" not in sys.modules
blocked = tuple(name for name in blocked if name != "cayu.budgets.base")
from cayu.budgets import _run_limit_accounting as accounting
from cayu.budgets.usage import SessionUsageSummary

limits = RunLimits(max_elapsed_seconds=60)
assert accounting.has_run_limit_accounting_authority(limits, ())
origin = datetime(2026, 1, 1, tzinfo=UTC)
context = accounting.RunLimitAccountingContext(
    started_at=origin, baseline=SessionUsageSummary(session_id="session", tool_calls=2),
)
paused = accounting.pause_run_limit_accounting_context(context, now=origin + timedelta(seconds=7))
saved = accounting.RunLimitAccountingContext.model_validate_json(paused.model_dump_json())
resumed = accounting.resume_run_limit_accounting_context(
    saved, resolved_at=origin + timedelta(days=3, seconds=7),
)
assert resumed.started_at == origin + timedelta(days=3)
assert resumed.baseline.tool_calls == 2 and resumed.pause_started_at is None
rebased = accounting.rebase_run_limit_accounting_context(
    resumed, session_id="session", limits=limits, budget_limits=(), events=[],
    reset_run_limits=True, reset_budgets=False, now=origin + timedelta(days=4),
)
assert rebased.started_at == origin + timedelta(days=4)
assert rebased.baseline.tool_calls == 0
assert accounting.run_budget_authorities_from_context(rebased, budget_limits=()) == {}
assert not any(name in sys.modules for name in blocked)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_run_limit_public_imports_and_historical_class_pickle_keep_identity():
    import cayu.runtime
    from cayu.runtime import stop_policy

    for name in ("RunLimits", "copy_run_limits", "has_run_limits"):
        canonical = getattr(run_limits, name)
        assert canonical.__module__ == run_limits.__name__
        assert getattr(cayu, name) is getattr(cayu.runtime, name) is canonical
        assert getattr(stop_policy, name) is canonical
        get_type_hints(canonical)
    # Protocol 0 GLOBAL from the original public owner must still resolve.
    old_class = pickle.loads(b"ccayu.runtime.stop_policy\nRunLimits\n.")
    assert old_class is run_limits.RunLimits
    limits = old_class(max_tool_calls=3)
    assert pickle.loads(pickle.dumps(limits)) == limits


def test_run_accounting_records_and_clock_adapter_share_canonical_types():
    from cayu.runtime import _run_limit_accounting as clock_adapter

    for name in (
        "RunBudgetAccountingAuthority",
        "RunLimitAccountingContext",
        "pause_run_limit_accounting_context",
        "resume_run_limit_accounting_context",
        "has_run_limit_accounting_authority",
        "run_budget_authorities_from_context",
        "rebase_run_limit_accounting_context",
    ):
        canonical = getattr(accounting, name)
        assert canonical.__module__ == accounting.__name__
        assert not hasattr(clock_adapter, name)
        get_type_hints(canonical)
    context = accounting.RunLimitAccountingContext(
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        baseline=SessionUsageSummary(session_id="session"),
    )
    assert pickle.loads(pickle.dumps(context)) == context
    assert get_type_hints(clock_adapter.capture_run_limit_accounting_context)["return"] is type(
        context
    )
    assert get_type_hints(clock_adapter.restore_run_limit_accounting_context)["context"] is type(
        context
    )
