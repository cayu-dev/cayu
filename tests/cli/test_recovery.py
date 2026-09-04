from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace

import cayu.cli.recovery as recovery_cli
from cayu.cli import main
from cayu.runtime import (
    RecoveryPlan,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
    RecoveryReceipt,
)


def _empty_plan(request: RecoveryPlanRequest | None = None) -> RecoveryPlan:
    return RecoveryPlan(
        plan_id="recovery-plan:sha256:" + "a" * 64,
        created_at=datetime(2026, 9, 3, tzinfo=UTC),
        request=request
        or RecoveryPlanRequest(selection=RecoveryPlanSelection(session_ids=("session-one",))),
        inspected_session_count=0,
    )


def test_recovery_plan_cli_builds_registered_app_once(monkeypatch, tmp_path, capsys) -> None:
    build_calls: list[str] = []

    class App:
        async def plan_recovery(self, request: RecoveryPlanRequest) -> RecoveryPlan:
            assert request.selection.session_ids == ("session-one", "session-two")
            return _empty_plan(request)

    monkeypatch.setattr(
        recovery_cli,
        "resolve_project",
        lambda target, **_kwargs: SimpleNamespace(root=tmp_path, target=target),
    )
    monkeypatch.setattr(recovery_cli, "project_context", lambda _root: nullcontext())

    def build(target: str, **_kwargs):
        build_calls.append(target)
        return App()

    monkeypatch.setattr(recovery_cli, "build_project_app", build)

    result = main(
        [
            "recovery",
            "plan",
            "project:build_app",
            "--session",
            "session-one",
            "--session",
            "session-two",
        ]
    )

    assert result == 0
    assert build_calls == ["project:build_app"]
    assert json.loads(capsys.readouterr().out)["record_type"] == "cayu.recovery-plan"


def test_recovery_execute_cli_loads_exact_plan_and_decisions(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    plan = _empty_plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text("[]", encoding="utf-8")
    build_calls: list[str] = []

    class App:
        async def execute_recovery(self, request):
            assert request.plan == plan
            assert request.execution_id == "operator-run-one"
            assert request.max_concurrency == 4
            return RecoveryReceipt(
                plan_id=plan.plan_id,
                execution_id=request.execution_id,
                items=(),
            )

    monkeypatch.setattr(
        recovery_cli,
        "resolve_project",
        lambda target, **_kwargs: SimpleNamespace(root=tmp_path, target=target),
    )
    monkeypatch.setattr(recovery_cli, "project_context", lambda _root: nullcontext())

    def build(target: str, **_kwargs):
        build_calls.append(target)
        return App()

    monkeypatch.setattr(recovery_cli, "build_project_app", build)

    result = main(
        [
            "recovery",
            "execute",
            str(plan_path),
            "--target",
            "project:build_app",
            "--execution-id",
            "operator-run-one",
            "--decisions",
            str(decisions_path),
            "--max-concurrency",
            "4",
        ]
    )

    assert result == 0
    assert build_calls == ["project:build_app"]
    assert json.loads(capsys.readouterr().out) == {
        "record_type": "cayu.recovery-receipt",
        "schema_version": 1,
        "plan_id": plan.plan_id,
        "execution_id": "operator-run-one",
        "items": [],
    }
