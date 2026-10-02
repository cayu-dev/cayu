from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace

import cayu.cli.recovery as recovery_cli
from cayu.cli import main
from cayu.sessions.recovery import (
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

    closed: list[float] = []

    class App:
        async def aclose(self, *, timeout_s: float):
            closed.append(timeout_s)
            return SimpleNamespace(settled=True)

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
    assert len(closed) == 1
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

    closed: list[float] = []

    class App:
        async def aclose(self, *, timeout_s: float):
            closed.append(timeout_s)
            return SimpleNamespace(settled=True)

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
    assert len(closed) == 1
    assert json.loads(capsys.readouterr().out) == {
        "record_type": "cayu.recovery-receipt",
        "schema_version": 1,
        "plan_id": plan.plan_id,
        "execution_id": "operator-run-one",
        "items": [],
    }


def test_recovery_cli_restores_a_startup_blocked_registration(monkeypatch, tmp_path, capsys):
    import asyncio

    from tests.core.test_startup_recovery_isolation import _app, _seed, _state

    from cayu import InMemorySessionStore, SessionStatus

    store = InMemorySessionStore()
    restored = _app(store, "3")
    replacement = _app(store, "4")

    async def seed():
        await _seed(store, restored, "blocked-cli")
        before = await _state(store, "blocked-cli")
        assert (
            await replacement.resume_pending_interruption_cascades(
                interrupting_inactive_for_seconds=0
            )
            == 0
        )
        assert before == await _state(store, "blocked-cli")

    asyncio.run(seed())
    monkeypatch.setattr(
        recovery_cli,
        "resolve_project",
        lambda *args, **kwargs: SimpleNamespace(root=tmp_path, target="app:build_app"),
    )
    monkeypatch.setattr(recovery_cli, "project_context", lambda _root: nullcontext())
    # Like the real CLI, every command builds (and then closes) its own app.
    monkeypatch.setattr(recovery_cli, "build_project_app", lambda *args, **kwargs: _app(store, "3"))
    plan_path = tmp_path / "recovery-plan.json"
    assert (
        main(
            [
                "recovery",
                "plan",
                "app:build_app",
                "--session",
                "blocked-cli",
                "--inactive-for-seconds",
                "0",
                "--output",
                str(plan_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert "automatic_repair" in json.loads(plan_path.read_text())["items"][0]["allowed_actions"]
    assert (
        main(
            [
                "recovery",
                "execute",
                str(plan_path),
                "--target",
                "app:build_app",
                "--execution-id",
                "restored-cli",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["items"][0]["status"] == "executed"
    assert asyncio.run(store.load("blocked-cli")).status == SessionStatus.INTERRUPTED
