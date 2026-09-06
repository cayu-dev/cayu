from __future__ import annotations

import asyncio
import sys

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EvalCase,
    EvalSuite,
    FinalOutputContains,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    load_eval_run,
    load_trajectory,
    render_html_report,
    run_eval_suite,
    write_eval_run_json,
    write_trajectory_json,
)
from cayu.core.events import Event, EventType
from cayu.evals.operation_outcomes import (
    HttpOperationOutcomeV1,
    OperationOutcomeSummary,
    summarize_operation_outcomes,
    trajectory_operation_outcomes,
)
from cayu.runners import LocalRunner
from cayu.tools.commands import ExecCommandTool


def _event(kind=EventType.TOOL_CALL_COMPLETED, *, call="call", **payload):
    return Event(
        type=kind,
        session_id="session",
        tool_name="exec_command",
        payload={"tool_call_id": call, "tool_round_id": "round", **payload},
    )


def test_runner_correlation_replay_retries_and_missing_evidence():
    terminal = _event(
        result={"structured": {"exit_code": 1, "timed_out": False, "cancelled": False}}
    )
    first = _event(
        EventType.RUNNER_EXEC_COMPLETED,
        execution_id="first",
        exit_code=1,
        timed_out=False,
        cancelled=False,
    )
    second = _event(
        EventType.RUNNER_EXEC_COMPLETED,
        execution_id="retry",
        exit_code=0,
        timed_out=False,
        cancelled=False,
    )
    pending = _event(EventType.RUNNER_EXEC_STARTED, execution_id="pending")
    events = [terminal, first, second, pending, first.model_copy(), terminal.model_copy()]
    summary = summarize_operation_outcomes(events)
    assert summary.evidence_state == "incomplete"
    assert summary.counts.invocation_completed == 1
    assert summary.counts.command_nonzero_exit == 1
    assert summary.counts.command_zero_exit == 1
    assert summary.counts.command_not_reported == 1
    assert summary.counts.http_not_reported == 1
    assert {row.execution_id for row in summary.evidence if row.dimension == "command"} == {
        "first",
        "retry",
        "pending",
    }
    assert OperationOutcomeSummary.model_validate_json(summary.model_dump_json()) == summary


def test_typed_http_and_expected_probe_do_not_change_invocation_health():
    events = [
        _event(
            call="probe",
            result={"structured": {"exit_code": 1, "timed_out": False, "cancelled": False}},
        ),
        _event(call="print", result={"structured": {"exit_code": 0, "stdout": "HTTP 403"}}),
        _event(
            call="http",
            result={
                "structured": {
                    "operation_outcome": HttpOperationOutcomeV1(status_code=403).model_dump()
                }
            },
        ),
        _event(call="builtin", result={"structured": {"error": "http_status", "status_code": 503}}),
        _event(
            call="malformed",
            result={
                "structured": {"operation_outcome": {"protocol": "http", "status_code": "403"}}
            },
        ),
        _event(
            call="cancelled",
            interrupted=True,
            result={"structured": {"exit_code": -9, "timed_out": False, "cancelled": True}},
        ),
        _event(
            call="timeout",
            result={"structured": {"exit_code": -9, "timed_out": True, "cancelled": False}},
        ),
    ]
    summary = summarize_operation_outcomes(events, evidence_complete=True)
    c = summary.counts
    assert c.invocation_completed == 6
    assert c.invocation_cancelled == 1
    assert c.invocation_failed == 0
    assert c.command_nonzero_exit == 1
    assert c.command_timed_out == c.command_cancelled == 1
    assert c.http_error == 2
    assert c.http_not_reported == 5
    assert c.command_not_reported == 3


def test_bounded_references_keep_full_counts_and_unknown_outcomes():
    events = [_event(call=str(i)) for i in range(300)]
    summary = summarize_operation_outcomes(events, evidence_complete=True)
    assert len(summary.evidence) == 256
    assert summary.omitted_evidence_count == 900 - 256
    assert summary.counts.command_not_reported == 300
    assert summary.counts.command_zero_exit == 0


def test_public_eval_local_commands_persist_export_and_report(tmp_path):
    scripts = [
        "pass",
        "raise SystemExit(1)",
        "import sys; sys.exit(1)",
        "print('HTTP 403')",
        "import time; time.sleep(10)",
    ]
    responses = []
    for index, script in enumerate(scripts):
        responses.append(
            [
                ModelStreamEvent.tool_call(
                    id=f"call-{index}",
                    name="exec_command",
                    arguments={
                        "kind": "process",
                        "argv": [sys.executable, "-c", script],
                        "timeout_s": 1,
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        )
    responses.append(
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({"finish_reason": "stop"})]
    )
    app = CayuApp(
        enable_logging=False, session_store=SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    )
    app.register_provider(ScriptedModelProvider(responses), default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), runner=LocalRunner(tmp_path)), default=True
    )
    app.register_agent(AgentSpec(name="agent", model="scripted-model"), tools=[ExecCommandTool()])
    suite = EvalSuite(
        id="outcomes",
        cases=(
            EvalCase(
                id="commands",
                request=RunRequest(agent_name="agent", messages=[Message.text("user", "run")]),
                assertions=(FinalOutputContains("done"),),
            ),
        ),
    )
    run = asyncio.run(run_eval_suite(app, suite, retain_trajectory=True))
    trial = run.cases[0].trials[0]
    assert trial.score == 1.0
    summary = trial.operation_outcomes
    assert summary is not None
    assert summary.counts.command_zero_exit == 2
    assert summary.counts.command_nonzero_exit == 2
    assert summary.counts.command_timed_out == 1
    assert summary.counts.http_error == 0
    assert summary.counts.invocation_completed + summary.counts.invocation_failed == 5
    assert summary.counts.command_not_reported == 0
    commands = [row for row in summary.evidence if row.dimension == "command"]
    assert len(commands) == 5
    assert all(row.execution_id and row.cancelled is not None for row in commands)
    assert trial.trajectory is not None
    event_ids = {event.id for event in trial.trajectory.events}
    assert all(row.event_id in event_ids for row in summary.evidence)
    write_eval_run_json(run, tmp_path / "run.json")
    assert load_eval_run(tmp_path / "run.json").cases[0].trials[0].operation_outcomes == summary
    write_trajectory_json(trial.trajectory, tmp_path / "trajectory.json")
    assert trajectory_operation_outcomes(load_trajectory(tmp_path / "trajectory.json")) == summary
    assert "2 nonzero exit" in render_html_report(run)


def test_legacy_missing_completion_and_malformed_linkage_remain_unknown():
    started = _event(EventType.RUNNER_EXEC_STARTED)
    completed = _event(result={"structured": {"exit_code": 0}})
    summary = summarize_operation_outcomes([started, completed], evidence_complete=True)
    assert summary.evidence_state == "incomplete"
    assert summary.counts.command_not_reported == 1
    assert summary.counts.command_zero_exit == 0
    malformed = _event(result={"structured": {"exit_code": False}}, tool_round_id={})
    assert summarize_operation_outcomes([malformed]).counts.command_not_reported == 1


def test_builtin_access_evidence_is_projected_without_page_content():
    from cayu.tools.web_access import WebAccessEvidenceSource, classify_http_access

    evidence = classify_http_access(
        "https://example.com",
        status_code=403,
        headers={},
        source=WebAccessEvidenceSource.HTTP_RESPONSE,
    )
    assert evidence is not None
    event = _event(result={"structured": {"access": evidence.model_dump(mode="json")}})
    summary = summarize_operation_outcomes([event])
    assert summary.counts.http_error == 1
    assert "example.com" not in summary.model_dump_json()
