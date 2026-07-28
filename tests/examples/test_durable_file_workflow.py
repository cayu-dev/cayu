from __future__ import annotations

import asyncio

from examples.durable_file_workflow.demo import build_app, run_demo

from cayu import TaskStatus, check_manifest


def test_demo_recovers_then_verifies_completion_and_blocks_missing_input(
    tmp_path,
) -> None:
    result = asyncio.run(run_demo(tmp_path))

    assert result.completed.status is TaskStatus.COMPLETED
    assert result.completed.result == {
        "artifact": "result.txt",
        "content": "CAYU\n",
        "verified": True,
    }
    assert result.second_completed.status is TaskStatus.COMPLETED
    assert result.second_completed.result == {
        "artifact": "result.txt",
        "content": "RUNTIME\n",
        "verified": True,
    }
    assert result.blocked.status is TaskStatus.BLOCKED
    assert result.blocked.status_reason == "Required source text is missing."
    assert result.blocked.status_payload == {"missing": "source_text"}
    assert result.provider_call_count == 6
    assert result.session_roots == (
        "session-transform-second-source",
        "session-transform-source",
    )
    assert result.completed.result["content"] != result.second_completed.result["content"]


def test_example_goal_prompt_is_outcome_oriented() -> None:
    from examples.durable_file_workflow.demo import GOAL_PROMPT

    assert "Goal:" in GOAL_PROMPT
    assert "Output contract:" in GOAL_PROMPT
    assert "Constraints:" in GOAL_PROMPT
    assert "1." not in GOAL_PROMPT


def test_example_declares_a_structurally_complete_workflow(tmp_path) -> None:
    app, _store, factory, _provider = build_app(tmp_path)

    report = check_manifest(app.describe())

    assert factory.roots == {}
    assert not {item.code for item in report.diagnostics if item.code.startswith("AGENT_WORKFLOW_")}
