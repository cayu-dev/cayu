from __future__ import annotations

import asyncio
import os

import pytest

from cayu import EnvironmentFactoryRequest, EvalStatus, run_eval_plan

_CASE_IDS = [
    "tool_roundtrip",
    "workspace_roundtrip",
    "context_observability",
    "knowledge_tool_roundtrip",
    "subagent_roundtrip",
    "usage_accounting",
    "budget_interrupt",
]
_LIVE_CREDENTIAL_ENV = {
    "ANTHROPIC_API_KEY",
    "E2B_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
}
_WORKSPACE_FILE = "runtime-acceptance/workspace-roundtrip.txt"


def test_internal_runtime_acceptance_plan_is_hermetic_and_isolated(monkeypatch) -> None:
    from cayu.evals.internal.runtime_acceptance import build

    environ_type = type(os.environ)
    original_getitem = environ_type.__getitem__
    original_contains = environ_type.__contains__

    def guarded_getitem(environ, key):
        if key in _LIVE_CREDENTIAL_ENV:
            raise AssertionError(f"internal eval read live credential {key}")
        return original_getitem(environ, key)

    def guarded_contains(environ, key):
        if key in _LIVE_CREDENTIAL_ENV:
            raise AssertionError(f"internal eval inspected live credential {key}")
        return original_contains(environ, key)

    monkeypatch.setattr(environ_type, "__getitem__", guarded_getitem)
    monkeypatch.setattr(environ_type, "__contains__", guarded_contains)

    async def run():
        plan = await build()
        result = await run_eval_plan(
            plan,
            case_timeout_seconds=5,
            retain_trajectory=True,
        )
        cases_by_id = {case.case_id: case for case in result.cases}
        workspace_session_id = cases_by_id["workspace_roundtrip"].trial_session_ids[0]
        tool_session_id = cases_by_id["tool_roundtrip"].trial_session_ids[0]

        factory = plan.app.get_environment_factory()
        workspace_environment = await factory.create(
            EnvironmentFactoryRequest(
                session_id=workspace_session_id,
                agent_name="runtime_acceptance_workspace",
                environment_name="runtime-acceptance-local",
            )
        )
        other_environment = await factory.create(
            EnvironmentFactoryRequest(
                session_id=tool_session_id,
                agent_name="runtime_acceptance_tool",
                environment_name="runtime-acceptance-local",
            )
        )
        return plan, result, workspace_environment, other_environment

    plan, result, workspace_environment, other_environment = asyncio.run(run())

    assert plan.suite.id == "cayu-internal-runtime-acceptance-v1"
    assert [case.id for case in plan.suite.cases] == _CASE_IDS
    assert result.status is EvalStatus.PASSED
    assert [case.case_id for case in result.cases] == _CASE_IDS
    assert all(case.status is EvalStatus.PASSED for case in result.cases)

    assert workspace_environment.environment.workspace is not None
    assert other_environment.environment.workspace is not None
    written = asyncio.run(workspace_environment.environment.workspace.read_bytes(_WORKSPACE_FILE))
    assert written.content == b"isolated workspace content"
    with pytest.raises(FileNotFoundError, match="Workspace file not found"):
        asyncio.run(other_environment.environment.workspace.read_bytes(_WORKSPACE_FILE))
