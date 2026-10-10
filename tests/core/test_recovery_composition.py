"""Recovery ownership stays composable across the execution boundary."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

import cayu
from cayu import AgentSpec, CayuApp, Event, EventType, Message, RunRequest, ScriptedModelProvider
from cayu.budgets.base import BudgetReservationResult
from cayu.budgets.usage import SessionUsageSummary
from cayu.execution_units import new_model_step_identity
from cayu.observability.hooks import RuntimeHookPhase
from cayu.runtime._model_step_executor import (
    ModelStepBudgetEvaluationRequest,
    ModelStepBudgetReservationFailureRequest,
    ModelStepLimitEvaluationRequest,
)
from cayu.runtime._recovery_requests import (
    RecoveryInterruptionRequest,
    RecoveryLimitStopRequest,
    RecoveryTerminalEventRequest,
)
from cayu.runtime._run_limits import BudgetEvaluation, LimitEvaluation
from cayu.runtime._tool_round_executor import ToolRoundLimitRequest
from cayu.runtime.stop_policy import StopDecision, StopLimit
from cayu.sessions.base import (
    _current_session_invocation_terminal_event,
    _mark_session_invocation_terminal_event,
)
from cayu.sessions.records import SessionIdentity


def _run_without_owners(blocked: tuple[str, ...], operation: str) -> None:
    script = (
        """
import importlib.abc
import sys

blocked = set(sys.argv[1:])
class RejectOwners(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Independent component imported {fullname}")
sys.meta_path.insert(0, RejectOwners())
"""
        + operation
        + """
assert not blocked.intersection(sys.modules)
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", script, *blocked],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_recovery_parts_work_without_application_or_execution():
    _run_without_owners(
        (
            "cayu.applications",
            "cayu.runtime._session_engine",
            "cayu.runtime._session_recovery",
            "cayu.runtime._recovery_coordinator",
            "cayu.runtime._incomplete_session_recovery",
        ),
        """
import asyncio
from cayu.runtime._pending_tool_round_recovery import PendingToolRoundRecovery
from cayu.runtime._workspace_observation_recovery import WorkspaceObservationRecovery
from cayu.runtime._recovery_ownership import RecoveryOwnership
from cayu.runtime._terminal_evidence_reader import TerminalEvidenceReader
from cayu.runtime._terminal_evidence_finalization import TerminalEvidenceFinalization
from cayu.runtime._terminal_event_publication import TerminalEventPublication
from cayu.runtime._session_finalization import SessionFinalization
from cayu.runtime._recovery_admission import RecoveryAdmission
from cayu.runtime._work_attempt_coordinator import WorkAttemptCoordinator
from cayu.runtime._durable_tool_round import DeferredInteractionInput
from cayu.sessions.base import InMemorySessionStore, RunRequest
from cayu.sessions.records import SessionIdentity
from cayu.messages import Message
from cayu.vaults.redaction import SecretRedactor

async def scenario():
    store = InMemorySessionStore()
    session = await store.create(
        RunRequest(agent_name="independent", messages=[Message.text("user", "hello")]),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    deferred = DeferredInteractionInput(store, SecretRedactor())
    before = await store.load_checkpoint(session.id)
    assert not await deferred.materialize_if_present(session.id)
    assert await store.load_checkpoint(session.id) == before
asyncio.run(scenario())
""",
    )


def test_execution_imports_without_continuation_recovery():
    _run_without_owners(
        (
            "cayu.applications",
            "cayu.runtime._session_recovery",
            "cayu.runtime._recovery_coordinator",
            "cayu.runtime._incomplete_session_recovery",
            "cayu.runtime._provider_disposition_recovery",
        ),
        """
from cayu.runtime._session_engine import SessionEngine
from cayu.runtime._work_attempt_coordinator import WorkAttemptEngine
assert callable(SessionEngine.prepare_resume)
assert callable(SessionEngine.resume)
assert callable(SessionEngine.continue_run)
""",
    )


@pytest.mark.parametrize(
    "operation",
    [
        "interruption",
        "limit",
        "terminal",
        "model_budget",
        "model_limit",
        "model_reservation",
        "tool_limit",
    ],
)
@pytest.mark.parametrize("close_early", [False, True])
def test_recovery_publication_preserves_caller_terminal_authority(
    operation, close_early, monkeypatch
):
    async def scenario():
        app = CayuApp(enable_logging=False)
        app.register_provider(ScriptedModelProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
        session = await app.session_store.create(
            RunRequest(agent_name="assistant", messages=[Message.text("user", "hello")]),
            identity=SessionIdentity(provider_name="scripted", model="scripted-model"),
        )
        caller_terminal = Event(type=EventType.SESSION_COMPLETED, session_id=session.id)
        recovered_terminal = Event(type=EventType.SESSION_INTERRUPTED, session_id=session.id)
        model_attempt = new_model_step_identity().new_attempt()
        _mark_session_invocation_terminal_event(caller_terminal)
        closed = False

        async def publish(**_kwargs):
            nonlocal closed
            _mark_session_invocation_terminal_event(recovered_terminal)
            try:
                yield recovered_terminal
            finally:
                assert _current_session_invocation_terminal_event(session.id) == recovered_terminal
                closed = True

        runtime = {
            "session": session,
            "registered_agent": app._get_registered_agent("assistant"),
            "registered_environment": None,
        }
        if operation == "interruption":
            monkeypatch.setattr(app._session_finalization, "handle_session_interrupted", publish)
            stream = app._session_finalization.interrupt_recovery(
                RecoveryInterruptionRequest(**runtime, environment_name=None)
            )
        elif operation == "limit":
            monkeypatch.setattr(
                app._session_finalization, "stop_session_for_limit_reached", publish
            )
            stream = app._session_finalization.stop_recovered_session_for_limit(
                RecoveryLimitStopRequest(
                    **runtime,
                    environment_name=None,
                    decision=StopDecision(
                        limit=StopLimit.TOOL_CALLS,
                        maximum=1,
                        actual=2,
                        message="Tool limit reached.",
                    ),
                    usage_summary=SessionUsageSummary(session_id=session.id),
                    cost_summary=None,
                    messages=[],
                    tool_calls=[],
                    completed_tool_outcomes=[],
                    pending_approval_to_clear=None,
                    deferred_messages=[],
                    requested_approval_decision=None,
                    approval_resolution_request_digest=None,
                )
            )
        elif operation == "terminal":
            monkeypatch.setattr(app._terminal_event_publication, "emit", publish)
            stream = app._terminal_event_publication.publish_recovered(
                RecoveryTerminalEventRequest(
                    **runtime,
                    event=recovered_terminal,
                    phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                )
            )
        else:
            execution = {
                **runtime,
                "environment_name": None,
                "messages": [],
                "run_started_at": 0.0,
                "turn_usage_tracker": None,
                "active_run": None,
                "execution_profile": None,
            }
            finalization = app._session_finalization
            if operation == "model_budget":
                monkeypatch.setattr(finalization, "apply_budget_evaluation", publish)
                stream = finalization.apply_model_step_budget_evaluation(
                    ModelStepBudgetEvaluationRequest(
                        **execution, evaluation=BudgetEvaluation(check=None)
                    )
                )
            elif operation == "model_reservation":
                monkeypatch.setattr(
                    finalization, "_stop_session_for_budget_reservation_failed", publish
                )
                stream = finalization.stop_for_model_step_budget_reservation_failure(
                    ModelStepBudgetReservationFailureRequest(
                        **execution,
                        result=BudgetReservationResult(
                            accepted=False,
                            budget_limit_id="blim_" + "a" * 64,
                            model_step_id=model_attempt.model_step_id,
                            model_attempt_id=model_attempt.model_attempt_id,
                            scope="session",
                            key=session.id,
                            currency="USD",
                            maximum=1,
                            action="interrupt",
                            requested=2,
                            actual=0,
                            message="Budget reservation denied.",
                        ),
                    )
                )
            else:
                monkeypatch.setattr(finalization, "apply_limit_evaluation", publish)
                evaluation = LimitEvaluation(
                    decision=None,
                    usage_summary=SessionUsageSummary(session_id=session.id),
                    cost_summary=None,
                )
                if operation == "model_limit":
                    stream = finalization.apply_model_step_limit_evaluation(
                        ModelStepLimitEvaluationRequest(**execution, evaluation=evaluation)
                    )
                else:
                    stream = finalization.apply_tool_round_limit(
                        ToolRoundLimitRequest(
                            **execution,
                            evaluation=evaluation,
                            tool_calls=[],
                            completed_tool_outcomes=[],
                            tool_round_identity=model_attempt.new_tool_round(),
                        )
                    )
        try:
            assert await anext(stream) == recovered_terminal
            assert _current_session_invocation_terminal_event(session.id) == caller_terminal
            if not close_early:
                with pytest.raises(StopAsyncIteration):
                    await anext(stream)
        finally:
            await stream.aclose()
        assert closed
        assert _current_session_invocation_terminal_event(session.id) == caller_terminal

    asyncio.run(scenario())
