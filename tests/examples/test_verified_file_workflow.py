from __future__ import annotations

import asyncio
import stat
import threading

import examples.durable_file_workflow.verified as verified
import pytest
from examples.durable_file_workflow.verified import (
    ARTIFACT,
    ARTIFACT_UNAVAILABLE,
    CONTENT_MISMATCH,
    CONTINUATION_TYPE,
    EXTRA_NEWLINE,
    FIRST_PROGRAM,
    INPUT_MODIFIED,
    INPUT_UNAVAILABLE,
    MISSING_NEWLINE,
    RESULT_MISMATCH,
    SOURCE_TEXT,
    SOURCE_UNAVAILABLE,
    ExactArtifactVerifier,
    ExpectedSource,
    GapAwareProvider,
    SessionWorkspaces,
    TaskInputSource,
    _evidence,
    artifact_result_reference,
    build_contract_draft,
    durable_evidence_preserved,
    judge_artifact,
    run_demo,
)

from cayu import (
    CompletionResultReference,
    CompletionVerdict,
    CriterionOutcomeStatus,
    ExecutionProfileBehaviorIdentity,
    InMemoryTaskStore,
    RunLimits,
    SQLiteTaskStore,
    TaskStatus,
    VerifiedTaskWorker,
    VerifiedTaskWorkerDraining,
    WorkCompletionConflict,
    completion_result_sha256,
)
from cayu.tasks.contracts import (
    CompletionDecisionCreate,
    validate_completion_decision_contract,
    work_contract_from_draft,
)


@pytest.mark.parametrize("store", ["memory", "sqlite"])
def test_rejected_attempt_continues_and_is_accepted_after_lost_acknowledgement(
    tmp_path, store: str
) -> None:
    result = asyncio.run(run_demo(tmp_path, store=store))

    expected_content = SOURCE_TEXT.upper() + "\n"
    assert result.task.status is TaskStatus.COMPLETED
    assert result.task.result is not None
    assert result.task.result["artifact"] == ARTIFACT
    assert result.task.result["content"] == expected_content

    # One rejected attempt with a cited gap, then one accepted continuation.
    first, second = result.attempts
    assert (first.ordinal, first.kind, first.verdict) == (1, "initial", "rejected")
    assert first.gap_codes == (MISSING_NEWLINE,)
    assert (second.ordinal, second.kind, second.verdict) == (2, "continuation", "accepted")
    assert second.gap_codes == ()

    # The next attempt received exactly the prior decision and its gaps.
    assert first.decision is not None
    (continuation,) = result.received_continuations
    assert continuation["type"] == CONTINUATION_TYPE
    assert continuation["decision_id"] == first.decision.decision_id
    assert continuation["gaps"] == [gap.model_dump(mode="json") for gap in first.decision.gaps]

    # The lost acknowledgement interrupted the worker after attempt 2's proposal
    # committed; recovery reran neither the model nor the external program.
    assert result.interrupted_by is not None
    assert result.provider_calls == result.provider_calls_before_restart == 4
    assert result.effects == result.effects_before_restart == 2
    assert result.preparations == 1
    assert result.proposals == 2
    assert result.verifier_calls == 2
    assert result.resolver_calls == 1
    assert result.resolved_events == 1
    assert result.unsettled_admissions == 0

    # Exact replay: everything durable before the restart is equal after it.
    before_first, before_second = result.attempts_before_restart
    assert before_first.proposal and before_first.decision and before_first.application
    assert before_second.proposal is not None
    assert before_second.decision is None and before_second.application is None
    assert durable_evidence_preserved(result.attempts_before_restart, result.attempts)
    # Asking for the accepted outcome again replays the stored task.
    assert result.replayed_task == result.task

    # Each step is attributable: who ran the attempt, which verifier and worker
    # decided it, and which application receipt moved the task.
    assert first.run_by == second.run_by == "worker-before-restart"
    assert first.decision.worker_id == "worker-before-restart"
    assert second.decision is not None
    assert second.decision.worker_id == "worker-after-restart"
    assert {first.decision.verifier.verifier_id, second.decision.verifier.verifier_id} == {
        "exact-artifact"
    }
    assert first.application is not None and second.application is not None
    assert first.application.decision_id == first.decision.decision_id
    assert first.application.task.status is TaskStatus.RUNNING
    assert second.application.decision_id == second.decision.decision_id
    assert second.application.task == result.task

    # Acceptance retired the contract binding and released the session.
    assert result.binding_retired is True
    assert result.active_contract_task is None


def test_worker_that_rewrites_its_input_cannot_get_accepted(tmp_path) -> None:
    # The program forges a matching pair: a new source.txt and the "right" output for it.
    forging_program = (
        "from pathlib import Path\n"
        "Path('source.txt').write_text('pwned', encoding='utf-8')\n"
        "Path('result.txt').write_text('PWNED\\n', encoding='utf-8')\n"
    )
    assert forging_program != FIRST_PROGRAM
    provider = GapAwareProvider(first_program=forging_program)

    result = asyncio.run(run_demo(tmp_path, lose_acknowledgement=False, provider=provider))

    # The forgery is rejected; the model cannot repair it, so the same gaps repeat
    # and the contract's repeated-gap ceiling stops the task.
    assert result.task.status is TaskStatus.NEEDS_ATTENTION
    assert result.task.status_reason == "work_contract_repeated_gap_limit"
    assert result.replayed_task is None
    assert [attempt.verdict for attempt in result.attempts] == ["rejected", "rejected"]
    assert set(result.attempts[0].gap_codes) == {CONTENT_MISMATCH, INPUT_MODIFIED}
    assert result.resolver_calls == 0


def test_a_pipe_in_place_of_the_artifact_cannot_stall_the_worker(tmp_path) -> None:
    # Opening a FIFO for reading blocks forever; bounded library reads refuse it.
    provider = GapAwareProvider(
        first_program="import os\nos.mkfifo('result.txt')\n",
    )

    # A blocking read would freeze the event loop, so guard from another thread.
    outcome: list[verified.VerifiedDemoResult] = []
    runner = threading.Thread(
        target=lambda: outcome.append(
            asyncio.run(run_demo(tmp_path, lose_acknowledgement=False, provider=provider))
        ),
        daemon=True,
    )
    runner.start()
    runner.join(timeout=300)
    assert not runner.is_alive(), "reading the pipe stalled the worker"
    (result,) = outcome

    assert result.task.status is TaskStatus.NEEDS_ATTENTION
    assert result.task.status_reason == "work_contract_repeated_gap_limit"
    first = result.attempts[0]
    assert first.proposal is not None
    (artifact,) = [
        item for item in first.proposal.evidence_references if item.requirement_id == "artifact"
    ]
    assert artifact.available is False
    assert set(first.gap_codes) == {ARTIFACT_UNAVAILABLE}
    # The artifact really is a pipe, so this is not the missing-file path.
    pipe = SessionWorkspaces(tmp_path / "workspaces").root_for(result.session_id) / ARTIFACT
    assert stat.S_ISFIFO(pipe.lstat().st_mode)


class _WorkspaceSource(ExpectedSource):
    """A different expected-input source; only its declared identity matters here."""

    @property
    def identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:workspace-source", behavior_version="1", implementation_version="1"
        )

    async def load(self, request):
        return None


def test_verifier_profile_names_the_injected_expected_source(tmp_path) -> None:
    workspaces = SessionWorkspaces(tmp_path)
    from_task = ExactArtifactVerifier(workspaces, TaskInputSource(InMemoryTaskStore()))
    from_workspace = ExactArtifactVerifier(workspaces, _WorkspaceSource())

    (task_component,) = from_task.execution_profile_components
    (workspace_component,) = from_workspace.execution_profile_components
    assert task_component.identity == TaskInputSource(InMemoryTaskStore()).identity
    assert task_component.identity != workspace_component.identity


@pytest.mark.parametrize("run_failure", [TimeoutError, asyncio.CancelledError])
@pytest.mark.parametrize("drains_forever", [False, True])
def test_stores_stay_open_until_a_draining_worker_closes(
    tmp_path, monkeypatch, drains_forever: bool, run_failure: type[BaseException]
) -> None:
    # Characterizes the example's handling of the shapes Cayu reports when closing
    # fails: a run error plus draining, or a cancellation carrying draining.
    closed_stores: list[object] = []
    real_close = SQLiteTaskStore.close

    async def record_close(self) -> None:
        closed_stores.append(self)
        await real_close(self)

    monkeypatch.setattr(SQLiteTaskStore, "close", record_close)
    close_attempts: list[int] = []

    class DrainingWorker(VerifiedTaskWorker):
        async def run(self, stop=None, max_tasks=None) -> int:
            raise run_failure()

        async def aclose(self) -> None:
            close_attempts.append(1)
            if drains_forever or len(close_attempts) == 1:
                raise VerifiedTaskWorkerDraining("in-flight work is still settling")
            await super().aclose()

    monkeypatch.setattr(verified, "VerifiedTaskWorker", DrainingWorker)

    cancelled = run_failure is asyncio.CancelledError
    if drains_forever:
        # The stores are left open because the worker still owns in-flight work.
        # A cancellation stays outermost and carries the unfinished drain.
        expected = asyncio.CancelledError if cancelled else VerifiedTaskWorkerDraining
        with pytest.raises(expected) as raised:
            asyncio.run(run_demo(tmp_path, store="sqlite", lose_acknowledgement=False))
        assert verified._still_draining(raised.value)
        # One close from `async with`, then the retries.
        assert len(close_attempts) == 1 + verified.CLOSE_ATTEMPTS
        assert closed_stores == []
    else:
        # The retried close succeeds, so the run's own failure surfaces and the
        # stores are closed normally.
        with pytest.raises(run_failure):
            asyncio.run(run_demo(tmp_path, store="sqlite", lose_acknowledgement=False))
        assert len(close_attempts) == 2
        assert len(closed_stores) == 1


def _validate_against_contract(decision) -> None:
    contract = work_contract_from_draft(build_contract_draft())
    validate_completion_decision_contract(
        contract,
        CompletionDecisionCreate(
            decision_id="decision",
            proposal_id="proposal",
            claim_id="claim",
            worker_id="worker",
            verifier=contract.verifier,
            verifier_profile_fingerprint="0" * 64,
            **decision.model_dump(),
        ),
    )


def _judge(
    artifact: str | None,
    *,
    source: str | None = SOURCE_TEXT,
    expected: str | None = SOURCE_TEXT,
    proposed_artifact: str | None = None,
    proposed_result: str | None = None,
):
    return judge_artifact(
        artifact=artifact,
        workspace_source=source,
        expected_source=expected,
        proposed={
            "artifact": _evidence("artifact", proposed_artifact or artifact),
            "source": _evidence("source", source),
        },
        proposed_result=artifact_result_reference(proposed_result or proposed_artifact or artifact),
    )


_UNAVAILABLE_ARTIFACT = (("content", ARTIFACT_UNAVAILABLE), ("format", ARTIFACT_UNAVAILABLE))


@pytest.mark.parametrize(
    ("decision_args", "verdict", "gaps"),
    [
        ({"artifact": "CAYU\n"}, CompletionVerdict.ACCEPTED, ()),
        ({"artifact": "CAYU"}, CompletionVerdict.REJECTED, (("format", MISSING_NEWLINE),)),
        ({"artifact": "CAYU\n\n"}, CompletionVerdict.REJECTED, (("format", EXTRA_NEWLINE),)),
        ({"artifact": "RUNTIME\n"}, CompletionVerdict.REJECTED, (("content", CONTENT_MISMATCH),)),
        (
            {"artifact": "runtime"},
            CompletionVerdict.REJECTED,
            (("content", CONTENT_MISMATCH), ("format", MISSING_NEWLINE)),
        ),
        ({"artifact": None}, CompletionVerdict.REJECTED, _UNAVAILABLE_ARTIFACT),
        (
            {"artifact": "CAYU\n", "proposed_artifact": "CAYU"},
            CompletionVerdict.REJECTED,
            _UNAVAILABLE_ARTIFACT,
        ),
        (
            {"artifact": "CAYU\n", "proposed_result": "SOMETHING ELSE\n"},
            CompletionVerdict.REJECTED,
            (("content", RESULT_MISMATCH), ("format", RESULT_MISMATCH)),
        ),
        (
            {"artifact": "PWNED\n", "source": "pwned"},
            CompletionVerdict.REJECTED,
            (("content", CONTENT_MISMATCH), ("source-unmodified", INPUT_MODIFIED)),
        ),
        (
            {"artifact": "CAYU\n", "source": None},
            CompletionVerdict.REJECTED,
            (("source-unmodified", SOURCE_UNAVAILABLE),),
        ),
        (
            {"artifact": "CAYU\n", "expected": None},
            CompletionVerdict.BLOCKED,
            (
                ("content", INPUT_UNAVAILABLE),
                ("format", INPUT_UNAVAILABLE),
                ("source-unmodified", INPUT_UNAVAILABLE),
            ),
        ),
    ],
    ids=[
        "correct",
        "missing-newline",
        "extra-newline",
        "wrong-text",
        "wrong-text-and-newline",
        "missing-artifact",
        "artifact-changed-after-proposal",
        "result-does-not-match-artifact",
        "forged-source-and-output",
        "missing-workspace-source",
        "missing-task-input",
    ],
)
def test_verifier_judges_each_outcome_and_satisfies_the_contract(
    decision_args, verdict: CompletionVerdict, gaps
) -> None:
    decision = _judge(**decision_args)

    assert decision.verdict is verdict
    assert tuple((gap.criterion_id or gap.constraint_id, gap.code) for gap in decision.gaps) == gaps
    # Cayu's own check that the decision covers exactly the frozen contract.
    _validate_against_contract(decision)


def test_readable_untouched_source_satisfies_the_constraint_when_output_is_wrong() -> None:
    decision = _judge("RUNTIME\n")

    (constraint,) = decision.constraint_outcomes
    assert constraint.status is CriterionOutcomeStatus.SATISFIED


def test_missing_artifact_is_proposed_as_unavailable_evidence() -> None:
    evidence = _evidence("artifact", None)

    assert evidence.available is False
    assert evidence.unavailable_reason == "artifact.missing"
    assert evidence.digest is None


def test_cancelling_the_run_propagates_and_closes_the_stores(tmp_path, monkeypatch) -> None:
    closed_stores: list[object] = []
    real_close = SQLiteTaskStore.close

    async def record_close(self) -> None:
        closed_stores.append(self)
        await real_close(self)

    monkeypatch.setattr(SQLiteTaskStore, "close", record_close)

    class StallingProvider(GapAwareProvider):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def stream(self, request):
            self.requests.append(request)
            self.started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover - never reached

    async def scenario() -> asyncio.Task:
        provider = StallingProvider()
        task = asyncio.create_task(
            run_demo(tmp_path, store="sqlite", lose_acknowledgement=False, provider=provider)
        )
        await asyncio.wait_for(provider.started.wait(), 120)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return task

    task = asyncio.run(scenario())

    # Delivered through Task.cancel(): the run reports cancellation, not a
    # converted error, and the stores close. This run has no owned work left in
    # flight; if a verifier ignored cancellation, Cayu's aclose() would currently
    # report only the cancellation, so the stores would close while it ran.
    assert task.cancelled()
    assert len(closed_stores) == 1


@pytest.mark.parametrize(
    ("scope", "status", "reason"),
    [
        ("run", TaskStatus.COMPLETED, None),
        ("session", TaskStatus.NEEDS_ATTENTION, "work_contract_execution_interrupted"),
    ],
)
def test_run_limits_reset_per_attempt_unless_session_scoped(
    tmp_path, scope, status: TaskStatus, reason: str | None
) -> None:
    # Each attempt makes three tool calls; the two attempts make six in total.
    limits = RunLimits(max_tool_calls=4, scope=scope)

    result = asyncio.run(run_demo(tmp_path, lose_acknowledgement=False, limits=limits))

    assert result.task.status is status
    assert result.task.status_reason == reason


def test_changing_the_expected_source_on_restart_fails_closed(tmp_path, monkeypatch) -> None:
    class RenamedSource(TaskInputSource):
        @property
        def identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:renamed-task-input-source",
                behavior_version="1",
                implementation_version="1",
            )

    real_build = verified.build_app
    built: list[object] = []

    def build_with_new_source_after_restart(sessions, tasks, parts):
        if built:
            monkeypatch.setattr(verified, "TaskInputSource", RenamedSource)
        built.append(tasks)
        return real_build(sessions, tasks, parts)

    monkeypatch.setattr(verified, "build_app", build_with_new_source_after_restart)

    # The restarted verifier declares a different expected-input source, so Cayu
    # refuses to verify the pending proposal with it.
    with pytest.raises(WorkCompletionConflict, match="explicit authorized adoption"):
        asyncio.run(run_demo(tmp_path))


def test_a_proposal_claiming_a_different_result_is_rejected(tmp_path, monkeypatch) -> None:
    class ForgingHandler(verified.FileArtifactHandler):
        async def propose(self, context):
            report = await super().propose(context)
            forged = CompletionResultReference(
                kind="workspace.file",
                reference_id=ARTIFACT,
                digest=completion_result_sha256({"forged": True}),
            )
            return report.model_copy(
                update={"proposal": report.proposal.model_copy(update={"result": forged})}
            )

    monkeypatch.setattr(verified, "FileArtifactHandler", ForgingHandler)

    result = asyncio.run(run_demo(tmp_path, lose_acknowledgement=False))

    # Rejected before resolution, rather than accepted and then stuck at the resolver.
    assert result.task.status is TaskStatus.NEEDS_ATTENTION
    assert result.task.status_reason == "work_contract_repeated_gap_limit"
    assert set(result.attempts[0].gap_codes) == {RESULT_MISMATCH}
    assert result.resolver_calls == 0
    assert result.unsettled_admissions == 0
