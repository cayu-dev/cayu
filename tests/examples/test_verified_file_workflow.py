from __future__ import annotations

import asyncio
import stat
import sys
import tempfile
import threading
from types import SimpleNamespace

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
    CayuApp,
    CompletionResultReference,
    CompletionVerdict,
    CriterionOutcomeStatus,
    ExecutionProfileBehaviorIdentity,
    InMemoryTaskStore,
    RunLimits,
    SQLiteSessionStore,
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
    closed_stores = _record_store_closes(monkeypatch)
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
    if cancelled:
        # A retry inside the cancelled task would be cancelled again, so the owner
        # finishes the drain from its own task; the cancellation still propagates,
        # carrying the drain it reported.
        with pytest.raises(asyncio.CancelledError) as raised:
            asyncio.run(run_demo(tmp_path, store="sqlite", lose_acknowledgement=False))
        assert verified._still_draining(raised.value)
        if drains_forever:
            # One close from `async with`, then the owner's retries; the worker
            # still owns in-flight work, so the stores stay open.
            assert len(close_attempts) == 1 + verified.CLOSE_ATTEMPTS
            assert closed_stores == []
        else:
            # The owner's first retry settles, then the stores close.
            assert len(close_attempts) == 2
            assert sorted(closed_stores) == ["sessions", "tasks"]
    elif drains_forever:
        # The stores are left open because the worker still owns in-flight work,
        # and the outcome says so beside the draining report.
        with pytest.raises(BaseExceptionGroup) as raised:
            asyncio.run(run_demo(tmp_path, store="sqlite", lose_acknowledgement=False))
        assert verified._still_draining(raised.value)
        assert raised.value.subgroup(
            lambda error: (
                isinstance(error, RuntimeError)
                and "the worker still owns in-flight work" in str(error)
            )
        )
        # One close from `async with`, the run's retries, then the owner's.
        assert len(close_attempts) == 1 + 2 * verified.CLOSE_ATTEMPTS
        assert closed_stores == []
    else:
        # The retried close succeeds, so the run's own failure surfaces and the
        # stores are closed normally.
        with pytest.raises(run_failure):
            asyncio.run(run_demo(tmp_path, store="sqlite", lose_acknowledgement=False))
        assert len(close_attempts) == 2
        assert sorted(closed_stores) == ["sessions", "tasks"]


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
    # converted error, and the stores close because no owned work is left in
    # flight. A verifier still running is covered by the next test.
    assert task.cancelled()
    assert len(closed_stores) == 1


def test_cancelling_a_close_retry_stops_retrying(monkeypatch) -> None:
    close_attempts: list[int] = []

    class DrainingWorker:
        async def aclose(self) -> None:
            # Like the real worker: a cancelled close still owning work raises
            # the cancellation carrying a draining report.
            close_attempts.append(1)
            if len(close_attempts) == 1:
                raise VerifiedTaskWorkerDraining("in-flight work is still settling")
            try:
                if len(close_attempts) == 2:
                    await asyncio.Event().wait()
                raise asyncio.CancelledError
            except asyncio.CancelledError as cancellation:
                raise cancellation from VerifiedTaskWorkerDraining("still settling")

    async def scenario() -> None:
        retrying = asyncio.create_task(verified._finish_draining(DrainingWorker()))
        while len(close_attempts) < 2 and not retrying.done():
            await asyncio.sleep(0)
        # The plain draining report was retried, and that retry is still waiting.
        assert not retrying.done()
        retrying.cancel()
        with pytest.raises(asyncio.CancelledError):
            await retrying

    asyncio.run(scenario())
    # The cancelled retry is not retried again inside the cancelled task.
    assert len(close_attempts) == 2


def test_owner_settles_then_raises_a_cancellation_received_while_settling(tmp_path) -> None:
    release = asyncio.Event()
    close_attempts: list[int] = []

    class DrainingWorker:
        async def aclose(self) -> None:
            close_attempts.append(1)
            await release.wait()

    async def scenario() -> verified.DemoOwner:
        owner = verified.DemoOwner(tmp_path)
        owner.retain(DrainingWorker())

        async def leave() -> None:
            async with owner:
                pass

        leaving = asyncio.create_task(leave())
        while not close_attempts:
            await asyncio.sleep(0)
        leaving.cancel("stop while settling")
        await asyncio.sleep(0)
        # The owner keeps waiting for its bounded settlement, then reports the
        # cancellation instead of dropping it.
        assert not leaving.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await leaving
        assert raised.value.args == ("stop while settling",)
        assert leaving.cancelled()
        return owner

    owner = asyncio.run(scenario())
    assert owner.settled
    assert len(close_attempts) == 1


def test_cancelled_cli_run_settles_before_removing_its_files(tmp_path, monkeypatch) -> None:
    # Through the real entrance: only the verifier's behavior and the temporary
    # directory's location are replaced, never a handle to the worker or stores.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["verified.py", "--store", "sqlite"])
    closed_stores = _record_store_closes(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    real_verify = ExactArtifactVerifier.verify

    async def resistant_verify(self, request):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.to_thread(release.wait)
            return await real_verify(self, request)

    monkeypatch.setattr(ExactArtifactVerifier, "verify", resistant_verify)

    async def scenario() -> None:
        run = asyncio.create_task(verified.main())
        try:
            await asyncio.to_thread(entered.wait, 120)
            (root,) = tmp_path.glob("cayu-verified-file-*")
            run.cancel()
            # The verifier still owns in-flight work: its files and stores stay.
            for _ in range(50):
                await asyncio.sleep(0.01)
            assert not run.done()
            assert (root / "state" / "tasks.sqlite").exists()
            assert closed_stores == []
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await run
        assert run.cancelled()
        # Once the work settles, the owner closes each store once, then the files go.
        assert sorted(closed_stores) == ["sessions", "tasks"]
        assert not root.exists()

    asyncio.run(scenario())


def _record_store_closes(monkeypatch, events: list[str] | None = None) -> list[str]:
    closed = [] if events is None else events
    for store_type, name in ((SQLiteTaskStore, "tasks"), (SQLiteSessionStore, "sessions")):
        real = store_type.close

        async def record_close(self, real=real, name=name) -> None:
            closed.append(name)
            await real(self)

        monkeypatch.setattr(store_type, "close", record_close)
    return closed


class _SettlingWorker:
    """A worker whose close reports draining until `settles` is set."""

    def __init__(self) -> None:
        self.settles = False
        self.closes = 0

    async def aclose(self) -> None:
        self.closes += 1
        if not self.settles:
            raise VerifiedTaskWorkerDraining("in-flight work is still settling")


def test_a_run_failure_is_reported_once_the_owner_finishes_the_drain(tmp_path, monkeypatch) -> None:
    closed_stores = _record_store_closes(monkeypatch)
    close_attempts: list[int] = []

    class DrainingWorker(VerifiedTaskWorker):
        async def run(self, stop=None, max_tasks=None) -> int:
            raise TimeoutError("the run timed out")

        async def aclose(self) -> None:
            close_attempts.append(1)
            # Outlasts the run's own retries; the owner's first retry settles.
            if len(close_attempts) <= 1 + verified.CLOSE_ATTEMPTS:
                raise VerifiedTaskWorkerDraining("in-flight work is still settling")
            await super().aclose()

    monkeypatch.setattr(verified, "VerifiedTaskWorker", DrainingWorker)

    with pytest.raises(TimeoutError, match="the run timed out") as raised:
        asyncio.run(run_demo(tmp_path, store="sqlite", lose_acknowledgement=False))
    assert _printed_once(raised.value, "TimeoutError: the run timed out")
    assert len(close_attempts) == 2 + verified.CLOSE_ATTEMPTS
    assert sorted(closed_stores) == ["sessions", "tasks"]


def test_owner_shuts_each_app_down_before_closing_its_stores(tmp_path, monkeypatch) -> None:
    events: list[str] = []
    _record_store_closes(monkeypatch, events)
    real_aclose = CayuApp.aclose

    async def record_aclose(self, **kwargs):
        outcome = await real_aclose(self, **kwargs)
        events.append("app" if outcome.settled else "unsettled app")
        return outcome

    monkeypatch.setattr(CayuApp, "aclose", record_aclose)

    asyncio.run(run_demo(tmp_path, store="sqlite"))

    # The restart closes the first app before its stores, and the owner the second.
    assert events == ["app", "tasks", "sessions", "app", "tasks", "sessions"]


def test_owner_keeps_the_stores_while_an_app_has_not_shut_down(tmp_path, monkeypatch) -> None:
    closed_stores = _record_store_closes(monkeypatch)
    real_aclose = CayuApp.aclose
    shutdown = {"settles": False}

    async def unsettled_aclose(self, **kwargs):
        outcome = await real_aclose(self, **kwargs)
        return outcome if shutdown["settles"] else SimpleNamespace(settled=False)

    monkeypatch.setattr(CayuApp, "aclose", unsettled_aclose)

    async def scenario() -> None:
        owner = verified.DemoOwner(tmp_path, store="sqlite")
        with pytest.raises(RuntimeError, match="an app has not finished shutting down"):
            async with owner:
                await owner.run(lose_acknowledgement=False)
        assert not owner.settled
        assert closed_stores == []
        # Settling is repeatable: once the app shuts down, the stores close.
        shutdown["settles"] = True
        assert await owner.settle()
        assert sorted(closed_stores) == ["sessions", "tasks"]

    asyncio.run(scenario())


def test_owner_settle_can_be_retried_after_its_bounded_attempts(tmp_path, monkeypatch) -> None:
    closed_stores = _record_store_closes(monkeypatch)

    async def scenario() -> None:
        owner = verified.DemoOwner(tmp_path, store="sqlite")
        worker = _SettlingWorker()
        owner.retain(worker)
        with pytest.raises(RuntimeError, match="the worker still owns in-flight work"):
            async with owner:
                pass
        assert not owner.settled
        assert worker.closes == verified.CLOSE_ATTEMPTS
        assert closed_stores == []
        worker.settles = True
        assert await owner.settle()
        assert owner.settled and owner.unsettled is None
        assert sorted(closed_stores) == ["sessions", "tasks"]
        assert await owner.settle()
        assert len(closed_stores) == 2

    asyncio.run(scenario())


def test_cli_keeps_its_files_when_the_drain_does_not_settle(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["verified.py", "--store", "sqlite"])
    closed_stores = _record_store_closes(monkeypatch)

    class DrainingWorker(VerifiedTaskWorker):
        async def run(self, stop=None, max_tasks=None) -> int:
            raise TimeoutError("the run timed out")

        async def aclose(self) -> None:
            raise VerifiedTaskWorkerDraining("in-flight work is still settling")

    monkeypatch.setattr(verified, "VerifiedTaskWorker", DrainingWorker)

    with pytest.raises(BaseExceptionGroup) as raised:
        asyncio.run(verified.main())
    assert verified._still_draining(raised.value)
    (root,) = tmp_path.glob("cayu-verified-file-*")
    assert (root / "state" / "tasks.sqlite").exists()
    assert closed_stores == []
    assert f"Keeping {root}: the worker still owns in-flight work." in capsys.readouterr().err


def test_a_cancellation_keeps_a_settlement_failure_as_its_cause(tmp_path, monkeypatch) -> None:
    close_failure = RuntimeError("closing the task store failed")

    async def failing_close(self) -> None:
        raise close_failure

    monkeypatch.setattr(SQLiteTaskStore, "close", failing_close)
    closed_sessions: list[object] = []
    real_session_close = SQLiteSessionStore.close

    async def record_session_close(self) -> None:
        closed_sessions.append(self)
        await real_session_close(self)

    monkeypatch.setattr(SQLiteSessionStore, "close", record_session_close)

    async def scenario() -> verified.DemoOwner:
        owner = verified.DemoOwner(tmp_path, store="sqlite")

        async def inside() -> None:
            async with owner:
                await asyncio.Event().wait()

        running = asyncio.create_task(inside())
        await asyncio.sleep(0)
        running.cancel("stop the demo")
        with pytest.raises(asyncio.CancelledError) as raised:
            await running
        assert raised.value.args == ("stop the demo",)
        handoff = raised.value.__cause__
        assert isinstance(handoff, verified.DemoNotSettled) and handoff.owner is owner
        assert handoff.__cause__ is close_failure
        return owner

    owner = asyncio.run(scenario())
    assert not owner.settled
    assert owner.unsettled == "closing the stores failed"
    # The session store still closes although the task store failed to.
    assert len(closed_sessions) == 1


def test_a_run_failure_stays_beside_a_settlement_failure(tmp_path, monkeypatch) -> None:
    close_failure = RuntimeError("closing the task store failed")

    async def failing_close(self) -> None:
        raise close_failure

    monkeypatch.setattr(SQLiteTaskStore, "close", failing_close)
    run_failure = ValueError("the run failed")

    async def scenario() -> None:
        with pytest.raises(BaseExceptionGroup) as raised:
            async with verified.DemoOwner(tmp_path, store="sqlite"):
                raise run_failure
        reported, handoff = raised.value.exceptions
        assert reported is run_failure
        assert isinstance(handoff, verified.DemoNotSettled)
        assert handoff.__cause__ is close_failure

    asyncio.run(scenario())


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


def _failing_task_store_close(monkeypatch, failure: BaseException) -> None:
    async def failing_close(self) -> None:
        raise failure

    monkeypatch.setattr(SQLiteTaskStore, "close", failing_close)


def test_a_finished_drain_reports_the_run_failure_beside_a_settlement_failure(
    tmp_path, monkeypatch
) -> None:
    close_failure = RuntimeError("closing the task store failed")
    _failing_task_store_close(monkeypatch, close_failure)
    run_failure = TimeoutError("the run timed out")

    async def scenario() -> None:
        owner = verified.DemoOwner(tmp_path, store="sqlite")
        worker = _SettlingWorker()
        worker.settles = True
        owner.retain(worker, run_failure=run_failure)
        with pytest.raises(BaseExceptionGroup) as raised:
            async with owner:
                raise VerifiedTaskWorkerDraining("reported before the drain finished")
        # The drain finished, so the stale draining report gives way to the
        # run's own failure, even though the stores did not close.
        reported, handoff = raised.value.exceptions
        assert reported is run_failure
        assert isinstance(handoff, verified.DemoNotSettled)
        assert handoff.__cause__ is close_failure

    asyncio.run(scenario())


def test_an_unsettled_owner_reports_beside_a_run_failure(tmp_path) -> None:
    run_failure = ValueError("the run failed")

    async def scenario() -> None:
        owner = verified.DemoOwner(tmp_path, store="sqlite")
        owner.retain(_SettlingWorker())
        with pytest.raises(BaseExceptionGroup) as raised:
            async with owner:
                raise run_failure
        first, unsettled = raised.value.exceptions
        assert first is run_failure
        assert isinstance(unsettled, RuntimeError)
        assert "the worker still owns in-flight work" in str(unsettled)
        owner._draining.settles = True
        assert await owner.settle()

    asyncio.run(scenario())


def test_an_unsettled_owner_reports_on_a_cancellation(tmp_path) -> None:
    async def scenario() -> None:
        owner = verified.DemoOwner(tmp_path, store="sqlite")
        worker = _SettlingWorker()
        owner.retain(worker)

        async def inside() -> None:
            async with owner:
                await asyncio.Event().wait()

        running = asyncio.create_task(inside())
        await asyncio.sleep(0)
        running.cancel("stop the demo")
        with pytest.raises(asyncio.CancelledError) as raised:
            await running
        assert raised.value.args == ("stop the demo",)
        assert isinstance(raised.value.__cause__, RuntimeError)
        assert "the worker still owns in-flight work" in str(raised.value.__cause__)
        worker.settles = True
        assert await owner.settle()

    asyncio.run(scenario())


def test_a_failed_worker_close_is_named_as_the_reason(tmp_path) -> None:
    close_failure = ValueError("the worker's close failed")

    class FailingWorker:
        async def aclose(self) -> None:
            raise close_failure

    async def scenario() -> verified.DemoOwner:
        owner = verified.DemoOwner(tmp_path, store="sqlite")
        owner.retain(FailingWorker())
        with pytest.raises(verified.DemoNotSettled) as raised:
            async with owner:
                pass
        assert raised.value.__cause__ is close_failure
        return owner

    owner = asyncio.run(scenario())
    assert owner.unsettled == "closing the worker failed"
    owner._draining = None
    assert asyncio.run(owner.settle())


def test_a_cancelled_settlement_does_not_replace_the_callers_cancellation(tmp_path) -> None:
    class CancellingWorker:
        async def aclose(self) -> None:
            raise asyncio.CancelledError("raised by the worker's own close")

    async def scenario() -> verified.DemoOwner:
        owner = verified.DemoOwner(tmp_path, store="sqlite")
        owner.retain(CancellingWorker())

        async def inside() -> None:
            async with owner:
                await asyncio.Event().wait()

        running = asyncio.create_task(inside())
        await asyncio.sleep(0)
        running.cancel("stop the demo")
        with pytest.raises(asyncio.CancelledError) as raised:
            await running
        # The caller's cancellation stays authoritative and names the failure.
        assert raised.value.args == ("stop the demo",)
        handoff = raised.value.__cause__
        assert isinstance(handoff, verified.DemoNotSettled)
        assert str(handoff.__cause__) == "Settling the demo was cancelled."
        return owner

    owner = asyncio.run(scenario())
    assert owner.unsettled == "settling was cancelled"
    owner._draining = None
    assert asyncio.run(owner.settle())


def test_a_cancellation_while_settling_keeps_the_run_failure_in_the_traceback(
    tmp_path,
) -> None:
    import traceback

    async def scenario() -> asyncio.CancelledError:
        owner = verified.DemoOwner(tmp_path, store="sqlite")
        worker = _SettlingWorker()
        owner.retain(worker)
        closing = asyncio.Event()
        real_aclose = worker.aclose

        async def slow_aclose() -> None:
            closing.set()
            await asyncio.sleep(0.05)
            await real_aclose()

        worker.aclose = slow_aclose

        async def inside() -> None:
            async with owner:
                raise ValueError("the run failed")

        running = asyncio.create_task(inside())
        await closing.wait()
        running.cancel("stop while settling")
        with pytest.raises(asyncio.CancelledError) as raised:
            await running
        worker.settles = True
        assert await owner.settle()
        return raised.value

    cancellation = asyncio.run(scenario())
    run_failure, unsettled = cancellation.__cause__.exceptions
    assert str(run_failure) == "the run failed"
    assert "the worker still owns in-flight work" in str(unsettled)
    printed = "".join(traceback.format_exception(cancellation))
    assert printed.count("ValueError: the run failed") == 1


def test_run_demo_hands_an_unsettled_owner_to_its_caller(tmp_path, monkeypatch) -> None:
    closed_stores = _record_store_closes(monkeypatch)
    work = {"settles": False}

    class DrainingWorker(VerifiedTaskWorker):
        async def run(self, stop=None, max_tasks=None) -> int:
            raise TimeoutError("the run timed out")

        async def aclose(self) -> None:
            if not work["settles"]:
                raise VerifiedTaskWorkerDraining("in-flight work is still settling")
            await super().aclose()

    monkeypatch.setattr(verified, "VerifiedTaskWorker", DrainingWorker)

    async def scenario() -> None:
        with pytest.raises(BaseExceptionGroup) as raised:
            await run_demo(tmp_path, store="sqlite", lose_acknowledgement=False)
        # The owner's bounded attempts ran out: the error hands the owner over.
        handoffs = raised.value.subgroup(lambda error: isinstance(error, verified.DemoNotSettled))
        assert handoffs is not None
        (handoff,) = handoffs.exceptions
        assert closed_stores == []
        # The work finishes later; the caller settles it through the handed-over owner.
        work["settles"] = True
        assert await handoff.owner.settle()
        assert sorted(closed_stores) == ["sessions", "tasks"]

    asyncio.run(scenario())


def test_a_cancellation_during_the_close_retry_keeps_the_run_failure(tmp_path, monkeypatch) -> None:
    import traceback

    close_attempts: list[int] = []

    class DrainingWorker(VerifiedTaskWorker):
        async def run(self, stop=None, max_tasks=None) -> int:
            raise TimeoutError("the run timed out")

        async def aclose(self) -> None:
            close_attempts.append(1)
            if len(close_attempts) == 1:
                raise VerifiedTaskWorkerDraining("in-flight work is still settling")
            if len(close_attempts) == 2:
                # The run's own close retry, which the caller cancels. Like the
                # real worker, the cancellation carries the draining report.
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as cancellation:
                    raise cancellation from VerifiedTaskWorkerDraining("still settling")
            await super().aclose()

    monkeypatch.setattr(verified, "VerifiedTaskWorker", DrainingWorker)

    async def scenario() -> asyncio.CancelledError:
        running = asyncio.create_task(
            run_demo(tmp_path, store="sqlite", lose_acknowledgement=False)
        )
        while len(close_attempts) < 2 and not running.done():
            await asyncio.sleep(0)
        assert not running.done()
        running.cancel("stop the demo")
        try:
            await running
        except asyncio.CancelledError as error:
            caught = error
        else:
            pytest.fail("An ordinary CancelledError handler must catch the cancellation.")
        assert caught.args == ("stop the demo",)
        assert running.cancelled()
        return caught

    cancellation = asyncio.run(scenario())
    # The owner's retry settled the worker, and the run's failure stays visible once.
    assert len(close_attempts) == 3
    printed = "".join(traceback.format_exception(cancellation))
    assert printed.count("TimeoutError: the run timed out") == 1


def _printed_once(error: BaseException, line: str) -> bool:
    import traceback

    return "".join(traceback.format_exception(error)).count(line) == 1


def test_a_run_failure_after_a_settled_close_retry_is_printed_once(tmp_path, monkeypatch) -> None:
    close_attempts: list[int] = []

    class DrainingWorker(VerifiedTaskWorker):
        async def run(self, stop=None, max_tasks=None) -> int:
            raise TimeoutError("the run timed out")

        async def aclose(self) -> None:
            close_attempts.append(1)
            if len(close_attempts) == 1:
                raise VerifiedTaskWorkerDraining("in-flight work is still settling")
            await super().aclose()

    monkeypatch.setattr(verified, "VerifiedTaskWorker", DrainingWorker)

    with pytest.raises(TimeoutError, match="the run timed out") as raised:
        asyncio.run(run_demo(tmp_path, store="sqlite", lose_acknowledgement=False))
    assert len(close_attempts) == 2
    assert _printed_once(raised.value, "TimeoutError: the run timed out")
