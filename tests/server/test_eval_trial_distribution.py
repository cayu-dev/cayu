"""The trials of one durable eval run, spread across several workers."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from tests.evals.eval_store_conformance import _trial_checkpoint
from tests.evals.test_corpus_execution import _source, _target

from cayu import AgentSpec
from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    FinalOutputEqualsAssertionSpec,
    RootStatusAssertionSpec,
    RunInputSpec,
    TrialRequestSpec,
)
from cayu.evals.execution import evaluation_target_identity, run_corpus_suite
from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1
from cayu.evals.store import (
    EvalRunClaimLost,
    EvalRunInvocation,
    EvalRunRequest,
    EvalRunStatus,
    EvalStoreTransientContention,
)
from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1
from cayu.messages import TextPart
from cayu.project_control_plane import ProjectEvalTarget
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.server.evals_registry import (
    EvalTargetRegistry,
    declared_eval_target_registrations,
    resolved_evals_runtime,
    target_for_eval_invocation,
)
from cayu.server.evals_worker import EvalRunCoordinator
from cayu.storage.evals_sqlite import SQLiteEvalStore

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TARGET_KEY = "shared-trials-agent"
_CASE_INPUTS = {"refund-approval": "approve refund 42", "refund-denial": "deny refund 7"}
_TRIALS = 3
_MAX_CONCURRENCY = 4
# Fields that differ between any two executions of the same trials.
_VOLATILE_FIELDS = frozenset(
    {
        "run_id",
        "session_id",
        "started_at",
        "completed_at",
        "duration_ms",
        "source_trial_revision",
    }
)


class _Trials:
    """What both workers' providers share: who ran which trial, and a start gate."""

    def __init__(self, *, open_at: int) -> None:
        self.open_at = open_at
        self.in_flight = 0
        self.gate = asyncio.Event()
        self.ran: list[tuple[str, str]] = []


class _EchoProvider(ModelProvider):
    """Answers with the trial's input once enough trials run at once in all workers."""

    name = "shared-trials-provider"

    def __init__(self, worker: str, trials: _Trials) -> None:
        self._worker = worker
        self._trials = trials

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        # Every worker's copy behaves alike, so a run admitted with one runs on all.
        return ExecutionProfileBehaviorIdentity(
            name="tests:shared-trials-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        part = request.messages[-1].content[0]
        assert isinstance(part, TextPart)
        text = part.text
        self._trials.ran.append((self._worker, text))
        self._trials.in_flight += 1
        if self._trials.in_flight >= self._trials.open_at:
            self._trials.gate.set()
        try:
            await asyncio.wait_for(self._trials.gate.wait(), timeout=30)
        finally:
            self._trials.in_flight -= 1
        yield ModelStreamEvent.text_delta(text)
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            }
        )


def _corpus() -> EvalCorpusDocument:
    suite = EvalSuiteSpec.create(
        id="refund-regressions",
        name="Refund regressions",
        trial_request=TrialRequestSpec(
            trials=_TRIALS,
            timeout_seconds=30,
            trial_policy=EvalSuiteTrialPolicyV1.create(
                trial_count=_TRIALS,
                minimum_passed_trials=_TRIALS,
                max_concurrency=_MAX_CONCURRENCY,
            ),
        ),
    )
    cases = tuple(
        EvalCaseSpec.create(
            id=case_id,
            suite_id=suite.id,
            name=case_id.replace("-", " ").capitalize(),
            source=_source(),
            input=RunInputSpec(messages=(CorpusUserMessageSpec(text=text),)),
            assertions=(
                RootStatusAssertionSpec(id="completed", expected="completed"),
                FinalOutputEqualsAssertionSpec(id="answer", expected=text),
            ),
        )
        for case_id, text in _CASE_INPUTS.items()
    )
    return EvalCorpusDocument.create(
        target_key=_TARGET_KEY,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=(suite,),
        cases=cases,
    )


def _declared(provider: ModelProvider) -> ProjectEvalTarget:
    target = _target(provider, key=_TARGET_KEY)
    return ProjectEvalTarget(
        reference="evals:shared_trials",
        target=target,
        execution_profile_policy=EvalExecutionProfilePolicyV1(
            fixture_strategy="application_managed",
            reset_strategy="application_managed",
            effect_posture="isolated_application_authority",
            isolation_revision="sha256:" + "5" * 64,
            max_trials=_TRIALS,
            max_concurrency=_MAX_CONCURRENCY,
        ),
    )


def _runtime(store: SQLiteEvalStore, declared: ProjectEvalTarget, *, worker_processes: int):
    runtime = resolved_evals_runtime(
        explicit=None,
        registry=EvalTargetRegistry(
            declared_eval_target_registrations((declared,), project_root=_PROJECT_ROOT)
        ),
        automatic_store=store,
    )
    assert runtime is not None
    return replace(
        runtime,
        poll_interval_seconds=0.02,
        lease_seconds=5,
        shutdown_grace_seconds=2.0,
        worker_processes=worker_processes,
    )


async def _admit(runtime, store: SQLiteEvalStore, declared: ProjectEvalTarget, run_id: str):
    target = declared.target
    corpus = _corpus()
    registration = runtime.registry.registration(target.key)
    invocation = EvalRunInvocation()
    prepared = await runtime.registry.prepare_execution_profile(
        target.key,
        effective_target=target_for_eval_invocation(registration.execution_target(), invocation),
    )
    await store.save_corpus(corpus, redact_json=target.app.redact_json)
    await store.admit_run(
        EvalRunRequest(
            run_id=run_id,
            idempotency_key="sha256:" + hashlib.sha256(run_id.encode()).hexdigest(),
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id=corpus.suites[0].id,
            suite_revision=corpus.suites[0].revision,
            max_concurrency=_MAX_CONCURRENCY,
            invocation=invocation.model_copy(
                update={
                    "execution_profile": prepared.binding,
                    "execution_profile_snapshot": prepared.snapshot,
                },
                deep=True,
            ),
        ),
        redact_json=target.app.redact_json,
    )
    return corpus


def _stable(value: Any) -> Any:
    """``value`` without what differs between two executions of the same trials."""

    if isinstance(value, dict):
        return {key: _stable(item) for key, item in value.items() if key not in _VOLATILE_FIELDS}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def _stable_run(result) -> dict[str, Any]:
    """A published run without its execution-specific fields or its content revision."""

    run = _stable(result.run.model_dump(mode="json"))
    # The run's content revision hashes its trials' times and sessions too.
    del run["revision"]
    return run


def test_two_workers_run_one_runs_trials_and_publish_the_single_process_result(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        trials = _Trials(open_at=_MAX_CONCURRENCY)
        owner_declared = _declared(_EchoProvider("owner", trials))
        helper_declared = _declared(_EchoProvider("helper", trials))
        # Two processes of one worker: each store connection is a process's own.
        owner_store = SQLiteEvalStore(tmp_path / "evals.db")
        helper_store = SQLiteEvalStore(tmp_path / "evals.db")
        owner_runtime = _runtime(owner_store, owner_declared, worker_processes=2)
        helper_runtime = _runtime(helper_store, helper_declared, worker_processes=2)
        corpus = await _admit(owner_runtime, owner_store, owner_declared, "shared-run")
        owner = EvalRunCoordinator(owner_runtime)
        helper = EvalRunCoordinator(helper_runtime)
        try:
            owner.start()
            # Let the owner claim the run before the helper starts looking.
            for _ in range(200):
                record = await owner_store.load_run("shared-run")
                assert record is not None
                if record.status is EvalRunStatus.RUNNING:
                    break
                await asyncio.sleep(0.01)
            helper.start()
            record = await owner_store.wait_for_run_terminal("shared-run", timeout_seconds=60)
            assert record is not None
            assert record.status is EvalRunStatus.COMPLETED
            distributed = await owner_store.load_result("shared-run")
        finally:
            await owner.stop()
            await helper.stop()
            await owner_store.close()
            await helper_store.close()

        # Each worker ran at most its half of the run's four concurrent trials, so
        # the gate that opens at four trials at once proves both ran trials.
        assert {worker for worker, _ in trials.ran} == {"owner", "helper"}
        assert len(trials.ran) == len(_CASE_INPUTS) * _TRIALS
        assert distributed is not None
        single = await run_corpus_suite(
            _target(_EchoProvider("single", _Trials(open_at=1)), key=_TARGET_KEY),
            corpus,
            corpus.suites[0].id,
            max_concurrency=_MAX_CONCURRENCY,
        )
        assert [
            (case.case_id, [trial.trial_number for trial in case.trials])
            for case in distributed.run.cases
        ] == [(case_id, [1, 2, 3]) for case_id in _CASE_INPUTS]
        assert _stable_run(distributed) == _stable_run(single)
        # The worker records its target under the project's manifest, as one
        # process running the whole run records it.
        assert distributed.target == evaluation_target_identity(
            owner_declared.target, project_root=_PROJECT_ROOT
        )

    asyncio.run(run())


class _HeldEchoProvider(_EchoProvider):
    """Starts answering only once ``held`` is set."""

    def __init__(self, worker: str, trials: _Trials, held: asyncio.Event) -> None:
        super().__init__(worker, trials)
        self._held = held

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        await self._held.wait()
        async for event in super().stream(request):
            yield event


def test_a_trial_whose_worker_was_lost_is_claimed_again_and_its_result_refused(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        trials = _Trials(open_at=1)
        held = asyncio.Event()
        declared = _declared(_HeldEchoProvider("owner", trials, held))
        store = SQLiteEvalStore(tmp_path / "evals.db")
        # The owner runs one trial at a time, so the lost worker can take another.
        runtime = _runtime(store, declared, worker_processes=_MAX_CONCURRENCY)
        await _admit(runtime, store, declared, "lost-worker-run")
        owner = EvalRunCoordinator(runtime)
        try:
            owner.start()
            lost = None
            for _ in range(500):
                lost = await store.claim_run_trial((_TARGET_KEY,), lease_seconds=1)
                if lost is not None:
                    break
                await asyncio.sleep(0.01)
            assert lost is not None
            assert lost.claim.epoch == 1
            # The lost worker never renews its claim or retains a result.
            held.set()
            record = await store.wait_for_run_terminal("lost-worker-run", timeout_seconds=60)
            assert record is not None
            assert record.status is EvalRunStatus.COMPLETED
            result = await store.load_result("lost-worker-run")
            assert result is not None
            assert [
                (case.case_id, [trial.trial_number for trial in case.trials])
                for case in result.run.cases
            ] == [(case_id, [1, 2, 3]) for case_id in _CASE_INPUTS]
            assert result.run.status == "passed"
            # Every trial ran once, in the owner; the lost worker's trial after its
            # claim expired.
            assert len(trials.ran) == len(_CASE_INPUTS) * _TRIALS
            with pytest.raises(EvalRunClaimLost):
                await store.save_claimed_trial_checkpoint(
                    lost.claim,
                    _trial_checkpoint(lost.claim.case_id, lost.claim.trial_number),
                    redact_json=declared.target.app.redact_json,
                )
        finally:
            held.set()
            await owner.stop()
            await store.close()

    asyncio.run(run())


def test_a_worker_running_until_idle_runs_trials_of_a_run_another_worker_owns(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        trials = _Trials(open_at=_MAX_CONCURRENCY)
        owner_declared = _declared(_EchoProvider("owner", trials))
        helper_declared = _declared(_EchoProvider("helper", trials))
        owner_store = SQLiteEvalStore(tmp_path / "evals.db")
        helper_store = SQLiteEvalStore(tmp_path / "evals.db")
        owner_runtime = _runtime(owner_store, owner_declared, worker_processes=2)
        helper_runtime = _runtime(helper_store, helper_declared, worker_processes=2)
        await _admit(owner_runtime, owner_store, owner_declared, "idle-helper-run")
        owner = EvalRunCoordinator(owner_runtime)
        try:
            owner.start()
            for _ in range(200):
                record = await owner_store.load_run("idle-helper-run")
                assert record is not None
                if record.status is EvalRunStatus.RUNNING:
                    break
                await asyncio.sleep(0.01)
            # It takes no run of its own, helps with the held one, and returns
            # only once that run is no longer outstanding.
            assert await EvalRunCoordinator(helper_runtime).run_until_idle() == (0, True)
            record = await owner_store.load_run("idle-helper-run")
            assert record is not None
            assert record.status is EvalRunStatus.COMPLETED
        finally:
            await owner.stop()
            await owner_store.close()
            await helper_store.close()
        assert {worker for worker, _ in trials.ran} == {"owner", "helper"}

    asyncio.run(run())


class _ProcessLocalEchoProvider(_EchoProvider):
    """Declares no identity, so a run admitted with another copy can't run on it."""

    @property
    def execution_profile_identity(self) -> None:
        return None


def test_a_worker_whose_target_differs_leaves_the_trials_to_the_owner(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflights = 0

    async def run() -> None:
        nonlocal preflights
        trials = _Trials(open_at=1)
        held = asyncio.Event()
        owner_declared = _declared(_HeldEchoProvider("owner", trials, held))
        helper_declared = _declared(_ProcessLocalEchoProvider("helper", trials))
        owner_store = SQLiteEvalStore(tmp_path / "evals.db")
        helper_store = SQLiteEvalStore(tmp_path / "evals.db")
        owner_runtime = _runtime(owner_store, owner_declared, worker_processes=_MAX_CONCURRENCY)
        helper_runtime = _runtime(helper_store, helper_declared, worker_processes=1)
        await _admit(owner_runtime, owner_store, owner_declared, "different-target-run")
        owner = EvalRunCoordinator(owner_runtime)
        helper = EvalRunCoordinator(helper_runtime)
        prepare = helper._preflight_run

        async def counted_preflight(run_record, registration):
            nonlocal preflights
            preflights += 1
            return await prepare(run_record, registration)

        monkeypatch.setattr(helper, "_preflight_run", counted_preflight)
        try:
            owner.start()
            helper.start()
            # The helper keeps finding the open trials while the owner is held.
            await asyncio.sleep(0.5)
            held.set()
            record = await owner_store.wait_for_run_terminal(
                "different-target-run", timeout_seconds=60
            )
            assert record is not None
            assert record.status is EvalRunStatus.COMPLETED
        finally:
            held.set()
            await owner.stop()
            await helper.stop()
            await owner_store.close()
            await helper_store.close()
        assert {worker for worker, _ in trials.ran} == {"owner"}
        assert len(trials.ran) == len(_CASE_INPUTS) * _TRIALS
        warnings = [
            record
            for record in caplog.records
            if record.getMessage().startswith("This worker can't run trials")
        ]
        assert len(warnings) == 1
        assert warnings[0].__dict__["eval_failure_reason"] == "execution_profile_changed"
        # It prepared the run once, not again each time it found an open trial.
        assert preflights == 1

    with caplog.at_level("WARNING", logger="cayu.server.evals_worker"):
        asyncio.run(run())


def test_a_launch_scoped_worker_runs_no_trials_of_other_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        trials = _Trials(open_at=1)
        store = SQLiteEvalStore(tmp_path / "evals.db")
        declared = _declared(_EchoProvider("launch", trials))
        runtime = _runtime(store, declared, worker_processes=1)
        claims: list[tuple[str, ...]] = []

        async def recorded_claim(target_keys, **_options):
            claims.append(target_keys)

        monkeypatch.setattr(store, "claim_run_trial", recorded_claim)
        try:
            launch = EvalRunCoordinator(runtime, launch_revision="sha256:" + "6" * 64)
            assert await launch._help_with_trials() is False
            assert claims == []
            # Any other worker looks for open trials of its targets.
            assert await EvalRunCoordinator(runtime)._help_with_trials() is False
            assert claims == [(_TARGET_KEY,)]
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 1), ("4", 4), ("1", 1), ("0", 1), ("-2", 1), ("four", 1), (" 3 ", 3)],
)
def test_a_supervised_worker_knows_how_many_processes_share_its_runs(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: int
) -> None:
    from cayu.runtime._process_workers import (
        WORKER_COUNT_ENVIRONMENT,
        process_worker_count,
        process_worker_environment,
    )

    if value is None:
        monkeypatch.delenv(WORKER_COUNT_ENVIRONMENT, raising=False)
    else:
        monkeypatch.setenv(WORKER_COUNT_ENVIRONMENT, value)
    assert process_worker_count() == expected
    assert process_worker_environment(0, 3)[WORKER_COUNT_ENVIRONMENT] == "3"


def test_store_errors_while_the_owner_claims_trials_are_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        trials = _Trials(open_at=1)
        declared = _declared(_EchoProvider("owner", trials))
        store = SQLiteEvalStore(tmp_path / "evals.db")
        runtime = _runtime(store, declared, worker_processes=1)
        await _admit(runtime, store, declared, "store-error-run")
        failures = {"claim_run_trial": 2, "unfinished_run_trials": 1}

        def failing(name: str):
            original = getattr(store, name)

            async def operation(*args, **kwargs):
                if failures[name]:
                    failures[name] -= 1
                    raise sqlite3.OperationalError("database is locked")
                return await original(*args, **kwargs)

            return operation

        for name in failures:
            monkeypatch.setattr(store, name, failing(name))
        owner = EvalRunCoordinator(runtime)
        try:
            owner.start()
            record = await store.wait_for_run_terminal("store-error-run", timeout_seconds=60)
        finally:
            await owner.stop()
            await store.close()
        assert record is not None
        assert record.status is EvalRunStatus.COMPLETED
        assert failures == {"claim_run_trial": 0, "unfinished_run_trials": 0}
        assert len(trials.ran) == len(_CASE_INPUTS) * _TRIALS

    asyncio.run(run())


def test_a_helper_keeps_its_claim_while_it_prepares_and_never_runs_a_trial_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        trials = _Trials(open_at=1)
        held = asyncio.Event()
        free = asyncio.Event()
        free.set()
        owner_declared = _declared(_HeldEchoProvider("owner", trials, held))
        # The same provider type, so both workers have one execution profile.
        helper_declared = _declared(_HeldEchoProvider("helper", trials, free))
        owner_store = SQLiteEvalStore(tmp_path / "evals.db")
        helper_store = SQLiteEvalStore(tmp_path / "evals.db")
        owner_runtime = _runtime(owner_store, owner_declared, worker_processes=2)
        # The helper's lease is shorter than its preparation takes.
        helper_runtime = replace(
            _runtime(helper_store, helper_declared, worker_processes=2), lease_seconds=1
        )
        await _admit(owner_runtime, owner_store, owner_declared, "slow-helper-run")
        owner = EvalRunCoordinator(owner_runtime)
        helper = EvalRunCoordinator(helper_runtime)
        preparing = asyncio.Event()
        prepare = helper._preflight_run

        async def slow_preflight(run_record, registration):
            preparing.set()
            await asyncio.sleep(2.5)
            return await prepare(run_record, registration)

        monkeypatch.setattr(helper, "_preflight_run", slow_preflight)
        try:
            owner.start()
            for _ in range(200):
                record = await owner_store.load_run("slow-helper-run")
                assert record is not None
                if record.status is EvalRunStatus.RUNNING:
                    break
                await asyncio.sleep(0.01)
            helper.start()
            await asyncio.wait_for(preparing.wait(), timeout=10)
            # The owner frees up while the helper still prepares its trial.
            held.set()
            record = await owner_store.wait_for_run_terminal("slow-helper-run", timeout_seconds=60)
            assert record is not None
            assert record.status is EvalRunStatus.COMPLETED
            await asyncio.sleep(0.5)
        finally:
            held.set()
            await owner.stop()
            await helper.stop()
            await owner_store.close()
            await helper_store.close()
        # Every trial ran exactly once: the helper's claim lived through its
        # preparation, so the owner never took that trial too.
        assert sorted(text for _, text in trials.ran) == sorted(
            text for text in _CASE_INPUTS.values() for _ in range(_TRIALS)
        )
        assert any(worker == "helper" for worker, _ in trials.ran)

    asyncio.run(run())


def test_contention_while_the_owner_opens_and_gathers_trials_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        trials = _Trials(open_at=1)
        declared = _declared(_EchoProvider("owner", trials))
        store = SQLiteEvalStore(tmp_path / "evals.db")
        runtime = _runtime(store, declared, worker_processes=1)
        await _admit(runtime, store, declared, "contended-owner-run")
        deferrals = {"open_run_trials": 1, "load_trial_checkpoints": 1}

        def deferred(name: str):
            original = getattr(store, name)

            async def operation(*args, **kwargs):
                if deferrals[name]:
                    deferrals[name] -= 1
                    raise EvalStoreTransientContention("The eval store is busy.")
                return await original(*args, **kwargs)

            return operation

        for name in deferrals:
            monkeypatch.setattr(store, name, deferred(name))
        owner = EvalRunCoordinator(runtime)
        try:
            owner.start()
            record = await store.wait_for_run_terminal("contended-owner-run", timeout_seconds=60)
        finally:
            await owner.stop()
            await store.close()
        assert record is not None
        assert record.status is EvalRunStatus.COMPLETED
        assert deferrals == {"open_run_trials": 0, "load_trial_checkpoints": 0}
        assert len(trials.ran) == len(_CASE_INPUTS) * _TRIALS

    asyncio.run(run())


def test_a_trial_whose_worker_changed_its_target_meanwhile_is_not_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cayu.evals import runner as runner_module

    async def run() -> None:
        trials = _Trials(open_at=1)
        held = asyncio.Event()
        free = asyncio.Event()
        free.set()
        owner_declared = _declared(_HeldEchoProvider("owner", trials, held))
        helper_declared = _declared(_HeldEchoProvider("helper", trials, free))
        observe = runner_module.observe_eval_trial_result
        changed = False

        def change_the_helper_after_its_first_trial(result) -> None:
            # The owner is held, so the first trial to finish is the helper's:
            # its app gains an agent after the trial ran, before it is retained.
            nonlocal changed
            if not changed:
                changed = True
                helper_declared.target.app.register_agent(
                    AgentSpec(name="late-agent", model="fixture-model")
                )
            observe(result)

        monkeypatch.setattr(
            runner_module, "observe_eval_trial_result", change_the_helper_after_its_first_trial
        )
        owner_store = SQLiteEvalStore(tmp_path / "evals.db")
        helper_store = SQLiteEvalStore(tmp_path / "evals.db")
        owner_runtime = _runtime(owner_store, owner_declared, worker_processes=2)
        helper_runtime = _runtime(helper_store, helper_declared, worker_processes=2)
        await _admit(owner_runtime, owner_store, owner_declared, "changed-target-run")
        owner = EvalRunCoordinator(owner_runtime)
        helper = EvalRunCoordinator(helper_runtime)
        try:
            owner.start()
            for _ in range(200):
                record = await owner_store.load_run("changed-target-run")
                assert record is not None
                if record.status is EvalRunStatus.RUNNING:
                    break
                await asyncio.sleep(0.01)
            helper.start()
            for _ in range(500):
                if changed:
                    break
                await asyncio.sleep(0.01)
            assert changed
            # Let the helper settle its trials before the owner runs on.
            await asyncio.sleep(1.0)
            held.set()
            record = await owner_store.wait_for_run_terminal(
                "changed-target-run", timeout_seconds=60
            )
            assert record is not None
            assert record.status is EvalRunStatus.COMPLETED
        finally:
            held.set()
            await owner.stop()
            await helper.stop()
            await owner_store.close()
            await helper_store.close()
        assert any(worker == "helper" for worker, _ in trials.ran)
        # None of the helper's results counted once its target had changed:
        # the owner ran every trial itself.
        assert sum(worker == "owner" for worker, _ in trials.ran) == len(_CASE_INPUTS) * _TRIALS

    asyncio.run(run())


def test_a_trial_survives_a_store_error_while_its_claim_is_renewed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        trials = _Trials(open_at=1)
        held = asyncio.Event()
        declared = _declared(_HeldEchoProvider("owner", trials, held))
        store = SQLiteEvalStore(tmp_path / "evals.db")
        # Renewed every half second, so a held trial sees several renewals.
        runtime = replace(_runtime(store, declared, worker_processes=1), lease_seconds=2)
        await _admit(runtime, store, declared, "renewal-error-run")
        heartbeat = store.heartbeat_run_trial
        claim = store.claim_run_trial
        heartbeats: dict[str, int] = {}
        failed = 0
        claimed = 0

        async def failing_first_renewal(trial_claim, **kwargs):
            nonlocal failed
            # A claim's first heartbeat confirms it before its start; the second
            # is its first renewal while the trial runs.
            heartbeats[trial_claim.claim_id] = heartbeats.get(trial_claim.claim_id, 0) + 1
            if heartbeats[trial_claim.claim_id] == 2 and not failed:
                failed += 1
                raise sqlite3.OperationalError("disk I/O error")
            return await heartbeat(trial_claim, **kwargs)

        async def counted_claim(*args, **kwargs):
            nonlocal claimed
            lease = await claim(*args, **kwargs)
            claimed += lease is not None
            return lease

        monkeypatch.setattr(store, "heartbeat_run_trial", failing_first_renewal)
        monkeypatch.setattr(store, "claim_run_trial", counted_claim)
        owner = EvalRunCoordinator(runtime)
        try:
            owner.start()
            for _ in range(500):
                if failed:
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.6)
            held.set()
            record = await store.wait_for_run_terminal("renewal-error-run", timeout_seconds=60)
        finally:
            held.set()
            await owner.stop()
            await store.close()
        assert record is not None
        assert record.status is EvalRunStatus.COMPLETED
        assert failed == 1
        # The failed renewal was retried: no trial was stopped and claimed again.
        assert claimed == len(_CASE_INPUTS) * _TRIALS
        assert len(trials.ran) == len(_CASE_INPUTS) * _TRIALS

    asyncio.run(run())


def test_contention_before_a_trial_starts_keeps_the_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        trials = _Trials(open_at=1)
        declared = _declared(_EchoProvider("owner", trials))
        store = SQLiteEvalStore(tmp_path / "evals.db")
        runtime = _runtime(store, declared, worker_processes=1)
        await _admit(runtime, store, declared, "start-contention-run")
        heartbeat = store.heartbeat_run_trial
        claim = store.claim_run_trial
        confirmed: set[str] = set()
        claimed = 0

        async def contended_first_heartbeat(trial_claim, **kwargs):
            # The first heartbeat of a claim is the check right before its start.
            if trial_claim.claim_id not in confirmed:
                confirmed.add(trial_claim.claim_id)
                raise EvalStoreTransientContention("The eval store is busy.")
            return await heartbeat(trial_claim, **kwargs)

        async def counted_claim(*args, **kwargs):
            nonlocal claimed
            lease = await claim(*args, **kwargs)
            claimed += lease is not None
            return lease

        monkeypatch.setattr(store, "heartbeat_run_trial", contended_first_heartbeat)
        monkeypatch.setattr(store, "claim_run_trial", counted_claim)
        owner = EvalRunCoordinator(runtime)
        try:
            owner.start()
            record = await store.wait_for_run_terminal("start-contention-run", timeout_seconds=60)
        finally:
            await owner.stop()
            await store.close()
        assert record is not None
        assert record.status is EvalRunStatus.COMPLETED
        # Each trial was claimed once: contention delayed its start, nothing more.
        assert claimed == len(_CASE_INPUTS) * _TRIALS
        assert len(trials.ran) == len(_CASE_INPUTS) * _TRIALS

    asyncio.run(run())


def test_a_helper_that_fails_waits_before_it_looks_for_trials_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        trials = _Trials(open_at=1)
        held = asyncio.Event()
        owner_declared = _declared(_HeldEchoProvider("owner", trials, held))
        helper_declared = _declared(_HeldEchoProvider("helper", trials, held))
        owner_store = SQLiteEvalStore(tmp_path / "evals.db")
        helper_store = SQLiteEvalStore(tmp_path / "evals.db")
        owner_runtime = _runtime(owner_store, owner_declared, worker_processes=2)
        helper_runtime = _runtime(helper_store, helper_declared, worker_processes=2)
        await _admit(owner_runtime, owner_store, owner_declared, "failing-helper-run")
        owner = EvalRunCoordinator(owner_runtime)
        helper = EvalRunCoordinator(helper_runtime)

        async def failing_trials(*_args, **_kwargs):
            raise RuntimeError("The helper's trials failed.")

        monkeypatch.setattr(helper, "_run_trials", failing_trials)
        try:
            owner.start()
            for _ in range(500):
                if trials.in_flight:
                    break
                await asyncio.sleep(0.01)
            # A failure reports no help, so the worker's loop waits a poll first.
            assert await helper._help_with_trials() is False
        finally:
            held.set()
            await owner.stop()
            await owner_store.close()
            await helper_store.close()

    asyncio.run(run())
