"""Emitted Evals lifetime with native workflows; not Docker/product-quality proof."""

import asyncio
import importlib
import threading
import traceback
import warnings
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from functools import partial
from types import SimpleNamespace

import pytest

from cayu import (
    CorpusExecutionResult,
    DockerImageIdentity,
    Environment,
    EnvironmentSpec,
    Event,
    EventType,
    ExecutionProfileBehaviorIdentity,
    ModelPrice,
    ModelStreamEvent,
    PriceBook,
    WorkflowBase,
    WorkflowSpec,
    compare_eval_results,
    run_eval_plan,
    step,
)
from cayu.cli.project import project_context
from cayu.evals.corpus import (
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationSourceIdentityV1,
    TrialRequestSpec,
)
from tests.cli.test_scaffold_coding_budget import denial_policy
from tests.core.test_queued_session_messages import RecordingOneShotProvider
from tests.evals.test_workflow_eval_target import _register_app
from tests.qualification.repository_maintenance_case import SEED_FILES, materialize_seed_repository
from tests.qualification.repository_maintenance_toolchain import maintenance_toolchain
from tests.qualification.test_repository_maintenance_application import project as project
from tests.qualification.test_repository_maintenance_cost import _completion, _session
from tests.qualification.test_repository_maintenance_request import consumer as consumer

_PINS = {
    "application_release_id": "controlled-maintenance-eval",
    "implementation_revision": "sha256:" + "a" * 64,
    "result_projector_revision": "sha256:" + "b" * 64,
    "execution_scope_revision": "sha256:" + "c" * 64,
}


def _corpus(module, task):
    return module.maintenance_coding_corpus(
        source=EvaluationSourceIdentityV1(
            application_release_id="controlled-source",
            app_manifest_schema_version="7",
            app_manifest_fingerprint="a" * 64,
            evidence_revision="sha256:" + "b" * 64,
        ),
        reviewed_instruction=task.instruction,
    )


@pytest.fixture
def evaluation(consumer, tmp_path, monkeypatch):
    application, task, _provider, _domain, requests, _workflow = consumer
    accepted = asyncio.run(requests.capture_accepted_request(application, task))
    module = importlib.import_module("evals.maintenance")
    monkeypatch.setattr(
        module,
        "maintenance_eval_plan",
        partial(module.maintenance_eval_plan, source_directory=tmp_path),
    )
    stores = importlib.import_module("operations.maintenance_runs")
    reservations = stores.SQLiteMaintenanceRunStore(tmp_path / "eval-runs.sqlite")
    asyncio.run(reservations.initialize())
    deployments = []
    actions = []
    candidate_output = ["fixture"]
    monkeypatch.setattr(module, "fixture_candidate_output", candidate_output, raising=False)

    class FixtureWorkflow(WorkflowBase):
        spec = WorkflowSpec(name="repository-maintenance-coding")

        def __init__(self, application, task, *, accepted):
            super().__init__(application.app)
            self.task = task
            application.fixture_task = task

        async def run(self, session_id):
            assert self.task.parent_session_id == self.task.causal_budget_id == session_id
            ctx = self.context(session_id)
            yield await ctx.start()
            output = await step(ctx, agent="first", step_id="check", prompt="fixture")
            yield await ctx.completed(
                {
                    "verdict": "verified" if output.text == "fixture" else "rejected",
                    "product_run_id": self.task.product_run_id,
                    "result_digest": "d" * 64,
                }
            )

    class FixtureDeployment:
        def __init__(self, *, workspace_root):
            self.number = len(deployments)
            self.closed = False
            self.reservations = reservations
            app = _register_app(
                [[ModelStreamEvent.text_delta(candidate_output[0]), ModelStreamEvent.completed()]]
            )
            app.register_environment(
                Environment(
                    EnvironmentSpec(
                        name="coding",
                        execution_profile_identity=ExecutionProfileBehaviorIdentity(
                            name="tests:maintenance-eval-environment",
                            behavior_version="1",
                            implementation_version="1",
                        ),
                    )
                ),
                default=True,
            )
            self.application = SimpleNamespace(
                app=app, agent_name="first", project_root=workspace_root
            )
            deployments.append(self)

        async def validate_startup_schema(self):
            actions.append((self.number, "validate"))

        async def quiesce(self, *, timeout_s):
            assert not self.closed
            actions.append((self.number, "quiesce"))
            return True

        async def aclose(self, *, timeout_s):
            assert not self.closed
            self.closed = True
            actions.append((self.number, "close"))
            return True

    async def capture(application, task):
        return accepted.model_copy(update={"repository_root": str(application.project_root)})

    monkeypatch.setattr(module, "MaintenanceCodingWorkflow", FixtureWorkflow)
    monkeypatch.setattr(module, "capture_accepted_request", capture)
    return module, task, FixtureDeployment, deployments, actions


def test_native_repeated_trials_keep_distinct_owners_and_delay_dependency_close(evaluation):
    module, task, build, deployments, actions = evaluation

    async def scenario():
        scope = module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS)
        async with scope as plan:
            result = await run_eval_plan(
                plan,
                corpus=_corpus(module, task),
                suite_id="maintenance-coding",
            )
            assert type(result) is CorpusExecutionResult
            assert result.run.status == "passed", result.model_dump_json()
            assert len(deployments) == 3
            assert len({id(item.application.app) for item in deployments}) == 3
            assert not any(item.closed for item in deployments)
            assert actions == [
                (0, "validate"),
                (1, "validate"),
                (1, "quiesce"),
                (2, "validate"),
                (2, "quiesce"),
            ]
            trials = result.run.cases[0].trials
            assert len(trials) == 2
            assert all(
                trial.evidence_complete and trial.usage is not None and trial.usage.model_steps == 1
                for trial in trials
            )
            assert (
                deployments[1].application.fixture_task.parent_session_id
                != deployments[2].application.fixture_task.parent_session_id
            )
            observations = scope.attempts
            assert type(observations) is tuple and len(observations) == 2
            assert [item.trial_number for item in observations] == [1, 2]
            assert len({item.run_id for item in observations}) == 1
            assert all(item.suite_id == "maintenance-coding" for item in observations)
            assert all(item.case_id == "inclusive-endpoint" for item in observations)
            assert [item.workflow_run_id for item in observations] == [
                item.application.fixture_task.parent_session_id for item in deployments[1:]
            ]
            assert all(
                item.idempotency_key == f"cayu-eval:{item.workflow_run_id}" for item in observations
            )
            assert not any(hasattr(item, "messages") for item in observations)
            assert all(item.workflow_run_id not in repr(item) for item in observations)
            with pytest.raises(FrozenInstanceError):
                observations[0].trial_number = 2
        assert all(item.closed for item in deployments)
        assert scope.attempts == observations
        assert actions[-3:] == [(2, "close"), (1, "close"), (0, "close")]

    asyncio.run(scenario())


@pytest.mark.parametrize("rejected", [False, True])
def test_host_observation_retains_native_result_and_attempts(evaluation, rejected):
    module, task, build, deployments, _actions = evaluation
    if rejected:
        module.fixture_candidate_output[0] = "incorrect"

    async def scenario():
        scope = module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS)
        async with module.maintenance_corpus_execution(scope, _corpus(module, task)) as observed:
            result = observed.result
            assert result.run.status == ("failed" if rejected else "passed")
            assert len(result.run.cases[0].trials) == len(observed.attempts) == 2
            assert observed.attempts == scope.attempts
            assert all(a is not b for a, b in zip(observed.attempts, scope.attempts, strict=True))
            assert len({attempt.run_id for attempt in observed.attempts}) == 1
            assert [attempt.workflow_run_id for attempt in observed.attempts] == [
                deployment.application.fixture_task.parent_session_id
                for deployment in deployments[1:]
            ]
            assert observed.result == result and observed.result is not result
            assert not any(deployment.closed for deployment in deployments)
            assert all(
                attempt.workflow_run_id not in repr(observed) for attempt in observed.attempts
            )
            with pytest.raises(FrozenInstanceError):
                observed.result_json = "{}"
        assert all(deployment.closed for deployment in deployments)
        assert observed.result == result

    asyncio.run(scenario())


def test_host_observation_cancellation_before_result_retains_scope(evaluation):
    module, task, build, deployments, _actions = evaluation

    async def scenario():
        entered = asyncio.Event()

        def blocking_build(*, workspace_root):
            deployment = build(workspace_root=workspace_root)
            if deployment.number == 1:

                async def validate():
                    entered.set()
                    await asyncio.Event().wait()

                deployment.validate_startup_schema = validate
            return deployment

        scope = module.maintenance_eval_plan(task=task, build_deployment=blocking_build, **_PINS)

        async def execute():
            async with module.maintenance_corpus_execution(scope, _corpus(module, task)):
                pytest.fail("Cancelled execution yielded a host observation")

        owner = asyncio.create_task(execute())
        await asyncio.wait_for(entered.wait(), timeout=10)
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert owner.cancelled() and owner.cancelling() == 1
        assert len(scope.attempts) == 1
        assert len(scope.deployments) == 2
        assert all(deployment.closed for deployment in deployments)

    asyncio.run(scenario())


def test_host_observation_rejects_foreign_scope_before_execution(evaluation):
    module, task, _build, deployments, _actions = evaluation

    async def scenario():
        with pytest.raises(ValueError, match="requires its evaluation scope"):
            async with module.maintenance_corpus_execution(object(), _corpus(module, task)):
                pytest.fail("Foreign scope was accepted")
        assert not deployments

    asyncio.run(scenario())


@pytest.mark.parametrize("consumer", ["memory", "sqlite"], indirect=True)
def test_cohort_costs_include_failed_trials_children_and_missing_roots(
    evaluation, consumer, monkeypatch, capsys, caplog
):
    module, task, build, deployments, _actions = evaluation
    reader = consumer[0].app
    reader.budget_policy = denial_policy()
    pricing = reader.budget_policy.limits[0].pricing

    def failing_second_trial(*, workspace_root):
        if len(deployments) == 2:
            module.fixture_candidate_output[0] = "incorrect"
        return build(workspace_root=workspace_root)

    async def scenario():
        scope = module.maintenance_eval_plan(
            task=task, build_deployment=failing_second_trial, **_PINS
        )
        corpus = _corpus(module, task)
        async with module.maintenance_corpus_execution(scope, corpus) as observation:
            assert [trial.status for trial in observation.result.run.cases[0].trials] == [
                "passed",
                "failed",
            ]
            first, second = (item.workflow_run_id for item in observation.attempts)
            missing_all = await observation.inspect_costs(reader, corpus, pricing)
            assert missing_all["missing_roots"] == 2
            assert missing_all["known_estimated_total"] is None
            await _session(reader, first, first)
            await _session(reader, first + "-child", first)
            empty = await observation.inspect_costs(reader, corpus, pricing)
            assert empty["empty_roots"] == 1
            assert empty["missing_roots"] == 1
            assert Decimal(empty["known_estimated_total"]) == 0
            await _completion(reader, first, 2)
            await _completion(reader, first + "-child", 3)
            missing = await observation.inspect_costs(reader, corpus, pricing)
            assert missing["missing_roots"] == 1
            assert Decimal(missing["known_estimated_total"]) == 5
            await _session(reader, second, second)
            await _completion(reader, second, 7)
            report = await observation.inspect_costs(reader, corpus, pricing)
            assert report["expected_trials"] == report["observed_trials"] == 2
            assert report["missing_roots"] == report["unobserved_trials"] == 0
            assert report["session_count"] == 3
            assert Decimal(report["known_estimated_total"]) == 12
            assert report["billing_completeness"] == "not_established"
            assert "verified_completions" not in report
            duplicate = replace(observation, attempts=observation.attempts * 2)
            assert await duplicate.inspect_costs(reader, corpus, pricing) == report
            absent = replace(observation, attempts=())
            no_observations = await absent.inspect_costs(reader, corpus, pricing)
            assert no_observations["unobserved_trials"] == 2
            assert no_observations["known_estimated_total"] is None
            conflicting = replace(
                observation,
                attempts=(
                    *observation.attempts,
                    replace(observation.attempts[0], idempotency_key="other"),
                ),
            )
            with pytest.raises(ValueError, match="Invalid maintenance cohort cost inputs"):
                await conflicting.inspect_costs(reader, corpus, pricing)
            assert all(root not in str(report) for root in (first, second))
            unpriced = await observation.inspect_costs(
                reader,
                corpus,
                PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="unmatched",
                            model="unmatched",
                            input_per_million=Decimal(1),
                            output_per_million=Decimal(1),
                        ),
                    )
                ),
            )
            assert unpriced["unpriced_line_items"] == 3
            assert Decimal(unpriced["known_estimated_total"]) == 0
            await reader.session_store.append_event(
                second,
                Event(
                    type=EventType.MODEL_HOSTED_TOOL_CALL,
                    session_id=second,
                    payload={
                        "tool_type": "web_search",
                        "call_id": "private-unknown-search",
                        "status": "outcome_unknown",
                        "provider_name": "unpriced-provider",
                        "model": "unpriced-model",
                    },
                ),
            )
            unknown = await observation.inspect_costs(reader, corpus, pricing)
            assert unknown["unknown_hosted_calls"] == 1
            assert unknown["unpriced_line_items"] == 1
            assert Decimal(unknown["known_estimated_total"]) == 12
            assert "private-unknown-search" not in str(unknown)
            original = reader.get_causal_budget_cost
            first_id = reader.project_causal_budget_id_for_exposure(first, session_ids=(first,))
            first_summary = await original(first_id, pricing, currency="USD")

            for changes in (
                {"trial_number": True},
                {"trial_number": 0},
                {"trial_number": 3},
                {"suite_id": "another-suite"},
                {"case_id": "another-case"},
                {"run_id": "another-run"},
            ):
                invalid = replace(
                    observation,
                    attempts=(
                        replace(observation.attempts[0], **changes),
                        observation.attempts[1],
                    ),
                )
                with pytest.raises(ValueError, match="Invalid maintenance cohort cost inputs"):
                    await invalid.inspect_costs(reader, corpus, pricing)

            for amount in (Decimal("NaN"), Decimal("Infinity"), Decimal(-1), Decimal("1e129")):

                async def invalid_amount(causal_id, _pricing, *, currency, amount=amount):
                    return first_summary.model_copy(
                        update={"causal_budget_id": causal_id, "total_cost": amount}
                    )

                monkeypatch.setattr(reader, "get_causal_budget_cost", invalid_amount)
                with pytest.raises(ValueError, match="Invalid maintenance cohort cost evidence"):
                    await observation.inspect_costs(reader, corpus, pricing)

            class Secret:
                def __repr__(self):
                    return "cohort-private-secret-canary"

                __str__ = __repr__

            with warnings.catch_warnings(record=True) as captured_warnings:
                malformed = replace(
                    observation,
                    attempts=(replace(observation.attempts[0], case_id=Secret()),),
                )
                with pytest.raises(
                    ValueError, match="Invalid maintenance cohort cost inputs"
                ) as error:
                    await malformed.inspect_costs(reader, corpus, pricing)
                assert "cohort-private-secret-canary" not in "".join(
                    traceback.format_exception(error.value)
                )

                async def malformed_summary(causal_id, _pricing, *, currency):
                    session = first_summary.session_costs[0]
                    line = session.line_items[0].model_copy(update={"provider_name": Secret()})
                    return first_summary.model_copy(
                        update={
                            "causal_budget_id": causal_id,
                            "session_costs": (
                                session.model_copy(update={"line_items": (line,)}),
                                *first_summary.session_costs[1:],
                            ),
                        }
                    )

                monkeypatch.setattr(reader, "get_causal_budget_cost", malformed_summary)
                with pytest.raises(
                    ValueError, match="Invalid maintenance cohort cost evidence"
                ) as error:
                    await observation.inspect_costs(reader, corpus, pricing)
                assert "cohort-private-secret-canary" not in "".join(
                    traceback.format_exception(error.value)
                )
            output = capsys.readouterr()
            assert "cohort-private-secret-canary" not in output.out + output.err + caplog.text
            assert not captured_warnings

            async def overlapping(causal_id, _pricing, *, currency):
                return first_summary.model_copy(update={"causal_budget_id": causal_id}, deep=True)

            monkeypatch.setattr(reader, "get_causal_budget_cost", overlapping)
            overlapped = await observation.inspect_costs(reader, corpus, pricing)
            assert overlapped["session_count"] == 2
            assert Decimal(overlapped["known_estimated_total"]) == 5

            async def conflicting_summary(causal_id, _pricing, *, currency):
                value = await overlapping(causal_id, _pricing, currency=currency)
                if causal_id != first_id:
                    session = value.session_costs[0]
                    line = session.line_items[0].model_copy(update={"provider_name": "changed"})
                    value = value.model_copy(
                        update={
                            "session_costs": (
                                session.model_copy(update={"line_items": (line,)}),
                                *value.session_costs[1:],
                            )
                        }
                    )
                return value

            monkeypatch.setattr(reader, "get_causal_budget_cost", conflicting_summary)
            with pytest.raises(ValueError, match="Conflicting maintenance cohort cost"):
                await observation.inspect_costs(reader, corpus, pricing)

            reading = asyncio.Event()

            async def blocked_read(causal_id, _pricing, *, currency):
                if causal_id != first_id:
                    reading.set()
                    await asyncio.Event().wait()
                return await original(causal_id, _pricing, currency=currency)

            monkeypatch.setattr(reader, "get_causal_budget_cost", blocked_read)
            owner = asyncio.create_task(observation.inspect_costs(reader, corpus, pricing))
            await asyncio.wait_for(reading.wait(), timeout=10)
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner
            assert owner.cancelled() and owner.cancelling() == 1

    asyncio.run(scenario())


def test_invalid_pins_close_reference_without_starting_trial(evaluation):
    module, task, build, deployments, actions = evaluation

    async def scenario():
        with pytest.raises(ValueError):
            async with module.maintenance_eval_plan(
                task=task, build_deployment=build, **{**_PINS, "implementation_revision": "bad"}
            ):
                pytest.fail("Invalid target was accepted")
        assert len(deployments) == 1 and deployments[0].closed
        assert actions == [(0, "validate"), (0, "close")]

    asyncio.run(scenario())


def test_repeated_trials_prepare_new_seeds_and_retain_prior_outputs(evaluation):
    module, task, build, deployments, _actions = evaluation
    roots = []
    original = SEED_FILES["range_ops.py"]
    repaired = original.replace("lower <= value < upper", "lower <= value <= upper")

    def fresh_deployment(*, workspace_root):
        assert (workspace_root / "range_ops.py").read_text() == original
        assert workspace_root not in roots
        # Stand in for source finalization here; actual product coverage is separate.
        if len(roots) == 2:
            assert (roots[1] / "range_ops.py").read_text() == repaired
        roots.append(workspace_root)
        deployment = build(workspace_root=workspace_root)
        if len(roots) == 2:
            (workspace_root / "range_ops.py").write_text(repaired)
        return deployment

    async def scenario():
        scope = module.maintenance_eval_plan(task=task, build_deployment=fresh_deployment, **_PINS)
        async with scope as plan:
            result = await run_eval_plan(
                plan, corpus=_corpus(module, task), suite_id="maintenance-coding"
            )
            assert type(result) is CorpusExecutionResult
            assert result.run.status == "passed", result.model_dump_json()
        assert len(scope.sources) == len(deployments) == 3
        assert tuple(source.root for source in scope.sources) == tuple(roots)
        assert all(source.prepared for source in scope.sources)
        assert (roots[0] / "range_ops.py").read_text() == original
        assert (roots[1] / "range_ops.py").read_text() == repaired
        assert (roots[2] / "range_ops.py").read_text() == original
        for source in scope.sources:
            await source.wait_prepared()

    asyncio.run(scenario())


def test_cancelled_seed_preparation_waits_for_dispatched_writer(evaluation, monkeypatch):
    module, task, build, deployments, _actions = evaluation
    materialize = module.materialize_seed_repository
    release = threading.Event()

    async def scenario():
        dispatched = asyncio.Event()
        loop = asyncio.get_running_loop()

        def blocked(root):
            loop.call_soon_threadsafe(dispatched.set)
            assert release.wait(10), "Test did not release the seed writer"
            return materialize(root)

        monkeypatch.setattr(module, "materialize_seed_repository", blocked)
        scope = module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS)

        async def enter():
            async with scope:
                pytest.fail("Cancelled preparation must not yield a plan")

        owner = asyncio.create_task(enter())
        try:
            await asyncio.wait_for(dispatched.wait(), 5)
            assert len(scope.sources) == 1 and not scope.sources[0].prepared
            owner.cancel("stop-source-preparation")
            await asyncio.sleep(0)
            assert owner.cancelling() == 1 and not owner.done()
            assert deployments == []
            release.set()
            with pytest.raises(asyncio.CancelledError, match="stop-source-preparation"):
                await asyncio.wait_for(owner, 5)
            assert owner.cancelled() and owner.cancelling() == 1
            assert deployments == []
            assert scope.sources[0].prepared
            await scope.sources[0].wait_prepared()
            assert (scope.sources[0].root / "range_ops.py").read_text() == SEED_FILES[
                "range_ops.py"
            ]
        finally:
            release.set()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("after_commit", [False, True])
def test_failed_seed_is_retained_without_constructing_application(
    evaluation, monkeypatch, after_commit
):
    module, task, build, deployments, _actions = evaluation
    materialize = module.materialize_seed_repository
    failure = RuntimeError("controlled seed preparation failure")

    def fail(root):
        if after_commit:
            materialize(root)
        raise failure

    monkeypatch.setattr(module, "materialize_seed_repository", fail)

    async def scenario():
        scope = module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS)
        with pytest.raises(RuntimeError) as caught:
            async with scope:
                pytest.fail("Failed seed must not authorize an application")
        assert caught.value is failure
        assert deployments == [] and len(scope.sources) == 1
        source = scope.sources[0]
        assert not source.prepared
        assert source.root.parent.is_dir()
        assert source.root.exists() is after_commit
        with pytest.raises(RuntimeError) as replay:
            await source.wait_prepared()
        assert replay.value is failure

    asyncio.run(scenario())


def test_factory_cannot_reuse_reference_source_for_trial(evaluation):
    module, task, build, deployments, _actions = evaluation

    def wrong_root(*, workspace_root):
        if deployments:
            workspace_root = deployments[0].application.project_root
        return build(workspace_root=workspace_root)

    async def scenario():
        scope = module.maintenance_eval_plan(task=task, build_deployment=wrong_root, **_PINS)
        async with scope as plan:
            result = await run_eval_plan(
                plan, corpus=_corpus(module, task), suite_id="maintenance-coding"
            )
            assert type(result) is CorpusExecutionResult
            assert result.run.status == "error"
            assert all(
                trial.code == "workflow_target_failed" for trial in result.run.cases[0].trials
            )
            assert all(not hasattr(item.application, "fixture_task") for item in deployments)
        assert len(scope.sources) == 3 and all(source.prepared for source in scope.sources)
        assert len({source.root for source in scope.sources}) == 3
        assert all(item.closed for item in deployments)

    asyncio.run(scenario())


def test_unexpected_seed_commit_never_authorizes_application(evaluation, monkeypatch):
    module, task, build, deployments, _actions = evaluation
    materialize = module.materialize_seed_repository

    def unexpected_revision(root):
        materialize(root)
        return "0" * 40

    monkeypatch.setattr(module, "materialize_seed_repository", unexpected_revision)

    async def scenario():
        scope = module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS)
        with pytest.raises(ValueError, match="unexpected base revision"):
            async with scope:
                pytest.fail("Unproven seed must not authorize an application")
        assert not deployments and len(scope.sources) == 1
        source = scope.sources[0]
        assert source.root.is_dir() and not source.prepared
        with pytest.raises(ValueError, match="unexpected base revision"):
            await source.wait_prepared()

    asyncio.run(scenario())


def test_cancelled_trial_construction_retains_native_observation(evaluation, monkeypatch):
    module, task, build, deployments, _actions = evaluation
    materialize = module.materialize_seed_repository
    release = threading.Event()

    async def scenario():
        dispatched = asyncio.Event()
        loop = asyncio.get_running_loop()
        scope = module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS)
        async with scope as plan:

            def blocked(root):
                loop.call_soon_threadsafe(dispatched.set)
                assert release.wait(10), "Test did not release the trial seed writer"
                return materialize(root)

            monkeypatch.setattr(module, "materialize_seed_repository", blocked)
            owner = asyncio.create_task(
                run_eval_plan(plan, corpus=_corpus(module, task), suite_id="maintenance-coding")
            )
            try:
                await asyncio.wait_for(dispatched.wait(), 5)
                assert len(scope.attempts) == 1
                observation = scope.attempts[0]
                assert observation.trial_number == 1
                assert observation.idempotency_key == f"cayu-eval:{observation.workflow_run_id}"
                owner.cancel("stop-observed-trial")
                await asyncio.sleep(0)
                assert owner.cancelling() == 1 and not owner.done()
                assert len(deployments) == 1
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(owner, 5)
                assert owner.cancelled() and owner.cancelling() == 1
                assert len(deployments) == 1
                assert scope.attempts == (observation,)
                assert scope.sources[1].prepared
            finally:
                release.set()
                await asyncio.gather(owner, return_exceptions=True)
        assert scope.attempts == (observation,)

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["validate_startup_schema", "quiesce"])
def test_trial_failure_retains_every_deployment_for_cleanup(evaluation, monkeypatch, phase):
    module, task, build, deployments, actions = evaluation
    original = getattr(build, phase)

    async def fail_trial(self, **kwargs):
        if self.number:
            raise RuntimeError("controlled trial failure")
        return await original(self, **kwargs)

    monkeypatch.setattr(build, phase, fail_trial)

    async def scenario():
        async with module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS) as plan:
            result = await run_eval_plan(
                plan,
                corpus=_corpus(module, task),
                suite_id="maintenance-coding",
            )
            assert type(result) is CorpusExecutionResult
            assert result.run.status == "error"
            assert len(deployments) == 3 and not any(item.closed for item in deployments)
        assert actions[-3:] == [(2, "close"), (1, "close"), (0, "close")]

    asyncio.run(scenario())


def test_context_cancellation_waits_for_owned_cleanup(evaluation, monkeypatch):
    module, task, build, deployments, _actions = evaluation

    async def scenario():
        entered, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original = build.aclose

        async def close(self, **kwargs):
            closing.set()
            await release.wait()
            return await original(self, **kwargs)

        monkeypatch.setattr(build, "aclose", close)

        async def owner():
            async with module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS):
                entered.set()
                await asyncio.Event().wait()

        work = asyncio.create_task(owner())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            work.cancel("eval-owner-stop")
            await asyncio.wait_for(closing.wait(), 5)
            assert work.cancelling() == 1 and not work.done()
            assert not deployments[0].closed
            release.set()
            with pytest.raises(asyncio.CancelledError, match="eval-owner-stop"):
                await work
            assert work.cancelled() and work.cancelling() == 1
            assert deployments[0].closed
        finally:
            release.set()
            await asyncio.gather(work, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("overflow", ["concurrency", "timeout", "trials"])
def test_native_corpus_ceiling_rejects_before_trial_construction(evaluation, overflow):
    module, task, build, deployments, _actions = evaluation

    async def scenario():
        async with module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS) as plan:
            corpus = _corpus(module, task)
            if overflow != "concurrency":
                suite = EvalSuiteSpec.create(
                    id="maintenance-coding",
                    name="Invalid exposure",
                    trial_request=TrialRequestSpec(
                        trials=3 if overflow == "trials" else 2,
                        timeout_seconds=181 if overflow == "timeout" else 180,
                    ),
                )
                corpus = EvalCorpusDocument.create(
                    target_key=corpus.target_key,
                    evidence_policy=corpus.evidence_policy,
                    suites=(suite,),
                    cases=corpus.cases,
                )
            with pytest.raises(ValueError):
                await run_eval_plan(
                    plan,
                    corpus=corpus,
                    suite_id="maintenance-coding",
                    max_concurrency=2 if overflow == "concurrency" else 1,
                )
            assert len(deployments) == 1
        assert deployments[0].closed

    asyncio.run(scenario())


def test_changed_input_rejects_without_constructing_trial(evaluation):
    module, task, build, deployments, _actions = evaluation

    async def scenario():
        scope = module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS)
        async with scope as plan:
            changed = _corpus(module, replace(task, instruction="different task"))
            result = await run_eval_plan(plan, corpus=changed, suite_id="maintenance-coding")
            assert type(result) is CorpusExecutionResult
            assert result.run.status == "error"
            assert len(deployments) == 1
            assert len(scope.attempts) == 2
            assert [item.trial_number for item in scope.attempts] == [1, 2]
            assert len({item.workflow_run_id for item in scope.attempts}) == 2
        assert deployments[0].closed
        assert len(scope.attempts) == 2

    asyncio.run(scenario())


def test_native_comparison_detects_seeded_regression_on_same_fixed_corpus(evaluation):
    module, task, build, deployments, _actions = evaluation

    async def scenario():
        corpus = _corpus(module, task)
        async with module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS) as plan:
            baseline = await run_eval_plan(plan, corpus=corpus, suite_id="maintenance-coding")
        module.fixture_candidate_output[0] = "wrong"
        candidate_pins = {
            **_PINS,
            "application_release_id": "controlled-regressed-maintenance-eval",
        }
        async with module.maintenance_eval_plan(
            task=task,
            build_deployment=build,
            **candidate_pins,
        ) as plan:
            current = await run_eval_plan(plan, corpus=corpus, suite_id="maintenance-coding")
        assert type(baseline) is CorpusExecutionResult
        assert type(current) is CorpusExecutionResult
        assert baseline.run.status == "passed" and current.run.status == "failed"
        comparison = compare_eval_results(baseline, current)
        assert comparison.compatibility.comparable
        assert comparison.regressions
        assert all(item.closed for item in deployments)

    asyncio.run(scenario())


def test_failed_cleanup_preserves_all_original_errors_and_reachable_owners(evaluation, monkeypatch):
    module, task, build, deployments, actions = evaluation
    primary = ValueError("controlled primary")
    cleanup_failures = [
        RuntimeError("reference close"),
        ExceptionGroup("nested trial close", [LookupError("first"), TypeError("second")]),
        OSError("last trial close"),
    ]

    async def fail(self, **kwargs):
        actions.append((self.number, "failed-close"))
        raise cleanup_failures[self.number]

    monkeypatch.setattr(build, "aclose", fail)

    async def scenario():
        scope = module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS)
        with pytest.raises(ExceptionGroup) as caught:
            async with scope as plan:
                result = await run_eval_plan(
                    plan, corpus=_corpus(module, task), suite_id="maintenance-coding"
                )
                assert type(result) is CorpusExecutionResult and result.run.status == "passed"
                raise primary
        assert scope.deployments == tuple(deployments)
        assert not any(item.closed for item in scope.deployments)
        assert actions[-3:] == [(2, "failed-close"), (1, "failed-close"), (0, "failed-close")]
        outer = caught.value
        assert outer.exceptions[0] is primary
        cleanup = outer.exceptions[1]
        assert isinstance(cleanup, ExceptionGroup)
        assert cleanup.exceptions == tuple(reversed(cleanup_failures))

    asyncio.run(scenario())


def test_cancelled_eval_preserves_prior_and_deployment_cleanup_failures(evaluation, monkeypatch):
    module, task, build, deployments, actions = evaluation
    prior = ExceptionGroup("earlier cleanup", [OSError("retained stream cleanup")])
    closing = [RuntimeError(f"deployment {number} close failure") for number in range(3)]

    async def fail(self, **kwargs):
        actions.append((self.number, "failed-close"))
        raise closing[self.number]

    monkeypatch.setattr(build, "aclose", fail)

    async def scenario():
        scope = module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS)
        entered, release = asyncio.Event(), asyncio.Event()

        async def run():
            async with scope as plan:
                result = await run_eval_plan(
                    plan, corpus=_corpus(module, task), suite_id="maintenance-coding"
                )
                assert type(result) is CorpusExecutionResult and result.run.status == "passed"
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError as cancellation:
                    # Represent a prior stream-cleanup failure without replacing
                    # the cancellation delivered by Task.cancel().
                    raise cancellation from prior

        owner = asyncio.create_task(run())
        try:
            await asyncio.wait_for(entered.wait(), 10)
            owner.cancel("stop-evaluation")
            with pytest.raises(asyncio.CancelledError) as caught:
                await owner
            assert owner.cancelled() and owner.cancelling() == 1
            cause = caught.value.__cause__
            assert isinstance(cause, ExceptionGroup)
            assert cause.exceptions[0] is prior
            later = cause.exceptions[1]
            assert isinstance(later, ExceptionGroup)
            assert later.exceptions == tuple(reversed(closing))
            assert scope.deployments == tuple(deployments)
            assert actions[-3:] == [(2, "failed-close"), (1, "failed-close"), (0, "failed-close")]
            assert not any(item.closed for item in deployments)
        finally:
            release.set()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("body_fails", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_cancellation_during_multi_deployment_cleanup_remains_cancellation(
    evaluation, monkeypatch, body_fails, cleanup_fails
):
    module, task, build, deployments, _actions = evaluation
    primary = ValueError("body failed")
    nested = ExceptionGroup("nested close", [LookupError("first"), TypeError("second")])
    last = OSError("reference close failed")
    original = build.aclose

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        closed = []

        async def close(self, **kwargs):
            closed.append(self.number)
            if self.number == 2:
                entered.set()
                await release.wait()
            elif cleanup_fails:
                raise nested if self.number == 1 else last
            return await original(self, **kwargs)

        monkeypatch.setattr(build, "aclose", close)
        scope = module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS)

        async def run():
            async with scope as plan:
                result = await run_eval_plan(
                    plan, corpus=_corpus(module, task), suite_id="maintenance-coding"
                )
                assert type(result) is CorpusExecutionResult
                assert result.run.status == "passed"
                if body_fails:
                    raise primary

        owner = asyncio.create_task(run())
        try:
            await asyncio.wait_for(entered.wait(), 10)
            owner.cancel("stop-during-cleanup")
            await asyncio.sleep(0)
            assert owner.cancelling() == 1 and not owner.done()
            assert closed == [2] and not any(item.closed for item in deployments)
            release.set()
            with pytest.raises(asyncio.CancelledError, match="stop-during-cleanup") as caught:
                await owner
            assert owner.cancelled() and owner.cancelling() == 1
            assert closed == [2, 1, 0]
            assert scope.deployments == tuple(deployments)
            cause = caught.value.__cause__
            if body_fails and cleanup_fails:
                assert isinstance(cause, ExceptionGroup)
                assert cause.exceptions[0] is primary
                cause = cause.exceptions[1]
            elif body_fails:
                assert cause is primary
            if cleanup_fails:
                assert isinstance(cause, ExceptionGroup)
                assert cause.exceptions == (nested, last)
                assert [item.closed for item in deployments] == [False, False, True]
            else:
                assert all(item.closed for item in deployments)
                if not body_fails:
                    assert cause is None
        finally:
            release.set()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


def test_failed_startup_and_cleanup_still_expose_reference_owner(evaluation, monkeypatch):
    module, task, build, deployments, _actions = evaluation
    startup = ValueError("startup failed")
    cleanup = RuntimeError("cleanup failed")

    async def fail_startup(self):
        raise startup

    async def fail_cleanup(self, **kwargs):
        raise cleanup

    monkeypatch.setattr(build, "validate_startup_schema", fail_startup)
    monkeypatch.setattr(build, "aclose", fail_cleanup)

    async def scenario():
        scope = module.maintenance_eval_plan(task=task, build_deployment=build, **_PINS)
        with pytest.raises(ExceptionGroup) as caught:
            async with scope:
                pytest.fail("Startup unexpectedly succeeded")
        assert scope.deployments == tuple(deployments) and len(scope.deployments) == 1
        assert caught.value.exceptions == (startup, cleanup)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite"])
def test_generated_profile_is_not_authority_for_source_path_or_seed_bytes(
    project, tmp_path, monkeypatch, backend
):
    """Actual generated profile/SQLite reads, without model or Docker dispatch."""
    assert backend == "sqlite"
    first_root, second_root = tmp_path / "first-seed", tmp_path / "second-seed"
    assert materialize_seed_repository(first_root) == materialize_seed_repository(second_root)
    profile = maintenance_toolchain(
        image_identity=DockerImageIdentity(reference="fixture@sha256:" + "a" * 64),
        architecture="amd64",
        build_context_sha256="sha256:" + "b" * 64,
    )
    provider = RecordingOneShotProvider()
    with project_context(project):
        operations = importlib.import_module("operations.coding")
        root_module = importlib.import_module("app")
        requests = importlib.import_module("operations.maintenance_requests")
        task_type = importlib.import_module("domain.coding_product").CodingProductTask
        monkeypatch.setattr(
            operations, "_configured_docker_authority", lambda root: (profile, "/usr/bin/docker")
        )
        applications = []

        async def scenario():
            try:
                for root in (first_root, second_root):
                    applications.append(
                        root_module.build_coding_product_application(
                            provider=provider,
                            workspace_root=root,
                        )
                    )
                task = task_type(
                    product_run_id="profile-probe",
                    session_id="profile-probe-child",
                    task_id="profile-probe-task",
                    instruction="Repair the inclusive endpoint.",
                )
                first = await requests.capture_accepted_request(applications[0], task)
                second = await requests.capture_accepted_request(applications[1], task)
                assert first.repository_root == str(first_root)
                assert second.repository_root == str(second_root)
                assert first.execution_profile_fingerprint == second.execution_profile_fingerprint
                assert first.model_copy(update={"repository_root": str(second_root)}) == second
                # Only test-owned data changes. A configuration fingerprint cannot
                # prove the source still contains the original seeded bug.
                (first_root / "range_ops.py").write_text(
                    SEED_FILES["range_ops.py"].replace("value < upper", "value <= upper")
                )
                assert await requests.capture_accepted_request(applications[0], task) == first
                assert (second_root / "range_ops.py").read_text() == SEED_FILES["range_ops.py"]
                assert not provider.requests
            finally:
                for application in reversed(applications):
                    await application.app.session_store.close()
                    await application.app.task_store.close()
                    await application.app.knowledge_store.close()

        asyncio.run(scenario())
