"""Actual generated coding-product execution under the existing workflow evaluator."""

import importlib
from dataclasses import replace
from decimal import Decimal

import pytest

from cayu import (
    ChildSessionCompleted,
    CodingProductArtifactRepository,
    CorpusExecutionResult,
    EvalCase,
    EvalStatus,
    EvalSuite,
    EventQuery,
    ExecutionProfileBehaviorIdentity,
    FinalOutputEqualsAssertionSpec,
    FinalOutputMatches,
    Message,
    ModelPrice,
    PriceBook,
    RunRequest,
    SavedWorkflowEvalCapture,
    SavedWorkflowEvalScore,
    ScriptedModelProvider,
    SessionTrajectoryError,
    WorkflowEvalExecution,
    WorkflowEvalInstanceScope,
    WorkflowEvalResult,
    WorkflowEvalTarget,
    capture_workflow_eval_attempt,
    current_execution_deadline,
    eval_run_to_json,
    load_eval_run,
    run_workflow_eval_suite,
    score_workflow_eval_capture,
    write_eval_run_json,
)
from cayu.evals.corpus import EvaluationEvidencePolicySpec, EvaluationSourceIdentityV1
from tests.qualification.repository_maintenance_case import SEED_FILES


async def exercise_repeated_coding_eval(
    build_application, task, scripts, *, directory, expected_model_steps, dirty_first=False
):
    """Actual product through the emitted two-trial target, with local runner/SQLite.

    The test-only deployment adapts schema/closure to SQLite; production remains
    PostgreSQL-only. Source preparation, target, workflow, native drains and
    independent product verification are not replaced.
    """
    evaluation = importlib.import_module("evals.maintenance")
    deployment_module = importlib.import_module("operations.maintenance_deployment")
    reservations_module = importlib.import_module("operations.maintenance_runs")
    reservations = reservations_module.SQLiteMaintenanceRunStore(
        directory / "repeated-evals.sqlite"
    )
    await reservations.initialize()
    providers = []
    closed = []

    class OwnedScriptedProvider(ScriptedModelProvider):
        @property
        def execution_profile_identity(self):
            return ExecutionProfileBehaviorIdentity(
                name="tests:maintenance-repeated-coding-provider",
                behavior_version="1",
                implementation_version="1",
            )

        async def aclose(self):
            closed.append(self)

    class SQLiteDeployment(deployment_module.MaintenanceDeployment):
        def _stores(self):
            app = self.application.app
            return app.session_store, app.task_store, app.knowledge_store

        async def validate_startup_schema(self):
            # Built-in SQLite stores validate synchronously during construction;
            # ensure_schema is the asynchronous PostgreSQL contract only.
            await self.reservations.check_ready()

    def build(*, workspace_root):
        assert (workspace_root / "range_ops.py").read_text() == SEED_FILES["range_ops.py"]
        if dirty_first and len(providers) == 1:
            # Alter a prepared input before admission, without changing its profile.
            (workspace_root / "range_ops.py").write_text(
                SEED_FILES["range_ops.py"] + "\n# changed after preparation\n"
            )
        provider = OwnedScriptedProvider(scripts)
        providers.append(provider)
        application, _sessions, _tasks = build_application(
            workspace_root=workspace_root,
            model_provider=provider,
        )
        return SQLiteDeployment(application=application, reservations=reservations)

    revision = "sha256:" + "1" * 64
    scope = evaluation.maintenance_eval_plan(
        source_directory=directory,
        task=task,
        build_deployment=build,
        application_release_id="maintenance-local-repeated-integration",
        implementation_revision=revision,
        result_projector_revision=revision,
        execution_scope_revision=revision,
    )
    corpus = evaluation.maintenance_coding_corpus(
        source=EvaluationSourceIdentityV1(
            application_release_id="controlled-source",
            app_manifest_schema_version="7",
            app_manifest_fingerprint="a" * 64,
            evidence_revision=revision,
        ),
        reviewed_instruction=task.instruction,
    )
    async with evaluation.maintenance_corpus_execution(scope, corpus) as host_observation:
        result = host_observation.result
        assert type(result) is CorpusExecutionResult
        assert result.run.status == ("error" if dirty_first else "passed"), result.run.cases[
            0
        ].model_dump_json(indent=2)
        trials = result.run.cases[0].trials
        assert len(trials) == 2
        successful = trials[1:] if dirty_first else trials
        if dirty_first:
            assert trials[0].code == "workflow_execution_failed"
            assert not providers[1].requests
        assert all(
            trial.output.text == "verified" and trial.evidence_complete for trial in successful
        )
        assert all(
            trial.usage is not None and trial.usage.model_steps == expected_model_steps
            for trial in successful
        )
        assert len(scope.deployments) == len(scope.sources) == len(providers) == 3
        assert len(scope.attempts) == 2
        assert [item.trial_number for item in scope.attempts] == [1, 2]
        assert len({item.workflow_run_id for item in scope.attempts}) == 2
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="unmatched-fixture",
                    model="unmatched-fixture",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        # All generated applications share the configured SQLite store and alias
        # namespace. Read the entire cohort through one still-open application.
        cohort = await host_observation.inspect_costs(
            scope.deployments[0].application.app, corpus, pricing
        )
        assert cohort["expected_trials"] == cohort["observed_trials"] == 2
        assert cohort["missing_roots"] == cohort["unobserved_trials"] == 0
        assert cohort["session_count"] == (3 if dirty_first else 4)
        assert cohort["empty_roots"] == int(dirty_first)
        assert cohort["unpriced_line_items"] == expected_model_steps * (1 if dirty_first else 2)
        assert Decimal(cohort["known_estimated_total"]) == 0
        assert cohort["billing_completeness"] == "not_established"
        assert all(item.workflow_run_id not in str(cohort) for item in scope.attempts)
        # Read back each native root (including a failed trial) through Runtime's
        # causal discovery. Missing pricing remains explicit, not free execution.
        for index, observation in enumerate(scope.attempts):
            app = scope.deployments[index + 1].application.app
            causal_id = app.project_causal_budget_id_for_exposure(
                observation.workflow_run_id, session_ids=(observation.workflow_run_id,)
            )
            cost = await app.get_causal_budget_cost(
                causal_id,
                pricing,
            )
            expected_steps = 0 if dirty_first and index == 0 else expected_model_steps
            assert cost.model_steps == expected_steps
            assert cost.unpriced_model_steps == expected_steps
            assert cost.session_count == (1 if dirty_first and index == 0 else 2)
            assert cost.priced_model_steps == 0
        assert not providers[0].requests
        active_providers = providers[2:] if dirty_first else providers[1:]
        assert all(len(provider.requests) == expected_model_steps for provider in active_providers)
        assert not closed
        roots = [source.root for source in scope.sources]
        assert len(set(roots)) == 3
        assert all(source.prepared for source in scope.sources)
        original = SEED_FILES["range_ops.py"]
        assert (roots[0] / "range_ops.py").read_text() == original
        repaired = original.replace("lower <= value < upper", "lower <= value <= upper")
        repaired_roots = roots[2:] if dirty_first else roots[1:]
        assert all((root / "range_ops.py").read_text() == repaired for root in repaired_roots)
        if dirty_first:
            assert (
                roots[1] / "range_ops.py"
            ).read_text() == original + "\n# changed after preparation\n"
    assert closed == list(reversed(providers))
    assert all(source.root.is_dir() for source in scope.sources)


async def exercise_coding_eval(
    application, task, expected_model_steps, *, reservation_path, monkeypatch, rejected=False
):
    workflow_type = importlib.import_module(
        "workflows.maintenance_coding"
    ).MaintenanceCodingWorkflow
    linked_tasks = []
    reservations = []
    identity_module = importlib.import_module("domain.maintenance_identity")
    store_module = importlib.import_module("operations.maintenance_runs")
    request_domain = importlib.import_module("domain.maintenance_request")
    requests = importlib.import_module("operations.maintenance_requests")
    store = store_module.SQLiteMaintenanceRunStore(reservation_path)
    await store.initialize()

    async def factory(invocation):
        assert invocation.messages == tuple(request.messages)
        deadline = current_execution_deadline()
        assert deadline.expires_at is not None
        accepted = await requests.capture_accepted_request(application, task)
        reserved = await store.reserve(
            identity_module.MaintenanceRunIntent(
                tenant="qualification-tenant",
                subject="qualification-evaluator",
                idempotency_key=invocation.idempotency_key,
                request_json=request_domain.encode_request(accepted),
            ),
            workflow_session_id=invocation.workflow_run_id,
            coding_expires_at=deadline.expires_at.isoformat(),
        )
        reservations.append(reserved)
        linked = replace(
            task,
            product_run_id=reserved.product_run_id,
            session_id=reserved.session_id,
            task_id=reserved.task_id,
            parent_session_id=reserved.workflow_session_id,
            causal_budget_id=reserved.workflow_session_id,
        )
        linked_tasks.append(linked)
        return WorkflowEvalExecution(
            app=application.app,
            workflow=workflow_type(
                application,
                linked,
                accepted=request_domain.decode_request(reserved.intent.request_json),
            ),
        )

    request = RunRequest(
        agent_name=application.agent_name,
        environment_name="coding",
        messages=[Message.text("user", task.instruction)],
    )
    # Qualification identities, not a pinned production release or live report.
    revision = "sha256:" + "1" * 64
    target = WorkflowEvalTarget(
        key="maintenance-coding",
        app=application.app,
        request_base=request.model_copy(update={"messages": []}),
        application_release_id="maintenance-local-integration",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        workflow_spec=workflow_type.spec,
        implementation_revision=revision,
        result_projector_revision=revision,
        execution_scope_revision=revision,
        instance_scope=WorkflowEvalInstanceScope.SHARED,
        workflow_factory=factory,
        result_projector=lambda evidence: WorkflowEvalResult(
            final_output=evidence.completion_event.payload["verdict"],
            structured_output={
                "product_run_id": evidence.completion_event.payload["product_run_id"],
                "result_digest": evidence.completion_event.payload["result_digest"],
                "verdict": evidence.completion_event.payload["verdict"],
            },
        ),
    )
    result = await run_workflow_eval_suite(
        target,
        EvalSuite(
            id="maintenance-coding",
            cases=[
                EvalCase(
                    id="inclusive-endpoint",
                    request=request,
                    assertions=[
                        ChildSessionCompleted(min_count=1),
                        FinalOutputMatches("^verified$"),
                    ],
                )
            ],
        ),
        retain_trajectory=True,
        case_timeout_seconds=180,
    )
    expected_status = EvalStatus.FAILED if rejected else EvalStatus.PASSED
    assert result.status is expected_status, result.model_dump_json(indent=2)
    trial = result.cases[0].trials[0]
    assert len(linked_tasks) == 1
    assert len(reservations) == 1
    reserved = reservations[0]
    assert (
        await store_module.SQLiteMaintenanceRunStore(reservation_path).load_owned(
            tenant=reserved.intent.tenant, public_id=reserved.public_id
        )
        == reserved
    )
    linked = linked_tasks[0]
    assert trial.trajectory is not None and len(trial.trajectory.children) == 1
    child = trial.trajectory.children[0]
    assert child.session is not None and child.session.id == linked.session_id
    assert child.session.execution_deadline.expires_at == reserved.coding_deadline().expires_at
    assert trial.usage_summary is not None
    assert trial.usage_summary["model_steps"] == expected_model_steps
    assert trial.structured_output is not None
    assert trial.final_output == ("rejected" if rejected else "verified")
    assert trial.structured_output["product_run_id"] == linked.product_run_id
    repository = CodingProductArtifactRepository(application.artifact_store)
    admitted = await repository.load_request(linked.product_run_id, session_id=linked.session_id)
    publication = await repository.load_publication(
        request_fingerprint=admitted.fingerprint, digest=trial.structured_output["result_digest"]
    )
    assert trial.structured_output["result_digest"] == publication.result_reference.digest
    await _exercise_saved_capture(
        application, target, request, result, reservation_path.parent, monkeypatch
    )
    return linked, publication


async def _exercise_saved_capture(application, target, request, result, directory, monkeypatch):
    """Controlled private evidence, not a reviewed live failure or a public export."""
    report_path = directory / "maintenance-eval-private.json"
    write_eval_run_json(result, report_path)
    saved = load_eval_run(report_path)
    source = saved.cases[0].trials[0]
    original = report_path.read_bytes()
    # Compare the durable contract, not excluded in-process trajectory handles.
    # Native capture below must independently authenticate the reconstructed
    # anchor and output against the original store.
    assert eval_run_to_json(saved) == eval_run_to_json(result)
    assert source.workflow_attempt is not None
    assert source.session_id is not None
    # EvalRun deliberately excludes its in-process trajectory from report JSON.
    # Recovery must recapture it from the store, not rely on a serialized cache.
    assert source.trajectory is None
    original_trajectory = result.cases[0].trials[0].trajectory
    assert original_trajectory is not None and len(original_trajectory.children) == 1
    child = original_trajectory.children[0].session
    assert child is not None
    query = EventQuery(session_ids=(source.session_id, child.id), limit=5000)
    before = await application.app.session_store.query_events(query)
    assert len(before) < query.limit

    def forbidden(*args, **kwargs):
        pytest.fail("Saved maintenance capture must not invoke application execution.")

    read_target = target.model_copy(
        update={"workflow_factory": forbidden, "result_projector": forbidden}
    )
    with monkeypatch.context() as guard:
        guard.setattr(application.app, "run", forbidden)
        guard.setattr(application, "run", forbidden)
        guard.setattr(application, "verify", forbidden)
        capture = await capture_workflow_eval_attempt(
            read_target,
            source,
            messages=tuple(request.messages),
            bounds=target.capture_bounds,
        )
        captured = SavedWorkflowEvalCapture.model_validate_json(capture.model_dump_json())
        assert captured.model_dump_json() == capture.model_dump_json()
        assert captured.source_trial.model_dump_json() == source.model_dump_json()
        assert captured.source_attempt == source.workflow_attempt
        assert len(captured.trajectory.children) == 1
        captured_child = captured.trajectory.children[0].session
        assert captured_child is not None and captured_child.id == child.id
        score = await score_workflow_eval_capture(
            read_target,
            captured,
            (FinalOutputEqualsAssertionSpec(id="independent-verdict", expected="verified"),),
        )
        assert SavedWorkflowEvalScore.model_validate_json(score.model_dump_json()) == score
        assert score.score == (1.0 if result.status is EvalStatus.PASSED else 0.0)
        assert score.model_calls == 0
        assert score.source_capture_id == captured.capture_id
        assert score.source_evidence_sha256 == captured.evidence_sha256
        assert score.assertion_revisions
        for changed_target, messages in (
            (read_target, (Message.text("user", "changed input"),)),
            (
                read_target.model_copy(update={"implementation_revision": "sha256:" + "2" * 64}),
                tuple(request.messages),
            ),
        ):
            with pytest.raises(SessionTrajectoryError):
                await capture_workflow_eval_attempt(
                    changed_target, source, messages=messages, bounds=target.capture_bounds
                )
    assert await application.app.session_store.query_events(query) == before
    assert report_path.read_bytes() == original
    assert eval_run_to_json(saved) == eval_run_to_json(result)
