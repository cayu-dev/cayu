from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator

import pytest
from pydantic import SecretStr

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    CorpusExecutionLimits,
    CorpusTarget,
    EvalRunInvocation,
    EvalRunRequest,
    EvalScenarioApprovalSubmission,
    EvalScenarioArtifactReference,
    EvalScenarioDocumentV2,
    EvalScenarioRunInvocation,
    EvalScenarioTrialFailureCode,
    EvalScenarioTrialPhase,
    ExecutionProfileBehaviorIdentity,
    InMemoryEvalStore,
    Message,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    RunRequest,
    ScenarioApprovalCheckpointEventV2,
    ScenarioInitialInputEventV2,
    ScenarioInputV2,
    ScenarioLaunchSettingsV2,
    ScenarioResumedInputEventV2,
    ScenarioTextPartV2,
    ScenarioUserMessageV2,
    SQLiteEvalStore,
    SQLiteSessionStore,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
    UserInputTool,
    compile_corpus_suite,
    corpus_for_eval_scenario,
    preflight_eval_scenario,
    run_compiled_eval_scenario,
)


class _ApprovalProvider(ModelProvider):
    name = "scenario-approval-provider"

    def __init__(self, *, request_approval: bool = True) -> None:
        self.request_count = 0
        self.request_approval = request_approval

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:scenario-approval-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.request_count += 1
        if self.request_approval and self.request_count == 1:
            yield ModelStreamEvent.tool_call(
                id="call-approval",
                name="review_action",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("approved answer")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _UserInputProvider(ModelProvider):
    name = "scenario-user-input-provider"

    def __init__(self) -> None:
        self.request_count = 0

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:scenario-user-input-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.request_count += 1
        if self.request_count == 1:
            yield ModelStreamEvent.tool_call(
                id="call-environment",
                name="ask_user",
                arguments={"question": "Which environment?"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("resumed with authored input")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _ReviewTool(Tool):
    spec = ToolSpec(
        name="review_action",
        description="Perform one reviewed action.",
        input_schema={"type": "object", "properties": {}},
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:scenario-review-tool",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(content="reviewed")


def _scenario(*, occurrence: int = 1) -> EvalScenarioDocumentV2:
    scenario_input = ScenarioInputV2.create(
        (ScenarioUserMessageV2.create((ScenarioTextPartV2(text="Review this request."),)),)
    )
    return EvalScenarioDocumentV2.create(
        id="review-request",
        target_key="assistant.default",
        name="Review request",
        events=(
            ScenarioInitialInputEventV2(
                sequence=0,
                id="initial",
                input=scenario_input,
            ),
            ScenarioApprovalCheckpointEventV2(
                sequence=1,
                id="review-approval",
                tool_name="review_action",
                occurrence=occurrence,
            ),
        ),
    )


def _approval_target(provider: ModelProvider, *, session_store=None) -> CorpusTarget:
    app = CayuApp(session_store=session_store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="scenario-model"),
        tools=[_ReviewTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )
    return CorpusTarget(
        key="assistant.default",
        app=app,
        request_base=RunRequest(
            agent_name="assistant",
            messages=[],
            max_steps=8,
        ),
        bootstrap_messages=(Message.text("system", "Follow policy."),),
        application_release_id="release-current",
        limits=CorpusExecutionLimits(max_trials=2, max_concurrency=2),
    )


def _user_input_target(provider: ModelProvider) -> CorpusTarget:
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="scenario-model"),
        tools=[UserInputTool()],
    )
    return CorpusTarget(
        key="assistant.default",
        app=app,
        request_base=RunRequest(agent_name="assistant", messages=[], max_steps=8),
        application_release_id="release-current",
    )


def _resumed_scenario() -> EvalScenarioDocumentV2:
    return EvalScenarioDocumentV2.create(
        id="resume-request",
        target_key="assistant.default",
        name="Resume request",
        events=(
            ScenarioInitialInputEventV2(
                sequence=0,
                id="initial",
                input=ScenarioInputV2.create(
                    (
                        ScenarioUserMessageV2.create(
                            (ScenarioTextPartV2(text="Ask which environment to use."),)
                        ),
                    )
                ),
            ),
            ScenarioResumedInputEventV2(
                sequence=1,
                id="environment-answer",
                resume_kind="user_input",
                input=ScenarioInputV2.create(
                    (ScenarioUserMessageV2.create((ScenarioTextPartV2(text="production"),)),)
                ),
            ),
        ),
    )


def _manual_resume_scenario() -> EvalScenarioDocumentV2:
    return EvalScenarioDocumentV2.create(
        id="manual-resume-request",
        target_key="assistant.default",
        name="Manual resume request",
        events=(
            ScenarioInitialInputEventV2(
                sequence=0,
                id="initial",
                input=ScenarioInputV2.create(
                    (
                        ScenarioUserMessageV2.create(
                            (ScenarioTextPartV2(text="Complete the first interaction."),)
                        ),
                    )
                ),
            ),
            ScenarioResumedInputEventV2(
                sequence=1,
                id="follow-up",
                resume_kind="manual_recovery",
                input=ScenarioInputV2.create(
                    (
                        ScenarioUserMessageV2.create(
                            (ScenarioTextPartV2(text="Complete the follow-up."),)
                        ),
                    )
                ),
            ),
        ),
    )


def test_scenario_execution_waits_for_fresh_approval_and_publishes_corpus_result() -> None:
    async def exercise() -> None:
        provider = _ApprovalProvider()
        target = _approval_target(provider)
        app = target.app
        scenario = _scenario()
        settings = ScenarioLaunchSettingsV2(timeout_seconds=30)
        preflight = await preflight_eval_scenario(
            scenario,
            target,
            settings,
            actor_authorized=True,
        )
        assert preflight.ready is True
        assert preflight.binding is not None
        binding = preflight.binding
        scenario_invocation = EvalScenarioRunInvocation(
            scenario_revision=scenario.revision,
            binding_revision=binding.revision,
            environment_name=binding.environment_name,
            trials=binding.trials,
            timeout_seconds=binding.timeout_seconds,
            artifact_references=tuple(
                EvalScenarioArtifactReference(
                    requirement_id=item.requirement_id,
                    artifact_id=item.artifact_id,
                )
                for item in binding.artifacts
            ),
        )
        invocation = EvalRunInvocation(
            max_steps=binding.max_steps,
            limits=binding.operator_run_limits,
            cost_budget=binding.cost_budget,
            scenario=scenario_invocation,
        )
        corpus = corpus_for_eval_scenario(scenario, binding, target)
        compiled = compile_corpus_suite(corpus, target, "scenario")
        store = InMemoryEvalStore()
        await store.save_scenario(scenario, redact_json=app.redact_json)
        await store.save_corpus(corpus, redact_json=app.redact_json)
        request = EvalRunRequest(
            run_id="scenario-run",
            idempotency_key="sha256:" + "a" * 64,
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id="scenario",
            suite_revision=compiled.run_contract.suite_revision,
            max_concurrency=1,
            invocation=invocation,
        )
        await store.admit_run(request, redact_json=app.redact_json)
        lease = await store.claim_run(target_key=target.key, lease_seconds=30)
        assert lease is not None
        execution = asyncio.create_task(
            run_compiled_eval_scenario(
                target,
                compiled,
                scenario,
                binding,
                store=store,
                claim=lease.claim,
                max_concurrency=1,
                poll_seconds=0.001,
            )
        )
        while True:
            run = await store.load_run(request.run_id)
            assert run is not None
            progress = run.scenario_progress
            if progress is not None and progress.trials[0].phase == "awaiting_approval":
                break
            await asyncio.sleep(0)
        await store.submit_scenario_approval(
            request.run_id,
            EvalScenarioApprovalSubmission(
                expected_progress_revision=progress.revision,
                trial_number=1,
                event_id="review-approval",
                decision="approve",
                actor_id="reviewer",
            ),
        )
        result = await execution
        published = await store.publish_result(
            lease.claim,
            result,
            redact_json=app.redact_json,
        )

        assert published.status == "completed"
        assert published.scenario_progress is not None
        assert published.scenario_progress.trials[0].phase == "completed"
        assert result.run.status == "passed"
        assert result.run.cases[0].trials[0].output.text == "approved answer"
        assert provider.request_count == 2

    asyncio.run(exercise())


def test_scenario_execution_rejects_the_wrong_approval_occurrence() -> None:
    async def exercise() -> None:
        target = _approval_target(_ApprovalProvider())
        scenario = _scenario(occurrence=2)
        preflight = await preflight_eval_scenario(
            scenario,
            target,
            ScenarioLaunchSettingsV2(timeout_seconds=30),
            actor_authorized=True,
        )
        assert preflight.binding is not None
        binding = preflight.binding
        corpus = corpus_for_eval_scenario(scenario, binding, target)
        compiled = compile_corpus_suite(corpus, target, "scenario")
        store = InMemoryEvalStore()
        await store.save_scenario(scenario, redact_json=target.app.redact_json)
        await store.save_corpus(corpus, redact_json=target.app.redact_json)
        request = EvalRunRequest(
            run_id="scenario-occurrence-run",
            idempotency_key="sha256:" + "e" * 64,
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id="scenario",
            suite_revision=compiled.run_contract.suite_revision,
            max_concurrency=1,
            invocation=EvalRunInvocation(
                max_steps=binding.max_steps,
                limits=binding.operator_run_limits,
                cost_budget=binding.cost_budget,
                scenario=EvalScenarioRunInvocation(
                    scenario_revision=scenario.revision,
                    binding_revision=binding.revision,
                    environment_name=binding.environment_name,
                    trials=binding.trials,
                    timeout_seconds=binding.timeout_seconds,
                ),
            ),
        )
        await store.admit_run(request, redact_json=target.app.redact_json)
        claimed = await store.claim_run(target_key=target.key, lease_seconds=30)
        assert claimed is not None

        result = await run_compiled_eval_scenario(
            target,
            compiled,
            scenario,
            binding,
            store=store,
            claim=claimed.claim,
            max_concurrency=1,
            poll_seconds=0.001,
        )

        run = await store.load_run(request.run_id)
        assert run is not None and run.scenario_progress is not None
        trial = run.scenario_progress.trials[0]
        assert trial.phase is EvalScenarioTrialPhase.ERROR
        assert trial.failure_code == "expected_approval_unavailable"
        assert result.run.status == "error"
        assert result.run.cases[0].trials[0].code == "execution_failed"

    asyncio.run(exercise())


def test_scenario_execution_timeout_closes_pending_trial_progress() -> None:
    async def exercise() -> None:
        target = _approval_target(_ApprovalProvider())
        scenario = _scenario()
        preflight = await preflight_eval_scenario(
            scenario,
            target,
            ScenarioLaunchSettingsV2(timeout_seconds=1),
            actor_authorized=True,
        )
        assert preflight.binding is not None
        binding = preflight.binding
        corpus = corpus_for_eval_scenario(scenario, binding, target)
        compiled = compile_corpus_suite(corpus, target, "scenario")
        store = InMemoryEvalStore()
        await store.save_scenario(scenario, redact_json=target.app.redact_json)
        await store.save_corpus(corpus, redact_json=target.app.redact_json)
        request = EvalRunRequest(
            run_id="scenario-timeout-run",
            idempotency_key="sha256:" + "a" * 64,
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id="scenario",
            suite_revision=compiled.run_contract.suite_revision,
            max_concurrency=1,
            invocation=EvalRunInvocation(
                max_steps=binding.max_steps,
                limits=binding.operator_run_limits,
                cost_budget=binding.cost_budget,
                scenario=EvalScenarioRunInvocation(
                    scenario_revision=scenario.revision,
                    binding_revision=binding.revision,
                    environment_name=binding.environment_name,
                    trials=binding.trials,
                    timeout_seconds=binding.timeout_seconds,
                ),
            ),
        )
        await store.admit_run(request, redact_json=target.app.redact_json)
        claimed = await store.claim_run(target_key=target.key, lease_seconds=30)
        assert claimed is not None

        result = await run_compiled_eval_scenario(
            target,
            compiled,
            scenario,
            binding,
            store=store,
            claim=claimed.claim,
            max_concurrency=1,
            poll_seconds=0.001,
        )

        run = await store.load_run(request.run_id)
        assert run is not None and run.scenario_progress is not None
        trial = run.scenario_progress.trials[0]
        assert trial.phase is EvalScenarioTrialPhase.ERROR
        assert trial.failure_code is EvalScenarioTrialFailureCode.EXECUTION_FAILED
        assert result.run.status == "error"
        assert result.run.cases[0].trials[0].code == "case_timeout"

    asyncio.run(exercise())


def test_scenario_resumed_input_recovers_the_same_checkpoint_after_claim_release() -> None:
    async def exercise() -> None:
        provider = _UserInputProvider()
        target = _user_input_target(provider)
        scenario = _resumed_scenario()
        preflight = await preflight_eval_scenario(
            scenario,
            target,
            ScenarioLaunchSettingsV2(timeout_seconds=30),
            actor_authorized=True,
        )
        assert preflight.binding is not None
        binding = preflight.binding
        corpus = corpus_for_eval_scenario(scenario, binding, target)
        compiled = compile_corpus_suite(corpus, target, "scenario")
        store = InMemoryEvalStore()
        await store.save_scenario(scenario, redact_json=target.app.redact_json)
        await store.save_corpus(corpus, redact_json=target.app.redact_json)
        request = EvalRunRequest(
            run_id="scenario-resume-run",
            idempotency_key="sha256:" + "d" * 64,
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id="scenario",
            suite_revision=compiled.run_contract.suite_revision,
            max_concurrency=1,
            invocation=EvalRunInvocation(
                max_steps=binding.max_steps,
                limits=binding.operator_run_limits,
                cost_budget=binding.cost_budget,
                scenario=EvalScenarioRunInvocation(
                    scenario_revision=scenario.revision,
                    binding_revision=binding.revision,
                    environment_name=binding.environment_name,
                    trials=binding.trials,
                    timeout_seconds=binding.timeout_seconds,
                ),
            ),
        )
        await store.admit_run(request, redact_json=target.app.redact_json)
        first = await store.claim_run(target_key=target.key, lease_seconds=30)
        assert first is not None
        checkpoint_written = asyncio.Event()
        hold_checkpoint = asyncio.Event()
        original_update = store.update_scenario_trial

        async def pause_after_resume_checkpoint(claim, trial):
            updated = await original_update(claim, trial)
            if trial.phase is EvalScenarioTrialPhase.AWAITING_RESUME:
                checkpoint_written.set()
                await hold_checkpoint.wait()
            return updated

        store.update_scenario_trial = pause_after_resume_checkpoint  # type: ignore[method-assign]
        first_execution = asyncio.create_task(
            run_compiled_eval_scenario(
                target,
                compiled,
                scenario,
                binding,
                store=store,
                claim=first.claim,
                max_concurrency=1,
                poll_seconds=0.001,
            )
        )
        await asyncio.wait_for(checkpoint_written.wait(), timeout=5)
        waiting = await store.load_run(request.run_id)
        assert waiting is not None and waiting.scenario_progress is not None
        waiting_trial = waiting.scenario_progress.trials[0]
        assert waiting_trial.phase is EvalScenarioTrialPhase.AWAITING_RESUME
        assert waiting_trial.pending_event_id == "environment-answer"
        assert waiting_trial.pending_input_id is not None
        first_execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_execution
        store.update_scenario_trial = original_update  # type: ignore[method-assign]
        await store.release_run(first.claim)

        second = await store.claim_run(target_key=target.key, lease_seconds=30)
        assert second is not None and second.run.scenario_progress is not None
        resumed = second.run.scenario_progress.trials[0]
        assert resumed.phase is EvalScenarioTrialPhase.AWAITING_RESUME
        assert resumed.session_id == waiting_trial.session_id
        result = await run_compiled_eval_scenario(
            target,
            compiled,
            scenario,
            binding,
            store=store,
            claim=second.claim,
            max_concurrency=1,
            poll_seconds=0.001,
        )
        published = await store.publish_result(
            second.claim,
            result,
            redact_json=target.app.redact_json,
        )
        assert published.status == "completed"
        assert result.run.status == "passed"
        assert result.run.cases[0].trials[0].output.text == "resumed with authored input"
        assert provider.request_count == 2

    asyncio.run(exercise())


def test_scenario_session_resume_recovers_the_same_checkpoint_after_claim_release() -> None:
    async def exercise() -> None:
        provider = _ApprovalProvider(request_approval=False)
        target = _approval_target(provider)
        scenario = _manual_resume_scenario()
        preflight = await preflight_eval_scenario(
            scenario,
            target,
            ScenarioLaunchSettingsV2(timeout_seconds=30),
            actor_authorized=True,
        )
        assert preflight.binding is not None
        binding = preflight.binding
        corpus = corpus_for_eval_scenario(scenario, binding, target)
        compiled = compile_corpus_suite(corpus, target, "scenario")
        store = InMemoryEvalStore()
        await store.save_scenario(scenario, redact_json=target.app.redact_json)
        await store.save_corpus(corpus, redact_json=target.app.redact_json)
        request = EvalRunRequest(
            run_id="scenario-manual-resume-run",
            idempotency_key="sha256:" + "f" * 64,
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id="scenario",
            suite_revision=compiled.run_contract.suite_revision,
            max_concurrency=1,
            invocation=EvalRunInvocation(
                max_steps=binding.max_steps,
                limits=binding.operator_run_limits,
                cost_budget=binding.cost_budget,
                scenario=EvalScenarioRunInvocation(
                    scenario_revision=scenario.revision,
                    binding_revision=binding.revision,
                    environment_name=binding.environment_name,
                    trials=binding.trials,
                    timeout_seconds=binding.timeout_seconds,
                ),
            ),
        )
        await store.admit_run(request, redact_json=target.app.redact_json)
        first = await store.claim_run(target_key=target.key, lease_seconds=30)
        assert first is not None
        checkpoint_written = asyncio.Event()
        hold_checkpoint = asyncio.Event()
        original_update = store.update_scenario_trial

        async def pause_after_resume_checkpoint(claim, trial):
            updated = await original_update(claim, trial)
            if (
                trial.phase is EvalScenarioTrialPhase.AWAITING_RESUME
                and trial.pending_resume_kind == "manual_recovery"
            ):
                checkpoint_written.set()
                await hold_checkpoint.wait()
            return updated

        store.update_scenario_trial = pause_after_resume_checkpoint  # type: ignore[method-assign]
        first_execution = asyncio.create_task(
            run_compiled_eval_scenario(
                target,
                compiled,
                scenario,
                binding,
                store=store,
                claim=first.claim,
                max_concurrency=1,
                poll_seconds=0.001,
            )
        )
        await asyncio.wait_for(checkpoint_written.wait(), timeout=5)
        first_execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_execution
        store.update_scenario_trial = original_update  # type: ignore[method-assign]
        await store.release_run(first.claim)

        second = await store.claim_run(target_key=target.key, lease_seconds=30)
        assert second is not None and second.run.scenario_progress is not None
        resumed = second.run.scenario_progress.trials[0]
        assert resumed.phase is EvalScenarioTrialPhase.AWAITING_RESUME
        assert resumed.pending_event_id == "follow-up"
        assert resumed.pending_resume_kind == "manual_recovery"
        result = await run_compiled_eval_scenario(
            target,
            compiled,
            scenario,
            binding,
            store=store,
            claim=second.claim,
            max_concurrency=1,
            poll_seconds=0.001,
        )
        assert result.run.status == "passed"
        assert result.run.cases[0].trials[0].output.text == "approved answer"
        assert provider.request_count == 2

    asyncio.run(exercise())


def test_scenario_approval_checkpoint_recovers_after_sqlite_coordinator_restart(
    tmp_path,
) -> None:
    async def exercise() -> None:
        eval_path = tmp_path / "evals.sqlite"
        runtime_path = tmp_path / "runtime.sqlite"
        key = base64.urlsafe_b64encode(bytes([7]) * 32).decode("ascii").rstrip("=")
        alias_codec = PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(
                active_key_id="test",
                keys={"test": SecretStr(key)},
            )
        )
        first_runtime_store = SQLiteSessionStore(
            runtime_path,
            public_authority_alias_codec=alias_codec,
        )
        first_provider = _ApprovalProvider()
        first_target = _approval_target(first_provider, session_store=first_runtime_store)
        scenario = _scenario()
        settings = ScenarioLaunchSettingsV2(timeout_seconds=30)
        preflight = await preflight_eval_scenario(
            scenario,
            first_target,
            settings,
            actor_authorized=True,
        )
        assert preflight.binding is not None
        binding = preflight.binding
        invocation = EvalRunInvocation(
            max_steps=binding.max_steps,
            limits=binding.operator_run_limits,
            cost_budget=binding.cost_budget,
            scenario=EvalScenarioRunInvocation(
                scenario_revision=scenario.revision,
                binding_revision=binding.revision,
                environment_name=binding.environment_name,
                trials=binding.trials,
                timeout_seconds=binding.timeout_seconds,
            ),
        )
        corpus = corpus_for_eval_scenario(scenario, binding, first_target)
        first_compiled = compile_corpus_suite(corpus, first_target, "scenario")
        first_store = SQLiteEvalStore(eval_path)
        await first_store.save_scenario(scenario, redact_json=first_target.app.redact_json)
        await first_store.save_corpus(corpus, redact_json=first_target.app.redact_json)
        request = EvalRunRequest(
            run_id="scenario-restart-run",
            idempotency_key="sha256:" + "c" * 64,
            corpus_revision=corpus.revision,
            target_key=first_target.key,
            suite_id="scenario",
            suite_revision=first_compiled.run_contract.suite_revision,
            max_concurrency=1,
            invocation=invocation,
        )
        await first_store.admit_run(request, redact_json=first_target.app.redact_json)
        first_lease = await first_store.claim_run(target_key=first_target.key, lease_seconds=30)
        assert first_lease is not None
        first_execution = asyncio.create_task(
            run_compiled_eval_scenario(
                first_target,
                first_compiled,
                scenario,
                binding,
                store=first_store,
                claim=first_lease.claim,
                max_concurrency=1,
                poll_seconds=0.001,
            )
        )
        while True:
            waiting = await first_store.load_run(request.run_id)
            assert waiting is not None
            if (
                waiting.scenario_progress is not None
                and waiting.scenario_progress.trials[0].phase == "awaiting_approval"
            ):
                break
            await asyncio.sleep(0)
        first_execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_execution
        await first_store.release_run(first_lease.claim)
        await first_runtime_store.close()
        await first_store.close()

        second_runtime_store = SQLiteSessionStore(
            runtime_path,
            public_authority_alias_codec=alias_codec,
        )
        second_provider = _ApprovalProvider(request_approval=False)
        second_target = _approval_target(second_provider, session_store=second_runtime_store)
        second_preflight = await preflight_eval_scenario(
            scenario,
            second_target,
            settings,
            actor_authorized=True,
        )
        assert second_preflight.binding == binding
        second_compiled = compile_corpus_suite(corpus, second_target, "scenario")
        second_store = SQLiteEvalStore(eval_path)
        try:
            second_lease = await second_store.claim_run(
                target_key=second_target.key,
                lease_seconds=30,
            )
            assert second_lease is not None
            resumed = second_lease.run.scenario_progress
            assert resumed is not None
            assert resumed.trials[0].phase == "awaiting_approval"
            assert resumed.trials[0].session_id is not None
            second_execution = asyncio.create_task(
                run_compiled_eval_scenario(
                    second_target,
                    second_compiled,
                    scenario,
                    binding,
                    store=second_store,
                    claim=second_lease.claim,
                    max_concurrency=1,
                    poll_seconds=0.001,
                )
            )
            await second_store.submit_scenario_approval(
                request.run_id,
                EvalScenarioApprovalSubmission(
                    expected_progress_revision=resumed.revision,
                    trial_number=1,
                    event_id="review-approval",
                    decision="approve",
                    actor_id="restart-reviewer",
                ),
            )
            result = await second_execution
            execution_record = await second_store.load_run(request.run_id)
            assert execution_record is not None
            assert execution_record.scenario_progress is not None
            execution_trial = execution_record.scenario_progress.trials[0]
            published = await second_store.publish_result(
                second_lease.claim,
                result,
                redact_json=second_target.app.redact_json,
            )
            assert published.status == "completed"
            trial = result.run.cases[0].trials[0]
            assert result.run.status == "passed", (
                trial.status,
                trial.code,
                trial.message,
                execution_trial.phase,
                execution_trial.failure_code,
                second_provider.request_count,
            )
            assert trial.output.text == "approved answer"
            assert first_provider.request_count == 1
            assert second_provider.request_count == 1
        finally:
            await second_runtime_store.close()
            await second_store.close()

    asyncio.run(exercise())
