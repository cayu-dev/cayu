"""Effective request target conformance; no public failover dispatch claim."""

from __future__ import annotations

import asyncio
import pickle
import time
from contextlib import nullcontext
from dataclasses import replace

import pytest
from tests.core._execution_profile_fixtures import versioned_test_provider_identity
from tests.core.test_model_completion_executor_publication import _atomic_publisher
from tests.core.test_model_failover_stages import (
    _initial_admission,
    _prepare,
    _StageMemoryStore,
    _StageSQLiteStore,
)

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ModelFailoverPolicy,
    ModelTarget,
    RecentTurnsContextPolicy,
    ResumeRequest,
    RunRequest,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu._exception_groups import iter_exception_tree
from cayu.context.counting import ContextCountingConfig, ContextCountingMode
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.base import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelStreamEvent,
)
from cayu.runtime._execution_profile_admission import ModelFailoverProfileResolution
from cayu.runtime._model_execution_selection import (
    ModelExecutionSelection,
    ModelFailoverAttempt,
    ModelFailoverTransition,
)
from cayu.runtime._model_failover import FailoverObservation
from cayu.runtime._model_step_executor import (
    ModelCompletionDispatch,
    ModelCompletionRecoveryContext,
    _model_request_fingerprint,
    _model_stream_event_to_runtime_event,
    _ModelFailoverCandidateExhausted,
)
from cayu.runtime._run_limits import RunLimitGate
from cayu.runtime._runtime_records import RegisteredProvider
from cayu.runtime.execution_profiles import ExecutionProfileMismatchError
from cayu.runtime.execution_units import ModelStepIdentity, new_model_step_identity
from cayu.runtime.retry_policy import RetryPolicy, retry_decision
from cayu.runtime.stop_policy import RunLimits


async def _selection():
    store = _StageMemoryStore()
    admission, _ = await _initial_admission(store)
    context = admission.invocation_context
    backup = RegisteredProvider("backup", ScriptedModelProvider([], name="backup"))
    resolution = ModelFailoverProfileResolution(
        plan=admission.successor.plan,
        candidate_profiles=admission.candidate_profiles,
        registered_providers=(context.registered_provider, backup),
    )
    selection = ModelExecutionSelection(
        invocation_context=context, resolution=resolution, candidate_index=1
    )
    session = await store.load("session")
    assert session is not None
    return store, session, selection


@pytest.mark.parametrize("with_tool", [False, True])
@pytest.mark.parametrize("native_history", [False, True])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("resume", [False, True], ids=["initial", "resume"])
def test_gated_public_run_uses_admitted_candidate_plan(
    monkeypatch, with_tool, native_history, backend, tmp_path, resume
):
    # Exercise the public path with native store capability and actual profile
    # admission; no runtime admission boundary is bypassed by this fixture.

    async def scenario(store):
        def unavailable(_request):
            if native_history and len(primary.requests) == 1:
                return [
                    ModelStreamEvent.thinking("primary-private-reasoning"),
                    ModelStreamEvent.tool_call(id="primary-call", name="echo", arguments={}),
                    ModelStreamEvent.completed(),
                ]
            raise ModelProviderError(
                "unavailable", provider="primary", status_code=503, retryable=True
            )

        primary = ScriptedModelProvider(name="primary", response_factory=unavailable)
        tool_calls = []

        class EchoTool(Tool):
            spec = ToolSpec(
                name="echo", description="Echo", input_schema={"type": "object", "properties": {}}
            )

            async def run(self, ctx, args):
                tool_calls.append(ctx.agent_name)
                return ToolResult(content="echoed")

        def respond(_request):
            if with_tool and len(backup.requests) == 1:
                return [
                    ModelStreamEvent.thinking("backup-private-reasoning"),
                    ModelStreamEvent.tool_call(id="echo-call", name="echo", arguments={}),
                    ModelStreamEvent.completed(),
                ]
            return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]

        backup = ScriptedModelProvider(response_factory=respond, name="backup")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"), tools=[EchoTool()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="public-failover",
                    messages=[Message.text("user", "hello")],
                    retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0),
                    failover=ModelFailoverPolicy(
                        fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                    ),
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED, [
            (event.type, event.payload) for event in events[-4:]
        ]
        assert len(primary.requests) == 2 + int(native_history) and len(backup.requests) == 1 + int(
            with_tool
        )
        assert tool_calls == ["agent"] * (int(with_tool) + int(native_history))
        assert primary.requests[0].model == "small" and backup.requests[0].model == "large"
        assert events[-1].type is EventType.SESSION_COMPLETED
        completed = [event for event in events if event.type is EventType.MODEL_COMPLETED]
        assert len(completed) == 1 + int(with_tool) + int(native_history)
        assert all(
            event.payload["provider_name"] == "backup" for event in completed[int(native_history) :]
        )
        assert all(
            event.payload["requested_model"] == "large"
            for event in completed[int(native_history) :]
        )
        assert all(
            "primary-private-reasoning" not in repr(request.messages) for request in backup.requests
        )
        if with_tool:
            assert "backup-private-reasoning" in repr(backup.requests[-1].messages)
        assert [event.type for event in events].count(EventType.MODEL_FAILOVER_SELECTED) == 2
        session = await store.load("public-failover")
        assert (
            session is not None and session.provider_name == "primary" and session.model == "small"
        )
        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is not None and checkpoint["model_failover"]["candidate_index"] == 1
        assert await store.load_active_model_completion_stage(session.id) is None
        if native_history:
            assert "primary-private-reasoning" in repr(await store.load_transcript(session.id))
        if resume:
            prior_progress = checkpoint["model_failover"]
            resumed_events = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session.id,
                        messages=[Message.text("user", "continue")],
                        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0),
                    )
                )
            ]
            assert resumed_events[-1].type is EventType.SESSION_COMPLETED, [
                (event.type, event.payload) for event in resumed_events[-4:]
            ]
            assert len(primary.requests) == 2 + int(native_history)
            assert len(backup.requests) == 2 + int(with_tool)
            assert backup.requests[-1].model == "large"
            assert "primary-private-reasoning" not in repr(backup.requests[-1].messages)
            if with_tool:
                assert "backup-private-reasoning" in repr(backup.requests[-1].messages)
            assert not any(
                event.type is EventType.MODEL_FAILOVER_SELECTED for event in resumed_events
            )
            checkpoint = await store.load_checkpoint(session.id)
            assert checkpoint is not None
            progress = checkpoint["model_failover"]
            assert progress["candidate_index"] == 1
            assert progress["plan"] == prior_progress["plan"]
            assert progress["source_run_epoch"] > prior_progress["source_run_epoch"]
            assert progress["logical_step_id"] != prior_progress["logical_step_id"]
            assert progress["attempts_used"] == 1
            assert await store.load_active_model_completion_stage(session.id) is None
        return checkpoint

    async def owned_scenario():
        path = tmp_path / "public-failover.sqlite"
        store = _StageMemoryStore() if backend == "memory" else _StageSQLiteStore(path)
        try:
            checkpoint = await scenario(store)
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()
        if backend == "sqlite":
            reopened = _StageSQLiteStore(path)
            try:
                assert await reopened.load_checkpoint("public-failover") == checkpoint
                assert await reopened.load_active_model_completion_stage("public-failover") is None
            finally:
                await reopened.close()

    asyncio.run(owned_scenario())


@pytest.mark.parametrize("change", [None, "policy", "provider", "add_policy"])
def test_gated_public_resume_reconstructs_selected_plan_after_sqlite_reopen(
    monkeypatch, tmp_path, change
):

    class VersionedProvider(ScriptedModelProvider):
        behavior_version = "1"

        @property
        def execution_profile_identity(self):
            return versioned_test_provider_identity(self, behavior_version=self.behavior_version)

    def application(store, *, restarted):
        def unavailable(_request):
            if change == "add_policy":
                return [ModelStreamEvent.text_delta("primary"), ModelStreamEvent.completed()]
            raise ModelProviderError(
                "unavailable", provider="primary", status_code=503, retryable=True
            )

        primary = VersionedProvider(name="primary", response_factory=unavailable)
        backup = VersionedProvider(
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]], name="backup"
        )
        if restarted and change == "provider":
            backup.behavior_version = "2"
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        return app, primary, backup

    async def scenario():
        path = tmp_path / "selected-plan.sqlite"
        store = _StageSQLiteStore(path)
        policy = ModelFailoverPolicy(
            fallbacks=(ModelTarget(provider_name="backup", model="large"),)
        )
        retry = RetryPolicy(max_attempts=1, initial_delay_s=0)
        try:
            app, primary, backup = application(store, restarted=False)
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="restart-failover",
                        messages=[Message.text("user", "first")],
                        retry_policy=retry,
                        failover=None if change == "add_policy" else policy,
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert len(primary.requests) == 1
            assert len(backup.requests) == (0 if change == "add_policy" else 1)
            before = await store.load("restart-failover")
            checkpoint = await store.load_checkpoint("restart-failover")
            transcript = await store.load_transcript("restart-failover")
            assert before is not None and checkpoint is not None
        finally:
            await store.close()

        reopened = _StageSQLiteStore(path)
        try:
            app, primary, backup = application(reopened, restarted=True)
            request = ResumeRequest(
                session_id="restart-failover",
                messages=[Message.text("user", "second")],
                retry_policy=retry,
                failover=(
                    ModelFailoverPolicy(
                        fallbacks=(ModelTarget(provider_name="backup", model="different"),)
                    )
                    if change == "policy"
                    else policy
                    if change == "add_policy"
                    else None
                ),
            )
            if change is not None:
                with pytest.raises(ExecutionProfileMismatchError):
                    _ = [event async for event in app.resume(request)]
                assert not primary.requests and not backup.requests
                after = await reopened.load(before.id)
                assert after is not None and after.run_epoch == before.run_epoch
                after_checkpoint = await reopened.load_checkpoint(before.id)
                assert after_checkpoint is not None
                assert after_checkpoint.get("model_failover") == checkpoint.get("model_failover")
                assert await reopened.load_transcript(before.id) == transcript
            else:
                events = [event async for event in app.resume(request)]
                assert events[-1].type is EventType.SESSION_COMPLETED
                assert not primary.requests and len(backup.requests) == 1
                assert backup.requests[0].model == "large"
                after_checkpoint = await reopened.load_checkpoint(before.id)
                assert after_checkpoint is not None
                assert after_checkpoint["model_failover"]["candidate_index"] == 1
                assert (
                    after_checkpoint["model_failover"]["plan"]
                    == checkpoint["model_failover"]["plan"]
                )
            assert await reopened.load_active_model_completion_stage(before.id) is None
        finally:
            await reopened.close()

    asyncio.run(scenario())


def test_candidate_request_keeps_root_session_and_invocation_unchanged():
    async def scenario():
        store, session, selection = await _selection()
        context = selection.invocation_context
        original_session = session.model_dump(mode="json")
        original_profile = context.profile.model_dump(mode="json")
        app = CayuApp(session_store=store, enable_logging=False)
        request = await app._model_step_executor.build_request(
            session=session,
            registered_agent=context.registered_agent,
            registered_environment=None,
            context_messages=[Message.text("user", "hello")],
            structured_output=None,
            thinking=None,
            step=1,
            model_execution_selection=selection,
        )
        assert request.model == "large"
        assert session.model == "small" and session.provider_name == "primary"
        assert session.model_dump(mode="json") == original_session
        assert context.profile.model_dump(mode="json") == original_profile
        assert context.registered_provider.name == "primary"
        assert selection.registered_provider.name == "backup"
        assert await store.load_active_model_completion_stage("session") is None
        assert isinstance(selection.registered_provider.provider, ScriptedModelProvider)
        assert not selection.registered_provider.provider.requests
        with pytest.raises(TypeError, match="serialization"):
            pickle.dumps(selection)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change", ["provider", "model", "session_target", "session_epoch", "profile"]
)
def test_selection_rejects_identical_raw_or_conflicting_authority(change):
    async def scenario():
        _, session, selection = await _selection()
        context = selection.invocation_context
        arguments = dict(
            invocation_context=context,
            session=session,
            registered_agent=context.registered_agent,
            registered_provider=selection.registered_provider,
            execution_profile=context.profile,
            model=selection.model,
        )
        selection.require_request_scope(**arguments)
        if change == "provider":
            arguments["registered_provider"] = replace(selection.registered_provider)
        elif change == "model":
            arguments["model"] = session.model
        elif change == "session_target":
            arguments["session"] = session.model_copy(
                update={"model": selection.model, "provider_name": "backup"}
            )
        elif change == "session_epoch":
            arguments["session"] = session.model_copy(update={"run_epoch": session.run_epoch + 1})
        elif change == "profile":
            arguments["execution_profile"] = context.profile.model_copy(deep=True)
        with pytest.raises(ValueError, match="substituted"):
            selection.require_request_scope(**arguments)

    asyncio.run(scenario())


@pytest.mark.parametrize("index", [True, False, -1, 2, 1.0, "1"])
def test_selection_rejects_invalid_candidate_index(index):
    async def scenario():
        _, _, selection = await _selection()
        with pytest.raises(ValueError, match="index"):
            replace(selection, candidate_index=index)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change",
    ["candidate", "session", "epoch", "cursor", "stage", "step", "ordinal", "model", "provider"],
)
def test_selection_checks_complete_prepared_stage_handoff(change):
    async def scenario():
        store = _StageMemoryStore()
        admission, attempt = await _initial_admission(store)
        context = admission.invocation_context
        selection = ModelExecutionSelection(
            invocation_context=context,
            resolution=ModelFailoverProfileResolution(
                plan=admission.successor.plan,
                candidate_profiles=admission.candidate_profiles,
                registered_providers=(
                    context.registered_provider,
                    RegisteredProvider("backup", ScriptedModelProvider([], name="backup")),
                ),
            ),
            candidate_index=0,
        )
        prepared = await _prepare(store, admission, attempt)
        selection.require_prepared_stage(prepared.stage)
        stage = prepared.stage.model_copy(deep=True)
        if change == "candidate":
            selection = replace(selection, candidate_index=1)
        elif change == "model":
            stage.intent["requested_model"] = "large"
        elif change == "provider":
            stage.intent["provider_name"] = "backup"
        else:
            field, value = {
                "session": ("session_id", "different-session"),
                "epoch": ("source_run_epoch", stage.source_run_epoch + 1),
                "cursor": ("source_transcript_cursor", stage.source_transcript_cursor + 1),
                "stage": ("stage_id", "different-stage"),
                "step": ("logical_step_id", "different-step"),
                "ordinal": ("dispatch_ordinal", stage.dispatch_ordinal + 1),
            }[change]
            stage = stage.model_copy(update={field: value})
        with pytest.raises(ValueError, match="conflicts"):
            selection.require_prepared_stage(stage)
        # A rejected handoff cannot modify the existing durable selection.
        active = await store.load_active_model_completion_stage("session")
        assert active is not None and active.stage == prepared.stage

    asyncio.run(scenario())


def test_completion_projection_uses_requested_candidate_without_relabeling_root():
    async def scenario():
        _, session, selection = await _selection()
        event = _model_stream_event_to_runtime_event(
            ModelStreamEvent.completed(),
            session=session,
            requested_model=selection.model,
            registered_agent=selection.invocation_context.registered_agent,
            environment_name=None,
            provider_name=selection.registered_provider.name,
            step=1,
            attempt=1,
            max_attempts=1,
            model_attempt_identity=new_model_step_identity().new_attempt(),
        )
        assert event.type is EventType.MODEL_COMPLETED
        assert event.payload["model"] == "large"
        assert event.payload["requested_model"] == "large"
        assert event.payload["provider_name"] == "backup"
        assert session.model == "small" and session.provider_name == "primary"

    asyncio.run(scenario())


@pytest.mark.parametrize("first_attempt_output", [False, True])
def test_candidate_retry_exhaustion_retains_prior_output_through_real_provider(
    first_attempt_output,
):
    class UnavailableProvider(ModelProvider):
        name = "primary"

        def __init__(self):
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1 and first_attempt_output:
                yield ModelStreamEvent.text_delta("partial")
            raise ModelProviderError(
                "temporarily unavailable", provider=self.name, status_code=503, retryable=True
            )

    async def scenario():
        store = _StageMemoryStore()
        provider = UnavailableProvider()
        initial, initial_attempt = await _initial_admission(store, provider=provider)
        context = initial.invocation_context
        selection = ModelExecutionSelection(
            invocation_context=context,
            resolution=ModelFailoverProfileResolution(
                plan=initial.successor.plan,
                candidate_profiles=initial.candidate_profiles,
                registered_providers=(
                    context.registered_provider,
                    RegisteredProvider("backup", ScriptedModelProvider([], name="backup")),
                ),
            ),
            candidate_index=0,
        )
        session = await store.load("session")
        assert session is not None
        app = CayuApp(session_store=store, enable_logging=False)
        request = await app._model_step_executor.build_request(
            session=session,
            registered_agent=context.registered_agent,
            registered_environment=None,
            context_messages=[Message.text("user", "hello")],
            structured_output=None,
            thinking=None,
            step=1,
            model_execution_selection=selection,
        )
        attempt = initial_attempt
        admission = initial
        stages = []
        events = []
        attempt_identities = []

        async def reserve(identity):
            nonlocal attempt
            attempt = identity
            return [], None, None

        async def refresh():
            pass

        async def before_dispatch(_identity):
            pass

        async def publish(*_args, **_kwargs):
            raise AssertionError("Failed attempts must not publish a completion.")

        async def prepare(
            model_request,
            _reference,
            _notifications,
            _consume_notifications,
            *,
            failover_attempt=None,
        ):
            nonlocal admission
            assert failover_attempt is not None
            assert failover_attempt.identity == attempt
            fingerprint = _model_request_fingerprint(
                provider_name="primary", model_request=model_request
            )
            admission = failover_attempt.stage_admission(
                previous=None if not stages else admission.successor,
                source_preparation_digest=None if not stages else stages[-1].preparation_digest,
                source_transcript_cursor=initial.successor.source_transcript_cursor,
                request_fingerprint=fingerprint,
            )
            assert admission.successor.candidate_attempt == len(stages) + 1
            prepared = await _prepare(store, admission, attempt)
            await store.mark_model_completion_stage_dispatched("session", stage=prepared.stage)
            stages.append(prepared.stage)
            return ModelCompletionDispatch(
                stage=prepared.stage,
                request_fingerprint=fingerprint,
                prepared_events=prepared.prepared_events,
            )

        with pytest.raises(_ModelFailoverCandidateExhausted) as caught:
            async for event, result in app._model_step_executor.run_with_retries(
                provider=provider,
                model_request=request,
                session=session,
                registered_agent=context.registered_agent,
                registered_provider=context.registered_provider,
                environment_name=None,
                step=1,
                model_step_identity=ModelStepIdentity(model_step_id=initial_attempt.model_step_id),
                initial_model_attempt_identity=initial_attempt,
                record_model_attempt_identity=attempt_identities.append,
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
                transcript_cursor_before_request=initial.successor.source_transcript_cursor,
                record_model_completion=lambda event: event,
                prepare_provider_dispatch=reserve,
                before_provider_dispatch=before_dispatch,
                validate_live_model_semantics=lambda: None,
                refresh_live_model_semantics=refresh,
                prepare_model_completion_dispatch=prepare,
                model_completion_publisher=publish,
                execution_profile=context.profile,
                invocation_context=context,
                model_execution_selection=selection,
            ):
                assert result is None
                if event is not None:
                    events.append(event)
        exhausted = caught.value
        assert exhausted.selection is selection
        assert exhausted.identity == attempt_identities[-1]
        assert exhausted.provider_effect_observed is first_attempt_output
        assert exhausted.failure.provider_effect_observed is False
        assert isinstance(exhausted.failure.cause, ModelProviderError)
        assert exhausted.failure.cause.status_code == 503
        assert exhausted.decision.attempt == 2 and not exhausted.decision.retry
        assert len(provider.requests) == len(stages) == 2
        assert len({identity.model_attempt_id for identity in attempt_identities}) == 2
        assert [event.type for event in events].count(EventType.MODEL_RETRY) == 1
        selections = [event for event in events if event.type is EventType.MODEL_FAILOVER_SELECTED]
        assert len(selections) == 1
        durable_selections = [
            event
            for event in await store.load_events("session")
            if event.type is EventType.MODEL_FAILOVER_SELECTED
        ]
        assert selections == durable_selections
        assert EventType.MODEL_COMPLETED not in [event.type for event in events]
        active = await store.load_active_model_completion_stage("session")
        assert active is not None and active.stage == stages[-1]
        # The handoff is not permission to silently consume another target.
        backup = selection.resolution.registered_providers[1].provider
        assert isinstance(backup, ScriptedModelProvider)
        assert not backup.requests

    asyncio.run(scenario())


@pytest.mark.parametrize("changed", [None, "digest", "progress", "step", "candidate"])
def test_live_fallback_handoff_binds_exact_predecessor(changed):
    async def scenario():
        store = _StageMemoryStore()
        initial, identity = await _initial_admission(store)
        prepared = await _prepare(store, initial, identity)
        context = initial.invocation_context
        source = initial.successor
        failure = ModelProviderError(
            "private diagnostic", provider="primary", status_code=503, retryable=True
        )
        fallback = ModelFailoverTransition(
            source=source,
            source_preparation_digest=prepared.stage.preparation_digest,
            failure=failure,
            retry=retry_decision(
                policy=RetryPolicy(max_attempts=1),
                attempt=1,
                error="unavailable",
                status_code=503,
                retryable=True,
            ),
            observation=FailoverObservation(
                provider_name="primary",
                caller_cancelled=False,
                completion_observed=False,
                provider_effect_observed=False,
                provider_operation_owned=False,
                cleanup_settled=True,
            ),
        )
        failure.retryable = False
        assert "private diagnostic" not in repr(fallback)
        with pytest.raises(TypeError, match="serialization"):
            pickle.dumps(fallback)
        selection = ModelExecutionSelection(
            invocation_context=context,
            resolution=ModelFailoverProfileResolution(
                plan=source.plan,
                candidate_profiles=initial.candidate_profiles,
                registered_providers=(
                    context.registered_provider,
                    RegisteredProvider("backup", ScriptedModelProvider([], name="backup")),
                ),
            ),
            candidate_index=0 if changed == "candidate" else 1,
        )
        next_attempt = ModelFailoverAttempt(
            selection=selection,
            identity=new_model_step_identity().new_attempt()
            if changed == "step"
            else identity.new_attempt(),
            prior_provider_effect_observed=False,
        )
        previous = (
            source
            if changed != "progress"
            else source.model_copy(update={"generation": source.generation + 1})
        )
        before = await store.load_checkpoint("session")
        with nullcontext() if changed is None else pytest.raises(ValueError, match="predecessor"):
            admission = next_attempt.stage_admission(
                previous=previous,
                source_preparation_digest="f" * 64
                if changed == "digest"
                else prepared.stage.preparation_digest,
                source_transcript_cursor=source.source_transcript_cursor,
                request_fingerprint="e" * 64,
                fallback=fallback,
            )
            assert admission.transition == "fallback"
            assert admission.successor.candidate_index == 1
            assert admission.successor.attempts_used == 2
            assert isinstance(admission.failure, ModelProviderError)
            assert admission.failure.retryable is True
        assert await store.load_checkpoint("session") == before

    asyncio.run(scenario())


async def _selected_run_fixture(
    store,
    provider,
    backup,
    *,
    counting=False,
    retry_attempts=2,
    context_overflow_policy=None,
    max_total_attempts=5,
):
    admission, _ = await _initial_admission(
        store,
        provider=provider,
        context_overflow_policy=context_overflow_policy,
        max_total_attempts=max_total_attempts,
    )
    context = admission.invocation_context
    selection = ModelExecutionSelection(
        invocation_context=context,
        resolution=ModelFailoverProfileResolution(
            plan=admission.successor.plan,
            candidate_profiles=admission.candidate_profiles,
            registered_providers=(
                context.registered_provider,
                RegisteredProvider("backup", backup),
            ),
        ),
        candidate_index=0,
    )
    session = await store.load("session")
    assert session is not None
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        context_counting=ContextCountingConfig(
            mode=ContextCountingMode.OBSERVE if counting else ContextCountingMode.OFF,
        ),
    )
    observed = []
    run = app._model_step_executor.create_run(
        provider=provider,
        session=session,
        registered_agent=context.registered_agent,
        registered_provider=context.registered_provider,
        registered_environment=None,
        environment_name=None,
        structured_output=None,
        thinking=None,
        knowledge_store=None,
        knowledge_access_scope=None,
        request_metadata={},
        retry_policy=RetryPolicy(max_attempts=retry_attempts, initial_delay_s=0),
        request_budget_limits=(),
        limit_gate=RunLimitGate(
            app._run_limit_controller,
            session=session,
            agent_name="agent",
            environment_name=None,
            limits=RunLimits(),
            budget_limits=(),
            run_started_at=time.monotonic(),
            run_baseline=None,
            budget_baseline_events=[],
            budget_notify_events=[],
        ),
        budget_policy=None,
        run_started_at=time.monotonic(),
        turn_usage_tracker=None,
        active_run=None,
        execution_profile=context.profile,
        invocation_context=context,
        validate_live_model_semantics=lambda: None,
        model_execution_selection=selection,
        model_completion_recovery_context_factory=lambda billing, _reservations: (
            ModelCompletionRecoveryContext(
                interaction_id=context.binding.interaction_id,
                billing_identity=billing,
                execution_profile_fingerprint=context.profile.fingerprint,
            )
        ),
        model_completion_publisher=_atomic_publisher(
            store,
            expected_run_epoch=session.run_epoch,
            observed=observed,
        ),
    )
    return app, run, context, session, observed


@pytest.mark.parametrize("outcome", ["recovered", "fallback", "partial", "cap", "overflow_again"])
def test_selected_context_overflow_preserves_attempts_and_effect_evidence(outcome):
    class OverflowProvider(ModelProvider):
        name = "primary"

        def __init__(self):
            self.requests = []

        def preflight_portable_messages(self, *, model, messages, tools):
            from cayu.providers.base import _preflight_provider_portable_messages

            _preflight_provider_portable_messages(
                model=model,
                messages=messages,
                tools=tools,
                supports_system_messages=True,
                supports_tool_history=True,
                supports_tool_definitions=True,
                supports_file_attachments=True,
            )

        async def stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1 or outcome == "overflow_again":
                if outcome == "partial":
                    yield ModelStreamEvent.text_delta("incomplete")
                raise ModelContextOverflowError(
                    "context overflow", provider="primary", status_code=400
                )
            if outcome == "recovered":
                yield ModelStreamEvent.text_delta("done")
                yield ModelStreamEvent.completed()
                return
            raise ModelProviderError(
                "unavailable", provider="primary", status_code=503, retryable=True
            )

    async def scenario():
        store = _StageMemoryStore()
        provider = OverflowProvider()
        backup = ScriptedModelProvider(
            [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()], name="backup"
        )
        _, run, context, session, observed = await _selected_run_fixture(
            store,
            provider,
            backup,
            context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
            max_total_attempts=1 if outcome == "cap" else 5,
        )
        await store.append_transcript_messages(
            session.id, [Message.text("user", "old"), Message.text("user", "new")]
        )
        step = new_model_step_identity()
        events = []
        succeeds = outcome in {"recovered", "fallback"}
        final_result = None
        with nullcontext() if succeeds else pytest.raises(ModelProviderError) as caught:
            async for event, result in run.execute(
                step=1,
                messages=await store.load_transcript(session.id),
                source_transcript_cursor=await store.load_transcript_cursor(session.id),
                model_step_identity=step,
            ):
                if event is not None:
                    events.append(event)
                if result is not None:
                    final_result = result
        assert (final_result is not None) is succeeds
        if not succeeds:
            assert caught is not None
            assert isinstance(caught.value, ModelContextOverflowError) is (
                outcome in {"cap", "overflow_again"}
            )
        expected_primary = (
            1 if outcome == "cap" else 2 if outcome in {"recovered", "overflow_again"} else 3
        )
        assert len(provider.requests) == expected_primary
        assert len(backup.requests) == int(outcome == "fallback")
        assert all(request.model == "small" for request in provider.requests)
        if expected_primary > 1:
            assert len(provider.requests[1].messages) < len(provider.requests[0].messages)
        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is not None
        progress = checkpoint["model_failover"]
        assert progress["attempts_used"] == expected_primary + len(backup.requests)
        assert progress["candidate_attempt"] == (1 if outcome == "fallback" else expected_primary)
        assert progress["provider_effect_observed"] is (outcome == "partial")
        assert progress["candidate_index"] == int(outcome == "fallback")
        assert len(observed) == int(succeeds)
        assert (await store.load_active_model_completion_stage(session.id) is None) is succeeds
        assert session.model == "small" and run.execution_profile is context.profile
        assert [event.type for event in events].count(EventType.CONTEXT_OVERFLOW_RECOVERING) == int(
            outcome != "cap"
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "primary_succeeds,backup_succeeds,failure_mode",
    [
        (True, False, "retryable"),
        (False, True, "retryable"),
        (False, False, "retryable"),
        (False, True, "partial_output"),
        (False, True, "permanent"),
        (False, True, "cancel"),
        (False, False, "chain_cap"),
        (False, False, "cap_at_primary"),
        (False, True, "settlement_failure"),
    ],
)
@pytest.mark.parametrize("counting", [False, True])
def test_model_step_run_prepares_retries_and_settles_routed_attempts(
    primary_succeeds, backup_succeeds, failure_mode, counting, monkeypatch
):
    should_fallback = not primary_succeeds and failure_mode in {"retryable", "chain_cap"}
    succeeds = primary_succeeds or (should_fallback and backup_succeeds)

    async def scenario():
        store = _StageMemoryStore()
        failures = []
        cancellation_ready = asyncio.Event()
        provider_stopped = asyncio.Event()

        class CandidateProvider(ScriptedModelProvider):
            async def stream(self, request):
                if failure_mode == "cancel":
                    self.requests.append(request)
                    cancellation_ready.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        provider_stopped.set()
                    return
                if not self.requests and failure_mode == "partial_output":
                    yield ModelStreamEvent.text_delta("incomplete")
                async for event in super().stream(request):
                    yield event

        def respond(_request):
            if len(provider.requests) == 1 or not primary_succeeds:
                failure = ModelProviderError(
                    "unavailable",
                    provider="primary",
                    status_code=503,
                    retryable=failure_mode != "permanent",
                )
                failures.append(failure)
                raise failure
            return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]

        provider = CandidateProvider(name="primary", response_factory=respond)

        def backup_respond(_request):
            if not backup_succeeds:
                failure = ModelProviderError(
                    "backup unavailable", provider="backup", status_code=503, retryable=True
                )
                failures.append(failure)
                raise failure
            return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]

        backup = ScriptedModelProvider(name="backup", response_factory=backup_respond)
        app, run, context, session, observed = await _selected_run_fixture(
            store,
            provider,
            backup,
            counting=counting,
            retry_attempts=3
            if failure_mode == "chain_cap"
            else 6
            if failure_mode == "cap_at_primary"
            else 2,
        )
        settlement_failure = OSError("settlement unavailable")
        if failure_mode == "settlement_failure":
            original_settle = app._run_limit_controller.settle_after_model_failure

            async def settle(*args, **kwargs):
                if kwargs.get("release_reason") == "selected model candidate exhausted":
                    raise settlement_failure
                async for event in original_settle(*args, **kwargs):
                    yield event

            monkeypatch.setattr(app._run_limit_controller, "settle_after_model_failure", settle)
        events = []
        outcome = None
        step = new_model_step_identity()

        async def consume():
            nonlocal outcome
            stream = run.execute(
                step=1,
                messages=await store.load_transcript("session"),
                source_transcript_cursor=await store.load_transcript_cursor("session"),
                model_step_identity=step,
            )
            try:
                async for event, result in stream:
                    if event is not None:
                        events.append(event)
                    if result is not None:
                        outcome = result
            finally:
                await stream.aclose()

        if failure_mode == "cancel":
            caller = asyncio.create_task(consume())
            await asyncio.wait_for(cancellation_ready.wait(), 10)
            caller.cancel()
            assert caller.cancelling() == 1
            try:
                await caller
            except asyncio.CancelledError:
                pass
            else:
                pytest.fail("Caller cancellation was swallowed.")
            assert caller.cancelled() and caller.cancelling() == 1
            assert provider_stopped.is_set()
        elif failure_mode == "settlement_failure":
            with pytest.raises(ExceptionGroup) as grouped:
                await consume()
            leaves = [
                item
                for item in iter_exception_tree(grouped.value)
                if not isinstance(item, BaseExceptionGroup)
            ]
            assert len(leaves) == 2
            assert isinstance(leaves[0], ModelProviderError)
            assert leaves[0].provider == "primary" and leaves[0].status_code == 503
            assert leaves[1] is settlement_failure
        else:
            with (
                nullcontext()
                if succeeds
                else pytest.raises(ModelProviderError, match="unavailable")
            ) as caught:
                await consume()
            if not succeeds:
                # The existing credential boundary detaches provider errors.
                # Preserve that public typed contract, not the raw SDK object.
                assert caught is not None
                assert caught.value.provider == failures[-1].provider
                assert caught.value.status_code == failures[-1].status_code
                assert caught.value.retryable == failures[-1].retryable
        assert (outcome is not None) is succeeds
        if not succeeds:
            assert "incomplete" not in str(await store.load_transcript("session"))
        primary_attempts = (
            1
            if failure_mode in {"permanent", "cancel"}
            else 3
            if failure_mode == "chain_cap"
            else 5
            if failure_mode == "cap_at_primary"
            else 2
        )
        assert len(provider.requests) == primary_attempts
        backup_attempts = 0 if not should_fallback else 1 if backup_succeeds else 2
        assert len(backup.requests) == backup_attempts
        assert all(request.model == "small" for request in provider.requests)
        assert all(request.model == "large" for request in backup.requests)
        assert session.provider_name == "primary" and session.model == "small"
        assert run.execution_profile is context.profile
        assert len(observed) == int(succeeds)
        active = await store.load_active_model_completion_stage("session")
        assert (active is None) is succeeds
        if succeeds:
            assert (await store.load_transcript("session"))[-1].content == Message.text(
                "assistant", "done"
            ).content
        checkpoint = await store.load_checkpoint("session")
        assert checkpoint is not None
        assert checkpoint["model_failover"]["attempts_used"] == primary_attempts + backup_attempts
        assert checkpoint["model_failover"]["candidate_index"] == int(should_fallback)
        assert (
            checkpoint["model_failover"]["stage_id"]
            == f"{step.model_step_id}:dispatch:{primary_attempts + backup_attempts - 1}"
        )
        assert [event.type for event in events].count(EventType.MODEL_FAILOVER_SELECTED) == 1 + int(
            should_fallback
        )
        assert [event.type for event in events].count(EventType.MODEL_COMPLETED) == int(succeeds)
        assert [event.type for event in events].count(EventType.MODEL_RETRY) == (
            primary_attempts - 1 + max(0, backup_attempts - 1)
        )
        if counting:
            types = [event.type for event in events]
            assert types.index(EventType.MODEL_FAILOVER_SELECTED) < types.index(
                EventType.CONTEXT_COUNTED
            )
            assert types.index(EventType.CONTEXT_COUNTED) < types.index(EventType.MODEL_STARTED)

    asyncio.run(scenario())
