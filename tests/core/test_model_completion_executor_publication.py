from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from cayu._exception_groups import iter_exception_tree
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    RetryPolicy,
    RunLimits,
    RunRequest,
    Session,
    SessionIdentity,
    SessionStatus,
)
from cayu.runtime._model_completion_publication import (
    LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    ModelStepPublicationCheckpoint,
)
from cayu.runtime._model_step_executor import (
    ModelCompletionDispatchNotAuthorized,
    ModelCompletionPublicationRequest,
    ModelCompletionPublicationResult,
    ModelStepRun,
)
from cayu.runtime._run_limits import RunLimitGate
from cayu.runtime.execution_units import new_model_step_identity
from cayu.runtime.sessions import (
    RuntimePublicationRequest,
    runtime_publication_checkpoint_mutation,
)


class _RetryThenCompleteProvider(ModelProvider):
    name = "retry-then-complete"

    def __init__(self, store: InMemorySessionStore, session_id: str) -> None:
        self.store = store
        self.session_id = session_id
        self.calls = 0
        self.active_dispatch_ordinals: list[int] = []

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        call = self.calls

        async def events() -> AsyncIterator[ModelStreamEvent]:
            active = await self.store.load_active_model_completion_stage(self.session_id)
            assert active is not None
            self.active_dispatch_ordinals.append(active.stage.dispatch_ordinal)
            if call == 1:
                raise ModelProviderError(
                    "retry this dispatch",
                    provider=self.name,
                    retryable=True,
                )
            yield ModelStreamEvent.text_delta("durable answer")
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "total_tokens": 5,
                    },
                }
            )

        return events()


class _CompletedThenLateProvider(ModelProvider):
    name = "completed-then-late"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        yield ModelStreamEvent.text_delta("must not become authoritative")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 3,
                    "total_tokens": 7,
                },
            }
        )
        yield ModelStreamEvent.text_delta("illegal late frame")


class _NeverDispatchedProvider(ModelProvider):
    name = "never-dispatched"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _CompletedThenWaitProvider(ModelProvider):
    name = "completed-then-wait"

    def __init__(self) -> None:
        self.waiting_after_completion = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("cancelled answer")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 2,
                    "total_tokens": 4,
                },
            }
        )
        self.waiting_after_completion.set()
        await self.release.wait()


class _CompletedThenRetryableFailureProvider(ModelProvider):
    name = "completed-then-retryable-failure"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        yield ModelStreamEvent.text_delta("terminal answer")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 2,
                    "total_tokens": 4,
                },
            }
        )
        raise ModelProviderError(
            "retryable transport failure after completion",
            provider=self.name,
            status_code=503,
            retryable=True,
        )


class _CompletedThenGroupedFailureProvider(ModelProvider):
    name = "completed-then-grouped-failure"

    def __init__(self) -> None:
        self.calls = 0
        self.failure = BaseExceptionGroup(
            "provider iteration failed after completion",
            [
                asyncio.CancelledError("provider iteration cancelled"),
                RuntimeError("provider iteration cleanup failed"),
            ],
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        yield ModelStreamEvent.text_delta("completed grouped answer")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                },
            }
        )
        raise self.failure


class _CompletedThenChildCancellationProvider(ModelProvider):
    name = "completed-then-child-cancellation"

    def __init__(self) -> None:
        self.calls = 0
        self.failure = asyncio.CancelledError("provider child cancelled itself")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        yield ModelStreamEvent.text_delta("completed child-cancellation answer")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                },
            }
        )
        raise self.failure


class _CompletedThenOrdinaryGroupedFailureProvider(ModelProvider):
    name = "completed-then-ordinary-grouped-failure"

    def __init__(self) -> None:
        self.calls = 0
        self.failure = ExceptionGroup(
            "provider iteration failed after completion",
            [
                RuntimeError("provider iteration failed"),
                RuntimeError("provider iterator cleanup failed"),
            ],
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        yield ModelStreamEvent.text_delta("completed ordinary grouped answer")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                },
            }
        )
        raise self.failure


class _CompletedThenGeneratorExitProvider(ModelProvider):
    name = "completed-then-generator-exit"

    def __init__(self) -> None:
        self.calls = 0
        self.failure = GeneratorExit("provider stream closed after completion")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        yield ModelStreamEvent.text_delta("completed generator-exit answer")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                },
            }
        )
        raise self.failure


class _CompletedThenGeneratorExitGroupProvider(ModelProvider):
    name = "completed-then-generator-exit-group"

    def __init__(self) -> None:
        self.calls = 0
        self.failure = BaseExceptionGroup(
            "provider stream closed after completion",
            [
                GeneratorExit("provider stream closed"),
                RuntimeError("provider stream cleanup failed"),
            ],
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        yield ModelStreamEvent.text_delta("completed grouped generator-exit answer")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                },
            }
        )
        raise self.failure


class _GroupedCloseFailureStream(AsyncIterator[ModelStreamEvent]):
    def __init__(self) -> None:
        self._index = 0
        self.waiting_after_completion = asyncio.Event()
        self.close_calls = 0
        self.cleanup_failure = BaseExceptionGroup(
            "provider iterator close failed after completion",
            [
                asyncio.CancelledError("provider iterator close cancelled"),
                RuntimeError("provider iterator close failed"),
            ],
        )

    def __aiter__(self) -> _GroupedCloseFailureStream:
        return self

    async def __anext__(self) -> ModelStreamEvent:
        if self._index == 0:
            self._index += 1
            return ModelStreamEvent.text_delta("completed close answer")
        if self._index == 1:
            self._index += 1
            return ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "total_tokens": 10,
                    },
                }
            )
        self.waiting_after_completion.set()
        await asyncio.Event().wait()
        raise AssertionError("Blocked provider iterator unexpectedly resumed.")

    async def aclose(self) -> None:
        self.close_calls += 1
        raise self.cleanup_failure


class _CompletedThenGroupedCloseFailureProvider(ModelProvider):
    name = "completed-then-grouped-close-failure"

    def __init__(self) -> None:
        self.calls = 0
        self.events = _GroupedCloseFailureStream()

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        return self.events


class _GeneratorExitCloseFailureStream(AsyncIterator[ModelStreamEvent]):
    def __init__(self) -> None:
        self._index = 0
        self.close_calls = 0
        self.failure = GeneratorExit("provider aclose failed after completion")

    def __aiter__(self) -> _GeneratorExitCloseFailureStream:
        return self

    async def __anext__(self) -> ModelStreamEvent:
        if self._index == 0:
            self._index += 1
            return ModelStreamEvent.text_delta("completed aclose answer")
        if self._index == 1:
            self._index += 1
            return ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "total_tokens": 10,
                    },
                }
            )
        raise RuntimeError("provider iteration failed after completion")

    async def aclose(self) -> None:
        self.close_calls += 1
        raise self.failure


class _CompletedThenGeneratorExitCloseFailureProvider(ModelProvider):
    name = "completed-then-generator-exit-close-failure"

    def __init__(self) -> None:
        self.calls = 0
        self.events = _GeneratorExitCloseFailureStream()

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        return self.events


class _AcloseLookupFailureStream(AsyncIterator[ModelStreamEvent]):
    def __init__(self) -> None:
        self._index = 0
        self.waiting_after_completion = asyncio.Event()
        self.close_lookups = 0

    def __aiter__(self) -> _AcloseLookupFailureStream:
        return self

    async def __anext__(self) -> ModelStreamEvent:
        if self._index == 0:
            self._index += 1
            return ModelStreamEvent.text_delta("completed lookup answer")
        if self._index == 1:
            self._index += 1
            return ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "total_tokens": 10,
                    },
                }
            )
        self.waiting_after_completion.set()
        await asyncio.Event().wait()
        raise AssertionError("Blocked provider iterator unexpectedly resumed.")

    @property
    def aclose(self):
        self.close_lookups += 1
        raise RuntimeError("provider-controlled aclose lookup failed")


class _CompletedThenAcloseLookupFailureProvider(ModelProvider):
    name = "completed-then-aclose-lookup-failure"

    def __init__(self) -> None:
        self.calls = 0
        self.events = _AcloseLookupFailureStream()

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        return self.events


class _PrepareAcknowledgementLostStore(InMemorySessionStore):
    """Commit preparation, then return the exact replay as an ambiguous acknowledgement."""

    def __init__(self) -> None:
        super().__init__()
        self.preparations = 0

    async def prepare_model_completion_stage(self, session_id: str, **kwargs):
        self.preparations += 1
        await super().prepare_model_completion_stage(session_id, **kwargs)
        return await super().prepare_model_completion_stage(session_id, **kwargs)


def _limit_gate(app: CayuApp, session: Session) -> RunLimitGate:
    return RunLimitGate(
        app._run_limit_controller,
        session=session,
        agent_name="assistant",
        environment_name=None,
        limits=RunLimits(),
        budget_limits=(),
        run_started_at=time.monotonic(),
        run_baseline=None,
        budget_baseline_events=[],
        budget_notify_events=[],
    )


async def _create_model_run(
    *,
    store: InMemorySessionStore,
    provider: ModelProvider,
    session_id: str,
    publisher,
    retry_policy: RetryPolicy,
) -> tuple[ModelStepRun, ModelRequest, Session, Message]:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    user_message = Message.text("user", "answer durably")
    session = await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[user_message],
        ),
        identity=SessionIdentity(provider_name=provider.name, model="fake-model"),
    )
    await store.append_transcript_messages(session.id, [user_message])
    session = await store.transition_status(
        session.id,
        from_statuses={SessionStatus.PENDING},
        to_status=SessionStatus.RUNNING,
    )
    registered_agent = app._agents["assistant"]
    registered_provider = app._providers[provider.name]
    request = await app._model_step_executor.build_request(
        session=session,
        registered_agent=registered_agent,
        registered_environment=None,
        context_messages=[user_message],
        structured_output=None,
        thinking=None,
        step=1,
    )
    run = app._model_step_executor.create_run(
        provider=provider,
        session=session,
        registered_agent=registered_agent,
        registered_provider=registered_provider,
        registered_environment=None,
        environment_name=None,
        structured_output=None,
        thinking=None,
        knowledge_store=None,
        knowledge_access_scope=None,
        request_metadata={},
        retry_policy=retry_policy,
        request_budget_limits=(),
        limit_gate=_limit_gate(app, session),
        budget_policy=None,
        run_started_at=time.monotonic(),
        turn_usage_tracker=None,
        active_run=None,
        validate_live_model_semantics=lambda: None,
        model_completion_publisher=publisher,
    )
    return run, request, session, user_message


def _atomic_publisher(
    store: InMemorySessionStore,
    *,
    expected_run_epoch: int,
    observed: list[ModelCompletionPublicationRequest],
):
    async def publish(
        request: ModelCompletionPublicationRequest,
    ) -> ModelCompletionPublicationResult:
        observed.append(request)
        source_checkpoint = await store.load_checkpoint(request.dispatch.stage.session_id)
        target_checkpoint = dict(source_checkpoint or {})
        classification = request.completion_event.payload["step_classification"]
        target_checkpoint[LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY] = (
            ModelStepPublicationCheckpoint(
                logical_step_id=request.dispatch.logical_step_id,
                stage_id=request.dispatch.stage_id,
                source_transcript_cursor=request.dispatch.stage.source_transcript_cursor,
                transcript_end_cursor=(
                    request.dispatch.stage.source_transcript_cursor
                    + int(request.authoritative_assistant_message is not None)
                ),
                completion_event_id=request.completion_event.id,
                classification=classification,
                assistant_message_published=(request.authoritative_assistant_message is not None),
            ).model_dump(mode="json")
        )
        publication = RuntimePublicationRequest(
            publication_id=request.dispatch.logical_step_id,
            kind="model-step",
            intent=request.dispatch.intent,
            mutation=runtime_publication_checkpoint_mutation(
                source_checkpoint,
                target_checkpoint,
            ),
            transcript_messages=(
                ()
                if request.authoritative_assistant_message is None
                else (request.authoritative_assistant_message,)
            ),
            events=(request.completion_event,),
        )
        completion = await store.complete_model_completion_stage(
            request.dispatch.stage.session_id,
            stage_id=request.dispatch.stage_id,
            publication=publication,
        )
        promoted = await store.promote_model_completion_stage(
            request.dispatch.stage.session_id,
            stage_id=request.dispatch.stage_id,
            expected_run_epoch=expected_run_epoch,
        )
        return ModelCompletionPublicationResult(
            completion=completion,
            publication=promoted,
        )

    return publish


def test_model_executor_stages_each_retry_and_promotes_before_returning_result() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _RetryThenCompleteProvider(store, "sess_model_stage_retry")
        observed: list[ModelCompletionPublicationRequest] = []

        # The publisher needs the run epoch, which is stable after the transition.
        publisher_slot = {}

        async def publisher(request: ModelCompletionPublicationRequest):
            return await publisher_slot["publish"](request)

        model_run, model_request, session, user_message = await _create_model_run(
            store=store,
            provider=provider,
            session_id="sess_model_stage_retry",
            publisher=publisher,
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
        )
        publisher_slot["publish"] = _atomic_publisher(
            store,
            expected_run_epoch=session.run_epoch,
            observed=observed,
        )

        events: list[Event] = []
        outcome = None
        model_step_identity = new_model_step_identity()
        async for event, candidate in model_run._execute_request(
            model_request=model_request,
            step=1,
            messages=[user_message],
            model_step_identity=model_step_identity,
        ):
            if event is not None:
                events.append(event)
            if candidate is not None:
                outcome = candidate
                durable_transcript = await store.load_transcript(session.id)
                assert durable_transcript[-1].role == "assistant"

        logical_step_id = model_step_identity.model_step_id
        first_stage = await store.load_model_completion_stage(
            session.id,
            f"{logical_step_id}:dispatch:0",
        )
        second_stage = await store.load_model_completion_stage(
            session.id,
            f"{logical_step_id}:dispatch:1",
        )
        assert second_stage is not None
        assert second_stage.intent["requested_model"] == model_request.model
        return (
            store,
            provider,
            session,
            user_message,
            events,
            outcome,
            observed,
            first_stage,
            second_stage,
        )

    (
        store,
        provider,
        session,
        user_message,
        events,
        outcome,
        observed,
        first_stage,
        second_stage,
    ) = asyncio.run(run())

    assert provider.calls == 2
    assert provider.active_dispatch_ordinals == [0, 1]
    assert first_stage is not None
    assert first_stage.state == "in_flight"
    assert first_stage.dispatch_ordinal == 0
    assert second_stage is not None
    assert second_stage.state == "completed"
    assert second_stage.dispatch_ordinal == 1
    assert first_stage.logical_step_id == second_stage.logical_step_id
    assert first_stage.intent["request_fingerprint"] == second_stage.intent["request_fingerprint"]
    assert len(observed) == 1
    assert observed[0].dispatch.stage_id == second_stage.stage_id
    assert observed[0].authoritative_assistant_message is not None
    assert outcome is not None
    assert outcome.assistant_step_result is not None
    assert outcome.assistant_step_result.text_content == "durable answer"
    assert [event.type for event in events].count(EventType.MODEL_COMPLETED) == 1
    assert asyncio.run(store.load_transcript(session.id)) == [
        user_message,
        observed[0].authoritative_assistant_message,
    ]


def test_model_executor_publishes_non_turn_evidence_for_late_provider_frame() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _CompletedThenLateProvider()
        observed: list[ModelCompletionPublicationRequest] = []
        publisher_slot = {}

        async def publisher(request: ModelCompletionPublicationRequest):
            return await publisher_slot["publish"](request)

        model_run, model_request, session, user_message = await _create_model_run(
            store=store,
            provider=provider,
            session_id="sess_model_stage_late_frame",
            publisher=publisher,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        publisher_slot["publish"] = _atomic_publisher(
            store,
            expected_run_epoch=session.run_epoch,
            observed=observed,
        )

        with pytest.raises(
            RuntimeError,
            match="Model provider emitted event after completed: text_delta",
        ):
            async for _ in model_run._execute_request(
                model_request=model_request,
                step=1,
                messages=[user_message],
                model_step_identity=new_model_step_identity(),
            ):
                pass
        return store, provider, session, user_message, observed

    store, provider, session, user_message, observed = asyncio.run(run())

    assert provider.calls == 1
    assert len(observed) == 1
    assert observed[0].assistant_step_result.text_content == ("must not become authoritative")
    assert observed[0].authoritative_assistant_message is None
    assert observed[0].completion_event.payload["step_classification"]["type"] == "failed"
    assert observed[0].completion_event.payload["transcript_cursor"] == 1
    assert asyncio.run(store.load_transcript(session.id)) == [user_message]
    durable_completions = [
        event
        for event in asyncio.run(store.load_events(session.id))
        if event.type == EventType.MODEL_COMPLETED
    ]
    assert durable_completions == [observed[0].completion_event]


def test_model_executor_does_not_dispatch_after_ambiguous_prepare_acknowledgement() -> None:
    async def run():
        store = _PrepareAcknowledgementLostStore()
        provider = _NeverDispatchedProvider()
        publication_calls = 0

        async def publisher(
            request: ModelCompletionPublicationRequest,
        ) -> ModelCompletionPublicationResult:
            nonlocal publication_calls
            publication_calls += 1
            raise AssertionError("No completion can publish without provider dispatch.")

        model_run, model_request, session, user_message = await _create_model_run(
            store=store,
            provider=provider,
            session_id="sess_model_stage_prepare_ack_loss",
            publisher=publisher,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        with pytest.raises(ModelCompletionDispatchNotAuthorized):
            async for _ in model_run._execute_request(
                model_request=model_request,
                step=1,
                messages=[user_message],
                model_step_identity=new_model_step_identity(),
            ):
                pass
        return store, provider, session, publication_calls

    store, provider, session, publication_calls = asyncio.run(run())

    assert store.preparations == 1
    assert provider.calls == 0
    assert publication_calls == 0
    active = asyncio.run(store.load_active_model_completion_stage(session.id))
    assert active is not None
    assert active.stage.state == "in_flight"


def test_model_executor_preserves_cancellation_after_non_turn_publication() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _CompletedThenWaitProvider()
        observed: list[ModelCompletionPublicationRequest] = []
        publisher_slot = {}

        async def publisher(request: ModelCompletionPublicationRequest):
            return await publisher_slot["publish"](request)

        model_run, model_request, session, user_message = await _create_model_run(
            store=store,
            provider=provider,
            session_id="sess_model_stage_cancel_after_completion",
            publisher=publisher,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        publisher_slot["publish"] = _atomic_publisher(
            store,
            expected_run_epoch=session.run_epoch,
            observed=observed,
        )

        async def consume() -> None:
            async for _ in model_run._execute_request(
                model_request=model_request,
                step=1,
                messages=[user_message],
                model_step_identity=new_model_step_identity(),
            ):
                pass

        task = asyncio.create_task(consume())
        await provider.waiting_after_completion.wait()
        task.cancel("authoritative test cancellation")
        task.cancel("repeated authoritative test cancellation")
        with pytest.raises(asyncio.CancelledError, match="authoritative test cancellation"):
            await task
        return (
            store,
            session,
            user_message,
            observed,
            task.cancelling(),
            task.cancelled(),
        )

    store, session, user_message, observed, cancelling, cancelled = asyncio.run(run())

    assert cancelling == 0
    assert cancelled is True
    assert len(observed) == 1
    assert observed[0].authoritative_assistant_message is None
    assert observed[0].completion_event.payload["step_classification"] == {
        "type": "failed",
        "reason": "model stream was cancelled before terminal validation completed",
    }
    assert observed[0].completion_event.payload["transcript_cursor"] == 1
    assert asyncio.run(store.load_transcript(session.id)) == [user_message]
    durable_completions = [
        event
        for event in asyncio.run(store.load_events(session.id))
        if event.type == EventType.MODEL_COMPLETED
    ]
    assert durable_completions == [observed[0].completion_event]


def test_model_executor_publishes_completion_before_grouped_iteration_failure() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _CompletedThenGroupedFailureProvider()
        observed: list[ModelCompletionPublicationRequest] = []
        publisher_slot = {}

        async def publisher(request: ModelCompletionPublicationRequest):
            return await publisher_slot["publish"](request)

        model_run, model_request, session, user_message = await _create_model_run(
            store=store,
            provider=provider,
            session_id="sess_model_stage_grouped_iteration_failure",
            publisher=publisher,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        publisher_slot["publish"] = _atomic_publisher(
            store,
            expected_run_epoch=session.run_epoch,
            observed=observed,
        )

        with pytest.raises(BaseExceptionGroup) as raised:
            async for _ in model_run._execute_request(
                model_request=model_request,
                step=1,
                messages=[user_message],
                model_step_identity=new_model_step_identity(),
            ):
                pass
        return store, provider, session, user_message, observed, raised.value

    store, provider, session, user_message, observed, raised = asyncio.run(run())

    assert raised is provider.failure
    assert provider.calls == 1
    assert len(observed) == 1
    publication = observed[0]
    assert publication.authoritative_assistant_message is None
    assert publication.completion_event.payload["step_classification"] == {
        "type": "failed",
        "reason": "model completion failed terminal validation",
    }
    usage = publication.completion_event.payload["usage"]
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == 10
    assert asyncio.run(store.load_transcript(session.id)) == [user_message]
    assert asyncio.run(store.load_active_model_completion_stage(session.id)) is None
    durable_completions = [
        event
        for event in asyncio.run(store.load_events(session.id))
        if event.type == EventType.MODEL_COMPLETED
    ]
    assert durable_completions == [publication.completion_event]


def test_model_executor_does_not_forge_caller_cancellation_from_provider_child() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _CompletedThenChildCancellationProvider()
        observed: list[ModelCompletionPublicationRequest] = []
        publisher_slot = {}

        async def publisher(request: ModelCompletionPublicationRequest):
            return await publisher_slot["publish"](request)

        model_run, model_request, session, user_message = await _create_model_run(
            store=store,
            provider=provider,
            session_id="sess_model_stage_child_cancellation",
            publisher=publisher,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        publisher_slot["publish"] = _atomic_publisher(
            store,
            expected_run_epoch=session.run_epoch,
            observed=observed,
        )

        async def consume() -> None:
            async for _ in model_run._execute_request(
                model_request=model_request,
                step=1,
                messages=[user_message],
                model_step_identity=new_model_step_identity(),
            ):
                pass

        task = asyncio.create_task(consume())
        with pytest.raises(RuntimeError, match="without caller cancellation") as raised:
            await task
        return (
            store,
            provider,
            session,
            user_message,
            observed,
            raised.value,
            task.cancelling(),
            task.cancelled(),
        )

    (
        store,
        provider,
        session,
        user_message,
        observed,
        raised,
        cancelling,
        cancelled,
    ) = asyncio.run(run())

    assert raised.__cause__ is provider.failure
    assert cancelling == 0
    assert cancelled is False
    assert provider.calls == 1
    assert len(observed) == 1
    publication = observed[0]
    assert publication.authoritative_assistant_message is None
    assert publication.completion_event.payload["step_classification"] == {
        "type": "failed",
        "reason": "model completion failed terminal validation",
    }
    usage = publication.completion_event.payload["usage"]
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == 10
    assert asyncio.run(store.load_transcript(session.id)) == [user_message]
    assert asyncio.run(store.load_active_model_completion_stage(session.id)) is None


@pytest.mark.parametrize(
    "provider_factory",
    [
        pytest.param(_CompletedThenGeneratorExitProvider, id="direct"),
        pytest.param(_CompletedThenGeneratorExitGroupProvider, id="grouped"),
    ],
)
def test_model_executor_publishes_completion_before_generator_exit(
    provider_factory,
) -> None:
    async def run():
        store = InMemorySessionStore()
        provider = provider_factory()
        observed: list[ModelCompletionPublicationRequest] = []
        publisher_slot = {}

        async def publisher(request: ModelCompletionPublicationRequest):
            return await publisher_slot["publish"](request)

        model_run, model_request, session, user_message = await _create_model_run(
            store=store,
            provider=provider,
            session_id=f"sess_model_stage_generator_exit_{provider.name}",
            publisher=publisher,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        publisher_slot["publish"] = _atomic_publisher(
            store,
            expected_run_epoch=session.run_epoch,
            observed=observed,
        )

        raised: BaseException | None = None
        try:
            async for _ in model_run._execute_request(
                model_request=model_request,
                step=1,
                messages=[user_message],
                model_step_identity=new_model_step_identity(),
            ):
                pass
        except BaseException as exc:
            raised = exc
        return store, provider, session, user_message, observed, raised

    store, provider, session, user_message, observed, raised = asyncio.run(run())

    assert raised is provider.failure
    assert provider.calls == 1
    assert len(observed) == 1
    publication = observed[0]
    assert publication.authoritative_assistant_message is None
    assert publication.completion_event.payload["step_classification"] == {
        "type": "failed",
        "reason": "model completion failed terminal validation",
    }
    usage = publication.completion_event.payload["usage"]
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == 10
    assert asyncio.run(store.load_transcript(session.id)) == [user_message]
    assert asyncio.run(store.load_active_model_completion_stage(session.id)) is None
    durable_completions = [
        event
        for event in asyncio.run(store.load_events(session.id))
        if event.type == EventType.MODEL_COMPLETED
    ]
    assert durable_completions == [publication.completion_event]


def test_model_executor_publishes_completion_before_generator_exit_from_aclose() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _CompletedThenGeneratorExitCloseFailureProvider()
        observed: list[ModelCompletionPublicationRequest] = []
        publisher_slot = {}

        async def publisher(request: ModelCompletionPublicationRequest):
            return await publisher_slot["publish"](request)

        model_run, model_request, session, user_message = await _create_model_run(
            store=store,
            provider=provider,
            session_id="sess_model_stage_generator_exit_aclose",
            publisher=publisher,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        publisher_slot["publish"] = _atomic_publisher(
            store,
            expected_run_epoch=session.run_epoch,
            observed=observed,
        )

        raised: BaseException | None = None
        try:
            async for _ in model_run._execute_request(
                model_request=model_request,
                step=1,
                messages=[user_message],
                model_step_identity=new_model_step_identity(),
            ):
                pass
        except BaseException as exc:
            raised = exc
        return store, provider, session, user_message, observed, raised

    store, provider, session, user_message, observed, raised = asyncio.run(run())

    assert raised is provider.events.failure
    assert provider.events.close_calls == 1
    assert provider.calls == 1
    assert len(observed) == 1
    publication = observed[0]
    assert publication.authoritative_assistant_message is None
    assert publication.completion_event.payload["step_classification"] == {
        "type": "failed",
        "reason": "model completion failed terminal validation",
    }
    usage = publication.completion_event.payload["usage"]
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == 10
    assert asyncio.run(store.load_transcript(session.id)) == [user_message]
    assert asyncio.run(store.load_active_model_completion_stage(session.id)) is None
    durable_completions = [
        event
        for event in asyncio.run(store.load_events(session.id))
        if event.type == EventType.MODEL_COMPLETED
    ]
    assert durable_completions == [publication.completion_event]


def test_model_executor_publishes_completion_before_ordinary_grouped_failure() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _CompletedThenOrdinaryGroupedFailureProvider()
        observed: list[ModelCompletionPublicationRequest] = []
        publisher_slot = {}

        async def publisher(request: ModelCompletionPublicationRequest):
            return await publisher_slot["publish"](request)

        model_run, model_request, session, user_message = await _create_model_run(
            store=store,
            provider=provider,
            session_id="sess_model_stage_ordinary_grouped_failure",
            publisher=publisher,
            retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
        )
        publisher_slot["publish"] = _atomic_publisher(
            store,
            expected_run_epoch=session.run_epoch,
            observed=observed,
        )

        with pytest.raises(ExceptionGroup) as raised:
            async for _ in model_run._execute_request(
                model_request=model_request,
                step=1,
                messages=[user_message],
                model_step_identity=new_model_step_identity(),
            ):
                pass
        return store, provider, session, user_message, observed, raised.value

    store, provider, session, user_message, observed, raised = asyncio.run(run())

    assert raised is provider.failure
    assert provider.calls == 1
    assert len(observed) == 1
    publication = observed[0]
    assert publication.authoritative_assistant_message is None
    assert publication.completion_event.payload["step_classification"] == {
        "type": "failed",
        "reason": "model completion failed terminal validation",
    }
    usage = publication.completion_event.payload["usage"]
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == 10
    assert asyncio.run(store.load_transcript(session.id)) == [user_message]
    assert asyncio.run(store.load_active_model_completion_stage(session.id)) is None
    durable_completions = [
        event
        for event in asyncio.run(store.load_events(session.id))
        if event.type == EventType.MODEL_COMPLETED
    ]
    assert durable_completions == [publication.completion_event]


def test_model_executor_publishes_completion_before_grouped_aclose_failure() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _CompletedThenGroupedCloseFailureProvider()
        observed: list[ModelCompletionPublicationRequest] = []
        publisher_slot = {}

        async def publisher(request: ModelCompletionPublicationRequest):
            return await publisher_slot["publish"](request)

        model_run, model_request, session, user_message = await _create_model_run(
            store=store,
            provider=provider,
            session_id="sess_model_stage_grouped_aclose_failure",
            publisher=publisher,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        publisher_slot["publish"] = _atomic_publisher(
            store,
            expected_run_epoch=session.run_epoch,
            observed=observed,
        )

        async def consume() -> None:
            async for _ in model_run._execute_request(
                model_request=model_request,
                step=1,
                messages=[user_message],
                model_step_identity=new_model_step_identity(),
            ):
                pass

        task = asyncio.create_task(consume())
        await provider.events.waiting_after_completion.wait()
        task.cancel("authoritative grouped-close cancellation")
        with pytest.raises(BaseExceptionGroup) as raised:
            await task
        return (
            store,
            provider,
            session,
            user_message,
            observed,
            raised.value,
            task.cancelling(),
            task.cancelled(),
        )

    (
        store,
        provider,
        session,
        user_message,
        observed,
        raised,
        cancelling,
        cancelled,
    ) = asyncio.run(run())

    assert cancelling == 0
    assert cancelled is False
    leaves = tuple(iter_exception_tree(raised))
    assert any(
        isinstance(error, asyncio.CancelledError)
        and error.args == ("authoritative grouped-close cancellation",)
        for error in leaves
    )
    assert provider.events.cleanup_failure in leaves
    assert provider.events.close_calls == 1
    assert provider.calls == 1
    assert len(observed) == 1
    publication = observed[0]
    assert publication.authoritative_assistant_message is None
    assert publication.completion_event.payload["step_classification"] == {
        "type": "failed",
        "reason": "model stream was cancelled before terminal validation completed",
    }
    usage = publication.completion_event.payload["usage"]
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == 10
    assert asyncio.run(store.load_transcript(session.id)) == [user_message]
    assert asyncio.run(store.load_active_model_completion_stage(session.id)) is None
    durable_completions = [
        event
        for event in asyncio.run(store.load_events(session.id))
        if event.type == EventType.MODEL_COMPLETED
    ]
    assert durable_completions == [publication.completion_event]


def test_model_executor_publishes_completion_when_aclose_lookup_fails() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _CompletedThenAcloseLookupFailureProvider()
        observed: list[ModelCompletionPublicationRequest] = []
        publisher_slot = {}

        async def publisher(request: ModelCompletionPublicationRequest):
            return await publisher_slot["publish"](request)

        model_run, model_request, session, user_message = await _create_model_run(
            store=store,
            provider=provider,
            session_id="sess_model_stage_aclose_lookup_failure",
            publisher=publisher,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        publisher_slot["publish"] = _atomic_publisher(
            store,
            expected_run_epoch=session.run_epoch,
            observed=observed,
        )

        async def consume() -> None:
            async for _ in model_run._execute_request(
                model_request=model_request,
                step=1,
                messages=[user_message],
                model_step_identity=new_model_step_identity(),
            ):
                pass

        task = asyncio.create_task(consume())
        await provider.events.waiting_after_completion.wait()
        task.cancel("authoritative lookup cancellation")
        with pytest.raises(asyncio.CancelledError, match="authoritative lookup cancellation"):
            await task
        return store, provider, session, user_message, observed

    store, provider, session, user_message, observed = asyncio.run(run())

    assert provider.calls == 1
    assert provider.events.close_lookups == 1
    assert len(observed) == 1
    publication = observed[0]
    assert publication.authoritative_assistant_message is None
    assert publication.completion_event.payload["step_classification"] == {
        "type": "failed",
        "reason": "model stream was cancelled before terminal validation completed",
    }
    usage = publication.completion_event.payload["usage"]
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == 10
    assert asyncio.run(store.load_transcript(session.id)) == [user_message]
    assert asyncio.run(store.load_active_model_completion_stage(session.id)) is None
    durable_completions = [
        event
        for event in asyncio.run(store.load_events(session.id))
        if event.type == EventType.MODEL_COMPLETED
    ]
    assert durable_completions == [publication.completion_event]


def test_model_executor_never_retries_after_authoritative_completed_frame() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _CompletedThenRetryableFailureProvider()
        observed: list[ModelCompletionPublicationRequest] = []
        publisher_slot = {}

        async def publisher(request: ModelCompletionPublicationRequest):
            return await publisher_slot["publish"](request)

        model_run, model_request, session, user_message = await _create_model_run(
            store=store,
            provider=provider,
            session_id="sess_model_stage_terminal_transport_failure",
            publisher=publisher,
            retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
        )
        publisher_slot["publish"] = _atomic_publisher(
            store,
            expected_run_epoch=session.run_epoch,
            observed=observed,
        )

        with pytest.raises(
            ModelProviderError,
            match="retryable transport failure after completion",
        ):
            async for _ in model_run._execute_request(
                model_request=model_request,
                step=1,
                messages=[user_message],
                model_step_identity=new_model_step_identity(),
            ):
                pass
        return store, provider, session, user_message, observed

    store, provider, session, user_message, observed = asyncio.run(run())

    assert provider.calls == 1
    assert len(observed) == 1
    assert observed[0].authoritative_assistant_message is None
    assert observed[0].completion_event.payload["step_classification"]["type"] == "failed"
    assert asyncio.run(store.load_transcript(session.id)) == [user_message]


def test_model_executor_fails_before_dispatch_when_transcript_cursor_drifted() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _NeverDispatchedProvider()
        publication_calls = 0

        async def publisher(
            request: ModelCompletionPublicationRequest,
        ) -> ModelCompletionPublicationResult:
            nonlocal publication_calls
            publication_calls += 1
            raise AssertionError("Cursor drift cannot produce completion material.")

        model_run, model_request, session, user_message = await _create_model_run(
            store=store,
            provider=provider,
            session_id="sess_model_stage_cursor_drift",
            publisher=publisher,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        drifted_messages = [
            user_message,
            Message.text("user", "not durably appended"),
        ]
        with pytest.raises(ValueError, match="transcript cursor is stale"):
            async for _ in model_run._execute_request(
                model_request=model_request,
                step=1,
                messages=drifted_messages,
                model_step_identity=new_model_step_identity(),
            ):
                pass
        return store, provider, session, publication_calls

    store, provider, session, publication_calls = asyncio.run(run())

    assert provider.calls == 0
    assert publication_calls == 0
    assert asyncio.run(store.load_active_model_completion_stage(session.id)) is None
