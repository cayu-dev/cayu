"""Shutdown waits for completion verification retained past its caller."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import traceback

import pytest
from tests.core.test_completion_result_resolvers import _contract as _resolver_contract
from tests.core.test_completion_result_resolvers import (
    _prepared_app,
    _prepared_app_with_stores,
    _resolution_request,
    _result,
)
from tests.core.test_completion_verifier_adapters import (
    _accepted_decision,
    _contract,
    _execution_request,
    _proposal,
    _TestCompletionVerifier,
)

from cayu import CompletionResultUnavailable, Event, InMemorySessionStore
from cayu.applications import CayuApp
from cayu.runtime.completion_result_resolvers import CompletionResultResolver
from cayu.runtime.completion_verifiers import CompletionVerifierExecutionError
from cayu.sessions.records import Session, StoreTimeCheckpointTransform
from cayu.tasks.memory import InMemoryTaskStore
from cayu.vaults import SecretRedactor


class _OwnedTaskStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _ResistantVerifier(_TestCompletionVerifier):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def verify(self, request):
        del request
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()
        self.finished.set()
        return _accepted_decision()


@pytest.mark.parametrize("left_by", ["timeout", "caller_cancellation"])
def test_shutdown_waits_for_a_verifier_retained_past_its_caller(left_by: str) -> None:
    async def scenario() -> None:
        store = _OwnedTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = _ResistantVerifier()
        app = CayuApp(task_store=store, enable_logging=False, owned_resources=(store,))
        app.register_completion_verifier(contract.verifier, verifier)

        if left_by == "timeout":
            request = _execution_request(proposal_id, lease_seconds=2, timeout_seconds=0.01)
            with pytest.raises(CompletionVerifierExecutionError, match="bounded execution timeout"):
                await app.verify_completion_proposal(request)
        else:
            request = _execution_request(proposal_id, lease_seconds=30, timeout_seconds=20)
            caller = asyncio.create_task(app.verify_completion_proposal(request))
            await asyncio.wait_for(verifier.started.wait(), 5)
            caller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await caller
        await asyncio.wait_for(verifier.cancelled.wait(), 5)

        # The public call returned, but its verifier, heartbeat and settlement
        # are still running: shutdown must not report settled or close the store.
        outcome = await app.aclose(timeout_s=0.3)
        assert outcome.status == "incomplete"
        step = outcome.step("verified_completions")
        assert step is not None and step.status == "incomplete"
        assert outcome.owned_resources == "retained" and not store.closed

        verifier.release.set()
        await asyncio.wait_for(verifier.finished.wait(), 5)
        settled = await app.aclose(timeout_s=5)
        assert settled.settled and store.closed

    asyncio.run(scenario())


def test_shutdown_waits_for_result_resolution_retained_past_its_caller() -> None:
    class ThreadedResolver(CompletionResultResolver):
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def blocking_read(self) -> dict[str, object]:
            self.started.set()
            if not self.release.wait(timeout=10):
                raise TimeoutError("resolver test thread was not released")
            return _result("1")

        async def resolve(self, request):
            del request
            return await asyncio.to_thread(self.blocking_read)

    async def scenario() -> None:
        app, _sessions, _tasks, decision_id = await _prepared_app()
        resolver = ThreadedResolver()
        app.register_completion_result_resolver(_resolver_contract().result_resolver, resolver)
        caller = asyncio.create_task(
            app.resolve_completion_result(_resolution_request(decision_id))
        )
        try:
            assert await asyncio.to_thread(resolver.started.wait, 5)
            caller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await caller
            outcome = await app.aclose(timeout_s=0.3)
            assert outcome.status == "incomplete"
            step = outcome.step("verified_completions")
            assert step is not None and step.status == "incomplete"
        finally:
            resolver.release.set()
        settled = await app.aclose(timeout_s=10)
        assert settled.settled

    asyncio.run(scenario())


_RELEASE_SECRET = "retained-release-failure-secret-canary"


class _FailingReleaseStore(InMemorySessionStore):
    """Fail the second, deferred publication-reservation release."""

    invocation_lifecycle_command_version = 1
    supports_completion_result_event_publication_reservations = True

    def __init__(self) -> None:
        super().__init__()
        self.empty_publications = 0
        self.release_started = asyncio.Event()
        self.release_allowed = asyncio.Event()
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def _publish_completion_result_event_publication(
        self,
        session_id: str,
        *,
        checkpoint_transform: StoreTimeCheckpointTransform,
        events: list[Event],
    ) -> Session:
        if not events:
            self.empty_publications += 1
            if self.empty_publications == 2:
                self.release_started.set()
                await self.release_allowed.wait()
                raise ConnectionError(_RELEASE_SECRET)
        return await super()._publish_completion_result_event_publication(
            session_id, checkpoint_transform=checkpoint_transform, events=events
        )


class _UnavailableResolver(CompletionResultResolver):
    async def resolve(self, request):
        del request
        raise CompletionResultUnavailable(_RELEASE_SECRET)


def test_shutdown_reports_a_retained_resolution_cleanup_that_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        sessions = _FailingReleaseStore()
        prepared, _sessions, tasks, decision_id = await _prepared_app_with_stores(
            session_store=sessions, secret_redactor=SecretRedactor(_RELEASE_SECRET)
        )
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
            secret_redactor=SecretRedactor(_RELEASE_SECRET),
            owned_resources=(sessions,),
        )
        del prepared
        app.register_completion_result_resolver(
            _resolver_contract().result_resolver, _UnavailableResolver()
        )
        caller = asyncio.create_task(
            app.resolve_completion_result(_resolution_request(decision_id))
        )
        await asyncio.wait_for(sessions.release_started.wait(), 5)
        # The caller leaves; its deferred reservation release then fails.
        caller.cancel()
        sessions.release_allowed.set()
        with contextlib.suppress(asyncio.CancelledError):
            await caller

        failed = await app.aclose(timeout_s=5)
        step = failed.step("verified_completions")
        assert step is not None and step.status == "failed"
        assert step.failure_type == "ConnectionError"
        assert failed.status == "failed" and failed.owned_resources == "retained"
        assert not sessions.closed
        # The evidence stays: the reservation is still durable.
        checkpoint = await sessions.load_checkpoint("session:application")
        assert checkpoint is not None and "completion_result_event_publications" in checkpoint

        # A retry neither reports the same failure again nor erases it.
        retried = await app.aclose(timeout_s=5)
        assert retried.settled and sessions.closed
        checkpoint = await sessions.load_checkpoint("session:application")
        assert checkpoint is not None and "completion_result_event_publications" in checkpoint

    with caplog.at_level(logging.DEBUG):
        asyncio.run(scenario())
    assert _RELEASE_SECRET not in caplog.text


def test_shutdown_reports_a_retained_verifier_that_failed_after_its_caller_left() -> None:
    class LateFailingVerifier(_ResistantVerifier):
        async def verify(self, request):
            await super().verify(request)
            raise RuntimeError("verifier failed after its caller left")

    async def scenario() -> None:
        store = _OwnedTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = LateFailingVerifier()
        app = CayuApp(task_store=store, enable_logging=False, owned_resources=(store,))
        app.register_completion_verifier(contract.verifier, verifier)
        request = _execution_request(proposal_id, lease_seconds=2, timeout_seconds=0.01)
        with pytest.raises(CompletionVerifierExecutionError, match="bounded execution timeout"):
            await app.verify_completion_proposal(request)
        await asyncio.wait_for(verifier.cancelled.wait(), 5)
        verifier.release.set()

        failed = await app.aclose(timeout_s=5)
        step = failed.step("verified_completions")
        assert step is not None and step.status == "failed"
        assert failed.owned_resources == "retained" and not store.closed

        retried = await app.aclose(timeout_s=5)
        assert retried.settled and store.closed

    asyncio.run(scenario())


def test_a_drain_reports_a_failure_without_taking_it_from_the_exact_retry() -> None:
    async def scenario() -> None:
        sessions = _FailingReleaseStore()
        app, _sessions, _tasks, decision_id = await _prepared_app_with_stores(
            session_store=sessions, secret_redactor=SecretRedactor(_RELEASE_SECRET)
        )
        app.register_completion_result_resolver(
            _resolver_contract().result_resolver, _UnavailableResolver()
        )
        caller = asyncio.create_task(
            app.resolve_completion_result(_resolution_request(decision_id))
        )
        await asyncio.wait_for(sessions.release_started.wait(), 5)
        caller.cancel()
        sessions.release_allowed.set()
        with contextlib.suppress(asyncio.CancelledError):
            await caller

        with pytest.raises(ConnectionError) as reported:
            await app.drain_verified_completions(timeout_s=5)
        rendered = "".join(traceback.format_exception(reported.value))
        assert _RELEASE_SECRET not in rendered
        assert await app.drain_verified_completions(timeout_s=5) is True
        # The exact retry still receives and acknowledges the retained failure.
        with pytest.raises(ConnectionError):
            await app.resolve_completion_result(_resolution_request(decision_id))

    asyncio.run(scenario())


def test_a_drain_leaves_a_verifier_failure_for_the_exact_retry() -> None:
    class LateFailingVerifier(_ResistantVerifier):
        async def verify(self, request):
            await super().verify(request)
            raise RuntimeError("verifier failed after its caller left")

    async def scenario() -> None:
        store = _OwnedTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = LateFailingVerifier()
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        request = _execution_request(proposal_id, lease_seconds=2, timeout_seconds=0.01)
        with pytest.raises(CompletionVerifierExecutionError, match="bounded execution timeout"):
            await app.verify_completion_proposal(request)
        await asyncio.wait_for(verifier.cancelled.wait(), 5)
        verifier.release.set()

        with pytest.raises(CompletionVerifierExecutionError, match="after its caller left"):
            await app.drain_verified_completions(timeout_s=5)
        assert await app.drain_verified_completions(timeout_s=5) is True
        with pytest.raises(CompletionVerifierExecutionError, match="after its caller left"):
            await app.verify_completion_proposal(request)

    asyncio.run(scenario())
