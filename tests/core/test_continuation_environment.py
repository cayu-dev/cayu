"""Publication boundaries retain the environment needed by continuation cleanup."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from types import SimpleNamespace

import pytest

from cayu.environments.factory import EnvironmentFactoryOperation
from cayu.events import Event, EventType
from cayu.runtime._continuation_environment import ContinuationEnvironment
from cayu.runtime._environment_lifecycle import (
    EnvironmentBindingResult,
    EnvironmentFactoryResolutionResult,
)


def _preparation(phase, *, error=None, rebind_error=None, barrier=None):
    old = SimpleNamespace(spec=SimpleNamespace(name="environment"))
    replacement = SimpleNamespace(spec=old.spec)
    profile = object()
    rebound = object()
    operations = []

    def rebind(environment, *, validated_profile):
        assert environment is replacement and validated_profile is profile
        if rebind_error is not None:
            raise rebind_error
        return rebound

    context = SimpleNamespace(with_registered_environment=rebind)
    start_type, end_type = (
        (EventType.ENVIRONMENT_FACTORY_STARTED, EventType.ENVIRONMENT_FACTORY_COMPLETED)
        if phase == "reconnect"
        else (EventType.ENVIRONMENT_BINDING_STARTED, EventType.ENVIRONMENT_BINDING_COMPLETED)
    )
    started = Event(type=start_type, session_id="session")
    completed = Event(type=end_type, session_id="session")

    async def start(**kwargs):
        assert kwargs["registered_environment"] is old
        assert kwargs["invocation_context"] is context
        return started

    async def perform(**kwargs):
        assert kwargs["started_event"] is started
        assert kwargs["registered_environment"] is old
        assert kwargs["invocation_context"] is context
        if phase == "reconnect":
            assert kwargs["operation"] is EnvironmentFactoryOperation.RECONNECT
        operations.append(phase)
        if barrier is not None:
            barrier[0].set()
            await barrier[1].wait()
        result_type = (
            EnvironmentFactoryResolutionResult if phase == "reconnect" else EnvironmentBindingResult
        )
        return result_type(replacement, [completed], error)

    lifecycle = SimpleNamespace(
        emit_factory_started=start,
        resolve_factory=perform,
        emit_binding_started=start,
        bind=perform,
    )
    preparation = ContinuationEnvironment(
        lifecycle=lifecycle,
        session=SimpleNamespace(id="session"),
        agent=object(),
        profile=profile,
        registered_environment=old,
        invocation_context=context,
    )
    return preparation, old, replacement, context, rebound, operations, started, completed


@pytest.mark.parametrize("phase", ["reconnect", "bind"])
@pytest.mark.parametrize("failure", [False, True])
def test_environment_authority_is_retained_before_result_publication(phase, failure):
    async def scenario():
        error = RuntimeError("environment operation failed") if failure else None
        preparation, old, replacement, context, rebound, calls, start, end = _preparation(
            phase, error=error
        )
        async with aclosing(getattr(preparation, phase)()) as stream:
            assert await anext(stream) is start
            assert preparation.registered_environment is old
            assert preparation.invocation_context is context
            assert not calls
            assert await anext(stream) is end
            assert preparation.registered_environment is replacement
            assert preparation.invocation_context is rebound
            assert preparation.error is error
        # A consumer can close on the result event. Its failure/abandonment
        # handler must still receive the replacement, without another effect.
        assert preparation.registered_environment is replacement
        assert calls == [phase]

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["reconnect", "bind"])
def test_rebinding_failure_retains_returned_environment_for_cleanup(phase):
    async def scenario():
        error = RuntimeError("changed invocation authority")
        preparation, _, replacement, context, _, calls, start, _ = _preparation(
            phase, rebind_error=error
        )
        async with aclosing(getattr(preparation, phase)()) as stream:
            assert await anext(stream) is start
            with pytest.raises(RuntimeError) as caught:
                await anext(stream)
        assert caught.value is error
        assert preparation.registered_environment is replacement
        assert preparation.invocation_context is context
        assert calls == [phase]

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["reconnect", "bind"])
def test_cancelled_environment_operation_keeps_prior_invocation_authority(phase):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        preparation, old, _, context, _, calls, start, _ = _preparation(
            phase, barrier=(entered, release)
        )
        async with aclosing(getattr(preparation, phase)()) as stream:
            assert await anext(stream) is start
            task = asyncio.create_task(anext(stream))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                task.cancel("cancelled during preparation")
                with pytest.raises(asyncio.CancelledError, match="cancelled during preparation"):
                    await task
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        assert preparation.registered_environment is old
        assert preparation.invocation_context is context
        assert calls == [phase]

    asyncio.run(scenario())
