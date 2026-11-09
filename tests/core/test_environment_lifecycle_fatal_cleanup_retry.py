"""Control signals from a binding cleanup retry must not become ordinary failures."""

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu._exception_groups import iter_exception_tree
from cayu.environments import Environment, EnvironmentSpec
from cayu.environments.bindings import WorkspaceBinding
from cayu.events import EventType
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runtime._binding_cleanup import binding_cleanup_status, record_binding_cleanup_failure
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus


class _Provider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, request):  # type: ignore[no-untyped-def]
        self.requests += 1
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _run_with_retry_error(retry_error: BaseException, *, session_suffix: str):
    bind_error = RuntimeError("bind failed")
    retry_calls = 0

    class FailingRetryBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            async def retry() -> None:
                nonlocal retry_calls
                retry_calls += 1
                raise retry_error

            record_binding_cleanup_failure(
                bind_error,
                RuntimeError("binding rollback failed"),
                retry=retry,
            )
            raise bind_error

        async def finalize(self, bound, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("a failed bind must not finalize")

    session_id = f"sess_retry_{session_suffix}"

    async def scenario():
        store = InMemorySessionStore()
        provider = _Provider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="retry"), binding=FailingRetryBinding()),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def collect() -> None:
            async for _event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            ):
                pass

        raised: BaseException | None = None
        try:
            await asyncio.wait_for(collect(), timeout=10)
        except BaseException as error:
            raised = error
        session = await store.load(session_id)
        assert session is not None
        return raised, await store.load_events(session_id), session, provider.requests

    raised, events, session, provider_requests = asyncio.run(scenario())
    assert retry_calls == 1
    assert provider_requests == 0
    return raised, events, session, bind_error


def _binding_failed_cleanup(events) -> dict:
    binding_failed = [
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FAILED
    ]
    assert len(binding_failed) == 1
    assert binding_failed[0].payload["error"] == "bind failed"
    cleanup = binding_failed[0].payload["binding_cleanup"]
    assert cleanup["retry_attempted"] is True
    assert cleanup["retry_completed"] is False
    return cleanup


def _assert_propagated(raised, bind_error, retry_error, signal) -> None:
    assert isinstance(raised, BaseExceptionGroup)
    assert len(raised.exceptions) == 2
    assert raised.exceptions[0] is bind_error
    # The operation boundary may rebuild a retry group, but the caller still
    # receives the exact signal object alongside the bind error.
    propagated_retry_error = raised.exceptions[1]
    assert type(propagated_retry_error) is type(retry_error)
    assert any(leaf is signal for leaf in iter_exception_tree(propagated_retry_error))
    cleanup_status = binding_cleanup_status(raised)
    assert cleanup_status is not None
    assert cleanup_status.retry_error is propagated_retry_error


@pytest.mark.parametrize("signal_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_fatal_cleanup_retry_signal_propagates_after_bind_failure_is_recorded(signal_type):
    signal = signal_type("cleanup retry signal")

    raised, events, session, bind_error = _run_with_retry_error(
        signal, session_suffix=signal_type.__name__
    )

    _assert_propagated(raised, bind_error, signal, signal)
    assert raised.__cause__ is signal
    assert _binding_failed_cleanup(events)["retry_error_type"] == signal_type.__name__
    assert not any(event.type is EventType.SESSION_FAILED for event in events)
    assert session.status is not SessionStatus.FAILED


@pytest.mark.parametrize("signal_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_grouped_fatal_cleanup_retry_signal_propagates(signal_type):
    signal = signal_type("cleanup retry signal")
    retry_error = BaseExceptionGroup(
        "cleanup retry failed and was interrupted",
        [RuntimeError("cleanup retry failed"), signal],
    )

    raised, events, session, bind_error = _run_with_retry_error(
        retry_error, session_suffix=f"grouped_{signal_type.__name__}"
    )

    _assert_propagated(raised, bind_error, retry_error, signal)
    assert _binding_failed_cleanup(events)["retry_error_type"] == "BaseExceptionGroup"
    assert not any(event.type is EventType.SESSION_FAILED for event in events)
    assert session.status is not SessionStatus.FAILED


def test_grouped_cancelled_cleanup_retry_still_propagates():
    signal = asyncio.CancelledError("cleanup retry cancelled")
    retry_error = BaseExceptionGroup(
        "cleanup retry failed and was cancelled",
        [RuntimeError("cleanup retry failed"), signal],
    )

    raised, events, session, bind_error = _run_with_retry_error(
        retry_error, session_suffix="grouped_cancelled"
    )

    _assert_propagated(raised, bind_error, retry_error, signal)
    assert raised.__cause__ is raised.exceptions[1]
    assert _binding_failed_cleanup(events)["retry_error_type"] == "BaseExceptionGroup"
    assert session.status is SessionStatus.INTERRUPTED


def test_child_only_cancelled_cleanup_retry_is_still_an_ordinary_failure():
    # Without caller cancellation, the operation boundary classifies a bare
    # CancelledError from the retry as an ordinary cleanup failure.
    raised, events, session, _bind_error = _run_with_retry_error(
        asyncio.CancelledError("cleanup retry cancelled"),
        session_suffix="child_cancelled",
    )

    assert raised is None
    cleanup = _binding_failed_cleanup(events)
    assert cleanup["retry_error_type"] == "RuntimeError"
    assert "cancelled without caller cancellation" in cleanup["retry_error"]
    assert session.status is SessionStatus.FAILED
