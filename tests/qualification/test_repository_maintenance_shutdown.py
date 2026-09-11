"""Owned shutdown observation, not a live PostgreSQL or process-supervisor proof."""

import asyncio
import warnings

import pytest

from cayu.providers._http import SharedAsyncClient
from tests.cli.test_scaffold_coding_budget import denial_policy
from tests.qualification.test_repository_maintenance_application import project as project
from tests.qualification.test_repository_maintenance_deployment import deployment as deployment

_DRAINS = (
    "drain_background_interruptions",
    "drain_recovery_cleanups",
    "drain_provider_operation_cancellations",
    "drain_environment_cleanups",
    "drain_knowledge_publications",
)
_CLOSES = ("provider", "session", "task", "knowledge", "budget")


@pytest.fixture
def shutdown(deployment, monkeypatch):
    module, source, provider = deployment
    owned = module.build_maintenance_deployment(
        budget_policy=denial_policy(), workspace_root=source
    )
    app = owned.application.app
    calls = []

    def operation(name, result=None):
        async def call(**kwargs):
            calls.append(name)
            return result

        return call

    registry = (
        app.get_agent(owned.application.agent_name).tools["subagent"].tool.background_task_registry
    )
    monkeypatch.setattr(registry, "drain", operation("registry", True))
    for name in _DRAINS:
        monkeypatch.setattr(app, name, operation(name, True))
    monkeypatch.setattr(provider, "aclose", operation("provider"), raising=False)
    stores = dict(
        zip(
            _CLOSES[1:],
            (app.session_store, app.task_store, app.knowledge_store, app.budget_ledger),
            strict=True,
        )
    )
    for name, store in stores.items():
        monkeypatch.setattr(store, "close", operation(name))
    return owned, app, provider, registry, stores, calls


def test_shutdown_closes_in_order_and_replays_without_reclosing(shutdown):
    owned, _app, _provider, _registry, _stores, calls = shutdown

    async def scenario():
        assert await owned.aclose(timeout_s=2) is True
        assert calls == ["registry", *_DRAINS, *_CLOSES]
        assert await owned.aclose(timeout_s=2) is True
        assert calls == ["registry", *_DRAINS, *_CLOSES]

    asyncio.run(scenario())


def test_quiescence_preserves_clients_until_final_shutdown(shutdown):
    owned, _app, _provider, _registry, _stores, calls = shutdown

    async def scenario():
        assert await owned.quiesce(timeout_s=2) is True
        assert calls == ["registry", *_DRAINS]
        assert not set(calls).intersection(_CLOSES)
        assert await owned.aclose(timeout_s=2) is True
        assert calls == ["registry", *_DRAINS, "registry", *_DRAINS, *_CLOSES]

    asyncio.run(scenario())


def test_concurrent_shutdown_observers_wait_for_registered_work(shutdown, monkeypatch):
    owned, _app, _provider, registry, _stores, calls = shutdown
    # Restore the concrete registry's drain. This is stream-owner integration,
    # not a generated PostgreSQL session or a remote-effect quiescence proof.
    monkeypatch.setattr(registry, "drain", type(registry).drain.__get__(registry))

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def stream():
            entered.set()
            await release.wait()

        child = asyncio.create_task(stream())
        registry.register(child, parent_session_id="parent", child_session_id="child")
        observers = []
        try:
            await asyncio.wait_for(entered.wait(), 5)
            observers = [asyncio.create_task(owned.aclose(timeout_s=5)) for _ in range(2)]
            done, _pending = await asyncio.wait(observers, timeout=0.01)
            assert not done and not calls
            assert not child.done() and child.cancelling() == 0
            observers[0].cancel("one-observer-left")
            with pytest.raises(asyncio.CancelledError, match="one-observer-left"):
                await observers[0]
            assert observers[0].cancelled() and observers[0].cancelling() == 1
            assert not observers[1].done() and not calls
            assert not child.done() and child.cancelling() == 0
            release.set()
            assert await observers[1] is True
            assert child.done() and not child.cancelled()
            assert calls == [*_DRAINS, *_CLOSES]
            assert await owned.aclose(timeout_s=1) is True
            assert calls == [*_DRAINS, *_CLOSES]
        finally:
            release.set()
            await asyncio.gather(child, *observers, return_exceptions=True)
            await owned.aclose(timeout_s=5)

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["registry", *_DRAINS])
@pytest.mark.parametrize("result", [False, 1])
def test_no_dependency_closes_without_every_positive_drain(shutdown, monkeypatch, phase, result):
    owned, app, _provider, registry, _stores, calls = shutdown
    target, method = (registry, "drain") if phase == "registry" else (app, phase)
    original = getattr(target, method)

    async def not_settled(**kwargs):
        calls.append(phase)
        return result

    monkeypatch.setattr(target, method, not_settled)

    async def scenario():
        assert await owned.aclose(timeout_s=1) is False
        assert not set(calls).intersection(_CLOSES)
        monkeypatch.setattr(target, method, original)
        assert await owned.aclose(timeout_s=1) is True
        assert all(calls.count(name) == 1 for name in _CLOSES)

    asyncio.run(scenario())


def test_cancel_and_timeout_observers_retain_actual_detached_client_close(shutdown, monkeypatch):
    owned, _app, provider, _registry, _stores, calls = shutdown

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        shared = SharedAsyncClient()
        client = shared.get()
        original_close = client.aclose
        close_calls = 0

        async def controlled_close():
            nonlocal close_calls
            close_calls += 1
            calls.append("provider")
            entered.set()
            await release.wait()
            await original_close()

        monkeypatch.setattr(client, "aclose", controlled_close)
        monkeypatch.setattr(provider, "aclose", shared.aclose)
        first = asyncio.create_task(owned.aclose(timeout_s=2))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 5)
            # Native helper has detached its client, but the deployment job
            # still owns this exact in-flight close coroutine.
            assert shared._client is None and not client.is_closed
            first.cancel("shutdown-observer-stop")
            with pytest.raises(asyncio.CancelledError, match="shutdown-observer-stop"):
                await first
            assert first.cancelled() and first.cancelling() == 1
            assert await owned.aclose(timeout_s=0.01) is False
            assert not set(calls).intersection(_CLOSES[1:])
            second = asyncio.create_task(owned.aclose(timeout_s=2))
            await asyncio.sleep(0)
            assert close_calls == 1
            release.set()
            assert await second is True
            assert client.is_closed and close_calls == 1
            assert calls == ["registry", *_DRAINS, *_CLOSES]
        finally:
            release.set()
            await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
            await owned.aclose(timeout_s=5)
            await original_close()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["provider", "task", "drain_environment_cleanups"])
def test_shutdown_failure_is_retained_not_blindly_retried(
    shutdown, monkeypatch, caplog, capsys, phase
):
    owned, app, provider, _registry, stores, calls = shutdown
    failure = RuntimeError("private-shutdown-canary")
    target = provider if phase == "provider" else app if phase in _DRAINS else stores[phase]
    method = "aclose" if phase == "provider" else phase if phase in _DRAINS else "close"

    async def fail(**kwargs):
        calls.append(phase)
        raise failure

    monkeypatch.setattr(target, method, fail)

    async def scenario():
        for _ in range(2):
            with pytest.raises(RuntimeError) as caught:
                await owned.aclose(timeout_s=2)
            assert caught.value is failure
        assert calls.count(phase) == 1
        if phase == "task":
            assert calls[-3:] == ["provider", "session", "task"]
            assert "knowledge" not in calls and "budget" not in calls
        elif phase in _DRAINS:
            assert not set(calls).intersection(_CLOSES)

    with warnings.catch_warnings(record=True) as recorded:
        asyncio.run(scenario())
    output = capsys.readouterr()
    assert "private-shutdown-canary" not in caplog.text + output.out + output.err
    assert not recorded


def test_shutdown_requires_explicit_provider_close(shutdown, monkeypatch):
    owned, _app, provider, _registry, _stores, calls = shutdown
    monkeypatch.delattr(provider, "aclose")

    async def scenario():
        with pytest.raises(ValueError, match="close capability"):
            await owned.aclose(timeout_s=1)
        assert not calls

    asyncio.run(scenario())


@pytest.mark.parametrize("timeout", [False, 0, -1, float("nan"), float("inf")])
def test_invalid_observation_timeout_does_not_begin_shutdown(shutdown, timeout):
    owned, _app, _provider, _registry, _stores, calls = shutdown

    async def scenario():
        with pytest.raises(ValueError, match="finite positive"):
            await owned.aclose(timeout_s=timeout)
        assert not calls
        assert await owned.aclose(timeout_s=1) is True

    asyncio.run(scenario())
