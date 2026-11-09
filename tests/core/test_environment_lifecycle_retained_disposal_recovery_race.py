"""Recovery must not replace authority captured by an in-flight cleanup."""

import asyncio

import pytest
from tests.environments.test_docker_coding import (
    _CONTAINER_ID,
    _coding_product_test_binding,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events

import cayu.runtime._environment_lifecycle as lifecycle_module
from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import Environment, EnvironmentSpec
from cayu.environments.factory import EnvironmentFactory, EnvironmentFactoryResult
from cayu.environments.lifecycle import (
    EnvironmentLifecyclePhase,
    EnvironmentLifecycleProgressStatus,
)
from cayu.providers import ModelStreamEvent
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest


@pytest.mark.parametrize("failure", ["disposal", "marker"])
def test_recovery_waits_for_inflight_retained_cleanup(tmp_path, monkeypatch, failure):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "result.txt").write_text("saved work")
    runner, source, binding = _coding_product_test_binding(source_root, target_root)
    close_available = False
    original_close = runner.close

    async def close():
        if not close_available:
            raise ConnectionError("Docker daemon temporarily unavailable")
        await original_close()

    monkeypatch.setattr(runner, "close", close)

    class Factory(EnvironmentFactory):
        async def create(self, request):
            return EnvironmentFactoryResult(
                Environment(
                    EnvironmentSpec(name="coding"),
                    workspace=source,
                    runner=runner,
                    binding=binding,
                ),
                reconnect_metadata={"container_id": _CONTAINER_ID},
            )

        async def recover_finalization_disposal(self, request, state):
            assert request.reconnect_metadata == {"container_id": _CONTAINER_ID}
            assert state == {
                "version": 1,
                "kind": "docker_coding_disposal",
                "container_id": _CONTAINER_ID,
                "attachment_ids": [],
            }
            await close()

    async def scenario():
        nonlocal close_available
        app = CayuApp(enable_logging=False)
        lifecycle = app._environment_lifecycle
        clear_available = False
        original_clear = lifecycle.clear_completion_finalization

        async def clear(**kwargs):
            if failure == "marker" and not clear_available:
                raise ConnectionError("Checkpoint temporarily unavailable")
            await original_clear(**kwargs)

        monkeypatch.setattr(lifecycle, "clear_completion_finalization", clear)
        app.register_provider(
            FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]), default=True
        )
        app.register_environment_factory(EnvironmentSpec(name="coding"), Factory(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        try:
            events = await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="disposal-retry",
                    messages=[Message.text("user", "save")],
                ),
            )
            assert events[-1].type is EventType.SESSION_FAILED
            assert not runner.is_closed
            checkpoint = await app.session_store.load_checkpoint("disposal-retry")
            assert checkpoint["pending_completion_finalization"]["disposal_state"] == {
                "version": 1,
                "kind": "docker_coding_disposal",
                "container_id": _CONTAINER_ID,
                "attachment_ids": [],
            }
            owner = app._environment_lifecycle._active_environment_setups["disposal-retry"]
            close_available = True
            if failure == "marker":
                await lifecycle._settle_retained_environment_cleanups()
                assert runner.is_closed
            assert owner.cleanup_requires_finalize_retry == (failure == "disposal")
            assert (owner.pending_completion_marker_clear is not None) == (failure == "marker")
            clear_available = True
            captured_context = owner.invocation_context
            cleanup_started = asyncio.Event()
            release_cleanup = asyncio.Event()
            recovery_entered = asyncio.Event()
            original_report = lifecycle_module._report_environment_lifecycle
            original_recover = lifecycle.recover_completion_disposal

            async def report(reporter, phase, status, **kwargs):
                if (
                    phase is EnvironmentLifecyclePhase.RETAINED_CLEANUP
                    and status is EnvironmentLifecycleProgressStatus.STARTED
                    and not cleanup_started.is_set()
                ):
                    cleanup_started.set()
                    await release_cleanup.wait()
                    assert owner.invocation_context is captured_context
                await original_report(reporter, phase, status, **kwargs)

            async def recover_disposal(**kwargs):
                recovery_entered.set()
                return await original_recover(**kwargs)

            monkeypatch.setattr(lifecycle_module, "_report_environment_lifecycle", report)
            monkeypatch.setattr(lifecycle, "recover_completion_disposal", recover_disposal)
            sweep = asyncio.create_task(lifecycle._settle_retained_environment_cleanups())
            await asyncio.wait_for(cleanup_started.wait(), timeout=3)
            settlement = owner.cleanup_settlement_task
            recovery = asyncio.create_task(
                app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id="disposal-retry")
                )
            )
            try:
                await asyncio.wait_for(recovery_entered.wait(), timeout=3)
                # Recovery has reached disposal while the old context is captured
                # by abort_environment_setup, before it retains that context.
                await asyncio.sleep(0)
                assert not settlement.done()
                assert owner.invocation_context is captured_context
            finally:
                release_cleanup.set()
                await sweep
            await asyncio.wait_for(recovery, timeout=3)
            if failure == "marker":
                # Marker-only cleanup can finish with its original context; do
                # not replace the owner or redispatch its disposed binding.
                assert owner.invocation_context is captured_context
            assert await app.drain_environment_cleanups(timeout_s=3), owner.cleanup_error
            assert runner.is_closed
            checkpoint = await app.session_store.load_checkpoint("disposal-retry")
            assert "pending_completion_finalization" not in checkpoint
            assert (source_root / "result.txt").read_text() == "saved work"
        finally:
            # The local runner substitutes for Docker; never leak a subprocess
            # if the assertion correctly exposes the failed recovery.
            await original_close()

    asyncio.run(scenario())
