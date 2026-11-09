"""Recovery of retained Docker disposal must not reuse its obsolete run epoch."""

import asyncio

import pytest
from tests.environments.test_docker_coding import (
    _CONTAINER_ID,
    _coding_product_test_binding,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import Environment, EnvironmentSpec
from cayu.environments.factory import EnvironmentFactory, EnvironmentFactoryResult
from cayu.providers import ModelStreamEvent
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest


@pytest.mark.parametrize("recover", [False, True], ids=["drain-before-refencing", "recovery"])
def test_retained_docker_disposal_remains_recoverable(tmp_path, monkeypatch, recover):
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
            assert owner.cleanup_requires_finalize_retry
            close_available = True
            if recover:
                recovery_error = None
                try:
                    await app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(session_id="disposal-retry")
                    )
                except Exception as error:
                    recovery_error = error
                current = await app.session_store.load("disposal-retry")
                assert recovery_error is None, (
                    str(recovery_error),
                    str(owner.cleanup_error),
                    {
                        "owner_epoch": owner.invocation_context.binding.run_epoch,
                        "recovery_epoch": current.run_epoch,
                    },
                )
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
