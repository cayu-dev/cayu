"""Cancellation after a binding returns success must not replay its settled effects."""

import asyncio

import pytest
from tests.environments.test_docker_coding import _CONTAINER_ID, _coding_product_test_binding
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import Environment, EnvironmentSpec
from cayu.environments.factory import EnvironmentFactory, EnvironmentFactoryResult
from cayu.providers import ModelStreamEvent
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest


@pytest.mark.parametrize(
    ("cancelled", "recover"),
    [(False, False), (True, False), (True, True)],
    ids=["ordinary-return", "cancelled-return-drain", "cancelled-return-recovery"],
)
def test_settled_finalization_return_does_not_rearm_disposed_binding(
    tmp_path, monkeypatch, cancelled, recover
):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "result.txt").write_text("saved work")
    runner, source, binding = _coding_product_test_binding(source_root, target_root)

    async def scenario():
        finalized = asyncio.Event()
        allow_return = asyncio.Event()
        original_finalize = binding.finalize
        finalize_calls = []

        async def settling_finalize(bound, **kwargs):
            finalize_calls.append(kwargs.get("outcome"))
            snapshot = await original_finalize(bound, **kwargs)
            finalized.set()
            try:  # noqa: SIM105
                await allow_return.wait()
            except asyncio.CancelledError:
                # An adapter may finish opaque teardown despite cancellation,
                # then return its successfully completed finalization result.
                pass
            return snapshot

        monkeypatch.setattr(binding, "finalize", settling_finalize)

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
                await runner.close()

        app = CayuApp(enable_logging=False)
        app.register_provider(
            FakeProvider([ModelStreamEvent.error("model service unavailable")]), default=True
        )
        app.register_environment_factory(EnvironmentSpec(name="coding"), Factory(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    session_id="settled-finalization-return",
                    agent_name="assistant",
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        try:
            await asyncio.wait_for(finalized.wait(), timeout=5)
            assert runner.is_closed
            assert not binding._coding_authorities
            if cancelled:
                task.cancel("stop after finalization settled")
                # Grouped cancellation is also an accepted lifecycle result;
                # this test is about the retained effect owner, not its shape.
                try:
                    await task
                except BaseException as error:
                    from cayu._exception_groups import exception_tree_contains

                    assert exception_tree_contains(error, asyncio.CancelledError)
                else:
                    pytest.fail("Caller cancellation was swallowed")
            else:
                allow_return.set()
                events = await task
                assert events[-1].type is EventType.SESSION_FAILED

            if recover and cancelled:
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id="settled-finalization-return")
                )

            drained = await app.drain_environment_cleanups(timeout_s=2)
            owner = app._environment_lifecycle._active_environment_setups.get(
                "settled-finalization-return"
            )
            assert drained, (
                "A successfully returned finalization was replayed after cancellation",
                finalize_calls,
                None if owner is None else str(owner.cleanup_error),
            )
            assert len(finalize_calls) == 1
            checkpoint = await app.session_store.load_checkpoint("settled-finalization-return")
            assert "pending_completion_finalization" not in checkpoint
        finally:
            allow_return.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await runner.close()

    asyncio.run(scenario())
