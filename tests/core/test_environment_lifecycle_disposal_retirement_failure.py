"""A retirement-store failure must not re-dispatch a disposed binding."""

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
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest


@pytest.mark.parametrize("failure", [None, "before-commit", "lost-ack"])
@pytest.mark.parametrize("recover", [False, True], ids=["drain", "recovery"])
def test_disposal_retirement_retries_durability_not_finalized_binding(tmp_path, failure, recover):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "result.txt").write_text("saved work")
    runner, source, binding = _coding_product_test_binding(source_root, target_root)

    class RetirementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self):
            super().__init__()
            self.inject_failure = failure is not None
            self.failures = 0

        async def transform_checkpoint(self, session_id, checkpoint_transform):
            failed_after_commit = False

            def transform(session, current):
                nonlocal failed_after_commit
                updated = checkpoint_transform(session, current)
                retiring = (
                    "coding" in (current or {}).get("environment_factory_pending_disposals", {})
                    and "coding"
                    not in (updated or {}).get("environment_factory_pending_disposals", {})
                    and "coding" in (updated or {}).get("environment_factory_retired_disposals", {})
                )
                if retiring and self.inject_failure:
                    self.inject_failure = False
                    self.failures += 1
                    if failure == "before-commit":
                        raise ConnectionError("retirement checkpoint unavailable")
                    failed_after_commit = True
                return updated

            await super().transform_checkpoint(session_id, transform)
            if failed_after_commit:
                raise ConnectionError("retirement committed but acknowledgement lost")

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
            assert state["container_id"] == _CONTAINER_ID
            await runner.close()

    async def scenario():
        store = RetirementStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            FakeProvider(
                [
                    ModelStreamEvent.error("model service unavailable"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ),
            default=True,
        )
        app.register_environment_factory(EnvironmentSpec(name="coding"), Factory(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        try:
            events = await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="retirement-store-failure",
                    messages=[Message.text("user", "run")],
                ),
            )
            assert events[-1].type is EventType.SESSION_FAILED
            assert store.failures == int(failure is not None)
            # Physical finalization is over, including the binding's private
            # generation. Only its durable retirement may still need retrying.
            assert runner.is_closed
            assert not binding._states
            assert not binding._coding_finalize_states
            assert not binding._coding_authorities

            if recover:
                recovery_error = None
                try:
                    await app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(session_id="retirement-store-failure")
                    )
                except Exception as error:
                    recovery_error = error
                assert recovery_error is None, (
                    "Recovery cannot settle the retirement of a disposed binding",
                    str(recovery_error),
                )
            drained = await app.drain_environment_cleanups(timeout_s=2)
            owner = app._environment_lifecycle._active_environment_setups.get(
                "retirement-store-failure"
            )
            assert drained, (
                "Durability retry re-entered an already disposed binding",
                None if owner is None else str(owner.cleanup_error),
            )
            checkpoint = await store.load_checkpoint("retirement-store-failure")
            assert "pending_completion_finalization" not in checkpoint
            assert "environment_factory_pending_disposals" not in checkpoint
            assert "environment_factory_reconnect" not in checkpoint
            assert "environment_factory_allocation_owner" not in checkpoint
            assert "coding" in checkpoint["environment_factory_retired_disposals"]
        finally:
            await runner.close()

    asyncio.run(scenario())
