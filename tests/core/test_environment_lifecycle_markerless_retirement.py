"""Markerless abort retirement readback survives recovery epoch changes."""

import asyncio

import pytest
from tests.environments.test_docker_coding import _CONTAINER_ID, _coding_product_test_binding
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.environments import Environment, EnvironmentSpec
from cayu.environments.factory import EnvironmentFactory, EnvironmentFactoryResult
from cayu.providers import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest


@pytest.mark.parametrize("failure", ["before-commit", "lost-ack"])
def test_markerless_abort_retirement_survives_public_recovery(tmp_path, monkeypatch, failure):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    runner, source, binding = _coding_product_test_binding(source_root, target_root)

    class RetirementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self):
            super().__init__()
            self.failures = 0

        async def transform_checkpoint(self, session_id, checkpoint_transform):
            lost_ack = False

            def transform(session, current):
                nonlocal lost_ack
                updated = checkpoint_transform(session, current)
                if (
                    not self.failures
                    and (current or {}).get("environment_factory_pending_disposals")
                    and not (updated or {}).get("environment_factory_pending_disposals")
                    and (updated or {}).get("environment_factory_retired_disposals")
                ):
                    self.failures += 1
                    if failure == "before-commit":
                        raise ConnectionError("retirement unavailable")
                    lost_ack = True
                return updated

            await super().transform_checkpoint(session_id, transform)
            if lost_ack:
                raise ConnectionError("retirement acknowledgement lost")

    class Factory(EnvironmentFactory):
        async def create(self, request):
            return EnvironmentFactoryResult(
                Environment(
                    EnvironmentSpec(name="coding"), workspace=source, runner=runner, binding=binding
                ),
                reconnect_metadata={"container_id": _CONTAINER_ID},
                allocation_fingerprint="a" * 64,
            )

        async def recover_finalization_disposal(self, request, state):
            raise AssertionError("Disposed binding must not be dispatched again")

    async def scenario():
        store = RetirementStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]), default=True
        )
        app.register_environment_factory(EnvironmentSpec(name="coding"), Factory(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        lifecycle = app._environment_lifecycle
        original_bind = lifecycle.bind

        async def bind_then_fail(**kwargs):
            result = await original_bind(**kwargs)
            assert result.error is None
            await lifecycle.abort_environment_setup(
                session_id=kwargs["session"].id,
                original_error=RuntimeError("setup interrupted after binding"),
            )
            raise RuntimeError("setup interrupted after binding")

        monkeypatch.setattr(lifecycle, "bind", bind_then_fail)
        try:
            with pytest.raises(ConnectionError, match="retirement"):
                await collect_events(
                    app,
                    RunRequest(
                        agent_name="assistant",
                        session_id="markerless-retirement",
                        messages=[Message.text("user", "run")],
                    ),
                )
            assert store.failures == 1
            checkpoint = await store.load_checkpoint("markerless-retirement")
            assert "pending_completion_finalization" not in checkpoint
            owner = lifecycle._active_environment_setups["markerless-retirement"]
            assert owner.pending_disposal_retirement is not None
            assert runner.is_closed
            assert not binding._coding_authorities
            epoch = owner.invocation_context.binding.run_epoch
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="markerless-retirement")
            )
            assert (await store.load("markerless-retirement")).run_epoch > epoch
            assert await app.drain_environment_cleanups(timeout_s=2)
            assert "markerless-retirement" not in lifecycle._active_environment_setups
            checkpoint = await store.load_checkpoint("markerless-retirement")
            assert "pending_completion_finalization" not in checkpoint
            assert "environment_factory_pending_disposals" not in checkpoint
            assert "environment_factory_reconnect" not in checkpoint
            assert "coding" in checkpoint["environment_factory_retired_disposals"]
        finally:
            await runner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "conflict",
    [
        None,
        "instance",
        "uncommitted",
        "different-retirement",
        "environment_factory_pending_disposals",
        "environment_factory_reconnect",
        "environment_factory_allocation_owner",
        "environment_factory_allocation_receipts",
        "environment_factory_allocation_identity",
        "environment_factory_allocation_identity_binding",
        "environment_factory_allocation_intents",
    ],
)
def test_stale_retirement_only_acknowledges_exact_committed_state(conflict):
    from tests.core.test_environment_lifecycle import _lifecycle

    from cayu.sessions.authority import SessionRunFenced
    from cayu.sessions.records import SessionIdentity

    async def scenario():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="retirement",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        expected = {"reconnect_metadata": {"container_id": _CONTAINER_ID}, "state": {"version": 1}}
        checkpoint = {
            "environment_factory_retired_disposals": {
                "coding": {**expected, "allocation_fingerprint": "a" * 64}
            },
            "unrelated": {"preserved": True},
        }
        if conflict == "uncommitted":
            checkpoint.pop("environment_factory_retired_disposals")
            checkpoint["environment_factory_pending_disposals"] = {"coding": expected}
            checkpoint["environment_factory_reconnect"] = {"coding": expected["reconnect_metadata"]}
            checkpoint["environment_factory_allocation_owner"] = {"coding": session.id}
        elif conflict == "different-retirement":
            checkpoint["environment_factory_retired_disposals"]["coding"]["state"] = {"version": 2}
        elif conflict is not None and conflict.startswith("environment_factory_"):
            checkpoint[conflict] = {"coding": {}}
        await store.checkpoint(session.id, checkpoint)

        # A stale epoch cannot perform a new retirement, even with an equal
        # tombstone beside successor authority. It may only acknowledge one.
        async def retire():
            await _lifecycle(store)._retire_disposed_allocation(
                session.id,
                "coding",
                expected,
                session_instance_id="another-instance"
                if conflict == "instance"
                else session.instance_id,
                run_epoch=session.run_epoch - 1,
            )

        if conflict is None:
            await retire()
        else:
            with pytest.raises(SessionRunFenced):
                await retire()
        assert await store.load_checkpoint(session.id) == checkpoint

    asyncio.run(scenario())
