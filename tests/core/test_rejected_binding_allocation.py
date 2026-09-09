from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from tests.core.test_environment_allocation_recovery import _FakeRemoteFactory, _FakeRemoteProvider

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentAllocationState,
    EnvironmentFactoryReleaseAction,
    EnvironmentSpec,
    IncompleteSessionRecoveryRequest,
    LocalWorkspace,
    Message,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    SyncBinding,
)
from cayu.environments.factory import (
    attach_environment_factory_cleanup_settlement_task,
    register_environment_factory_cleanup_retry,
)


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "reap_error",
        "reap_cancel",
        "timeout",
        "fence_ack",
        "reaped_ack",
        "release_handoff",
        "release_grouped",
        "release_timeout",
        "release_retry_error",
        "release_retry_cancel",
        "release_reap_error",
    ],
)
def test_new_allocation_binding_rejection_retains_exact_cleanup(tmp_path, fault):
    source_path = tmp_path / "source"
    source_path.mkdir()
    source = LocalWorkspace(source_path)
    target_path = tmp_path / "owner"
    target_path.mkdir()
    rejected_path = tmp_path / "rejected"
    rejected_path.mkdir()
    provider = _FakeRemoteProvider()
    unblock = asyncio.Event()
    blocked = fault in {
        "reap_error",
        "reap_cancel",
        "release_retry_error",
        "release_retry_cancel",
        "release_reap_error",
    }
    timed_out = fault in {"timeout", "release_timeout"}

    class Store(SQLiteSessionStore):
        invocation_lifecycle_command_version = 1
        lost_ack = False

        async def transform_checkpoint(self, session_id, transform):
            await super().transform_checkpoint(session_id, transform)
            checkpoint = await self.load_checkpoint(session_id) or {}
            record = checkpoint.get("environment_factory_allocation_intents", {}).get("remote")
            if (
                not self.lost_ack
                and record is not None
                and record["state"] == {"fence_ack": "reaping", "reaped_ack": "reaped"}.get(fault)
            ):
                self.lost_ack = True
                raise TimeoutError("lost cleanup acknowledgement")

    class Factory(_FakeRemoteFactory):
        reaps = 0
        releases = 0
        detached = False

        def _result(self, request, resource, *, allocation=None):
            result = super()._result(request, resource, allocation=allocation)

            async def release(action):
                self.releases += 1
                assert action is EnvironmentFactoryReleaseAction.PRESERVE

                async def detach():
                    if fault == "release_timeout":
                        await unblock.wait()
                    if blocked and fault == "release_retry_error":
                        raise OSError("host detach unavailable")
                    if blocked and fault == "release_retry_cancel":
                        raise asyncio.CancelledError("host detach unavailable")
                    self.detached = True

                def start():
                    task = asyncio.create_task(detach())
                    register_environment_factory_cleanup_retry(task, start)
                    return task

                error = RuntimeError("host detach handed off")
                attach_environment_factory_cleanup_settlement_task(error, start())
                if fault == "release_grouped":
                    raise ExceptionGroup("grouped host release handoff", [error])
                raise error

            return replace(
                result,
                release=release if fault.startswith("release_") else result.release,
                environment=Environment(
                    EnvironmentSpec(name="remote"),
                    workspace=source,
                    binding=SyncBinding(
                        target_workspace=LocalWorkspace(rejected_path), sync_back="never"
                    ),
                ),
                release_timeout_s=0.05 if timed_out else 2,
            )

        async def reap_allocation(self, request, allocation):
            self.reaps += 1
            if fault.startswith("release_"):
                assert self.detached, "Provider deletion must wait for every host detach owner"
            if blocked and fault in {"reap_error", "reap_cancel", "release_reap_error"}:
                if fault == "reap_cancel":
                    raise asyncio.CancelledError("reaper unavailable")
                raise OSError("reaper unavailable")
            if fault == "timeout":
                await unblock.wait()
            assert await allocation.mark_reaping()
            intent = allocation.intent
            provider.reap(
                intent.provider_metadata["resource_name"],
                allocation_id=intent.allocation_id,
                session_id=intent.session_id,
                environment_name=intent.environment_name,
                adapter_generation=intent.adapter_generation,
            )
            await allocation.mark_reaped()

    async def scenario():
        nonlocal blocked
        owner = SyncBinding(target_workspace=LocalWorkspace(target_path), sync_back="never")
        bound = await owner.bind(source, None, session_id="existing-owner")
        store = Store(tmp_path / "sessions.db")
        factory = Factory(provider)
        app = CayuApp(session_store=store, enable_logging=False)
        model = ScriptedModelProvider([])
        app.register_provider(model, default=True)
        app.register_agent(AgentSpec(name="probe", model="scripted-model"))
        app.register_environment_factory(EnvironmentSpec(name="remote"), factory, default=True)
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="probe",
                        session_id="rejected",
                        messages=[Message.text("user", "finish")],
                    )
                )
            ]
            assert any(
                "already bound by an active session" in str(event.payload) for event in events
            )
            assert events[-1].type == "session.failed"
            assert not model.requests
            assert not any(event.type == "model.requested" for event in events)
            assert len(provider.create_calls) == 1
            if blocked or timed_out:
                failure = next(
                    event for event in events if event.type == "environment.binding.failed"
                )
                assert failure.payload["environment_factory_release"]["completed"] is False
                assert not await app.drain_environment_cleanups(timeout_s=0.05)
                assert len(provider.resources) == 1
                if fault in {"release_timeout", "release_retry_error", "release_retry_cancel"}:
                    assert factory.reaps == 0
                    checkpoint = await store.load_checkpoint("rejected")
                    assert checkpoint.get("environment_factory_allocation_receipts")
            blocked = False
            unblock.set()
            assert await app.drain_environment_cleanups(timeout_s=3)
            assert not provider.resources
            checkpoint = await store.load_checkpoint("rejected")
            record = checkpoint["environment_factory_allocation_intents"]["remote"]
            assert record["state"] == EnvironmentAllocationState.REAPED.value
            assert not checkpoint.get("environment_factory_allocation_receipts")
            assert not checkpoint.get("environment_factory_reconnect")
            reaps = factory.reaps
            for _ in range(2):
                assert await app.drain_environment_cleanups(timeout_s=1)
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id="rejected", reason="verify_cleanup")
                )
            assert not model.requests
            assert factory.reaps == reaps
            if fault.startswith("release_"):
                assert factory.releases == 1
            assert len(provider.create_calls) == 1
            contender = SyncBinding(target_workspace=LocalWorkspace(rejected_path))
            with pytest.raises(ValueError, match="already bound by an active session"):
                await contender.bind(source, None, session_id="still-rejected")
        finally:
            unblock.set()
            await owner.finalize(bound, outcome="completed")
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("conflict", ["epoch", "receipt", "owner"])
def test_rejected_publication_fence_rejects_changed_authority(conflict):
    from tests.core.test_environment_allocation_recovery import _create_session, _resolve

    from cayu import EnvironmentFactoryOperation, InMemorySessionStore
    from cayu.runtime._environment_allocation import (
        EnvironmentAllocationCoordinator,
        EnvironmentAllocationTransitionConflict,
    )
    from cayu.vaults import SecretRedactor

    async def scenario():
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        result = await _resolve(
            store,
            session,
            _FakeRemoteFactory(provider),
            operation=EnvironmentFactoryOperation.CREATE,
        )
        assert result.error is None
        coordinator = EnvironmentAllocationCoordinator(
            session_store=store,
            checkpoint_transform=lambda candidate: lambda _session, _current: candidate,
            secret_redactor=SecretRedactor(),
        )
        checkpoint = await store.load_checkpoint(session.id)
        receipt = coordinator.receipt_from_checkpoint(checkpoint, environment_name="remote")
        assert receipt is not None
        if conflict == "epoch":
            session = session.model_copy(update={"run_epoch": session.run_epoch + 1})
        elif conflict == "receipt":
            receipt = replace(receipt, reconnect_metadata={"resource_name": "different"})
        else:
            receipt = replace(receipt, intent=replace(receipt.intent, session_id="other-session"))
        with pytest.raises(EnvironmentAllocationTransitionConflict):
            await coordinator.reclaim_rejected_binding_publication(session=session, receipt=receipt)
        assert await store.load_checkpoint(session.id) == checkpoint
        assert len(provider.resources) == 1
        assert not provider.reap_calls

    asyncio.run(scenario())
