"""Completed invocation replacement requires exact provider disposal evidence."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_environment_allocation_recovery import (
    _CompletingModelProvider,
    _FakeRemoteFactory,
    _FakeRemoteProvider,
)
from tests.core.test_human_review import identity

from cayu import (
    AgentSpec,
    CayuApp,
    EnvironmentFactoryOperation,
    EnvironmentSpec,
    EventType,
    Message,
    ResumeRequest,
    RunRequest,
    SessionStatus,
    SQLiteSessionStore,
)
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._environment_lifecycle import (
    ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY as OWNERS,
)
from cayu.runtime._environment_lifecycle import (
    ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY as RECEIPTS,
)
from cayu.runtime._environment_lifecycle import (
    ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY as RECONNECT,
)


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize(
    "proof",
    [
        True,
        False,
        "unavailable",
        "conflicting_receipt",
        "changed_receipt",
        "noncompleted",
        "truthy",
        "failed",
        "pending_disposal",
    ],
)
def test_completed_allocation_resume_requires_disposal_proof(tmp_path, restart, proof):
    async def scenario():
        path = tmp_path / "session.sqlite"
        store = SQLiteSessionStore(path)
        remote = _FakeRemoteProvider()
        checked = []
        recovered = []
        original = None

        class Factory(_FakeRemoteFactory):
            execution_profile_identity = identity("completed-allocation-factory")

            async def recover_finalization_disposal(self, request, state):
                assert request.reconnect_metadata == original
                assert state == {"allocation": original["allocation_id"]}
                recovered.append(request)

            async def is_allocation_disposed(self, request):
                assert request.operation is EnvironmentFactoryOperation.RECONNECT
                assert request.reconnect_metadata == original
                checked.append(request)
                if proof == "unavailable":
                    raise RuntimeError("provider ownership is unavailable")
                if proof == "truthy":
                    return 1
                if proof == "changed_receipt":

                    def change(_session, current):
                        current[RECEIPTS]["remote"]["intent"]["allocation_id"] = (
                            "another-allocation"
                        )
                        return current

                    await runtime_checkpoint_session_store(store).transform_checkpoint(
                        "completed", change
                    )
                    return True
                return proof is True or proof == "failed"

        class Provider(_CompletingModelProvider):
            execution_profile_identity = identity("completed-allocation-provider")

        def make_app():
            factory, provider = Factory(remote), Provider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_environment_factory(
                EnvironmentSpec(name="remote", execution_profile_identity=identity("remote")),
                factory,
                default=True,
            )
            app.register_agent(AgentSpec(name="agent", model="fake-model"))
            return app, factory, provider

        app, factory, provider = make_app()
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="completed",
                        agent_name="agent",
                        messages=[Message.text("user", "save this draft")],
                    )
                )
            ]
            assert events[-1].type == EventType.SESSION_COMPLETED
            before = await store.load("completed")
            transcript = await store.load_transcript("completed")
            private = runtime_checkpoint_session_store(store)
            checkpoint = await private.load_checkpoint("completed")
            original = checkpoint[RECONNECT]["remote"]
            assert not checked
            # The fake provider does not own a real runner. Model its already
            # settled disposal separately from its durable affirmative proof.
            remote.resources.clear()
            if proof == "conflicting_receipt":

                def conflict(_session, current):
                    current[RECEIPTS]["remote"]["reconnect_metadata"]["resource_name"] = "other"
                    return current

                await private.transform_checkpoint("completed", conflict)
            if proof == "noncompleted":
                await store.update_status("completed", SessionStatus.INTERRUPTED)
            elif proof in {"failed", "pending_disposal"}:
                await store.update_status("completed", SessionStatus.FAILED)
                if proof == "pending_disposal":

                    def pending(_session, current):
                        current["environment_factory_pending_disposals"] = {
                            "remote": {
                                "reconnect_metadata": original,
                                "state": {"allocation": original["allocation_id"]},
                            },
                        }
                        return current

                    await private.transform_checkpoint("completed", pending)
            if restart:
                await store.close()
                store = SQLiteSessionStore(path)
                app, factory, provider = make_app()
                private = runtime_checkpoint_session_store(store)
            count = len(provider.requests)
            events = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="completed",
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]
            after = await store.load("completed")
            checkpoint = await private.load_checkpoint("completed")
            if proof is True or proof in {"failed", "pending_disposal"}:
                assert events[-1].type == EventType.SESSION_COMPLETED
                assert len(remote.create_calls) == 2
                assert factory.requests[-1].operation is EnvironmentFactoryOperation.CREATE
                assert factory.requests[-1].reconnect_metadata == {}
                assert checkpoint[RECONNECT]["remote"] != original
                assert checkpoint[OWNERS]["remote"] == before.id
                assert (
                    checkpoint[RECEIPTS]["remote"]["reconnect_metadata"]
                    == checkpoint[RECONNECT]["remote"]
                )
                assert len(provider.requests) == count + 1
                assert (await store.load_transcript("completed"))[: len(transcript)] == transcript
                assert after.instance_id == before.instance_id
            else:
                assert events[-1].type == EventType.SESSION_FAILED
                assert len(remote.create_calls) == 1
                assert len(provider.requests) == count
                assert checkpoint[RECONNECT]["remote"] == original
            assert len(checked) == (
                0 if proof in {"conflicting_receipt", "noncompleted", "pending_disposal"} else 1
            )
            assert len(recovered) == (1 if proof == "pending_disposal" else 0)
        finally:
            await store.close()

    asyncio.run(scenario())
