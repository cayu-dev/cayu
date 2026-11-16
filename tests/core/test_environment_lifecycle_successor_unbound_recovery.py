"""A settled predecessor binding is not cleanup authority for a new allocation."""

import asyncio

import pytest
from tests.environments.test_docker_coding import _coding_product_test_binding
from tests.runtime.runtime_provider_support import FakeProvider

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu.environments import Environment, EnvironmentSpec
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryResult,
)
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.storage import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite-reopened"])
@pytest.mark.parametrize(
    "fail_successor_publication",
    [False, True],
    ids=["bound-successor", "unbound-successor"],
)
def test_terminal_recovery_ignores_a_released_predecessor_binding(
    tmp_path, monkeypatch, fail_successor_publication, backend
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "result.txt").write_text("saved output")
    allocations = []
    for number in range(2):
        target = tmp_path / f"target-{number}"
        target.mkdir()
        runner, source, binding = _coding_product_test_binding(source_root, target)
        # The reusable harness substitutes a local directory for Docker I/O.
        # Give each allocation a distinct exact container identity as well.
        runner.container_id = str(number + 1) * 64
        allocations.append((runner, source, binding))

    class Factory(EnvironmentFactory):
        def __init__(self):
            self.creates = 0
            self.release_actions = []

        async def create(self, request):
            assert request.operation is EnvironmentFactoryOperation.CREATE
            runner, source, binding = allocations[self.creates]
            self.creates += 1

            async def release(action):
                self.release_actions.append(action)
                # PRESERVE detaches this result; its exact remote allocation
                # stays available to a later invocation.
                assert action is EnvironmentFactoryReleaseAction.PRESERVE

            return EnvironmentFactoryResult(
                Environment(
                    EnvironmentSpec(name="coding"),
                    workspace=source,
                    runner=runner,
                    binding=binding,
                ),
                reconnect_metadata={"container_id": runner.container_id},
                allocation_fingerprint=str(self.creates) * 64,
                release=release,
            )

        async def is_allocation_disposed(self, request):
            container_id = request.reconnect_metadata["container_id"]
            return next(
                runner.is_closed
                for runner, _, _ in allocations
                if runner.container_id == container_id
            )

        async def recover_finalization_disposal(self, request, state):
            raise AssertionError("Neither allocation has unfinished binding disposal")

    async def scenario():
        factory = Factory()
        database = tmp_path / "sessions.sqlite"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(database)

        def make_app():
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(
                FakeProvider([[ModelStreamEvent.completed({"finish_reason": "stop"})]] * 2),
                default=True,
            )
            app.register_environment_factory(EnvironmentSpec(name="coding"), factory, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            return app

        app = make_app()
        emit = app._event_writer.emit
        failed_publications = 0

        async def emit_with_failure(event):
            nonlocal failed_publications
            if (
                fail_successor_publication
                and factory.creates == 2
                and event.type is EventType.ENVIRONMENT_FACTORY_COMPLETED
                and not failed_publications
            ):
                failed_publications += 1
                raise ConnectionError("successor factory publication unavailable")
            return await emit(event)

        monkeypatch.setattr(app._event_writer, "emit", emit_with_failure)
        try:
            first = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="predecessor-binding",
                        messages=[Message.text("user", "first run")],
                    )
                )
            ]
            assert first[-1].type is EventType.SESSION_COMPLETED
            assert allocations[0][0].is_closed
            assert any(
                event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload.get("phase") == "release"
                and event.payload.get("outcome") == "released"
                and event.payload.get("release_action") == "discard"
                for event in first
            )
            second = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="predecessor-binding",
                        messages=[Message.text("user", "next run")],
                    )
                )
            ]
            assert factory.creates == 2
            assert failed_publications == int(fail_successor_publication)
            assert second[-1].type is (
                EventType.SESSION_FAILED
                if fail_successor_publication
                else EventType.SESSION_COMPLETED
            )
            assert factory.release_actions == (
                [EnvironmentFactoryReleaseAction.PRESERVE] if fail_successor_publication else []
            )
            assert await app.drain_environment_cleanups(timeout_s=2)
            # Recovery runs with the event writer healthy again. SQLite also
            # exercises a fresh application with no process-local setup owners.
            monkeypatch.setattr(app._event_writer, "emit", emit)
            if backend == "sqlite-reopened":
                await store.close()
                store = SQLiteSessionStore(database)
                app = make_app()
            recovered = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="predecessor-binding")
            )
            assert recovered.status.value == (
                "failed" if fail_successor_publication else "completed"
            )
        finally:
            for runner, _, _ in allocations:
                await runner.close()
            if backend == "sqlite-reopened":
                await store.close()

    asyncio.run(scenario())
