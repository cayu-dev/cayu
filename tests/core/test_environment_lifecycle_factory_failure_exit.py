"""A preserved pre-adoption allocation still needs the process-local exit backstop."""

from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace

import pytest
from tests.core.test_environment_lifecycle import _collect_events, _FakeProvider
from tests.docker_toolchain import docker_toolchain_profile
from tests.environments.test_docker_coding import _image_identity

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import DockerCodingEnvironmentFactory, EnvironmentSpec
from cayu.runners import _exit_cleanup
from cayu.runners.docker import DockerRunner
from cayu.sessions import InMemorySessionStore
from cayu.workspaces import LocalWorkspace


@pytest.mark.parametrize(
    "lose_publication", [False, True], ids=["held-result", "publication-fails"]
)
def test_factory_publication_failure_keeps_process_local_exit_owner(
    tmp_path, monkeypatch, lose_publication
):
    container_id = "c" * 64
    removed = []
    original_run = subprocess.run

    def docker_rm(argv, **kwargs):
        if list(argv[:2]) != ["/usr/bin/docker", "rm"]:
            return original_run(argv, **kwargs)
        removed.append(list(argv))
        return SimpleNamespace(returncode=0)

    # Exercise the actual DockerRunner exit close, without a Docker daemon.
    monkeypatch.setattr("cayu.runners.docker.subprocess.run", docker_rm)

    async def scenario():
        runner = DockerRunner(
            "owned",
            close_action="remove",
            docker_path="/usr/bin/docker",
            _container_id=container_id,
        )
        factory = DockerCodingEnvironmentFactory(
            source_workspace=LocalWorkspace(tmp_path, workspace_id="source"),
            toolchain_profile=docker_toolchain_profile(image_identity=_image_identity()),
            docker_path="/usr/bin/docker",
            git_baseline=False,
        )
        created = []

        async def create_runner(name, **kwargs):
            created.append((name, kwargs["process_local_owner"]))
            return runner

        async def toolchain_probes(*args, **kwargs):
            pass

        # Only provider allocation and admission probes are substituted. The
        # built-in factory creates and publishes its real recoverable result,
        # including its own PRESERVE callback and Docker coding binding.
        monkeypatch.setattr(factory, "_create_or_recover_runner", create_runner)
        monkeypatch.setattr(
            "cayu.environments.docker_coding._run_toolchain_admission_probes", toolchain_probes
        )
        monkeypatch.setattr(
            factory,
            "_require_final_evidence",
            lambda _runner, requirements: factory._configured_candidate(requirements).evidence,
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        provider = _FakeProvider([])
        app.register_provider(provider, default=True)
        app.register_environment_factory(EnvironmentSpec(name="docker"), factory, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        emit = app._event_writer.emit

        async def fail_completion(event):
            if lose_publication and event.type is EventType.ENVIRONMENT_FACTORY_COMPLETED:
                raise ConnectionError("factory-completed event store unavailable")
            return await emit(event)

        monkeypatch.setattr(app._event_writer, "emit", fail_completion)
        request = RunRequest(
            session_id="process-local-publication-failure",
            agent_name="assistant",
            messages=[Message.text("user", "run")],
        )
        stream = None
        try:
            if lose_publication:
                events = await _collect_events(app, request)
                assert events[-1].type is EventType.SESSION_FAILED
                failure = next(e for e in events if e.type is EventType.ENVIRONMENT_FACTORY_FAILED)
                assert failure.payload["environment_factory_release"]["action"] == "preserve"
                assert failure.payload["environment_factory_release"]["completed"] is True
            else:
                # Control: a returned result that reached promotion is visible
                # to the exit hook, even before its workspace has been bound.
                stream = app.run(request)
                async for event in stream:
                    if event.type is EventType.ENVIRONMENT_FACTORY_COMPLETED:
                        break
            assert provider.requests == []
            assert len(created) == 1 and created[0][1] is not None
            assert not runner.is_closed
            checkpoint = await store.load_checkpoint("process-local-publication-failure")
            assert (
                checkpoint["environment_factory_reconnect"]["docker"]["container_id"]
                == container_id
            )
            assert app._environment_lifecycle._exit_hook is not None
            assert not _exit_cleanup._unsettled  # No runner close has started.

            # Run the same registered hook a normal interpreter exit runs. The
            # only reconnect state is in memory and cannot survive this process.
            _exit_cleanup._release_held_at_exit()
            assert removed == [["/usr/bin/docker", "rm", "-f", container_id]], (
                "The process-local allocation disappeared from exit cleanup ownership",
                list(app._environment_lifecycle._active_environment_setups),
            )
            assert runner.container_removed
        finally:
            if stream is not None:
                await stream.aclose()

    asyncio.run(scenario())
