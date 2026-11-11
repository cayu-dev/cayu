"""Cancellation cannot turn an unproven mutation into a captured endpoint."""

import asyncio

import pytest
from tests.core.test_workspace_mutation_receipts import (
    DeterministicWorkspaceBinding,
    _DetachedRunnerMutationTool,
    _FailingDeferredBackgroundMutationRunner,
    _portable_environment_spec,
)

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import Environment
from cayu.evals import ScriptedModelProvider
from cayu.observability.events import EventSink
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.tools._resources import InvocationWorkspaceMutationOwner
from cayu.workspaces import LocalWorkspace


@pytest.mark.parametrize("cancel_during_settlement", [False, True])
def test_unproven_settlement_never_captures_a_live_writer(
    tmp_path, monkeypatch, cancel_during_settlement
):
    async def scenario():
        sealing = asyncio.Event()
        release_failure = asyncio.Event()
        unsafe_observation = asyncio.Event()
        observation_finalized = asyncio.Event()
        receipts_before_quiescence = []
        original_seal = InvocationWorkspaceMutationOwner.seal_and_wait
        settlement_errors = []

        async def seal(owner):
            sealing.set()
            try:
                return await original_seal(owner)
            except BaseException as error:
                settlement_errors.append(error)
                raise

        monkeypatch.setattr(InvocationWorkspaceMutationOwner, "seal_and_wait", seal)

        class DisconnectedRunner(_FailingDeferredBackgroundMutationRunner):
            async def exec(self, command, **kwargs):
                await super().exec(command, **kwargs)
                await release_failure.wait()
                raise OSError("connection lost while the remote writer remains active")

        runner = DisconnectedRunner(tmp_path)

        class Binding(DeterministicWorkspaceBinding):
            async def observe_revision(self, bound):
                if runner.started.is_set() and not runner.mutation_finished.is_set():
                    unsafe_observation.set()
                return await super().observe_revision(bound)

        class WatchEvidence(EventSink):
            async def emit(self, event):
                if event.type is EventType.WORKSPACE_MUTATION_RECORDED:  # noqa: SIM102
                    if not runner.mutation_finished.is_set():
                        receipts_before_quiescence.append(event)
                if event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED:
                    observation_finalized.set()

        store = InMemorySessionStore()
        app = CayuApp(session_store=store, event_sinks=[WatchEvidence()], enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="write", name="detached_runner_mutation", arguments={}
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
                ]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                runner=runner,
                workspace=LocalWorkspace(tmp_path, workspace_id="cancelled-settlement"),
                binding=Binding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="worker", model="scripted-model"), tools=[_DetachedRunnerMutationTool()]
        )

        async def consume():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="cancelled-settlement",
                        agent_name="worker",
                        messages=[Message.text("user", "write")],
                    )
                )
            ]

        consumer = asyncio.create_task(consume())
        observation_wait = asyncio.create_task(observation_finalized.wait())
        try:
            await asyncio.wait_for(sealing.wait(), 10)
            await asyncio.wait_for(runner.started.wait(), 10)
            if cancel_during_settlement:
                consumer.cancel("cancel while waiting for the detached writer")
                # Deliver caller control while seal_and_wait is still waiting.
                for _ in range(12):
                    await asyncio.sleep(0)
            release_failure.set()
            await asyncio.wait(
                {consumer, observation_wait}, timeout=5, return_when=asyncio.FIRST_COMPLETED
            )
            observed_before_quiescence = unsafe_observation.is_set()
        finally:
            release_failure.set()
            runner.release_mutation.set()
            await asyncio.gather(consumer, return_exceptions=True)
            observation_wait.cancel()
            await asyncio.gather(observation_wait, return_exceptions=True)
            if runner._mutation is not None:
                await runner._mutation
            await app.drain_environment_cleanups(timeout_s=10)
            await app.aclose()

        assert settlement_errors, "The detached runner must fail settlement"
        if cancel_during_settlement:
            assert isinstance(settlement_errors[0], asyncio.CancelledError)
            assert consumer.cancelled()
        durable = await store.load_events("cancelled-settlement")
        assert observation_finalized.is_set(), "The workspace outcome must be closed"
        assert (tmp_path / "uncertain-runner.txt").read_bytes() == b"late"
        assert not receipts_before_quiescence, (
            "A receipt was published before the uncertain writer stopped: "
            f"{[event.payload['status'] for event in receipts_before_quiescence]}"
        )
        assert not observed_before_quiescence, (
            "The runtime observed an after endpoint while the writer's settlement was unproven"
        )
        terminal = next(event for event in durable if event.type is EventType.TOOL_CALL_COMPLETED)
        assert terminal.payload["workspace_mutation_capture_status"] == "failed"
        assert (
            terminal.payload["workspace_mutation_capture_detail_code"]
            == "mutation_settlement_unproven"
        )

    asyncio.run(scenario())
