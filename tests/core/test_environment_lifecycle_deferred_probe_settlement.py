"""A deferred admission failure must settle its live probe before disposal."""

import asyncio

from tests.core.test_deferred_environments import DeferredLocalFactory, _app, _turns
from tests.runtime.test_execution_admission_dispatch import _candidate, _EvidenceRunner

from cayu import EventType, Message, RunRequest
from cayu.environments import Environment, EnvironmentFactoryResult, EnvironmentSpec
from cayu.environments.deferred import DeferredMaterialization
from cayu.environments.factory import attach_environment_factory_cleanup_settlement_task
from cayu.sessions import InMemorySessionStore


def test_deferred_admission_settles_failed_probe_before_closing_runner(tmp_path):
    async def scenario():
        class ProbeRunner(_EvidenceRunner):
            def __init__(self):
                super().__init__("deferred-probe")
                self.probe_started = asyncio.Event()
                self.allow_settlement = asyncio.Event()
                self.probe_settled = False
                self.close_overtook_probe = False
                self.settlement = None

            async def collect_execution_admission_candidate(self):
                async def settle_probe():
                    await self.allow_settlement.wait()
                    self.probe_settled = True

                self.settlement = asyncio.create_task(settle_probe())
                error = ConnectionError("Admission probe acknowledgement lost")
                attach_environment_factory_cleanup_settlement_task(error, self.settlement)
                self.probe_started.set()
                raise error

            async def close(self):
                self.close_overtook_probe |= not self.probe_settled
                await super().close()

        runner = ProbeRunner()

        class Factory(DeferredLocalFactory):
            def execution_admission_candidate(self, request):
                return _candidate("deferred-probe", state="available")

            async def create(self, request):
                async def materialize(mode):
                    return runner, None

                state = DeferredMaterialization(
                    materialize,
                    default_cwd=str(tmp_path),
                    configured_candidate=lambda requirements: self.execution_admission_candidate(
                        request
                    ),
                )
                return EnvironmentFactoryResult(
                    environment=Environment(
                        EnvironmentSpec(name=request.environment_name),
                        runner=state.runner,
                        binding=state.binding,
                    ),
                    reconnect_metadata={"kind": "deferred-probe"},
                )

        app = _app(Factory(tmp_path, tmp_path), InMemorySessionStore(), _turns("write"))

        async def collect():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="deferred-probe-settlement",
                        messages=[Message.text("user", "Use the runner once")],
                    )
                )
            ]

        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(runner.probe_started.wait(), timeout=10)
            # Let the failed collection unwind. A correct implementation either
            # waits here or retains cleanup, but cannot close the concrete runner.
            await asyncio.sleep(0.05)
            runner.allow_settlement.set()
            events = await asyncio.wait_for(task, timeout=10)
            await runner.settlement
            assert any(e.type is EventType.ENVIRONMENT_MATERIALIZATION_FAILED for e in events)
            assert not runner.close_overtook_probe, (
                "Deferred admission closed the allocation while its authenticated probe "
                "settlement was still pending"
            )
            assert await app.drain_environment_cleanups(timeout_s=2)
        finally:
            runner.allow_settlement.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if runner.settlement is not None:
                await runner.settlement
            await app.close_idle_environment_resources()

    asyncio.run(scenario())
