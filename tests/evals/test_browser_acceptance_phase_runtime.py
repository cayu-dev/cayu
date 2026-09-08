"""Real tool hooks and durable browser publication; backend/site behavior is controlled.

This tests phase attribution, not another live operator-login proof. The operator
projection is supplied from confirmed observation records; handback itself has its
own runtime and live tests.
"""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from tests.core.test_browser_session import _FakeBrowserBackend
from tests.core.test_environment_allocation_recovery import _FakeRemoteFactory, _FakeRemoteProvider
from tests.evals.test_browser_acceptance_operator_oracle import (
    _project_operator_record,
    _settled_record,
)

from cayu import (
    AgentSpec,
    CayuApp,
    EnvironmentSpec,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
)
from cayu.evals import (
    BrowserAcceptanceAuthenticationCollector,
    EvalCase,
    EvalSuite,
    SessionCompleted,
    run_eval_suite,
)
from cayu.evals.browser_acceptance import _case_authentication_evidence
from cayu.evals.corpus import _content_revision
from cayu.tools.browser_session import (
    BrowserSessionTool,
    _durable_browser_operation_key,
    _RunnerBrowserSessionBackend,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("first_only", [False, True])
def test_runtime_phase_samples_cannot_assign_first_browser_requests_to_restoration(
    tmp_path, monkeypatch, backend, first_only
):
    async def scenario():
        count = 0
        navigations = 0
        fake = _FakeBrowserBackend()
        collector = BrowserAcceptanceAuthenticationCollector(
            lambda: count, observer_revision="sha256:" + "a" * 64
        )
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "session.sqlite")
        )
        app = CayuApp(session_store=store, enable_logging=False, runtime_hooks=[collector])

        async def preflight(self, ctx, args):
            return await fake.preflight(ctx, args)

        async def execute(self, ctx, args):
            nonlocal count, navigations
            if args["operation"] == "navigate":
                navigations += 1
                count += (2 if navigations == 1 else 0) if first_only else 1
                fake.title = "Login" if first_only and navigations == 2 else "Authenticated"
            return await fake.execute(ctx, args)

        monkeypatch.setattr(_RunnerBrowserSessionBackend, "preflight", preflight)
        monkeypatch.setattr(_RunnerBrowserSessionBackend, "execute", execute)
        operation_ids = (
            "open",
            "acceptance-post-handback",
            "close",
            "reopen",
            "acceptance-restored",
            "reclose",
        )

        class Provider(ScriptedModelProvider):
            index = 0

            async def stream(self, request):
                if self.index == 6:
                    yield ModelStreamEvent.text_delta("done")
                    yield ModelStreamEvent.completed({"finish_reason": "stop"})
                    return
                operation = ("navigate", "observe", "close")[self.index % 3]
                args = {"operation": operation, "operation_id": operation_ids[self.index]}
                if operation == "navigate":
                    args["url"] = "https://example.test/form"
                else:
                    args["session_id"] = fake.session_id
                    if operation == "observe":
                        args["page_id"] = fake.page_id
                self.index += 1
                yield ModelStreamEvent.tool_call(
                    id=f"call-{self.index}", name="browser_session", arguments=args
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

        app.register_provider(Provider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="browser"), _FakeRemoteFactory(_FakeRemoteProvider()), default=True
        )
        app.register_agent(AgentSpec(name="agent", model="model"), tools=[BrowserSessionTool()])
        collector._begin()
        try:
            run = await run_eval_suite(
                app,
                EvalSuite(
                    id="phase",
                    cases=[
                        EvalCase(
                            id="phase",
                            request=RunRequest(
                                agent_name="agent",
                                messages=[Message.text("user", "test")],
                                max_steps=7,
                            ),
                            assertions=[SessionCompleted()],
                        )
                    ],
                ),
                retain_trajectory=True,
            )
            samples = collector._finish()
            trial = run.cases[0].trials[0]
            assert trial.status.value == "passed", trial
            assert len(samples) == 6
            assert count == 2
            records = [
                await store.load_session_operation(
                    trial.session_id, _durable_browser_operation_key(key)
                )
                for key in operation_ids
            ]

            def browser(index):
                return _content_revision(
                    {"session_id": samples[index].browser_session_id},
                    "browser acceptance browser session",
                )

            operator = _project_operator_record(_settled_record()).model_copy(
                update={
                    "browser_session_revision": browser(0),
                    "restored_browser_session_revision": browser(3),
                    "fresh_observation_revision": _content_revision(
                        records[1], "browser operator fresh observation"
                    ),
                    "restored_observation_revision": _content_revision(
                        records[4], "browser operator fresh observation"
                    ),
                }
            )
            phases = await _case_authentication_evidence(app, trial, samples, operator)
            assert [phase.authenticated_requests for phase in phases] == (
                [2, 0] if first_only else [1, 1]
            )
            for field, value in (
                ("tool_call_id", "other"),
                ("run_epoch", 999),
                ("arguments_sha256", "b" * 64),
                ("operation_id", "missing"),
            ):
                altered = (replace(samples[0], **{field: value}), *samples[1:])
                with pytest.raises(ValueError):
                    await _case_authentication_evidence(app, trial, altered, operator)
            assert not collector._active and collector._samples == []
        finally:
            collector._finish()
            assert await app.drain_environment_cleanups(timeout_s=5)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


def test_collector_rejects_overlap_and_unpaired_hook_results():
    async def scenario():
        collector = BrowserAcceptanceAuthenticationCollector(
            lambda: 0, observer_revision="sha256:" + "a" * 64
        )
        collector._begin()
        before = SimpleNamespace(tool_name="browser_session", tool_call_id="first")
        await collector.before_tool_call(before)
        await collector.before_tool_call(before)
        assert collector._finish() == ()
        collector._begin()
        await collector.after_tool_call(SimpleNamespace(tool_call_id="unpaired"))
        assert collector._finish() == ()

    asyncio.run(scenario())
