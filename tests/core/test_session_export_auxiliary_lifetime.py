"""Tool admission expires before the auxiliary owner's asynchronous drain."""

import asyncio
import threading

import pytest
from tests.core.test_session_export_content_release import run_async
from tests.core.test_session_export_runtime import RuntimePolicy
from tests.core.test_session_exports import CONTEXT, harness, published
from tests.core.test_session_exports import backend as backend

from cayu import AuxiliaryInferencePolicy, InferenceLimits, ModelRequest
from cayu.agents import AgentSpec
from cayu.collaboration.exports import SessionExportDenied
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import RunRequest
from cayu.sessions.invocation import InvocationOriginClaim
from cayu.tools.base import Tool, ToolResult, ToolSpec


@pytest.mark.parametrize("tool_fails", [False, True])
@run_async
async def test_export_admission_closed_during_auxiliary_cleanup(backend, tool_fails):
    async with harness(backend) as case:
        app, store, _, projector = case.app(policy=RuntimePolicy())
        await case.create(store)
        first = await case.request(app, "already-admitted")
        late = await case.request(app, "after-tool-exit")
        dispatched = asyncio.Event()
        cleaning = asyncio.Event()
        release_cleanup = asyncio.Event()
        checked = asyncio.Event()
        children = []
        limits = InferenceLimits(max_input_tokens=100, max_output_tokens=10, timeout_seconds=20)

        class Provider(ScriptedModelProvider):
            async def stream(self, request):
                if request.messages == [Message.text("user", "nested")]:
                    self.requests.append(request.model_copy(deep=True))
                    dispatched.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        cleaning.set()
                        await release_cleanup.wait()
                        raise
                    return
                async for event in super().stream(request):
                    yield event

        class ExportAndAbandon(Tool):
            spec = ToolSpec(
                name="export_and_abandon",
                description="Exercise retained cleanup",
                input_schema={"type": "object"},
                auxiliary_inference=AuxiliaryInferencePolicy(
                    limits=limits, purposes=("tool.summary",)
                ),
            )

            async def run(self, ctx, args):
                children.append(
                    asyncio.create_task(
                        ctx.inference.invoke(
                            ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                            purpose="tool.summary",
                            limits=limits,
                        )
                    )
                )
                await dispatched.wait()
                projector.release = threading.Event()
                observer = asyncio.create_task(
                    app.export_session(first, context=CONTEXT, invocation=ctx)
                )
                assert await asyncio.to_thread(projector.entered.wait, 5)
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                assert observer.cancelled() and observer.cancelling() == 1

                async def attempt_during_cleanup():
                    await cleaning.wait()
                    with pytest.raises(SessionExportDenied):
                        await app.export_session(late, context=CONTEXT, invocation=ctx)
                    projector.release.set()
                    await asyncio.gather(*app._session_export_coordinator.owners.pending)
                    receipt = (await app.lookup_session_export(first, context=CONTEXT)).receipt
                    assert receipt.expected.intent.authorization.runtime.tool_call_id == "origin"
                    assert (
                        await app.lookup_session_export(late, context=CONTEXT)
                    ).status == "not_found"
                    checked.set()

                children.append(asyncio.create_task(attempt_during_cleanup()))
                if tool_fails:
                    raise ValueError("Tool body failed before auxiliary drain")
                return ToolResult(content="Tool body returned before auxiliary drain")

        tool = ExportAndAbandon()
        app.register_provider(
            Provider(
                [
                    [
                        ModelStreamEvent.tool_call(id="origin", name=tool.spec.name, arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="model"), tools=(tool,))

        async def consume():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=case.session_id + "-caller",
                        agent_name="assistant",
                        messages=[Message.text("user", "export")],
                        invocation_origin=InvocationOriginClaim(subject=CONTEXT.principal),
                    )
                )
            ]

        run = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(checked.wait(), 10)
            assert cleaning.is_set() and not run.done()
            assert projector.calls == 1
            assert len(await published(store, case.session_id)) == 1
            # End the deliberately abandoned-inference run with an explicit
            # caller cancellation; the admitted export has already settled.
            run.cancel()
            release_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(run), 10)
            assert run.cancelled() and run.cancelling() == 1
        finally:
            release_cleanup.set()
            if projector.release is not None:
                projector.release.set()
            if not run.done():
                run.cancel()
            await asyncio.gather(run, *children, return_exceptions=True)
