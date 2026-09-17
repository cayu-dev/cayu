"""Real run/tool entrance derives caller provenance without borrowing source identity."""

import asyncio
import threading
from contextlib import asynccontextmanager

import pytest
from tests.core.test_session_export_content_release import run_async
from tests.core.test_session_exports import CONTEXT, Policy, harness, published
from tests.core.test_session_exports import backend as backend

from cayu.agents import AgentSpec
from cayu.collaboration.exports import SessionExportDenied
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import RunRequest
from cayu.sessions.invocation import InvocationOriginClaim
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.vaults.redaction import SecretRedactor


class RuntimePolicy(Policy):
    def __init__(self):
        super().__init__()
        self.origins = []

    @asynccontextmanager
    async def acquire_runtime(self, context, *, origin, **kwargs):
        if origin.invocation.origin.subject != context.principal:
            raise SessionExportDenied()
        self.origins.append(origin)
        async with self.acquire(context, **kwargs) as authorization:
            yield authorization


class ExportTool(Tool):
    spec = ToolSpec(
        name="export_checked",
        description="Export reviewed source.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        effect=ToolEffect.EXTERNAL,
    )

    def __init__(self, app, request, *, denied=False):
        self.app = app
        self.request = request
        self.context = None
        self.receipt = None
        self.denied = denied

    async def run(self, ctx, args):
        self.context = ctx
        for copied in (ctx.model_copy(), ToolContext(session_id=ctx.session_id)):
            with pytest.raises(SessionExportDenied):
                await self.app.export_session(self.request, context=CONTEXT, invocation=copied)
        if self.denied:
            with pytest.raises(SessionExportDenied):
                await self.app.export_session(self.request, context=CONTEXT, invocation=ctx)
            return ToolResult(content="Export denied.")
        self.receipt = await self.app.export_session(self.request, context=CONTEXT, invocation=ctx)
        assert (
            await self.app.export_session(self.request, context=CONTEXT, invocation=ctx)
            == self.receipt
        )
        return ToolResult(content="Export retained.")


@pytest.mark.parametrize("retained_work", [False, True])
@run_async
async def test_completed_tool_cannot_admit_during_next_tool(backend, retained_work):
    async with harness(backend) as case:
        app, store, _, projector = case.app(policy=RuntimePolicy())
        await case.create(store)
        first = await case.request(app, "first")
        second = await case.request(app, "second")
        late_start = asyncio.Event()

        class TwoCalls(Tool):
            spec = ExportTool.spec
            previous = None
            receipt = None
            late = None

            async def run(self, ctx, args):
                if self.previous is None:
                    self.previous = ctx

                    async def late_export():
                        await late_start.wait()
                        with pytest.raises(SessionExportDenied):
                            await app.export_session(second, context=CONTEXT, invocation=ctx)

                    self.late = asyncio.create_task(late_export())
                    if retained_work:
                        projector.release = threading.Event()
                        observer = asyncio.create_task(
                            app.export_session(first, context=CONTEXT, invocation=ctx)
                        )
                        assert await asyncio.to_thread(projector.entered.wait, 5)
                        observer.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await observer
                        assert observer.cancelled() and observer.cancelling() == 1
                    return ToolResult(content="First call finished.")
                with pytest.raises(SessionExportDenied):
                    await app.export_session(second, context=CONTEXT, invocation=self.previous)
                late_start.set()
                await self.late
                if retained_work:
                    projector.release.set()
                    await asyncio.gather(*app._session_export_coordinator.owners.pending)
                    prior = await app.lookup_session_export(first, context=CONTEXT)
                    assert prior.receipt.expected.intent.authorization.runtime.tool_call_id == "a"
                self.receipt = await app.export_session(second, context=CONTEXT, invocation=ctx)
                return ToolResult(content="Second call finished.")

        tool = TwoCalls()
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(id=call, name=tool.spec.name, arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ]
                    for call in ("a", "b")
                ]
                + [
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="scripted-model"), tools=(tool,))
        events = [
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
        assert any(event.type is EventType.SESSION_COMPLETED for event in events)
        assert tool.receipt is not None
        assert tool.receipt.expected.intent.authorization.runtime.tool_call_id == "b"
        assert projector.calls == (2 if retained_work else 1)


@pytest.mark.parametrize("case_kind", ["authorized", "wrong_principal", "host_only_policy"])
@run_async
async def test_real_runtime_export_and_historical_readback(backend, case_kind):
    async with harness(backend) as case:
        policy = Policy() if case_kind == "host_only_policy" else RuntimePolicy()
        app, store, _, projector = case.app(policy=policy)
        await case.create(store)
        request = await case.request(app)
        denied = case_kind != "authorized"
        tool = ExportTool(app, request, denied=denied)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="export-call", name=tool.spec.name, arguments={}
                        ),
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
        app.register_agent(AgentSpec(name="assistant", model="scripted-model"), tools=(tool,))
        caller_id = case.session_id + "-caller"
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id=caller_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "export")],
                    invocation_origin=InvocationOriginClaim(
                        subject="mallory" if case_kind == "wrong_principal" else CONTEXT.principal
                    ),
                )
            )
        ]
        assert any(event.type is EventType.SESSION_COMPLETED for event in events), [
            (event.type, event.payload) for event in events[-2:]
        ]
        if denied:
            assert tool.receipt is None
            assert projector.calls == 0
            assert not await published(store, case.session_id)
            return
        assert tool.receipt is not None
        caller = await store.load(caller_id)
        origin = tool.receipt.expected.intent.authorization.runtime
        assert origin.session_id == caller_id != request.ref.session_id
        assert origin.session_instance_id == caller.instance_id
        assert origin.invocation == caller.invocation
        assert origin.tool_call_id == "export-call"
        assert tool.receipt.expected.initiator.invocation_id == caller.invocation.root_invocation_id
        assert projector.calls == 1
        assert len(await published(store, case.session_id)) == 1
        with pytest.raises(SessionExportDenied):
            await app.export_session(request, context=CONTEXT, invocation=tool.context)
        # Authorized readback recovers historical provenance from the source owner;
        # it does not reconstruct a live ToolContext or rerun projection.
        reopened, _, _, _ = case.app(projectors=())
        result = await reopened.lookup_session_export(request, context=CONTEXT)
        assert result.receipt == tool.receipt


def test_runtime_provenance_controls_are_exact_and_secret_safe():
    from typing import get_args

    from cayu.collaboration._contracts import CollaborationContractError
    from cayu.collaboration._preparation import prepare_contract
    from cayu.collaboration.exports import SessionExportRuntimeOrigin
    from cayu.sessions.invocation import InvocationOriginTrust, SessionExecutionSource

    fields = SessionExportRuntimeOrigin.model_fields
    assert set(get_args(fields["invocation_trust"].annotation)) == {
        item.value for item in InvocationOriginTrust
    }
    assert set(get_args(fields["invocation_source"].annotation)) == {
        item.value for item in SessionExecutionSource
    }
    material = dict(
        session_id="requester",
        session_instance_id="instance",
        run_epoch=1,
        invocation_trust="host_asserted",
        invocation_subject="alice",
        root_invocation_id="a9e577cd-385f-4e22-9f9c-87de7a3171dc",
        root_session_id="requester",
        invocation_source="sdk_run",
        interaction_id="interaction",
        model_step_id="step",
        model_attempt_id="attempt",
        tool_round_id="round",
        tool_call_id="call",
        tool_name="export_checked",
        idempotency_key="key",
        effective_arguments_sha256="a" * 64,
        execution_profile_fingerprint="b" * 64,
    )
    for secret in ("host_asserted", "sdk_run", "invocation_trust"):
        redactor = SecretRedactor((secret,))
        origin = prepare_contract(SessionExportRuntimeOrigin, material, redactor=redactor)
        assert (
            prepare_contract(
                SessionExportRuntimeOrigin, origin.model_dump(mode="json"), redactor=redactor
            )
            == origin
        )
        with pytest.raises(CollaborationContractError):
            prepare_contract(
                SessionExportRuntimeOrigin,
                {**material, "invocation_subject": secret},
                redactor=redactor,
            )
