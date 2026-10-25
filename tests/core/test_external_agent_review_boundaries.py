from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from tests.core.test_external_agents import ScriptedAdapter, request

from cayu import CayuApp, EventType
from cayu.approvals import PendingToolApprovalEventView
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineExceeded
from cayu.external_agents import (
    ExternalAgentError,
    ExternalAgentEvent,
    ExternalCleanupIncomplete,
    run_external_agent,
)
from cayu.external_agents.base import ExternalAgentBudgetExceeded, ExternalAgentContext
from cayu.external_agents.recording import ExternalAgentRecorder
from cayu.external_agents.server import ExternalAgentGateway, ExternalModelRoute
from cayu.external_agents.tools import CayuToolBridge, ExternalPermissionTool
from cayu.resource_access import ResourceAccessDenied
from cayu.sessions import SessionStatus
from cayu.tasks import InMemoryTaskStore, TaskQuery, TaskStatus
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy


@pytest.mark.parametrize(
    "error", [ExternalAgentBudgetExceeded, ResourceAccessDenied, ExternalAgentError]
)
def test_live_approval_blocks_when_parent_admission_check_raises(error, monkeypatch):
    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        deny = False
        original = ExternalAgentRecorder.check_budget

        async def check(recorder):
            if deny:
                raise error()
            await original(recorder)

        monkeypatch.setattr(ExternalAgentRecorder, "check_budget", check)
        executed = []

        class Effect(ExternalPermissionTool):
            async def run(self, ctx, args):
                executed.append(args)
                return await super().run(ctx, args)

        bridge = CayuToolBridge(
            app, name="bridge", tools=[Effect("effect")], policy=AlwaysRequireApprovalToolPolicy()
        )

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                result = await bridge.call(
                    "effect",
                    {},
                    parent_session_id=context.session_id,
                    execution_deadline=context.execution_deadline,
                    call_id="effect",
                )
                assert result.is_error
                yield ExternalAgentEvent("result", {"output": "blocked"}, "final")

        running = asyncio.create_task(
            run_external_agent(app, request(), Adapter([]), prompt="test")
        )
        async with asyncio.timeout(5):
            pending = None
            while pending is None:
                for child_id in tuple(bridge.active):
                    if await app.session_store.load(child_id) is None:
                        continue
                    pending = next(
                        (
                            event
                            for event in await app.session_store.load_events(child_id)
                            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                        ),
                        None,
                    )
                if pending is None:
                    await asyncio.sleep(0.01)
        approval = PendingToolApprovalEventView.from_event(pending)
        deny = True
        events = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=child_id,
                    approval_id=approval.approval_id,
                    tool_round_id=approval.tool_round_id,
                    tool_call_id=approval.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        assert any(event.type is EventType.TOOL_CALL_BLOCKED for event in events)
        assert not executed
        with pytest.raises(ExternalAgentError):
            await running
        assert not bridge.invocations
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("second_method", ["model", "tools/call"])
def test_concurrent_gateway_request_waits_for_successful_usage_publication(second_method):
    async def scenario():
        publishing, release = asyncio.Event(), asyncio.Event()
        receipts, calls = [], []

        async def record(event):
            receipts.append(event)
            if len(receipts) == 1:
                publishing.set()
                await release.wait()

        class Tools:
            async def close(self):
                pass

            async def dispatch(self, method, params):
                calls.append(method)
                return {"content": []}

        context = ExternalAgentContext(
            "run", "test", ExecutionDeadline.after(5), record_model_usage=record
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200, json={"usage": {"input_tokens": 1, "output_tokens": 1}}
                )
            )
        ) as upstream:
            gateway = ExternalAgentGateway(
                client=upstream,
                public_base_url="http://gateway",
                routes={"model": ExternalModelRoute("https://model.test", {}, ("/responses",))},
            )
            surface = FastAPI()
            surface.include_router(gateway.router)
            async with (
                httpx.AsyncClient(transport=httpx.ASGITransport(app=surface)) as guest,
                gateway.grant(
                    context, route="model", models=frozenset({"test"}), tools=Tools()
                ) as grant,
            ):
                headers = {"Authorization": f"Bearer {grant.token}"}
                first = asyncio.create_task(
                    guest.post(
                        grant.model_base_url + "/responses", headers=headers, json={"model": "test"}
                    )
                )
                await publishing.wait()
                if second_method == "model":
                    second = asyncio.create_task(
                        guest.post(
                            grant.model_base_url + "/responses",
                            headers=headers,
                            json={"model": "test"},
                        )
                    )
                else:
                    second = asyncio.create_task(
                        guest.post(
                            grant.mcp_url,
                            headers=headers,
                            json={
                                "jsonrpc": "2.0",
                                "id": 1,
                                "method": second_method,
                                "params": {"name": "effect"},
                            },
                        )
                    )
                await asyncio.sleep(0.02)
                assert not second.done()
                release.set()
                assert (await first).status_code == (await second).status_code == 200
        assert len(receipts) == (2 if second_method == "model" else 1)
        assert calls == ([] if second_method == "model" else [second_method])

    asyncio.run(scenario())


@pytest.mark.parametrize("deadline", [False, True])
def test_cancellation_during_cleanup_retries_settles_after_quiescence(deadline):
    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        retrying, release = asyncio.Event(), asyncio.Event()

        async def retry():
            retrying.set()
            await release.wait()

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                raise ExternalCleanupIncomplete(retry)
                yield

        req = request().model_copy(
            update={"execution_deadline": ExecutionDeadline.after(0.3 if deadline else 5)}
        )
        running = asyncio.create_task(run_external_agent(app, req, Adapter([]), prompt="test"))
        await asyncio.wait_for(retrying.wait(), 5)
        if deadline:
            await asyncio.sleep(max(0, req.execution_deadline.remaining_seconds() or 0) + 0.02)
        else:
            running.cancel("operator")
        await asyncio.sleep(0.02)
        (task,) = await app.task_store.list_tasks(TaskQuery())
        assert task.status is TaskStatus.RUNNING
        assert app._session_control.execution_presence.groups
        assert not running.done()
        release.set()
        with pytest.raises(ExecutionDeadlineExceeded if deadline else asyncio.CancelledError):
            await running
        current = await app.task_store.load_task(task.id)
        assert current.status is (TaskStatus.FAILED if deadline else TaskStatus.CANCELLED)
        session = await app.session_store.load(task.session_id)
        assert session.status is (SessionStatus.FAILED if deadline else SessionStatus.INTERRUPTED)
        assert not app._session_control.execution_presence.groups
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("write_fails", [False, True])
def test_sdk_observation_waits_for_inflight_gateway_usage_write(monkeypatch, write_fails):
    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        publishing, release, observing, closing = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )
        original_emit = app.emit_event

        async def emit(event):
            if event.type is EventType.MODEL_COMPLETED:
                publishing.set()
                await release.wait()
                if write_fails:
                    raise OSError("usage publication failed")
            return await original_emit(event)

        monkeypatch.setattr(app, "emit_event", emit)

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                context.claim_model_usage()
                pending = asyncio.create_task(
                    context.record_model_usage(
                        ExternalAgentEvent(
                            "usage", {"input_tokens": 1, "output_tokens": 1}, "gateway:one"
                        )
                    )
                )
                try:
                    await publishing.wait()
                    observing.set()
                    yield ExternalAgentEvent("text", {"text": "tool progress"}, "progress")
                    await pending
                    yield ExternalAgentEvent("result", {"output": "done"}, "result")
                finally:
                    closing.set()
                    await asyncio.gather(pending, return_exceptions=True)

        running = asyncio.create_task(
            run_external_agent(app, request(), Adapter([]), prompt="test")
        )
        try:
            await asyncio.wait_for(observing.wait(), 5)
            await asyncio.sleep(0.02)
            assert not closing.is_set()
            release.set()
            if write_fails:
                with pytest.raises(ExternalAgentError):
                    await asyncio.wait_for(running, 5)
            else:
                result = await asyncio.wait_for(running, 5)
                assert result.output == "done"
                events = await app.session_store.load_events(result.session_id)
                assert sum(event.type is EventType.MODEL_COMPLETED for event in events) == 1
        finally:
            release.set()
            await asyncio.gather(running, return_exceptions=True)
            await app.aclose()

    asyncio.run(scenario())
