from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack

import pytest
from pydantic import SecretStr
from tests.core.test_workspace_mutation_receipts import _portable_environment_spec

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    IncompleteSessionRecoveryRequest,
    Message,
    RecoveryBlockerCode,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
    ResumeRequest,
    RunRequest,
    StaticToolExposurePolicy,
    TargetedToolGrant,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolResult,
    ToolSpec,
)
from cayu._exception_groups import exception_cause
from cayu.core import ToolResultPart
from cayu.environments import DeterministicWorkspaceBinding, Environment
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runtime import InMemorySessionStore, UserInputResponse
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.workspace_observation_recovery import workspace_observations_from_checkpoint
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.user_input import UserInputTool
from cayu.vaults import SecretRedactor
from cayu.workspaces import LocalWorkspace


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("pause", ["ordinary", "approval", "user_input", "gateway", "workspace"])
def test_cancelled_preparation_recovers_without_dispatch(backend, pause, tmp_path):
    async def scenario():
        committed = asyncio.Event()
        release = asyncio.Event()
        invocations = []
        model_results = []
        effect_key = None
        if pause == "workspace":
            (tmp_path / "workspace").mkdir()
        codec = PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(
                active_key_id="test",
                keys={"test": SecretStr("A" * 43)},
            )
        )

        class Barrier:
            invocation_lifecycle_command_version = 1

            async def publish_session_operation(self, session_id, **kwargs):
                nonlocal effect_key
                result = await super().publish_session_operation(session_id, **kwargs)
                key = kwargs["idempotency_key"]
                if key.startswith("tool-effect:v1:"):
                    record = await self.load_session_operation(session_id, key)
                    if record["state"] == "prepared" and not committed.is_set():
                        effect_key = key
                        committed.set()
                        await release.wait()
                return result

        class Memory(Barrier, InMemorySessionStore):
            invocation_lifecycle_command_version = 1

        class SQLite(Barrier, SQLiteSessionStore):
            invocation_lifecycle_command_version = 1

        class External(Tool):
            spec = ToolSpec(
                name="external",
                workspace_mutation=pause == "workspace",
                parallel_safe=pause != "workspace",
            )

            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:prepared-external", behavior_version="1", implementation_version="1"
                )

            async def run(self, ctx, args):
                invocations.append(args)
                return ToolResult(content="must not execute")

        class Provider(ModelProvider):
            name = "preparation-test"

            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:preparation", behavior_version="1", implementation_version="1"
                )

            async def stream(self, request):
                results = [
                    part
                    for message in request.messages
                    for part in message.content
                    if isinstance(part, ToolResultPart)
                ]
                if results:
                    if pause == "gateway":
                        assert [part.tool_name for part in results] == ["call_tool"]
                    model_results.extend(results)
                    yield ModelStreamEvent.text_delta("Recovered.")
                    yield ModelStreamEvent.completed({"finish_reason": "stop"})
                else:
                    if pause == "gateway":
                        context = next(
                            message.content[0].text
                            for message in request.messages
                            if message.role == "user"
                            and message.content[0].text.startswith(
                                "Cayu runtime targeted-tool context"
                            )
                        )
                        [descriptor] = json.loads(context.rsplit("\n", 1)[1])["tools"]
                        yield ModelStreamEvent.tool_call(
                            id="call",
                            name="call_tool",
                            arguments={
                                "tool_ref": descriptor["tool_ref"],
                                "arguments": {"value": "private argument"},
                            },
                        )
                        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                        return
                    yield ModelStreamEvent.tool_call(
                        id="call", name="external", arguments={"value": "private argument"}
                    )
                    if pause == "user_input":
                        yield ModelStreamEvent.tool_call(
                            id="input", name="ask_user", arguments={"question": "Continue?"}
                        )
                    yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

        def application(store):
            app = CayuApp(
                session_store=store,
                enable_logging=False,
                secret_redactor=SecretRedactor("tool_effect_not_dispatched"),
            )
            app.register_provider(Provider(), default=True)
            if pause == "workspace":
                app.register_environment(
                    Environment(
                        _portable_environment_spec("local"),
                        workspace=LocalWorkspace(tmp_path / "workspace", workspace_id="prepared"),
                        binding=DeterministicWorkspaceBinding(),
                    ),
                    default=True,
                )
            app.register_agent(
                AgentSpec(name="agent", model="test"),
                tools=[External(), UserInputTool()] if pause == "user_input" else [External()],
                tool_policy=AlwaysRequireApprovalToolPolicy(tools=["external"])
                if pause == "approval"
                else None,
                **(
                    {
                        "targeted_tool_mode": "call_tool",
                        "tool_exposure_policy": StaticToolExposurePolicy(
                            profile_id="targeted-only", tools=()
                        ),
                    }
                    if pause == "gateway"
                    else {}
                ),
            )
            return app

        async with AsyncExitStack() as stack:
            store = (
                Memory()
                if backend == "memory"
                else SQLite(tmp_path / "prepared.sqlite", public_authority_alias_codec=codec)
            )
            if backend == "sqlite":
                stack.push_async_callback(store.close)
            app = application(store)
            approval_request = None
            input_response = None

            async def run():
                nonlocal approval_request, input_response
                events = [
                    e
                    async for e in app.run(
                        RunRequest(
                            session_id="prepared",
                            agent_name="agent",
                            messages=[Message.text("user", "go")],
                            tool_grants=(
                                TargetedToolGrant(
                                    request_id="prepared-grant",
                                    tool_id="cayu:external",
                                    max_calls=1,
                                ),
                            )
                            if pause == "gateway"
                            else (),
                        )
                    )
                ]
                if pause == "user_input":
                    awaiting = next(
                        e for e in events if e.type.value == "session.awaiting_user_input"
                    )
                    input_response = UserInputResponse(
                        session_id="prepared", input_id=awaiting.payload["input_id"], answer="yes"
                    )
                    _ = [e async for e in app.resolve_user_input(input_response)]
                elif pause == "approval":
                    approval = next(
                        e for e in events if e.type.value == "tool.call.approval_requested"
                    )
                    approval_request = ToolApprovalRequest(
                        session_id="prepared",
                        approval_id=approval.payload["approval"]["approval_id"],
                        tool_round_id=approval.payload["tool_round_id"],
                        tool_call_id=approval.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                    _ = [e async for e in app.resolve_tool_approval(approval_request)]

            task = asyncio.create_task(run())
            try:
                await asyncio.wait_for(committed.wait(), 15)
                prior = await store.load_session_operation("prepared", effect_key)
                assert prior["state"] == "prepared" and prior["dispatch_id"] is None
                task.cancel()
                assert task.cancelling() == 1
                release.set()
                with pytest.raises(asyncio.CancelledError) as cancelled:
                    await task
                assert task.cancelled() and task.cancelling() == 1
                if pause == "workspace":
                    assert exception_cause(cancelled.value) is None
                    observations = workspace_observations_from_checkpoint(
                        await store.load_checkpoint("prepared")
                    )
                    assert len(observations) == 1
                    [observation] = observations.values()
                    assert observation.tool_outcome_event_id is None
                    assert not any(
                        event.type.value == "tool.effect.outcome_unknown"
                        for event in await store.load_events("prepared")
                    )
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            assert invocations == []
            assert await store.load_session_operation("prepared", effect_key) == prior
            if backend == "sqlite":
                await store.close()
                store = SQLite(tmp_path / "prepared.sqlite", public_authority_alias_codec=codec)
                stack.push_async_callback(store.close)
            app = application(store)
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="prepared",
                    inactive_for_seconds=0,
                )
            )
            selected = await store.load_session_operation("prepared", effect_key)
            assert selected["state"] == "failed"
            assert selected["dispatch_id"] is None and selected["terminal"]["receipt"] is None
            assert selected["intent"] == prior["intent"]
            if pause in {"ordinary", "gateway", "workspace"}:
                before_plan = await store.load_events("prepared")
                plan = await app.plan_recovery(
                    RecoveryPlanRequest(selection=RecoveryPlanSelection(session_ids=("prepared",)))
                )
                assert plan.items[0].allowed_actions == (RecoveryPlanAction.LEAVE_INTACT,)
                assert RecoveryBlockerCode.TOOL_EFFECT_CONTINUATION_REQUIRED in {
                    blocker.code for blocker in plan.items[0].blockers
                }
                assert await store.load_events("prepared") == before_plan
                assert await store.load_session_operation("prepared", effect_key) == selected
            events = [
                e
                async for e in (
                    app.resolve_user_input(input_response)
                    if input_response is not None
                    else app.resolve_tool_approval(approval_request)
                    if approval_request is not None
                    else app.resume(
                        ResumeRequest(
                            session_id="prepared", messages=[Message.text("user", "continue")]
                        )
                    )
                )
            ]
            assert events[-1].type.value == "session.completed"
            assert invocations == [] and len(model_results) == (2 if pause == "user_input" else 1)
            if pause == "workspace":
                assert not workspace_observations_from_checkpoint(
                    await store.load_checkpoint("prepared")
                )
            terminals = [
                e
                for e in await store.load_events("prepared")
                if e.type.value in {"tool.call.failed", "tool.call.completed"}
                and e.tool_name == "external"
            ]
            assert len(terminals) == 1 and terminals[0].id == selected["terminal"]["event_id"]
            assert terminals[0].payload["result"]["structured"]["executed"] is False
            # Diagnostic redaction must not erase the boolean non-dispatch proof
            # or expose the private arguments in the recovered terminal.
            assert "private argument" not in str(terminals[0].payload)
            assert "tool_effect_not_dispatched" not in str(terminals[0].payload)
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="prepared",
                    inactive_for_seconds=0,
                )
            )
            assert await store.load_session_operation("prepared", effect_key) == selected
            assert [
                e
                for e in await store.load_events("prepared")
                if e.type.value in {"tool.call.failed", "tool.call.completed"}
                and e.tool_name == "external"
            ] == terminals
            owner = ToolEffectStateOwner(store)
            assert (
                await owner.resolve_call(
                    await store.load("prepared"),
                    tool_round_id=prior["intent"]["tool_round_id"],
                    tool_call_id="call",
                )
            ).revision == 1

    asyncio.run(scenario())
