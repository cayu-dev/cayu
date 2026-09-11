"""Crash barriers around real foreground execution, using the shared worker owner."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from worker_harness import _append_json_line, _public_authority_alias_codec, _write_json_atomic

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    IncompleteSessionRecoveryRequest,
    Message,
    RecoveryDecision,
    RecoveryExecutionRequest,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
    ResumeRequest,
    RunRequest,
    SQLiteSessionStore,
    SubagentSpec,
    SubagentTool,
)
from cayu.core import ToolResultPart
from cayu.core.runtime_authority import SessionRunFenced
from cayu.providers import ModelProvider, ModelStreamEvent


async def run_foreground_subagent_worker(config):
    starting = config["action"] == "start"
    phase = config["crash_phase"] if starting else None
    now = datetime.now(UTC) + timedelta(seconds=0 if starting else 600)
    parent_id = config["session_id"]
    child_id = config.get("child_session_id")
    live_result = None
    late_return = asyncio.Event()

    async def pause():
        _write_json_atomic(
            Path(config["phase_path"]),
            {"phase": phase, "child_session_id": child_id, "live_result": live_result},
        )
        if config.get("stale_worker"):
            # Simulate a dispatched remote operation that cannot be aborted.
            # The harness owns this worker and explicitly releases it after a
            # different process has acquired and settled the expired run.
            while not Path(config["control_path"]).exists():
                try:
                    await asyncio.sleep(0.02)
                except asyncio.CancelledError:
                    continue
            _append_json_line(Path(config["marker_path"]), {"late_child_return": True})
            late_return.set()
        else:
            await asyncio.Future()

    class Provider(ModelProvider):
        name = "foreground-process-recovery"

        @property
        def execution_profile_identity(self):
            return ExecutionProfileBehaviorIdentity(
                name="tests:foreground-process-provider",
                behavior_version="1",
                implementation_version="1",
            )

        async def stream(self, request):
            is_child = request.model == "child-model"
            has_result = any(
                isinstance(part, ToolResultPart)
                for message in request.messages
                for part in message.content
            )
            _append_json_line(
                Path(config["marker_path"]),
                {"child": is_child, "continuation": has_result},
            )
            if not starting:
                assert not is_child and has_result, "Recovery redispatched unfinished work"
            if is_child:
                if phase == "during_child":
                    await pause()
                yield ModelStreamEvent.text_delta("Review 😀é日本語 " * 4)
            elif has_result:
                yield ModelStreamEvent.text_delta("Parent continued")
            else:
                yield ModelStreamEvent.tool_call(
                    id="spawn", name="subagent", arguments={"agent": "child", "task": "review"}
                )
            yield ModelStreamEvent.completed(
                {"finish_reason": "stop" if is_child or has_result else "tool_calls"}
            )

    class Runtime:
        def __init__(self, app):
            self.app = app
            self.session_store = store

        async def run(self, request):
            nonlocal child_id
            child_id = request.session_id
            async for event in self.app.run(request):
                if phase == "after_child_creation" and event.type.value == "session.started":
                    await pause()
                yield event

        def interrupt_session(self, request):
            return self.app.interrupt_session(request)

        async def _submit_durable_subagent(self, **kwargs):
            raise AssertionError("Foreground recovery must not submit durable tasks")

        async def _reconcile_durable_subagent(self, **kwargs):
            raise AssertionError("Foreground recovery must not reconcile durable tasks")

    class Tool(SubagentTool):
        async def run(self, context, arguments):
            nonlocal live_result
            if phase == "before_child_creation":
                await pause()
            result = await super().run(context, arguments)
            live_result = result.model_dump(mode="json")
            if phase == "after_child_completion":
                await pause()
            return result

    store = SQLiteSessionStore(
        config["backend"]["session_path"],
        ownership_clock=lambda: now,
        public_authority_alias_codec=_public_authority_alias_codec(),
    )
    try:
        app = CayuApp(session_store=store, clock=lambda: now, enable_logging=False)
        app.register_provider(Provider(), default=True)
        app.register_agent(
            AgentSpec(name="parent", model="parent-model"),
            tools=[
                Tool(
                    Runtime(app),
                    agents={"child": SubagentSpec(agent_name="child", result_max_chars=12)},
                    execution_profile_identity=ExecutionProfileBehaviorIdentity(
                        name="tests:foreground-process-tool",
                        behavior_version="1",
                        implementation_version="1",
                    ),
                )
            ],
        )
        app.register_agent(AgentSpec(name="child", model="child-model"))
        if starting:
            try:
                async for event in app.run(
                    RunRequest(
                        session_id=parent_id,
                        agent_name="parent",
                        messages=[Message.text("user", "go")],
                    )
                ):
                    if phase == "after_parent_terminal" and event.type == "tool.call.completed":
                        assert any(
                            stored.type == "tool.call.completed"
                            for stored in await store.load_events(parent_id)
                        )
                        await pause()
            except (SessionRunFenced, asyncio.CancelledError):
                if not config.get("stale_worker"):
                    raise
            if config.get("stale_worker"):
                await asyncio.wait_for(late_return.wait(), timeout=30)
                return {"late_return_observed": True}
            raise AssertionError("Crash barrier was not reached")

        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=parent_id, inactive_for_seconds=0)
        )
        parent = await store.load(parent_id)
        assert parent is not None
        if parent.status.value == "completed":
            return {"status": parent.status.value, "recovery": recovery.model_dump(mode="json")}
        if child_id is not None:
            child = await store.load(child_id)
            assert child is not None
            if child.status.value in {"pending", "running", "interrupting"}:
                assert recovery.pending_subagent_session_ids == (child_id,)
                if config["crash_phase"] == "during_child":
                    # A killed provider call is ambiguous. An explicit operator
                    # decision settles its existing model fence; child recovery
                    # must not silently discard that authority boundary.
                    plan = await app.plan_recovery(
                        RecoveryPlanRequest(
                            selection=RecoveryPlanSelection(session_ids=(child_id,))
                        )
                    )
                    assert len(plan.items) == 1
                    item = plan.items[0]
                    assert RecoveryPlanAction.MODEL_MARK_INTERRUPTED in item.allowed_actions
                    request = RecoveryExecutionRequest(
                        plan=plan,
                        execution_id="foreground-child-interruption",
                        decisions=(
                            RecoveryDecision(
                                item_id=item.item_id,
                                action=RecoveryPlanAction.MODEL_MARK_INTERRUPTED,
                            ),
                        ),
                    )
                    receipt = await app.execute_recovery(request)
                    replay = await app.execute_recovery(request)
                    assert receipt.items[0].status.value == "executed"
                    assert not receipt.items[0].replayed
                    assert replay.items == tuple(
                        item.model_copy(update={"replayed": True}) for item in receipt.items
                    )
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=child_id, inactive_for_seconds=0)
                )
                child = await store.load(child_id)
                assert child is not None and child.status.value == "interrupted"
            async for _ in app.resume(
                ResumeRequest(session_id=parent_id, messages=[Message.text("user", "continue")])
            ):
                pass
        session = await store.load(parent_id)
        assert session is not None
        return {"status": session.status.value, "recovery": recovery.model_dump(mode="json")}
    finally:
        await store.close()
