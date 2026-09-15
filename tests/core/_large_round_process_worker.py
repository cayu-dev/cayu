"""Separate-process native execution for large-round recovery controls."""

import asyncio
import json
import sys
from pathlib import Path

from tests.core._workload_secret_support import FakeProvider, collect_events

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import EventType
from cayu.messages import Message, ToolResultPart
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.base import IncompleteSessionRecoveryRequest, ResumeRequest, RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec


class Echo(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo a synthetic value",
        parallel_safe=False,
        effect=ToolEffect.IDEMPOTENT,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:large-round:process-echo",
            behavior_version="1",
            implementation_version="1",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "hold": {"type": "boolean"},
            },
            "required": ["value", "hold"],
        },
    )

    def __init__(self, root, store):
        self.root, self.store = root, store

    async def run(self, ctx, args):
        execution_log = self.root / "executions.jsonl"
        prior = execution_log.read_text().splitlines() if execution_log.exists() else []
        assert args["value"] not in prior, "Recovery must not re-execute a dispatched call"
        with execution_log.open("a") as stream:
            stream.write(args["value"] + "\n")
        if args["hold"]:
            events = await self.store.load_events("process-large-round")
            assert any(
                e.type == EventType.TOOL_CALL_COMPLETED
                and e.payload.get("tool_call_id") == "call-0"
                and e.payload["result"]["content"] == "0"
                for e in events
            )
            (self.root / "ready.json").write_text(json.dumps({"first_result_durable": True}))
            await asyncio.Event().wait()
        return ToolResult(content=args["value"])


async def run(mode, root):
    store = SQLiteSessionStore(root / "session.sqlite3")
    app = CayuApp(session_store=store, enable_logging=False)
    final = [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]
    batches = (
        [final]
        if mode == "resume"
        else [
            [
                *[
                    ModelStreamEvent.tool_call(
                        id=f"call-{i}",
                        name="echo",
                        arguments={
                            "value": str(i),
                            "hold": mode == "prepare" and i == 1,
                        },
                    )
                    for i in range(257)
                ],
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            final,
        ]
    )
    app.register_provider(FakeProvider(batches), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[Echo(root, store)])
    try:
        if mode == "resume":
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="process-large-round",
                    reason="synthetic worker process exited",
                )
            )
            events = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="process-large-round",
                        messages=[Message.text("user", "Continue from durable work.")],
                    )
                )
            ]
        else:
            events = await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="process-large-round",
                    messages=[Message.text("user", "Echo the synthetic values.")],
                ),
            )
        assert events[-1].type == EventType.SESSION_COMPLETED
        retained = await store.load_events("process-large-round")
        terminal = [
            e
            for e in retained
            if e.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        ]
        assert len(terminal) == 257
        assert {e.payload["tool_call_id"] for e in terminal} == {f"call-{i}" for i in range(257)}
        assert any(
            e.type == EventType.TOOL_CALL_COMPLETED and e.payload["result"]["content"] == "0"
            for e in terminal
        )
        if mode == "normal":
            assert all(e.type == EventType.TOOL_CALL_COMPLETED for e in terminal)
        assert "pending_tool_round" not in await store.load_checkpoint("process-large-round")
        transcript = await store.load_transcript("process-large-round")
        parts = [
            p for message in transcript for p in message.content if isinstance(p, ToolResultPart)
        ]
        assert len(parts) == 257
        (root / "result.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "terminals": len(terminal),
                    "completed": sum(e.type == EventType.TOOL_CALL_COMPLETED for e in terminal),
                    "first_execution_count": (root / "executions.jsonl")
                    .read_text()
                    .splitlines()
                    .count("0"),
                }
            )
        )
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1], Path(sys.argv[2])))
