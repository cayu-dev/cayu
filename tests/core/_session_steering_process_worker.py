"""Independent runtime/control processes for cooperative steering acceptance."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    EnqueueSessionMessageRequest,
    ExecutionProfileBehaviorIdentity,
    IncompleteSessionRecoveryRequest,
    Message,
    ModelProvider,
    ModelStreamEvent,
    PostgresSessionStore,
    ResumeRequest,
    RunRequest,
    SessionStatus,
    SQLiteSessionStore,
    StopAfterCurrentToolRoundRequest,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint

SESSION_ID = "process-safe-steering"
CORRECTION = "focus on Y"


def _identity(name: str) -> ExecutionProfileBehaviorIdentity:
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version="1", implementation_version="1"
    )


class LoopProvider(ModelProvider):
    name = "steering-process-provider"

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @property
    def execution_profile_identity(self):
        return _identity("steering-process-provider")

    async def stream(self, request):
        with (self.directory / "provider-calls").open("a") as journal:
            journal.write("dispatch\n")
        if any(
            getattr(part, "text", None) == CORRECTION
            for message in request.messages
            for part in message.content
        ):
            yield ModelStreamEvent.text_delta("corrected")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
        else:
            yield ModelStreamEvent.tool_call(id="current-tool", name="barrier", arguments={})
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class BarrierTool(Tool):
    spec = ToolSpec(
        name="barrier",
        description="A real dispatched tool held behind an external barrier.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        execution_profile_identity=_identity("steering-process-barrier"),
    )

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory

    async def run(self, ctx, args):
        (self.directory / "tool-started").touch()
        async with asyncio.timeout(60):
            while not (self.directory / "release-tool").exists():
                await asyncio.sleep(0.02)
        with (self.directory / "tool-effects").open("a") as journal:
            journal.write("effect\n")
        return ToolResult(content="tool settled")


def open_store(backend: str, directory: Path, *, hold_promotion: bool = False):
    base = SQLiteSessionStore if backend == "sqlite" else PostgresSessionStore
    selected = base
    if hold_promotion:

        class HeldPromotionStore(base):
            invocation_lifecycle_command_version = 1
            terminal_interaction_publication_version = 1

            async def transition_status_and_checkpoint(self, *args, **kwargs):
                result = await super().transition_status_and_checkpoint(*args, **kwargs)
                if kwargs.get("to_status") is SessionStatus.INTERRUPTING:
                    (directory / "promotion-committed").touch()
                    await asyncio.Event().wait()
                return result

        selected = HeldPromotionStore
    if backend == "sqlite":
        return selected(directory / "sessions.db")
    return selected(os.environ["CAYU_STEERING_TEST_DSN"], min_size=1, max_size=2)


async def execute(mode: str, backend: str, directory: Path) -> None:
    store = open_store(backend, directory, hold_promotion=mode == "run-crash")
    app = CayuApp(session_store=store, enable_logging=False)
    try:
        if mode in {"accept", "accept-crash"}:
            if mode == "accept-crash":
                original_publish = store.publish_session_operation

                async def publish_then_wait(*args, **kwargs):
                    result = await original_publish(*args, **kwargs)
                    if kwargs["idempotency_key"].startswith("cayu.session-steering.v1:"):
                        (directory / "acceptance-committed").touch()
                        await asyncio.Event().wait()
                    return result

                store.publish_session_operation = publish_then_wait
            session = await store.load(SESSION_ID)
            assert session is not None
            profile = active_invocation_execution_profile_from_checkpoint(
                await store.load_checkpoint(SESSION_ID)
            )
            assert profile is not None
            await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=SESSION_ID,
                    idempotency_key="correction",
                    content=CORRECTION,
                    delivery_mode="next_turn",
                )
            )
            request = StopAfterCurrentToolRoundRequest(
                session_id=SESSION_ID,
                session_instance_id=session.instance_id,
                interaction_id=profile.interaction_id,
                expected_run_epoch=session.run_epoch,
                idempotency_key="stop-current-round",
            )
            receipt = await app.stop_after_current_tool_round(request)
            (directory / "acceptance.json").write_text(receipt.model_dump_json())
            return
        app.register_provider(LoopProvider(directory))
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"), tools=[BarrierTool(directory)]
        )
        if mode == "recover-interruption":
            # The parent has killed and reaped the previous owner. This is
            # positive process-loss evidence, not a timeout-based takeover.
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=SESSION_ID, inactive_for_seconds=0)
            )
            return
        stream = (
            app.run(
                RunRequest(
                    session_id=SESSION_ID,
                    agent_name="assistant",
                    messages=[Message.text("user", "keep investigating")],
                    max_steps=64,
                )
            )
            if mode in {"run", "run-crash"}
            else app.resume(
                ResumeRequest(
                    session_id=SESSION_ID,
                    messages=[Message.text("user", "continue")],
                    max_steps=64,
                )
            )
        )
        async for _ in stream:
            pass
        session = await store.load(SESSION_ID)
        assert session is not None
        print(json.dumps({"status": session.status.value}), flush=True)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(execute(sys.argv[1], sys.argv[2], Path(sys.argv[3])))
