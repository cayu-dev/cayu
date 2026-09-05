"""Public-API process fixture; producer exits inside the opaque provider dispatch."""

from __future__ import annotations

import asyncio
import json
import os
import sys

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    ModelCompletionManualRecoveryRequest,
    ModelProvider,
    ModelStreamEvent,
    RunRequest,
    SessionRunFenced,
    SQLiteSessionStore,
    Tool,
    ToolResult,
    ToolSpec,
)


async def main():
    if sys.argv[2].startswith("postgresql://"):
        from cayu import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        store = PostgresSessionStore(sys.argv[2], schema_mode=SchemaMode.MIGRATE)
    else:
        store = SQLiteSessionStore(sys.argv[2])
    app = CayuApp(session_store=store, enable_logging=False)
    if sys.argv[1] == "produce":

        class OpaqueTool(Tool):
            spec = ToolSpec(
                name="unused",
                description="Process-local callable",
                input_schema={"type": "object", "properties": {}},
            )

            async def run(self, ctx, args):
                raise AssertionError("Recovery must not execute this tool.")
                return ToolResult(content="unreachable")

        class OpaqueProvider(ModelProvider):
            name = "opaque"

            async def stream(self, request):
                session = await store.load("opaque-model")
                active = await store.load_active_model_completion_stage(session.id)
                print(
                    ModelCompletionManualRecoveryRequest(
                        session_id=session.id,
                        expected_session_instance_id=session.instance_id,
                        expected_run_epoch=session.run_epoch,
                        stage_id=active.stage.stage_id,
                        terminal_status="failed",
                        terminalization_only=True,
                        inactive_for_seconds=0,
                    ).model_dump_json(),
                    flush=True,
                )
                # Deliberately omit Runtime/store cleanup with a live dispatched stage.
                os._exit(0)
                yield ModelStreamEvent.completed({"finish_reason": "stop"})

        app.register_provider(OpaqueProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="opaque"), tools=[OpaqueTool()])
        async for _event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="opaque-model",
                messages=[Message.text("user", "execute")],
            )
        ):
            pass
        raise AssertionError("Producer did not reach a dispatched model stage.")
    try:
        request = ModelCompletionManualRecoveryRequest.model_validate_json(sys.argv[3])
        result = await app.recover_model_completion_stage(request)
        print(
            json.dumps({"status": result.session.status, "replayed": result.replayed}), flush=True
        )
    except SessionRunFenced:
        print(json.dumps({"status": "fenced"}), flush=True)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
