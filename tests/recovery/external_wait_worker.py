"""Disposable process used to kill a real external-wait execution before parking."""

import asyncio
import json
import sys
from datetime import timedelta

from tests.core.test_external_wait_request_policies import RequestStopPolicy
from tests.core.test_tool_completion import FinalTool, call
from tests.external_wait_support import CONTEXT, Policy

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    ResumeRequest,
    RunRequest,
    StructuredOutputSpec,
    ToolCompletionPolicy,
)
from cayu.evals.testing import ScriptedModelProvider, scripted_structured_output
from cayu.external_waits import ExternalEventWaits
from cayu.providers.base import ModelStreamEvent
from cayu.runtime import _recovery_ownership as recovery_ownership_module
from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions.execution import SessionExecutionConfig
from cayu.sessions.external_waits import ExternalWaitRegistration
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


async def main():
    setup = json.loads(sys.stdin.readline())
    if setup.get("receipt_limit") is not None:
        from cayu.sessions import _invocation_lifecycle

        _invocation_lifecycle.INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS = setup["receipt_limit"]
        recovery_ownership_module._INCOMPLETE_RECOVERY_CLAIM_LEASE = timedelta(seconds=2)
    store = (
        SQLiteSessionStore(setup["database"])
        if setup["backend"] == "sqlite"
        else PostgresSessionStore(setup["database"], min_size=1, max_size=2)
    )
    registered = ExternalWaitRegistration.model_validate_json(setup["registration"])
    waits = ExternalEventWaits(store=store, access_policy=Policy())
    partial_tool = setup.get("boundary") == "tool_publication"
    final_tool = setup.get("boundary") == "final_tool_publication"
    tool = FinalTool()
    provider = ScriptedModelProvider(
        [
            call()
            if final_tool
            else list(scripted_structured_output({"answer": "OK"}, id="answer"))
            if partial_tool
            else [
                ModelStreamEvent.text_delta("Submitted"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    stop_policy = RequestStopPolicy()
    parked_before_release = setup.get("boundary") == "parked_before_release"
    cleanup_boundary = setup.get("boundary") in {"parked_before_release", "before_park_cleanup"}
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        session_execution=SessionExecutionConfig(
            heartbeat_interval_seconds=0.1,
            lease_seconds=5 if cleanup_boundary else 0.6,
        ),
        loop_policies=[stop_policy] if cleanup_boundary or setup.get("repeated_recovery") else [],
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="root", model="model"), tools=[tool] if final_tool else [])

    async def pause(self, invocation):
        assert tool.calls == int(final_tool)
        if cleanup_boundary:
            assert stop_policy.calls == 1
            presence = await store.inspect_session_execution(setup["session_id"])
            assert presence.state == "executing", presence
        print(json.dumps({"ready": True, "requests": len(provider.requests)}), flush=True)
        await asyncio.Event().wait()

    if setup.get("boundary") == "recovery_admission":
        from cayu.runtime import _invocation_lifecycle
        from cayu.sessions.recovery import IncompleteSessionRecoveryRequest

        original_apply = _invocation_lifecycle.apply_invocation_lifecycle_command

        async def committed_rebind(native, command):
            result = await original_apply(native, command)
            if command.kind.value == "rebind":
                await pause(None, None)
            return result

        _invocation_lifecycle.apply_invocation_lifecycle_command = committed_rebind
        recovery_result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=setup["session_id"], inactive_for_seconds=0)
        )
        raise AssertionError(f"Recovery did not reach its durable rebind: {recovery_result!r}")
    if partial_tool or final_tool:
        from cayu.runtime import _tool_round_publication

        async def pause_publication(prepared, **kwargs):
            await pause(None, None)

        _tool_round_publication.publish_tool_round_publication = pause_publication
    elif setup.get("boundary") == "before_ticket":
        _ExternalExecutionToWait.prepare = pause
    elif parked_before_release:
        original_park = _ExternalExecutionToWait.park

        async def park_then_pause(boundary, invocation):
            await original_park(boundary, invocation)
            await pause(boundary, invocation)

        _ExternalExecutionToWait.park = park_then_pause
    else:
        _ExternalExecutionToWait.park = pause
    if setup.get("resume_existing"):
        async for _ in app.run(
            RunRequest(
                agent_name="root",
                session_id=setup["session_id"],
                messages=[Message.text("user", "Ready")],
            )
        ):
            pass
        await SessionExternalWaitAdapter(app, waits).resume_to_wait(
            ResumeRequest(
                session_id=setup["session_id"], messages=[Message.text("user", "Submit job")]
            ),
            registered,
            context=CONTEXT,
        )
        raise AssertionError("Resumed execution did not reach the boundary.")
    await SessionExternalWaitAdapter(app, waits).run_to_wait(
        RunRequest(
            agent_name="root",
            session_id=setup["session_id"],
            messages=[Message.text("user", "Submit job")],
            structured_output=(
                StructuredOutputSpec(
                    strategy="tool",
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                )
                if partial_tool
                else None
            ),
            tool_completion=(
                ToolCompletionPolicy(tool_names=["ask_customer"]) if final_tool else None
            ),
        ),
        registered,
        context=CONTEXT,
    )


if __name__ == "__main__":
    asyncio.run(main())
