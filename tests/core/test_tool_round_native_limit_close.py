"""Tool-round limit closure must not orphan native asynchronous generators."""

import asyncio

from tests.core._execution_profile_fixtures import rebind_test_invocation
from tests.core.test_tool_round_executor import (
    _app_with_completed_session,
    _tool_call,
    _tool_round_identity,
    _tool_round_run,
)

from cayu import EventType
from cayu.budgets.run_limits import RunLimits
from cayu.runtime._tool_round_recovery import checkpoint_with_pending_tool_round


def test_native_limit_stream_teardown_has_no_asyncgen_errors():
    app, store, _ = _app_with_completed_session("round-limit-native-close")

    async def scenario():
        errors = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        session = await rebind_test_invocation(store, "round-limit-native-close")
        runner = await _tool_round_run(app, session, limits=RunLimits(max_total_tokens=10))
        calls = [_tool_call()]
        checkpoint, _ = checkpoint_with_pending_tool_round(
            await store.load_checkpoint(session.id),
            agent_name="assistant",
            environment_name=None,
            task_id=None,
            tool_calls=calls,
            policy_outcomes=None,
            structured_output=None,
            tool_round_identity=_tool_round_identity(),
        )
        await store.checkpoint(session.id, checkpoint)
        stream = runner.run(
            messages=await store.load_transcript(session.id),
            tool_calls=calls,
            tool_round_identity=_tool_round_identity(),
        )
        assert (await anext(stream)).type is EventType.SESSION_LIMIT_REACHED
        await stream.aclose()
        await loop.shutdown_asyncgens()
        await asyncio.sleep(0)
        failures = [(context.get("message"), repr(context.get("exception"))) for context in errors]
        assert failures == []

    asyncio.run(scenario())
