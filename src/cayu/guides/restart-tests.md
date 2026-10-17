# Restart tests

A Cayu agent that pauses for an approval or a user's answer usually resumes in a
different process: a server worker, a CLI command, or the next deploy. Test that
directly. `cayu.testing.RestartHarness` runs your application factory, stops it,
and builds a fresh application on the same durable stores, so a test reads like
the real sequence of processes.

## The pattern

1. Build the app from its factory and run until the session pauses.
2. Restart: close that app and build a new one from the factory on the same stores.
3. Approve, deny, or answer the pending action from the new app.
4. Check that the protected tool ran exactly once.

```python
import asyncio

from cayu import AgentSpec, CayuApp, Message, RunRequest, Tool, ToolEffect, ToolResult, ToolSpec
from cayu.providers import ModelStreamEvent
from cayu.runtime import ExecutionProfileBehaviorIdentity
from cayu.testing import RestartHarness
from cayu.tools import ToolPolicy, ToolPolicyDecision, ToolPolicyResult

REFUNDS = []  # Stands in for the payment service, which outlives every process.


class IssueRefund(Tool):
    spec = ToolSpec(
        name="issue_refund",
        description="Refund an order.",
        input_schema={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        effect=ToolEffect.IDEMPOTENT,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="issue_refund", behavior_version="1", implementation_version="1"
        ),
    )

    async def run(self, ctx, args):
        REFUNDS.append(args["order_id"])
        return ToolResult(content=f"Refunded {args['order_id']}.")


class RefundsNeedApproval(ToolPolicy):
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="refunds-need-approval", behavior_version="1", implementation_version="1"
        )

    async def authorize(self, request):
        if request.tool_name == "issue_refund":
            return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


def build_app(*, provider, session_store, task_store) -> CayuApp:
    app = CayuApp(session_store=session_store, task_store=task_store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="refunds", model="scripted-model"),
        tools=[IssueRefund()],
        tool_policy=RefundsNeedApproval(),
    )
    return app


def call(name, arguments):
    return [
        ModelStreamEvent.tool_call(name=name, arguments=arguments),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def final(text):
    return [ModelStreamEvent.text_delta(text), ModelStreamEvent.completed()]


def test_refund_survives_restart_before_approval(tmp_path):
    async def scenario():
        async with RestartHarness(
            build_app,
            store_dir=tmp_path,
            model_script=[call("issue_refund", {"order_id": "A-1"}), final("Refunded.")],
        ) as harness:
            paused = await harness.run(
                RunRequest(
                    agent_name="refunds",
                    session_id="refund-A-1",
                    messages=[Message.text("user", "Please refund order A-1.")],
                )
            )
            [approval] = paused.pending_approvals
            assert REFUNDS == []

            await harness.restart()
            done = await harness.approve(approval, actor="ops@example.test")

            assert done.completed
            assert done.final_text == "Refunded."
            await harness.assert_tool_ran_once("refund-A-1", "issue_refund")
            assert REFUNDS == ["A-1"]

    asyncio.run(scenario())
```

The factory is called with `provider=`, `session_store=` and `task_store=`. The
`build_app` that `cayu new` generates accepts those keywords, so a generated
project passes it as is. For any other factory, pass a small function that
forwards those three values.

`model_script` is the whole conversation, as `ScriptedModelProvider` batches.
Each generation gets a fresh provider that starts at the first turn the earlier
ones did not use. `harness.model_requests` holds every request across
generations and `harness.remaining_model_turns` counts what is left.

## What each call does

| Call | Effect |
| --- | --- |
| `await harness.run(request)` | Runs a `RunRequest` (or `ResumeRequest`) until the session completes, fails or pauses. |
| `await harness.restart()` | Closes the app (`aclose()`), then the stores the harness opened, and builds the next generation. |
| `await harness.approve(action, actor=...)` / `deny(...)` | Resolves a pending tool approval through `resolve_tool_approval` and runs on. |
| `await harness.answer_user_input(action, answer)` | Answers a pending `ask_user` question through `resolve_user_input` and runs on. |
| `await harness.resume(session_id, messages=[...])` | Continues the conversation with new messages. |
| `await harness.pending_actions(session_id)` | Lists durable pending actions, as the server's pending-action route does. |
| `await harness.assert_tool_ran_once(session_id, tool_name)` | Fails unless the durable history shows exactly one start of that tool and its completion. |

`action` is a pending action from a step's `pending_approvals` or
`pending_user_inputs`, or a session ID when that session has exactly one action
of that kind. Every call returns a `RestartStep` with the session's durable
`status`, `pending_actions`, the step's `events` and its `final_text`.

`assert_tool_ran_once` checks what Cayu dispatched. A real service also needs
its own idempotency key and record; assert on that too.

## Stores

Pass `store_dir=` for a SQLite database in that directory, or `database_url=` for
an absolute `sqlite:///` file or a PostgreSQL URL. PostgreSQL needs its schema
migrated first (`cayu storage migrate`), as in a deployment. The harness opens
the stores with `open_application_stores`, the same function generated projects
use.

In-memory stores are refused with a `ValueError`, including a factory that
ignores the stores it is given and builds its own `InMemorySessionStore`. They
lose every session on restart, so a passing test would prove nothing.

## Identity checks still apply

The harness does not relax execution-profile admission. If the factory builds a
tool with a different declared identity after a restart, resolving the approval
raises `ExecutionProfileMismatchError`, as it would in a redeployed process. A
tool or policy without `execution_profile_identity` gets a process-local
identity, which also fails after a restart. Declare stable identities from the
first run; see `cayu guide durable-service-tools`.

## Not covered yet

The harness restarts cleanly between steps. It does not inject a crash in the
middle of a model call or tool dispatch, and it has no clock control for
timers. Crash recovery after a lost process is described in
`cayu guide order-support#troubleshooting-and-limits` and
`cayu guide durable-operations`.
