"""Live coding loop on an AWS Lambda MicroVM: edit, failing check, repair, passing check.

One MicroVM from the first-party image (which ships ``git`` and ``rg``) runs a
real ``CayuApp`` session. Every tool is admitted through live executable
evidence from that exact MicroVM before the model sees it. The model writes a
buggy module, a named check fails, the model repairs the file with a
revision-guarded edit, the same check passes, and ``git_changes`` and
``search_text`` operate on the same admitted workspace. Named checks keep their
timeout, output bound, and exact command authority.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncIterator

from examples._live_checks import require

from cayu import AgentSpec, CayuApp, ExecCommand, LambdaMicroVMRunner, Message, RunRequest
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import EventType
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.tools.commands import CommandPolicy, CommandPolicyDecision, CommandPolicyResult
from cayu.tools.files import EditFileTool, WriteFileTool
from cayu.tools.git import GitChangesTool
from cayu.tools.named_checks import NamedCheck, RunCheckTool
from cayu.tools.search import SearchTextTool
from cayu.workspaces import RunnerWorkspace

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
_PROJECT = "project"
_BUGGY = "def add(a, b):\n    return a - b\n"
_TEST = "from calc import add\n\nassert add(2, 3) == 5, add(2, 3)\nprint('ok')\n"
_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="examples.lambda_microvm_coding.check_test",
    behavior_version="1",
    implementation_version="1",
)
_CHECK = NamedCheck(
    name="test",
    description="Run the project's assertion script.",
    command=ExecCommand.process("python3", "test_calc.py"),
    timeout_s=30,
    max_output_bytes=4096,
    execution_profile_identity=_IDENTITY,
    required_executables=("python3",),
)


class _ExactCheckPolicy(CommandPolicy):
    """Allow only the declared check command."""

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _IDENTITY

    async def evaluate(self, ctx, request) -> CommandPolicyResult:
        del ctx
        allowed = tuple(request.command.argv or ()) == tuple(_CHECK.command.argv or ())
        return CommandPolicyResult(
            decision=CommandPolicyDecision.ALLOW if allowed else CommandPolicyDecision.DENY,
            reason=None if allowed else "only the declared check may run",
        )


class _CodingModel(ModelProvider):
    """A deterministic coder that reads revisions from its own tool results."""

    name = "lambda-coding-script"

    def __init__(self) -> None:
        self.requests = 0

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="examples.lambda_microvm_coding.model",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        # Stateless: the next action follows from the tool results already in
        # the request, so retried or repeated model requests stay consistent.
        messages = [m.model_dump(mode="json") for m in request.messages]
        completed = sum(1 for message in messages if message.get("role") == "tool")
        transcript = json.dumps(messages)
        self.requests += 1
        calls = {
            0: ("write_file", {"path": "calc.py", "content": _BUGGY, "mode": "create"}),
            1: ("run_check", {"check": "test"}),
            2: ("edit_file", None),
            3: ("run_check", {"check": "test"}),
            4: ("git_changes", {"mode": "diff"}),
            5: ("search_text", {"pattern": "def add", "mode": "content"}),
        }
        if completed in calls:
            name, arguments = calls[completed]
            if arguments is None:
                arguments = {
                    "path": "calc.py",
                    "expected_revision": _last_revision(transcript),
                    "edits": [{"old_text": "a - b", "new_text": "a + b"}],
                }
            yield ModelStreamEvent.tool_call(id=f"call-{completed}", name=name, arguments=arguments)
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("fixed")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _last_revision(transcript: str) -> str:
    matches = re.findall(r"Revision: ([^\s\\\"]+)", transcript)
    require(bool(matches), "write_file did not report a revision")
    return matches[-1]


def _tool_results(events) -> list[dict]:
    return [
        event.payload["result"]
        for event in events
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
    ]


async def main() -> None:
    if os.environ.get("CAYU_LAMBDA_MICROVM_CODING_LIVE") != "1":
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_CODING_LIVE=1 to run this contract.")
    image = os.environ.get("CAYU_LAMBDA_MICROVM_IMAGE", "")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not image.startswith("arn:") or not region:
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_IMAGE (a built image ARN) and AWS_REGION.")
    started = time.monotonic()
    runner: LambdaMicroVMRunner | None = None
    allocator: LambdaMicroVMRunner | None = None
    try:
        allocator = await LambdaMicroVMRunner.create(
            image,
            region_name=region,
            ingress_network_connectors=[
                f"arn:aws:lambda:{region}:aws:network-connector:aws-network-connector:ALL_INGRESS"
            ],
            maximum_duration_in_seconds=900,
            close_action="none",
        )
        microvm_id = allocator.microvm_id
        created = await allocator.exec(
            ExecCommand.process(
                "python3",
                "-c",
                "import os, sys; os.makedirs(sys.argv[1], exist_ok=True); "
                "open(os.path.join(sys.argv[1], 'test_calc.py'), 'w').write(sys.stdin.read())",
                _PROJECT,
            ),
            stdin=_TEST,
        )
        require(created.exit_code == 0, f"project files failed: {created.stderr}")
        setup = await allocator.exec(
            ExecCommand.bash(
                f"cd {_PROJECT} && git init -q && "
                "git config user.email live@cayu.invalid && git config user.name cayu && "
                "git add . && git commit -qm baseline"
            )
        )
        require(setup.exit_code == 0, f"project setup failed: {setup.stderr}")
        await allocator.close()
        # Checks and commands run in the runner root, so the session's runner
        # is rooted at the project inside the same MicroVM.
        runner = await LambdaMicroVMRunner.from_existing(
            microvm_id,
            region_name=region,
            default_cwd=f"/workspace/{_PROJECT}",
            close_action="terminate",
        )
        workspace = RunnerWorkspace(runner, workspace_id="lambda-coding")
        model = _CodingModel()
        app = CayuApp(enable_logging=False)
        app.register_provider(model, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="lambda"), runner=runner, workspace=workspace),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="coder", model="scripted"),
            tools=[
                WriteFileTool(),
                EditFileTool(),
                RunCheckTool(checks=[_CHECK], command_policy=_ExactCheckPolicy()),
                GitChangesTool(),
                SearchTextTool(),
            ],
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="coder",
                    session_id="lambda-coding-live",
                    messages=[Message.text("user", "Implement add and make the check pass.")],
                )
            )
        ]
        failed = next((event for event in events if event.type is EventType.SESSION_FAILED), None)
        require(failed is None, f"coding session failed: {failed.payload if failed else None}")
        require(
            EventType.SESSION_COMPLETED in {event.type for event in events},
            f"coding session did not complete: {[event.type.value for event in events[-6:]]}",
        )
        results = _tool_results(events)
        require(len(results) == 6, f"expected six completed tool calls, got {len(results)}")
        write, first_check, edit, second_check, diff, _search = (
            result["structured"] for result in results
        )
        require(write.get("mode") == "create", "write_file did not create the module")
        require(first_check.get("status") == "failed", "the buggy module did not fail its check")
        require(edit.get("replacement_count") == 1, "edit_file did not repair the module")
        require(
            second_check.get("status") == "passed",
            f"the repaired module did not pass: {results[3]['content'][:500]}",
        )
        require(
            "calc.py" in {change.get("path") for change in diff.get("changes", [])},
            f"git_changes did not report the new module: {diff.get('changes')}",
        )
        require(
            not results[5]["is_error"] and "def add" in results[5]["content"],
            "search_text did not find the function with ripgrep",
        )
        final = await workspace.read_bytes("calc.py")
        require(final.content == b"def add(a, b):\n    return a + b\n", "repair not persisted")
        verify = await runner.exec(ExecCommand.process("python3", "test_calc.py"))
        require(verify.exit_code == 0 and verify.stdout.strip() == "ok", "independent check failed")
        print(
            EVIDENCE_PREFIX
            + json.dumps(
                {
                    "adapter": "lambda-microvm",
                    "region": region,
                    "tools_admitted_live": [
                        "write_file",
                        "edit_file",
                        "run_check",
                        "git_changes",
                        "search_text",
                    ],
                    "edit_fail_repair_pass": "verified",
                    "git_changes": "verified",
                    "search_text_rg": "verified",
                    "same_admitted_workspace": "verified",
                    "seconds": round(time.monotonic() - started, 1),
                },
                sort_keys=True,
            )
        )
    finally:
        if runner is not None:
            await runner.close()
        elif allocator is not None:
            allocator.close_action = "terminate"
            await allocator.close()


if __name__ == "__main__":
    asyncio.run(main())
