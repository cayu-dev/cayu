"""Run a deterministic coding agent behind E2B's one-way guest handoff.

This example uses no virtual credentials and makes no model-provider call. It
does require ``E2B_API_KEY`` and consumes one E2B sandbox:

    uv run --extra e2b python examples/e2b_hardened_coding_agent_live.py
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

from _live_checks import require_equal, require_exec_success
from cayu import (
    AgentSpec,
    CayuApp,
    E2BGuestProvisioner,
    E2BRunner,
    E2BWorkspace,
    Environment,
    EnvironmentSpec,
    ExecCommand,
    ExecCommandTool,
    Message,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ReadFileTool,
    RunRequest,
    WriteFileTool,
)

_VERIFICATION_ROOT = "/opt/cayu-verification"
_EXPECTED_PATH = f"{_VERIFICATION_ROOT}/expected.txt"
_VERIFIER_PATH = f"{_VERIFICATION_ROOT}/verify.py"
_EXPECTED_RESULT = "factorial(6)=720\n"

_VERIFIER = f"""\
from pathlib import Path

expected = Path({_EXPECTED_PATH!r}).read_text(encoding="utf-8")
actual = Path("/home/user/workspace/result.txt").read_text(encoding="utf-8")
if actual != expected:
    raise SystemExit("independent verification failed")
print("verified")
"""


class CodingProvider(ModelProvider):
    """A deterministic model substitute that exercises the coding tool path."""

    name = "e2b-handoff-demo"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._batches = [
            [
                ModelStreamEvent.tool_call(
                    id="write_solution",
                    name="write_file",
                    arguments={
                        "path": "solution.py",
                        "content": (
                            "from math import factorial\n"
                            "from pathlib import Path\n"
                            'Path("result.txt").write_text('
                            'f"factorial(6)={factorial(6)}\\n", encoding="utf-8")\n'
                        ),
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="run_solution",
                    name="exec_command",
                    arguments={"kind": "process", "argv": ["python3", "solution.py"]},
                ),
                ModelStreamEvent.tool_call(
                    id="read_result",
                    name="read_file",
                    arguments={"path": "result.txt"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("coding task complete"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self._batches[len(self.requests) - 1]:
            yield event


async def main() -> None:
    if not os.environ.get("E2B_API_KEY"):
        print("Set E2B_API_KEY to run this live E2B handoff example.")
        return

    retained: list[E2BGuestProvisioner] = []

    async def trusted_bootstrap(provisioner: E2BGuestProvisioner) -> None:
        retained.append(provisioner)
        await provisioner.install_file(_EXPECTED_PATH, _EXPECTED_RESULT, mode=0o444)
        await provisioner.install_file(_VERIFIER_PATH, _VERIFIER, mode=0o555)

    print("creating offline sandbox and performing trusted bootstrap")
    async with await E2BRunner.create_hardened(
        template=os.environ.get("CAYU_E2B_TEMPLATE"),
        sandbox_timeout_s=int(os.environ.get("CAYU_E2B_SANDBOX_TIMEOUT_S", "300")),
        close_action="kill",
        bootstrap=trusted_bootstrap,
    ) as runner:
        print(f"sandbox_id {runner.sandbox_id}")
        print("handoff complete: privileged provisioning is sealed")
        require_equal(retained[0].is_sealed, True, "provisioner_sealed")

        workspace = E2BWorkspace(runner, workspace_id="e2b-hardened-coding-agent")
        provider = CodingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="e2b-offline-coding",
                    metadata={"network": "offline", "guest": "hardened"},
                ),
                runner=runner,
                workspace=workspace,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="builder", model="deterministic-coding-model"),
            tools=[WriteFileTool(), ReadFileTool(), ExecCommandTool()],
        )

        async for event in app.run(
            RunRequest(
                agent_name="builder",
                session_id="demo_e2b_hardened_coding_agent",
                messages=[Message.text("user", "implement and run factorial(6)")],
            )
        ):
            print(event.type, event.tool_name or "-", event.payload)

        verifier = await runner.exec(
            ExecCommand.process("python3", _VERIFIER_PATH),
            timeout_s=20,
        )
        require_exec_success(verifier, stdout="verified\n", label="protected_verifier")
        require_equal(len(provider.requests), 2, "model_requests")

        mutation = await runner.exec(
            ExecCommand.bash(f"printf tampered >{_EXPECTED_PATH}"),
            timeout_s=10,
        )
        if mutation.exit_code == 0:
            raise RuntimeError("guest modified the protected verification input")
        print("independent verification passed; protected input remained immutable")

    print("sandbox closed")


if __name__ == "__main__":
    asyncio.run(main())
