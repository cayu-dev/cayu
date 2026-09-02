"""Hermetic durable workflow with isolated files, bounded commands, and trusted outcomes.

Run with:
    uv run python examples/durable_file_workflow/demo.py
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    ExecCommandTool,
    InMemoryTaskStore,
    LocalRunner,
    LocalWorkspace,
    Message,
    NativeBinding,
    ParameterConstrainedToolPolicy,
    ProcessCommandPolicy,
    ReadFileTool,
    RequiredAllowlistRule,
    RequiredFieldRule,
    RunRequest,
    Task,
    TaskCreate,
    TaskQuery,
    WriteFileTool,
    complete_managed_task,
    run_task_worker,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent

GOAL_PROMPT = """\
Goal: Transform the supplied source text into the required artifact.
Inputs: The user message contains the source text.
Output contract: Create result.txt containing the upper-cased source plus one newline.
Constraints: Work only in the session workspace. Use process commands, never shell commands.
Recover from tool failures, then report completion only after reading the final artifact.
"""


@dataclass(frozen=True)
class DemoResult:
    completed: Task
    second_completed: Task
    blocked: Task
    provider_call_count: int
    session_roots: tuple[str, ...]


class IsolatedLocalFactory(EnvironmentFactory):
    """Create one native local environment per session."""

    def __init__(self, base_root: Path) -> None:
        self._base_root = base_root
        self.roots: dict[str, Path] = {}

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        root = self._base_root / request.session_id
        root.mkdir(parents=True, exist_ok=False)
        source = request.metadata.get("source_text")
        if isinstance(source, str):
            (root / "source.txt").write_text(source, encoding="utf-8")
        self.roots[request.session_id] = root
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(root),
                runner=LocalRunner(root, inherit_env=False),
                binding=NativeBinding(),
            )
        )


class RecoveryProvider(ModelProvider):
    """Deterministic provider that observes a failed command and repairs its file."""

    name = "recovery-script"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._turns = (
            (
                ModelStreamEvent.tool_call(
                    id="write-broken",
                    name="write_file",
                    arguments={
                        "path": "transform.py",
                        "content": "raise RuntimeError('repair me')\n",
                        "mode": "create",
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="run-broken",
                    name="exec_command",
                    arguments={
                        "kind": "process",
                        "argv": [sys.executable, "transform.py"],
                        "timeout_s": 30,
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ),
            (
                ModelStreamEvent.tool_call(
                    id="write-repair",
                    name="write_file",
                    arguments={
                        "path": "transform.py",
                        "content": (
                            "from pathlib import Path\n"
                            "source = Path('source.txt').read_text(encoding='utf-8')\n"
                            "Path('result.txt').write_text("
                            "source.strip().upper() + '\\n', encoding='utf-8')\n"
                        ),
                        "mode": "overwrite",
                        "expected_revision": (
                            "sha256:"
                            + hashlib.sha256(b"raise RuntimeError('repair me')\n").hexdigest()
                        ),
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="run-repair",
                    name="exec_command",
                    arguments={
                        "kind": "process",
                        "argv": [sys.executable, "transform.py"],
                        "timeout_s": 30,
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="read-result",
                    name="read_file",
                    arguments={"path": "result.txt"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ),
            (
                ModelStreamEvent.text_delta("Recovered and produced the artifact."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self._turns[(len(self.requests) - 1) % len(self._turns)]:
            yield event


def build_app(
    base_root: Path,
) -> tuple[CayuApp, InMemoryTaskStore, IsolatedLocalFactory, RecoveryProvider]:
    task_store = InMemoryTaskStore()
    factory = IsolatedLocalFactory(base_root)
    provider = RecoveryProvider()
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="isolated-local"),
        factory,
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="file-worker",
            model="scripted",
            system_prompt=GOAL_PROMPT,
            workflow_tool_names=("write_file", "exec_command", "read_file"),
        ),
        tools=[
            WriteFileTool(),
            ExecCommandTool(
                policy=ProcessCommandPolicy(
                    allowed_executables=(sys.executable,),
                    allowed_cwds=(str(base_root.resolve()),),
                    max_timeout_s=30,
                )
            ),
            ReadFileTool(),
        ],
        tool_policy=ParameterConstrainedToolPolicy(
            {
                "write_file": (
                    RequiredAllowlistRule("path", values=("transform.py",)),
                    RequiredFieldRule("content"),
                ),
                "exec_command": (
                    RequiredAllowlistRule("kind", values=("process",)),
                    RequiredFieldRule("argv"),
                ),
                "read_file": (RequiredAllowlistRule("path", values=("result.txt",)),),
            }
        ),
    )
    return app, task_store, factory, provider


async def run_demo(base_root: Path) -> DemoResult:
    app, task_store, factory, provider = build_app(base_root)
    await task_store.create_task(
        TaskCreate(
            task_id="transform-source",
            type="transform_file",
            assigned_agent_name="file-worker",
            input={"source_text": "cayu"},
        )
    )
    await task_store.create_task(
        TaskCreate(
            task_id="transform-second-source",
            type="transform_file",
            assigned_agent_name="file-worker",
            input={"source_text": "runtime"},
        )
    )
    await task_store.create_task(
        TaskCreate(
            task_id="missing-source",
            type="transform_file",
            assigned_agent_name="file-worker",
        )
    )

    async def handle(_app: CayuApp, task: Task, worker_id: str) -> None:
        source = task.input.get("source_text")
        if not isinstance(source, str) or not source.strip():
            await task_store.block_task(
                task.id,
                reason="Required source text is missing.",
                payload={"missing": "source_text"},
            )
            return

        session_id = f"session-{task.id}"
        async for _event in _app.run(
            RunRequest(
                agent_name="file-worker",
                session_id=session_id,
                metadata={"source_text": source},
                messages=[Message.text("user", f"Transform this source text: {source}")],
            )
        ):
            pass

        output_path = factory.roots[session_id] / "result.txt"
        output = output_path.read_text(encoding="utf-8")
        expected = source.strip().upper() + "\n"
        if output != expected:
            await task_store.mark_task_needs_attention(
                task.id,
                reason="Generated artifact failed trusted verification.",
                payload={"artifact": "result.txt"},
            )
            return
        await complete_managed_task(
            task_store,
            task,
            worker_id,
            {"artifact": "result.txt", "content": output, "verified": True},
        )

    handled = await run_task_worker(
        app,
        task_store,
        handle,
        worker_id="demo-worker",
        query=TaskQuery(type="transform_file"),
        max_tasks=3,
        poll_interval_s=0.01,
        reclaim=False,
    )
    if handled != 3:
        raise RuntimeError(f"Expected three handled tasks, got {handled}.")
    completed = await task_store.load_task("transform-source")
    second_completed = await task_store.load_task("transform-second-source")
    blocked = await task_store.load_task("missing-source")
    if completed is None or second_completed is None or blocked is None:
        raise RuntimeError("Demo tasks disappeared.")
    return DemoResult(
        completed=completed,
        second_completed=second_completed,
        blocked=blocked,
        provider_call_count=len(provider.requests),
        session_roots=tuple(sorted(factory.roots)),
    )


async def main() -> None:
    with TemporaryDirectory(prefix="cayu-durable-file-") as temp:
        result = await run_demo(Path(temp))
        print(result.completed.id, result.completed.status, result.completed.result)
        print(
            result.second_completed.id,
            result.second_completed.status,
            result.second_completed.result,
        )
        print(result.blocked.id, result.blocked.status, result.blocked.status_reason)


if __name__ == "__main__":
    asyncio.run(main())
