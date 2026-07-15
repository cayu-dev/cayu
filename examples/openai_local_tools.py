from __future__ import annotations

import asyncio
import os
import sys
from decimal import Decimal
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    ExecCommandTool,
    ListFilesTool,
    LocalRunner,
    LocalWorkspace,
    Message,
    ModelPrice,
    OpenAIProvider,
    PriceBook,
    ReadFileTool,
    RunRequest,
    WriteFileTool,
)


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this live OpenAI example.")
        return

    model = os.environ.get("CAYU_OPENAI_MODEL", "gpt-5.6")
    root = Path(__file__).resolve().parents[1] / ".examples-workspaces" / "openai-local-tools"
    root.mkdir(parents=True, exist_ok=True)
    workspace = LocalWorkspace(root, workspace_id="openai-local-demo")
    runner = LocalRunner(root, inherit_env=False)

    print("workspace_root", root)

    app = CayuApp()
    app.register_provider(OpenAIProvider(), default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-dev", metadata={"kind": "local"}),
            workspace=workspace,
            runner=runner,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="assistant",
            model=model,
            system_prompt=(
                "You are testing Cayu local tools. Use tools when needed. "
                "Do not write progress narration before tool calls. Use exactly "
                "one tool call per assistant turn. Cayu currently executes tool "
                "calls sequentially, so never say work ran simultaneously, in "
                "parallel, concurrently, or at the same time. Keep the final "
                "answer short."
            ),
        ),
        tools=[
            WriteFileTool(),
            ReadFileTool(),
            ListFilesTool(),
            ExecCommandTool(),
        ],
    )

    request = RunRequest(
        agent_name="assistant",
        session_id="demo_openai_local_tools",
        messages=[
            Message.text(
                "user",
                (
                    "Create notes/result.txt with the text 'openai ok'. "
                    "After that tool result is returned, list txt files. "
                    "After that tool result is returned, run a process command "
                    "that prints the Python executable path using "
                    f"{sys.executable!r}. Do not describe progress until all "
                    "tool calls are complete."
                ),
            )
        ],
    )
    async for event in app.run(request):
        print(
            event.type,
            event.environment_name or "-",
            event.tool_name or "-",
            event.payload,
        )

    print("workspace_files", list((await workspace.list("**/*")).paths))
    usage = await app.get_session_usage("demo_openai_local_tools")
    print("usage_input_tokens", usage.usage.input_tokens)
    print("usage_output_tokens", usage.usage.output_tokens)
    print("usage_cache_read_tokens", usage.usage.cache.read_tokens)
    print("usage_cache_write_tokens", usage.usage.cache.write_tokens)
    print("usage_cached_input_tokens", usage.usage.cache.cached_input_tokens)

    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="openai",
                model=os.environ.get("CAYU_OPENAI_PRICING_MODEL_PREFIX", model),
                match="prefix",
                input_per_million=Decimal(os.environ.get("CAYU_OPENAI_INPUT_PER_MILLION", "0")),
                output_per_million=Decimal(os.environ.get("CAYU_OPENAI_OUTPUT_PER_MILLION", "0")),
                cache_read_input_per_million=Decimal(
                    os.environ.get("CAYU_OPENAI_CACHE_READ_INPUT_PER_MILLION", "0")
                ),
                cache_write_input_per_million=Decimal(
                    os.environ.get("CAYU_OPENAI_CACHE_WRITE_INPUT_PER_MILLION", "0")
                ),
            ),
        )
    )
    cost = await app.get_session_cost("demo_openai_local_tools", pricing)
    print("estimated_cost", cost.total_cost)
    print("estimated_cost_unpriced_model_steps", cost.unpriced_model_steps)


if __name__ == "__main__":
    asyncio.run(main())
