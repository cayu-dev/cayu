from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    ChatCompletionsProvider,
    Environment,
    EnvironmentSpec,
    ExecCommandTool,
    ListFilesTool,
    LocalRunner,
    LocalWorkspace,
    Message,
    ReadFileTool,
    RunRequest,
    WriteFileTool,
)

# ChatCompletionsProvider targets any OpenAI-compatible /v1/chat/completions
# service. This example runs against Google Gemini (AI Studio) because it has a
# free tier, but the provider is generic. To target another service, change only
# the provider construction below, e.g.:
#
#   # Azure OpenAI (uses the default document_encoding="file")
#   ChatCompletionsProvider(
#       name="azure",
#       api_key_env="AZURE_OPENAI_API_KEY",
#       base_url="https://<resource>.openai.azure.com/openai/deployments/<deployment>",
#       api_version="2024-10-21",
#   )
#
#   # Together AI
#   ChatCompletionsProvider(
#       name="together",
#       api_key_env="TOGETHER_API_KEY",
#       base_url="https://api.together.xyz/v1",
#   )
#
# document_encoding="image_url" is set below because Gemini's compatible endpoint
# carries PDFs through the image_url content part; OpenAI/Azure use the default
# "file" part instead.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


async def main() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY to run this live Chat Completions example.")
        return

    model = os.environ.get("CAYU_GEMINI_MODEL", "gemini-2.5-flash")
    root = (
        Path(__file__).resolve().parents[1]
        / ".examples-workspaces"
        / "chat-completions-local-tools"
    )
    root.mkdir(parents=True, exist_ok=True)
    workspace = LocalWorkspace(root, workspace_id="chat-completions-local-demo")
    runner = LocalRunner(root, inherit_env=False)

    print("workspace_root", root)

    app = CayuApp()
    app.register_provider(
        ChatCompletionsProvider(
            name="gemini",
            api_key_env="GEMINI_API_KEY",
            base_url=GEMINI_BASE_URL,
            document_encoding="image_url",
        ),
        default=True,
    )
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
        session_id="demo_chat_completions_local_tools",
        messages=[
            Message.text(
                "user",
                (
                    "Create notes/result.txt with the text 'chat completions ok'. "
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
    usage = await app.get_session_usage("demo_chat_completions_local_tools")
    print("usage_input_tokens", usage.usage.input_tokens)
    print("usage_output_tokens", usage.usage.output_tokens)
    print("usage_cache_read_tokens", usage.usage.cache.read_tokens)
    print("usage_cache_write_tokens", usage.usage.cache.write_tokens)
    print("usage_cached_input_tokens", usage.usage.cache.cached_input_tokens)


if __name__ == "__main__":
    asyncio.run(main())
