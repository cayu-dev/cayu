"""Minimal cayu server example.

Usage:
    pip install cayu[server]
    CAYU_SERVER_PASSWORD=... OPENAI_API_KEY=... python examples/server_example.py
    # or:
    CAYU_SERVER_PASSWORD=... ANTHROPIC_API_KEY=... python examples/server_example.py
    # or (Google Gemini via the OpenAI-compatible endpoint):
    CAYU_SERVER_PASSWORD=... GEMINI_API_KEY=... python examples/server_example.py

    # Then:
    # Open http://localhost:8000/cayu/ for the CAYU runtime dashboard.
    curl http://localhost:8000/api/health
    curl -N -u "${CAYU_SERVER_USERNAME:-admin}:$CAYU_SERVER_PASSWORD" \
      -X POST http://localhost:8000/api/run \
      -H "Content-Type: application/json" \
      -d '{"prompt": "List files in the workspace"}'
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from cayu import (
    AgentSpec,
    CayuApp,
    ContextPolicy,
    ContextRequest,
    Environment,
    EnvironmentSpec,
    ListFilesTool,
    LocalRunner,
    LocalWorkspace,
    Message,
    ReadFileTool,
    SQLiteSessionStore,
    SQLiteTaskStore,
    WriteFileTool,
    trim_context_turns,
)
from cayu.server import BasicAuth, ServerConfig, create_server

WORKSPACE = Path(__file__).parent / ".examples-workspaces" / "server"
DATA_DIR = WORKSPACE / "data"


class RecentContextPolicy(ContextPolicy):
    async def build(self, request: ContextRequest) -> list[Message]:
        return trim_context_turns(request.messages, max_user_turns=5, preserve_system=True)


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    database = DATA_DIR / "cayu.db"
    session_store = SQLiteSessionStore(database)
    task_store = SQLiteTaskStore(database)

    app = CayuApp(session_store=session_store, task_store=task_store)
    model = _register_provider(app)

    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=LocalWorkspace(WORKSPACE, workspace_id="workspace"),
            runner=LocalRunner(WORKSPACE),
        ),
        default=True,
    )

    app.register_agent(
        AgentSpec(
            name="assistant",
            model=model,
            system_prompt="You are a helpful assistant with workspace access.",
        ),
        tools=[ReadFileTool(), WriteFileTool(), ListFilesTool()],
        context_policy=RecentContextPolicy(),
    )

    server = create_server(
        app,
        config=ServerConfig.protected(
            _server_auth(),
            deployment_name=os.environ.get("CAYU_SERVER_DEPLOYMENT_NAME", "development"),
        ),
    )
    uvicorn.run(server, host="0.0.0.0", port=8000)


def _server_auth() -> BasicAuth:
    password = os.environ.get("CAYU_SERVER_PASSWORD")
    if not password:
        raise RuntimeError("Set CAYU_SERVER_PASSWORD before starting examples/server_example.py.")
    return BasicAuth(
        username=os.environ.get("CAYU_SERVER_USERNAME", "admin"),
        password=password,
    )


def _register_provider(app: CayuApp) -> str:
    if os.environ.get("OPENAI_API_KEY"):
        from cayu import OpenAIProvider

        app.register_provider(OpenAIProvider(), default=True)
        return os.environ.get("CAYU_OPENAI_MODEL", "gpt-5.6")

    if os.environ.get("ANTHROPIC_API_KEY"):
        from cayu import AnthropicProvider

        app.register_provider(AnthropicProvider(), default=True)
        return os.environ.get("CAYU_ANTHROPIC_MODEL", "claude-sonnet-4-6")

    if os.environ.get("GEMINI_API_KEY"):
        from cayu import ChatCompletionsProvider

        app.register_provider(
            ChatCompletionsProvider(
                name="google",
                api_key_env="GEMINI_API_KEY",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                document_encoding="image_url",
            ),
            default=True,
        )
        return os.environ.get("CAYU_GEMINI_MODEL", "gemini-3.5-flash")

    raise RuntimeError(
        "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY before starting "
        "examples/server_example.py."
    )


if __name__ == "__main__":
    main()
