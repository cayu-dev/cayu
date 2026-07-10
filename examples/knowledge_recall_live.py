"""Demo-only live knowledge-recall example.

This exercises a real provider with knowledge tools, but it does not assert
model prose. Treat it as smoke coverage in nightly reports.
"""

from __future__ import annotations

import asyncio
import os

from cayu import (
    AgentSpec,
    AnthropicProvider,
    CayuApp,
    Environment,
    EnvironmentSpec,
    InMemoryKnowledgeStore,
    KnowledgeIndexer,
    KnowledgeIndexRequest,
    ListKnowledgeTool,
    Message,
    OpenAIProvider,
    ReadKnowledgeTool,
    RunRequest,
    SearchKnowledgeTool,
)


async def main() -> None:
    provider_name = _provider_name()
    model = _model(provider_name)
    if provider_name == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY or choose CAYU_PROVIDER=anthropic.")
        return
    if provider_name == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY or choose CAYU_PROVIDER=openai.")
        return

    knowledge_store = InMemoryKnowledgeStore()
    await _seed_knowledge(knowledge_store)

    app = CayuApp()
    if provider_name == "openai":
        app.register_provider(OpenAIProvider(), default=True)
    else:
        app.register_provider(AnthropicProvider(), default=True)

    app.register_environment(
        Environment(
            EnvironmentSpec(name="knowledge-live"),
            knowledge_store=knowledge_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="assistant",
            model=model,
            system_prompt=(
                "Use knowledge tools for durable project guidance. Start with "
                "list_knowledge when you do not know the exact terms, then use "
                "search_knowledge and read_knowledge. Do not use negative search "
                "terms unless a previous result proves that term is irrelevant. "
                "Keep the final answer concise and cite entry ids."
            ),
        ),
        tools=[
            ListKnowledgeTool(),
            SearchKnowledgeTool(),
            ReadKnowledgeTool(),
        ],
    )

    print("provider", provider_name)
    print("model", model)
    print("seeded_entries", ["remote_git_credentials", "test_credentials", "sendgrid_proxy"])

    request = RunRequest(
        agent_name="assistant",
        session_id=f"demo_{provider_name}_knowledge_recall",
        messages=[
            Message.text(
                "user",
                (
                    "Find the project guidance for pushing to GitHub from a remote "
                    "sandbox without exposing credentials. Use the knowledge tools "
                    "instead of answering from general knowledge. First discover what "
                    "knowledge exists, then search and read the relevant entry. "
                    "Explain the recommended approach briefly."
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


async def _seed_knowledge(store: InMemoryKnowledgeStore) -> None:
    indexer = KnowledgeIndexer(store)
    await indexer.index_text(
        KnowledgeIndexRequest(
            entry_id="remote_git_credentials",
            namespace="project:cayu",
            title="Remote sandbox Git credential boundary",
            kind="procedure",
            labels={"area": "sandbox-git", "project": "cayu"},
            aspects=["credentials", "remote-sandbox", "git"],
            impact_targets=["sandbox.git.push"],
            source_type="design-note",
            text=(
                "For GitHub clone or push from a remote sandbox, prefer a brokered "
                "Git HTTP proxy. The agent runs normal git commands against the proxy "
                "URL inside the sandbox. The trusted Cayu side forwards Git smart HTTP "
                "requests to GitHub and injects the credential outside the sandbox, so "
                "the raw token is never present in sandbox environment variables, files, "
                "process arguments, or command output. Avoid putting long-lived GitHub "
                "tokens into generic exec_command environments."
            ),
        )
    )
    await indexer.index_text(
        KnowledgeIndexRequest(
            entry_id="test_credentials",
            namespace="project:cayu",
            title="Fixture credentials",
            kind="example",
            labels={"area": "tests", "project": "cayu"},
            aspects=["credentials", "testing"],
            source_type="test-fixture",
            text=(
                "Test credentials in unit tests are fake fixture values. They are useful "
                "for checking redaction and policy behavior, but they are not production "
                "guidance for GitHub push from a remote sandbox."
            ),
        )
    )
    await indexer.index_text(
        KnowledgeIndexRequest(
            entry_id="sendgrid_proxy",
            namespace="project:cayu",
            title="SendGrid credential proxy",
            kind="procedure",
            labels={"area": "email", "project": "cayu"},
            aspects=["credentials", "proxy"],
            impact_targets=["email.send"],
            source_type="design-note",
            text=(
                "For SendGrid, prefer a trusted tool or credential proxy that performs "
                "the API request outside the sandbox. Do not expose the SendGrid API key "
                "through generic shell access."
            ),
        )
    )


def _provider_name() -> str:
    requested = os.environ.get("CAYU_PROVIDER")
    if requested is not None:
        requested = requested.strip().lower()
        if requested in {"openai", "anthropic"}:
            return requested
        raise RuntimeError("CAYU_PROVIDER must be openai or anthropic.")
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


def _model(provider_name: str) -> str:
    if provider_name == "openai":
        return os.environ.get("CAYU_OPENAI_MODEL", "gpt-5.5")
    return os.environ.get("CAYU_ANTHROPIC_MODEL", "claude-sonnet-4-6")


if __name__ == "__main__":
    asyncio.run(main())
