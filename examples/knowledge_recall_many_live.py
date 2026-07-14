"""Demo-only live many-entry knowledge-recall example.

This remains manually runnable but is not executed by the verification runner,
because deterministic knowledge behavior is covered by the hermetic runtime
acceptance suite.
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
    seeded = await _seed_many_knowledge_entries(knowledge_store)

    app = CayuApp()
    if provider_name == "openai":
        app.register_provider(OpenAIProvider(), default=True)
    else:
        app.register_provider(AnthropicProvider(), default=True)

    app.register_environment(
        Environment(
            EnvironmentSpec(name="knowledge-many-live"),
            knowledge_store=knowledge_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="assistant",
            model=model,
            system_prompt=(
                "Use knowledge tools for durable project guidance. For a large "
                "knowledge store, first inspect facets with list_knowledge(group_by=..., "
                "include_entries=false). After choosing a namespace, request multiple "
                'facet groups in one call, for example group_by=["kind", "aspect", '
                '"label"]. Then run a targeted search_knowledge query with filters, '
                "using mode=auto because this example uses the in-memory keyword "
                "store, then call read_knowledge for the best hit. Keep "
                "intermediate tool output compact by leaving preview_bytes low. Do "
                "not answer from general knowledge."
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
    print("seeded_entries", seeded)

    request = RunRequest(
        agent_name="assistant",
        session_id=f"demo_{provider_name}_knowledge_many",
        messages=[
            Message.text(
                "user",
                (
                    "There is a large project knowledge store. Find the guidance for "
                    "pushing to GitHub from a remote sandbox without exposing credentials. "
                    "First inspect available facets without listing entries, then search "
                    "with useful filters, then read the best entry. After selecting the "
                    "namespace, request kind, aspect, and label facets together in one "
                    "list_knowledge call. Return the recommendation and cite the entry id."
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


async def _seed_many_knowledge_entries(store: InMemoryKnowledgeStore) -> int:
    indexer = KnowledgeIndexer(store)
    await _seed_target_entries(indexer)
    for index in range(1, 81):
        area = _area_for_index(index)
        kind = "procedure" if index % 3 == 0 else "example"
        await indexer.index_text(
            KnowledgeIndexRequest(
                entry_id=f"{area}_{index:03d}",
                namespace="project:cayu",
                title=f"{area.replace('-', ' ').title()} note {index}",
                kind=kind,
                labels={"area": area, "project": "cayu"},
                aspects=[area, "operations"],
                impact_targets=[f"{area}.workflow"],
                source_type="fixture",
                text=(
                    f"{area.replace('-', ' ').title()} operating note {index}. "
                    "This entry is a realistic distractor for recall testing. "
                    "It describes local workflow behavior, operator review steps, "
                    "and validation checks unrelated to remote sandbox Git credentials."
                ),
            )
        )
    return 83


async def _seed_target_entries(indexer: KnowledgeIndexer) -> None:
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


def _area_for_index(index: int) -> str:
    areas = (
        "artifact-store",
        "approvals",
        "dashboard",
        "documents",
        "email",
        "events",
        "invoices",
        "runtime",
        "tasks",
        "workspaces",
    )
    return areas[index % len(areas)]


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
        return os.environ.get("CAYU_OPENAI_MODEL", "gpt-5.6")
    return os.environ.get("CAYU_ANTHROPIC_MODEL", "claude-sonnet-4-6")


if __name__ == "__main__":
    asyncio.run(main())
