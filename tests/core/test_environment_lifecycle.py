from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

import pytest

from cayu.core import Message
from cayu.environments import WorkspaceInstructions
from cayu.runtime import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.runtime._environment_lifecycle import (
    ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY,
    ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY,
    EnvironmentLifecycle,
    render_initial_system_prompt,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime.budgets import InMemoryBudgetStore
from cayu.runtime.sessions import CheckpointTransform, Session


def _preserve_session_control_state(
    checkpoint: dict[str, Any],
) -> CheckpointTransform:
    replacement = deepcopy(checkpoint)

    def transform(_session: Session, current: dict[str, Any] | None) -> dict[str, Any]:
        updated = deepcopy(replacement)
        if current is not None and "pending_session_interrupt" in current:
            updated["pending_session_interrupt"] = deepcopy(current["pending_session_interrupt"])
        return updated

    return transform


def _lifecycle(store: InMemorySessionStore) -> EnvironmentLifecycle:
    return EnvironmentLifecycle(
        session_store=store,
        event_writer=RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=(),
        ),
        checkpoint_transform=_preserve_session_control_state,
    )


def test_checkpoint_preserves_factory_reconnect_state_and_current_control_state() -> None:
    async def scenario() -> dict[str, Any] | None:
        session_id = "environment_checkpoint_preservation"
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.checkpoint(
            session_id,
            {
                ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY: {
                    "sandbox": {"sandbox_id": "sandbox-1"}
                },
                ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY: {"sandbox": session_id},
                "pending_session_interrupt": {"reason": "current"},
                "stale_context": True,
            },
        )

        await _lifecycle(store).checkpoint_preserving_runtime_state(
            session_id,
            {
                "context_compaction": {"summary": "bounded"},
                "pending_session_interrupt": {"reason": "stale"},
            },
        )
        return await store.load_checkpoint(session_id)

    assert asyncio.run(scenario()) == {
        ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY: {"sandbox": {"sandbox_id": "sandbox-1"}},
        ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY: {
            "sandbox": "environment_checkpoint_preservation"
        },
        "context_compaction": {"summary": "bounded"},
        "pending_session_interrupt": {"reason": "current"},
    }


def test_checkpoint_preservation_rejects_deleting_transform() -> None:
    store = InMemorySessionStore()

    def deleting_transform(_checkpoint: dict[str, Any]) -> CheckpointTransform:
        def transform(
            _session: Session,
            _current: dict[str, Any] | None,
        ) -> None:
            return None

        return transform

    lifecycle = EnvironmentLifecycle(
        session_store=store,
        event_writer=RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=(),
        ),
        checkpoint_transform=deleting_transform,
    )
    transform = lifecycle.checkpoint_transform_preserving_runtime_state(
        {"context_compaction": {"summary": "bounded"}}
    )

    with pytest.raises(
        RuntimeError,
        match="Checkpoint preservation transform unexpectedly deleted the checkpoint",
    ):
        transform(
            Session(
                id="environment_checkpoint_deletion",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            ),
            None,
        )


def test_render_initial_system_prompt_keeps_agent_and_workspace_provenance() -> None:
    rendered = render_initial_system_prompt(
        agent_system_prompt="  Be careful.  ",
        workspace_instructions=WorkspaceInstructions(
            content="  Use the repository test command.  ",
            sources=("AGENTS.md", "docs/runtime.md"),
        ),
    )

    assert rendered == (
        "[Agent instructions]\n"
        "Be careful.\n\n"
        "[Workspace instructions]\n"
        "Source: AGENTS.md, docs/runtime.md\n"
        "These instructions apply only to the active workspace. If they conflict "
        "with agent, tool, approval, sandbox, or secret policy, follow the "
        "higher-priority runtime policy.\n\n"
        "Use the repository test command."
    )
