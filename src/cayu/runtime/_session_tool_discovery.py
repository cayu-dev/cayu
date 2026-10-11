"""Initialize and validate durable tool discovery at session boundaries."""

from __future__ import annotations

from typing import Any

from cayu.runtime import _runtime_records as runtime_records
from cayu.sessions._runtime_publication import SessionOperationInitializer
from cayu.sessions.base import SessionStore
from cayu.sessions.records import Session
from cayu.tools.discovery import (
    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
    current_tool_discovery_view,
    initial_tool_discovery_operation_records,
    tool_discovery_generation_id,
)
from cayu.tools.exposure import ToolCapabilityCeiling


def _tool_discovery_operation_initializer(
    registered_agent: runtime_records.RegisteredAgentState,
    ceiling: ToolCapabilityCeiling,
) -> SessionOperationInitializer | None:
    """Freeze one pure initializer for a discovery-enabled session branch."""

    if registered_agent.tool_discovery_mode is None:
        return None
    catalogue = registered_agent.tool_catalogue
    agent_name = registered_agent.spec.name

    def initialize(session: Session) -> dict[str, dict[str, Any]]:
        if session.agent_name != agent_name:
            raise ValueError("Tool discovery initializer received a different agent.")
        return initial_tool_discovery_operation_records(
            session_id=session.id,
            root_invocation_id=session.invocation.root_invocation_id,
            agent_name=agent_name,
            catalogue=catalogue,
            ceiling=ceiling,
        )

    return initialize


async def _require_tool_discovery_view(
    store: SessionStore,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    ceiling: ToolCapabilityCeiling,
    *,
    require_empty: bool,
) -> None:
    """Validate exact persisted discovery authority after a lifecycle boundary."""

    if registered_agent.tool_discovery_mode is None:
        return
    state = current_tool_discovery_view(
        await store.load_session_operation(
            session.id,
            TOOL_DISCOVERY_VIEW_OPERATION_KEY,
        ),
        session_id=session.id,
        generation_id=tool_discovery_generation_id(
            session_id=session.id,
            root_invocation_id=session.invocation.root_invocation_id,
        ),
        agent_name=registered_agent.spec.name,
        catalogue=registered_agent.tool_catalogue,
        ceiling=ceiling,
    )
    if require_empty and (state.revision != 0 or state.grants):
        raise RuntimeError("New session branch did not start with an empty discovery view.")
