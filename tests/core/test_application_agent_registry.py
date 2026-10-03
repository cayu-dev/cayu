"""Agent registration composes independently and publishes one current catalogue."""

from __future__ import annotations

import asyncio
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
from tests.core.test_mcp import (
    BlockingPublicationMcpSession,
    _fake_server_spec,
    _fake_tool_definitions,
    _fake_toolset,
)

import cayu
from cayu import AgentSpec, CayuApp, McpToolset


def test_agent_registry_composes_without_application_controllers() -> None:
    script = """
import asyncio
import importlib.abc
import sys

blocked = {
    "cayu.applications",
    "cayu.runtime._session_engine",
    "cayu.runtime._model_step_executor",
    "cayu.runtime._recovery_coordinator",
}

class RejectControllers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Registry imported {fullname}")

sys.meta_path.insert(0, RejectControllers())
from cayu import (
    AgentSpec, InMemorySessionStore, McpInitializeResult, McpServerSpec,
    McpSession, McpToolDefinition, McpToolset, SecretRedactor,
)
from cayu._application_agent_registry import ApplicationAgentRegistry
from cayu.runtime.application_lifecycle import ApplicationAdmission, ApplicationAdmissionsSealed

class Session(McpSession):
    definitions = (McpToolDefinition(name="first", input_schema={"type": "object"}),)

    @property
    def initialize_result(self):
        return McpInitializeResult(protocol_version="2025-06-18")

    async def list_tools(self):
        if hasattr(self, "list_started"):
            self.list_started.set()
            await self.allow_list.wait()
        return self.definitions

    async def call_tool(self, name, arguments):
        raise AssertionError("Registration must not dispatch a tool")

    async def list_resources(self):
        return ()

    async def read_resource(self, uri):
        raise AssertionError("Registration must not read a resource")

    async def close(self):
        pass

async def run():
    admission = ApplicationAdmission()
    registry = ApplicationAgentRegistry(
        admission=admission,
        session_store=InMemorySessionStore(), secret_redactor=SecretRedactor(),
        tool_timeout_seconds=None, mcp_manifest_policy=None,
    )
    session = Session()
    toolset = McpToolset(
        server=McpServerSpec(name="source", command=["unused"], connection_id="source"),
        session=session, definitions=session.definitions,
    )
    try:
        for name in ("alpha", "beta"):
            registry.register_agent(AgentSpec(name=name, model="model"), mcp_toolsets=(toolset,))
        before = registry.registrations
        assert registry.list_agents() == ("alpha", "beta")
        session.definitions = (
            *session.definitions,
            McpToolDefinition(name="second", input_schema={"type": "object"}),
        )
        lease = admission.acquire()
        try:
            result = await registry.refresh_mcp_toolset(toolset)
        finally:
            admission.release(lease)
        assert result.status == "accepted"
        for name in registry.list_agents():
            assert len(before[name].tools) == 1
            assert len(registry.get_agent(name).tools) == 2
            assert registry.registrations[name].mcp_toolsets == (result.toolset,)
        copied = registry.get_agent("alpha")
        copied.tools["mcp__source__first"].schema["type"] = "array"
        assert registry.get_agent("alpha").tools["mcp__source__first"].schema["type"] == "object"
        session.list_started, session.allow_list = asyncio.Event(), asyncio.Event()
        refresh = asyncio.create_task(registry._refresh_after_notification(id(toolset._refresh_source)))
        await asyncio.wait_for(session.list_started.wait(), 5)
        assert admission.in_flight == 1
        admission.seal()
        assert not await registry.release_mcp_toolsets(0.001)
        session.allow_list.set()
        await asyncio.wait_for(refresh, 5)
        assert admission.in_flight == 0
        assert await registry.release_mcp_toolsets(1)
        assert not registry.mcp_refreshes_pending
        try:
            await registry._refresh_after_notification(id(toolset._refresh_source))
        except ApplicationAdmissionsSealed:
            pass
        else:
            raise AssertionError("A sealed registry accepted a notification refresh")
        try:
            registry.register_agent(AgentSpec(name="late", model="model"), mcp_toolsets=(result.toolset,))
        except ApplicationAdmissionsSealed:
            pass
        else:
            raise AssertionError("A sealed registry reclaimed MCP ownership")
        next_registry = ApplicationAgentRegistry(
            admission=ApplicationAdmission(), session_store=InMemorySessionStore(),
            secret_redactor=SecretRedactor(), tool_timeout_seconds=None, mcp_manifest_policy=None,
        )
        next_registry.register_agent(AgentSpec(name="next", model="model"), mcp_toolsets=(result.toolset,))
        assert len(next_registry.get_agent("next").tools) == 2
        assert not blocked.intersection(sys.modules)
    finally:
        await toolset.close()

asyncio.run(run())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_refresh_preserves_public_registration_site_and_updates_manifest() -> None:
    async def run() -> None:
        app = CayuApp(enable_logging=False)
        toolset = _fake_toolset()
        try:
            app.register_agent(AgentSpec(name="agent", model="model"), mcp_toolsets=(toolset,))
            registered = app._get_registered_agent("agent")
            assert registered.registration_source == __file__
            assert registered.registration_symbol == (
                f"{__name__}:test_refresh_preserves_public_registration_site_and_updates_manifest"
                ".<locals>.run"
            )
            before = app._agent_registry.registrations
            manifest_before = app.describe()
            toolset.session.definitions = _fake_tool_definitions("echo", "second")
            await app.refresh_mcp_toolset(toolset)
            after = app._get_registered_agent("agent")
            assert after is not registered
            assert before["agent"] is registered
            assert app._agents["agent"] is after
            assert after.registration_source == registered.registration_source
            assert after.registration_symbol == registered.registration_symbol
            assert len(app.get_agent("agent").tools) == 2
            manifest_after = app.describe()
            assert len(manifest_before.agents[0].tools) == 1
            assert len(manifest_after.agents[0].tools) == 2
        finally:
            await toolset.close()
            await app.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("independent_source", [False, True])
def test_refresh_preserves_registrations_completed_during_discovery_publication(
    independent_source: bool,
) -> None:
    async def run() -> None:
        session = BlockingPublicationMcpSession(definitions=_fake_tool_definitions("echo"))
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(update={"connection_id": "refreshing"}),
            session=session,
            definitions=session.definitions,
        )
        other = _fake_toolset(connection_id="independent") if independent_source else None
        app = CayuApp(enable_logging=False)
        refresh = None
        try:
            app.register_agent(AgentSpec(name="first", model="model"), mcp_toolsets=(toolset,))
            session.definitions = _fake_tool_definitions("echo", "search")
            refresh = asyncio.create_task(app.refresh_mcp_toolset(toolset))
            await asyncio.wait_for(session.publication_started.wait(), timeout=5)
            app.register_agent(
                AgentSpec(name="during", model="model"),
                mcp_toolsets=() if other is None else (other,),
            )
            registered_during = app._get_registered_agent("during")
            assert app.list_agents() == ("during", "first")
            session.release_publication.set()
            result = await asyncio.wait_for(refresh, timeout=5)

            assert result.status == "accepted"
            assert app.list_agents() == ("during", "first")
            assert app._get_registered_agent("during") is registered_during
            assert len(app.get_agent("first").tools) == 2
            assert tuple(agent.name for agent in app.describe().agents) == ("during", "first")
            if other is not None:
                # Its owner claim and the registry's source map must agree too.
                unchanged = await app.refresh_mcp_toolset(other)
                assert unchanged.status == "unchanged"
                assert unchanged.toolset is other
        finally:
            session.release_publication.set()
            if refresh is not None and not refresh.done():
                refresh.cancel()
                await asyncio.gather(refresh, return_exceptions=True)
            await toolset.close()
            if other is not None:
                await other.close()
            await app.aclose()

    asyncio.run(run())


def test_registry_helpers_preserve_legacy_import_and_pickle_identity() -> None:
    from cayu import _application_agent_registry as registry
    from cayu import _application_registration as validation
    from cayu import applications

    for owner, names in (
        (
            registry,
            (
                "_copy_refreshable_mcp_toolsets",
                "_mcp_refresh_source_key",
                "_registered_agent_contains_mcp_source",
                "_agents_contain_mcp_source",
                "_registered_agent_after_mcp_refresh",
            ),
        ),
        (
            validation,
            ("_validate_runtime_hooks", "_snapshot_context_behavior_execution_profile_identities"),
        ),
    ):
        for name in names:
            canonical = getattr(owner, name)
            assert getattr(applications, name) is canonical
            assert pickle.loads(f"ccayu.applications\n{name}\n.".encode()) is canonical
