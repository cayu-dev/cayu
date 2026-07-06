"""Tests for CayuApp registry introspection (list_agents/providers/environments)."""

from __future__ import annotations

from cayu import AgentSpec, CayuApp, Environment, EnvironmentSpec, ScriptedModelProvider


def test_registry_introspection_lists_sorted_names() -> None:
    app = CayuApp()
    provider = ScriptedModelProvider([])
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="beta", model="m"))
    app.register_agent(AgentSpec(name="alpha", model="m"))
    app.register_environment(Environment(EnvironmentSpec(name="env-1")), default=True)

    assert app.list_agents() == ("alpha", "beta")
    assert app.list_providers() == (provider.name,)
    assert app.list_environments() == ("env-1",)


def test_empty_registries_are_empty_tuples() -> None:
    app = CayuApp()
    assert app.list_agents() == ()
    assert app.list_providers() == ()
    assert app.list_environments() == ()
