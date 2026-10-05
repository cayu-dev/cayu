"""Environment registration owns declarations and closure publication together."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import cayu
from cayu._application_environment_registry import ApplicationEnvironmentRegistry
from cayu.artifacts import LocalArtifactStore
from cayu.environments import Environment, EnvironmentSpec
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.session_closure import RetainedSessionClosureStore
from cayu.sessions.base import InMemorySessionStore
from cayu.vaults.redaction import SecretRedactor


class UncalledFactory(EnvironmentFactory):
    def __init__(self) -> None:
        self.identity = ExecutionProfileBehaviorIdentity(
            name="factory", behavior_version="1", implementation_version="1"
        )

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return self.identity

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        raise AssertionError("Registration and inspection must not materialize a factory")


def registry(**kwargs) -> ApplicationEnvironmentRegistry:
    return ApplicationEnvironmentRegistry(
        session_store=InMemorySessionStore(),
        secret_redactor=SecretRedactor(),
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        **kwargs,
    )


def test_environment_registry_composes_without_application_controllers() -> None:
    script = """
import importlib.abc
import sys
from datetime import UTC, datetime
blocked = {
    "cayu.applications", "cayu.runtime._session_engine",
    "cayu.runtime._model_step_executor", "cayu.runtime._recovery_coordinator",
    "cayu.runtime._tool_round_executor",
}
class RejectControllers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Registry imported {fullname}")
sys.meta_path.insert(0, RejectControllers())
from cayu._application_environment_registry import ApplicationEnvironmentRegistry
from cayu.environments import Environment, EnvironmentSpec
from cayu.sessions.base import InMemorySessionStore
from cayu.vaults.redaction import SecretRedactor
registry = ApplicationEnvironmentRegistry(
    session_store=InMemorySessionStore(), secret_redactor=SecretRedactor(),
    clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
)
environment = Environment(EnvironmentSpec(name="local"))
assert registry.register(environment, default=True) is environment
assert registry.get_concrete().spec.name == "local"
assert registry.names() == ("local",)
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_environment_registry_snapshots_factory_declarations_and_public_metadata() -> None:
    owner = registry()
    factory = UncalledFactory()
    spec = EnvironmentSpec(name="factory", metadata={"nested": {"values": [1]}})
    assert (
        owner.register_factory(
            spec, factory, default=True, registration_site=("app.py", "app:build")
        )
        is factory
    )
    stored = owner.get()
    assert stored is not None
    factory.identity = factory.identity.model_copy(update={"behavior_version": "2"})
    spec.metadata["nested"]["values"].append(2)
    inspected = owner.list_registrations()[0]
    inspected.spec.metadata["nested"]["values"].append(3)
    inspected.environment.spec.metadata["nested"]["values"].append(4)
    assert stored.spec.metadata == {"nested": {"values": [1]}}
    assert stored.environment.spec.metadata == {"nested": {"values": [1]}}
    assert stored.factory_execution_profile_identity.behavior_version == "1"
    assert (inspected.registration_source, inspected.registration_symbol) == ("app.py", "app:build")
    assert owner.get_factory() is factory
    with pytest.raises(RuntimeError, match="factory-backed"):
        owner.get_concrete()
    with pytest.raises(TypeError):
        owner.registrations["other"] = stored


@pytest.mark.parametrize("kind", ["concrete", "factory"])
def test_environment_registry_rejects_closure_conflicts_before_publication(tmp_path, kind) -> None:
    owner = registry(
        session_closure_stores=(RetainedSessionClosureStore("artifact-store:blocked"),)
    )
    first = Environment(EnvironmentSpec(name="first"))
    owner.register(first, default=True)
    before = owner.session_closure
    view = owner.registrations

    def register(store_id):
        store = LocalArtifactStore(tmp_path / store_id, store_id=store_id)
        spec = EnvironmentSpec(name="second")
        if kind == "concrete":
            owner.register(Environment(spec, artifact_store=store), default=True)
        else:
            owner.register_factory(spec, UncalledFactory(), artifact_store=store, default=True)

    with pytest.raises(ValueError, match="store identity"):
        register("blocked")
    assert tuple(view) == ("first",)
    assert owner.default_name == "first"
    assert owner.session_closure is before
    assert owner.artifact_store_registration_fingerprints(limit=1) == ((), 0)
    register("accepted")
    assert tuple(view) == ("first", "second")
    assert owner.default_name == "second"
    assert owner.session_closure is not before
    assert owner.artifact_store_registration_count() == 1


@pytest.mark.parametrize("kind", ["concrete", "factory"])
def test_environment_registry_requires_explicit_default_and_deduplicates_stores(tmp_path, kind):
    owner = registry()
    store = LocalArtifactStore(tmp_path / "shared", store_id="shared")
    for name in ("second", "first"):
        spec = EnvironmentSpec(name=name)
        if kind == "concrete":
            owner.register(Environment(spec, artifact_store=store))
        else:
            owner.register_factory(spec, UncalledFactory(), artifact_store=store)
    assert owner.names() == ("first", "second")
    assert owner.get() is None
    assert owner.has_registered_artifact_store()
    assert owner.artifact_store_registration_count() == 1
    assert owner.get("first").environment.artifact_store is store


@pytest.mark.parametrize("kind", ["concrete", "factory"])
def test_environment_registration_keeps_public_callsite_provenance(kind) -> None:
    from cayu import CayuApp

    app = CayuApp(enable_logging=False)
    spec = EnvironmentSpec(name="local")
    if kind == "concrete":
        app.register_environment(Environment(spec))
    else:
        app.register_environment_factory(spec, UncalledFactory())
    registered = app.list_environment_registrations()[0]
    assert registered.registration_source.endswith("test_application_environment_registry.py")
    assert registered.registration_symbol.endswith(
        ":test_environment_registration_keeps_public_callsite_provenance"
    )
