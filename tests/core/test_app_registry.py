"""Tests for CayuApp registry introspection (list_agents/providers/environments)."""

from __future__ import annotations

import hashlib

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    ScriptedModelProvider,
)
from cayu.artifacts import LocalArtifactStore


class _UncalledEnvironmentFactory(EnvironmentFactory):
    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        del request
        raise AssertionError("registry introspection must not materialize environment factories")


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


@pytest.mark.parametrize("first_kind", ["concrete", "factory"])
@pytest.mark.parametrize("conflicting_kind", ["concrete", "factory"])
def test_cayu_app_rejects_conflicting_artifact_store_ids_atomically(
    first_kind: str,
    conflicting_kind: str,
    tmp_path,
) -> None:
    app = CayuApp()
    first_store = LocalArtifactStore(tmp_path / "first", store_id="shared-id")
    conflicting_store = LocalArtifactStore(tmp_path / "second", store_id="shared-id")

    def register(
        kind: str,
        name: str,
        store: LocalArtifactStore,
        *,
        default: bool,
    ) -> _UncalledEnvironmentFactory | None:
        if kind == "concrete":
            app.register_environment(
                Environment(EnvironmentSpec(name=name), artifact_store=store),
                default=default,
            )
            return None
        factory = _UncalledEnvironmentFactory()
        app.register_environment_factory(
            EnvironmentSpec(name=name),
            factory,
            artifact_store=store,
            default=default,
        )
        return factory

    first_factory = register(first_kind, "first", first_store, default=True)
    expected_fingerprint = f"sha256:{hashlib.sha256(b'shared-id').hexdigest()}"
    assert app.artifact_store_registration_fingerprints(limit=64) == (
        (expected_fingerprint,),
        1,
    )

    with pytest.raises(ValueError, match="different registered store: shared-id"):
        register(conflicting_kind, "conflicting", conflicting_store, default=True)

    assert app.list_environments() == ("first",)
    assert app.has_registered_artifact_store() is True
    assert app.artifact_store_registration_fingerprints(limit=64) == (
        (expected_fingerprint,),
        1,
    )
    if first_kind == "concrete":
        assert app.get_environment().spec.name == "first"
    else:
        assert app.get_environment_factory() is first_factory

    register("concrete", "shared", first_store, default=False)
    assert app.list_environments() == ("first", "shared")
    assert app.artifact_store_registration_fingerprints(limit=64) == (
        (expected_fingerprint,),
        1,
    )


def test_cayu_app_allows_one_artifact_store_instance_in_multiple_environments(tmp_path) -> None:
    app = CayuApp()
    shared_store = LocalArtifactStore(tmp_path / "shared", store_id="shared-id")
    app.register_environment(
        Environment(EnvironmentSpec(name="concrete"), artifact_store=shared_store),
        default=True,
    )
    factory = _UncalledEnvironmentFactory()

    app.register_environment_factory(
        EnvironmentSpec(name="factory"),
        factory,
        artifact_store=shared_store,
    )

    assert app.list_environments() == ("concrete", "factory")
    assert app.has_registered_artifact_store() is True
    assert app.artifact_store_registration_fingerprints(limit=1) == (
        (f"sha256:{hashlib.sha256(b'shared-id').hexdigest()}",),
        1,
    )


def test_cayu_app_bounds_artifact_store_registration_fingerprint_snapshots(tmp_path) -> None:
    app = CayuApp()
    for index in range(3):
        app.register_environment(
            Environment(
                EnvironmentSpec(name=f"environment-{index}"),
                artifact_store=LocalArtifactStore(
                    tmp_path / f"store-{index}",
                    store_id=f"store-{index}",
                ),
            )
        )

    fingerprints, total_count = app.artifact_store_registration_fingerprints(limit=2)

    assert fingerprints == tuple(
        f"sha256:{hashlib.sha256(f'store-{index}'.encode()).hexdigest()}" for index in range(2)
    )
    assert total_count == 3
    with pytest.raises(TypeError, match="must be an integer"):
        app.artifact_store_registration_fingerprints(limit=True)
    with pytest.raises(ValueError, match="must be positive"):
        app.artifact_store_registration_fingerprints(limit=0)
