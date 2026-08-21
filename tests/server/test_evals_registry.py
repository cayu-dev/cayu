from __future__ import annotations

from pathlib import Path

import pytest
from tests.evals.test_corpus_execution import _provider, _target

from cayu import AgentSpec
from cayu.evals.execution import evaluation_target_identity
from cayu.server.evals_registry import (
    DEFAULT_EVAL_PROFILE_ID,
    EvalTargetRegistry,
    derive_eval_target_key,
    explicit_eval_target_registry,
    generated_eval_target_registry,
)


def _generated_registry(*, release_id: str = "release-one") -> EvalTargetRegistry:
    target = _target(_provider())
    app = target.app
    app.register_agent(AgentSpec(name="beta", model="fixture-model"))
    manifest = app.describe()
    registry = generated_eval_target_registry(
        app,
        project_id="registry-project",
        application_release_id=release_id,
        app_manifest_fingerprint=manifest.fingerprint,
    )
    assert registry is not None
    return registry


def test_generated_target_keys_are_stable_unambiguous_and_release_independent() -> None:
    baseline = derive_eval_target_key(
        project_id="project",
        agent_name="agent",
        profile_id="default",
    )

    assert baseline == derive_eval_target_key(
        project_id="project",
        agent_name="agent",
        profile_id="default",
    )
    assert baseline == "eval.b1cbe3c1b491a3bd35c3c259200cca26d82c526bb6ebde8a248ccb834af80dbc"
    assert len(baseline) == 69
    assert (
        len(
            {
                baseline,
                derive_eval_target_key(
                    project_id="project-two",
                    agent_name="agent",
                    profile_id="default",
                ),
                derive_eval_target_key(
                    project_id="project",
                    agent_name="agent-two",
                    profile_id="default",
                ),
                derive_eval_target_key(
                    project_id="project",
                    agent_name="agent",
                    profile_id="isolated",
                ),
                derive_eval_target_key(
                    project_id="ab",
                    agent_name="c",
                    profile_id="default",
                ),
                derive_eval_target_key(
                    project_id="a",
                    agent_name="bc",
                    profile_id="default",
                ),
            }
        )
        == 6
    )


def test_generated_registry_maps_each_agent_to_normal_authority_without_serializing_app() -> None:
    first = _generated_registry(release_id="release-one")
    second = _generated_registry(release_id="release-two")

    assert first.target_keys == second.target_keys
    assert len(first.target_keys) == 2
    catalog = first.catalog()
    assert [entry.agent_name for entry in catalog.items] == ["agent", "beta"]
    assert {entry.profile_id for entry in catalog.items} == {DEFAULT_EVAL_PROFILE_ID}
    assert {entry.project_id for entry in catalog.items} == {"registry-project"}
    assert {entry.application_release_id for entry in catalog.items} == {"release-one"}
    assert catalog.default_target_key == first.target_keys[0]
    assert "CayuApp" not in catalog.model_dump_json()
    assert repr(first) == "EvalTargetRegistry(target_count=2)"

    for entry in catalog.items:
        runtime_target = first.get(entry.target_key)
        assert runtime_target is not None
        assert runtime_target.request_base.agent_name == entry.agent_name
        assert runtime_target.request_base.messages == []
        assert runtime_target.application_release_id == "release-one"


def test_generated_registry_preserves_project_root_manifest_provenance() -> None:
    target = _target(_provider())
    project_root = Path(__file__).resolve().parents[2]
    rooted_manifest = target.app.describe(project_root=project_root)
    unrooted_manifest = target.app.describe()

    assert rooted_manifest.fingerprint != unrooted_manifest.fingerprint
    with pytest.raises(ValueError, match="catalog manifest"):
        generated_eval_target_registry(
            target.app,
            project_id="registry-project",
            application_release_id="release-one",
            app_manifest_fingerprint=unrooted_manifest.fingerprint,
            app_manifest_project_root=project_root,
        )

    registry = generated_eval_target_registry(
        target.app,
        project_id="registry-project",
        application_release_id="release-one",
        app_manifest_fingerprint=rooted_manifest.fingerprint,
        app_manifest_project_root=project_root,
    )
    assert registry is not None
    entry = registry.catalog().items[0]
    registration = registry.registration(entry.target_key)
    assert registration is not None
    runtime_identity = evaluation_target_identity(
        registration.target,
        project_root=registration.manifest_project_root,
    )

    assert entry.app_manifest_fingerprint == rooted_manifest.fingerprint
    assert runtime_identity.app_manifest == rooted_manifest


def test_generated_registry_rejects_hash_collisions(monkeypatch) -> None:
    monkeypatch.setattr(
        "cayu.server.evals_registry.derive_eval_target_key",
        lambda **_kwargs: "eval." + "0" * 64,
    )

    with pytest.raises(ValueError, match="key collision"):
        _generated_registry()


def test_explicit_v1_target_is_adapted_without_changing_its_authority() -> None:
    target = _target(_provider())
    registry = explicit_eval_target_registry(target)
    entry = registry.catalog().items[0]

    assert registry.target_keys == (target.key,)
    assert registry.get(target.key) is target
    assert entry.target_key == target.key
    assert entry.agent_name == target.request_base.agent_name
    assert entry.profile_id == "explicit"
    assert entry.project_id is None
    assert entry.source == "explicit"
