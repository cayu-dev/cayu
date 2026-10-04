"""Provider ownership preserves declarations, routing and isolated replay."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest

import cayu
from cayu._application_provider_registry import ApplicationProviderRegistry
from cayu.applications import CayuApp
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent, UsageDialect
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.vaults.redaction import SecretRedactor


class DeclaredProvider(ModelProvider):
    def __init__(self, name: str) -> None:
        self.name = name
        self.identity = ExecutionProfileBehaviorIdentity(
            name=name, behavior_version="1", implementation_version="1"
        )

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return self.identity

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        raise AssertionError("Registration and inspection must not dispatch the provider")
        yield  # pragma: no cover


def test_provider_registry_composes_without_application_controllers() -> None:
    script = """
import importlib.abc
import sys
blocked = {
    "cayu.applications", "cayu.runtime._session_engine",
    "cayu.runtime._model_step_executor", "cayu.runtime._recovery_coordinator",
}
class RejectControllers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Registry imported {fullname}")
sys.meta_path.insert(0, RejectControllers())
from cayu._application_provider_registry import ApplicationProviderRegistry
from cayu.providers.base import ModelProvider
from cayu.vaults.redaction import SecretRedactor
class Provider(ModelProvider):
    name = "sample"
    async def stream(self, request):
        raise AssertionError("Unexpected dispatch")
        yield
registry = ApplicationProviderRegistry(secret_redactor=SecretRedactor())
provider = Provider()
assert registry.register(provider, model_patterns=["sample-*"]) is provider
assert registry.get().provider is provider
assert registry.route(model="sample-model").provider is provider
assert registry.names() == ("sample",)
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


def test_provider_registration_snapshots_declarations_and_keeps_live_provider() -> None:
    registry = ApplicationProviderRegistry(secret_redactor=SecretRedactor())
    provider = DeclaredProvider("first")
    provider.usage_dialect = UsageDialect.OPENAI
    patterns = ["model-*", "model-exact"]
    assert (
        registry.register(
            provider, model_patterns=patterns, registration_site=("app.py", "app:build")
        )
        is provider
    )
    record = registry.get()
    assert record.execution_profile_identity == provider.identity
    assert record.execution_profile_identity is not provider.identity
    provider.identity = provider.identity.model_copy(update={"behavior_version": "2"})
    provider.usage_dialect = UsageDialect.GENERIC
    patterns.append("other-*")
    provider.name = "changed"
    assert record.name == "first"
    assert record.provider is provider
    assert record.execution_profile_identity.behavior_version == "1"
    assert record.usage_dialect is UsageDialect.OPENAI
    assert record.model_patterns == ("model-*", "model-exact")
    assert (record.registration_source, record.registration_symbol) == ("app.py", "app:build")
    assert registry.route(model="other-model") is None
    assert registry.route(model="model-exact") is record
    with pytest.raises(TypeError):
        registry.registrations["new"] = record  # type: ignore[index]


@pytest.mark.parametrize("failure", ["duplicate", "secret", "dialect", "patterns", "default"])
def test_rejected_provider_registration_does_not_publish_or_change_default(failure: str) -> None:
    registry = ApplicationProviderRegistry(secret_redactor=SecretRedactor("private-token"))
    first = DeclaredProvider("first")
    registry.register(first)
    provider = DeclaredProvider("candidate")
    default = True
    patterns = ["candidate-*"]
    if failure == "duplicate":
        provider.name = "first"
    elif failure == "secret":
        provider.identity = provider.identity.model_copy(update={"name": "private-token"})
    elif failure == "dialect":
        provider.usage_dialect = "invalid"  # type: ignore[assignment]
    elif failure == "patterns":
        patterns = [" "]
    else:
        default = 1  # type: ignore[assignment]
    with pytest.raises((ValueError, TypeError)):
        registry.register(provider, default=default, model_patterns=patterns)
    assert registry.names() == ("first",)
    assert registry.default_name == "first"
    assert registry.get().provider is first
    assert registry.route(model="candidate-model") is None


def test_provider_registry_preserves_defaults_and_ambiguity_order() -> None:
    registry = ApplicationProviderRegistry(secret_redactor=SecretRedactor())
    with pytest.raises(RuntimeError, match="No model provider registered"):
        registry.get()
    first, second, third = (DeclaredProvider(name) for name in ("z-first", "a-second", "third"))
    registry.register(first, model_patterns=["model-*", "model-exact"])
    registry.register(second, model_patterns=["model-e*"])
    assert registry.get().provider is first
    assert registry.names() == ("a-second", "z-first")
    assert [record.provider for record in registry.matching(model="model-exact")] == [first, second]
    with pytest.raises(ValueError, match="model-exact -> z-first, a-second"):
        registry.route(model="model-exact")
    assert registry.route(model="MODEL-exact") is None
    assert registry.get("a-second").provider is second
    registry.register(third, default=True)
    assert registry.get().provider is third
    with pytest.raises(KeyError, match="Provider not registered: missing"):
        registry.get("missing")
    with pytest.raises(ValueError):
        registry.route(model=" ")


def test_replay_install_preserves_admitted_records_and_owns_its_mapping() -> None:
    source = ApplicationProviderRegistry(secret_redactor=SecretRedactor())
    original = DeclaredProvider("original")
    source.register(original, model_patterns=["model-*"], registration_site=("app.py", "build"))
    admitted = source.get()
    recorded = DeclaredProvider("recorded-wrapper")
    declarations = {"original": replace(admitted, provider=recorded)}
    replay = ApplicationProviderRegistry(secret_redactor=SecretRedactor())
    replay.replace_for_replay(declarations, default_name="original")
    declarations.clear()
    source.register(DeclaredProvider("later"), default=True)
    assert replay.names() == ("original",)
    assert replay.default_name == "original"
    assert replay.get().provider is recorded
    assert replay.get().execution_profile_identity is admitted.execution_profile_identity
    assert replay.get().registration_source == "app.py"
    assert replay.route(model="model-exact") is replay.get()
    assert source.get("original").provider is original
    with pytest.raises(ValueError, match="Replay default provider not registered: missing"):
        replay.replace_for_replay({}, default_name="missing")
    assert replay.get().provider is recorded


def test_application_provider_registration_keeps_public_provenance() -> None:
    app = CayuApp(enable_logging=False)
    provider = DeclaredProvider("public")
    assert app.register_provider(provider) is provider
    record = app._provider_registry.get()
    assert record.registration_source == __file__
    assert (
        record.registration_symbol
        == f"{__name__}:test_application_provider_registration_keeps_public_provenance"
    )
    assert app.list_providers() == ("public",)
    assert app.get_provider() is provider
    assert not hasattr(app, "_providers")
    assert not hasattr(app, "_default_provider_name")
