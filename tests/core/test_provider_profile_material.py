from __future__ import annotations

import hashlib
import json

import pytest

from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.anthropic import AnthropicProvider
from cayu.providers.bedrock import BedrockProvider
from cayu.providers.chat_completions import ChatCompletionsProvider
from cayu.providers.openai import OpenAIProvider
from cayu.runtime._execution_profile_admission import _cayu_provider_material


def cases():
    return {
        "openai": OpenAIProvider(api_key="test-key"),
        "openai-background": OpenAIProvider(api_key="test-key", background=True),
        "openai-search": OpenAIProvider(
            api_key="test-key", hosted_tool_search_models=("test-model",)
        ),
        "chat": ChatCompletionsProvider(api_key="test-key"),
        "openrouter": ChatCompletionsProvider(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            openrouter_http_referer="https://test.example",
            openrouter_app_title="test",
            openrouter_router_metadata=True,
        ),
        "anthropic": AnthropicProvider(api_key="test-key"),
        "anthropic-tokens": AnthropicProvider(api_key="test-key", max_tokens=8192),
        "bedrock": BedrockProvider(region_name="us-east-1"),
        "bedrock-route": BedrockProvider(
            region_name="us-west-2", endpoint_url="https://bedrock.test.example"
        ),
        "scripted": ScriptedModelProvider([]),
        "scripted-background": ScriptedModelProvider([], background=True),
    }


# Fixed fingerprints include 300-second idle defaults, text progress v2,
# semantic cleanup policy v1, and HTTP cleanup observation policy v1.
# Captured against current main before checking the concept migration.
_BASE_MATERIAL_SHA256 = {
    "anthropic": "cc868ed1d23c846c0ca4eb2662b46fdea1ef8388fa1e9805b03e750fdcc992b8",
    "anthropic-tokens": "725b7e11c5a592e9e53ece501a973e4d7c2cb4eb2badcef88651234f75de5ac8",
    "bedrock": "369bf7120cf10c9212db958362187d33b99808531a11d3c564b97d27ae902390",
    "bedrock-route": "3527dc6dba95186eb5a71bc8e76b400945959c4aa5e937d6ada8bb757732ab85",
    "chat": "d50e4dab344cc06426827e645bf50e85bdb39ca8b3c99d8930c7178b8db029ef",
    "openai": "6eb9e24f4030fdaafd474a737c8626a3242a24ce81216bceaa48fafb84764f27",
    "openai-background": "23a3d61609b86ddd06a8e26240b43a3ddbd52ec7b1a235ea92d78def886f21db",
    "openai-search": "6a35bfd929b5e74d1923e92bf9b08d1952d1158e9b1174d8c14c6b7624c0943b",
    "openrouter": "a793fc4ae759234dbd792c85bc6a3e6131f3ddfec706298d251d9d824f58fff5",
    "scripted": "06548d202a0bac8275f5b0440ef995d442f547383da12717e99a84d23d30e803",
    "scripted-background": "019d2934ee5f333b455cd092d2adb2ca99d8c7d399253444befad745921ada5a",
}


@pytest.mark.parametrize("name", tuple(_BASE_MATERIAL_SHA256))
def test_builtin_provider_material_preserves_existing_fingerprint_input(name):
    material = _cayu_provider_material(cases()[name])
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == _BASE_MATERIAL_SHA256[name]


@pytest.mark.parametrize(
    "baseline,changed",
    [
        ("openai", "openai-background"),
        ("openai", "openai-search"),
        ("chat", "openrouter"),
        ("anthropic", "anthropic-tokens"),
        ("bedrock", "bedrock-route"),
        ("scripted", "scripted-background"),
    ],
)
def test_behavior_changes_invalidate_builtin_provider_material(baseline, changed):
    providers = cases()
    assert _cayu_provider_material(providers[baseline]) != _cayu_provider_material(
        providers[changed]
    )


def test_provider_subclasses_and_opaque_transports_cannot_claim_builtin_identity():
    class CustomProvider(OpenAIProvider):
        def _execution_profile_material(self):
            raise AssertionError("Untrusted provider hook was invoked")

    assert _cayu_provider_material(CustomProvider(api_key="test-key")) is None
    assert _cayu_provider_material(OpenAIProvider(api_key="test-key", transport=object())) is None
    assert (
        _cayu_provider_material(AnthropicProvider(api_key="test-key", transport=object())) is None
    )
    assert (
        _cayu_provider_material(ChatCompletionsProvider(api_key="test-key", transport=object()))
        is None
    )
    assert (
        _cayu_provider_material(BedrockProvider(client=object(), region_name="us-east-1")) is None
    )
    assert (
        _cayu_provider_material(
            OpenAIProvider(api_key="test-key", extra_headers={"x-private": "secret"})
        )
        is None
    )
