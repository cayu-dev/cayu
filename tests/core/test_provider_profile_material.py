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
# The Anthropic entries changed deliberately when prompt caching became the
# default (AnthropicProvider().cache_policy is now CachePolicy(), #2289).
# Every streaming provider changed deliberately when the absolute stream deadline
# began counting from the last output (absolute output-idle policy v1, #2800).
_BASE_MATERIAL_SHA256 = {
    "anthropic": "5752745f4e9163a4c25e92d1a6807617e11d175535d5ba4e202ebbf28cee8359",
    "anthropic-tokens": "f0ab2706bce94d182651a849de81c9dd8628bb4df87281a7e2ace761a81d1d68",
    "bedrock": "d044630621b700d8cf9ccae560ecc8e7d7c775367838c1c51114474d0022c6e8",
    "bedrock-route": "3870ad81f70e1544041527d7b480d00484287480ec1b011e746f862fface6044",
    "chat": "3a152f2176015ef97e16dca84cf78da7a7b08dbc710bd73b7557464c2c74ad09",
    "openai": "6c2a30c9eefa61c650f0b986e28f85390e4ff432775b7844f6c132d450a05956",
    "openai-background": "49c21fb2d544ac9dc29466e452c40816fbbc194541dd3fb4689f7344a7c16634",
    "openai-search": "ec5ed69c8dff9d6d75086fe7ebef3996b3be17b2535a3b31dd26ce4a1893edaa",
    "openrouter": "0b651ea447a9f186c9ec420bbf92f139dc2053ab7ad86d67720fabeab221afff",
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
