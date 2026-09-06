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


# Fixed fingerprints including text progress policy version 2 (#1467).
_BASE_MATERIAL_SHA256 = {
    "anthropic": "12d1a8e612796c36b3c6d2278536c7e5c57f6461d339b5dd8e1fb8e5c83db7bf",
    "anthropic-tokens": "4dd622ebd3f943a406dd11b466d147fdc4ce39612ae0c514d8820a244de73189",
    "bedrock": "982928a290c5d0115bbdaf50f573c760c7599531fa3ecc00d79aa7bdbe317462",
    "bedrock-route": "a8b21551d1dbe3c46d97e4eae270ed5532ef978b89ab5c529ed2b99dd28a61c0",
    "chat": "82a4c51b5839bdc12027721ad7c7f8d3d459f0ce4b34619a01cedb5fb74b01d9",
    "openai": "a72a355846eaa76d769be8c5801a86c3ee18af4685f8df99146f1bedcb5efade",
    "openai-background": "88add28f1ba0e82fe2cabdf3dd1744bb0c0929bd3f4546131196518cb2ef18f8",
    "openai-search": "48b24a27d216779766b12f35fe4e0b663127058412e5780ac199dec27e05b8f5",
    "openrouter": "61af0ada9d0f2784461b9a9fa71d3f389735211dc515e9ba4c3f6955213c90da",
    "scripted": "cd8a343352e9c14f261c619ea4bfdb538bbfb31c8b9ea36856035a046d159c45",
    "scripted-background": "756612fad3d73854b2b4dcddb5bc60c7644af36dfb7965404ee64959f765267f",
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
