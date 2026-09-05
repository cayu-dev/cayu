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


# Captured from b4662938727f16f5a0eb1d06aa866fd96bc8e6ba before relocating extraction.
_BASE_MATERIAL_SHA256 = {
    "anthropic": "66779e94aa8c23e7e338d8606c07807783065aaa906506635de85b4175f3b933",
    "anthropic-tokens": "79ef682397846f23bd931266bc212ca446329e42b379e01a8ce13a3b5324e58d",
    "bedrock": "c8823786844bd231d594f1a9f7c72872c2d6f53081b3b6ff9cb0717db652e705",
    "bedrock-route": "369f610698653deaf94cb35517035e5f7e856ca4bf0ca4dcd34385d85cfb15c0",
    "chat": "d8a1bd749c4e56f7a89db2d13588857468a6f1b0d7fedbc3c448d4caf8e4ae6a",
    "openai": "5a36c821a9b59506f35a823af1914647012b81b0eded69520187c4978cb59245",
    "openai-background": "1fcf694e44bbdb6b9ae99ff30bacd83b73aacd75ed899c32c86818e5477bff6a",
    "openai-search": "88a55fbc695499330ef737c6c9ced436064673969f8053eab35a7afe2d2948d5",
    "openrouter": "2efc346b93bc93531a317b6e8741fa13d69b457bd19353e96c6be98ea15aae46",
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
