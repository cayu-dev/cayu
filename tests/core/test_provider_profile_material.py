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
# and semantic cleanup policy v1.
_BASE_MATERIAL_SHA256 = {
    "anthropic": "c4292274f58f832f7f29291c9dbf972e843488bf0e682ad0e3b801f63f4c782d",
    "anthropic-tokens": "1175e0534936e52740a06703a041d22d23d2927850efa811843e57633da2e2d7",
    "bedrock": "bc83166b1ce8ee51eeba8ae2d06581c48fcb7fe570b794ad9da1f8c678a2e888",
    "bedrock-route": "239ae134fa3556d80111abc93903136c219b9f7c20086fbaa3db08449a142592",
    "chat": "ef06fc7b45743da89ce0afb48035b7462f635d9fbe822ce46f1eb2077a9f5d33",
    "openai": "91f0102224946afe1b65b1e1c5f1bfa8852f7958b73c07afe74e804671bd2a82",
    "openai-background": "12d4c2badb4988320ae573ffd75f602ffb46a1fe444b82901966b28789cd9b4a",
    "openai-search": "902b8510f3b60f4d5b22c831fe4d592656390b7f0c64146134a6551a554a7014",
    "openrouter": "238bdd76b4e6cbead2f89ed3462935c660a7f8445dd439acc34f267ff5cb7312",
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
