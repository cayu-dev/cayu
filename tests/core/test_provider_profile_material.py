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


# Fixed fingerprints include text progress v2 and semantic cleanup policy v1.
_BASE_MATERIAL_SHA256 = {
    "anthropic": "e80fda7cb981ae133375342f5f3eedb1451ca9ff4000682e6f15ab69e34b57d6",
    "anthropic-tokens": "c1b216be87df40fc0076b8f550151e2bfdcd7a314e6392a678ddae58dc9dfc90",
    "bedrock": "b7ad27364266b326041261b23cedaa17b48634ed744b26149fb27f882e8749d0",
    "bedrock-route": "11bed9d201d2e9b326ffe8ca4eee658be51e11a18c23e1badf5a464652043f92",
    "chat": "5e23079ace357b25bcbe6a13a667be063897356a7bb3c051ed9d67e6aae4b95f",
    "openai": "2db00d154db1b256468a722b21b1330b882886bd7545f6272e95362d7be0a261",
    "openai-background": "ea67a1b1b3328d3cc253f1917b0633e0ada145fba3d0bb9f3e77910a241607c7",
    "openai-search": "af24f1fe667c3bf9e79747be2063339d056063decdf56beab48f8050714960ce",
    "openrouter": "e1a8f89bc858116512cfdecac43151e76a7bc16fb5b0678b44d340233fa7ca5c",
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
