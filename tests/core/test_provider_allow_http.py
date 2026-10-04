"""OpenAI and Anthropic accept plain HTTP endpoints only when asked to."""

from __future__ import annotations

import pytest

from cayu.providers.anthropic import (
    AnthropicProvider,
    HttpxAnthropicTransport,
)
from cayu.providers.anthropic import (
    _execution_profile_material as anthropic_profile_material,
)
from cayu.providers.openai import HttpxOpenAITransport, OpenAIProvider
from cayu.providers.openai import _execution_profile_material as openai_profile_material

_LOCAL = "http://127.0.0.1:8080/v1"


@pytest.mark.parametrize(
    ("factory", "label"),
    [
        (lambda **kw: OpenAIProvider(api_key="test-key", **kw), "OpenAI"),
        (lambda **kw: AnthropicProvider(api_key="test-key", **kw), "Anthropic"),
    ],
    ids=("openai", "anthropic"),
)
def test_plain_http_base_url_requires_allow_http(factory, label: str) -> None:
    with pytest.raises(ValueError, match=rf"{label} base_url must use https \(set allow_http"):
        factory(base_url=_LOCAL)

    provider = factory(base_url=_LOCAL, allow_http=True)

    assert provider.base_url == _LOCAL.rstrip("/")
    assert provider.allow_http is True
    # The default transport inherits the opt-in so the local endpoint connects.
    assert provider.transport.allow_http is True


@pytest.mark.parametrize("value", [1, "yes", None])
def test_allow_http_must_be_a_bool(value) -> None:
    with pytest.raises(TypeError, match="allow_http must be a bool"):
        OpenAIProvider(api_key="test-key", allow_http=value)
    with pytest.raises(TypeError, match="allow_http must be a bool"):
        AnthropicProvider(api_key="test-key", allow_http=value)
    with pytest.raises(TypeError, match="allow_http must be a bool"):
        HttpxOpenAITransport(allow_http=value)
    with pytest.raises(TypeError, match="allow_http must be a bool"):
        HttpxAnthropicTransport(allow_http=value)


def test_allow_http_is_part_of_execution_profile_material_only_when_set() -> None:
    assert "allow_http" not in openai_profile_material(OpenAIProvider(api_key="test-key"))
    assert "allow_http" not in anthropic_profile_material(AnthropicProvider(api_key="test-key"))

    openai = OpenAIProvider(api_key="test-key", base_url=_LOCAL, allow_http=True)
    anthropic = AnthropicProvider(api_key="test-key", base_url=_LOCAL, allow_http=True)

    assert openai_profile_material(openai)["allow_http"] is True
    assert anthropic_profile_material(anthropic)["allow_http"] is True


def test_https_endpoints_work_with_allow_http_set() -> None:
    provider = OpenAIProvider(
        api_key="test-key", base_url="https://gateway.example/v1", allow_http=True
    )

    assert provider.base_url == "https://gateway.example/v1"
