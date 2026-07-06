"""Provider construction errors must name the env var (and the deferred path)."""

from __future__ import annotations

from typing import Any

import pytest

from cayu import AnthropicProvider, ChatCompletionsProvider, OpenAIProvider


def test_openai_error_names_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIProvider()


def test_openai_rejects_non_string_api_key() -> None:
    api_key: Any = 123
    with pytest.raises(TypeError, match="api_key must be a string"):
        OpenAIProvider(api_key=api_key)


def test_anthropic_error_names_env_var_and_deferred_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()
    with pytest.raises(ValueError, match="api_key_ref"):
        AnthropicProvider()


def test_anthropic_rejects_non_string_api_key() -> None:
    api_key: Any = 123
    with pytest.raises(TypeError, match="api_key must be a string"):
        AnthropicProvider(api_key=api_key)


def test_chat_completions_error_names_configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        ChatCompletionsProvider()
    monkeypatch.delenv("CUSTOM_LLM_KEY", raising=False)
    with pytest.raises(ValueError, match="CUSTOM_LLM_KEY"):
        ChatCompletionsProvider(api_key_env="CUSTOM_LLM_KEY")


def test_chat_completions_rejects_non_string_api_key() -> None:
    api_key: Any = 123
    with pytest.raises(TypeError, match="api_key must be a string"):
        ChatCompletionsProvider(api_key=api_key)
