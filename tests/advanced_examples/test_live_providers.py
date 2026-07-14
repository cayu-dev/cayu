from __future__ import annotations

import sys
from pathlib import Path

import pytest
from examples._advanced_support import ScenarioResult, live_provider
from examples._advanced_support.cli import run_cli

from cayu import AnthropicProvider, ChatCompletionsProvider, OpenAIProvider


@pytest.mark.parametrize(
    ("provider_name", "key_name", "model_name", "expected_type", "expected_model"),
    [
        (
            "gemini",
            "GEMINI_API_KEY",
            "CAYU_GEMINI_MODEL",
            ChatCompletionsProvider,
            "gemini-3.1-flash-lite",
        ),
        ("openai", "OPENAI_API_KEY", "CAYU_OPENAI_MODEL", OpenAIProvider, "gpt-5.6-luna"),
        (
            "anthropic",
            "ANTHROPIC_API_KEY",
            "CAYU_ANTHROPIC_MODEL",
            AnthropicProvider,
            "claude-sonnet-4-6",
        ),
    ],
)
def test_live_provider_selects_supported_backend(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    key_name: str,
    model_name: str,
    expected_type: type,
    expected_model: str,
) -> None:
    monkeypatch.setenv(key_name, "test-key")
    monkeypatch.delenv(model_name, raising=False)

    provider, model = live_provider(provider_name)

    assert isinstance(provider, expected_type)
    assert model == expected_model


def test_live_provider_requires_selected_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        live_provider("openai")


def test_live_provider_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="gemini, openai, or anthropic"):
        live_provider("other")


def test_cli_forwards_explicit_live_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected: list[str | None] = []

    async def deterministic(root: Path) -> ScenarioResult:
        raise AssertionError(f"deterministic runner called for {root}")

    async def live(root: Path, provider_name: str | None) -> ScenarioResult:
        selected.append(provider_name)
        return ScenarioResult(
            scenario="provider-routing",
            mode="live",
            status="verified",
            assertions={"provider_selected": True},
            sessions=[],
        )

    monkeypatch.setattr(
        sys,
        "argv",
        ["advanced-example", "--mode", "live", "--provider", "anthropic", "--root", str(tmp_path)],
    )

    run_cli(deterministic=deterministic, live=live)

    assert selected == ["anthropic"]
