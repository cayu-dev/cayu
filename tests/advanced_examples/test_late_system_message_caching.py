from __future__ import annotations

import asyncio
import json

import pytest
from examples.late_system_message_caching.deterministic import run
from examples.late_system_message_caching.live import run as live_run


def test_deterministic_late_system_caching_is_verified(tmp_path) -> None:
    result = asyncio.run(run(tmp_path))

    assert result.status == "verified"
    assert all(result.assertions.values()), result.assertions
    written = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert written["metrics"]["token_counts"].startswith("simulated")
    for provider in ("openai", "anthropic"):
        late = written["metrics"]["providers"][provider]["late_system"]
        assert [turn["turn"] for turn in late["turns"]] == [1, 2, 3, 4, 5, 6]
        assert late["turns"][0]["cache_read_tokens"] == 0
        assert late["turns_2_plus_cached_share"] >= 0.9
        assert late["cost_usd"] is not None


def test_live_mode_requires_a_supported_provider_and_key(tmp_path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="--provider openai or anthropic"):
        asyncio.run(live_run(tmp_path, "gemini"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        asyncio.run(live_run(tmp_path, "anthropic"))
