from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Event,
    EventType,
    InMemorySessionStore,
    ModelCatalog,
    ModelInfo,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    PriceTier,
    ToolContext,
    default_model_catalog,
)
from maintenance.model_catalog import agent_tools as catalog_tools
from maintenance.model_catalog import browser, refresh, search
from maintenance.model_catalog.agent_tools import ReadPageTool
from maintenance.model_catalog.browser_verifier import (
    BrowserVerifier,
    browsed_pricing_modes,
    browsed_urls,
    verifier_model_info,
)
from maintenance.model_catalog.diff import catalog_diff, format_catalog_diff, markdown_code_span
from maintenance.model_catalog.guidance import WORKSPACE_GUIDANCE
from maintenance.model_catalog.policy import (
    CATALOG_MAX_AGE_DAYS,
    VERIFY_MAX_AGE_DAYS,
    VERIFY_SCHEDULE_DAYS,
    model_policy_errors,
    suspicious_price_changes,
    validate_catalog,
)
from maintenance.model_catalog.security import PROVIDER_METADATA_KEY, validate_official_url
from maintenance.model_catalog.verified import VERIFIED_SCHEMA, _dec, parse_verified
from maintenance.model_catalog.verify import RecommendationOutcome, VerifyOutcome


def _provider_model(provider_name: str) -> ModelInfo:
    return next(
        model for model in default_model_catalog().models if model.provider_name == provider_name
    )


_OPENAI_PAGE = "https://developers.openai.com/api/pricing"
_OPENAI_HOSTS = frozenset({"developers.openai.com", "openai.com", "platform.openai.com"})


def _browser_context(provider_name: str = "openai") -> ToolContext:
    return ToolContext(
        session_id="verification",
        metadata={PROVIDER_METADATA_KEY: provider_name},
    )


def test_default_catalog_satisfies_release_policy() -> None:
    catalog = default_model_catalog()
    validate_catalog(catalog, today=date.fromisoformat(catalog.generated_at))


def test_release_policy_rejects_stale_provenance() -> None:
    catalog = default_model_catalog()
    oldest = min(date.fromisoformat(model.provenance.as_of) for model in catalog.models)
    stale_day = max(
        date.fromisoformat(catalog.generated_at),
        oldest + timedelta(days=31),
    )
    with pytest.raises(ValueError, match="provenance is stale"):
        validate_catalog(catalog, today=stale_day)


def test_release_policy_rejects_non_official_provider_host() -> None:
    catalog = default_model_catalog()
    model = catalog.models[0]
    bad_model = model.model_copy(
        update={
            "provenance": model.provenance.model_copy(update={"url": "https://example.com/pricing"})
        }
    )
    candidate = catalog.model_copy(update={"models": (bad_model, *catalog.models[1:])})

    with pytest.raises(ValueError, match="allowed official host"):
        validate_catalog(candidate, today=date.fromisoformat(catalog.generated_at))


def test_release_policy_rejects_provider_owned_but_unapproved_subdomain() -> None:
    catalog = default_model_catalog()
    index, model = next(
        (index, model)
        for index, model in enumerate(catalog.models)
        if model.provider_name == "vertex"
    )
    bad_model = model.model_copy(
        update={
            "provenance": model.provenance.model_copy(
                update={"url": "https://sites.google.com/view/model-pricing"}
            )
        }
    )
    models = list(catalog.models)
    models[index] = bad_model

    with pytest.raises(ValueError, match="allowed official host"):
        validate_catalog(
            catalog.model_copy(update={"models": tuple(models)}),
            today=date.fromisoformat(catalog.generated_at),
        )


def test_release_policy_enforces_narrow_bundled_match_rules() -> None:
    model = default_model_catalog().models[0]
    broad = model.model_copy(update={"match": "prefix"})
    unrelated = model.model_copy(update={"match_prefixes": ("other-model-20",)})
    unsafe_delimiter = model.model_copy(update={"match_prefixes": (f"{model.model}x20",)})

    assert any(
        "canonical model match must be 'exact'" in error
        for error in model_policy_errors(broad, today=date(2026, 7, 14), max_age_days=None)
    )
    for candidate in (unrelated, unsafe_delimiter):
        assert any(
            "must strictly extend the canonical model" in error
            for error in model_policy_errors(candidate, today=date(2026, 7, 14), max_age_days=None)
        )


def test_verification_schedule_leaves_release_freshness_margin() -> None:
    assert VERIFY_MAX_AGE_DAYS == 14
    assert VERIFY_MAX_AGE_DAYS + VERIFY_SCHEDULE_DAYS < CATALOG_MAX_AGE_DAYS


def test_catalog_diff_reports_field_changes_without_another_schema() -> None:
    catalog = default_model_catalog()
    model = catalog.models[0]
    changed_model = model.model_copy(update={"context_window": (model.context_window or 0) + 1})
    candidate = ModelCatalog(
        catalog_version="2026-07-14",
        generated_at="2026-07-14",
        models=(changed_model, *catalog.models[1:]),
    )

    diff = catalog_diff(catalog, candidate)
    assert diff["added"] == []
    assert diff["removed"] == []
    assert diff["changed"] == [
        {
            "model": f"{model.provider_name}/{model.model}",
            "fields": {
                "context_window": {
                    "before": model.context_window,
                    "after": (model.context_window or 0) + 1,
                }
            },
        }
    ]
    report = format_catalog_diff(catalog, candidate)
    assert "1 changed" in report
    assert f"`{model.provider_name}/{model.model}`" in report


def test_catalog_diff_uses_non_breakable_code_spans() -> None:
    assert markdown_code_span('"value with ` delimiter"') == '`` "value with ` delimiter" ``'
    assert markdown_code_span("") == "<code></code>"


def test_verifier_schema_is_strict_and_covers_rich_catalog_pricing() -> None:
    assert VERIFIED_SCHEMA["additionalProperties"] is False
    assert set(VERIFIED_SCHEMA["required"]) == set(VERIFIED_SCHEMA["properties"])
    assert VERIFIED_SCHEMA["properties"]["context_window"]["minimum"] == 1
    tier_schema = VERIFIED_SCHEMA["properties"]["context_tiers"]["items"]
    assert tier_schema["properties"]["up_to_tokens"]["minimum"] == 1
    assert {
        "batch_input_per_million",
        "batch_output_per_million",
        "cache_write_5m_per_million",
        "cache_write_1h_per_million",
        "context_tiers",
        "evidence",
    } <= set(VERIFIED_SCHEMA["properties"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (".75", Decimal("0.75")),
        ("$.075", Decimal("0.075")),
        ("5.00 USD", Decimal("5.00")),
        (True, None),
        ("3.50-4.00", None),
        ("3.50 to 4.00", None),
    ],
)
def test_verified_price_parser_handles_leading_decimals_and_rejects_ambiguity(
    value, expected
) -> None:
    assert _dec(value) == expected


def test_search_results_decode_ddg_target_exactly_once() -> None:
    html = '<a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fa%252Fb">result</a>'

    assert search.parse_results(html) == ["https://example.com/a%2Fb"]


def test_verified_official_data_updates_the_canonical_model_info() -> None:
    original = _provider_model("anthropic")

    outcome = parse_verified(
        {
            "confirmed": True,
            "context_window": 1_000_001,
            "input_per_million": "$3.25",
            "output_per_million": "16 USD",
            "cache_read_per_million": "0.325",
            "cache_write_5m_per_million": "4.0625",
            "cache_write_1h_per_million": "6.5",
            "batch_input_per_million": "1.625",
            "batch_output_per_million": "8",
            "deprecated": False,
            "source_url": "https://platform.claude.com/docs/pricing",
            "evidence": "Claude Sonnet pricing row",
        },
        original,
        as_of="2026-07-13",
        browsed_urls={"https://platform.claude.com/docs/pricing"},
        browsed_pricing_modes={"https://platform.claude.com/docs/pricing": {"standard", "batch"}},
    )

    assert outcome.verified is True
    assert outcome.evidence == "Claude Sonnet pricing row"
    assert outcome.model is not None
    assert outcome.model.context_window == 1_000_001
    assert outcome.model.pricing.base().input_per_million == Decimal("3.25")
    assert outcome.model.pricing.batch is not None
    assert outcome.model.pricing.batch.output_per_million == Decimal("8")
    assert outcome.model.provenance.source == "official"
    assert outcome.model.provenance.as_of == "2026-07-13"


def _verified_payload(**updates):
    payload = {
        "confirmed": True,
        "context_window": None,
        "input_per_million": "2",
        "output_per_million": "8",
        "cache_read_per_million": None,
        "cache_write_5m_per_million": None,
        "cache_write_1h_per_million": None,
        "batch_input_per_million": None,
        "batch_output_per_million": None,
        "context_tiers": None,
        "deprecated": False,
        "source_url": "https://openai.com/api/pricing/",
        "evidence": "official pricing row",
        "note": "verified",
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize("evidence", ["", "   ", None])
def test_verified_confirmation_requires_grounding_evidence(evidence) -> None:
    original = _provider_model("openai")

    outcome = parse_verified(
        _verified_payload(evidence=evidence),
        original,
        as_of="2026-07-13",
        browsed_urls={"https://openai.com/api/pricing/"},
        browsed_pricing_modes={"https://openai.com/api/pricing/": {"standard"}},
    )

    assert outcome.verified is False
    assert outcome.model is None
    assert outcome.evidence == ""
    assert outcome.note == "official-page verification returned no grounding evidence"


def test_verified_explicit_nulls_remove_obsolete_pricing_dimensions() -> None:
    model = _provider_model("openai")
    base = model.pricing.base().model_copy(update={"max_input_tokens": 100_000})
    upper = base.model_copy(
        update={
            "max_input_tokens": None,
            "input_per_million": base.input_per_million * 2,
            "output_per_million": base.output_per_million * 2,
        }
    )
    original = model.model_copy(
        update={
            "pricing": model.pricing.model_copy(
                update={
                    "standard": (base, upper),
                    "cache_write_5m_per_million": Decimal("3"),
                    "cache_write_1h_per_million": Decimal("4"),
                    "batch": PriceTier(
                        input_per_million=Decimal("1"),
                        output_per_million=Decimal("4"),
                    ),
                }
            )
        }
    )

    outcome = parse_verified(
        _verified_payload(),
        original,
        as_of="2026-07-13",
        browsed_urls={"https://openai.com/api/pricing/"},
        browsed_pricing_modes={"https://openai.com/api/pricing/": {"standard", "batch"}},
    )

    assert outcome.verified and outcome.model is not None
    pricing = outcome.model.pricing
    assert len(pricing.standard) == 1
    assert pricing.base().cache_read_input_per_million is None
    assert pricing.cache_write_5m_per_million is None
    assert pricing.cache_write_1h_per_million is None
    assert pricing.batch is None


def test_verified_malformed_tier_boundary_retains_curated_tiers() -> None:
    model = _provider_model("openai")
    base = model.pricing.base().model_copy(update={"max_input_tokens": 100_000})
    upper = base.model_copy(
        update={
            "max_input_tokens": None,
            "input_per_million": base.input_per_million * 2,
            "output_per_million": base.output_per_million * 2,
        }
    )
    original = model.model_copy(
        update={"pricing": model.pricing.model_copy(update={"standard": (base, upper)})}
    )

    outcome = parse_verified(
        _verified_payload(
            context_tiers=[
                {
                    "up_to_tokens": True,
                    "input_per_million": "2",
                    "output_per_million": "8",
                    "cache_read_per_million": None,
                }
            ]
        ),
        original,
        as_of="2026-07-13",
        browsed_urls={"https://openai.com/api/pricing/"},
        browsed_pricing_modes={"https://openai.com/api/pricing/": {"standard"}},
    )

    assert outcome.verified and outcome.model is not None
    assert outcome.model.pricing.standard[1:] == original.pricing.standard[1:]


def test_verified_missing_or_malformed_pricing_does_not_clear_curated_values() -> None:
    model = _provider_model("openai")
    base = model.pricing.base().model_copy(update={"max_input_tokens": 100_000})
    upper = base.model_copy(
        update={
            "max_input_tokens": None,
            "input_per_million": base.input_per_million * 2,
            "output_per_million": base.output_per_million * 2,
        }
    )
    original = model.model_copy(
        update={
            "pricing": model.pricing.model_copy(
                update={
                    "standard": (base, upper),
                    "cache_write_5m_per_million": Decimal("3"),
                    "cache_write_1h_per_million": Decimal("4"),
                    "batch": PriceTier(
                        input_per_million=Decimal("1"),
                        output_per_million=Decimal("4"),
                    ),
                }
            )
        }
    )
    payload = _verified_payload(
        context_tiers="malformed",
        cache_read_per_million="not-a-price",
        cache_write_1h_per_million="not-a-price",
        batch_input_per_million=None,
        batch_output_per_million="not-a-price",
    )
    payload.pop("cache_write_5m_per_million")

    outcome = parse_verified(
        payload,
        original,
        as_of="2026-07-13",
        browsed_urls={"https://openai.com/api/pricing/"},
        browsed_pricing_modes={"https://openai.com/api/pricing/": {"standard", "batch"}},
    )

    assert outcome.verified and outcome.model is not None
    expected_standard = (
        original.pricing.base().model_copy(
            update={
                "input_per_million": Decimal("2"),
                "output_per_million": Decimal("8"),
            }
        ),
        *original.pricing.standard[1:],
    )
    assert outcome.model.pricing == original.pricing.model_copy(
        update={"standard": expected_standard}
    )


def test_verified_confirmation_requires_the_claimed_source_to_have_been_browsed() -> None:
    original = _provider_model("openai")

    outcome = parse_verified(
        _verified_payload(source_url="https://openai.com/api/pricing/?region=us"),
        original,
        as_of="2026-07-13",
        browsed_urls={"https://platform.openai.com/docs/pricing"},
        browsed_pricing_modes={"https://platform.openai.com/docs/pricing": {"standard"}},
    )

    assert outcome.verified is False
    assert outcome.model is None
    assert outcome.note == "claimed source URL was not visited by the page-reading tools"


def test_verified_confirmation_requires_standard_mode_browser_proof() -> None:
    original = _provider_model("openai")

    outcome = parse_verified(
        _verified_payload(),
        original,
        as_of="2026-07-13",
        browsed_urls={"https://openai.com/api/pricing/"},
        browsed_pricing_modes={},
    )

    assert outcome.verified is False
    assert outcome.note == "standard pricing mode was not verified on the claimed source URL"


def test_verified_rejects_batch_values_without_batch_mode_browser_proof() -> None:
    original = _provider_model("openai")

    outcome = parse_verified(
        _verified_payload(
            batch_input_per_million="0.375",
            batch_output_per_million="2.25",
        ),
        original,
        as_of="2026-07-13",
        browsed_urls={"https://openai.com/api/pricing/"},
        browsed_pricing_modes={"https://openai.com/api/pricing/": {"standard"}},
    )

    assert outcome.verified is False
    assert "without verified Batch mode" in outcome.note


def test_verified_null_batch_without_batch_mode_preserves_curated_batch() -> None:
    model = _provider_model("openai")
    curated_batch = PriceTier(
        input_per_million=Decimal("0.5"),
        output_per_million=Decimal("3"),
    )
    original = model.model_copy(
        update={"pricing": model.pricing.model_copy(update={"batch": curated_batch})}
    )

    outcome = parse_verified(
        _verified_payload(),
        original,
        as_of="2026-07-13",
        browsed_urls={"https://openai.com/api/pricing/"},
        browsed_pricing_modes={"https://openai.com/api/pricing/": {"standard"}},
    )

    assert outcome.verified and outcome.model is not None
    assert outcome.model.pricing.batch == curated_batch


def test_release_policy_rejects_zero_and_implausibly_large_prices() -> None:
    catalog = default_model_catalog()
    model = catalog.models[0]
    for price, message in (("0", "greater than zero"), ("1001", "exceeds 1000")):
        tier = model.pricing.base().model_copy(update={"input_per_million": Decimal(price)})
        changed = model.model_copy(
            update={"pricing": model.pricing.model_copy(update={"standard": (tier,)})}
        )
        candidate = catalog.model_copy(update={"models": (changed, *catalog.models[1:])})
        with pytest.raises(ValueError, match=message):
            validate_catalog(candidate, today=date.fromisoformat(catalog.generated_at))


def test_automated_price_changes_flag_large_deltas() -> None:
    model = default_model_catalog().models[0]
    tier = model.pricing.base().model_copy(
        update={"input_per_million": model.pricing.base().input_per_million * 5}
    )
    changed = model.model_copy(
        update={"pricing": model.pricing.model_copy(update={"standard": (tier,)})}
    )

    warnings = suspicious_price_changes(model, changed)

    assert any("standard[0].input changed" in warning for warning in warnings)


def test_catalog_policy_bounds_context_window_updates() -> None:
    model = default_model_catalog().models[0]
    too_large = model.model_copy(update={"context_window": 10_000_001})
    wrong_type = model.model_copy(update={"context_window": True})
    nonpositive = model.model_copy(update={"context_window": 0})

    assert any(
        "context_window exceeds 10000000" in error
        for error in model_policy_errors(too_large, today=date(2026, 7, 13), max_age_days=None)
    )
    assert any(
        "context_window must be a positive integer" in error
        for error in model_policy_errors(wrong_type, today=date(2026, 7, 13), max_age_days=None)
    )
    assert any(
        "context_window must be a positive integer" in error
        for error in model_policy_errors(nonpositive, today=date(2026, 7, 13), max_age_days=None)
    )
    assert suspicious_price_changes(model, nonpositive) == []


def test_automated_context_window_changes_flag_large_deltas() -> None:
    model = default_model_catalog().models[0].model_copy(update={"context_window": 100_000})
    changed = model.model_copy(update={"context_window": 500_000})

    warnings = suspicious_price_changes(model, changed)

    assert "context_window changed from 100000 to 500000" in warnings


def test_verified_boolean_context_window_does_not_replace_curated_value() -> None:
    original = _provider_model("openai")

    outcome = parse_verified(
        _verified_payload(context_window=True),
        original,
        as_of="2026-07-13",
        browsed_urls={"https://openai.com/api/pricing/"},
        browsed_pricing_modes={"https://openai.com/api/pricing/": {"standard"}},
    )

    assert outcome.verified and outcome.model is not None
    assert outcome.model.context_window == original.context_window


def test_browser_verifier_supplies_pricing_guidance_without_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    verifier = BrowserVerifier(as_of="2026-07-13", max_cost_usd=None)
    prompt = verifier.app.get_agent("model-verifier").spec.system_prompt
    environment = verifier.app.get_environment()

    assert prompt is not None
    assert WORKSPACE_GUIDANCE in prompt
    assert environment.environment.artifact_store is not None


def test_browser_verifier_records_only_page_reading_urls() -> None:
    events = [
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id="verification",
            tool_name="search_web",
            payload={
                "tool_call_id": "search",
                "arguments": {"url": "https://search.example/result"},
            },
        ),
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id="verification",
            tool_name="search_web",
            payload={"tool_call_id": "search"},
        ),
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id="verification",
            tool_name="read_page",
            payload={
                "tool_call_id": "reused-call",
                "tool_round_id": "round-1",
                "arguments": {"url": "https://openai.com/api/pricing/"},
            },
        ),
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id="verification",
            tool_name="read_page",
            payload={
                "tool_call_id": "reused-call",
                "tool_round_id": "round-1",
                "result": {
                    "is_error": False,
                    "structured": {
                        "pricing_mode": "standard",
                        "pricing_mode_verified": True,
                        "effective_url": "https://developers.openai.com/api/pricing",
                    },
                },
            },
        ),
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id="verification",
            tool_name="read_page",
            payload={
                "tool_call_id": "reused-call",
                "tool_round_id": "round-2",
                "arguments": {"url": "https://attacker.example/pricing"},
            },
        ),
        Event(
            type=EventType.TOOL_CALL_FAILED,
            session_id="verification",
            tool_name="read_page",
            payload={"tool_call_id": "reused-call", "tool_round_id": "round-2"},
        ),
    ]

    assert browsed_urls(events) == {"https://developers.openai.com/api/pricing"}
    assert browsed_pricing_modes(events) == {
        "https://developers.openai.com/api/pricing": {"standard"}
    }


def test_browser_verifier_records_every_mode_verified_by_all_mode_read() -> None:
    url = "https://developers.openai.com/api/docs/pricing"
    events = [
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id="verification",
            tool_name="read_page",
            payload={
                "tool_call_id": "all-prices",
                "tool_round_id": "round-1",
                "arguments": {"url": url, "pricing_mode": "all"},
            },
        ),
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id="verification",
            tool_name="read_page",
            payload={
                "tool_call_id": "all-prices",
                "tool_round_id": "round-1",
                "result": {
                    "is_error": False,
                    "structured": {
                        "pricing_mode": "all",
                        "pricing_mode_verified": True,
                        "pricing_modes_verified": ["standard", "batch"],
                    },
                },
            },
        ),
    ]

    assert browsed_pricing_modes(events) == {url: {"standard", "batch"}}


def test_host_browser_subprocess_scrubs_secrets_and_uses_non_login_shell(monkeypatch) -> None:
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-browser")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(browser.subprocess, "run", fake_run)

    assert browser.run_bash_host("agent-browser snapshot") == "ok"
    assert captured["args"] == ["bash", "-c", "agent-browser snapshot"]
    assert captured["env"]["PATH"] == "/usr/bin"
    assert "OPENAI_API_KEY" not in captured["env"]


def test_host_browser_subprocess_reports_exit_and_stderr(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        return CompletedProcess(args=args, returncode=127, stdout="", stderr="not found\n")

    monkeypatch.setattr(browser.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="exit 127: not found"):
        browser.run_bash_host("missing-browser")


@pytest.mark.parametrize(
    "command",
    [
        browser.open_page_command(
            _OPENAI_PAGE,
            session="verification",
            allowed_hosts=_OPENAI_HOSTS,
        ),
        browser.click_command("e102", session="verification", allowed_hosts=_OPENAI_HOSTS),
        browser.screenshot_current_page_command(
            "/tmp/page.png",
            session="verification",
            allowed_hosts=_OPENAI_HOSTS,
        ),
    ],
)
def test_browser_command_chains_stop_after_primary_action_failure(
    command: str, tmp_path: Path, monkeypatch
) -> None:
    executable = tmp_path / "agent-browser"
    calls = tmp_path / "calls"
    executable.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> {calls}\n"
        'case " $* " in\n'
        '  *" open "*|*" click "*|*" screenshot "*) exit 7 ;;\n'
        "esac\n"
        "exit 0\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/bin:/usr/bin")

    with pytest.raises(RuntimeError, match="browser command failed with exit 7"):
        browser.run_bash_host(command)

    assert len(calls.read_text().splitlines()) == 1


def test_read_page_falls_back_when_compressed_snapshot_is_empty(monkeypatch) -> None:
    responses = iter(("", _OPENAI_PAGE, "cookie navigation " * 30, "Official price: $1 input"))

    async def fake_exec(ctx, command, **kwargs):
        return next(responses)

    monkeypatch.setattr(catalog_tools, "_exec", fake_exec)

    result = asyncio.run(
        ReadPageTool().run(
            _browser_context(),
            {
                "url": "https://developers.openai.com/api/pricing",
                "pricing_mode": "standard",
            },
        )
    )

    assert result.is_error is False
    assert result.structured == {
        "source": "text",
        "pricing_mode": "standard",
        "pricing_mode_verified": True,
        "selection_note": "default page state",
        "effective_url": _OPENAI_PAGE,
    }
    assert result.content == (
        "[verified pricing_mode=standard; default page state]\nOfficial price: $1 input"
    )


def test_read_page_selects_and_verifies_batch_pricing_control(monkeypatch) -> None:
    standard_snapshot = (
        '- StaticText "Batch API price"\n'
        '- switch "Batch API price" [checked=false] [ref=e102]\n'
        + '- row "gpt-5.4-mini $0.75 $4.50 standard input output"\n'
        * 8
    )
    batch_snapshot = (
        '- StaticText "Batch API price"\n'
        '- switch "Batch API price" [checked=true] [ref=e102]\n'
        + '- row "gpt-5.4-mini $0.375 $2.25 batch input output"\n'
        * 8
    )
    responses = iter(("", _OPENAI_PAGE, standard_snapshot, "", _OPENAI_PAGE, batch_snapshot))
    commands = []

    async def fake_exec(ctx, command, **kwargs):
        commands.append(command)
        return next(responses)

    monkeypatch.setattr(catalog_tools, "_exec", fake_exec)

    result = asyncio.run(
        ReadPageTool().run(
            _browser_context(),
            {
                "url": "https://developers.openai.com/api/pricing",
                "pricing_mode": "batch",
            },
        )
    )

    assert result.is_error is False
    assert commands[3] == browser.click_command(
        "e102", session="verification", allowed_hosts=_OPENAI_HOSTS
    )
    assert commands[5] == browser.snapshot_command(
        session="verification", allowed_hosts=_OPENAI_HOSTS
    )
    assert result.structured is not None
    assert result.structured["pricing_mode"] == "batch"
    assert result.structured["pricing_mode_verified"] is True
    assert "$0.375 $2.25" in result.content


def test_read_page_all_mode_automatically_captures_standard_and_batch(monkeypatch) -> None:
    standard_snapshot = (
        '- radio "Standard" [checked=true, ref=e10]\n'
        '- radio "Batch" [checked=false, ref=e20]\n'
        + '- row "gpt-5.4-mini $0.75 $4.50 standard input output"\n'
        * 8
    )
    batch_snapshot = (
        '- radio "Standard" [checked=false, ref=e11]\n'
        '- radio "Batch" [checked=true, ref=e21]\n'
        + '- row "gpt-5.4-mini $0.375 $2.25 batch input output"\n'
        * 8
    )
    responses = iter(
        (
            "",
            _OPENAI_PAGE,
            standard_snapshot,
            "",
            _OPENAI_PAGE,
            standard_snapshot,
            "",
            _OPENAI_PAGE,
            batch_snapshot,
        )
    )

    async def fake_exec(ctx, command, **kwargs):
        return next(responses)

    monkeypatch.setattr(catalog_tools, "_exec", fake_exec)

    result = asyncio.run(
        ReadPageTool().run(
            _browser_context(),
            {"url": "https://developers.openai.com/api/pricing"},
        )
    )

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["pricing_mode"] == "all"
    assert result.structured["pricing_modes_verified"] == ["standard", "batch"]
    assert "[verified pricing_mode=standard" in result.content
    assert "$0.75 $4.50" in result.content
    assert "[verified pricing_mode=batch" in result.content
    assert "$0.375 $2.25" in result.content


def test_read_page_standard_mode_does_not_read_unselected_batch_values(monkeypatch) -> None:
    standard_snapshot = (
        '- StaticText "Batch API price"\n'
        '- switch "Batch API price" [checked=false] [ref=e102]\n'
        + '- row "gpt-5.4-mini $0.75 $4.50 standard input output"\n'
        * 8
    )
    commands = []
    responses = iter(("", _OPENAI_PAGE, standard_snapshot))

    async def fake_exec(ctx, command, **kwargs):
        commands.append(command)
        return next(responses)

    monkeypatch.setattr(catalog_tools, "_exec", fake_exec)

    result = asyncio.run(
        ReadPageTool().run(
            _browser_context(),
            {
                "url": "https://developers.openai.com/api/pricing",
                "pricing_mode": "standard",
            },
        )
    )

    assert result.is_error is False
    assert len(commands) == 3
    assert result.structured is not None
    assert result.structured["pricing_mode"] == "standard"
    assert result.structured["pricing_mode_verified"] is True
    assert "$0.75 $4.50" in result.content


def test_read_page_selects_requested_mode_for_every_pricing_table(monkeypatch) -> None:
    initial = (
        '- radio "Batch" [checked=false, ref=e10]\n'
        '- radio "Batch" [checked=false, ref=e20]\n' + '- row "model $1 $2 batch pricing"\n' * 10
    )
    first_selected = initial.replace(
        'radio "Batch" [checked=false, ref=e10]',
        'radio "Batch" [checked=true, ref=e11]',
    ).replace("ref=e20", "ref=e21")
    all_selected = first_selected.replace(
        'radio "Batch" [checked=false, ref=e21]',
        'radio "Batch" [checked=true, ref=e22]',
    )
    responses = iter(
        (
            "",
            _OPENAI_PAGE,
            initial,
            "",
            _OPENAI_PAGE,
            first_selected,
            "",
            _OPENAI_PAGE,
            all_selected,
        )
    )
    commands = []

    async def fake_exec(ctx, command, **kwargs):
        commands.append(command)
        return next(responses)

    monkeypatch.setattr(catalog_tools, "_exec", fake_exec)

    result = asyncio.run(
        ReadPageTool().run(
            _browser_context(),
            {"url": _OPENAI_PAGE, "pricing_mode": "batch"},
        )
    )

    assert result.is_error is False
    assert (
        browser.click_command("e10", session="verification", allowed_hosts=_OPENAI_HOSTS)
        in commands
    )
    assert (
        browser.click_command("e21", session="verification", allowed_hosts=_OPENAI_HOSTS)
        in commands
    )
    assert result.structured is not None
    assert result.structured["selection_note"] == "batch pricing controls selected"


def test_browser_click_command_rejects_shell_metacharacters() -> None:
    assert browser.click_command("e102", session="verification", allowed_hosts=_OPENAI_HOSTS) == (
        "agent-browser --session verification --allowed-domains "
        "developers.openai.com,openai.com,platform.openai.com click @e102 && "
        "agent-browser --session verification --allowed-domains "
        "developers.openai.com,openai.com,platform.openai.com wait 500 >/dev/null 2>&1"
    )
    with pytest.raises(ValueError, match="browser refs"):
        browser.click_command(
            "e102;touch-pwned", session="verification", allowed_hosts=_OPENAI_HOSTS
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://developers.openai.com/api/pricing",
        "https://user@developers.openai.com/api/pricing",
        "https://developers.openai.com:8443/api/pricing",
        "https://attacker.example/api/pricing",
        "https://localhost/internal",
        "https://127.0.0.1/internal",
        "https://[::1]/internal",
    ],
)
def test_catalog_browser_rejects_non_official_navigation(url: str) -> None:
    with pytest.raises(ValueError):
        validate_official_url(url, provider_name="openai")


def test_search_web_returns_only_provider_official_urls(monkeypatch) -> None:
    monkeypatch.setattr(
        search,
        "search_web",
        lambda query: [
            "https://attacker.example/injected",
            "https://developers.openai.com/api/pricing#token-prices",
        ],
    )

    result = asyncio.run(
        catalog_tools.SearchWebTool().run(_browser_context(), {"query": "pricing"})
    )

    assert result.structured == {"urls": [_OPENAI_PAGE]}


def test_read_page_rejects_redirect_to_non_official_host(monkeypatch) -> None:
    responses = iter(("", "https://attacker.example/injected"))

    async def fake_exec(ctx, command, **kwargs):
        return next(responses)

    monkeypatch.setattr(catalog_tools, "_exec", fake_exec)

    with pytest.raises(ValueError, match="not official"):
        asyncio.run(
            ReadPageTool().run(
                _browser_context(),
                {"url": _OPENAI_PAGE, "pricing_mode": "standard"},
            )
        )


def test_browser_verifier_applies_cost_limit_to_caller_supplied_app() -> None:
    class CapturingApp:
        def __init__(self) -> None:
            self.requests = []

        async def run(self, request):
            self.requests.append(request)
            if False:
                yield

    app = CapturingApp()
    verifier = BrowserVerifier(
        as_of="2026-07-13",
        app=app,
        max_cost_usd=0.03,
    )

    asyncio.run(verifier.averify(_provider_model("openai")))

    assert len(app.requests) == 1
    assert len(app.requests[0].budget_limits) == 1
    assert app.requests[0].budget_limits[0].max_estimated_cost == Decimal("0.03")
    assert app.requests[0].budget_limits[0].allow_unpriced is False


def test_browser_verifier_rejects_unpriced_configured_model() -> None:
    with pytest.raises(ValueError, match="not a bundled canonical record"):
        BrowserVerifier(
            as_of="2026-07-13",
            model="unlisted-verifier-model",
            app=object(),
            max_cost_usd=None,
        )


def test_verifier_model_must_be_canonical_active_and_tool_capable() -> None:
    catalog = default_model_catalog()
    luna = catalog.match(provider_name="openai", model="gpt-5.6-luna")
    assert luna is not None
    without_luna = catalog.model_copy(
        update={"models": tuple(item for item in catalog.models if item != luna)}
    )
    deprecated = catalog.model_copy(
        update={
            "models": tuple(
                item.model_copy(update={"deprecated": True}) if item == luna else item
                for item in catalog.models
            )
        }
    )

    with pytest.raises(ValueError, match="not a bundled canonical record"):
        verifier_model_info(without_luna)
    with pytest.raises(ValueError, match="deprecated"):
        verifier_model_info(deprecated)


def test_browser_verifier_stops_after_unknown_resolved_model() -> None:
    class UnknownResolvedModelProvider(ModelProvider):
        name = "openai"
        supports_native_structured_output = True

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.tool_call(
                id="call_search",
                name="search_web",
                arguments={"query": "official model pricing"},
            )
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "model": "gpt-5.6-luna-unlisted",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            )

    class RecordingCayuApp(CayuApp):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.emitted_events: list[Event] = []

        async def run(self, request):
            async for event in super().run(request):
                self.emitted_events.append(event)
                yield event

    async def exercise():
        provider = UnknownResolvedModelProvider()
        store = InMemorySessionStore()
        app = RecordingCayuApp(session_store=store)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="model-verifier", model="gpt-5.6-luna"),
            tools=[catalog_tools.SearchWebTool()],
        )
        verifier = BrowserVerifier(as_of="2026-07-13", app=app)

        outcome = await verifier.averify(_provider_model("openai"))
        return provider, outcome, app.emitted_events

    provider, outcome, events = asyncio.run(exercise())

    assert outcome.verified is False
    assert len(provider.requests) == 1
    limit_event = next(event for event in events if event.type == EventType.SESSION_LIMIT_REACHED)
    assert limit_event.payload["limit"] == "estimated_cost"
    assert "no matching pricing" in limit_event.payload["message"]
    assert not any(event.type == EventType.TOOL_CALL_STARTED for event in events)


def test_refresh_paths_are_repository_relative_after_chdir(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    repository_root = Path(refresh.__file__).resolve().parents[2]

    assert repository_root == refresh.REPOSITORY_ROOT
    assert repository_root / "src/cayu/data/default_model_catalog.json" == refresh.CATALOG_PATH
    assert repository_root / "model-catalog-refresh.md" == refresh.REPORT_PATH
    assert refresh.CATALOG_PATH.is_file()


def test_refresh_report_escapes_and_bounds_untrusted_verifier_text() -> None:
    rendered = refresh._safe_report_text(
        "provider's <script>alert(1)</script> [click](javascript:x)"
    )

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "[click](javascript:x)" not in rendered
    assert "provider's" in rendered
    assert "&\\#x27;" not in rendered
    assert len(refresh._safe_report_text("x" * 2_000)) == refresh._REPORT_TEXT_LIMIT
    evidence = "e" * 1_500
    catalog = default_model_catalog()
    report = refresh._format_report(
        catalog,
        catalog,
        flagged=[],
        evidence=[
            refresh._VerificationEvidence(
                identity="provider/model",
                source_url="https://example.com/pricing",
                quote=evidence,
            )
        ],
    )
    assert evidence in report


def test_refresh_removes_newly_deprecated_models_before_policy_validation(
    tmp_path, monkeypatch
) -> None:
    original = default_model_catalog()
    deprecated_model = next(
        model
        for model in original.models
        if sum(item.provider_name == model.provider_name for item in original.models) > 1
    )
    deprecated_identity = (deprecated_model.provider_name, deprecated_model.model)

    class FakeBrowserVerifier:
        def __init__(self, **kwargs) -> None:
            pass

        async def averify(self, model):
            deprecated = (model.provider_name, model.model) == deprecated_identity
            return VerifyOutcome(
                verified=True,
                model=model.model_copy(update={"deprecated": deprecated}),
                note="verified",
                evidence="official evidence",
            )

    catalog_path = tmp_path / "catalog.json"
    report_path = tmp_path / "report.md"
    monkeypatch.setattr(refresh, "BrowserVerifier", FakeBrowserVerifier)
    monkeypatch.setattr(refresh, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(refresh, "REPORT_PATH", report_path)

    asyncio.run(refresh._run(force_all=True, max_age_days=30))

    candidate = ModelCatalog.model_validate_json(catalog_path.read_text(encoding="utf-8"))
    assert deprecated_identity not in {
        (model.provider_name, model.model) for model in candidate.models
    }
    report = report_path.read_text(encoding="utf-8")
    assert "official source marks model deprecated; removed from candidate" in report
    assert "source:" in report
    assert "official evidence" in report


def test_refresh_preserves_active_verifier_when_marked_deprecated(tmp_path, monkeypatch) -> None:
    original = default_model_catalog()
    verifier_model = original.match(provider_name="openai", model="gpt-5.6-luna")
    assert verifier_model is not None
    identity = f"{verifier_model.provider_name}/{verifier_model.model}"

    class FakeBrowserVerifier:
        def __init__(self, **kwargs) -> None:
            pass

        async def averify(self, model):
            return VerifyOutcome(
                verified=True,
                model=model.model_copy(update={"deprecated": True}),
                evidence="official retirement notice",
            )

    report_path = tmp_path / "report.md"
    catalog_path = tmp_path / "catalog.json"
    monkeypatch.setattr(refresh, "BrowserVerifier", FakeBrowserVerifier)
    monkeypatch.setattr(refresh, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(refresh, "REPORT_PATH", report_path)

    asyncio.run(refresh._run(force_all=True, max_age_days=30, only=(identity,)))

    assert not catalog_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "active maintenance verifier was marked deprecated" in report
    assert "preserved until a replacement verifier is configured" in report


def test_refresh_rejects_deprecation_from_non_official_host(tmp_path, monkeypatch) -> None:
    original = default_model_catalog()
    selected = next(
        model
        for model in original.models
        if sum(item.provider_name == model.provider_name for item in original.models) > 1
    )
    identity = f"{selected.provider_name}/{selected.model}"

    class FakeBrowserVerifier:
        def __init__(self, **kwargs) -> None:
            pass

        async def averify(self, model):
            provenance = model.provenance.model_copy(update={"url": "https://example.com/pricing"})
            return VerifyOutcome(
                verified=True,
                model=model.model_copy(update={"deprecated": True, "provenance": provenance}),
                note="verified",
                evidence="injected deprecation claim",
            )

    report_path = tmp_path / "report.md"
    monkeypatch.setattr(refresh, "BrowserVerifier", FakeBrowserVerifier)
    monkeypatch.setattr(refresh, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(refresh, "REPORT_PATH", report_path)

    asyncio.run(refresh._run(force_all=True, max_age_days=30, only=(identity,)))

    report = report_path.read_text(encoding="utf-8")
    assert "automated deprecation rejected" in report
    assert "not on an allowed official host" in report
    assert r"example\.com/pricing" in report
    assert "injected deprecation claim" in report


def test_refresh_preserves_one_model_when_provider_deprecates_every_record(
    tmp_path, monkeypatch
) -> None:
    original = default_model_catalog()
    provider = next(
        provider
        for provider in {model.provider_name for model in original.models}
        if provider != "openai"
        and sum(model.provider_name == provider for model in original.models) > 1
    )
    selected = tuple(
        f"{model.provider_name}/{model.model}"
        for model in original.models
        if model.provider_name == provider
    )

    class FakeBrowserVerifier:
        def __init__(self, **kwargs) -> None:
            pass

        async def averify(self, model):
            return VerifyOutcome(
                verified=True,
                model=model.model_copy(update={"deprecated": True}),
                note="verified",
                evidence="official retirement notice",
            )

    catalog_path = tmp_path / "catalog.json"
    report_path = tmp_path / "report.md"
    monkeypatch.setattr(refresh, "BrowserVerifier", FakeBrowserVerifier)
    monkeypatch.setattr(refresh, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(refresh, "REPORT_PATH", report_path)

    asyncio.run(refresh._run(force_all=True, max_age_days=30, only=selected))

    candidate = ModelCatalog.model_validate_json(catalog_path.read_text(encoding="utf-8"))
    assert sum(model.provider_name == provider for model in candidate.models) == 1
    assert "last bundled model deprecated" in report_path.read_text(encoding="utf-8")


def test_refresh_rejects_one_invalid_record_without_aborting_other_updates(
    tmp_path, monkeypatch
) -> None:
    original = default_model_catalog()
    invalid_identity = (original.models[0].provider_name, original.models[0].model)
    safe_identity = (original.models[1].provider_name, original.models[1].model)

    class FakeBrowserVerifier:
        def __init__(self, **kwargs) -> None:
            pass

        async def averify(self, model):
            identity = (model.provider_name, model.model)
            if identity == invalid_identity:
                zero = model.pricing.base().model_copy(update={"input_per_million": Decimal("0")})
                updated = model.model_copy(
                    update={"pricing": model.pricing.model_copy(update={"standard": (zero,)})}
                )
            elif identity == safe_identity:
                updated = model.model_copy(
                    update={"context_window": (model.context_window or 1) + 1}
                )
            else:
                updated = model
            return VerifyOutcome(
                verified=True,
                model=updated,
                note="verified",
                evidence="official evidence",
            )

    catalog_path = tmp_path / "catalog.json"
    report_path = tmp_path / "report.md"
    monkeypatch.setattr(refresh, "BrowserVerifier", FakeBrowserVerifier)
    monkeypatch.setattr(refresh, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(refresh, "REPORT_PATH", report_path)

    asyncio.run(refresh._run(force_all=True, max_age_days=30))

    candidate = ModelCatalog.model_validate_json(catalog_path.read_text(encoding="utf-8"))
    invalid = candidate.resolve(provider_name=invalid_identity[0], model=invalid_identity[1])
    safe = candidate.resolve(provider_name=safe_identity[0], model=safe_identity[1])
    assert invalid == original.models[0]
    assert safe is not None
    assert safe.context_window == (original.models[1].context_window or 1) + 1
    assert "automated update rejected" in report_path.read_text(encoding="utf-8")


def test_refresh_writes_report_before_candidate_validation_failure(tmp_path, monkeypatch) -> None:
    original = default_model_catalog()
    selected = original.models[0]
    identity = f"{selected.provider_name}/{selected.model}"

    class FakeBrowserVerifier:
        def __init__(self, **kwargs) -> None:
            pass

        async def averify(self, model):
            return VerifyOutcome(
                verified=True,
                model=model.model_copy(update={"aliases": (" ",)}),
                note="verified",
                evidence="official evidence",
            )

    catalog_path = tmp_path / "catalog.json"
    report_path = tmp_path / "report.md"
    monkeypatch.setattr(refresh, "BrowserVerifier", FakeBrowserVerifier)
    monkeypatch.setattr(refresh, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(refresh, "REPORT_PATH", report_path)

    with pytest.raises(ValueError):
        asyncio.run(refresh._run(force_all=True, max_age_days=30, only=(identity,)))

    assert not catalog_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "candidate validation failed" in report
    assert "aliases" in report


def test_refresh_only_verifies_selected_catalog_identity(tmp_path, monkeypatch) -> None:
    original = default_model_catalog()
    selected = original.models[0]
    selected_identity = f"{selected.provider_name}/{selected.model}"
    observed: list[str] = []

    class FakeBrowserVerifier:
        def __init__(self, **kwargs) -> None:
            pass

        async def averify(self, model):
            observed.append(f"{model.provider_name}/{model.model}")
            return VerifyOutcome(
                verified=True,
                model=model,
                note="verified",
                evidence="official evidence",
            )

    monkeypatch.setattr(refresh, "BrowserVerifier", FakeBrowserVerifier)
    monkeypatch.setattr(refresh, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(refresh, "REPORT_PATH", tmp_path / "report.md")

    asyncio.run(
        refresh._run(
            force_all=True,
            max_age_days=30,
            only=(selected_identity,),
        )
    )

    assert observed == [selected_identity]


def test_recommendation_audit_reports_unbundled_official_model(tmp_path, monkeypatch) -> None:
    original = default_model_catalog()
    selected = original.match(provider_name="openai", model="gpt-5.6-luna")
    assert selected is not None
    identity = f"{selected.provider_name}/{selected.model}"

    class FakeBrowserVerifier:
        def __init__(self, **kwargs) -> None:
            pass

        async def adiscover_recommendations(self, provider_name, existing_models):
            assert provider_name == "openai"
            assert "gpt-5.6-luna" in existing_models
            return RecommendationOutcome(
                verified=True,
                provider_name=provider_name,
                models=("gpt-5.6-sol", "gpt-6-new"),
                source_url="https://developers.openai.com/api/docs/guides/latest-model",
                evidence="Use gpt-6-new for frontier workloads.",
            )

    report_path = tmp_path / "report.md"
    monkeypatch.setattr(refresh, "BrowserVerifier", FakeBrowserVerifier)
    monkeypatch.setattr(refresh, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(refresh, "REPORT_PATH", report_path)

    asyncio.run(
        refresh._run(
            force_all=False,
            max_age_days=10_000,
            only=(identity,),
            audit_recommendations=True,
        )
    )

    report = report_path.read_text(encoding="utf-8")
    assert "recommended model missing from bundled catalog" in report
    assert r"gpt\-6\-new" in report
    assert r"Use gpt\-6\-new for frontier workloads" in report


def test_recommendation_audit_accepts_models_covered_by_exact_aliases(
    tmp_path, monkeypatch
) -> None:
    original = default_model_catalog()
    selected = original.match(provider_name="openai", model="gpt-5.6-luna")
    assert selected is not None
    identity = f"{selected.provider_name}/{selected.model}"

    class FakeBrowserVerifier:
        def __init__(self, **kwargs) -> None:
            pass

        async def adiscover_recommendations(self, provider_name, existing_models):
            return RecommendationOutcome(
                verified=True,
                provider_name=provider_name,
                models=("gpt-5.6", "gpt-5.6-luna", "gpt-5.6-terra"),
                source_url="https://developers.openai.com/api/docs/guides/latest-model",
                evidence="Sol, Luna, and Terra are current.",
            )

    monkeypatch.setattr(refresh, "BrowserVerifier", FakeBrowserVerifier)
    monkeypatch.setattr(refresh, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(refresh, "REPORT_PATH", tmp_path / "report.md")

    asyncio.run(
        refresh._run(
            force_all=False,
            max_age_days=10_000,
            only=(identity,),
            audit_recommendations=True,
        )
    )


def test_refresh_fails_when_every_attempt_is_inconclusive(tmp_path, monkeypatch) -> None:
    original = default_model_catalog()
    selected = original.models[0]
    selected_identity = f"{selected.provider_name}/{selected.model}"

    class FakeBrowserVerifier:
        def __init__(self, **kwargs) -> None:
            pass

        async def averify(self, model):
            return VerifyOutcome(verified=False, note="browser unavailable")

    report_path = tmp_path / "report.md"
    monkeypatch.setattr(refresh, "BrowserVerifier", FakeBrowserVerifier)
    monkeypatch.setattr(refresh, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(refresh, "REPORT_PATH", report_path)

    with pytest.raises(RuntimeError, match="all attempted model verifications"):
        asyncio.run(
            refresh._run(
                force_all=True,
                max_age_days=30,
                only=(selected_identity,),
            )
        )

    assert "browser unavailable" in report_path.read_text(encoding="utf-8")


def test_refresh_verifier_construction_failure_writes_report(tmp_path, monkeypatch) -> None:
    original = default_model_catalog()
    selected = original.models[0]
    identity = f"{selected.provider_name}/{selected.model}"

    class BrokenBrowserVerifier:
        def __init__(self, **kwargs) -> None:
            raise ValueError("OPENAI_API_KEY is missing")

    report_path = tmp_path / "report.md"
    monkeypatch.setattr(refresh, "BrowserVerifier", BrokenBrowserVerifier)
    monkeypatch.setattr(refresh, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(refresh, "REPORT_PATH", report_path)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        asyncio.run(refresh._run(force_all=True, max_age_days=30, only=(identity,)))

    report = report_path.read_text(encoding="utf-8")
    assert "verifier construction failed" in report
    assert r"OPENAI\_API\_KEY is missing" in report


def test_fresh_refresh_does_not_construct_live_verifier(tmp_path, monkeypatch) -> None:
    class UnexpectedBrowserVerifier:
        def __init__(self, **kwargs) -> None:
            raise AssertionError("fresh refresh should not require provider credentials")

    monkeypatch.setattr(refresh, "BrowserVerifier", UnexpectedBrowserVerifier)
    monkeypatch.setattr(refresh, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(refresh, "REPORT_PATH", tmp_path / "report.md")

    asyncio.run(refresh._run(force_all=False, max_age_days=10_000))


def test_refresh_invalid_cost_limit_still_writes_report(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "report.md"
    monkeypatch.setenv("MAX_VERIFY_COST_USD", "not-a-number")
    monkeypatch.setattr(refresh, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(refresh, "REPORT_PATH", report_path)

    with pytest.raises(ValueError, match="MAX_VERIFY_COST_USD"):
        asyncio.run(refresh._run(force_all=False, max_age_days=10_000))

    assert "invalid cost limit" in report_path.read_text(encoding="utf-8")
