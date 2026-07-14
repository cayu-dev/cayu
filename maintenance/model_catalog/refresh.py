"""Re-verify Cayu's committed catalog using Cayu plus a real browser.

Run locally with ``python -m maintenance.model_catalog.refresh``. The scheduled workflow
installs ``agent-browser`` and Chromium, then opens a review PR when verified records
change. Runtime catalog loading and release builds never call this module.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from cayu import ModelCatalog, default_model_catalog, dump_model_catalog
from maintenance.model_catalog.browser_verifier import (
    DEFAULT_MAX_VERIFY_COST_USD,
    DEFAULT_VERIFIER_MODEL,
    RECOMMENDATION_PAGES,
    VERIFIER_PROVIDER_NAME,
    BrowserVerifier,
    verifier_model_info,
)
from maintenance.model_catalog.decide import Action, Decision, decide
from maintenance.model_catalog.diff import format_catalog_diff, markdown_code_span
from maintenance.model_catalog.policy import (
    VERIFY_MAX_AGE_DAYS,
    model_policy_errors,
    suspicious_price_changes,
    validate_catalog,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPOSITORY_ROOT / "src/cayu/data/default_model_catalog.json"
REPORT_PATH = REPOSITORY_ROOT / "model-catalog-refresh.md"
_REPORT_TEXT_LIMIT = 800
_EVIDENCE_TEXT_LIMIT = 2_000


@dataclass(frozen=True)
class _VerificationFlag:
    identity: str
    reason: str
    note: str


@dataclass(frozen=True)
class _VerificationEvidence:
    identity: str
    source_url: str
    quote: str


def _safe_report_text(value: str, *, limit: int = _REPORT_TEXT_LIMIT) -> str:
    """Render untrusted verifier/page text as one bounded, inert Markdown line."""

    collapsed = " ".join(value.split())[:limit]
    html_escaped = html.escape(collapsed, quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>-])", r"\\\1", html_escaped)


def _verifier_rates(catalog: ModelCatalog) -> tuple[float, float]:
    info = verifier_model_info(
        catalog,
        provider_name=VERIFIER_PROVIDER_NAME,
        model=DEFAULT_VERIFIER_MODEL,
    )
    base = info.pricing.base()
    return float(base.input_per_million), float(base.output_per_million)


def _max_verify_cost() -> float:
    raw = os.environ.get("MAX_VERIFY_COST_USD", str(DEFAULT_MAX_VERIFY_COST_USD))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("MAX_VERIFY_COST_USD must be a finite number greater than zero") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("MAX_VERIFY_COST_USD must be a finite number greater than zero")
    return value


def _format_report(
    original: ModelCatalog,
    candidate: ModelCatalog,
    *,
    flagged: list[_VerificationFlag],
    evidence: list[_VerificationEvidence],
) -> str:
    report = format_catalog_diff(original, candidate)
    if flagged:
        report += "\n### Verification flags\n\n"
        for flag in flagged:
            report += (
                f"- {markdown_code_span(flag.identity)} — {_safe_report_text(flag.reason)}: "
                f"{_safe_report_text(flag.note)}\n"
            )
    if evidence:
        report += "\n### Browser-grounded evidence\n\n"
        for item in evidence:
            report += (
                f"- {markdown_code_span(item.identity)} — "
                f"source: {_safe_report_text(item.source_url)}; "
                f"evidence: {_safe_report_text(item.quote, limit=_EVIDENCE_TEXT_LIMIT)}\n"
            )
    return report


async def _run(
    *,
    force_all: bool,
    max_age_days: int,
    only: tuple[str, ...] = (),
    audit_recommendations: bool = False,
) -> None:
    today = datetime.now(UTC).date().isoformat()
    original = default_model_catalog()
    requested = set(only)
    known = {f"{model.provider_name}/{model.model}" for model in original.models}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError("unknown catalog model(s): " + ", ".join(unknown))
    verifier: BrowserVerifier | None = None
    kept = []
    flagged: list[_VerificationFlag] = []
    evidence: list[_VerificationEvidence] = []
    provider_survivors = Counter(model.provider_name for model in original.models)
    input_tokens = output_tokens = 0
    attempted = verified = 0
    discovery_attempted = discovery_verified = missing_recommendations = 0
    try:
        max_cost = _max_verify_cost()
    except ValueError as exc:
        flagged.append(
            _VerificationFlag(
                identity="refresh configuration",
                reason="invalid cost limit",
                note=str(exc),
            )
        )
        REPORT_PATH.write_text(
            _format_report(original, original, flagged=flagged, evidence=evidence),
            encoding="utf-8",
        )
        raise

    for model in original.models:
        identity = f"{model.provider_name}/{model.model}"
        if requested and identity not in requested:
            kept.append(model)
            continue
        decision = (
            Decision(Action.VERIFY, "forced verification")
            if force_all
            else decide(model, now=today, max_age_days=max_age_days)
        )
        if decision.action is Action.ACCEPT:
            kept.append(model)
            continue
        attempted += 1
        if verifier is None:
            try:
                verifier = BrowserVerifier(as_of=today, max_cost_usd=max_cost)
            except Exception as exc:
                kept.append(model)
                flagged.append(
                    _VerificationFlag(
                        identity=identity,
                        reason=decision.reason,
                        note=f"verifier construction failed: {type(exc).__name__}: {exc}",
                    )
                )
                REPORT_PATH.write_text(
                    _format_report(original, original, flagged=flagged, evidence=evidence),
                    encoding="utf-8",
                )
                raise
        try:
            outcome = await verifier.averify(model)
        except Exception as exc:
            kept.append(model)
            flagged.append(
                _VerificationFlag(
                    identity=identity,
                    reason=decision.reason,
                    note=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        if outcome.usage:
            input_tokens += outcome.usage.get("input_tokens", 0)
            output_tokens += outcome.usage.get("output_tokens", 0)
        if outcome.verified and outcome.model is not None:
            verified += 1
            evidence.append(
                _VerificationEvidence(
                    identity=identity,
                    source_url=outcome.model.provenance.url,
                    quote=outcome.evidence,
                )
            )
            if outcome.model.deprecated:
                if (
                    model.provider_name == VERIFIER_PROVIDER_NAME
                    and model.model == DEFAULT_VERIFIER_MODEL
                ):
                    kept.append(model)
                    flagged.append(
                        _VerificationFlag(
                            identity=identity,
                            reason=decision.reason,
                            note=(
                                "active maintenance verifier was marked deprecated; record "
                                "preserved until a replacement verifier is configured; source: "
                                f"{outcome.model.provenance.url}"
                            ),
                        )
                    )
                    continue
                policy_errors = model_policy_errors(
                    outcome.model,
                    today=date.fromisoformat(today),
                    max_age_days=None,
                    allow_deprecated=True,
                )
                if policy_errors:
                    kept.append(model)
                    flagged.append(
                        _VerificationFlag(
                            identity=identity,
                            reason=decision.reason,
                            note="automated deprecation rejected: "
                            + "; ".join(policy_errors)
                            + f"; source: {outcome.model.provenance.url}",
                        )
                    )
                elif provider_survivors[model.provider_name] == 1:
                    kept.append(model)
                    flagged.append(
                        _VerificationFlag(
                            identity=identity,
                            reason=decision.reason,
                            note="official source marks the provider's last bundled model deprecated; "
                            "record preserved for manual replacement; source: "
                            f"{outcome.model.provenance.url}",
                        )
                    )
                else:
                    provider_survivors[model.provider_name] -= 1
                    flagged.append(
                        _VerificationFlag(
                            identity=identity,
                            reason=decision.reason,
                            note="official source marks model deprecated; removed from candidate; "
                            f"source: {outcome.model.provenance.url}",
                        )
                    )
            else:
                policy_errors = model_policy_errors(
                    outcome.model,
                    today=date.fromisoformat(today),
                    max_age_days=None,
                )
                suspicious = suspicious_price_changes(model, outcome.model)
                if policy_errors or suspicious:
                    kept.append(model)
                    flagged.append(
                        _VerificationFlag(
                            identity=identity,
                            reason=decision.reason,
                            note="automated update rejected: "
                            + "; ".join([*policy_errors, *suspicious]),
                        )
                    )
                else:
                    kept.append(outcome.model)
        else:
            kept.append(model)
            flagged.append(
                _VerificationFlag(
                    identity=identity,
                    reason=decision.reason,
                    note=outcome.note,
                )
            )

    if audit_recommendations:
        selected_providers = (
            {identity.split("/", 1)[0] for identity in requested}
            if requested
            else set(RECOMMENDATION_PAGES)
        )
        for provider_name in sorted(selected_providers):
            discovery_attempted += 1
            if verifier is None:
                try:
                    verifier = BrowserVerifier(as_of=today, max_cost_usd=max_cost)
                except Exception as exc:
                    flagged.append(
                        _VerificationFlag(
                            identity=f"{provider_name} recommendation audit",
                            reason="verifier construction failed",
                            note=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    REPORT_PATH.write_text(
                        _format_report(original, original, flagged=flagged, evidence=evidence),
                        encoding="utf-8",
                    )
                    raise
            existing_models = tuple(
                item.model for item in original.models if item.provider_name == provider_name
            )
            try:
                discovered = await verifier.adiscover_recommendations(
                    provider_name, existing_models
                )
            except Exception as exc:
                flagged.append(
                    _VerificationFlag(
                        identity=f"{provider_name} recommendation audit",
                        reason="recommendation discovery failed",
                        note=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            if discovered.usage:
                input_tokens += discovered.usage.get("input_tokens", 0)
                output_tokens += discovered.usage.get("output_tokens", 0)
            if not discovered.verified:
                flagged.append(
                    _VerificationFlag(
                        identity=f"{provider_name} recommendation audit",
                        reason="recommendation discovery inconclusive",
                        note=discovered.note,
                    )
                )
                continue
            discovery_verified += 1
            evidence.append(
                _VerificationEvidence(
                    identity=f"{provider_name} recommendations",
                    source_url=discovered.source_url,
                    quote=discovered.evidence,
                )
            )
            missing = tuple(
                model_id
                for model_id in discovered.models
                if original.resolve(provider_name=provider_name, model=model_id) is None
            )
            if missing:
                missing_recommendations += len(missing)
                flagged.append(
                    _VerificationFlag(
                        identity=f"{provider_name} recommendation audit",
                        reason="recommended model missing from bundled catalog",
                        note=", ".join(missing),
                    )
                )

    candidate_models = tuple(kept)
    records_changed = candidate_models != original.models
    provisional = ModelCatalog(
        catalog_version=today if records_changed else original.catalog_version,
        generated_at=today if records_changed else original.generated_at,
        models=candidate_models,
    )
    # ``model_copy(update=...)`` is intentionally used while merging partial verifier
    # results. Re-parse the complete candidate here so every nested ModelInfo/pricing
    # invariant is enforced before repository data can be written.
    try:
        candidate = ModelCatalog.model_validate(provisional.model_dump(mode="json"))
        # A refresh PR may expose stale, inconclusive records for review, but it may never
        # introduce unsupported providers, non-official sources, or invalid catalog structure.
        validate_catalog(candidate, max_age_days=None)
        # The refresh may not write a catalog that cannot price and run its own maintenance
        # agent. Replacement is a deliberate code + catalog change, never an automated deletion.
        verifier_model_info(
            candidate,
            provider_name=VERIFIER_PROVIDER_NAME,
            model=DEFAULT_VERIFIER_MODEL,
        )
    except (TypeError, ValueError) as exc:
        flagged.append(
            _VerificationFlag(
                identity="candidate catalog",
                reason="candidate validation failed",
                note=f"{type(exc).__name__}: {exc}",
            )
        )
        REPORT_PATH.write_text(
            _format_report(original, provisional, flagged=flagged, evidence=evidence),
            encoding="utf-8",
        )
        raise

    report = _format_report(original, candidate, flagged=flagged, evidence=evidence)
    REPORT_PATH.write_text(report, encoding="utf-8")
    if records_changed:
        CATALOG_PATH.write_text(dump_model_catalog(candidate), encoding="utf-8")

    rate_in, rate_out = _verifier_rates(original)
    cost = input_tokens / 1e6 * rate_in + output_tokens / 1e6 * rate_out
    print(
        f"models={len(kept)} attempted={attempted} verified={verified} "
        f"discovery={discovery_attempted}/{discovery_verified} "
        f"missing_recommendations={missing_recommendations} "
        f"changed={records_changed} flagged={len(flagged)} "
        f"tokens={input_tokens}+{output_tokens} estimated_cost_usd={cost:.4f}"
    )
    if attempted and not verified:
        raise RuntimeError(
            "all attempted model verifications were inconclusive; check provider credentials "
            "and browser access"
        )
    if discovery_attempted and not discovery_verified:
        raise RuntimeError(
            "all recommendation audits were inconclusive; check provider credentials and "
            "browser access"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="force re-verification of every model")
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=VERIFY_MAX_AGE_DAYS,
        help=f"verify records older than this many days (default: {VERIFY_MAX_AGE_DAYS})",
    )
    parser.add_argument(
        "--audit-recommendations",
        action="store_true",
        help="audit official provider recommendation pages for unbundled current models",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PROVIDER/MODEL",
        help="verify only this catalog identity (repeatable; combine with --all to force it)",
    )
    args = parser.parse_args(argv)
    asyncio.run(
        _run(
            force_all=args.all,
            max_age_days=args.max_age_days,
            only=tuple(args.only),
            audit_recommendations=args.audit_recommendations,
        )
    )


if __name__ == "__main__":
    main()
