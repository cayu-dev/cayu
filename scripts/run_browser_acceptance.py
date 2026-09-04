#!/usr/bin/env python3
"""Run Cayu's browser acceptance profile without installing external prerequisites."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from pathlib import Path

from cayu.cli._targets import load_target
from cayu.evals import (
    BROWSER_ACCEPTANCE_REPORT_MAX_BYTES,
    BrowserAcceptanceConformance,
    BrowserAcceptanceFixtureV1,
    BrowserAcceptanceMode,
    BrowserAcceptancePlanV1,
    browser_acceptance_report_from_json,
    deterministic_browser_acceptance_manifest,
    live_public_browser_acceptance_manifest,
    run_browser_acceptance,
    write_browser_acceptance_report,
)

_DETERMINISTIC_TARGET = "cayu.evals.internal.browser_acceptance:build"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        help=(
            "Trusted live-public module:attribute returning BrowserAcceptancePlanV1. "
            "Deterministic mode always uses Cayu's checked-in target."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in BrowserAcceptanceMode),
        default=BrowserAcceptanceMode.DETERMINISTIC.value,
    )
    parser.add_argument(
        "--resume-report",
        type=Path,
        help="Exact prior report whose selected trials should be retried.",
    )
    parser.add_argument(
        "--retry",
        action="append",
        default=[],
        metavar="CASE_ID:TRIAL",
        help="Retry one exact prior trial; repeat for a bounded subset.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("browser-acceptance-results"),
    )
    return parser.parse_args()


async def _load_plan(
    target: str,
    *,
    deterministic_fixture: BrowserAcceptanceFixtureV1 | None = None,
) -> BrowserAcceptancePlanV1:
    loaded = load_target(target, label="Browser acceptance target", normalize_errors=True)
    if callable(loaded):
        loaded = loaded(deterministic_fixture) if deterministic_fixture is not None else loaded()
    if inspect.isawaitable(loaded):
        loaded = await loaded
    if type(loaded) is not BrowserAcceptancePlanV1:
        raise TypeError("Browser acceptance target must return BrowserAcceptancePlanV1.")
    return loaded


async def _run(args: argparse.Namespace) -> int:
    requested_mode = BrowserAcceptanceMode(args.mode)
    if requested_mode is BrowserAcceptanceMode.LIVE_AUTHENTICATED:
        raise RuntimeError("Authenticated browser acceptance is disabled in schema v1.")
    if requested_mode is BrowserAcceptanceMode.DETERMINISTIC:
        if args.target not in {None, _DETERMINISTIC_TARGET}:
            raise ValueError("Deterministic browser acceptance uses Cayu's checked-in target.")
        with BrowserAcceptanceFixtureV1() as fixture:
            plan = await _load_plan(_DETERMINISTIC_TARGET, deterministic_fixture=fixture)
            return await _run_plan(args, plan=plan, deterministic_fixture=fixture)
    if args.target is None:
        raise ValueError("Live-public browser acceptance requires an explicit target.")
    plan = await _load_plan(args.target)
    return await _run_plan(args, plan=plan)


async def _run_plan(
    args: argparse.Namespace,
    *,
    plan: BrowserAcceptancePlanV1,
    deterministic_fixture: BrowserAcceptanceFixtureV1 | None = None,
) -> int:
    requested_mode = BrowserAcceptanceMode(args.mode)
    if plan.manifest.mode is not requested_mode:
        raise ValueError("Browser acceptance target mode conflicts with --mode.")
    canonical_manifest = (
        deterministic_browser_acceptance_manifest()
        if requested_mode is BrowserAcceptanceMode.DETERMINISTIC
        else live_public_browser_acceptance_manifest()
    )
    if plan.manifest != canonical_manifest:
        raise ValueError("Browser acceptance target does not use the canonical manifest.")
    if not plan.manifest.enabled:
        raise RuntimeError("The selected browser acceptance manifest is disabled.")
    resume_report = getattr(args, "resume_report", None)
    retry_values = getattr(args, "retry", [])
    previous_report = (
        None
        if resume_report is None
        else browser_acceptance_report_from_json(_read_bounded_report(resume_report))
    )
    retry_trials = tuple(_retry_trial(value) for value in retry_values)
    if (previous_report is None) != (not retry_trials):
        raise ValueError("--resume-report and at least one --retry must be supplied together.")
    report = await run_browser_acceptance(
        plan,
        deterministic_fixture=deterministic_fixture,
        receipt_directory=args.output_directory / ".trials",
        previous_report=previous_report,
        retry_trials=retry_trials,
    )
    json_path, html_path = write_browser_acceptance_report(report, args.output_directory)
    sys.stdout.write(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "report_revision": report.revision,
                "manifest_revision": report.manifest.revision,
                "runtime_identity_revision": report.runtime_identity.revision,
                "overall_status": report.aggregate.overall_status.value,
                "json_report": str(json_path),
                "html_report": str(html_path),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    return 0 if report.aggregate.overall_status is BrowserAcceptanceConformance.PASSED else 1


def _retry_trial(value: str) -> tuple[str, int]:
    case_id, separator, raw_trial = value.rpartition(":")
    if not separator or not case_id or not raw_trial.isascii() or not raw_trial.isdecimal():
        raise ValueError("--retry must use CASE_ID:TRIAL.")
    trial_number = int(raw_trial)
    if trial_number < 1:
        raise ValueError("--retry trial number must be positive.")
    return case_id, trial_number


def _read_bounded_report(path: Path) -> bytes:
    with path.open("rb") as handle:
        payload = handle.read(BROWSER_ACCEPTANCE_REPORT_MAX_BYTES + 1)
    if len(payload) > BROWSER_ACCEPTANCE_REPORT_MAX_BYTES:
        raise ValueError("Browser acceptance source report exceeds its byte bound.")
    return payload


def main() -> int:
    args = _arguments()
    try:
        return asyncio.run(_run(args))
    except Exception as exc:
        # Targets and browser dependencies are application-controlled. Keep the
        # command diagnostic stable and content-free; detailed bounded evidence
        # belongs in a successfully built report.
        sys.stderr.write(
            f"browser acceptance unavailable ({type(exc).__module__}.{type(exc).__qualname__})\n"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
