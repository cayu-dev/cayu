"""CLI adapter for the native benchmark package and campaign contracts."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from cayu.cli.project import project_context, resolve_eval_project
from cayu.evals.benchmark_campaign import (
    BenchmarkCampaignSettingsV1,
    admit_benchmark_campaign,
    execute_benchmark_campaign,
    prepare_benchmark_campaign,
)
from cayu.evals.benchmark_inspection import (
    inspect_benchmark_campaign,
    render_benchmark_campaign_html,
)
from cayu.evals.benchmark_package import benchmark_suite_selection, load_benchmark_package
from cayu.evals.store import EvalRunCostBudget, EvalRunRecoveryPolicyV1
from cayu.runtime.stop_policy import RunLimits
from cayu.sessions.base import ModelTarget
from cayu.storage.evals_sqlite import SQLiteEvalStore


def add_benchmark_arguments(run: argparse.ArgumentParser, inner) -> None:
    run.add_argument(
        "--package",
        metavar="PATH",
        help="Run a portable benchmark package through native durable Evals.",
    )
    run.add_argument(
        "--case",
        dest="benchmark_case_ids",
        action="append",
        metavar="CASE_ID",
        help="Explicit packaged-benchmark case selection; repeat to select multiple cases.",
    )
    run.add_argument(
        "--campaign-directory",
        metavar="NEW_DIRECTORY",
        help="New private directory for native campaign admission, catalog, and inspection.",
    )
    run.add_argument(
        "--admit-only",
        action="store_true",
        help="Persist a packaged campaign without dispatching it; use eval resume to start it.",
    )
    run.add_argument("--trials", type=int, help="Independent trials per package case.")
    run.add_argument(
        "--minimum-passed-trials",
        type=int,
        help="Passed trials needed by the package's native trial policy.",
    )
    run.add_argument(
        "--max-steps", type=int, help="Narrow the packaged target's per-trial model-step limit."
    )
    run.add_argument(
        "--max-total-tokens",
        type=int,
        help="Per-trial observed-token stop threshold; in-flight usage can overshoot.",
    )
    run.add_argument(
        "--max-tool-calls",
        type=int,
        help="Narrow the packaged target's per-trial tool-call allowance.",
    )
    run.add_argument(
        "--max-estimated-cost",
        type=Decimal,
        help="Per-trial priced budget; requires the target's trusted PriceBook.",
    )
    run.add_argument("--currency", help="Currency for --max-estimated-cost (default: USD).")
    run.add_argument(
        "--provider", help="Registered provider for the packaged run; requires --model."
    )
    run.add_argument("--model", help="Model on the explicitly selected --provider.")
    run.add_argument("--environment", help="Existing target environment for the packaged run.")
    run.add_argument(
        "--max-execution-attempts",
        type=int,
        default=None,
        help="Explicitly authorize up to this many full execution attempts after interruption (1-10).",
    )
    run.add_argument(
        "--max-retry-attempts",
        type=int,
        default=None,
        help="Reserve up to 3 selective successor trials per original slot; default 0.",
    )
    run.add_argument(
        "--retry-category",
        action="append",
        dest="retry_categories",
        help="Eligible typed failure category; repeat. Wrong answers are excluded by default.",
    )
    run.add_argument(
        "--retry-backoff-seconds",
        type=int,
        default=None,
        help="Minimum elapsed time after the preceding attempt, 0-3600 seconds.",
    )
    resume = inner.add_parser(
        "resume",
        help="Resume a native benchmark campaign using its saved admission.",
        description="Resume saved native work under its admitted recovery policy. Inspect it with `cayu eval status DIRECTORY`.",
    )
    resume.add_argument("directory")
    resume.add_argument("target", nargs="?")
    cancel = inner.add_parser(
        "cancel",
        help="Request native cancellation for a benchmark campaign.",
        description="Request cancellation of admitted campaign runs. Inspect settlement with `cayu eval status DIRECTORY`.",
    )
    cancel.add_argument("directory")
    rescore = inner.add_parser(
        "rescore",
        help="Evaluate a new static scorer using saved output, without candidate or judge calls.",
        description="Apply a new static scorer package to retained output. Review the new receipt at --output; original results remain available through eval status.",
    )
    rescore.add_argument("directory")
    rescore.add_argument("--package", required=True, dest="scorer_package")
    rescore.add_argument(
        "--output", required=True, help="New scorer receipt outside the original campaign."
    )
    retry = inner.add_parser(
        "retry",
        help="Execute a bounded, explicitly selected successor trial.",
        description="Retry one eligible original trial within its reserved allowance. Inspect original and successor evidence with `cayu eval status DIRECTORY`.",
    )
    retry.add_argument("directory")
    retry.add_argument("target", nargs="?")
    retry.add_argument("--case", required=True, dest="case_id")
    retry.add_argument("--trial", required=True, type=int)
    retry.add_argument("--attempt", type=int, default=1)
    retry.add_argument(
        "--allow-reexecution",
        action="store_true",
        help="Explicit decision to replay when the target has no application reset contract.",
    )
    package = inner.add_parser(
        "package",
        help="Create, validate, and inspect portable benchmark packages.",
        description="Manage portable native benchmark definitions. Start with `cayu eval package init DIRECTORY`, then validate the package.",
    )
    commands = package.add_subparsers(dest="package_command", required=True)
    initialize = commands.add_parser(
        "init",
        help="Create the installed synthetic benchmark package in a new directory.",
        description="Create a synthetic benchmark package without external calls. Next use `cayu eval package validate DIRECTORY` or launch with `cayu eval run --package DIRECTORY`.",
    )
    initialize.add_argument("directory")
    initialize.add_argument(
        "--failure-modes",
        action="store_true",
        help="Include synthetic provider and scoring failures.",
    )
    initialize.add_argument(
        "--interruption-case",
        action="store_true",
        help="Include a synthetic pending call for explicit process-loss qualification.",
    )
    initialize.add_argument(
        "--scorer-version",
        default="1",
        help="Explicit synthetic scorer version for saved-output qualification.",
    )
    for name in ("validate", "inspect"):
        command = commands.add_parser(
            name,
            help=f"{name.capitalize()} a package without importing any execution target.",
            description=f"{name.capitalize()} exact package material and cohort identity. Launch a validated package with `cayu eval run TARGET --package PATH`.",
        )
        command.add_argument("path")
        command.add_argument("--case", dest="case_ids", action="append")


def require_package_for_benchmark_options(args) -> None:
    if any(
        getattr(args, name, None) is not None
        for name in (
            "benchmark_case_ids",
            "campaign_directory",
            "trials",
            "minimum_passed_trials",
            "max_steps",
            "max_total_tokens",
            "max_tool_calls",
            "max_estimated_cost",
            "currency",
            "provider",
            "model",
            "environment",
            "max_execution_attempts",
            "max_retry_attempts",
            "retry_categories",
            "retry_backoff_seconds",
        )
    ) or getattr(args, "admit_only", False):
        raise ValueError("Benchmark campaign options require --package PATH.")


def package_command(args) -> int:
    if args.package_command == "init":
        from cayu.evals.benchmark_synthetic import write_synthetic_benchmark_package

        path = write_synthetic_benchmark_package(
            args.directory,
            failure_modes=args.failure_modes,
            interruption_case=args.interruption_case,
            scorer_version=args.scorer_version,
        )
        print(f"Created synthetic benchmark package: {path}")
        return 0
    loaded = load_benchmark_package(args.path)
    suite, selection = benchmark_suite_selection(loaded.package, args.case_ids)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "package_id": loaded.package.id,
                "version": loaded.package.version,
                "revision": loaded.package.revision,
                "target_key": suite.target_key,
                "scorer_id": loaded.package.scorer_id,
                "scorer_version": loaded.package.scorer_version,
                "selection": selection.model_dump(mode="json"),
                "input_files": len(loaded.file_contents),
                "input_bytes": sum(len(content) for _, content in loaded.file_contents),
                "requirements": loaded.package.requirements.model_dump(mode="json"),
            },
            indent=2,
        )
    )
    return 0


def _settings(args) -> BenchmarkCampaignSettingsV1:
    if (args.provider is None) != (args.model is None):
        raise ValueError("Select both --provider and --model for an explicit model override.")
    if args.currency is not None and args.max_estimated_cost is None:
        raise ValueError("--currency requires --max-estimated-cost.")
    timeout = args.case_timeout_seconds
    if timeout is not None:
        if not timeout.is_integer():
            raise ValueError("Packaged case timeouts require a whole number of seconds.")
        timeout = int(timeout)
    limits = None
    if args.max_total_tokens is not None or args.max_tool_calls is not None:
        limits = RunLimits(
            max_total_tokens=args.max_total_tokens, max_tool_calls=args.max_tool_calls
        )
    return BenchmarkCampaignSettingsV1(
        trials=args.trials,
        minimum_passed_trials=args.minimum_passed_trials,
        max_concurrency=args.max_concurrency,
        case_timeout_seconds=timeout,
        max_steps=args.max_steps,
        limits=limits,
        cost_budget=None
        if args.max_estimated_cost is None
        else EvalRunCostBudget(
            max_estimated_cost=args.max_estimated_cost, currency=args.currency or "USD"
        ),
        model_target=None
        if args.model is None
        else ModelTarget(provider_name=args.provider, model=args.model),
        environment_name=args.environment,
        stagger_seconds=args.stagger_seconds,
        recovery_policy=EvalRunRecoveryPolicyV1(
            mode="caller_authorized"
            if (args.max_execution_attempts or 1) > 1
            else "checkpoint_only",
            max_execution_attempts=args.max_execution_attempts
            if args.max_execution_attempts is not None
            else 1,
        ),
        max_retry_attempts=args.max_retry_attempts if args.max_retry_attempts is not None else 0,
        retry_backoff_seconds=args.retry_backoff_seconds
        if args.retry_backoff_seconds is not None
        else 0,
        retry_categories=tuple(args.retry_categories)
        if args.retry_categories is not None
        else BenchmarkCampaignSettingsV1().retry_categories,
    )


def _protect_outputs(
    args, directory: Path, protected: tuple[tuple[str, str | None], ...] = ()
) -> None:
    from cayu.cli.evals import _reject_output_path_aliases

    outputs = (("--output", args.output), ("--html-output", getattr(args, "html_output", None)))
    for _, output in outputs:
        if output is not None and Path(output).resolve().is_relative_to(directory.resolve()):
            raise ValueError("Benchmark report output must be outside its campaign directory.")
    _reject_output_path_aliases(outputs=outputs, protected=protected)


def benchmark_protected_stores(inspection):
    paths = {
        trial.session_sqlite_path
        for trial in inspection.trials
        if trial.session_sqlite_path is not None
    }
    paths.update(
        reference["sqlite_path"]
        for trial in inspection.trials
        for reference in trial.prior_session_references
        if reference.get("sqlite_path") is not None
    )
    return tuple(
        ("linked session store", str(path) + suffix)
        for path in paths
        for suffix in ("", "-wal", "-shm", "-journal")
    )


async def run_benchmark(args) -> int:
    from cayu.cli.evals import _load_eval_plan, _write_or_print

    if args.corpus is not None or args.suite is not None:
        raise ValueError("--package owns its suite; do not combine it with --corpus or --suite.")
    if args.processes != 1 or args.process_directory is not None:
        raise ValueError(
            "Packaged campaigns use durable native workers; process launch is unsupported."
        )
    settings = _settings(args)
    project = resolve_eval_project(args.target)
    with project_context(project.root):
        loaded = load_benchmark_package(args.package)
        directory = Path(args.campaign_directory or f".cayu/evals/campaigns/{uuid4()}").resolve()
        protected = [
            ("package manifest", str(loaded.manifest_path or loaded.root / "benchmark.json"))
        ]
        protected.extend(
            ("package input", str(loaded.root / binding.path))
            for binding, _ in loaded.file_contents
        )
        if directory.exists():
            raise ValueError(
                "Campaign directory already exists; use eval resume for an admitted campaign."
            )
        _protect_outputs(args, directory, tuple(protected))
        plan = await _load_eval_plan(project.target, label="Benchmark eval target")
        target = plan.corpus_target or plan.workflow_target
        if target is not None:
            protected.extend(
                ("target store", str(path) + suffix)
                for path in target.app.session_store.durable_state_paths()
                for suffix in ("", "-wal", "-shm", "-journal")
            )
        _protect_outputs(args, directory, tuple(protected))
        prepared = await prepare_benchmark_campaign(
            loaded, plan, settings=settings, case_ids=args.benchmark_case_ids
        )
        campaign = await admit_benchmark_campaign(prepared, directory)
        exposure = campaign.runs[0].spec.invocation.authored_suite_exposure
        assert exposure is not None
        print(
            f"Campaign: {campaign.id}\nPackage: {campaign.package_id}@{campaign.package_version} {campaign.package_revision}\nCohort: {campaign.selection.revision}\nAdmission: {directory / 'campaign.json'}\nInspect: cayu eval status {directory}",
            file=sys.stderr,
        )
        print(
            json.dumps(
                {
                    "settings": settings.model_dump(mode="json"),
                    "maximum_work": exposure.model_dump(mode="json"),
                    "maximum_candidate_attempts_including_recovery_and_retries": exposure.candidate_trials
                    * settings.recovery_policy.max_execution_attempts
                    * (1 + settings.max_retry_attempts),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        if not args.admit_only:
            store = SQLiteEvalStore(directory / "evals.sqlite3")
            try:
                await execute_benchmark_campaign(
                    campaign, directory, registry=prepared.registry, store=store
                )
            finally:
                await store.close()
        inspection = await inspect_benchmark_campaign(directory)
        _write_or_print(inspection.model_dump_json(indent=2), args.output)
        if args.html_output is not None:
            Path(args.html_output).write_text(
                render_benchmark_campaign_html(inspection), encoding="utf-8"
            )
        return (
            0
            if args.admit_only or inspection.status == "passed"
            else 1
            if inspection.status == "failed"
            else 2
        )


async def inspect_benchmark(args) -> int:
    from cayu.cli.evals import _write_or_print

    if args.session_evidence is not None:
        raise ValueError(
            "Campaign session links come from exact native observations; --session-evidence is a process-run option."
        )
    if Path(args.directory).is_file():
        from cayu.evals.benchmark_inspection import load_benchmark_export

        if args.sessions:
            raise ValueError("Offline export inspection cannot open linked session stores.")
        inspection = load_benchmark_export(args.directory)
    else:
        inspection = await inspect_benchmark_campaign(
            args.directory,
            include_sessions=args.sessions,
            case_id=args.case_id,
            max_sessions=args.max_sessions,
            max_diagnostics=args.max_diagnostics,
        )
    protected = benchmark_protected_stores(inspection)
    if args.case_id is not None and args.case_id not in {
        case.id for case in inspection.campaign.selection.cases
    }:
        raise ValueError("Requested case is not in the campaign cohort.")
    if Path(args.directory).is_file():
        protected += (("benchmark export", args.directory),)
    _protect_outputs(args, Path(args.directory), protected)
    rows = tuple(
        trial
        for trial in inspection.trials
        if (args.case_id is None or trial.case_id == args.case_id)
        and (args.eval_command != "failures" or trial.status not in {"passed", "pending"})
    )
    if args.output_format == "json":
        document = inspection.model_dump(mode="json")
        document["trials"] = [trial.model_dump(mode="json") for trial in rows]
        _write_or_print(json.dumps(document, indent=2), args.output)
    else:
        text = [
            f"{inspection.campaign.id}: {inspection.status}",
            json.dumps(inspection.counts, sort_keys=True),
            "CASE\tTRIAL\tSTATUS\tCATEGORY\tSCORE\tEVIDENCE\tSESSION",
        ]
        text.extend(
            "\t".join(
                str(value)
                for value in (
                    trial.case_id,
                    trial.trial_number,
                    trial.status,
                    trial.failure_category,
                    trial.score if trial.score is not None else "unavailable",
                    trial.evidence_state,
                    trial.session_id or "unavailable",
                )
            )
            for trial in rows
        )
        _write_or_print("\n".join(text), args.output)
    return 0


async def maintain_benchmark(args) -> int:
    from cayu.cli.evals import _load_eval_plan
    from cayu.evals.benchmark_campaign import load_benchmark_campaign, resume_benchmark_campaign
    from cayu.evals.benchmark_retry import retry_benchmark_trial

    if args.eval_command == "rescore":
        from cayu.evals.benchmark_rescore import rescore_benchmark_campaign

        _protect_outputs(args, Path(args.directory))
        if Path(args.output).exists():
            raise ValueError("Rescore output must be a new file.")
        result = await rescore_benchmark_campaign(args.directory, args.scorer_package)
        with Path(args.output).open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        print(f"Saved static scorer receipt: {args.output}")
        return (
            2
            if any(
                assertion["outcome"] in {"unavailable", "error"}
                for row in result["trials"]
                for assertion in row["assertions"]
            )
            else 0
        )

    if args.eval_command == "cancel":
        campaign = load_benchmark_campaign(args.directory)
        store = SQLiteEvalStore(Path(args.directory) / "evals.sqlite3")
        try:
            for run in campaign.runs:
                await store.request_cancel(run.spec.id)
        finally:
            await store.close()
        print(f"Cancellation requested: {campaign.id}")
        return 0
    project = resolve_eval_project(args.target)
    with project_context(project.root):
        plan = await _load_eval_plan(project.target, label="Benchmark eval target")
        directory = Path(args.directory)
        if args.eval_command == "resume":
            await resume_benchmark_campaign(directory, plan)
        else:
            directory = await retry_benchmark_trial(
                directory,
                plan,
                case_id=args.case_id,
                trial_number=args.trial,
                attempt=args.attempt,
                allow_reexecution=args.allow_reexecution,
            )
        inspection = await inspect_benchmark_campaign(directory)
        print(inspection.model_dump_json(indent=2))
        return 0 if inspection.status == "passed" else 1 if inspection.status == "failed" else 2
