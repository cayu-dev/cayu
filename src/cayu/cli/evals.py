from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from cayu.cli._output import add_output_options
from cayu.cli._targets import TargetResolutionError, load_target
from cayu.cli.project import project_context, resolve_eval_project
from cayu.evals import (
    CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES,
    EVAL_RESULT_REPORT_MAX_BYTES,
    MEMORY_EXPERIMENT_REPORT_MAX_BYTES,
    CapturedEvaluationResultV1,
    CorpusExecutionResult,
    CorpusTarget,
    EvalCorpusDocument,
    EvalPlan,
    EvalRun,
    EvalStatus,
    EvalSuite,
    MemoryExperimentReport,
    build_memory_experiment_report,
    captured_evaluation_result_from_json,
    compare_eval_results,
    compare_eval_runs,
    comparison_to_json,
    corpus_execution_comparison_to_json,
    corpus_execution_result_to_json,
    eval_corpus_inspection_to_json,
    eval_result_report_from_json,
    eval_result_report_to_json,
    eval_run_to_json,
    inspect_eval_corpus,
    load_corpus_execution_result,
    load_eval_corpus,
    load_eval_run,
    memory_experiment_report_from_json,
    memory_experiment_report_to_json,
    memory_experiment_request_from_json,
    merge_eval_corpus_files,
    present_eval_result,
    render_comparison_html,
    render_corpus_execution_comparison_html,
    render_corpus_execution_html,
    render_eval_result_html,
    render_html_report,
    render_memory_experiment_report_html,
    run_eval_plan,
)
from cayu.runtime.app import CayuApp


def add_eval_parser(subparsers: Any) -> None:
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run and report Cayu runtime-native evals.",
        description=(
            "Run and report Cayu runtime-native evals. Start with `cayu eval run` "
            "for the project-configured hermetic proof."
        ),
    )
    inner = eval_parser.add_subparsers(dest="eval_command", required=True)

    run = inner.add_parser(
        "run",
        help="Run a configured or explicit eval plan.",
        description=(
            "Run a configured or explicit eval plan and emit a stable JSON result. "
            "Use `--output FILE` to save it."
        ),
    )
    run.add_argument(
        "target",
        nargs="?",
        help=(
            "Python target that returns a direct-suite or corpus-target EvalPlan. "
            "Defaults to [tool.cayu].eval_target."
        ),
    )
    add_output_options(run, formats=("json",))
    run.add_argument("--html-output", metavar="FILE", help="Also write an HTML report to FILE.")
    run.add_argument(
        "--corpus",
        metavar="FILE",
        help="Run a portable corpus through the target's trusted CorpusTarget.",
    )
    run.add_argument(
        "--suite",
        metavar="SUITE_ID",
        help="Corpus suite to run (optional only when the corpus has one suite).",
    )
    run.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        metavar="COUNT",
        help="Maximum concurrently executing cases (default: 1).",
    )
    run.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Limit each direct-suite case to SECONDS; corpus timeouts are declared in JSON.",
    )

    report = inner.add_parser(
        "report",
        help="Render a JSON or HTML report from eval results.",
        description=(
            "Render saved eval results as HTML by default or JSON explicitly. "
            "Use `--output FILE` to save the report."
        ),
    )
    report.add_argument("input", metavar="RESULTS_JSON", help="Eval JSON results file.")
    add_output_options(report, formats=("html", "json"), default="html")

    memory_report = inner.add_parser(
        "memory-report",
        help="Build a paired repeated-trial memory experiment report.",
        description=(
            "Validate an exact memory experiment request and build its deterministic "
            "fixed-candidate report without launching additional work."
        ),
    )
    memory_report.add_argument(
        "input",
        metavar="REQUEST_JSON",
        help="Memory experiment report request JSON file.",
    )
    add_output_options(memory_report, formats=("html", "json"), default="json")

    compare = inner.add_parser(
        "compare",
        help="Compare baseline and current eval results.",
        description=(
            "Compare baseline and current eval results. JSON is the default; "
            "a nonzero exit reports regressions."
        ),
    )
    compare.add_argument("baseline", metavar="BASELINE_JSON")
    compare.add_argument("current", metavar="CURRENT_JSON")
    add_output_options(compare, formats=("html", "json"))
    compare.add_argument(
        "--score-tolerance",
        type=float,
        default=0.0,
        metavar="DELTA",
        help="Allowed score drop before a regression is flagged (default: 0.0).",
    )

    validate = inner.add_parser(
        "validate",
        help="Validate a portable eval corpus.",
        description=(
            "Validate one bounded portable eval corpus without loading an application target."
        ),
    )
    validate.add_argument("corpus", metavar="CORPUS_JSON")
    add_output_options(validate, formats=("json", "table"), default="table")

    inspect_parser = inner.add_parser(
        "inspect",
        help="Inspect a validated portable eval corpus.",
        description=(
            "Show the target, suites, cases, assertions, and expanded result count for one "
            "validated corpus."
        ),
    )
    inspect_parser.add_argument("corpus", metavar="CORPUS_JSON")
    add_output_options(inspect_parser, formats=("json", "table"), default="table")

    merge = inner.add_parser(
        "merge",
        help="Atomically merge portable eval corpora.",
        description=(
            "Validate and merge compatible corpus files, then atomically replace the destination."
        ),
    )
    merge.add_argument("destination", metavar="DESTINATION_JSON")
    merge.add_argument("inputs", nargs="+", metavar="CORPUS_JSON")
    merge.add_argument(
        "--replace-conflicts",
        action="store_true",
        help="Replace same-ID content conflicts in command-line order.",
    )
    add_output_options(merge, formats=("json", "table"), default="table")


def run_eval_command(args: argparse.Namespace) -> int:
    try:
        if args.eval_command == "run":
            return asyncio.run(_run(args))
        if args.eval_command == "report":
            return _report(args)
        if args.eval_command == "memory-report":
            return _memory_report(args)
        if args.eval_command == "compare":
            return _compare(args)
        if args.eval_command == "validate":
            return _validate_corpus(args)
        if args.eval_command == "inspect":
            return _inspect_corpus(args)
        if args.eval_command == "merge":
            return _merge_corpora(args)
    except Exception as exc:
        if getattr(args, "output_format", None) == "json":
            print(
                json.dumps(
                    {
                        "schema_version": "1",
                        "error": {"code": "EVAL_COMMAND_FAILED", "message": str(exc)},
                    },
                    sort_keys=True,
                )
            )
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


async def _run(args: argparse.Namespace) -> int:
    project = resolve_eval_project(args.target)
    label = (
        "Command-line eval target"
        if args.target is not None
        else f"Configured eval target from {project.root / 'pyproject.toml'}"
    )
    with project_context(project.root):
        # Relative eval paths are interpreted from the resolved project root,
        # so collision checks must run in that same filesystem context.
        _reject_output_path_aliases(
            outputs=(
                ("--output", args.output),
                ("--html-output", args.html_output),
            ),
            protected=(("--corpus", args.corpus),),
        )
        plan = await _load_eval_plan(project.target, label=label)
        if args.corpus is None:
            if plan.corpus_target is not None:
                raise ValueError("Corpus EvalPlan execution requires --corpus FILE.")
            if args.suite is not None:
                raise ValueError("--suite requires --corpus FILE.")
            run = await run_eval_plan(
                plan,
                max_concurrency=args.max_concurrency,
                case_timeout_seconds=args.case_timeout_seconds,
            )
            if type(run) is not EvalRun:
                raise TypeError("Direct EvalPlan returned an unexpected corpus result.")
            output = eval_run_to_json(run)
            _write_or_print(output, args.output)
            if args.html_output is not None:
                Path(args.html_output).write_text(render_html_report(run), encoding="utf-8")
            return _status_exit_code(run.status)

        if plan.corpus_target is None:
            raise ValueError("--corpus requires an EvalPlan configured with corpus_target.")
        if args.case_timeout_seconds is not None:
            raise ValueError("Corpus timeout comes from the corpus; omit --case-timeout-seconds.")
        corpus = load_eval_corpus(args.corpus)
        suite_id = _selected_suite_id(corpus, args.suite)
        result = await run_eval_plan(
            plan,
            corpus=corpus,
            suite_id=suite_id,
            max_concurrency=args.max_concurrency,
        )
        if type(result) is not CorpusExecutionResult:
            raise TypeError("Corpus EvalPlan returned an unexpected direct result.")
        _write_or_print(corpus_execution_result_to_json(result), args.output)
        if args.html_output is not None:
            Path(args.html_output).write_text(
                render_corpus_execution_html(result),
                encoding="utf-8",
            )
        return _status_exit_code(result.run.status)


async def _load_eval_plan(target: str, *, label: str) -> EvalPlan:
    try:
        loaded = load_target(target, label=label)
    except TargetResolutionError:
        raise
    except Exception as exc:
        raise RuntimeError(f"{label} could not be loaded ({type(exc).__name__}): {exc}") from exc

    try:
        if callable(loaded):
            loaded = loaded()
        if inspect.isawaitable(loaded):
            loaded = await loaded
    except Exception as exc:
        raise RuntimeError(f"{label} failed ({type(exc).__name__}): {exc}") from exc

    try:
        return _coerce_plan(loaded)
    except Exception as exc:
        raise TypeError(f"{label} returned an invalid eval plan: {exc}") from exc


def _report(args: argparse.Namespace) -> int:
    result = _load_saved_eval_result(args.input)
    if type(result) is CapturedEvaluationResultV1:
        output = (
            eval_result_report_to_json(result)
            if args.output_format == "json"
            else render_eval_result_html(result)
        )
        _write_or_print(output, args.output)
        return 0
    if type(result) is CorpusExecutionResult:
        output = (
            eval_result_report_to_json(result)
            if args.output_format == "json"
            else render_corpus_execution_html(result)
        )
        _write_or_print(output, args.output)
        return 0
    if type(result) is MemoryExperimentReport:
        output = (
            memory_experiment_report_to_json(result)
            if args.output_format == "json"
            else render_memory_experiment_report_html(result)
        )
        _write_or_print(output, args.output)
        return 0
    if type(result) is not EvalRun:
        raise TypeError("Unsupported eval result document type.")
    output = (
        eval_run_to_json(result) if args.output_format == "json" else render_html_report(result)
    )
    _write_or_print(output, args.output)
    return 0


def _memory_report(args: argparse.Namespace) -> int:
    with Path(args.input).open("rb") as handle:
        request = memory_experiment_request_from_json(
            handle.read(MEMORY_EXPERIMENT_REPORT_MAX_BYTES + 1)
        )
    report = build_memory_experiment_report(request)
    output = (
        memory_experiment_report_to_json(report)
        if args.output_format == "json"
        else render_memory_experiment_report_html(report)
    )
    _write_or_print(output, args.output)
    return 0


def _compare(args: argparse.Namespace) -> int:
    baseline = _load_saved_eval_result(args.baseline)
    current = _load_saved_eval_result(args.current)
    published_types = {CorpusExecutionResult, CapturedEvaluationResultV1}
    if type(baseline) in published_types and type(current) in published_types:
        baseline_result = cast(
            "CorpusExecutionResult | CapturedEvaluationResultV1",
            baseline,
        )
        current_result = cast(
            "CorpusExecutionResult | CapturedEvaluationResultV1",
            current,
        )
        comparison = compare_eval_results(
            baseline_result,
            current_result,
            score_tolerance=args.score_tolerance,
        )
        output = (
            corpus_execution_comparison_to_json(comparison)
            if args.output_format == "json"
            else render_corpus_execution_comparison_html(comparison)
        )
        _write_or_print(output, args.output)
        if not comparison.compatibility.comparable:
            return 2
        if comparison.structured_judge_comparison_state in {
            "observation_identity_mismatch",
            "source_detail_unavailable",
        }:
            return 2
        if comparison.tool_json_comparison_state in {
            "observation_identity_mismatch",
            "source_detail_unavailable",
        }:
            return 2
        if _status_exit_code(comparison.baseline.status) == 2:
            return 2
        current_exit = _status_exit_code(comparison.current.status)
        if current_exit == 2:
            return 2
        for published_result in (baseline_result, current_result):
            dimensions = present_eval_result(published_result).dimensions
            if (
                dimensions.evaluator_health in {"error", "unavailable"}
                or dimensions.runtime in {"failed", "unavailable"}
                or dimensions.evidence in {"incomplete", "unavailable"}
            ):
                return 2
        if (
            current_exit == 1
            or comparison.regressions
            or any(item.regressed for item in comparison.structured_judgments)
            or any(item.regressed for item in comparison.tool_json_assertions)
        ):
            return 1
        return 0

    if type(baseline) is not type(current):
        raise ValueError(
            "Cannot compare direct EvalRun and captured/fresh published result documents."
        )

    if type(baseline) is not EvalRun or type(current) is not EvalRun:
        raise TypeError("Unsupported eval result document type.")
    comparison = compare_eval_runs(
        baseline,
        current,
        score_tolerance=args.score_tolerance,
    )
    output = (
        comparison_to_json(comparison)
        if args.output_format == "json"
        else render_comparison_html(comparison)
    )
    _write_or_print(output, args.output)
    if _status_exit_code(baseline.status) == 2:
        return 2
    current_exit = _status_exit_code(current.status)
    if current_exit == 2:
        return 2
    if current_exit == 1 or comparison.regressions:
        return 1
    return 0


def _status_exit_code(status: EvalStatus | str) -> int:
    value = status.value if isinstance(status, EvalStatus) else status
    if value == "passed":
        return 0
    if value == "failed":
        return 1
    return 2


def _load_saved_eval_result(
    path: str,
) -> EvalRun | CorpusExecutionResult | CapturedEvaluationResultV1 | MemoryExperimentReport:
    result_path = Path(path)
    with result_path.open("rb") as handle:
        raw = handle.read(
            max(CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES, EVAL_RESULT_REPORT_MAX_BYTES) + 1
        )
    if len(raw) > EVAL_RESULT_REPORT_MAX_BYTES:
        raise ValueError(f"Eval result JSON exceeds {EVAL_RESULT_REPORT_MAX_BYTES} bytes.")
    try:
        document = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("Eval result JSON must be UTF-8.") from exc
    if not isinstance(document, dict):
        raise ValueError("Eval result JSON must be an object.")
    if document.get("record_type") == "cayu.eval-result-report":
        return eval_result_report_from_json(raw).result
    if document.get("origin") == "captured_session":
        return captured_evaluation_result_from_json(raw.decode("utf-8"))
    if document.get("record_type") == "cayu.memory-experiment-report":
        return memory_experiment_report_from_json(raw)
    if document.get("schema_version") == 1 or {"target", "run"} <= set(document):
        return load_corpus_execution_result(result_path)
    return load_eval_run(result_path)


def _validate_corpus(args: argparse.Namespace) -> int:
    _reject_output_path_aliases(
        outputs=(("--output", args.output),),
        protected=(("corpus input", args.corpus),),
    )
    inspection = inspect_eval_corpus(load_eval_corpus(args.corpus))
    output = (
        eval_corpus_inspection_to_json(inspection)
        if args.output_format == "json"
        else f"Valid eval corpus {inspection.revision}\n"
    )
    _write_or_print(output, args.output)
    return 0


def _inspect_corpus(args: argparse.Namespace) -> int:
    _reject_output_path_aliases(
        outputs=(("--output", args.output),),
        protected=(("corpus input", args.corpus),),
    )
    inspection = inspect_eval_corpus(load_eval_corpus(args.corpus))
    output = (
        eval_corpus_inspection_to_json(inspection)
        if args.output_format == "json"
        else _corpus_inspection_table(inspection)
    )
    _write_or_print(output, args.output)
    return 0


def _merge_corpora(args: argparse.Namespace) -> int:
    _reject_output_path_aliases(
        outputs=(("--output", args.output),),
        protected=(
            ("merge destination", args.destination),
            *(("merge input", path) for path in args.inputs),
        ),
    )
    corpus = merge_eval_corpus_files(
        args.destination,
        tuple(args.inputs),
        replace_conflicts=args.replace_conflicts,
    )
    inspection = inspect_eval_corpus(corpus)
    output = (
        eval_corpus_inspection_to_json(inspection)
        if args.output_format == "json"
        else _corpus_inspection_table(inspection)
    )
    _write_or_print(output, args.output)
    return 0


def _selected_suite_id(corpus: EvalCorpusDocument, requested: str | None) -> str:
    if requested is not None:
        if not any(suite.id == requested for suite in corpus.suites):
            raise ValueError(f"Eval corpus does not contain suite {requested!r}.")
        return requested
    if len(corpus.suites) != 1:
        choices = ", ".join(suite.id for suite in corpus.suites)
        raise ValueError(f"--suite is required; available corpus suites: {choices}.")
    return corpus.suites[0].id


def _corpus_inspection_table(inspection: Any) -> str:
    lines = [
        f"Corpus: {inspection.revision}",
        f"Target: {inspection.target_key}",
        f"Suites: {inspection.suite_count}",
        f"Cases: {inspection.case_count}",
        f"Assertions: {inspection.assertion_count}",
        f"Expanded results: {inspection.expanded_assertion_result_count}",
        "",
    ]
    lines.extend(
        f"{suite.id}: {suite.case_count} case(s), {suite.assertion_count} assertion(s), "
        f"{suite.trials} trial(s), {suite.timeout_seconds}s timeout"
        for suite in inspection.suites
    )
    return "\n".join(lines) + "\n"


def _coerce_plan(value: Any) -> EvalPlan:
    if type(value) is EvalPlan:
        if value.corpus_target is not None:
            return EvalPlan(corpus_target=value.corpus_target)
        return _validate_plan(value.app, value.suite)
    if isinstance(value, tuple | list) and len(value) == 2:
        app, suite = value
        return _validate_plan(app, suite)
    app = getattr(value, "app", None)
    suite = getattr(value, "suite", None)
    corpus_target = getattr(value, "corpus_target", None)
    if (app is not None or suite is not None) and corpus_target is not None:
        raise ValueError("Eval target cannot configure direct and corpus modes together.")
    if app is not None or suite is not None:
        return _validate_plan(app, suite)
    if corpus_target is not None:
        return _validate_corpus_plan(corpus_target)
    if isinstance(value, dict):
        has_direct = "app" in value or "suite" in value
        has_corpus = "corpus_target" in value
        if has_direct and has_corpus:
            raise ValueError("Eval target cannot configure direct and corpus modes together.")
        if has_direct:
            return _validate_plan(value.get("app"), value.get("suite"))
        if has_corpus:
            return _validate_corpus_plan(value["corpus_target"])
    raise TypeError(
        "Eval target must return EvalPlan, (CayuApp, EvalSuite), app/suite attributes, "
        "or a corpus_target attribute."
    )


def _validate_plan(app: Any, suite: Any) -> EvalPlan:
    if not isinstance(app, CayuApp):
        raise TypeError("Eval plan app must be a CayuApp.")
    if type(suite) is not EvalSuite:
        suite = EvalSuite.model_validate(suite)
    return EvalPlan(app=app, suite=suite)


def _validate_corpus_plan(target: Any) -> EvalPlan:
    if type(target) is not CorpusTarget:
        raise TypeError("Eval plan corpus_target must be an exact CorpusTarget.")
    return EvalPlan(corpus_target=target)


def _write_or_print(content: str, path: str | None) -> None:
    if path is None:
        print(content, end="")
        return
    Path(path).write_text(content, encoding="utf-8")


def _reject_output_path_aliases(
    *,
    outputs: tuple[tuple[str, str | None], ...],
    protected: tuple[tuple[str, str | None], ...],
) -> None:
    selected_outputs = tuple((label, path) for label, path in outputs if path is not None)
    for index, (label, path) in enumerate(selected_outputs):
        for other_label, other_path in selected_outputs[index + 1 :]:
            if _paths_alias(path, other_path):
                raise ValueError(f"{label} and {other_label} must use different files.")
        for protected_label, protected_path in protected:
            if protected_path is not None and _paths_alias(path, protected_path):
                raise ValueError(f"{label} must not overwrite {protected_label}.")


def _paths_alias(first: str, second: str) -> bool:
    first_path = Path(first)
    second_path = Path(second)
    try:
        if (
            first_path.exists()
            and second_path.exists()
            and os.path.samefile(
                first_path,
                second_path,
            )
        ):
            return True
    except OSError:
        pass
    try:
        return first_path.resolve(strict=False) == second_path.resolve(strict=False)
    except OSError:
        return Path(os.path.abspath(first_path)) == Path(os.path.abspath(second_path))
