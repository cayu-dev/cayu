#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

from cayu.evals.causal_memory_campaign import (
    load_causal_memory_reference_corpus,
    run_causal_memory_reference_campaign,
)
from cayu.evals.memory_reporting import (
    MemoryExperimentReport,
    memory_experiment_report_to_json,
    render_memory_experiment_report_html,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Cayu's API-key-free causal-memory reference campaign through the "
            "canonical runtime and published comparison path."
        )
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/memory/causal-memory-campaign-corpus-v1.json"),
        help="The checked standard Evals corpus.",
    )
    parser.add_argument(
        "--state-directory",
        type=Path,
        help="Retain durable campaign state here instead of using a temporary directory.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "html"),
        default="json",
        help="Render the stable machine report or the human report.",
    )
    parser.add_argument("--output", type=Path, help="Write the report instead of stdout.")
    parser.add_argument(
        "--recover-only",
        action="store_true",
        help="Recover a completed campaign without allowing provider dispatch.",
    )
    return parser.parse_args()


def _write_report(
    report: MemoryExperimentReport,
    *,
    output_format: str,
    output: Path | None,
) -> None:
    rendered = (
        memory_experiment_report_to_json(report) + "\n"
        if output_format == "json"
        else render_memory_experiment_report_html(report)
    )
    if output is None:
        print(rendered, end="" if rendered.endswith("\n") else "\n")
    else:
        output.write_text(rendered, encoding="utf-8")


def _recover_in_fresh_process(
    *,
    corpus: Path,
    state_directory: Path,
    expected_json: str,
    scratch_directory: Path,
) -> None:
    recovered_path = scratch_directory / "recovered-report.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(Path(__file__).resolve()),
            "--corpus",
            str(corpus.resolve()),
            "--state-directory",
            str(state_directory.resolve()),
            "--format",
            "json",
            "--output",
            str(recovered_path),
            "--recover-only",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Fresh-process campaign recovery failed; rerun with --recover-only against "
            "the retained state directory for diagnostics."
        )
    recovered_json = recovered_path.read_text(encoding="utf-8").rstrip("\n")
    if recovered_json != expected_json:
        raise RuntimeError(
            "Fresh-process recovery produced evidence different from the original campaign."
        )


def main() -> None:
    arguments = _arguments()
    corpus = load_causal_memory_reference_corpus(arguments.corpus)
    with tempfile.TemporaryDirectory(prefix="cayu-causal-memory-campaign-") as temporary:
        scratch = Path(temporary)
        state_directory = arguments.state_directory or scratch / "state"
        report = asyncio.run(
            run_causal_memory_reference_campaign(
                corpus,
                state_directory,
                recover_only=arguments.recover_only,
            )
        )
        canonical_json = memory_experiment_report_to_json(report)
        if not arguments.recover_only:
            _recover_in_fresh_process(
                corpus=arguments.corpus,
                state_directory=state_directory,
                expected_json=canonical_json,
                scratch_directory=scratch,
            )
        _write_report(
            report,
            output_format=arguments.format,
            output=arguments.output,
        )


if __name__ == "__main__":
    main()
