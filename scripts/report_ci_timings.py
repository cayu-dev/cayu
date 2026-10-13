#!/usr/bin/env python3
"""Report observed GitHub Actions timings without changing runs or runner settings.

See docs/ci-timing.md for metric definitions, exclusions and offline replay.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


def _get(endpoint: str, *, paginate: bool = False):
    command = ["gh", "api", "--hostname", "github.com", "--method", "GET"]
    if paginate:
        command += ["--paginate", "--slurp"]
    command += ["-H", "X-GitHub-Api-Version: 2022-11-28", endpoint]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout)


def fetch_snapshot(repository: str, run_id: int) -> dict:
    """Read every attempt, checking pagination and concurrent reruns."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or run_id < 1:
        raise ValueError("expected OWNER/REPOSITORY and a positive run ID")
    endpoint = f"repos/{repository}/actions/runs/{run_id}"
    run = _get(endpoint)
    if run["status"] != "completed":
        raise ValueError("the workflow must finish before taking a timing snapshot")
    attempts = []
    for number in range(1, run["run_attempt"] + 1):
        attempt_endpoint = f"{endpoint}/attempts/{number}"
        attempt = _get(attempt_endpoint)
        pages = _get(f"{attempt_endpoint}/jobs?per_page=100", paginate=True)
        jobs = [job for page in pages for job in page["jobs"]]
        if not pages or any(page["total_count"] != len(jobs) for page in pages):
            raise ValueError(f"incomplete job pages for attempt {number}")
        attempts.append({"run": attempt, "total_count": len(jobs), "jobs": jobs})
    fresh = _get(endpoint)
    keys = ("id", "head_sha", "run_attempt", "status", "conclusion", "updated_at")
    if any(run[key] != fresh[key] for key in keys):
        raise ValueError("workflow changed while reading it; take a new snapshot")
    return {"schema_version": 1, "repository": repository, "run": run, "attempts": attempts}


def _time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"timestamp has no timezone: {value}")
    return result


def _seconds(
    start: str | None, end: str | None, *, metric: str, anomalies: list[str]
) -> float | None:
    if start is None or end is None:
        return None
    left, right = _time(start), _time(end)
    seconds = (right - left).total_seconds()
    if seconds < 0:
        anomalies.append(metric)
        return None
    return seconds


def _completion(jobs: list[dict]) -> str | None:
    active = [job for job in jobs if job["conclusion"] != "skipped"]
    if not active or any(
        job["completed_at"] is None or "execution_seconds" in job["timing_anomalies"]
        for job in active
    ):
        return None
    return max((job["completed_at"] for job in active), key=_time)


def build_report(snapshot: dict, *, exclude_jobs: tuple[str, ...] = ()) -> dict:
    """Measure selected jobs; never infer branch protection or required coverage."""
    run = snapshot["run"]
    if snapshot["schema_version"] != 1 or run["status"] != "completed" or run["run_attempt"] < 1:
        raise ValueError("expected a completed workflow in a version 1 snapshot")
    attempts = snapshot["attempts"]
    if [item["run"]["run_attempt"] for item in attempts] != list(range(1, run["run_attempt"] + 1)):
        raise ValueError("snapshot must include every attempt in order")
    excluded = set(exclude_jobs)
    seen_names = set()
    executions = {}
    observed_execution = 0.0
    observed_waiting = 0.0
    unavailable_execution, unavailable_waiting = [], []
    result_attempts: list[dict] = []
    previous_completion = None
    for item in attempts:
        attempt = item["run"]
        if len(item["jobs"]) != item["total_count"]:
            raise ValueError("snapshot job count does not match the recorded inventory")
        if (
            attempt["id"] != run["id"]
            or attempt["head_sha"] != run["head_sha"]
            or attempt["status"] != "completed"
        ):
            raise ValueError("attempt does not belong to this completed workflow revision")
        number = attempt["run_attempt"]
        if attempt["run_started_at"] is None:
            raise ValueError("attempt start time is missing")
        started = _time(attempt["run_started_at"])
        ids, names = set(), set()
        jobs = []
        for job in item["jobs"]:
            if job["id"] in ids or job["name"] in names:
                raise ValueError("duplicate job ID or ambiguous job name within an attempt")
            ids.add(job["id"])
            names.add(job["name"])
            if job["run_id"] != run["id"] or job["run_attempt"] != number:
                raise ValueError("job belongs to a different run or attempt")
            if job["status"] != "completed" or not job["conclusion"]:
                raise ValueError("snapshot contains an unfinished job")
            seen_names.add(job["name"])
            signature = tuple(
                job.get(key) for key in ("name", "started_at", "completed_at", "conclusion")
            )
            job_start = None if job.get("started_at") is None else _time(job["started_at"])
            skipped = job["conclusion"] == "skipped"
            carried = None
            if not skipped and job_start is not None and job_start < started:
                carried = executions.get(signature)
                if carried is None:
                    raise ValueError(
                        "job predates its attempt without a matching earlier execution"
                    )
            selected = job["name"] not in excluded
            anomalies: list[str] = []
            execution = _seconds(
                job.get("started_at"),
                job.get("completed_at"),
                metric="execution_seconds",
                anomalies=anomalies,
            )
            if skipped:
                execution = None
            waiting = (
                None
                if skipped or carried is not None
                else _seconds(
                    job.get("created_at"),
                    job.get("started_at"),
                    metric="created_to_started_seconds",
                    anomalies=anomalies,
                )
            )
            if not skipped and carried is None:
                executions[signature] = job["id"]
                if selected:
                    if execution is None:
                        unavailable_execution.append(job["id"])
                    else:
                        observed_execution += execution
                    if waiting is None:
                        unavailable_waiting.append(job["id"])
                    else:
                        observed_waiting += waiting
            jobs.append(
                {
                    "id": job["id"],
                    "name": job["name"],
                    "conclusion": job["conclusion"],
                    "selected": selected,
                    "carried_from_job_id": carried,
                    "created_at": job.get("created_at"),
                    "started_at": job.get("started_at"),
                    "completed_at": job.get("completed_at"),
                    "timing_anomalies": anomalies,
                    "run_to_created_seconds": (
                        None
                        if carried is not None
                        else _seconds(
                            run["created_at"],
                            job.get("created_at"),
                            metric="run_to_created_seconds",
                            anomalies=anomalies,
                        )
                    ),
                    "created_to_started_seconds": waiting,
                    "execution_seconds": execution,
                    "steps": [
                        {
                            "number": step["number"],
                            "name": step["name"],
                            "conclusion": step["conclusion"],
                            "started_at": step.get("started_at"),
                            "completed_at": step.get("completed_at"),
                            "seconds": _seconds(
                                step.get("started_at"),
                                step.get("completed_at"),
                                metric=f"steps.{step['number']}.seconds",
                                anomalies=anomalies,
                            ),
                        }
                        for step in job.get("steps", [])
                    ],
                }
            )
        selected_jobs = [job for job in jobs if job["selected"]]
        completion = _completion(selected_jobs)
        gap = (
            None
            if previous_completion is None
            else (started - _time(previous_completion)).total_seconds()
        )
        result_attempts.append(
            {
                "number": number,
                "conclusion": attempt["conclusion"],
                "started_at": attempt["run_started_at"],
                "gap_after_previous_selected_completion_seconds": gap,
                "selected_completion_at": completion,
                "jobs": jobs,
            }
        )
        previous_completion = completion
    if excluded - seen_names:
        raise ValueError(f"excluded job names were not found: {sorted(excluded - seen_names)}")
    latest = [job for job in result_attempts[-1]["jobs"] if job["selected"]]
    if not latest:
        raise ValueError("no selected jobs in the final attempt")
    report_anomalies: list[str] = []
    elapsed = _seconds(
        run["created_at"],
        _completion(latest),
        metric="elapsed_to_selected_completion_seconds",
        anomalies=report_anomalies,
    )
    green = any(job["conclusion"] == "success" for job in latest) and all(
        job["conclusion"] in {"success", "skipped"} for job in latest
    )
    return {
        "schema_version": 1,
        "repository": snapshot["repository"],
        "run_id": run["id"],
        "head_sha": run["head_sha"],
        "workflow_path": run["path"],
        "event": run["event"],
        "run_conclusion": run["conclusion"],
        "created_at": run["created_at"],
        "excluded_job_names": sorted(excluded),
        "latest_selected_conclusions": dict(
            sorted(Counter(job["conclusion"] for job in latest).items())
        ),
        "selected_jobs_green": green,
        "elapsed_to_selected_completion_seconds": elapsed,
        "successful_selected_elapsed_seconds": elapsed if green else None,
        "timing_anomalies": report_anomalies,
        "observed_selected_job_execution_seconds": observed_execution,
        "observed_selected_job_waiting_seconds": observed_waiting,
        "jobs_with_unavailable_execution_seconds": unavailable_execution,
        "jobs_with_unavailable_waiting_seconds": unavailable_waiting,
        "attempts": result_attempts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--repo", help="GitHub OWNER/REPOSITORY; uses authenticated gh GET requests"
    )
    source.add_argument("--input", type=Path, help="replay a saved snapshot without network access")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--snapshot", type=Path, help="save API evidence for offline replay")
    parser.add_argument(
        "--exclude-job", action="append", default=[], help="exact job name; repeatable"
    )
    args = parser.parse_args(argv)
    if bool(args.repo) != (args.run_id is not None) or (args.input and args.snapshot):
        parser.error("use --repo with --run-id, or --input; --snapshot applies only to fetching")
    try:
        snapshot = (
            json.loads(args.input.read_text(encoding="utf-8"))
            if args.input
            else fetch_snapshot(args.repo, args.run_id)
        )
        report = build_report(snapshot, exclude_jobs=tuple(args.exclude_job))
        if args.snapshot:
            args.snapshot.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Cannot report CI timings: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
