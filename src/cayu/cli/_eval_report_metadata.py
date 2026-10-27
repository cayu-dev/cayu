"""Bounded publisher-supplied CI attribution; never authentication or native identity."""

from __future__ import annotations

import json
import os
from pathlib import Path

from cayu.evals.reported import EvalReportOrigin

_FIELDS = (
    "provider",
    "repository",
    "branch",
    "commit",
    "pr_head_commit",
    "pull_request",
    "run_url",
    "run_id",
    "run_attempt",
    "job",
)


def add_metadata_arguments(parser):
    for field in _FIELDS:
        parser.add_argument(
            "--report-" + field.replace("_", "-"),
            type=int if field in {"pull_request", "run_attempt"} else str,
        )


def report_origin(args) -> EvalReportOrigin:
    values = {}
    sources = {}
    if os.environ.get("GITHUB_ACTIONS") == "true":
        values.update(kind="ci", provider="github")
        sources.update(kind="environment", provider="environment")
        mappings = {
            "repository": "GITHUB_REPOSITORY",
            "commit": "GITHUB_SHA",
            "run_id": "GITHUB_RUN_ID",
            "run_attempt": "GITHUB_RUN_ATTEMPT",
            "job": "GITHUB_JOB",
        }
        for field, variable in mappings.items():
            value = os.environ.get(variable)
            if value:
                values[field] = int(value) if field == "run_attempt" else value
                sources[field] = "environment"
        branch = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME")
        if branch:
            values["branch"] = branch
            sources["branch"] = "environment"
        if values.get("repository") and values.get("run_id"):
            server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
            values["run_url"] = f"{server}/{values['repository']}/actions/runs/{values['run_id']}"
            sources["run_url"] = "environment"
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if event_path:
            with Path(event_path).open("rb") as stream:
                raw = stream.read((1 << 20) + 1)
            if len(raw) > 1 << 20:
                raise ValueError("GitHub event attribution exceeds the byte limit.")
            event = json.loads(raw)
            if type(event) is not dict:
                raise ValueError("GitHub event attribution requires an object.")
            pull = event.get("pull_request")
            if type(pull) is dict:
                if "number" in event:
                    values["pull_request"] = event["number"]
                    sources["pull_request"] = "environment"
                head = pull.get("head")
                if type(head) is dict and "sha" in head:
                    values["pr_head_commit"] = head["sha"]
                    sources["pr_head_commit"] = "environment"
    for field in _FIELDS:
        override = getattr(args, "report_" + field, None)
        if override is not None:
            values[field] = override
            sources[field] = "flag"
    return EvalReportOrigin.model_validate({**values, "sources": sources})
