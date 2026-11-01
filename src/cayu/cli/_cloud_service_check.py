"""`cayu cloud service check --application NAME`: run the Agent's own post-deploy check.

The Agent declares the check in its root `cayu-cloud.toml`:

```toml
[operator_check]
path = "/operator-check"
```

The command asks Cayu Cloud (`POST /v1/applications/{id}/service/check`) to call exactly that
route on the running Agent with the Agent's operator login, which Cloud never sends to any other
app route, and prints what the Agent answered: the verdict, its HTTP status, how long it took and a
summary of its body. `cayu.server.operator_check` is the app side.

A sleeping Agent wakes for the check. While Cloud says it is starting (`agent_starting`, the check
not sent), the command asks again after Cloud's `Retry-After`, for up to `--wait-seconds`. Nothing
else is sent twice: a check can run a live workflow.

It prints a short report by default and the JSON envelope with `--json`. A check that fails, or
can't be run, exits 2 like any refused command, with the result in the error.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections.abc import Callable, Mapping
from typing import Any

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_releases import add_output_argument
from cayu.cli._cloud_settings import _cell

SCHEMA_VERSION = 1
DEFAULT_WAIT_SECONDS = 300.0
#: Longer than Cloud's longest check (its 45-second limit and a short wait while the Agent wakes).
_REQUEST_TIMEOUT_SECONDS = 90.0
_RETRY_SECONDS = (1.0, 10.0)


class OperatorCheckFailedError(CloudApiError):
    """The Agent answered its check, and the answer is a failure; `details` carries it."""

    def __init__(self, message: str, *, result: dict[str, Any]) -> None:
        super().__init__("operator_check_failed", message)
        self.details: dict[str, Any] = {"result": result}


def add_service_check_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wait-seconds",
        type=_seconds,
        default=DEFAULT_WAIT_SECONDS,
        help=(
            "How long to keep asking while the Agent wakes for the check "
            f"(default {DEFAULT_WAIT_SECONDS:g}). The check itself is never sent twice."
        ),
    )
    add_output_argument(parser, render_check, failure=render_check_failure)


def _seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number of seconds") from None
    if not 0 <= seconds <= 3600:
        raise argparse.ArgumentTypeError("must be between 0 and 3600 seconds")
    return seconds


def service_check(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    application_id: str,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run the check once, asking again only while Cloud says it wasn't sent."""

    patient = dataclasses.replace(
        client, timeout_seconds=max(client.timeout_seconds, _REQUEST_TIMEOUT_SECONDS)
    )
    deadline = monotonic() + arguments.wait_seconds
    while True:
        try:
            answer = patient.request("POST", f"/v1/applications/{application_id}/service/check")
            break
        except CloudApiError as exc:
            if exc.status_code == 405 or (exc.status_code == 404 and exc.code is None):
                raise CloudApiError(
                    "service_check_unsupported",
                    "This Cayu Cloud can't run an Agent's operator check yet.",
                ) from None
            if exc.code != "agent_starting":
                raise
            wait = _retry_seconds(exc)
            if monotonic() + wait > deadline:
                raise CloudApiError(
                    "agent_starting",
                    f"The Agent was still starting after {arguments.wait_seconds:g} s, so its "
                    "check wasn't sent. Run the command again, or raise --wait-seconds.",
                ) from None
            sleep(wait)
    result = _result(answer)
    if result["outcome"] != "passed":
        raise OperatorCheckFailedError(
            f"The Agent's check failed (HTTP {result['http_status']}).", result=result
        )
    return {"operation": "service.check", "result": result}


def _retry_seconds(exc: CloudApiError) -> float:
    retry_after = exc.body.get("retry_after") if isinstance(exc.body, Mapping) else None
    seconds = retry_after if isinstance(retry_after, int | float) else 3.0
    return min(max(float(seconds), _RETRY_SECONDS[0]), _RETRY_SECONDS[1])


def _result(answer: Mapping[str, Any]) -> dict[str, Any]:
    outcome = answer.get("outcome")
    status = answer.get("http_status")
    body = answer.get("body")
    if (
        outcome not in {"passed", "failed"}
        or type(status) is not int
        or not isinstance(body, Mapping)
        or not isinstance(answer.get("path"), str)
    ):
        raise CloudApiError("api_response_invalid", "Operator check response is invalid.")
    return {
        "schema_version": SCHEMA_VERSION,
        "application_id": answer.get("application_id"),
        "deployment_id": answer.get("deployment_id"),
        "method": answer.get("method"),
        "path": answer["path"],
        "outcome": outcome,
        "http_status": status,
        "duration_ms": answer.get("duration_ms"),
        "checked_at": answer.get("checked_at"),
        "body": dict(body),
        "hint": answer.get("hint"),
    }


def render_check(output: Mapping[str, Any]) -> str:
    return "\n".join(_report(output["result"]))


def render_check_failure(error: Mapping[str, Any]) -> str:
    result = error.get("result")
    if isinstance(result, Mapping):
        return "\n".join(_report(result))
    return f"Error: {_cell(error.get('message'), limit=500)}"


def _report(result: Mapping[str, Any]) -> list[str]:
    verdict = "passed" if result.get("outcome") == "passed" else "FAILED"
    lines = [
        f"Check {verdict}: {_cell(result.get('method'))} {_cell(result.get('path'), limit=200)}"
        f" answered HTTP {_cell(result.get('http_status'))} in "
        f"{_cell(result.get('duration_ms'))} ms",
        f"Release: {_cell(result.get('deployment_id'))}  Checked at: "
        f"{_cell(result.get('checked_at'))}",
    ]
    body = result.get("body")
    body = body if isinstance(body, Mapping) else {}
    kind = body.get("kind")
    if kind == "json":
        if body.get("summary"):
            lines.append(f"Summary: {_cell(body['summary'], limit=500)}")
        checks = body.get("checks")
        for step in checks if isinstance(checks, list) else []:
            if not isinstance(step, Mapping):
                continue
            mark = {True: "ok  ", False: "FAIL"}.get(step.get("ok"), "?   ")
            detail = step.get("detail")
            suffix = f"  {_cell(detail, limit=500)}" if detail else ""
            lines.append(f"  {mark} {_cell(step.get('name'), limit=100)}{suffix}")
        if body.get("checks_truncated"):
            lines.append("  (more checks than Cloud keeps)")
    elif kind in {"text", "unreadable_json"}:
        lines.append("Body:")
        # The Agent's own text: each line with control characters replaced, so it can't drive the
        # terminal (an OSC 52 clipboard write, a rewritten line).
        excerpt = str(body.get("excerpt", ""))
        lines.extend(f"  {_cell(line, limit=500)}" for line in excerpt.splitlines()[:40])
        if body.get("truncated"):
            lines.append("  (cut short)")
    elif kind == "binary":
        lines.append(f"Body: {_cell(body.get('bytes'))} bytes, not text")
    else:
        lines.append("Body: empty")
    if result.get("hint"):
        lines.append(f"Hint: {_cell(result['hint'], limit=500)}")
    return lines
