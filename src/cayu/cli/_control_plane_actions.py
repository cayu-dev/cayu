"""CLI adapters for existing authorized control-plane mutations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

from cayu._validation import require_durable_clean_nonblank
from cayu.cli._cloud_agent_api import CayuApi, CayuApiError, open_cloud_agent_api
from cayu.cli._json_inputs import json_object_file
from cayu.cli._output import add_output_options, output_destination, terminal_safe
from cayu.cli._read_targets import _positive_seconds
from cayu.cli.diagnostics import (
    DEFAULT_AUTHORIZATION_ENV,
    DEFAULT_SERVER_URL,
    read_authorization,
    request_cost_endpoint,
)
from cayu.cli.session import _redact_sensitive


def add_action_target_options(
    parser: argparse.ArgumentParser, *, output_format: str = "table"
) -> None:
    """Add ``--server-url``/``--application`` and the output options (default ``output_format``)."""

    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--server-url", help=f"Running server mount URL (default: {DEFAULT_SERVER_URL})."
    )
    target.add_argument("--application", help="Cayu Cloud Agent name or slug.")
    parser.add_argument("--context", type=Path, help="Cloud context file (with --application).")
    parser.add_argument(
        "--authorization-env",
        default=DEFAULT_AUTHORIZATION_ENV,
        help="Environment variable holding the complete Authorization header.",
    )
    parser.add_argument("--timeout-seconds", type=_positive_seconds, default=30.0)
    add_output_options(parser, formats=("json", "table"), default=output_format)


def open_action_api(args: argparse.Namespace) -> CayuApi:
    if args.application is not None:
        name = require_durable_clean_nonblank(args.application, "application")
        return open_cloud_agent_api(
            name, context=args.context, timeout_seconds=args.timeout_seconds
        )
    if args.context is not None:
        raise ValueError("--context applies only with --application.")
    url = action_server_url(args)
    request_cost_endpoint(url)  # Same canonical HTTPS-or-loopback policy as diagnostics.
    return CayuApi(
        url,
        authorization=read_authorization(args.authorization_env),
        timeout_seconds=args.timeout_seconds,
    )


def action_server_url(args: argparse.Namespace) -> str:
    """The running server an action without ``--application`` reaches."""

    return args.server_url or DEFAULT_SERVER_URL


def _receipt_identity(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        require_durable_clean_nonblank(value, "receipt identity")
    except ValueError:
        return False
    return _redact_sensitive(value) == value


def _mutation_result(args: argparse.Namespace, body: dict[str, Any], result: Any) -> dict[str, Any]:
    """Validate evidence of success before rendering structural IDs or user content."""
    invalid = CayuApiError("outcome_unknown", "The mutation returned an unusable receipt.")
    if not isinstance(result, dict):
        raise invalid
    if args.command == "task":
        if args.task_command == "create":
            if not _receipt_identity(result.get("task_id")):
                raise invalid
            return {"task_id": result["task_id"]}
        if args.task_command == "graph-create":
            task_ids, nodes = result.get("task_ids"), body.get("nodes")
            if (
                not _receipt_identity(result.get("graph_id"))
                or not isinstance(task_ids, dict)
                or not task_ids
                or not isinstance(nodes, list)
                or not all(isinstance(node, dict) for node in nodes)
            ):
                raise invalid
            expected = [node.get("key") for node in nodes]
            if (
                not all(_receipt_identity(key) for key in expected)
                or set(task_ids) != set(expected)
                or not all(_receipt_identity(value) for value in task_ids.values())
                or len(set(task_ids.values())) != len(task_ids)
            ):
                raise invalid
            # These keys are caller-selected node identities, not secret field names.
            return {"graph_id": result["graph_id"], "task_ids": dict(task_ids)}
        if (
            not _receipt_identity(result.get("id"))
            or result["id"] != args.task_id
            or result.get("status") != ("paused" if args.task_command == "hold" else "blocked")
        ):
            raise invalid
    elif (
        not _receipt_identity(result.get("entry_id"))
        or result["entry_id"] != args.entry_id
        or result.get("status") != ("active" if args.knowledge_command == "keep" else "archived")
        or type(result.get("revision")) is not int
        or result["revision"] < 1
    ):
        raise invalid
    return _redact_sensitive(result)


def _output_aliases_input(args: argparse.Namespace) -> bool:
    if args.output is None:
        return False
    output = Path(args.output)
    for name in ("file", "input_file", "payload_file", "context"):
        source = getattr(args, name, None)
        if source is None:
            continue
        if output.resolve() == source.resolve():
            return True
        if output.exists() and source.exists() and output.samefile(source):
            return True
    return False


def run_action(args: argparse.Namespace) -> int:
    api = None
    try:
        if _output_aliases_input(args):
            print("error: output must not refer to an input file.", file=sys.stderr)
            return 1
        with output_destination(args.output):
            try:
                operation = getattr(args, "operation_id", None)
                if operation is not None:
                    require_durable_clean_nonblank(operation, "operation ID")
                    if len(operation) > 256 or any(ord(c) < 32 or ord(c) > 126 for c in operation):
                        raise ValueError("Operation ID must be 1-256 printable ASCII characters.")
                if args.command == "task":
                    command = args.task_command
                    if command == "create":
                        path, body = (
                            "/tasks",
                            {
                                "operation_id": operation,
                                "type": args.type,
                                "input": json_object_file(args.input_file),
                            },
                        )
                    elif command == "graph-create":
                        path, body = "/task-graphs", json_object_file(args.file)
                        body["operation_id"] = operation
                    else:
                        action = "pause" if command == "hold" else command
                        path = f"/tasks/{quote(args.task_id, safe='')}/{action}"
                        body = {
                            "reason": args.reason,
                            "payload": json_object_file(args.payload_file),
                        }
                else:
                    action = "approve" if args.knowledge_command == "keep" else "reject"
                    path, body = f"/knowledge/{quote(args.entry_id, safe='')}/{action}", {}
                api = open_action_api(args)
                result = api.mutate(
                    path,
                    body,
                    resource="task" if args.command == "task" else "knowledge",
                    operation_id=operation,
                )
                payload = {"schema_version": "1", "result": _mutation_result(args, body, result)}
                if operation is not None:
                    payload["operation_id"] = operation
                try:
                    api.ensure_safe_output(payload)
                except CayuApiError:
                    raise CayuApiError(
                        "outcome_unknown", "The mutation returned an unsafe receipt."
                    ) from None
                text = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=None if args.output_format == "json" else 2,
                )
                print(
                    text if args.output_format == "json" else terminal_safe(text, keep_lines=True)
                )
                return 0
            except (CayuApiError, ValueError, OSError) as exc:
                # Never echo server-controlled detail or arbitrary file contents on failure.
                message = (
                    str(exc)
                    if not isinstance(exc, CayuApiError)
                    else (
                        "Mutation outcome is unknown; inspect the result and retain the same operation ID and input before retrying."
                        if exc.category == "outcome_unknown"
                        else f"Server refused the request ({exc.category}, HTTP {exc.status_code})."
                    )
                )
                if args.output_format == "json":
                    print(json.dumps({"schema_version": "1", "error": {"message": message}}))
                else:
                    print(f"error: {terminal_safe(message)}", file=sys.stderr)
                return 1
    except OSError:
        print(
            "error: could not write command output; inspect the server before retrying a mutation.",
            file=sys.stderr,
        )
        return 1
    finally:
        if api is not None:
            api.close()
