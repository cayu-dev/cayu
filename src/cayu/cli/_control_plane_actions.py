"""CLI adapters for existing authorized control-plane mutations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

from cayu._validation import require_durable_clean_nonblank, require_durable_nonblank
from cayu.cli._cloud_agent_api import (
    CayuApi,
    CayuApiError,
    add_wake_options,
    open_cloud_agent_api,
)
from cayu.cli._cloud_private_state import write_private_json
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
    add_wake_options(parser)
    add_output_options(parser, formats=("json", "table"), default=output_format)


def open_action_api(args: argparse.Namespace) -> CayuApi:
    if args.application is not None:
        name = require_durable_clean_nonblank(args.application, "application")
        return open_cloud_agent_api(
            name,
            context=args.context,
            timeout_seconds=args.timeout_seconds,
            wake_timeout_seconds=args.wake_timeout_seconds,
            offers_wake_option=True,
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
        if args.task_command == "replay":
            if not _receipt_identity(result.get("task_id")):
                raise invalid
            return {"task_id": result["task_id"], "replayed_task_id": args.task_id}
        if (
            not _receipt_identity(result.get("id"))
            or result["id"] != args.task_id
            or not _task_action_settled(args.task_command, result)
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
    elif body.get("reason") is not None and "reason" not in result:
        # An Agent on a Cayu from before reasons: the decision stands, without the reason.
        print(
            "warning: this Agent's Cayu doesn't record review reasons; the decision was "
            "made without it. Deploy it with a newer Cayu to record reasons.",
            file=sys.stderr,
        )
    return _redact_sensitive(result)


def _task_action_settled(command: str, task: dict[str, Any]) -> bool:
    """Whether the task the server returned is in the state the action leaves it in."""

    status = task.get("status")
    if command == "cancel":
        # A running task stays running, marked, until its worker stops.
        return status == "cancelled" or task.get("status_reason") in {
            "cancellation_requested",
            "retry_cancellation_requested",
        }
    if command == "resume":
        # Back in the queue, or waiting there for its prerequisites or its group.
        return status in {"pending", "waiting_dependencies", "waiting_group"}
    return status == {"hold": "paused", "block": "blocked"}[command]


def _replay_request(
    api: CayuApi, task_id: str, operation: str | None, *, record: Path
) -> dict[str, Any]:
    """The replay of ``task_id`` to send: the one sent before under this operation ID, or
    the one the agent offers now (its declared type and incarnation proof).

    The task store binds a replay's operation ID to its exact request, proof included, and the
    proof changes once the source task is attached to a session or deleted. So the first
    request is kept (``_replay_record``), and a retry sends it again unchanged: a replay
    whose answer was lost then returns the task it created instead of a conflict. A request
    the server refused outright is dropped, so the next try asks for a fresh offer.
    """

    try:
        kept = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        kept = None
    if isinstance(kept, dict) and kept.get("operation_id") == operation:
        return kept
    request = _offered_replay(api, task_id, operation)
    write_private_json(record, request)
    return request


def _replay_record(target: str, task_id: str, operation: str | None) -> Path:
    """Where the replay sent under ``operation`` is kept, per target and task."""

    root = os.environ.get("XDG_STATE_HOME")
    base = Path(root).expanduser() if root else Path.home() / ".local" / "state"
    digest = hashlib.sha256(
        json.dumps([target, task_id, operation], separators=(",", ":")).encode()
    ).hexdigest()
    return base / "cayu" / "task-replays" / f"{digest}.json"


def _offered_replay(api: CayuApi, task_id: str, operation: str | None) -> dict[str, Any]:
    quoted = quote(task_id, safe="")
    offer = api.get(f"/tasks/{quoted}/replay", resource="task")
    if not isinstance(offer, dict) or offer.get("available") is not True:
        reason = offer.get("unavailable_reason") if isinstance(offer, dict) else None
        raise ValueError(
            f"Task {task_id} can't be replayed ({reason or 'unavailable'}). "
            "`cayu task show` has the task."
        )
    task = api.get(f"/tasks/{quoted}", resource="task")
    if not isinstance(task, dict) or not isinstance(task.get("type"), str):
        raise CayuApiError("not_cayu", "The agent returned a task without its type.")
    return {"operation_id": operation, "type": task["type"], "source": offer.get("source")}


def _output_aliases_input(args: argparse.Namespace) -> bool:
    if args.output is None:
        return False
    output = Path(args.output)
    for name in ("file", "input_file", "payload_file", "context", "evidence", "labels"):
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
                    elif command in {"cancel", "resume", "replay"}:
                        path = f"/tasks/{quote(args.task_id, safe='')}/{command}"
                        reason = getattr(args, "reason", None)
                        body = {} if reason is None else {"reason": reason}
                    else:
                        action = "pause" if command == "hold" else command
                        path = f"/tasks/{quote(args.task_id, safe='')}/{action}"
                        body = {
                            "reason": args.reason,
                            "payload": json_object_file(args.payload_file),
                        }
                else:
                    action = {"keep": "approve", "drop": "reject", "archive": "retire"}[
                        args.knowledge_command
                    ]
                    path, body = f"/knowledge/{quote(args.entry_id, safe='')}/{action}", {}
                    if args.reason is not None:
                        body["reason"] = require_durable_nonblank(args.reason, "--reason")
                    if getattr(args, "revision", None) is not None:
                        body["expected_revision"] = args.revision
                api = open_action_api(args)
                replay_record: Path | None = None
                if args.command == "task" and args.task_command == "replay":
                    # The Agent the API resolved to, not the name: two contexts may each have
                    # an Agent of that name.
                    replay_record = _replay_record(api.origin, args.task_id, operation)
                    body = _replay_request(api, args.task_id, operation, record=replay_record)
                try:
                    result = api.mutate(
                        path,
                        body,
                        resource="task" if args.command == "task" else "knowledge",
                        operation_id=operation,
                    )
                except CayuApiError as exc:
                    if replay_record is not None and exc.category == "request_rejected":
                        # The route refused this exact request, so it wasn't admitted: a retry
                        # asks for a fresh offer. Any other failure (unreachable, waking,
                        # credentials, an unknown outcome) may follow an admitted attempt, so
                        # the kept request stays for the retry.
                        replay_record.unlink(missing_ok=True)
                    raise
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
                # Never echo arbitrary file contents or a server body on failure; only the
                # route's own bounded reason and code, which `CayuApi` already vetted.
                error: dict[str, Any] = {}
                if not isinstance(exc, CayuApiError):
                    message = str(exc)
                elif exc.category == "outcome_unknown":
                    message = (
                        "Mutation outcome is unknown; inspect the result and retain the same "
                        "operation ID and input before retrying."
                    )
                else:
                    message = (
                        f"Server refused the request ({exc.category}, HTTP {exc.status_code})"
                        + (f": {exc.detail}" if exc.detail else ".")
                        + (f" ({exc.code})" if exc.code else "")
                    )
                    error = {
                        "category": exc.category,
                        "status_code": exc.status_code,
                        **({"detail": exc.detail} if exc.detail else {}),
                        **({"code": exc.code} if exc.code else {}),
                    }
                if args.output_format == "json":
                    print(
                        json.dumps({"schema_version": "1", "error": {"message": message, **error}})
                    )
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
