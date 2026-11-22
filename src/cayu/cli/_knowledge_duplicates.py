"""`cayu knowledge drop-duplicates`: decide a cluster of similar proposals in one step.

Either keep one proposal and drop the others as duplicates of it, or drop them all as
duplicates of an entry that is already kept. Each decision goes through the server's
own keep (``approve``) and drop (``reject``) routes, so each entry is decided atomically
and records its own reviewer and reason, as a single keep or drop does. Every decision
has its own operation ID, derived from the command's ``--operation-id``, the entry and
its role: running the same command again replays the decisions already made and makes
the rest, which is how to finish after a failure part way through.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any
from urllib.parse import quote

from cayu._validation import require_durable_clean_nonblank, require_durable_nonblank
from cayu.cli._cloud_agent_api import CayuApi, CayuApiError, CayuApiNotFound
from cayu.cli._output import output_destination, terminal_safe
from cayu.cli.session import _redact_sensitive

#: The most proposals one command drops.
MAX_DUPLICATE_DROPS = 50


def duplicate_decision_operation_id(operation_id: str, role: str, entry_id: str) -> str:
    """The operation ID of one entry's decision within a cluster decision."""

    digest = hashlib.sha256(
        json.dumps(["cayu.knowledge.duplicates.v1", operation_id, role, entry_id]).encode()
    ).hexdigest()
    return f"duplicates-{role}-{digest[:40]}"


def duplicate_reason(target: str, reason: str | None) -> str:
    """The reason each dropped entry records: what it duplicates, then the reviewer's."""

    return f"Duplicate of {target}." if reason is None else f"Duplicate of {target}. {reason}"


def run_drop_duplicates(args: argparse.Namespace) -> int:
    from cayu.cli._control_plane_actions import open_action_api

    api: CayuApi | None = None
    decided: list[dict[str, Any]] = []
    try:
        with output_destination(args.output):
            try:
                plan = _plan(args)
                api = open_action_api(args)
                if plan["keep"] is None:
                    _require_kept(api, plan["target"])
                else:
                    decided.append(
                        _decide(
                            api,
                            plan["keep"],
                            "approve",
                            operation_id=duplicate_decision_operation_id(
                                plan["operation_id"], "keep", plan["keep"]
                            ),
                            body={} if plan["reason"] is None else {"reason": plan["reason"]},
                        )
                    )
                for entry_id in plan["drop"]:
                    decided.append(
                        _decide(
                            api,
                            entry_id,
                            "reject",
                            operation_id=duplicate_decision_operation_id(
                                plan["operation_id"], "drop", entry_id
                            ),
                            body={"reason": duplicate_reason(plan["target"], plan["reason"])},
                        )
                    )
            except (CayuApiError, ValueError) as exc:
                return _report_failure(args, exc, decided)
            payload = {
                "schema_version": "1",
                "operation_id": plan["operation_id"],
                "duplicate_of": plan["target"],
                "decisions": decided,
            }
            if args.output_format == "json":
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                for decision in decided:
                    print(
                        terminal_safe(
                            f"{decision['decision']} {decision['entry_id']} "
                            f"(revision {decision['revision']})"
                        )
                    )
            return 0
    except OSError:
        print(
            "error: could not write command output; run the same command again to replay "
            "the decisions and see them.",
            file=sys.stderr,
        )
        return 1
    finally:
        if api is not None:
            api.close()


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    from cayu.cli._control_plane_actions import require_operation_id

    operation_id = require_operation_id(args.operation_id)
    keep = None if args.keep is None else require_durable_clean_nonblank(args.keep, "--keep")
    kept = None if args.of is None else require_durable_clean_nonblank(args.of, "--of")
    if (keep is None) == (kept is None):
        raise ValueError("Pass exactly one of --keep and --of.")
    target = keep if keep is not None else kept
    drop = [require_durable_clean_nonblank(entry_id, "entry ID") for entry_id in args.entry_ids]
    if len(set(drop)) != len(drop):
        raise ValueError("An entry ID is listed twice.")
    if target in drop:
        raise ValueError(f"{target} can't be dropped as a duplicate of itself.")
    if len(drop) > MAX_DUPLICATE_DROPS:
        raise ValueError(f"Drop at most {MAX_DUPLICATE_DROPS} entries in one command.")
    reason = None if args.reason is None else require_durable_nonblank(args.reason, "--reason")
    return {
        "operation_id": operation_id,
        "keep": keep,
        "target": target,
        "drop": drop,
        "reason": reason,
    }


def _require_kept(api: CayuApi, entry_id: str) -> None:
    try:
        entry = api.get(f"/knowledge/kept/{quote(entry_id, safe='')}", resource="knowledge")
    except CayuApiNotFound:
        raise ValueError(f"{entry_id} isn't a kept knowledge entry this agent can read.") from None
    if not isinstance(entry, dict) or entry.get("status") != "active":
        raise ValueError(f"{entry_id} isn't a kept knowledge entry this agent can read.")


def _decide(
    api: CayuApi, entry_id: str, action: str, *, operation_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    result = api.mutate(
        f"/knowledge/{quote(entry_id, safe='')}/{action}",
        body,
        resource="knowledge",
        operation_id=operation_id,
    )
    expected = "active" if action == "approve" else "archived"
    if (
        not isinstance(result, dict)
        or result.get("entry_id") != entry_id
        or result.get("status") != expected
        or type(result.get("revision")) is not int
    ):
        raise CayuApiError("outcome_unknown", "The decision returned an invalid receipt.")
    return _redact_sensitive(
        {
            "entry_id": entry_id,
            "decision": "kept" if action == "approve" else "dropped",
            "revision": result["revision"],
            "reason": result.get("reason"),
            "operation_id": operation_id,
        }
    )


def _report_failure(
    args: argparse.Namespace, exc: CayuApiError | ValueError, decided: list[dict[str, Any]]
) -> int:
    if isinstance(exc, CayuApiError) and exc.category != "outcome_unknown":
        message = f"Server refused the request ({exc.category}, HTTP {exc.status_code})" + (
            f": {exc.detail}" if exc.detail else "."
        )
    elif isinstance(exc, CayuApiError):
        message = "A decision's outcome is unknown."
    else:
        message = str(exc)
    if decided:
        message += (
            f" {len(decided)} decision(s) were made first; run the same command again to "
            "replay them and make the rest."
        )
    if args.output_format == "json":
        print(
            json.dumps(
                {"schema_version": "1", "error": {"message": message}, "decisions": decided},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        print(f"error: {terminal_safe(message)}", file=sys.stderr)
    return 1
