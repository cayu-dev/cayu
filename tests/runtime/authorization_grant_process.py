"""One phase of the cross-process grant scenario, run in its own interpreter.

Each phase opens fresh SQLite session and grant stores in ``state`` and prints a
JSON summary, so a later phase observes only durable state.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from tests.runtime.authorization_grant_support import Harness, _command_turn

from cayu.approvals import (
    ToolAuthorizationGrantRequest,
    ToolAuthorizationGrantScopeKind,
    ToolAuthorizationGrantStatus,
    ToolAuthorizationGrantUseDecision,
)
from cayu.events import EventType
from cayu.storage import SQLiteToolAuthorizationGrantStore
from cayu.storage.sqlite import SQLiteSessionStore

COMMAND = ["make", "test"]
# A different command, so the first workspace grant does not already cover it.
LINT = ["make", "lint"]
WORKSPACE = ToolAuthorizationGrantRequest(scope=ToolAuthorizationGrantScopeKind.WORKSPACE)


async def _phase(name: str, state: Path) -> dict:
    sessions = SQLiteSessionStore(state / "sessions.sqlite")
    grants = SQLiteToolAuthorizationGrantStore(state / "grants.sqlite")
    command = LINT if name == "approve-without-settling" else COMMAND
    harness = Harness(
        state, turns=_command_turn("call_1", command), sessions=sessions, grants=grants
    )
    try:
        if name == "approve-and-remember":
            await harness.run("s1")
            await harness.remember("s1", WORKSPACE)
            (grant,) = await harness.service.list_grants()
            return {"grant_id": grant.grant_id, "status": grant.status.value}
        if name == "approve-without-settling":
            # The process stops after the approval ran but before the grant settled.
            await harness.run("s2")
            prepared = await harness.service.prepare(await harness.approval("s2"), WORKSPACE)
            async for _event in harness.app.resolve_tool_approval(prepared.approval_request):
                pass
            grant = await harness.service.get(prepared.grant.grant_id)
            return {"grant_id": grant.grant_id, "status": grant.status.value}
        if name == "settle-and-reuse":
            settled = await harness.service.reconcile_staged()
            events = await harness.run("s3")
            granted = await harness.service.list_grants()
            uses = [
                use
                for grant in granted
                for use in await harness.service.list_uses(grant.grant_id)
                if use.session_id == "s3"
            ]
            return {
                "settled": {grant.grant_id: grant.status.value for grant in settled},
                "paused": any(e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED for e in events),
                "allowed_uses": [
                    use.grant_id
                    for use in uses
                    if use.decision is ToolAuthorizationGrantUseDecision.ALLOWED
                ],
                "active": sorted(
                    grant.grant_id
                    for grant in granted
                    if grant.status is ToolAuthorizationGrantStatus.ACTIVE
                ),
            }
        raise SystemExit(f"unknown phase: {name}")
    finally:
        await grants.close()
        await sessions.close()


def main() -> None:
    name, state = sys.argv[1], Path(sys.argv[2])
    state.mkdir(parents=True, exist_ok=True)
    print(json.dumps(asyncio.run(_phase(name, state))))


if __name__ == "__main__":
    main()
