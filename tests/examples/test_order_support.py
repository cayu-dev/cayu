"""Independent process/store evidence for the installed support application."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

if TYPE_CHECKING:
    from tests.sqlite_resources import SQLiteResourceScope

from cayu import EventQuery, PendingActionQuery, SQLiteSessionStore

ROOT = Path(__file__).resolve().parents[2]


def invoke(tmp: Path, *arguments: str, succeeds: bool = True) -> str:
    # CAYU_EXAMPLE_PYTHON also runs this identical suite against an installed wheel.
    python = os.environ.get("CAYU_EXAMPLE_PYTHON", sys.executable)
    env = dict(os.environ)
    if "CAYU_EXAMPLE_PYTHON" in env:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [python, "-m", "cayu.examples.order_support", *map(str, arguments)],
        cwd=tmp,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    output = result.stdout + result.stderr
    assert (result.returncode == 0) == succeeds, output
    return output


def rows(state: Path, table: str, resources: SQLiteResourceScope) -> list[tuple]:
    assert table in {"proposals", "replacements"}
    connection = resources.own(
        sqlite3.connect(f"file:{state / 'service.sqlite'}?mode=ro", uri=True), kind="connection"
    )
    cursor = resources.own(connection.execute(f"SELECT * FROM {table}"), kind="cursor")
    return cursor.fetchall()


async def native(state: Path, resources: SQLiteResourceScope) -> dict:
    store = resources.own(SQLiteSessionStore(state / "sessions.sqlite"))
    session = await store.load("support-example")
    assert session is not None
    actions = await store.query_pending_actions(PendingActionQuery(session_id=session.id))
    events = await store.query_events(EventQuery(session_id=session.id, limit=200))
    transcript = await store.load_transcript_snapshot(session.id)
    return {
        "transcript": transcript.model_dump(mode="json"),
        "instance": session.instance_id,
        "pending": [a.model_dump(mode="json") for a in actions.actions],
        "events": [r.event.model_dump(mode="json") for r in events],
    }


async def prepare(tmp: Path, resources: SQLiteResourceScope) -> tuple[Path, Path, Path, Path]:
    state, operator, review, receipt = (
        tmp / name for name in ("state", "operator", "review.json", "receipt.json")
    )
    invoke(tmp, "init", state, operator)
    invoke(tmp, "start", state)
    before = await native(state, resources)
    assert not before["pending"]
    assert not rows(state, "proposals", resources)
    invoke(tmp, "clarify", state)
    paused = await native(state, resources)
    assert before["instance"] == paused["instance"]
    assert len(paused["pending"]) == 1
    assert len(rows(state, "proposals", resources)) == 1
    assert not rows(state, "replacements", resources)
    assert "tool.call.approval_requested" in {event["type"] for event in paused["events"]}
    invoke(tmp, "review", state, review)
    binding = json.loads(review.read_text())
    action = paused["pending"][0]
    # Proposal creation is complete before any representative receipt is issued;
    # execution alone is held for approval across the process restart.
    assert action["tool_name"] == "execute_replacement"
    assert not receipt.exists()
    assert binding["approval_id"] == action["approval_id"]
    assert binding["tool_round_id"] == action["round_id"]
    assert binding["tool_call_id"] == action["tool_call_id"]
    assert binding["proposal"] == json.loads(rows(state, "proposals", resources)[0][1])
    return state, operator, review, receipt


@pytest.mark.parametrize("decision", ["approve", "deny"])
def test_restart_approval_repeat_and_independent_verification(
    tmp_path: Path, decision: str, sqlite_resources: SQLiteResourceScope
) -> None:
    async def scenario() -> None:
        async with sqlite_resources as resources:
            state, operator, review, receipt = await prepare(tmp_path, resources)
            invoke(tmp_path, "issue", state, operator, review, receipt, decision)
            assert not rows(state, "replacements", resources)
            invoke(tmp_path, "deliver", state, receipt)
            effects = rows(state, "replacements", resources)
            assert len(effects) == (1 if decision == "approve" else 0)
            if effects:
                proposal = json.loads(rows(state, "proposals", resources)[0][1])
                assert effects[0][0] == proposal["action_id"]
                assert json.loads(effects[0][2])["quantity"] == 1
            settled = await native(state, resources)
            assert not settled["pending"]
            invoke(tmp_path, "deliver", state, receipt)
            assert rows(state, "replacements", resources) == effects
            assert (await native(state, resources))["events"] == settled["events"]
            invoke(tmp_path, "verify", state)
            assert rows(state, "replacements", resources) == effects
            events = (await native(state, resources))["events"]
            assert any(
                e["tool_name"] == "verify_replacement" and e["type"] == "tool.call.completed"
                for e in events
            )
            # Conversation evidence survives all process boundaries, not just final prose.
            serialized = json.dumps((await native(state, resources))["transcript"])
            assert "Which item was damaged" in serialized
            assert "The mug was damaged" in serialized

    asyncio.run(scenario())


def test_invalid_receipts_and_changed_version_never_execute(
    tmp_path: Path, sqlite_resources: SQLiteResourceScope
) -> None:
    async def scenario() -> None:
        async with sqlite_resources as resources:
            state, operator, review, receipt = await prepare(tmp_path, resources)
            invoke(tmp_path, "issue", state, operator, review, receipt, "approve")
            original = json.loads(receipt.read_text())
            for field, value in (
                ("proposal", {}),
                ("receipt_id", "unknown"),
                ("approval_id", "wrong"),
            ):
                altered = json.loads(json.dumps(original))
                altered["payload"][field] = value
                bad = tmp_path / "bad.json"
                bad.write_text(json.dumps(altered))
                assert "Invalid representative signature" in invoke(
                    tmp_path, "deliver", state, bad, succeeds=False
                )
                assert not rows(state, "replacements", resources)
            # A valid signature alone is insufficient: this receipt was never issued.
            unknown = json.loads(json.dumps(original))
            unknown["payload"]["receipt_id"] = "unknown-signed-receipt"
            signer = Ed25519PrivateKey.from_private_bytes((operator / "signing.key").read_bytes())
            material = json.dumps(
                unknown["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode()
            unknown["signature"] = base64.b64encode(signer.sign(material)).decode()
            unknown_path = tmp_path / "unknown.json"
            unknown_path.write_text(json.dumps(unknown))
            assert "Unknown representative receipt" in invoke(
                tmp_path, "deliver", state, unknown_path, succeeds=False
            )
            assert not rows(state, "replacements", resources)
            assert "conversation or approver authority mismatch" in invoke(
                tmp_path, "deliver", state, receipt, "--session", "other", succeeds=False
            )
            assert "ExecutionProfileMismatchError" in invoke(
                tmp_path, "deliver", state, receipt, "--version", "2", succeeds=False
            )
            assert not rows(state, "replacements", resources)
            assert len((await native(state, resources))["pending"]) == 1
            # Even a correctly signed receipt must match the pending native identities.
            wrong_review = json.loads(review.read_text())
            wrong_review["tool_call_id"] = "different-call"
            review.write_text(json.dumps(wrong_review))
            invoke(tmp_path, "issue", state, operator, review, tmp_path / "wrong.json", "approve")
            assert "does not bind the current Cayu" in invoke(
                tmp_path, "deliver", state, tmp_path / "wrong.json", succeeds=False
            )
            assert not rows(state, "replacements", resources)

    asyncio.run(scenario())
