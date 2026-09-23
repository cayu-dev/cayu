"""Restartable order support: python -m cayu.examples.order_support --help.

Local signing files model a separate trusted representative service. They are a
teaching fixture, not production authentication. Tools never receive that key.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    IncompleteSessionRecoveryRequest,
    Message,
    ModelStreamEvent,
    OpenAIProvider,
    PendingActionQuery,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolContext,
    ToolEffect,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    ToolResult,
    ToolSpec,
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def identity(name: str, version: str = "1") -> ExecutionProfileBehaviorIdentity:
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version=version, implementation_version="1"
    )


@contextmanager
def database(state: Path):
    connection = sqlite3.connect(state / "service.sqlite")
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize(state: Path, operator: Path) -> None:
    state.mkdir(parents=True, exist_ok=False)
    operator.mkdir(parents=True, exist_ok=False, mode=0o700)
    key = Ed25519PrivateKey.generate()
    private = key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    # A local OS-protected fixture stand-in for the representative's signer.
    descriptor = os.open(operator / "signing.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(private)
    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    with database(state) as db:
        db.executescript("""
        CREATE TABLE config (public_key TEXT NOT NULL);
        CREATE TABLE orders (id TEXT PRIMARY KEY, items TEXT NOT NULL, tracking TEXT NOT NULL);
        CREATE TABLE proposals (id TEXT PRIMARY KEY, body TEXT NOT NULL, session_id TEXT UNIQUE NOT NULL);
        CREATE TABLE issued_receipts (id TEXT PRIMARY KEY, digest TEXT NOT NULL);
        CREATE TABLE replacements (action_id TEXT PRIMARY KEY, proposal_id TEXT UNIQUE NOT NULL,
                                   receipt TEXT NOT NULL);
        """)
        db.execute("INSERT INTO config VALUES (?)", (public,))
        db.execute(
            "INSERT INTO orders VALUES (?, ?, ?)",
            ("order-42", '["mug", "plate"]', "delivered; customer reports damage"),
        )
    print("Initialized service and independent representative signer; no replacement exists.")


def proposal_body(session: str, item: str) -> dict:
    if item not in {"mug", "plate"}:
        raise ValueError("Choose an item from the inspected order")
    return {
        "session_id": session,
        "order_id": "order-42",
        "item": item,
        "quantity": 1,
        "action_id": "replacement-"
        + hashlib.sha256(canonical([session, "order-42", item])).hexdigest()[:24],
    }


def proposal_id(body: dict) -> str:
    return "proposal-" + hashlib.sha256(canonical(body)).hexdigest()[:24]


def load_proposal(db: sqlite3.Connection, identifier: str, session: str) -> dict:
    row = db.execute("SELECT body FROM proposals WHERE id = ?", (identifier,)).fetchone()
    if row is None:
        raise ValueError("Unknown proposal")
    body = json.loads(row[0])
    if body.get("session_id") != session or proposal_id(body) != identifier:
        raise ValueError("Proposal binding mismatch")
    if body != proposal_body(session, body["item"]):
        raise ValueError("Proposal is outside the bounded fixture policy")
    return body


class SupportTool(Tool):
    def __init__(self, state: Path, name: str, version: str) -> None:
        self.state = state
        schemas = {
            "inspect_order": {},
            "propose_replacement": {"item": {"type": "string", "enum": ["mug", "plate"]}},
            "execute_replacement": {"proposal_id": {"type": "string"}},
            "verify_replacement": {},
        }
        descriptions = {
            "inspect_order": "Read order, tracking and replacement policy evidence.",
            "propose_replacement": "After the customer clarifies the damaged item, persist an immutable bounded replacement proposal.",
            "execute_replacement": "Execute the exact proposal; Cayu requires representative approval first.",
            "verify_replacement": "Independently read replacement service receipts for this conversation.",
        }
        properties = schemas[name]
        self.spec = ToolSpec(
            name=name,
            description=descriptions[name],
            input_schema={
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
            effect=ToolEffect.IDEMPOTENT
            if name in {"propose_replacement", "execute_replacement"}
            else ToolEffect.NONE,
            parallel_safe=False,
            execution_profile_identity=identity(name, version),
        )
        super().__init__()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        with database(self.state) as db:
            if self.spec.name == "inspect_order":
                row = db.execute("SELECT * FROM orders WHERE id = 'order-42'").fetchone()
                result = {
                    "order": dict(row),
                    "policy": "Replace one damaged item after customer clarification and representative approval. Ask which item: mug or plate.",
                }
            elif self.spec.name == "propose_replacement":
                body = proposal_body(ctx.session_id, args["item"])
                identifier = proposal_id(body)
                db.execute(
                    "INSERT OR IGNORE INTO proposals VALUES (?, ?, ?)",
                    (identifier, canonical(body).decode(), ctx.session_id),
                )
                load_proposal(db, identifier, ctx.session_id)
                result = {"proposal_id": identifier, **body}
            elif self.spec.name == "execute_replacement":
                body = load_proposal(db, args["proposal_id"], ctx.session_id)
                # The downstream action key is stable across Runtime retries/restarts.
                receipt = {
                    "replacement_id": body["action_id"],
                    "item": body["item"],
                    "quantity": 1,
                    "status": "created",
                }
                db.execute(
                    "INSERT OR IGNORE INTO replacements VALUES (?, ?, ?)",
                    (body["action_id"], args["proposal_id"], canonical(receipt).decode()),
                )
                result = json.loads(
                    db.execute(
                        "SELECT receipt FROM replacements WHERE action_id = ?", (body["action_id"],)
                    ).fetchone()[0]
                )
            else:
                rows = db.execute(
                    "SELECT p.body, r.receipt FROM replacements r JOIN proposals p ON p.id = r.proposal_id"
                ).fetchall()
                result = {
                    "replacements": [
                        json.loads(row["receipt"])
                        for row in rows
                        if json.loads(row["body"])["session_id"] == ctx.session_id
                    ]
                }
        return ToolResult(content=json.dumps(result), structured=result)


class SupportPolicy(ToolPolicy):
    def __init__(self, state: Path) -> None:
        self.state = state

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return identity("support-approval-policy")

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name == "execute_replacement":
            with database(self.state) as db:
                row = db.execute(
                    "SELECT id FROM proposals WHERE session_id = ?", (request.session.id,)
                ).fetchone()
                if row is None or request.arguments != {"proposal_id": row[0]}:
                    return ToolPolicyResult(
                        decision=ToolPolicyDecision.DENY,
                        reason="Call must match the immutable proposal for this conversation",
                    )
                load_proposal(db, row[0], request.session.id)
            return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


def scripted(phase: str, session: str, item: str) -> ScriptedModelProvider:
    def call(name: str, args: dict) -> list[ModelStreamEvent]:
        return [
            ModelStreamEvent.tool_call(id=f"{phase}-{name}", name=name, arguments=args),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]

    def say(text: str) -> list[ModelStreamEvent]:
        return [ModelStreamEvent.text_delta(text), ModelStreamEvent.completed()]

    if phase == "start":
        batches = [call("inspect_order", {}), say("Which item was damaged: the mug or the plate?")]
    elif phase == "clarify":
        batches = [
            call("propose_replacement", {"item": item}),
            call("execute_replacement", {"proposal_id": proposal_id(proposal_body(session, item))}),
        ]
    else:
        batches = [
            call("verify_replacement", {}),
            say("The independent service read above shows the outcome."),
        ]
    return ScriptedModelProvider(batches)


async def review(app: CayuApp, session: str, state: Path) -> dict:
    await app.recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session))
    pending = await app.session_store.query_pending_actions(PendingActionQuery(session_id=session))
    if len(pending.actions) != 1:
        raise ValueError("Expected exactly one pending protected action")
    action = pending.actions[0]
    if action.tool_name != "execute_replacement":
        raise ValueError("Expected a replacement approval")
    # Pending-action arguments are quarantined. The policy above binds the call
    # to this sole immutable application proposal before requesting approval.
    with database(state) as db:
        row = db.execute("SELECT id FROM proposals WHERE session_id = ?", (session,)).fetchone()
        if row is None:
            raise ValueError("No immutable proposal for the pending approval")
        identifier = row[0]
        body = load_proposal(db, identifier, session)
    return {
        "session_id": session,
        "proposal_id": identifier,
        "proposal": body,
        "approval_id": action.approval_id,
        "tool_round_id": action.round_id,
        "tool_call_id": action.tool_call_id,
    }


def issue(state: Path, operator: Path, review_path: Path, output: Path, decision: str) -> None:
    request = json.loads(review_path.read_text())
    with database(state) as db:
        body = load_proposal(db, request["proposal_id"], request["session_id"])
        if body != request["proposal"]:
            raise ValueError("Altered proposal in representative review")
        payload = {
            **request,
            "receipt_id": str(uuid4()),
            "decision": decision,
            "approver": "fixture-representative",
            "authority": "replace-one-damaged-item",
        }
        key = Ed25519PrivateKey.from_private_bytes((operator / "signing.key").read_bytes())
        public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        if public != db.execute("SELECT public_key FROM config").fetchone()[0]:
            raise ValueError("Unrecognized representative signer")
        envelope = {
            "payload": payload,
            "signature": base64.b64encode(key.sign(canonical(payload))).decode(),
        }
        db.execute(
            "INSERT INTO issued_receipts VALUES (?, ?)",
            (payload["receipt_id"], hashlib.sha256(canonical(envelope)).hexdigest()),
        )
    output.write_bytes(canonical(envelope))
    print("Representative issued a bound receipt; no replacement executed.")


async def receipt_to_native_request(
    app: CayuApp, state: Path, session: str, path: Path
) -> ToolApprovalRequest:
    if path.stat().st_size > 16_384:
        raise ValueError("Receipt exceeds fixture limit")
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    with database(state) as db:
        public = bytes.fromhex(db.execute("SELECT public_key FROM config").fetchone()[0])
        try:
            Ed25519PublicKey.from_public_bytes(public).verify(
                base64.b64decode(envelope["signature"], validate=True), canonical(payload)
            )
        except (InvalidSignature, ValueError) as exc:
            raise ValueError("Invalid representative signature") from exc
        registered = db.execute(
            "SELECT digest FROM issued_receipts WHERE id = ?", (payload["receipt_id"],)
        ).fetchone()
        if registered is None or registered[0] != hashlib.sha256(canonical(envelope)).hexdigest():
            raise ValueError("Unknown representative receipt")
        if (
            payload["session_id"] != session
            or payload["approver"] != "fixture-representative"
            or payload["authority"] != "replace-one-damaged-item"
        ):
            raise ValueError("Receipt conversation or approver authority mismatch")
        body = load_proposal(db, payload["proposal_id"], session)
        if payload["proposal"] != body:
            raise ValueError("Receipt proposal mismatch")
    # Read native identities from Runtime; receipt/action IDs cannot substitute.
    pending = await app.session_store.query_pending_actions(PendingActionQuery(session_id=session))
    if pending.actions:
        current = await review(app, session, state)
        if any(payload.get(key) != value for key, value in current.items()):
            raise ValueError("Receipt does not bind the current Cayu approval/round/call")
    # On repeated delivery, Runtime validates the exact completed resolution.
    # No direct service mutation and no locally synthesized replacement approval.
    return ToolApprovalRequest(
        session_id=session,
        approval_id=payload["approval_id"],
        tool_round_id=payload["tool_round_id"],
        tool_call_id=payload["tool_call_id"],
        decision=ToolApprovalDecision(payload["decision"]),
        resolved_by=ResolutionActor(
            subject=payload["approver"], source=ResolutionActorSource.REQUEST
        ),
        reason="Verified representative receipt",
        metadata={"external_receipt_id": payload["receipt_id"]},
    )


async def run(args: argparse.Namespace) -> None:
    state, session = args.state, args.session
    if not (state / "service.sqlite").is_file():
        raise ValueError("Initialize fresh state first")
    provider = OpenAIProvider() if args.live else scripted(args.command, session, args.item)
    store = SQLiteSessionStore(state / "sessions.sqlite")
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="support",
            model=os.environ["SUPPORT_MODEL"] if args.live else "scripted",
            system_prompt="Investigate the damaged order with inspect_order. Ask the user which item is damaged before proposing. After clarification propose_replacement, then execute_replacement with its proposal_id. After approval or rejection independently verify_replacement and explain the result. Never invent approval or receipts.",
        ),
        tools=[
            SupportTool(state, name, args.version)
            for name in (
                "inspect_order",
                "propose_replacement",
                "execute_replacement",
                "verify_replacement",
            )
        ],
        tool_policy=SupportPolicy(state),
    )
    try:
        if args.command == "review":
            result = await review(app, session, state)
            args.output.write_bytes(canonical(result))
            print(json.dumps(result))
            return
        if args.command == "start":
            stream = app.run(
                RunRequest(
                    agent_name="support",
                    session_id=session,
                    messages=[
                        Message.text("user", "Something in order-42 arrived damaged. Please help.")
                    ],
                )
            )
        elif args.command == "clarify":
            stream = app.resume(
                ResumeRequest(
                    session_id=session,
                    messages=[
                        Message.text("user", f"The {args.item} was damaged. Please replace it.")
                    ],
                )
            )
        elif args.command == "deliver":
            request = await receipt_to_native_request(app, state, session, args.receipt)
            stream = app.resolve_tool_approval(request)
        else:
            stream = app.resume(
                ResumeRequest(
                    session_id=session,
                    messages=[
                        Message.text("user", "Verify the replacement with a separate service read.")
                    ],
                )
            )
        async for event in stream:
            print(json.dumps(event.model_dump(mode="json"), default=str))
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("state", type=Path)
    init.add_argument("operator", type=Path)
    representative = sub.add_parser("issue")
    representative.add_argument("state", type=Path)
    representative.add_argument("operator", type=Path)
    representative.add_argument("review_path", type=Path)
    representative.add_argument("output", type=Path)
    representative.add_argument("decision", choices=["approve", "deny"])
    for name in ("start", "clarify", "review", "deliver", "verify"):
        command = sub.add_parser(name)
        command.add_argument("state", type=Path)
        command.add_argument("--session", default="support-example")
        command.add_argument("--item", choices=["mug", "plate"], default="mug")
        command.add_argument("--version", default="1")
        command.add_argument("--live", action="store_true")
        if name == "review":
            command.add_argument("output", type=Path)
        if name == "deliver":
            command.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.state, args.operator)
    elif args.command == "issue":
        issue(args.state, args.operator, args.review_path, args.output, args.decision)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
