"""An example downstream service, independent of Cayu's session database.

SQLite stands in for a receipt-observable external service. Its idempotency and
precondition checks are application guarantees, not universal Cayu guarantees.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from cayu import (
    ExecutionProfileBehaviorIdentity,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconcilerSpec,
    ToolEffectReconciliationContext,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationResult,
)


def _arguments(service: str, release: str, expected_version: int) -> str:
    return json.dumps(
        {"service": service, "release": release, "expected_version": expected_version},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class DeploymentState:
    def __init__(self, path: Path, *, lose_acknowledgement_once: bool = False) -> None:
        self.path = path
        self.lose_acknowledgement_once = lose_acknowledgement_once
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS service (id INTEGER PRIMARY KEY, version INTEGER, "
                "mutation_count INTEGER, release TEXT)"
            )
            connection.execute("INSERT OR IGNORE INTO service VALUES (1, 7, 0, NULL)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS operations (key TEXT PRIMARY KEY, call_id TEXT, "
                "arguments TEXT, arguments_digest TEXT, receipt TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS invocations (key TEXT PRIMARY KEY, count INTEGER)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def snapshot(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            return dict(connection.execute("SELECT * FROM service WHERE id=1").fetchone())

    @property
    def version(self) -> int:
        return self.snapshot()["version"]

    @property
    def mutation_count(self) -> int:
        return self.snapshot()["mutation_count"]

    @property
    def deployed_release(self) -> str | None:
        return self.snapshot()["release"]

    @property
    def receipts(self) -> dict[str, ToolEffectReceipt]:
        with closing(self._connect()) as connection:
            return {
                row["key"]: ToolEffectReceipt.model_validate_json(row["receipt"])
                for row in connection.execute(
                    "SELECT key, receipt FROM operations WHERE receipt IS NOT NULL"
                )
            }

    def invocation_count(self, key: str) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT count FROM invocations WHERE key=?", (key,)).fetchone()
            return 0 if row is None else row["count"]

    def lookup(self, key: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM operations WHERE key=?", (key,)).fetchone()
            return None if row is None else dict(row)

    def begin(
        self,
        *,
        service: str,
        release: str,
        expected_version: int,
        idempotency_key: str,
        tool_call_id: str,
    ) -> ToolResult | None:
        """Record downstream admission without pretending that it completed."""
        arguments = _arguments(service, release, expected_version)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO invocations VALUES (?, 1) ON CONFLICT(key) DO UPDATE SET count=count+1",
                (idempotency_key,),
            )
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM operations WHERE key=?", (idempotency_key,)
            ).fetchone()
            if existing is not None:
                if existing["call_id"] != tool_call_id or existing["arguments"] != arguments:
                    raise ValueError("Deployment key conflicts with the original operation.")
                return None
            version = connection.execute("SELECT version FROM service WHERE id=1").fetchone()[0]
            if expected_version != version:
                return ToolResult(
                    content="External state changed; deployment rejected.",
                    is_error=True,
                    structured={"expected_version": expected_version, "actual_version": version},
                )
            connection.execute(
                "INSERT INTO operations VALUES (?, ?, ?, ?, NULL)",
                (idempotency_key, tool_call_id, arguments, sha256(arguments.encode()).hexdigest()),
            )
        return None

    def complete(self, key: str) -> ToolResult:
        """Commit mutation and receipt in one downstream transaction."""
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM operations WHERE key=?", (key,)).fetchone()
            if row is None:
                raise ValueError("Deployment has no admitted operation.")
            if row["receipt"] is not None:
                receipt = ToolEffectReceipt.model_validate_json(row["receipt"])
                return ToolResult(
                    content=receipt.message,
                    structured=receipt.structured,
                    is_error=receipt.outcome == "failed",
                )
            arguments = json.loads(row["arguments"])
            state = dict(connection.execute("SELECT * FROM service WHERE id=1").fetchone())
            completed = arguments["expected_version"] == state["version"]
            if completed:
                state["version"] += 1
                state["mutation_count"] += 1
                connection.execute(
                    "UPDATE service SET version=?, mutation_count=?, release=? WHERE id=1",
                    (state["version"], state["mutation_count"], arguments["release"]),
                )
            receipt = ToolEffectReceipt(
                receipt_id="deployment:" + sha256(key.encode()).hexdigest(),
                receipt_schema="counterfactual-deployment",
                receipt_schema_version=1,
                tool_call_id=row["call_id"],
                tool_name="deploy_service",
                idempotency_key=key,
                external_system="example-deployment-service",
                outcome="completed" if completed else "failed",
                message=(
                    f"Deployed {arguments['release']} to {arguments['service']}."
                    if completed
                    else "Deployment precondition changed before completion."
                ),
                structured={
                    "service": arguments["service"],
                    "release": arguments["release"],
                    "version": state["version"],
                    "mutation_count": state["mutation_count"],
                },
                resource_versions={"service_version": str(state["version"])},
                integrity={"arguments_sha256": row["arguments_digest"]},
                observed_at=datetime.now(UTC),
                source="adapter",
            )
            connection.execute(
                "UPDATE operations SET receipt=? WHERE key=?", (receipt.model_dump_json(), key)
            )
        return ToolResult(
            content=receipt.message, structured=receipt.structured, is_error=not completed
        )

    def deploy(self, **arguments) -> ToolResult:
        rejected = self.begin(**arguments)
        if rejected is not None:
            return rejected
        result = self.complete(arguments["idempotency_key"])
        if self.lose_acknowledgement_once:
            self.lose_acknowledgement_once = False
            raise ConnectionError("Downstream deployment committed; acknowledgement lost.")
        return result


class DeployServiceTool(Tool):
    spec = ToolSpec(
        name="deploy_service",
        description="Deploy only when the expected external-state version still matches.",
        input_schema={
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "release": {"type": "string"},
                "expected_version": {"type": "integer"},
            },
            "required": ["service", "release", "expected_version"],
            "additionalProperties": False,
        },
        effect=ToolEffect.EXTERNAL,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="examples:counterfactual-approval:deploy-service",
            behavior_version="2",
            implementation_version="2",
        ),
    )

    def __init__(self, state: DeploymentState) -> None:
        self.state = state

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        call_id = ctx.metadata.get("tool_call_id")
        if ctx.idempotency_key is None or type(call_id) is not str or not call_id:
            raise RuntimeError("Deployment requires runtime call and idempotency identities.")
        return self.state.deploy(**args, idempotency_key=ctx.idempotency_key, tool_call_id=call_id)


class DeploymentReconciler:
    def __init__(self, state: DeploymentState) -> None:
        self.state = state

    async def reconcile(
        self,
        *,
        context: ToolEffectReconciliationContext,
        receipt: ToolEffectReceipt | None,
    ) -> ToolEffectReconciliationResult:
        operation = self.state.lookup(context.idempotency_key)
        if operation is None:
            return ToolEffectReconciliationResult(
                outcome="not_found", observation="outcome_unknown"
            )
        if (
            operation["call_id"] != context.tool_call_id
            or operation["arguments_digest"] != context.arguments_digest
        ):
            return ToolEffectReconciliationResult(outcome="conflict", observation="outcome_unknown")
        if operation["receipt"] is None:
            return ToolEffectReconciliationResult(outcome="not_found", observation="partial")
        authoritative = ToolEffectReceipt.model_validate_json(operation["receipt"])
        if receipt is not None and receipt != authoritative:
            return ToolEffectReconciliationResult(outcome="conflict", observation="outcome_unknown")
        return ToolEffectReconciliationResult(
            outcome=authoritative.outcome,
            receipt=authoritative,
            observation="sent" if authoritative.outcome == "completed" else "not_sent",
        )


def deployment_reconciliation(state: DeploymentState) -> ToolEffectReconciliationRegistration:
    return ToolEffectReconciliationRegistration(
        reconciler=DeploymentReconciler(state),
        spec=ToolEffectReconcilerSpec(
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="examples:deployment-reconciler",
                behavior_version="1",
                implementation_version="1",
            ),
            receipt_schema="counterfactual-deployment",
            receipt_schema_version=1,
            required=True,
            supports_lookup=True,
            result_schema={
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "release": {"type": "string"},
                    "version": {"type": "integer"},
                    "mutation_count": {"type": "integer"},
                },
                "required": ["service", "release", "version", "mutation_count"],
                "additionalProperties": False,
            },
            integrity_fields=("arguments_sha256",),
            resource_version_fields=("service_version",),
        ),
    )
