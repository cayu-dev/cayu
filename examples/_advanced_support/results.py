from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class SessionEvidence:
    session_id: str
    role: str
    parent_session_id: str | None
    causal_budget_id: str
    status: str
    usage: dict[str, int] = field(default_factory=dict)
    model_steps: int = 0
    tool_calls: int = 0
    recovery_state: str = "not-required"
    taint_labels: list[str] = field(default_factory=list)
    compaction_count: int = 0
    receipt_ids: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "role": self.role,
            "parent_session_id": self.parent_session_id,
            "causal_budget_id": self.causal_budget_id,
            "status": self.status,
            "usage": self.usage,
            "model_steps": self.model_steps,
            "tool_calls": self.tool_calls,
            "recovery_state": self.recovery_state,
            "taint_labels": self.taint_labels,
            "compaction_count": self.compaction_count,
            "receipt_ids": self.receipt_ids,
        }


@dataclass
class ScenarioResult:
    scenario: str
    mode: str
    status: str
    assertions: dict[str, bool]
    sessions: list[SessionEvidence]
    provider_name: str | None = None
    model: str | None = None
    run_id: str = field(default_factory=lambda: uuid4().hex)
    metrics: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    output_path: Path | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "schema": "cayu.advanced-example-result.v1",
            "scenario": self.scenario,
            "mode": self.mode,
            "status": self.status,
            "run_id": self.run_id,
            "provider_name": self.provider_name,
            "model": self.model,
            "assertions": self.assertions,
            "sessions": [session.as_json() for session in self.sessions],
            "metrics": self.metrics,
            "outputs": self.outputs,
        }

    def write(self, root: Path) -> Path:
        destination = root / ".cayu-example-results" / self.scenario / f"{self.run_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.output_path = destination
        return destination

    def require_verified(self) -> None:
        failures = [name for name, passed in self.assertions.items() if not passed]
        if failures:
            raise RuntimeError(f"{self.scenario} failed assertions: {', '.join(failures)}")
        if self.status != "verified":
            raise RuntimeError(f"{self.scenario} status is {self.status!r}, not 'verified'.")
