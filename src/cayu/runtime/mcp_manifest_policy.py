from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cayu._validation import copy_json_value


class McpManifestPolicyAction(StrEnum):
    ALLOW = "allow"
    ALERT = "alert"
    BLOCK = "block"


class McpManifestPolicyDecision(BaseModel):
    """Decision for one checked MCP tool manifest."""

    model_config = ConfigDict(extra="forbid")

    action: McpManifestPolicyAction
    status: str
    matched_changes: list[str] = Field(default_factory=list)
    reason: str


class McpManifestPolicy(BaseModel):
    """Controls whether changed MCP tool manifests may reach the model."""

    model_config = ConfigDict(extra="forbid")

    on_first_seen: McpManifestPolicyAction = McpManifestPolicyAction.ALLOW
    on_unchanged: McpManifestPolicyAction = McpManifestPolicyAction.ALLOW
    on_changed: McpManifestPolicyAction = McpManifestPolicyAction.BLOCK
    on_server_changed: McpManifestPolicyAction | None = None
    on_tools_added: McpManifestPolicyAction | None = None
    on_tools_removed: McpManifestPolicyAction | None = None
    on_tools_changed: McpManifestPolicyAction | None = None

    def decide(
        self,
        *,
        status: str,
        diff: Mapping[str, Any],
    ) -> McpManifestPolicyDecision:
        if status == "first_seen":
            return McpManifestPolicyDecision(
                action=self.on_first_seen,
                status=status,
                reason=_decision_reason(self.on_first_seen, "MCP manifest is first seen."),
            )
        if status == "unchanged":
            return McpManifestPolicyDecision(
                action=self.on_unchanged,
                status=status,
                reason=_decision_reason(self.on_unchanged, "MCP manifest is unchanged."),
            )

        candidates: list[tuple[str, McpManifestPolicyAction]] = []
        if diff.get("server_changed") is True:
            candidates.append(("server_changed", self.on_server_changed or self.on_changed))
        if _has_items(diff.get("added_tools")):
            candidates.append(("tools_added", self.on_tools_added or self.on_changed))
        if _has_items(diff.get("removed_tools")):
            candidates.append(("tools_removed", self.on_tools_removed or self.on_changed))
        if _has_items(diff.get("changed_tools")):
            candidates.append(("tools_changed", self.on_tools_changed or self.on_changed))
        if not candidates:
            candidates.append(("changed", self.on_changed))

        max_rank = max(_ACTION_RANK[action] for _, action in candidates)
        matched_changes = [
            change for change, action in candidates if _ACTION_RANK[action] == max_rank
        ]
        action = next(action for _, action in candidates if _ACTION_RANK[action] == max_rank)
        change_text = ", ".join(matched_changes)
        return McpManifestPolicyDecision(
            action=action,
            status=status,
            matched_changes=matched_changes,
            reason=_decision_reason(action, f"MCP manifest changed: {change_text}."),
        )


class McpManifestPolicyError(RuntimeError):
    """Raised when MCP manifest policy blocks a model step."""


def copy_mcp_manifest_policy(
    policy: McpManifestPolicy | None,
) -> McpManifestPolicy | None:
    if policy is None:
        return None
    if not isinstance(policy, McpManifestPolicy):
        raise TypeError("mcp_manifest_policy must be a McpManifestPolicy.")
    return McpManifestPolicy.model_validate(policy.model_dump(mode="json"))


def mcp_manifest_policy_payload(
    decision: McpManifestPolicyDecision,
) -> dict[str, Any]:
    return copy_json_value(decision.model_dump(mode="json"), "policy")


_ACTION_RANK = {
    McpManifestPolicyAction.ALLOW: 0,
    McpManifestPolicyAction.ALERT: 1,
    McpManifestPolicyAction.BLOCK: 2,
}


def _has_items(value: object) -> bool:
    return isinstance(value, list | tuple) and len(value) > 0


def _decision_reason(action: McpManifestPolicyAction, detail: str) -> str:
    if action == McpManifestPolicyAction.BLOCK:
        return f"{detail} Policy action: block."
    if action == McpManifestPolicyAction.ALERT:
        return f"{detail} Policy action: alert."
    return f"{detail} Policy action: allow."
