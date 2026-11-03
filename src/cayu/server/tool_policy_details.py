"""Content-safe explanations of maintained tool policies, never policy evaluation.

Private allowlist values and patterns are deliberately not projected. Extensions
and custom rules remain explicitly incomplete instead of inheriting base semantics.
"""

from __future__ import annotations

from cayu.tools.policy import (
    AllowAllToolPolicy,
    AllowlistRule,
    AlwaysRequireApprovalToolPolicy,
    DenyPatternRule,
    EnvironmentScopedToolPolicy,
    EveryCallRule,
    GuardedToolPolicy,
    ParameterConstrainedToolPolicy,
    RequiredAllowlistRule,
    RequiredFieldRule,
    StaticToolPolicy,
    TaintAwareToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
)
from cayu.tools.structured_commands import StructuredCommandToolPolicy


def _violation_action(decision: ToolPolicyDecision) -> str:
    return (
        "requires approval"
        if decision is ToolPolicyDecision.REQUIRE_APPROVAL
        else "denies the call"
    )


def policy_details(policy: ToolPolicy, name: str) -> tuple[list[str], bool]:
    """Return policy-only conditions and whether all conditions can be described.

    These declarations never grant authority: argument validation, runtime hooks,
    grants, mandatory controls and execution admission still apply per call.
    """
    return _details(policy, name, frozenset())


def _details(policy: ToolPolicy, name: str, seen: frozenset[int]) -> tuple[list[str], bool]:
    if id(policy) in seen:
        return ["Policy composition is cyclic; inspect the application policy."], False
    seen = seen | {id(policy)}
    if type(policy) is StructuredCommandToolPolicy:
        base_details, base_complete = (
            (["This policy has no base restrictions for this tool."], True)
            if policy._base_policy is None
            else _details(policy._base_policy, name, seen)
        )
        if name != "run_command":
            return base_details, base_complete
        return [
            "The base policy is evaluated first.",
            *base_details,
            "Structured commands must then match admitted selector, argument, working-directory and execution rules. A selector can require approval. Exact authority configuration is not published.",
        ], False
    if type(policy) is AllowAllToolPolicy:
        return ["This policy allows every call without asking for approval."], True
    if type(policy) is StaticToolPolicy:
        denied = name in policy.deny or (policy.allow is not None and name not in policy.allow)
        return [
            "This policy denies this tool."
            if denied
            else "This policy allows this tool without approval."
        ], True
    if type(policy) is AlwaysRequireApprovalToolPolicy:
        if policy.tools is not None and name not in policy.tools:
            return ["This tool is outside the policy's approval set; this policy allows it."], True
        expiry = (
            "Approval does not expire under this policy."
            if policy.expires_in_seconds is None
            else f"An unanswered approval expires after {policy.expires_in_seconds:g} seconds and denies the call."
        )
        return ["Every call to this tool requires a person's approval.", expiry], True
    if type(policy) is EnvironmentScopedToolPolicy:
        environments = sorted(policy.allow.get(name, ()))
        if not environments:
            return ["This policy denies this tool in every environment."], True
        return [
            "Allowed only in these trusted execution environments: "
            + ", ".join(environments)
            + ". Missing or other environments deny the call."
        ], True
    if type(policy) is GuardedToolPolicy:
        details, complete = _details(policy.then, name, seen)
        if not policy.guards:
            return details, complete
        return [
            "Application guards run first and can deny the call. Their conditions are not published.",
            *details,
        ], False
    if type(policy) is ParameterConstrainedToolPolicy:
        rules = policy.rules.get(name, ())
        if not rules:
            return ["No parameter rules constrain this tool under this policy."], True
        action = _violation_action(policy.decision)
        details: list[str] = [
            f"An authority-rule violation {action}. Invalid arguments deny the call even when approval is configured."
        ]
        complete = True
        for rule in rules:
            kind = type(rule)
            if kind is EveryCallRule:
                details.append("Every call violates the authority rule.")
            elif kind is RequiredFieldRule:
                details.append(
                    f"{rule.parameter}: a non-empty value is required; missing or empty values deny the call."
                )
            elif kind in {AllowlistRule, RequiredAllowlistRule}:
                required = (
                    "A non-empty string is required; invalid values deny the call. "
                    if kind is RequiredAllowlistRule
                    else "Omitting this argument is allowed. "
                )
                details.append(
                    f"{rule.parameter}: {required}A value outside the configured allowlist violates the authority rule. The allowlist is private and is not published."
                )
                complete = False
            elif kind is DenyPatternRule:
                details.append(
                    f"{rule.parameter}: a string matching a configured deny pattern violates the authority rule. The patterns are private and are not published."
                )
                complete = False
            else:
                details.append(
                    "An application-defined parameter rule applies; its conditions are not published."
                )
                complete = False
        return details, complete
    if type(policy) is TaintAwareToolPolicy:
        if not policy.protected_tools.get(name):
            return ["This tool is not protected by this policy's untrusted-output rules."], True
        action = _violation_action(policy.decision)
        return [
            f"This policy {action} when prior or same-round untrusted-output labels match this tool's protection rules. Exact application labels are not published."
        ], False
    return [
        "An application-defined policy decides per call. It has not published inspectable conditions; ask the Agent developer to inspect its policy source."
    ], False
