from __future__ import annotations

import asyncio

import pytest

from cayu import (
    ANY_TAINT_LABEL,
    TAINT_LABELS_METADATA_KEY,
    AgentSpec,
    AllowlistRule,
    DenyPatternRule,
    ParameterConstrainedToolPolicy,
    RequiredFieldRule,
    TaintAwareToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    metadata_with_taint_labels,
    taint_labels_from_metadata,
)
from cayu.runtime import Session, SessionStatus


def _request(
    *,
    tool_name: str = "send_email",
    arguments: dict | None = None,
) -> ToolPolicyRequest:
    return ToolPolicyRequest(
        session=Session(
            id="sess_policy",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            status=SessionStatus.RUNNING,
        ),
        agent=AgentSpec(name="assistant", model="fake-model"),
        tool_name=tool_name,
        tool_call_id="call_1",
        arguments=arguments or {},
    )


def test_allowlist_rule_allows_only_explicit_string_values() -> None:
    rule = AllowlistRule("to", values=["ops@example.com", "billing@example.com"])

    assert rule.check({"to": "ops@example.com"}) is None
    assert rule.check({}) is None
    assert rule.check({"to": "attacker@example.net"}) == "Parameter 'to' value is not allowed."
    assert rule.check({"to": 123}) == ("Parameter 'to' must be a string for allowlist validation.")


def test_allowlist_rule_supports_nested_argument_paths() -> None:
    rule = AllowlistRule("request.host", values=["api.internal"])

    assert rule.check({"request": {"host": "api.internal"}}) is None
    assert rule.check({"request": {"host": "evil.example"}}) == (
        "Parameter 'request.host' value is not allowed."
    )


def test_deny_pattern_rule_rejects_matching_string_values() -> None:
    rule = DenyPatternRule("shell", patterns=[r"\bcurl\b", r"\bwget\b"])

    assert rule.check({"shell": "ls -la"}) is None
    assert rule.check({}) is None
    assert rule.check({"shell": "curl https://evil.example"}) == (
        "Parameter 'shell' matches a denied pattern."
    )
    assert rule.check({"shell": ["curl"]}) == (
        "Parameter 'shell' must be a string for pattern validation."
    )


def test_required_field_rule_rejects_missing_blank_or_empty_values() -> None:
    rule = RequiredFieldRule("payload.message")

    assert rule.check({"payload": {"message": "hello"}}) is None
    assert rule.check({"payload": {}}) == "Required parameter 'payload.message' is missing."
    assert rule.check({"payload": {"message": "   "}}) == (
        "Required parameter 'payload.message' is empty."
    )
    assert rule.check({"payload": {"message": []}}) == (
        "Required parameter 'payload.message' is empty."
    )


def test_parameter_constrained_policy_allows_unconstrained_tools() -> None:
    policy = ParameterConstrainedToolPolicy(
        {"send_email": [AllowlistRule("to", values=["ops@example.com"])]}
    )

    result = asyncio.run(policy.authorize(_request(tool_name="echo", arguments={"text": "hi"})))

    assert result.decision == ToolPolicyDecision.ALLOW


def test_parameter_constrained_policy_denies_first_rule_violation_with_metadata() -> None:
    policy = ParameterConstrainedToolPolicy(
        {
            "send_email": [
                RequiredFieldRule("to"),
                AllowlistRule("to", values=["ops@example.com"]),
            ]
        }
    )

    result = asyncio.run(
        policy.authorize(_request(arguments={"to": "attacker@example.net", "body": "hi"}))
    )

    assert result.decision == ToolPolicyDecision.DENY
    assert result.reason == "Parameter 'to' value is not allowed."
    assert result.metadata == {
        "policy": "parameter_constrained",
        "tool_name": "send_email",
        "parameter": "to",
        "rule": "AllowlistRule",
        "rule_index": 1,
    }


def test_parameter_constrained_policy_can_require_approval_on_violation() -> None:
    policy = ParameterConstrainedToolPolicy(
        {"http_request": [DenyPatternRule("url", patterns=[r"^https://external\.example"])]},
        decision=ToolPolicyDecision.REQUIRE_APPROVAL,
    )

    result = asyncio.run(
        policy.authorize(
            _request(tool_name="http_request", arguments={"url": "https://external.example"})
        )
    )

    assert result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
    assert result.reason == "Parameter 'url' matches a denied pattern."
    assert result.metadata["policy"] == "parameter_constrained"


def test_parameter_constrained_policy_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="cannot be ALLOW"):
        ParameterConstrainedToolPolicy({}, decision=ToolPolicyDecision.ALLOW)

    with pytest.raises(ValueError, match="'invalid' is not a valid ToolPolicyDecision"):
        ParameterConstrainedToolPolicy({}, decision="invalid")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="dict"):
        ParameterConstrainedToolPolicy([])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="ParameterRule"):
        ParameterConstrainedToolPolicy({"send_email": ["not a rule"]})  # type: ignore[list-item]

    with pytest.raises(ValueError, match="cannot be empty"):
        ParameterConstrainedToolPolicy({"send_email": []})

    with pytest.raises(ValueError, match="cannot be empty"):
        AllowlistRule("to", values=[])

    with pytest.raises(ValueError, match="cannot be empty"):
        DenyPatternRule("shell", patterns=[])

    with pytest.raises(ValueError, match="cannot be blank"):
        RequiredFieldRule("payload..message")


def test_taint_aware_policy_allows_before_untrusted_source() -> None:
    policy = TaintAwareToolPolicy(
        taint_sources={"read_email": ["external_email"]},
        protected_tools={"send_email": ["external_email"]},
    )

    result = asyncio.run(policy.authorize(_request(tool_name="send_email")))

    assert result.decision == ToolPolicyDecision.ALLOW


def test_taint_aware_policy_requires_approval_after_matching_taint() -> None:
    policy = TaintAwareToolPolicy(
        taint_sources={"read_email": ["external_email"]},
        protected_tools={"send_email": ["external_email"]},
    )

    result = asyncio.run(
        policy.authorize(
            _request(
                tool_name="send_email",
                arguments={},
            ).model_copy(
                update={
                    "metadata": metadata_with_taint_labels(
                        {},
                        ["external_email", "artifact_pdf"],
                    )
                }
            )
        )
    )

    assert result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
    assert result.metadata == {
        "policy": "taint_aware",
        "tool_name": "send_email",
        "taint_labels": ["artifact_pdf", "external_email"],
        "matched_taint_labels": ["external_email"],
        "protected_taint_labels": ["external_email"],
    }


def test_taint_aware_policy_does_not_count_current_source_before_result() -> None:
    policy = TaintAwareToolPolicy(
        taint_sources={"read_and_send": ["external_email"]},
        protected_tools={"read_and_send": ["external_email"]},
    )

    result = asyncio.run(policy.authorize(_request(tool_name="read_and_send")))

    assert result.decision == ToolPolicyDecision.ALLOW


def test_taint_aware_policy_supports_any_taint_label() -> None:
    policy = TaintAwareToolPolicy(
        taint_sources={"read_pdf": ["artifact_pdf"]},
        protected_tools={"send_email": [ANY_TAINT_LABEL]},
        decision=ToolPolicyDecision.DENY,
    )

    result = asyncio.run(
        policy.authorize(
            _request(
                tool_name="send_email",
            ).model_copy(update={"metadata": metadata_with_taint_labels({}, ["artifact_pdf"])})
        )
    )

    assert result.decision == ToolPolicyDecision.DENY
    assert result.metadata["matched_taint_labels"] == ["artifact_pdf"]


def test_taint_aware_policy_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="cannot be ALLOW"):
        TaintAwareToolPolicy(
            taint_sources={"read_email": ["external_email"]},
            protected_tools={"send_email": ["external_email"]},
            decision=ToolPolicyDecision.ALLOW,
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        TaintAwareToolPolicy(taint_sources={}, protected_tools={"send_email": ["external"]})

    with pytest.raises(ValueError, match="cannot contain"):
        TaintAwareToolPolicy(
            taint_sources={"read_email": [ANY_TAINT_LABEL]},
            protected_tools={"send_email": ["external_email"]},
        )


def test_taint_metadata_helpers_copy_and_validate_labels() -> None:
    metadata = {"tenant": {"id": "acme"}}
    copied = metadata_with_taint_labels(metadata, ["external_email"])

    copied["tenant"]["id"] = "mutated"
    assert metadata == {"tenant": {"id": "acme"}}
    assert copied[TAINT_LABELS_METADATA_KEY] == ["external_email"]
    assert taint_labels_from_metadata(copied) == frozenset({"external_email"})

    with pytest.raises(ValueError, match="cannot contain"):
        metadata_with_taint_labels({}, [ANY_TAINT_LABEL])
