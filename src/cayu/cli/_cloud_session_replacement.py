"""Strict source declaration for Cloud's approval replacement consent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from cayu.cli._cloud_api import CloudApiError

_ERROR = (
    "[session_replacement] requires 1 to 20 unique approvals with exactly agent_name, "
    "tool_name and reason. Names must be 1 to 128 printable characters without "
    "surrounding whitespace or '*'. Reasons must be 1 to 512 printable characters without leading "
    "or trailing whitespace."
)


@dataclass(frozen=True)
class CloudApprovalReplacementConsent:
    agent_name: str
    tool_name: str
    reason: str


def _valid_name(value: object) -> bool:
    return (
        type(value) is str
        and 1 <= len(value) <= 128
        and value == value.strip()
        and value.isprintable()
        and "*" not in value
    )


def _validate_rule(rule: CloudApprovalReplacementConsent) -> None:
    if (
        not _valid_name(rule.agent_name)
        or not _valid_name(rule.tool_name)
        or type(rule.reason) is not str
        or not 1 <= len(rule.reason) <= 512
        or rule.reason != rule.reason.strip()
        or not rule.reason.isprintable()
    ):
        raise CloudApiError("manifest_invalid", _ERROR)


def validate_session_replacement(
    rules: tuple[CloudApprovalReplacementConsent, ...] | None,
) -> None:
    if rules is None:
        return
    if type(rules) is not tuple or not 1 <= len(rules) <= 20:
        raise CloudApiError("manifest_invalid", _ERROR)
    seen: set[tuple[str, str]] = set()
    for rule in rules:
        if type(rule) is not CloudApprovalReplacementConsent:
            raise CloudApiError("manifest_invalid", _ERROR)
        _validate_rule(rule)
        pair = (rule.agent_name, rule.tool_name)
        if pair in seen:
            raise CloudApiError("manifest_invalid", _ERROR)
        seen.add(pair)


def session_replacement(value: object) -> tuple[CloudApprovalReplacementConsent, ...]:
    if type(value) is not dict or set(value) != {"approvals"}:
        raise CloudApiError("manifest_invalid", _ERROR)
    approvals = cast("dict[str, object]", value)["approvals"]
    if type(approvals) is not list or not 1 <= len(approvals) <= 20:
        raise CloudApiError("manifest_invalid", _ERROR)
    rules = []
    for item in approvals:
        if type(item) is not dict or set(item) != {"agent_name", "tool_name", "reason"}:
            raise CloudApiError("manifest_invalid", _ERROR)
        rule = CloudApprovalReplacementConsent(**cast("dict[str, str]", item))
        rules.append(rule)
    result = tuple(rules)
    validate_session_replacement(result)
    return tuple(sorted(result, key=lambda rule: (rule.agent_name, rule.tool_name)))


def render_session_replacement(rules: tuple[CloudApprovalReplacementConsent, ...]) -> str:
    validate_session_replacement(rules)
    return "".join(
        "\n[[session_replacement.approvals]]\n"
        f"agent_name = {json.dumps(rule.agent_name, ensure_ascii=False)}\n"
        f"tool_name = {json.dumps(rule.tool_name, ensure_ascii=False)}\n"
        f"reason = {json.dumps(rule.reason, ensure_ascii=False)}\n"
        for rule in sorted(rules, key=lambda rule: (rule.agent_name, rule.tool_name))
    )
