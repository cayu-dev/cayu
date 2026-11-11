"""What the connector toolkits share: the client, refusals as tool errors, bounded answers and
the approval wrapper."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import Any

from cayu.connections import (
    CloudConnectionBroker,
    ConnectionCallError,
    ConnectionClient,
    ConnectionResponse,
    retry_after_seconds,
)
from cayu.tools._errors import invalid_tool_arguments_result
from cayu.tools.base import ToolResult
from cayu.tools.policy import (
    AllowAllToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)

#: The largest answer the model reads: a result's ``content``, in UTF-8 bytes.
ANSWER_BUDGET_BYTES = 60 * 1024
#: The terminal payload each toolkit tool declares. A result carries its answer twice, as the
#: model's ``content`` (escaped once more inside the payload's JSON) and as ``structured``, so
#: the bound holds a full answer twice with room for its escaping.
MAX_TERMINAL_PAYLOAD_BYTES = 160 * 1024
#: The largest serialized result a tool returns, leaving the runtime its own room.
_RESULT_BUDGET_BYTES = MAX_TERMINAL_PAYLOAD_BYTES - 4 * 1024
#: The longest argument an approval reason quotes.
_REASON_ARGUMENT_CHARACTERS = 80

_ADMIN = "An admin must bind the slot {slot} to a ready connection, or reconnect it."
_NOT_ALLOWED = "This call isn't allowed for this Agent on the slot {slot}; don't retry it."
_ADVICE = {
    "connection_unbound": _ADMIN,
    "connection_not_ready": _ADMIN,
    "connection_unavailable": _ADMIN,
    "credential_invalid": _ADMIN,
    "scope_not_allowed": _NOT_ALLOWED,
    "path_not_allowed": _NOT_ALLOWED,
    "method_not_allowed": _NOT_ALLOWED,
    "connector_not_brokered": _NOT_ALLOWED,
    "agent_credential_invalid": (
        "The Agent's own connections credential was refused; redeploy the Agent. Don't retry."
    ),
    "upstream_response_too_large": (
        "The answer was too large; ask for less (fewer rows or columns, or a smaller limit)."
    ),
    "body_too_large": "The request is too large; send less.",
    "request_too_large": "The request is too large; send less.",
}
TEMPORARY = "A temporary failure; retry later."
_RATE_LIMITED = "Too many calls; retry after {delay}."


def resolved_client(client: ConnectionClient | None) -> ConnectionClient | None:
    """The given client, else the broker Cayu Cloud bound this process to."""

    return client if client is not None else CloudConnectionBroker.from_environment()


def no_client_result() -> ToolResult:
    return error_result(
        "no_connection_client",
        "This process has no connection client.",
        "Outside Cayu Cloud, give the toolkit a client, such as LocalConnections.",
    )


def error_result(code: str, message: str, advice: str | None = None, **details: Any) -> ToolResult:
    """A tool error the model can act on: what failed, its code and what to do."""

    structured: dict[str, Any] = {"error": code, "message": message}
    if advice is not None:
        structured["advice"] = advice
    structured.update({name: value for name, value in details.items() if value is not None})
    content = f"{message} ({code})" + (f" {advice}" if advice else "")
    return ToolResult(content=content, structured=structured, is_error=True)


def invalid_arguments(message: str) -> ToolResult:
    return invalid_tool_arguments_result(ValueError(message))


def call_error_result(
    error: ConnectionCallError, slot: str, advice: Mapping[str, str] | None = None
) -> ToolResult:
    """A call the client or broker refused, as a tool error naming its code and what to do;
    ``advice`` gives a toolkit's own words for some codes."""

    code = error.code
    if code is None:
        return error_result(
            "connection_failed", str(error), TEMPORARY, retry_after_seconds=error.retry_after
        )
    if code == "too_many_calls":
        return rate_limited_result(str(error), error.retry_after)
    known = {**_ADVICE, **(advice or {})}
    if code in known:
        said = known[code].format(slot=slot)
    elif code.startswith("upstream_"):
        said = TEMPORARY
    else:
        said = "The call was refused; don't retry it unchanged."
    return error_result(code, str(error), said, retry_after_seconds=error.retry_after)


def rate_limited_result(message: str, retry_after: float | None) -> ToolResult:
    delay = "a short wait" if retry_after is None else f"{math.ceil(retry_after)} seconds"
    return error_result(
        "too_many_calls",
        message,
        _RATE_LIMITED.format(delay=delay),
        retry_after_seconds=retry_after,
    )


def connector_rate_limited(response: ConnectionResponse, connector: str) -> ToolResult:
    return rate_limited_result(
        f"{connector} rate-limited the call.", retry_after_seconds(response.header("retry-after"))
    )


def json_body(response: ConnectionResponse) -> dict[str, Any] | None:
    """The connector's answer as a JSON object; None when it isn't one."""

    try:
        answer = json.loads(response.body)
    except ValueError:
        return None
    return answer if isinstance(answer, dict) else None


def answer_result(answer: Mapping[str, Any]) -> ToolResult:
    """A successful answer: compact JSON for the model, the same object for applications."""

    return ToolResult(
        content=json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
        structured=dict(answer),
    )


def answer_fits(result: ToolResult) -> bool:
    """Whether the model's answer is within its budget and the whole result within the
    tool's declared terminal payload."""

    if len(result.content.encode()) > ANSWER_BUDGET_BYTES:
        return False
    size = len(json.dumps(result.model_dump(mode="json"), ensure_ascii=False).encode())
    return size <= _RESULT_BUDGET_BYTES


def shortened(text: str, limit: int) -> tuple[str, bool]:
    """``text`` cut to ``limit`` characters, ending in an ellipsis when cut."""

    if len(text) <= limit:
        return text, False
    return text[: limit - 1] + "…", True


def reason_argument(value: object) -> str:
    """A model-given argument as an approval reason quotes it: printable, short."""

    text = "".join(character for character in str(value) if character.isprintable()).strip()
    return shortened(text, _REASON_ARGUMENT_CHARACTERS)[0] or "(none)"


class _ApprovalPolicy(ToolPolicy):
    """Ask for approval on named write tools; every other decision is ``inner``'s."""

    def __init__(
        self,
        inner: ToolPolicy | None,
        reasons: Mapping[str, Callable[[Mapping[str, Any]], str]],
    ) -> None:
        inner = AllowAllToolPolicy() if inner is None else inner
        if not isinstance(inner, ToolPolicy):
            raise TypeError("inner must be a ToolPolicy.")
        self.inner = inner
        self._reasons = dict(reasons)

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        decided = await self.inner.authorize(request)
        reason = self._reasons.get(request.tool_name)
        if reason is None or decided.decision is not ToolPolicyDecision.ALLOW:
            return decided
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            reason=reason(request.arguments),
            metadata=decided.metadata,
        )


def approval_policy(
    inner: ToolPolicy | None,
    reasons: Mapping[str, Callable[[Mapping[str, Any]], str]],
) -> ToolPolicy:
    return _ApprovalPolicy(inner, reasons)
