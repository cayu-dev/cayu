"""Runtime-owned control input that raises one application-proposed tool call.

A proposal session runs through the ordinary session loop: its tool call
passes the agent's tool policy, pauses in the ordinary approval checkpoint and
recovers like any model-raised call. Only the model step is replaced. This
provider reads the proposal from the session's own durable transcript, so a
worker that restarts before the first step completes raises the same call.
It never calls a model and is priced at zero.

The transcript is redacted when it is written, and a redacted secret reads as
``REDACTED_SECRET``. So that arguments which merely quote that text stay
distinguishable, the proposal writes the marker's opening bracket as a JSON
unicode escape, which ``json.loads`` reads back exactly. A bare marker in the
stored text can then only come from redaction, and the provider refuses to
raise that proposal: the call it would raise is not the call that was proposed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from cayu._validation import copy_durable_json_object, require_clean_nonblank
from cayu.budgets.billing import BillingIdentity
from cayu.messages import Message, TextPart, ToolResultPart
from cayu.providers.base import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    UsageDialect,
)
from cayu.vaults.redaction import REDACTED_SECRET

TOOL_CALL_PROPOSAL_PROVIDER_NAME = "cayu-tool-proposal"
TOOL_CALL_PROPOSAL_MODEL = "tool-proposal"
# The runtime-owned zero-priced identity shared with native tool dispatch.
_BILLING_PROVIDER_NAME = "cayu-tool-dispatch"
_BILLING_RESOURCE_ID = "tool-dispatch"
_PROPOSAL_KEY = "cayu_tool_call_proposal"
_PROPOSAL_VERSION = 1
_ZERO_USAGE = {"usage": {"input_tokens": 0, "output_tokens": 0}}
# JSON text never has a "[" right after a backslash, so this escape only ever
# rewrites the marker inside a string and never changes what it decodes to.
_ESCAPED_REDACTED_SECRET = "\\u005b" + REDACTED_SECRET[1:]
# How every stored proposal starts: its body has exactly this one key.
_PROPOSAL_ENVELOPE_PREFIX = '{"' + _PROPOSAL_KEY + '": {'


def proposal_message(*, tool_name: str, arguments: dict[str, Any], tool_call_id: str) -> Message:
    """Return the user message that records one proposal in the transcript."""

    body = {
        _PROPOSAL_KEY: {
            "version": _PROPOSAL_VERSION,
            "tool_name": require_clean_nonblank(tool_name, "tool_name"),
            "tool_call_id": require_clean_nonblank(tool_call_id, "tool_call_id"),
            "arguments": copy_durable_json_object(arguments, "arguments"),
        }
    }
    text = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return Message.text("user", text.replace(REDACTED_SECRET, _ESCAPED_REDACTED_SECRET))


def parse_proposal_message(message: Message) -> tuple[str, dict[str, Any], str] | None:
    """Return ``(tool_name, arguments, tool_call_id)`` from a proposal message.

    Context policies may add parts to the message the provider sees, such as a
    recall manifest prepended to the first user message, so the proposal is the
    one text part that parses as a proposal; other parts are ignored. A
    proposal the transcript redacted raises ``ValueError``.
    """

    if message.role != "user":
        return None
    proposals = [
        proposal
        for part in message.content
        if isinstance(part, TextPart) and (proposal := _parse_proposal_text(part.text)) is not None
    ]
    return proposals[0] if len(proposals) == 1 else None


def _parse_proposal_text(text: str) -> tuple[str, dict[str, Any], str] | None:
    try:
        body = json.loads(text)
    except ValueError:
        # Redaction can also break the JSON of a proposal (a short secret
        # matching inside an escape); that is a redacted proposal, not text.
        # Only the envelope proposal_message writes counts, so other parts a
        # context policy adds are still ignored.
        if text.startswith(_PROPOSAL_ENVELOPE_PREFIX) and REDACTED_SECRET in text:
            raise _redacted_proposal() from None
        return None
    if type(body) is not dict or set(body) != {_PROPOSAL_KEY}:
        return None
    proposal = body[_PROPOSAL_KEY]
    if (
        type(proposal) is not dict
        or proposal.get("version") != _PROPOSAL_VERSION
        or type(proposal.get("tool_name")) is not str
        or type(proposal.get("tool_call_id")) is not str
        or type(proposal.get("arguments")) is not dict
    ):
        return None
    if REDACTED_SECRET in text:
        raise _redacted_proposal()
    return proposal["tool_name"], proposal["arguments"], proposal["tool_call_id"]


def _redacted_proposal() -> ValueError:
    return ValueError(
        "The tool call proposal was redacted in the session transcript, so its exact "
        "call cannot be raised; bind the credential to the tool instead of passing it "
        "as an argument."
    )


class ToolCallProposalProvider(ModelProvider):
    """Emit the proposed call once, then finish after its result is recorded."""

    name = TOOL_CALL_PROPOSAL_PROVIDER_NAME
    billing_provider_name = _BILLING_PROVIDER_NAME
    usage_dialect = UsageDialect.GENERIC

    async def billing_identity_for_request(self, request: ModelRequest) -> BillingIdentity:
        return BillingIdentity(
            provider_name=_BILLING_PROVIDER_NAME, resource_id=_BILLING_RESOURCE_ID
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        text = "This session only runs tool call proposals; nothing was proposed."
        for message in reversed(request.messages):
            results = [part for part in message.content if isinstance(part, ToolResultPart)]
            if results:
                text = (
                    "The proposed tool call did not run successfully."
                    if any(part.is_error for part in results)
                    else "The proposed tool call completed."
                )
                break
            if message.role != "user":
                continue
            try:
                proposal = parse_proposal_message(message)
            except ValueError as exc:
                raise ModelProviderError(
                    str(exc), provider=self.name, error_type="RedactedProposal", retryable=False
                ) from None
            if proposal is None:
                break
            tool_name, arguments, tool_call_id = proposal
            yield ModelStreamEvent.tool_call(name=tool_name, arguments=arguments, id=tool_call_id)
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls", **_ZERO_USAGE})
            return
        yield ModelStreamEvent.text_delta(text)
        yield ModelStreamEvent.completed({"finish_reason": "stop", **_ZERO_USAGE})


def _execution_profile_material(provider: ToolCallProposalProvider) -> dict[str, Any]:
    """The provider has no configuration; its runtime version is its identity."""

    del provider
    return {}


__all__ = [
    "TOOL_CALL_PROPOSAL_MODEL",
    "TOOL_CALL_PROPOSAL_PROVIDER_NAME",
    "ToolCallProposalProvider",
    "parse_proposal_message",
    "proposal_message",
]
