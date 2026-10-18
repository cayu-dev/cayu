"""Admit changed arguments for an approval before the approval is claimed."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from cayu._validation import canonical_durable_json_bytes, copy_durable_metadata
from cayu.approvals.authorization_grants import (
    APPROVED_ARGUMENTS_REAUTHORIZATION_METADATA_KEY,
    MANDATORY_CONTROL_OFFER_REASON,
    TOOL_AUTHORIZATION_GRANT_METADATA_KEY,
    TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY,
    TOOL_MANDATORY_APPROVAL_METADATA_KEY,
)
from cayu.approvals.tools import (
    PendingToolApproval,
    ToolApprovalArgumentsRefused,
    ToolApprovalRequest,
    ToolPolicyEvidence,
)
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._tool_invocation.admission import ToolInvocationAdmission
from cayu.runtime._tool_round_staging import _redactor_for_tool_calls
from cayu.runtime.public_authority import PublicAuthorityAliasCodec
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions._checkpoint_secret_validation import durable_value_contains_secret
from cayu.sessions.records import Session
from cayu.tools.gateway import tool_argument_validation_error
from cayu.tools.policy import (
    TOOL_POLICY_REAUTHORIZATION_METADATA_KEY,
    ToolPolicyDecision,
)
from cayu.vaults.redaction import SecretRedactor, contains_redacted_secret

_MAX_CHANGED_FIELDS = 64
_MAX_CHANGED_FIELD_CHARS = 64


# Runtime-owned metadata that never expresses the authority an approval covers.
_RUNTIME_POLICY_METADATA_KEYS = frozenset(
    {
        TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY,
        TOOL_AUTHORIZATION_GRANT_METADATA_KEY,
        TOOL_POLICY_REAUTHORIZATION_METADATA_KEY,
        APPROVED_ARGUMENTS_REAUTHORIZATION_METADATA_KEY,
    }
)


def _same_arguments(approved: dict[str, Any], proposed: dict[str, Any]) -> bool:
    """Whether approved arguments are byte-for-byte the model's proposal."""

    try:
        return canonical_durable_json_bytes(
            approved, "approved_arguments"
        ) == canonical_durable_json_bytes(proposed, "approved_arguments")
    except (TypeError, ValueError):
        return False


def _approval_authority(metadata: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """What an approval covers: all policy-expressed metadata and the mandatory flag.

    Fails closed: any policy key outside the runtime-owned grant and
    re-authorization keys (an approval gate, a business routing tier, ...) is
    authority, so an edit is covered only when every such value is equal.
    """

    offer = metadata.get(TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY)
    mandatory = metadata.get(TOOL_MANDATORY_APPROVAL_METADATA_KEY) is True or (
        type(offer) is dict
        and offer.get("grantable") is False
        and offer.get("reason") == MANDATORY_CONTROL_OFFER_REASON
    )
    expressed = {
        key: value for key, value in metadata.items() if key not in _RUNTIME_POLICY_METADATA_KEYS
    }
    return expressed, mandatory


async def admit_approved_tool_arguments(
    request: ToolApprovalRequest,
    *,
    approval: PendingToolApproval,
    request_metadata: dict[str, Any],
    requested_at: datetime | None,
    clock: Callable[[], datetime],
    session: Session,
    invocation_context: InvocationContext,
    admission: ToolInvocationAdmission,
    redactor: SecretRedactor,
    can_create_resolution_intent: bool,
) -> pending_approval_reader.ApprovedToolArguments | None:
    """Validate and re-authorize changed arguments, or refuse them.

    Runs before the claim, so a refusal leaves the approval pending and runs no
    tool. Re-authorization sees the paused round's own ``request_metadata`` (the
    run context the original decision used), never the approver's resolution
    metadata, and is marked read-only: an authorization grant is never
    consulted, used or sampled for it.
    """

    arguments = request.arguments
    if arguments is None:
        return None
    if request.tool_call_id != approval.tool_call_id:
        raise ToolApprovalArgumentsRefused("Only the gating call can be approved with changes.")
    call = next(call for call in approval.tool_calls if call.tool_call_id == approval.tool_call_id)
    if _same_arguments(arguments, call.arguments):
        # An approval form that resubmits the proposal unchanged is a plain
        # approval: no revision evidence, no edit classification and no notice.
        # The resolution request digest still binds the submitted arguments.
        return None
    if not can_create_resolution_intent:
        raise ToolApprovalArgumentsRefused(
            "This approval has resolution activity without a durable request identity, "
            "so changed arguments cannot be recorded; approve it unchanged or deny it."
        )
    if (
        call.targeted_tool_grant_id is not None
        or call.model_tool_name is not None
        or call.targeted_tool_invocation is not None
        or call.targeted_tool_rejection is not None
    ):
        raise ToolApprovalArgumentsRefused(
            "A targeted tool call cannot be approved with changed arguments."
        )
    original = approval_support.policy_result_from_pending_tool_call(call)
    if (
        pending_approval_reader.effective_tool_policy_evidence(call)
        is not ToolPolicyEvidence.AUTHORITATIVE
        or original is None
        or original.decision is not ToolPolicyDecision.REQUIRE_APPROVAL
    ):
        raise ToolApprovalArgumentsRefused(
            "Only a call that tool policy sent to approval can be approved with changes."
        )
    registered_agent = invocation_context.registered_agent
    registered_tool = registered_agent.executable_tool(call.tool_name)
    if registered_tool is None:
        raise ToolApprovalArgumentsRefused(
            f"Tool is not registered, so its arguments cannot be changed: {call.tool_name}"
        )
    if registered_tool.child_session_recovery is not None:
        raise ToolApprovalArgumentsRefused(
            "A call that runs a child session cannot be approved with changed arguments."
        )
    if contains_redacted_secret(arguments) or durable_value_contains_secret(
        arguments, redactor=redactor, path=("approved_arguments",)
    ):
        raise ToolApprovalArgumentsRefused(
            "Approved arguments must not contain a workload secret or a redaction marker."
        )
    schema_error = tool_argument_validation_error(arguments, registered_tool.schema)
    if schema_error is not None:
        raise ToolApprovalArgumentsRefused(schema_error)
    result = await admission.authorize(
        session=session,
        registered_agent=registered_agent,
        registered_environment=invocation_context.registered_environment,
        tool_call=runtime_records.ToolCallRequest(
            id=call.tool_call_id, name=call.tool_name, arguments=arguments
        ),
        request_metadata={
            **request_metadata,
            TOOL_POLICY_REAUTHORIZATION_METADATA_KEY: True,
            APPROVED_ARGUMENTS_REAUTHORIZATION_METADATA_KEY: True,
        },
        taint_labels=call.active_taint_labels,
        tool_round_id=approval.tool_round_id,
        recorded_scope=approval,
    )
    if result.decision is ToolPolicyDecision.DENY:
        raise ToolApprovalArgumentsRefused("Tool policy denied the approved arguments.")
    if result.decision is ToolPolicyDecision.REQUIRE_APPROVAL:
        # The pause stored its policy metadata through the round's redactor;
        # compare the re-authorized result through the same projection.
        pause_redactor = _redactor_for_tool_calls(
            redactor,
            registered_agent=registered_agent,
            tool_calls=[
                approval_support.tool_call_request_from_pending(pending)
                for pending in approval.tool_calls
            ],
        )
        reauthorized = copy_durable_metadata(pause_redactor.redact_json_values(result.metadata))
        if _approval_authority(reauthorized) != _approval_authority(original.metadata):
            raise ToolApprovalArgumentsRefused(
                "The approved arguments need authority this approval did not cover."
            )
        window = result.approval_expires_in_seconds
        if window is not None and (
            # Read after the awaited re-authorization, never before it.
            requested_at is None or requested_at + timedelta(seconds=window) <= clock()
        ):
            raise ToolApprovalArgumentsRefused(
                "The approval window the approved arguments require has passed."
            )
    return pending_approval_reader.ApprovedToolArguments(
        arguments=arguments,
        arguments_sha256=pending_approval_reader.approved_arguments_sha256(arguments),
    )


def arguments_revision_payload(
    *,
    session_id: str,
    tool_call_id: str,
    proposed: dict[str, Any],
    approved: dict[str, Any],
    schema: dict[str, Any] | None,
    digest_codec: PublicAuthorityAliasCodec | None,
) -> dict[str, Any]:
    """Content-free evidence that a person changed a call before approving it.

    Session and call identities are public beside this evidence, so an unkeyed
    digest of a short argument could be confirmed by guessing. Digests are
    therefore keyed with the session store's public-authority alias keyring and
    are omitted when the store has none. Changed fields name only
    schema-declared top-level properties.
    """

    def field(arguments: dict[str, Any], key: str) -> bytes | None:
        if key not in arguments:
            return None
        return canonical_durable_json_bytes(arguments[key], "arguments_revision")

    properties = (schema or {}).get("properties")
    declared: set[str] = (
        {key for key in properties if type(key) is str} if type(properties) is dict else set()
    )
    changed = sorted(
        {
            key[:_MAX_CHANGED_FIELD_CHARS]
            for key in declared
            if field(proposed, key) != field(approved, key)
        }
    )[:_MAX_CHANGED_FIELDS]
    payload: dict[str, Any] = {"changed_fields": changed}
    if digest_codec is None:
        return payload

    def digest(arguments: dict[str, Any]) -> tuple[str, str]:
        return digest_codec.evidence_digest(
            canonical_durable_json_bytes(
                [session_id, tool_call_id, arguments], "arguments_revision"
            ),
            context="arguments_revision",
        )

    key_id, proposed_digest = digest(proposed)
    _, approved_digest = digest(approved)
    return {
        **payload,
        "proposed_arguments_digest": proposed_digest,
        "approved_arguments_digest": approved_digest,
        "digest_key_id": key_id,
    }
