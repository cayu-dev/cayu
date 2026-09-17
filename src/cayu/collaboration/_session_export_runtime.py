"""Derive requester provenance from the existing runtime-owned tool boundary."""

from dataclasses import dataclass

from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.exports import SessionExportDenied, SessionExportRuntimeOrigin
from cayu.sessions.base import (
    SessionStore,
    _current_session_interaction_id,
    _current_session_run_epoch,
)
from cayu.tools._invocation_lifetime import has_live_tool_invocation
from cayu.tools.base import ToolContext, _runtime_tool_invocation_authority
from cayu.vaults.redaction import SecretRedactor


@dataclass(frozen=True)
class RuntimeExportAdmission:
    context: ToolContext
    origin: SessionExportRuntimeOrigin

    def require_live(self) -> None:
        if (
            _runtime_tool_invocation_authority(self.context) is None
            or self.context.session_id != self.origin.session_id
            or _current_session_run_epoch(self.origin.session_id) != self.origin.run_epoch
            or _current_session_interaction_id(self.origin.session_id) != self.origin.interaction_id
        ):
            raise SessionExportDenied()

    async def validate(self, store: SessionStore) -> None:
        self.require_live()
        current = await store.load(self.origin.session_id)
        if (
            current is None
            or current.id != self.origin.session_id
            or current.instance_id != self.origin.session_instance_id
            or current.run_epoch != self.origin.run_epoch
            or current.invocation != self.origin.invocation
        ):
            raise SessionExportDenied()
        self.require_live()


def capture_runtime_export(
    context: ToolContext, redactor: SecretRedactor
) -> RuntimeExportAdmission:
    # Test provenance before serialization or formatting; hostile public values
    # cannot borrow authority merely by copying all visible ToolContext fields.
    if type(context) is not ToolContext or type(context.session_id) is not str:
        raise SessionExportDenied()
    # This gates a new request, not settlement of work already handed to the
    # export owner. Retained provenance remains usable for that owned work.
    if not has_live_tool_invocation(context):
        raise SessionExportDenied()
    authority = _runtime_tool_invocation_authority(context)
    interaction_id = _current_session_interaction_id(context.session_id)
    if (
        authority is None
        or _current_session_run_epoch(context.session_id) != authority.parent_run_epoch
        or interaction_id is None
    ):
        raise SessionExportDenied()
    lineage = authority.current_session_lineage
    invocation = lineage["invocation"]
    origin_identity = invocation["origin"]
    origin = prepare_contract(
        SessionExportRuntimeOrigin,
        {
            "session_id": lineage.get("session_id"),
            "session_instance_id": lineage.get("session_instance_id"),
            "run_epoch": authority.parent_run_epoch,
            "invocation_schema_version": invocation["schema_version"],
            "invocation_trust": origin_identity["trust"],
            "invocation_subject": origin_identity["subject"],
            "invocation_tenant": origin_identity["tenant"],
            "root_invocation_id": invocation["root_invocation_id"],
            "root_session_id": invocation["root_session_id"],
            "invocation_source": invocation["source"],
            "interaction_id": interaction_id,
            "model_step_id": authority.model_step_id,
            "model_attempt_id": authority.model_attempt_id,
            "tool_round_id": authority.tool_round_id,
            "tool_call_id": authority.tool_call_id,
            "tool_name": authority.tool_name,
            "idempotency_key": authority.idempotency_key,
            "effective_arguments_sha256": authority.effective_arguments_sha256,
            "execution_profile_fingerprint": authority.execution_profile_fingerprint,
        },
        redactor=redactor,
    )
    result = RuntimeExportAdmission(context, origin)
    result.require_live()
    return result
