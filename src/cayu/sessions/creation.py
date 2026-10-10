"""Prepared session creation authority and runtime metadata rules."""

from __future__ import annotations

import hashlib
from typing import Any

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_metadata,
    copy_session_metadata,
)
from cayu.deadlines import (
    EXECUTION_DEADLINE_METADATA_KEY,
    ExecutionDeadline,
    current_execution_deadline,
    effective_deadline,
)
from cayu.execution_profiles import (
    ExecutionProfileComponentClass,
    direct_tool_capability_ceiling_component,
)
from cayu.sessions import requests as session_request_contracts
from cayu.sessions._execution_profile_checkpoint import (
    EXECUTION_PROFILE_METADATA_KEY,
    execution_profile_session_metadata,
)
from cayu.sessions.forks import FORK_EXECUTION_PROFILE_METADATA_KEY
from cayu.sessions.records import RUNTIME_BUILD_PROVENANCE_METADATA_KEY, Session, SessionIdentity
from cayu.sessions.requests import RunRequest, copy_run_request
from cayu.tools.exposure import (
    TOOL_CAPABILITY_CEILING_METADATA_KEY,
    ToolCapabilityCeiling,
    session_metadata_with_tool_capability_ceiling,
)


def _prepared_work_attempt_creation_sha256(request: RunRequest, identity: SessionIdentity) -> str:
    return hashlib.sha256(
        canonical_durable_json_bytes(
            {
                "request": request.model_dump(mode="json", warnings=False),
                "private_authority": session_request_contracts._run_request_invocation_lifecycle_authority_sha256(
                    request
                ),
                "identity": identity.model_dump(mode="json", warnings=False),
            },
            "prepared_work_attempt_creation",
        )
    ).hexdigest()


def run_request_with_prepared_work_attempt_creation(
    request: RunRequest, *, identity: SessionIdentity, admission: object
) -> RunRequest:
    """Internal runtime entrance; raw request equality cannot confer this authority."""
    from cayu.tasks.admission import (
        WorkAttemptAdmissionState,
        require_work_attempt_admission_result,
    )

    prepared = require_work_attempt_admission_result(
        admission, operation_name="Prepared session creation authority"
    )
    copied = copy_run_request(request)
    if (
        prepared.state is not WorkAttemptAdmissionState.PREPARING
        or prepared.kind != "initial"
        or prepared.execution_entry is not None
        or prepared.run_semantics is None
        or copied.session_id != prepared.session_id
        or copied.task_id != prepared.task_id
        or session_request_contracts._authenticated_session_instance_id_for_run_request(
            copied, session_id=prepared.session_id
        )
        != prepared.session_invocation.session_instance_id
        or copied.execution_deadline.model_dump() != prepared.run_semantics.deadline.model_dump()
        or identity.execution_profile is None
        or identity.execution_profile.fingerprint != prepared.source_execution_profile_fingerprint
    ):
        raise ValueError("Prepared session creation conflicts with admission authority.")
    copied._runtime_work_attempt_creation = session_request_contracts._PreparedWorkAttemptCreation(
        _prepared_work_attempt_creation_sha256(copied, identity),
        session_request_contracts._PREPARED_WORK_ATTEMPT_CREATION_TOKEN,
    )
    return copied


def session_metadata_for_creation(
    metadata: dict[str, Any],
    *,
    identity: SessionIdentity,
    tool_capability_ceiling: ToolCapabilityCeiling | None = None,
    execution_deadline: ExecutionDeadline | None = None,
    parent_session: Session | None = None,
    prepared_request: RunRequest | None = None,
) -> dict[str, Any]:
    """Combine caller metadata with runtime-owned creation authority."""

    copied = copy_durable_metadata(metadata)
    if EXECUTION_DEADLINE_METADATA_KEY in copied:
        raise ValueError("Session metadata contains runtime-owned deadline authority.")
    boundary = effective_deadline(
        execution_deadline or ExecutionDeadline(),
        current_execution_deadline(),
        parent_session.execution_deadline if parent_session is not None else ExecutionDeadline(),
    )
    prepared_creation = (
        None if prepared_request is None else prepared_request._runtime_work_attempt_creation
    )
    if prepared_creation is not None:
        if (
            type(prepared_creation) is not session_request_contracts._PreparedWorkAttemptCreation
            or prepared_creation.token
            is not session_request_contracts._PREPARED_WORK_ATTEMPT_CREATION_TOKEN
            or prepared_request is None
            or prepared_creation.request_sha256
            != _prepared_work_attempt_creation_sha256(prepared_request, identity)
            or prepared_request.metadata != metadata
            or prepared_request.tool_capability_ceiling != tool_capability_ceiling
            or prepared_request.execution_deadline.model_dump() != boundary.model_dump()
        ):
            raise ValueError("Prepared session creation authority changed.")
    else:
        boundary.require_admission("session_creation")
    if boundary.expires_at is not None:
        copied[EXECUTION_DEADLINE_METADATA_KEY] = boundary.model_dump(mode="json")
    if EXECUTION_PROFILE_METADATA_KEY in copied:
        raise ValueError("Session metadata contains runtime-owned execution-profile authority.")
    if RUNTIME_BUILD_PROVENANCE_METADATA_KEY in copied:
        raise ValueError("Session metadata contains runtime-owned build-provenance authority.")
    if FORK_EXECUTION_PROFILE_METADATA_KEY in copied:
        raise ValueError("Session metadata contains runtime-owned fork-profile authority.")
    if TOOL_CAPABILITY_CEILING_METADATA_KEY in copied:
        raise ValueError(
            "Session metadata contains runtime-owned tool-capability-ceiling authority."
        )
    if tool_capability_ceiling is not None and identity.execution_profile is None:
        raise ValueError("A tool capability ceiling requires a durable session execution profile.")
    if identity.execution_profile is not None:
        if tool_capability_ceiling is not None and identity.execution_profile.component(
            ExecutionProfileComponentClass.TOOL_VIEW_GRANTS
        ) != direct_tool_capability_ceiling_component(tool_capability_ceiling.tool_names):
            raise ValueError(
                "Session execution profile conflicts with its tool capability ceiling."
            )
        copied[EXECUTION_PROFILE_METADATA_KEY] = execution_profile_session_metadata(
            identity.execution_profile
        )
    copied[RUNTIME_BUILD_PROVENANCE_METADATA_KEY] = identity.runtime_build_provenance.model_dump(
        mode="json"
    )
    if tool_capability_ceiling is not None:
        copied = session_metadata_with_tool_capability_ceiling(
            copied,
            tool_capability_ceiling,
        )
    return copy_session_metadata(copied)
