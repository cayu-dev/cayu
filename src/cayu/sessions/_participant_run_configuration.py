"""Restore inert participant run controls from their durable creation evidence."""

from typing import TYPE_CHECKING

from cayu._validation import canonical_durable_json_bytes
from cayu.execution_profiles import ExecutionProfileIdentity
from cayu.runtime._application_run_defaults import _RunConfigurationRequest
from cayu.runtime._resume_configuration import (
    _configuration_matches_profile,
    invocation_configuration,
)
from cayu.sessions._invocation_configuration import InvocationConfiguration
from cayu.sessions.requests import RunRequest

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.runtime._session_preparation import _PreparedInitialRun
    from cayu.sessions.records import Session


def participant_configuration_json(
    app: "CayuApp", prepared: "_PreparedInitialRun", session: "Session"
) -> str:
    record = invocation_configuration(
        prepared.request,
        session_id=session.id,
        session_instance_id=session.instance_id,
        profile=prepared.execution_profile,
        retry_policy=app._effective_retry_policy(prepared.request.retry_policy),
    )
    return canonical_durable_json_bytes(
        record.model_dump(mode="json"), "participant run configuration"
    ).decode("utf-8")


def inherit_participant_configuration(
    request: _RunConfigurationRequest,
    configuration_json: str,
    *,
    session_instance_id: str | None,
    profile: ExecutionProfileIdentity | None,
) -> _RunConfigurationRequest:
    """Retain omitted controls; ordinary preparation still checks the full profile."""
    if type(request) is not RunRequest:
        raise TypeError("Participant creation controls require a RunRequest.")
    record = InvocationConfiguration.model_validate_json(configuration_json)
    if (
        profile is None
        or record.session_id != request.session_id
        or record.session_instance_id != session_instance_id
        or not _configuration_matches_profile(record, profile)
    ):
        raise ValueError("Participant run configuration conflicts with its admitted profile.")
    fields = request.model_fields_set
    inherited = request.model_copy(
        update={
            name: getattr(record, name)
            for name in (
                "structured_output",
                "max_steps",
                "limits",
                "retry_policy",
                "tool_completion",
            )
            if name not in fields or (name == "retry_policy" and request.retry_policy is None)
        }
    )
    object.__setattr__(inherited, "__pydantic_fields_set__", set(fields))
    return inherited
