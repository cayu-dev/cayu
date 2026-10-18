"""Resolve application run controls without changing explicit-field authority."""

from __future__ import annotations

from typing import TypeVar

from cayu.configuration import CayuConfig, CayuConfigSource
from cayu.sessions.invocation import SessionExecutionSource
from cayu.sessions.records import Session
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.tasks.dispatch import DispatchRequest

_RunConfigurationRequest = TypeVar(
    "_RunConfigurationRequest", RunRequest, ResumeRequest, DispatchRequest
)


def with_application_run_defaults(
    request: _RunConfigurationRequest,
    config: CayuConfig,
    max_steps_source: CayuConfigSource,
    session: Session | None = None,
) -> _RunConfigurationRequest:
    """Resolve omitted run controls once before admission or dispatch."""

    updates: dict[str, object] = {}
    fields_set = request.model_fields_set
    if "max_steps" not in fields_set:
        updates["max_steps"] = (
            None
            if max_steps_source == "framework"
            and (
                (type(request) is RunRequest and request.task_id is not None)
                or (
                    session is not None and session.invocation.source is SessionExecutionSource.TASK
                )
            )
            else config.run.max_steps
        )
    if "limits" not in fields_set:
        updates["limits"] = config.run.copy_limits()
    if not updates:
        return request
    # Loop policies and other request collaborators can hold runtime
    # objects that are intentionally not deepcopy-safe. The resolved
    # configuration values above are already detached copies. Preserve the
    # caller's explicit-field set so durable continuation can distinguish
    # an inherited value from an explicit override.
    resolved = request.model_copy(update=updates)
    object.__setattr__(resolved, "__pydantic_fields_set__", set(fields_set))
    return resolved
