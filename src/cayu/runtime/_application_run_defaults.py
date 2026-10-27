"""Resolve application run controls without changing explicit-field authority."""

from __future__ import annotations

from typing import TypeVar

from cayu.configuration import CayuConfig, CayuConfigSource
from cayu.providers.retry_policy import copy_retry_policy
from cayu.sessions.invocation import SessionExecutionSource
from cayu.sessions.records import Session
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.tasks.dispatch import DispatchRequest

_RunConfigurationRequest = TypeVar(
    "_RunConfigurationRequest", RunRequest, ResumeRequest, DispatchRequest
)


def with_resolved_retry_policy(request: RunRequest, config: CayuConfig) -> RunRequest:
    """Fill an omitted retry policy from ``config`` without marking it explicit."""

    if request.retry_policy is not None:
        return request
    fields_set = set(request.model_fields_set)
    resolved = request.model_copy(
        update={"retry_policy": copy_retry_policy(config.run.retry_policy)}
    )
    object.__setattr__(resolved, "__pydantic_fields_set__", fields_set)
    return resolved


def with_application_run_defaults(
    request: _RunConfigurationRequest,
    config: CayuConfig,
    max_steps_source: CayuConfigSource,
    session: Session | None = None,
    *,
    freeze_retry_policy: bool = False,
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
    if freeze_retry_policy and type(request) is RunRequest and request.retry_policy is None:
        # The application's config can change between a new run's preparation and
        # execution (Cayu Cloud's managed settings), so resolve its retry policy once
        # here. Resumes and dispatches keep None and inherit as before.
        updates["retry_policy"] = copy_retry_policy(config.run.retry_policy)
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
