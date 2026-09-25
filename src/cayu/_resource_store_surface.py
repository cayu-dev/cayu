"""Explicit model-facing native store surface; all other methods are operator-only."""

from functools import wraps
from inspect import iscoroutinefunction

_SESSION_OPERATIONS = frozenset(
    {
        "create",
        "load",
        "list_sessions",
        "update_labels",
        "update_metadata",
        "delete_session",
        "summarize_events",
        "summarize_outcome",
        "load_latest_transcript_text",
        "query_transcript",
        "load_events",
        "load_transcript",
        "load_transcript_snapshot",
        "load_transcript_window",
        "load_checkpoint",
        "load_session_operation",
        "query_events",
        "query_events_bounded",
        "read_usage_accounting",
        "read_cost_accounting",
        "event_exists",
        "enqueue_session_message",
        "inspect_session_messages",
        "apply_session_message_action",
        "snapshot_session_message_source",
        "append_peer_content",
    }
)
_TASK_OPERATIONS = frozenset(
    {
        "create_task",
        "create_running_task",
        "create_task_graph",
        "create_task_group",
        "load_task",
        "list_tasks",
        "cancel_task",
        "pause_task",
        "resume_task",
        "block_task",
        "mark_task_needs_attention",
        "load_task_graph",
        "load_task_group",
        "list_task_graph_events",
        "list_task_group_events",
    }
)
_ARTIFACT_OPERATIONS = frozenset({"put_bytes", "read_bytes", "read_range", "list", "delete"})


def model_store_surface(kind):
    """Fail closed on native methods outside the implemented resource boundary.

    Raw stores outside model tool execution remain explicit trusted operator
    APIs. This is not a sandbox for arbitrary Python tools or database clients.
    """
    supported = {
        "sessions": _SESSION_OPERATIONS,
        "tasks": _TASK_OPERATIONS,
        "artifacts": _ARTIFACT_OPERATIONS,
    }[kind]

    def decorate(cls):
        def guarded(operation):
            @wraps(operation)
            async def call(self, *args, **kwargs):
                from cayu.resource_access import _active, _model_data_access

                try:
                    if _active.get() is not None and _model_data_access.get():
                        raise NotImplementedError(
                            f"{operation.__name__} requires the trusted operator interface."
                        )
                    return await operation(self, *args, **kwargs)
                finally:
                    del args, kwargs

            return call

        for name in dir(cls):
            operation = getattr(cls, name)
            if (
                not name.startswith("_")
                and name not in supported
                and iscoroutinefunction(operation)
            ):
                setattr(cls, name, guarded(operation))
        return cls

    return decorate
