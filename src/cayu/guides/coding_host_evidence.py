"""Evidence policy for the installed coding-host example.

Not a Runtime recovery engine. Consume only inspector-owned original invocation
records after native release; unknown/custom effects remain fenced.
"""

from cayu import EventType
from cayu.providers.base import ModelCompletion, ModelFinishReason


class MaintenanceReconciliationUnavailable(ValueError):
    def __init__(self):
        super().__init__("Exact maintenance cancellation evidence is unavailable or conflicting.")


def _completed_workspace_tool_result(name, kind, result):
    """Recognize maintained file-tool success, not arbitrary tool completion.

    Callers must first authenticate the original invocation/profile and release.
    This projection is not accepted from the HTTP caller as recovery authority.
    """
    if kind != "tool.call.completed" or type(result) is not dict:
        return False
    if result.get("is_error") is not False:
        return False
    value = result.get("structured")
    if type(value) is not dict or "error" in value:
        return False

    def text_fields(*fields):
        return all(type(value.get(field)) is str and bool(value[field]) for field in fields)

    def counts(*fields):
        return all(type(value.get(field)) is int and value[field] >= 0 for field in fields)

    if name == "read_file":
        return (
            value.get("source") == "workspace"
            and value.get("encoding") == "utf-8"
            and text_fields("path")
            and counts("bytes", "total_bytes", "offset")
            and type(value.get("truncated")) is bool
        )
    if name == "write_file":
        return (
            text_fields("path", "revision", "sha256")
            and counts("bytes")
            and value.get("encoding") == "utf-8"
            and value.get("mode") in {"create", "overwrite"}
        )
    if name == "edit_file":
        return (
            text_fields(
                "path", "before_revision", "after_revision", "before_sha256", "after_sha256"
            )
            and counts("before_bytes", "after_bytes", "edit_count", "replacement_count")
            and value["edit_count"] > 0
            and value["replacement_count"] > 0
        )
    if name == "apply_patch":
        return (
            text_fields("patch_id", "behavior_profile_id")
            and type(value.get("version")) is int
            and value["version"] == 2
            and value.get("outcome") == "applied"
            and "failure_category" in value
            and value["failure_category"] is None
            and value.get("requires_fresh_read") is False
            and counts("operation_count")
            and value["operation_count"] > 0
        )
    if name == "git_changes":
        # Maintained Git inspection awaits its read-only runner operations with
        # optional Git locks disabled. Capture truncation is not an unknown effect.
        return (
            value.get("mode") in {"status", "summary", "diff"}
            and value.get("scope") in {"all", "staged", "unstaged"}
            and type(value.get("changes")) is list
            and counts("returned", "offset", "limit")
            and value["returned"] == len(value["changes"])
            and type(value.get("truncated")) is bool
        )
    if name == "delete_file":
        return text_fields("path", "deleted_revision", "deleted_sha256") and counts("deleted_bytes")
    if name == "list_files":
        # A completed bounded scan may not know the full directory cardinality.
        # Explicit truncation is settled read evidence, not an unknown effect;
        # a missing field or an invalid count is still rejected.
        total = value.get("total_files")
        return (
            text_fields("pattern")
            and type(value.get("files")) is list
            and counts("offset")
            and type(value.get("truncated")) is bool
            and "total_files" in value
            and (
                (type(total) is int and total >= 0)
                or (total is None and value["truncated"] is True)
            )
        )
    if name == "search_text":
        return (
            text_fields("pattern", "path")
            and value.get("mode") in {"files", "content", "count"}
            and type(value.get("matches")) is list
            and counts("returned", "offset", "limit", "stdout_bytes")
            and value["returned"] == len(value["matches"])
            and type(value.get("truncated")) is bool
        )
    return False


def _require_serial_check_quiescence(events, *, tool_call_ordinals=None):
    # Public event identities can be redacted. Do not compare redacted aliases as
    # if they were private dispatch IDs. Inspector-owned ordinals correlate a
    # batch's starts and terminals; each result still needs positive settlement.
    # Without those ordinals only a single outstanding call can be checked.
    model = None
    pending = {}
    seen = set()
    if tool_call_ordinals is not None and (
        type(tool_call_ordinals) is not tuple or len(tool_call_ordinals) != len(events)
    ):
        raise MaintenanceReconciliationUnavailable()
    model_count = 0
    check_count = 0
    for index, event in enumerate(events):
        if type(event.type) is not EventType:
            raise MaintenanceReconciliationUnavailable()
        kind, payload = event.type.value, event.payload
        if kind.startswith("model.") and kind not in {
            "model.started",
            "model.completed",
            "model.text.delta",
            "model.thinking.delta",
            "model.citation",
            "model.http_cleanup",
        }:
            raise MaintenanceReconciliationUnavailable()
        if kind.startswith("tool.") and kind not in {
            "tool.call.started",
            "tool.call.completed",
            "tool.call.failed",
            "tool.exposure.recorded",
            "tool.effect.outcome_unknown",
        }:
            raise MaintenanceReconciliationUnavailable()
        if kind.startswith(("subagent.", "provider.operation.")) or kind == "session.resumed":
            raise MaintenanceReconciliationUnavailable()
        if kind == "model.started":
            if model is not None or pending:
                raise MaintenanceReconciliationUnavailable()
            if any(type(payload.get(key)) is not int for key in ("step", "attempt")):
                raise MaintenanceReconciliationUnavailable()
            if (
                payload["step"] < 1
                or payload["attempt"] < 1
                or type(payload.get("model")) is not str
                or not payload["model"]
            ):
                raise MaintenanceReconciliationUnavailable()
            model = (payload["step"], payload["attempt"], payload.get("model"))
        elif kind == "model.completed":
            if any(type(payload.get(key)) is not int for key in ("step", "attempt")):
                raise MaintenanceReconciliationUnavailable()
            if model is None or model != (
                payload.get("step"),
                payload.get("attempt"),
                payload.get("model"),
            ):
                raise MaintenanceReconciliationUnavailable()
            try:
                completion = ModelCompletion.model_validate(payload.get("completion"))
            except ValueError:
                raise MaintenanceReconciliationUnavailable() from None
            if (
                payload.get("status") is not None and payload["status"] != completion.status
            ) or not (
                completion.status == "completed"
                or (
                    completion.status is None
                    and completion.finish_reason
                    in {ModelFinishReason.STOP, ModelFinishReason.TOOL_CALLS}
                )
            ):
                raise MaintenanceReconciliationUnavailable()
            model = None
            model_count += 1
        elif kind == "tool.call.started":
            if (
                model is not None
                or (tool_call_ordinals is None and pending)
                or event.tool_name
                not in {
                    "run_check",
                    "read_file",
                    "write_file",
                    "edit_file",
                    "apply_patch",
                    "git_changes",
                    "delete_file",
                    "list_files",
                    "search_text",
                }
            ):
                raise MaintenanceReconciliationUnavailable()
            ordinal = 0 if tool_call_ordinals is None else tool_call_ordinals[index]
            if (
                type(ordinal) is not int
                or not 0 <= ordinal < len(events)
                or (tool_call_ordinals is not None and ordinal in seen)
            ):
                raise MaintenanceReconciliationUnavailable()
            seen.add(ordinal)
            pending[ordinal] = event.tool_name
        elif kind in {"tool.call.completed", "tool.call.failed"}:
            ordinal = 0 if tool_call_ordinals is None else tool_call_ordinals[index]
            if type(ordinal) is not int:
                raise MaintenanceReconciliationUnavailable()
            tool = pending.get(ordinal)
            if tool is None or event.tool_name != tool:
                raise MaintenanceReconciliationUnavailable()
            result = payload.get("result")
            if tool != "run_check":
                if not _completed_workspace_tool_result(tool, kind, result):
                    raise MaintenanceReconciliationUnavailable()
                del pending[ordinal]
                continue
            structured = result.get("structured") if type(result) is dict else None
            if (
                type(structured) is not dict
                or structured.get("workspace_mutation_settlement") != "complete"
                or structured.get("cleanup_uncertain") is not False
                or structured.get("status") not in {"passed", "failed", "timed_out"}
            ):
                raise MaintenanceReconciliationUnavailable()
            del pending[ordinal]
            check_count += 1
    if model is not None or pending or not model_count or not check_count:
        raise MaintenanceReconciliationUnavailable()
