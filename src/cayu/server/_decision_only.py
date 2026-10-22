"""Keep read-and-decide access from overriding a resumed invocation."""

from __future__ import annotations

from starlette.requests import HTTPConnection, Request

from cayu.approvals.revisions import APPROVAL_REVISION_METADATA_KEY

_EXECUTION_OVERRIDES = frozenset(
    {
        "arguments",
        "approval_revision",
        "max_steps",
        "limits",
        "budget_limits",
        "retry_policy",
        "structured_output",
        "thinking",
    }
)


async def has_decision_overrides(request: HTTPConnection) -> bool:
    """Inspect decision bodies after the caller has authorized the route."""

    if not isinstance(request, Request) or request.method != "POST":
        return False
    template = getattr(request.scope.get("route"), "path", "")
    if not template.endswith(
        ("/tool-approvals/resolve", "/user-input/resolve", "/inbox/{item_id}/settle")
    ):
        return False
    try:
        body = await request.json()
    except ValueError:
        return False  # The route's body validation reports malformed JSON.
    if not isinstance(body, dict):
        return False
    if any(body.get(field) is not None for field in _EXECUTION_OVERRIDES):
        return True
    metadata = body.get("metadata")
    return isinstance(metadata, dict) and APPROVAL_REVISION_METADATA_KEY in metadata
