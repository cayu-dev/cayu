"""Shared terminal-proof validation and bounded retirement commitment."""

from hashlib import sha256

from cayu.sessions.external_waits import (
    EXTERNAL_WAIT_PAGE_LIMIT,
    ExternalWaitConflict,
    ExternalWaitRecord,
    ExternalWaitRetirement,
    ExternalWaitRetirementRequest,
    ExternalWaitScope,
    external_wait_digest,
)


def require_prune_limit(limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= EXTERNAL_WAIT_PAGE_LIMIT:
        raise ValueError("External wait page limit is invalid.")


class RetirementAccumulator:
    def __init__(self, request: ExternalWaitRetirementRequest):
        self.request = request
        self.count = 0
        self.digest = sha256()
        self.last_key = ""

    def add(self, record: ExternalWaitRecord) -> None:
        key = record.correlation.request.correlation_key
        if (
            record.correlation.request.scope != self.request.scope
            or record.correlation.limits != self.request.limits
            or key <= self.last_key
            or self.count >= self.request.limits.correlations
        ):
            raise ExternalWaitConflict("External retirement snapshot conflicts.")
        if (
            record.outcome is None
            or record.pending_handoff
            or (record.timer is not None and not record.timer_published)
        ):
            raise ExternalWaitConflict("External scope retains unresolved responsibility.")
        self.digest.update(bytes.fromhex(external_wait_digest(record)))
        self.count += 1
        self.last_key = key

    def finish(self, now_ms: int) -> ExternalWaitRetirement:
        return ExternalWaitRetirement(
            request=self.request,
            retired_at_ms=now_ms,
            correlation_count=self.count,
            records_sha256=self.digest.hexdigest(),
        )


def read_retirement(
    value: str | None, request: ExternalWaitRetirementRequest
) -> ExternalWaitRetirement:
    if value is None:
        raise ExternalWaitConflict("External retirement evidence is unavailable.")
    receipt = ExternalWaitRetirement.model_validate_json(value)
    if receipt.request != request:
        raise ExternalWaitConflict("External retirement operation conflicts.")
    return receipt


def read_scope_retirement(row, scope: ExternalWaitScope) -> ExternalWaitRetirement | None:
    """Cross-check the permanent tombstone against its indexed namespace/limits."""
    if row is None:
        return None
    retired, value, limits_json = row
    if not retired:
        if value is not None:
            raise ExternalWaitConflict("External retirement state conflicts.")
        return None
    if retired != 1 or value is None:
        raise ExternalWaitConflict("External retirement evidence is unavailable.")
    receipt = ExternalWaitRetirement.model_validate_json(value)
    if receipt.request.scope != scope or receipt.request.limits.model_dump_json() != limits_json:
        raise ExternalWaitConflict("External retirement indexed identity conflicts.")
    return receipt
