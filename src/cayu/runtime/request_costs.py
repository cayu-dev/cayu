"""Content-free request cost summaries for the protected control plane.

The server's request timing middleware records per-request wall time, CPU time,
and response size under route templates and keyed client hashes. These models
are the portable summary of that bounded in-memory record, shared by the
``GET /api/diagnostics/requests`` route, ``cayu diagnostics requests``, and the
diagnostic support bundle. They never carry raw paths, query values, cookies,
credentials, or client addresses.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

MAX_REQUEST_COST_ROUTES = 200
MAX_REQUEST_COST_WINDOW_SECONDS = 24 * 60 * 60
DEFAULT_REQUEST_COST_WINDOW_SECONDS = 5 * 60
UNMATCHED_ROUTE_TEMPLATE = "(unmatched)"


class _RequestCostModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RequestCostRoute(_RequestCostModel):
    """Aggregated cost of one method and route template inside the window."""

    method: str = Field(max_length=16)
    route: str = Field(max_length=512)
    requests: StrictInt = Field(ge=0)
    requests_per_minute: float = Field(ge=0)
    streaming_requests: StrictInt = Field(ge=0)
    status_2xx: StrictInt = Field(ge=0)
    status_3xx: StrictInt = Field(ge=0)
    status_4xx: StrictInt = Field(ge=0)
    status_5xx: StrictInt = Field(ge=0)
    wall_ms_p50: float | None = Field(default=None, ge=0)
    wall_ms_p95: float | None = Field(default=None, ge=0)
    wall_ms_max: float | None = Field(default=None, ge=0)
    cpu_seconds: float = Field(ge=0)
    cpu_seconds_per_minute: float = Field(ge=0)
    vcpu_share: float = Field(ge=0)
    response_bytes: StrictInt = Field(ge=0)
    clients: StrictInt = Field(ge=0)


class RequestCostSummary(_RequestCostModel):
    """Per-route request rate, latency, and CPU over a recent window.

    ``window_seconds`` is the span actually covered by retained records: it is
    shorter than ``requested_window_seconds`` when the process started more
    recently or when the bounded buffer has already evicted records from the
    requested span (``truncated``). Rates and the vCPU share use the covered
    span. ``vcpu_share`` is request CPU seconds per second divided by ``vcpu``.
    ``routes`` is ordered by CPU seconds, highest first; latency percentiles
    exclude streaming responses, whose duration is the stream lifetime.
    """

    observed_at: datetime
    enabled: StrictBool
    requested_window_seconds: float = Field(gt=0, le=MAX_REQUEST_COST_WINDOW_SECONDS)
    window_seconds: float = Field(ge=0, le=MAX_REQUEST_COST_WINDOW_SECONDS)
    truncated: StrictBool
    buffer_capacity: StrictInt = Field(ge=0)
    vcpu: float = Field(gt=0)
    requests: StrictInt = Field(ge=0)
    requests_per_minute: float = Field(ge=0)
    cpu_seconds: float = Field(ge=0)
    cpu_seconds_per_minute: float = Field(ge=0)
    vcpu_share: float = Field(ge=0)
    route_count: StrictInt = Field(ge=0)
    routes: tuple[RequestCostRoute, ...] = Field(max_length=MAX_REQUEST_COST_ROUTES)

    @field_validator("observed_at")
    @classmethod
    def normalize_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware.")
        return value.astimezone(UTC)


def disabled_request_cost_summary(
    *,
    requested_window_seconds: float,
    vcpu: float,
    observed_at: datetime | None = None,
) -> RequestCostSummary:
    """Return the explicit empty summary for a server without request timing."""

    return RequestCostSummary(
        observed_at=observed_at or datetime.now(UTC),
        enabled=False,
        requested_window_seconds=requested_window_seconds,
        window_seconds=0.0,
        truncated=False,
        buffer_capacity=0,
        vcpu=vcpu,
        requests=0,
        requests_per_minute=0.0,
        cpu_seconds=0.0,
        cpu_seconds_per_minute=0.0,
        vcpu_share=0.0,
        route_count=0,
        routes=(),
    )


__all__ = [
    "DEFAULT_REQUEST_COST_WINDOW_SECONDS",
    "MAX_REQUEST_COST_ROUTES",
    "MAX_REQUEST_COST_WINDOW_SECONDS",
    "UNMATCHED_ROUTE_TEMPLATE",
    "RequestCostRoute",
    "RequestCostSummary",
    "disabled_request_cost_summary",
]
