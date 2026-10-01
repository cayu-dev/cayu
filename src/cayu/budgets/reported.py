"""Bounded completion-time cost observations, never local budget authority."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

MAX_REPORTED_COST_RECORDS = 100
_USD_AMOUNT = re.compile(r"[0-9]{1,8}\.[0-9]{9}")


class ReportedCostObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    event_id: str
    timestamp: datetime
    request_id: str | None = Field(min_length=1, max_length=256)
    provider_name: str | None = Field(min_length=1, max_length=256)
    model: str | None = Field(min_length=1, max_length=256)
    cost: str | None = Field(max_length=18)
    currency: Literal["USD"] | None
    status: Literal["reported", "pending", "unavailable"]

    @model_validator(mode="after")
    def validate_observation(self) -> ReportedCostObservation:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("Reported-cost timestamps must be timezone-aware.")
        if self.status == "reported":
            if self.currency != "USD" or self.cost is None or not _USD_AMOUNT.fullmatch(self.cost):
                raise ValueError("Reported cost requires an exact nonnegative USD amount.")
        elif self.cost is not None:
            raise ValueError("Unknown reported cost must not contain an amount.")
        if self.status == "pending" and self.currency != "USD":
            raise ValueError("Pending cost requires its reported currency.")
        return self


class ReportedCostPage(BaseModel):
    """Latest retained observations, not a total or a current financial readback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[ReportedCostObservation, ...] = Field(
        default=(), max_length=MAX_REPORTED_COST_RECORDS
    )
    truncated: StrictBool = False


def reported_cost_observation(
    *, session_id: str, event_id: str, timestamp: datetime, values: Mapping[str, object]
) -> ReportedCostObservation:
    def text(name: str) -> str | None:
        value = values.get(name)
        return value if type(value) is str and 0 < len(value) <= 256 else None

    amount = values.get("cost")
    currency = "USD" if values.get("currency") == "USD" else None
    status: Literal["reported", "pending", "unavailable"] = "unavailable"
    if currency is not None:
        if values.get("status") == "reported" and type(amount) is str:
            if _USD_AMOUNT.fullmatch(amount):
                status = "reported"
        elif values.get("status") == "pending" and amount is None:
            status = "pending"
    return ReportedCostObservation(
        session_id=session_id,
        event_id=event_id,
        timestamp=timestamp,
        request_id=text("request_id"),
        provider_name=text("provider_name"),
        model=text("model"),
        cost=amount if status == "reported" and type(amount) is str else None,
        currency=currency,
        status=status,
    )


class ReportedCostCollector:
    def __init__(self) -> None:
        self.records: list[ReportedCostObservation] = []
        self.truncated = False

    def add(self, *, session_id: str, event_id: str, timestamp: datetime, payload: dict) -> None:
        usage = payload.get("usage")
        if not isinstance(usage, dict) or "cost_status" not in usage:
            return
        self.records.append(
            reported_cost_observation(
                session_id=session_id,
                event_id=event_id,
                timestamp=timestamp,
                values={
                    "request_id": payload.get("id"),
                    "provider_name": payload.get("provider_name"),
                    "model": payload.get("model"),
                    "cost": usage.get("cost"),
                    "currency": usage.get("cost_currency"),
                    "status": usage.get("cost_status"),
                },
            )
        )
        self.records.sort(
            key=lambda row: (row.timestamp, row.session_id, row.event_id), reverse=True
        )
        if len(self.records) > MAX_REPORTED_COST_RECORDS:
            self.records.pop()
            self.truncated = True

    def page(self) -> ReportedCostPage:
        return ReportedCostPage(records=tuple(self.records), truncated=self.truncated)
