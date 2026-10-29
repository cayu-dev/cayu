"""Bounded inventory activity from current retained session records."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import require_durable_clean_nonblank
from cayu.budgets.aggregates import AggregateCount

ENVIRONMENT_ACTIVITY_BATCH_LIMIT = 500


class EnvironmentActivityQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    environment_names: tuple[str, ...] = Field(
        min_length=1, max_length=ENVIRONMENT_ACTIVITY_BATCH_LIMIT
    )

    @field_validator("environment_names")
    @classmethod
    def validate_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for name in value:
            require_durable_clean_nonblank(name, "environment name")
        if len(set(value)) != len(value):
            raise ValueError("Environment names must be distinct.")
        return value


def copy_environment_activity_query(query: EnvironmentActivityQuery) -> EnvironmentActivityQuery:
    if type(query) is not EnvironmentActivityQuery:
        raise TypeError("Environment activity requires a typed query.")
    return EnvironmentActivityQuery(environment_names=query.environment_names)


class EnvironmentSessionActivity(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    environment_name: str
    session_count: AggregateCount = Field(ge=0)
    last_activity_at: datetime | None

    @field_validator("environment_name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "environment name")

    @field_validator("last_activity_at")
    @classmethod
    def normalize_activity(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @model_validator(mode="after")
    def validate_activity(self) -> EnvironmentSessionActivity:
        if (self.session_count == 0) is not (self.last_activity_at is None):
            raise ValueError("Environment activity timestamp must match its session population.")
        return self


class EnvironmentActivitySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    as_of: datetime
    environments: tuple[EnvironmentSessionActivity, ...] = Field(
        max_length=ENVIRONMENT_ACTIVITY_BATCH_LIMIT
    )

    @field_validator("as_of")
    @classmethod
    def normalize_as_of(cls, value: datetime) -> datetime:
        return _utc(value)

    def validate_for_query(self, query: EnvironmentActivityQuery) -> EnvironmentActivitySnapshot:
        query = copy_environment_activity_query(query)
        if type(self.environments) is not tuple or len(self.environments) != len(
            query.environment_names
        ):
            raise ValueError("Environment activity did not cover the requested names.")
        # Rebuild each bounded row explicitly, without serializing extension objects.
        rows = []
        for name, row in zip(query.environment_names, self.environments, strict=True):
            if (
                type(row) is not EnvironmentSessionActivity
                or type(row.environment_name) is not str
                or row.environment_name != name
            ):
                raise ValueError("Environment activity returned unexpected names.")
            rows.append(
                EnvironmentSessionActivity(
                    environment_name=row.environment_name,
                    session_count=row.session_count,
                    last_activity_at=row.last_activity_at,
                )
            )
        return EnvironmentActivitySnapshot(as_of=self.as_of, environments=tuple(rows))


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Environment activity timestamps must be timezone-aware.")
    return value.astimezone(UTC)


def environment_activity_snapshot(
    query: EnvironmentActivityQuery, as_of: datetime, values: dict[str, tuple[int, datetime]]
) -> EnvironmentActivitySnapshot:
    """Fill requested empty groups without treating unknown populations as zero."""
    return EnvironmentActivitySnapshot(
        as_of=as_of,
        environments=tuple(
            EnvironmentSessionActivity(
                environment_name=name,
                session_count=values[name][0] if name in values else 0,
                last_activity_at=values[name][1] if name in values else None,
            )
            for name in query.environment_names
        ),
    )
