"""Resolved, source-agnostic configuration for the Cayu server surface."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn
from unicodedata import category as unicode_category

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    TypeAdapter,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticSerializationError, to_jsonable_python

from cayu._validation import (
    copy_json_value,
    freeze_json_value,
    require_clean_nonblank,
    require_durable_json_text,
    require_unicode_scalar_text,
    thaw_json_value,
)
from cayu.runtime.sessions import IncompleteSessionsRecoveryRequest, SessionStatus
from cayu.server.contracts import SERVER_API_PREFIX

DEFAULT_SERVER_DEPLOYMENT_NAME = "development"
DEFAULT_SERVER_TITLE = "Cayu"
DEFAULT_DASHBOARD_PATH = "/cayu"
DEFAULT_LOCAL_CORS_ORIGIN = "http://localhost:5173"
DEFAULT_REPLAY_IDLE_TIMEOUT_SECONDS = 300.0
DEFAULT_RECOVERY_INACTIVE_AFTER_SECONDS = 300
DEFAULT_EVENT_SIDE_EFFECT_STARTUP_TIMEOUT_SECONDS = 30.0
DEFAULT_INTERRUPTION_SHUTDOWN_GRACE_SECONDS = 10.0
_GENERATED_DOCS_PATHS = frozenset(
    {
        "/docs",
        "/docs/oauth2-redirect",
        "/openapi.json",
        "/redoc",
    }
)
_HTTP_TOKEN_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_PERCENT_ENCODED_OCTET_RE = re.compile(r"%[0-9A-Fa-f]{2}")

__all__ = [
    "AuthenticatedAccess",
    "CorsConfig",
    "DashboardConfig",
    "DocsConfig",
    "OpenAccess",
    "ServerAccessConfig",
    "ServerApiConfig",
    "ServerConfig",
    "ServerLifecycleConfig",
]


def _raise_redacted_config_error(
    title: str,
    errors: list[InitErrorDetails],
) -> NoReturn:
    raise ValidationError.from_exception_data(
        title,
        errors,
        hide_input=True,
    ) from None


def _redacted_config_errors(
    exc: ValidationError,
    *,
    message: str,
) -> list[InitErrorDetails]:
    errors = exc.errors()
    if all(error.get("input") is None for error in errors):
        return _config_errors_without_inputs(exc)
    return [
        InitErrorDetails(
            type="value_error",
            loc=error["loc"],
            input=None,
            ctx={"error": ValueError(message)},
        )
        for error in errors
    ]


def _config_errors_without_inputs(exc: ValidationError) -> list[InitErrorDetails]:
    redacted_errors: list[InitErrorDetails] = []
    for error in exc.errors(include_input=False):
        details: InitErrorDetails = {
            "type": error["type"],
            "loc": error["loc"],
            "input": None,
        }
        if "ctx" in error:
            details["ctx"] = {
                key: ValueError(str(value)) if isinstance(value, BaseException) else value
                for key, value in error["ctx"].items()
            }
        redacted_errors.append(details)
    return redacted_errors


def _validate_with_redacted_errors(
    title: str,
    value: Any,
    handler: Any,
    *,
    message: str,
    preserve_messages: bool = False,
) -> Any:
    try:
        return handler(value)
    except ValidationError as exc:
        errors = (
            _config_errors_without_inputs(exc)
            if preserve_messages
            else _redacted_config_errors(exc, message=message)
        )
    _raise_redacted_config_error(title, errors)


class OpenAccess(BaseModel):
    """Deliberate unauthenticated access to the configured server surface."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: Literal["open"] = "open"

    @model_validator(mode="wrap")
    @classmethod
    def redact_validation_inputs(cls, value: Any, handler: Any) -> OpenAccess:
        return _validate_with_redacted_errors(
            cls.__name__,
            value,
            handler,
            message="Invalid open-access configuration.",
        )


class AuthenticatedAccess(BaseModel):
    """Access guarded by an application-provided Cayu auth dependency.

    The callable is runtime wiring rather than serializable configuration. It
    is consequently excluded from representations and model serialization so
    a dependency carrying credentials cannot leak through diagnostics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: Literal["authenticated"] = "authenticated"
    dependency: Any = Field(exclude=True, repr=False)

    @model_validator(mode="wrap")
    @classmethod
    def validate_dependency(cls, value: Any, handler: Any) -> AuthenticatedAccess:
        if isinstance(value, Mapping) and "dependency" in value:
            dependency = value["dependency"]
            if not callable(dependency):
                _raise_redacted_config_error(
                    cls.__name__,
                    [
                        InitErrorDetails(
                            type="value_error",
                            loc=("dependency",),
                            input=None,
                            ctx={
                                "error": ValueError(
                                    "Authenticated access requires a callable auth dependency."
                                )
                            },
                        )
                    ],
                )
        return _validate_with_redacted_errors(
            cls.__name__,
            value,
            handler,
            message="Invalid authenticated-access configuration.",
        )


ServerAccessConfig = Annotated[
    OpenAccess | AuthenticatedAccess,
    Field(discriminator="kind"),
]
_SERVER_ACCESS_ADAPTER = TypeAdapter(ServerAccessConfig)


def _validate_server_access(value: Any) -> ServerAccessConfig:
    try:
        return _SERVER_ACCESS_ADAPTER.validate_python(value)
    except ValidationError as exc:
        errors = _redacted_config_errors(
            exc,
            message="Invalid server access policy.",
        )
    _raise_redacted_config_error("ServerAccessConfig", errors)


class ServerApiConfig(BaseModel):
    """Control-plane API exposure and mount policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    enabled: StrictBool = True
    path: str = SERVER_API_PREFIX

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return normalize_api_path(value)


class DashboardConfig(BaseModel):
    """Bundled dashboard exposure, mount, and authorization policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    enabled: StrictBool = True
    path: str = DEFAULT_DASHBOARD_PATH
    directory: Path | None = None
    access: ServerAccessConfig | None = None
    runtime_config: Mapping[str, Any] = Field(default_factory=dict)

    @model_validator(mode="wrap")
    @classmethod
    def redact_validation_inputs(cls, value: Any, handler: Any) -> DashboardConfig:
        return _validate_with_redacted_errors(
            cls.__name__,
            value,
            handler,
            message="Invalid dashboard configuration.",
            preserve_messages=True,
        )

    @field_validator("directory", mode="before")
    @classmethod
    def validate_directory(cls, value: object) -> object:
        if isinstance(value, str):
            return require_clean_nonblank(value, "dashboard.directory")
        return value

    @field_validator("access", mode="before")
    @classmethod
    def validate_access(cls, value: Any) -> ServerAccessConfig | None:
        if value is None:
            return None
        return _validate_server_access(value)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return normalize_dashboard_path(value)

    @field_validator("runtime_config", mode="before")
    @classmethod
    def copy_runtime_config(cls, value: object) -> dict[str, Any]:
        return normalize_dashboard_runtime_config(value)

    @field_validator("runtime_config")
    @classmethod
    def freeze_runtime_config(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_json_value(dict(value))

    @field_serializer("runtime_config")
    def serialize_runtime_config(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return dict(thaw_json_value(value))


class DocsConfig(BaseModel):
    """FastAPI OpenAPI and interactive documentation exposure."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    enabled: StrictBool = False


class CorsConfig(BaseModel):
    """Explicit browser cross-origin request policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    allowed_origins: tuple[str, ...] = ()
    allow_methods: tuple[str, ...] = ("*",)
    allow_headers: tuple[str, ...] = ("*",)
    allow_credentials: StrictBool = False

    @field_validator("allowed_origins", "allow_methods", "allow_headers", mode="before")
    @classmethod
    def validate_string_sequence(cls, value: object, info) -> tuple[str, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise ValueError(f"{info.field_name} must be a sequence of strings.")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"{info.field_name} must contain only strings.")
            item = require_clean_nonblank(item, info.field_name)
            require_unicode_scalar_text(item, info.field_name)
            if info.field_name in {"allow_methods", "allow_headers"}:
                if _HTTP_TOKEN_RE.fullmatch(item) is None:
                    raise ValueError(f"{info.field_name} entries must be valid HTTP tokens.")
            elif any(unicode_category(character) == "Cc" for character in item):
                raise ValueError(f"{info.field_name} entries must not contain control characters.")
            normalized.append(item)
        return tuple(normalized)

    @model_validator(mode="after")
    def validate_credentials_policy(self) -> CorsConfig:
        if self.allow_credentials and any(
            "*" in values
            for values in (self.allowed_origins, self.allow_methods, self.allow_headers)
        ):
            raise ValueError(
                "CORS wildcard origins, methods, or headers cannot be combined with "
                "allow_credentials=True."
            )
        return self


class ServerLifecycleConfig(BaseModel):
    """Recovery, replay, and shutdown behavior owned by the Cayu server."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    replay_idle_timeout_s: float = DEFAULT_REPLAY_IDLE_TIMEOUT_SECONDS
    startup_recovery_statuses: frozenset[SessionStatus] | None = None
    recovery_inactive_after_seconds: int = DEFAULT_RECOVERY_INACTIVE_AFTER_SECONDS
    event_side_effect_startup_timeout_seconds: float = (
        DEFAULT_EVENT_SIDE_EFFECT_STARTUP_TIMEOUT_SECONDS
    )
    interruption_shutdown_grace_seconds: float = DEFAULT_INTERRUPTION_SHUTDOWN_GRACE_SECONDS

    @field_validator(
        "replay_idle_timeout_s",
        "event_side_effect_startup_timeout_seconds",
        "interruption_shutdown_grace_seconds",
        mode="before",
    )
    @classmethod
    def validate_positive_seconds(cls, value: object, info) -> float:
        return _positive_seconds(value, info.field_name)

    @field_validator("recovery_inactive_after_seconds", mode="before")
    @classmethod
    def validate_recovery_inactivity(cls, value: object) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("recovery_inactive_after_seconds must be a non-negative integer.")
        return value

    @field_validator("startup_recovery_statuses", mode="before")
    @classmethod
    def validate_startup_recovery_statuses(
        cls,
        value: object,
    ) -> frozenset[SessionStatus] | None:
        if value is None:
            return None
        candidate = set(value) if isinstance(value, frozenset) else value
        validated = IncompleteSessionsRecoveryRequest.model_validate(
            {"statuses": candidate}
        ).statuses
        return frozenset(validated)


class ServerConfig(BaseModel):
    """Fully resolved Cayu server identity and policy.

    This model deliberately knows nothing about where configuration originated.
    Applications may construct it directly after resolving secrets from any
    provider. Deployment identity is metadata only and never selects policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    deployment_name: str = DEFAULT_SERVER_DEPLOYMENT_NAME
    title: str = DEFAULT_SERVER_TITLE
    access: ServerAccessConfig
    api: ServerApiConfig = Field(default_factory=ServerApiConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    docs: DocsConfig = Field(default_factory=DocsConfig)
    cors: CorsConfig = Field(default_factory=CorsConfig)
    lifecycle: ServerLifecycleConfig = Field(default_factory=ServerLifecycleConfig)

    @model_validator(mode="wrap")
    @classmethod
    def redact_validation_inputs(cls, value: Any, handler: Any) -> ServerConfig:
        try:
            config = handler(value)
            config._validate_mount_relationships()
            return config
        except ValidationError as exc:
            errors = _config_errors_without_inputs(exc)
        except ValueError as exc:
            errors = [
                InitErrorDetails(
                    type="value_error",
                    loc=(),
                    input=None,
                    ctx={"error": ValueError(str(exc))},
                )
            ]
        _raise_redacted_config_error(cls.__name__, errors)

    @field_validator("access", mode="before")
    @classmethod
    def validate_access(cls, value: Any) -> ServerAccessConfig:
        return _validate_server_access(value)

    @field_validator("deployment_name", "title")
    @classmethod
    def validate_nonblank_text(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        return require_unicode_scalar_text(value, info.field_name)

    def _validate_mount_relationships(self) -> None:
        if (
            self.api.enabled
            and self.dashboard.enabled
            and _is_same_or_child_path(self.dashboard.path, self.api.path)
        ):
            raise ValueError("dashboard.path must not live inside api.path.")
        if self.dashboard.enabled and not self.api.enabled:
            raise ValueError("dashboard.enabled requires api.enabled so the dashboard can operate.")
        if (
            self.dashboard.enabled
            and self.docs.enabled
            and self.dashboard.path in _GENERATED_DOCS_PATHS
        ):
            raise ValueError(
                "dashboard.path conflicts with a generated documentation route while "
                "docs.enabled is true."
            )

    @classmethod
    def local_development(
        cls,
        *,
        deployment_name: str = DEFAULT_SERVER_DEPLOYMENT_NAME,
        api: ServerApiConfig | None = None,
        dashboard: DashboardConfig | None = None,
        lifecycle: ServerLifecycleConfig | None = None,
    ) -> ServerConfig:
        """Build an explicitly open local configuration with docs and Vite CORS."""

        return cls(
            deployment_name=deployment_name,
            access=OpenAccess(),
            api=api or ServerApiConfig(),
            dashboard=dashboard or DashboardConfig(),
            docs=DocsConfig(enabled=True),
            cors=CorsConfig(allowed_origins=(DEFAULT_LOCAL_CORS_ORIGIN,)),
            lifecycle=lifecycle or ServerLifecycleConfig(),
        )

    @classmethod
    def protected(
        cls,
        dependency: Any,
        *,
        deployment_name: str = DEFAULT_SERVER_DEPLOYMENT_NAME,
        api: ServerApiConfig | None = None,
        dashboard: DashboardConfig | None = None,
        docs: DocsConfig | None = None,
        cors: CorsConfig | None = None,
        lifecycle: ServerLifecycleConfig | None = None,
    ) -> ServerConfig:
        """Build a protected configuration around an application auth dependency."""

        return cls(
            deployment_name=deployment_name,
            access=AuthenticatedAccess(dependency=dependency),
            api=api or ServerApiConfig(),
            dashboard=dashboard or DashboardConfig(),
            docs=docs or DocsConfig(),
            cors=cors or CorsConfig(),
            lifecycle=lifecycle or ServerLifecycleConfig(),
        )

    def safe_summary(self) -> dict[str, Any]:
        """Return non-secret effective policy suitable for operator diagnostics."""

        dashboard_access = self.dashboard.access or self.access
        return {
            "deployment_name": self.deployment_name,
            "title": self.title,
            "access": self.access.kind,
            "api": {"enabled": self.api.enabled, "path": self.api.path},
            "dashboard": {
                "enabled": self.dashboard.enabled,
                "path": self.dashboard.path,
                "access": dashboard_access.kind,
            },
            "docs": {"enabled": self.docs.enabled},
            "cors": {
                "allowed_origins": list(self.cors.allowed_origins),
                "allow_methods": list(self.cors.allow_methods),
                "allow_headers": list(self.cors.allow_headers),
                "allow_credentials": self.cors.allow_credentials,
            },
            "lifecycle": {
                "replay_idle_timeout_s": self.lifecycle.replay_idle_timeout_s,
                "startup_recovery_statuses": (
                    None
                    if self.lifecycle.startup_recovery_statuses is None
                    else sorted(status.value for status in self.lifecycle.startup_recovery_statuses)
                ),
                "recovery_inactive_after_seconds": (self.lifecycle.recovery_inactive_after_seconds),
                "event_side_effect_startup_timeout_seconds": (
                    self.lifecycle.event_side_effect_startup_timeout_seconds
                ),
                "interruption_shutdown_grace_seconds": (
                    self.lifecycle.interruption_shutdown_grace_seconds
                ),
            },
        }


def auth_dependency_for(access: ServerAccessConfig) -> Any | None:
    """Resolve an access policy into the existing server auth dependency contract."""

    if isinstance(access, OpenAccess):
        return None
    if isinstance(access, AuthenticatedAccess):
        return access.dependency
    raise TypeError("access must be OpenAccess or AuthenticatedAccess.")


def normalize_api_path(path: str, *, field_name: str = "api.path") -> str:
    """Normalize and validate a control-plane API mount path."""

    return _normalize_mount_path(path, field_name=field_name, allow_root=False)


def normalize_dashboard_path(path: str, *, field_name: str = "dashboard.path") -> str:
    """Normalize and validate a dashboard mount path."""

    return _normalize_mount_path(path, field_name=field_name, allow_root=True)


def _normalize_mount_path(path: str, *, field_name: str, allow_root: bool) -> str:
    value = require_clean_nonblank(path, field_name)
    require_unicode_scalar_text(value, field_name)
    if "?" in value or "#" in value or "://" in value:
        raise ValueError(f"{field_name} must be a URL path, not a URL.")
    if "\\" in value or any(unicode_category(character) == "Cc" for character in value):
        raise ValueError(f"{field_name} must not contain backslashes or control characters.")
    if value != "/" and (value.startswith("//") or value.endswith("//")):
        raise ValueError(f"{field_name} must not contain repeated path separators.")
    normalized = "/" + value.strip("/")
    if "//" in normalized:
        raise ValueError(f"{field_name} must not contain repeated path separators.")
    if _PERCENT_ENCODED_OCTET_RE.search(normalized):
        raise ValueError(
            f"{field_name} must use decoded path characters, not percent-encoded octets."
        )
    segments = normalized.split("/")[1:]
    if any(segment in {".", ".."} for segment in segments):
        raise ValueError(f"{field_name} must not contain dot path segments.")
    if normalized == "/" and not allow_root:
        raise ValueError(f"{field_name} must not be the site root.")
    return "/" if normalized == "/" else normalized


def normalize_dashboard_runtime_config(
    value: object,
    *,
    field_name: str = "dashboard.runtime_config",
) -> dict[str, Any]:
    """Return an owned JSON-object copy suitable for browser injection."""

    try:
        json_value = to_jsonable_python(value)
    except (PydanticSerializationError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain JSON-serializable values.") from exc
    copied = copy_json_value(json_value, field_name)
    if type(copied) is not dict:
        raise ValueError(f"{field_name} must be an object.")
    require_durable_json_text(copied, field_name)
    return copied


def _positive_seconds(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a finite positive number.")
    return float(value)


def _is_same_or_child_path(path: str, parent: str) -> bool:
    if path == parent:
        return True
    if parent == "/":
        return True
    return path.startswith(f"{parent}/")
