"""Optional environment and dotenv loading for resolved server configuration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Self, get_args, get_origin

try:
    from pydantic_settings import (
        BaseSettings,
        DotEnvSettingsSource,
        EnvSettingsSource,
        PydanticBaseSettingsSource,
        SecretsSettingsSource,
        SettingsConfigDict,
        SettingsError,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised without the optional extra
    raise RuntimeError(
        "Cayu server settings require the optional settings packages. "
        'Install them with `pip install "cayu[server-settings]"`.'
    ) from exc

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

if TYPE_CHECKING:
    from pydantic.config import ExtraValues

from cayu._validation import require_clean_nonblank
from cayu.runtime.sessions import SessionStatus
from cayu.server.auth import BasicAuth
from cayu.server.config import (
    DEFAULT_DASHBOARD_PATH,
    DEFAULT_EVENT_SIDE_EFFECT_STARTUP_TIMEOUT_SECONDS,
    DEFAULT_INTERRUPTION_SHUTDOWN_GRACE_SECONDS,
    DEFAULT_RECOVERY_INACTIVE_AFTER_SECONDS,
    DEFAULT_REPLAY_IDLE_TIMEOUT_SECONDS,
    DEFAULT_SERVER_DEPLOYMENT_NAME,
    DEFAULT_SERVER_TITLE,
    AuthenticatedAccess,
    CorsConfig,
    DashboardConfig,
    DocsConfig,
    OpenAccess,
    ServerAccessConfig,
    ServerApiConfig,
    ServerConfig,
    ServerLifecycleConfig,
)
from cayu.server.contracts import SERVER_API_PREFIX

__all__ = [
    "CorsSettings",
    "DashboardSettings",
    "DocsSettings",
    "ServerAccessSettings",
    "ServerApiSettings",
    "ServerLifecycleSettings",
    "ServerSettings",
]


class ServerAccessSettings(BaseModel):
    """Source-friendly access selection for open, basic, or external auth."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    mode: Literal["open", "basic", "external"]
    username: str | None = None
    password: SecretStr | None = None
    realm: str | None = None
    subject: str | None = None
    tenant: str | None = None

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        return _model_validate_json_with_redaction(
            cls,
            super().model_validate_json,
            json_data,
            strict=strict,
            extra=extra,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @model_validator(mode="wrap")
    @classmethod
    def redact_validation_inputs(cls, value: Any, handler: Any) -> ServerAccessSettings:
        return _validate_settings_model(
            cls,
            value,
            handler,
            sensitive_context_prefixes=((),),
        )

    @field_validator("username", "realm", "subject", "tenant")
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class DashboardSettings(BaseModel):
    """Source-friendly dashboard settings with optional distinct access."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    enabled: bool = True
    path: str = DEFAULT_DASHBOARD_PATH
    directory: Path | None = None
    access: ServerAccessSettings | None = None
    runtime_config: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        return _model_validate_json_with_redaction(
            cls,
            super().model_validate_json,
            json_data,
            strict=strict,
            extra=extra,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @model_validator(mode="wrap")
    @classmethod
    def redact_validation_inputs(cls, value: Any, handler: Any) -> DashboardSettings:
        return _validate_settings_model(
            cls,
            value,
            handler,
            sensitive_context_prefixes=(("access",), ("runtime_config",)),
        )

    @field_validator("directory", mode="before")
    @classmethod
    def validate_directory(cls, value: object) -> object:
        if isinstance(value, str):
            return require_clean_nonblank(value, "dashboard.directory")
        return value


class ServerApiSettings(BaseModel):
    """Environment-coercible API settings validated again by ``ServerConfig``."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    enabled: bool = True
    path: str = SERVER_API_PREFIX


class DocsSettings(BaseModel):
    """Environment-coercible documentation exposure."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    enabled: bool = False


class CorsSettings(BaseModel):
    """Environment-coercible CORS settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    allowed_origins: tuple[str, ...] = ()
    allow_methods: tuple[str, ...] = ("*",)
    allow_headers: tuple[str, ...] = ("*",)
    allow_credentials: bool = False


class ServerLifecycleSettings(BaseModel):
    """Environment-coercible lifecycle values with core validation on resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    replay_idle_timeout_s: float = DEFAULT_REPLAY_IDLE_TIMEOUT_SECONDS
    startup_recovery_statuses: frozenset[SessionStatus] | None = None
    recovery_inactive_after_seconds: int = DEFAULT_RECOVERY_INACTIVE_AFTER_SECONDS
    event_side_effect_startup_timeout_seconds: float = (
        DEFAULT_EVENT_SIDE_EFFECT_STARTUP_TIMEOUT_SECONDS
    )
    interruption_shutdown_grace_seconds: float = DEFAULT_INTERRUPTION_SHUTDOWN_GRACE_SECONDS


class _ServerSettingsSource(PydanticBaseSettingsSource):
    """Preserve a standard source while surfacing unknown Cayu root keys."""

    keep_only_prefixed_model_fields = False
    source_description = "environment variables"

    def __init__(self, source: EnvSettingsSource) -> None:
        super().__init__(source.settings_cls)
        self._source = source

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._source.get_field_value(field, field_name)

    def __call__(self) -> dict[str, Any]:
        source = self._source
        env_vars = source.env_vars
        env_prefix = source.env_prefix
        case_sensitive = source.case_sensitive
        delimiter = source.env_nested_delimiter
        source_maxsplit = getattr(source, "maxsplit", -1)
        path_maxsplit = source_maxsplit if source_maxsplit < 0 else source_maxsplit + 1

        def normalize(name: str) -> str:
            return name if case_sensitive else name.lower()

        model_fields = {
            normalize(name): field for name, field in self.settings_cls.model_fields.items()
        }
        if not case_sensitive:
            env_prefix = env_prefix.lower()

        prefixed_roots: set[str] = set()
        invalid_paths: list[tuple[str, ...]] = []
        for env_name in env_vars:
            if not env_name.startswith(env_prefix):
                continue
            relative_name = env_name[len(env_prefix) :]
            name_parts = (
                relative_name.split(delimiter, maxsplit=path_maxsplit)
                if delimiter
                else [relative_name]
            )
            root_name = name_parts[0]
            if not root_name:
                continue
            normalized_root = normalize(root_name)
            prefixed_roots.add(normalized_root)
            normalized_path = tuple(normalize(part) for part in name_parts)
            if not _environment_path_is_valid(
                self.settings_cls,
                normalized_path,
                normalize=normalize,
            ):
                invalid_paths.append(normalized_path)

        if invalid_paths:
            _raise_redacted_settings_error(
                self.settings_cls,
                [
                    InitErrorDetails(type="extra_forbidden", loc=path, input=None)
                    for path in invalid_paths
                ],
            )

        source_data: dict[str, Any] = {}
        source_parsed = False
        try:
            source_data = source()
        except (SettingsError, UnicodeError):
            pass
        else:
            source_parsed = True
        if not source_parsed:
            _raise_redacted_settings_parse_error(self.source_description)

        if self.keep_only_prefixed_model_fields:
            data = {
                name: value
                for name, value in source_data.items()
                if normalize(name) in model_fields and normalize(name) in prefixed_roots
            }
        else:
            data = dict(source_data)
        _raise_for_forbidden_source_fields(self.settings_cls, data)
        return data


def _raise_for_forbidden_source_fields(
    settings_cls: type[BaseSettings],
    data: dict[str, Any],
) -> None:
    source_errors = [
        InitErrorDetails(
            type="extra_forbidden",
            loc=(field_name,),
            input=None,
        )
        for field_name in data
        if field_name not in settings_cls.model_fields
    ]
    for field_name, field in settings_cls.model_fields.items():
        if field_name not in data:
            continue
        try:
            TypeAdapter(field.annotation).validate_python(data[field_name])
        except ValidationError as exc:
            for error in exc.errors(include_input=False):
                if error["type"] != "extra_forbidden":
                    continue
                source_errors.append(
                    InitErrorDetails(
                        type="extra_forbidden",
                        loc=(field_name, *error["loc"]),
                        input=None,
                    )
                )
    if source_errors:
        _raise_redacted_settings_error(settings_cls, source_errors)


def _raise_redacted_settings_error(
    settings_cls: type[BaseModel],
    errors: list[InitErrorDetails],
) -> NoReturn:
    raise ValidationError.from_exception_data(
        settings_cls.__name__,
        errors,
        hide_input=True,
    ) from None


def _raise_redacted_settings_parse_error(source_description: str) -> NoReturn:
    raise SettingsError(
        f"Unable to parse Cayu server settings from {source_description}."
    ) from None


def _model_validate_json_with_redaction(
    settings_cls: type[BaseModel],
    validate_json: Callable[..., BaseModel],
    json_data: str | bytes | bytearray,
    *,
    strict: bool | None,
    extra: ExtraValues | None,
    context: Any | None,
    by_alias: bool | None,
    by_name: bool | None,
) -> Any:
    try:
        return validate_json(
            json_data,
            strict=strict,
            extra=extra,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )
    except ValidationError as exc:
        if not any(error["type"] == "json_invalid" for error in exc.errors()):
            raise
    _raise_redacted_settings_error(
        settings_cls,
        [
            InitErrorDetails(
                type=PydanticCustomError(
                    "json_invalid",
                    "Invalid JSON for server settings.",
                ),
                loc=(),
                input=None,
            )
        ],
    )


def _redacted_error_details(
    exc: ValidationError,
    *,
    sensitive_context_prefixes: tuple[tuple[str, ...], ...],
) -> list[InitErrorDetails]:
    details: list[InitErrorDetails] = []
    for error in exc.errors(include_input=False):
        location = error["loc"]
        context = error.get("ctx")
        context_is_sensitive = any(
            location[: len(prefix)] == prefix for prefix in sensitive_context_prefixes
        )
        if error["type"] == "invalid_server_settings" or (
            context is not None
            and (
                context_is_sensitive
                or any(not isinstance(value, BaseException) for value in context.values())
            )
        ):
            details.append(
                InitErrorDetails(
                    type=PydanticCustomError(
                        "invalid_server_settings",
                        "Invalid server settings.",
                    ),
                    loc=location,
                    input=None,
                )
            )
            continue

        item: InitErrorDetails = {
            "type": error["type"],
            "loc": location,
            "input": None,
        }
        if context is not None:
            item["ctx"] = {key: ValueError(str(value)) for key, value in context.items()}
        details.append(item)
    return details


def _validate_settings_model(
    settings_cls: type[BaseModel],
    value: Any,
    handler: Any,
    *,
    sensitive_context_prefixes: tuple[tuple[str, ...], ...],
) -> Any:
    try:
        return handler(value)
    except ValidationError as exc:
        errors = _redacted_error_details(
            exc,
            sensitive_context_prefixes=sensitive_context_prefixes,
        )
    _raise_redacted_settings_error(settings_cls, errors)


def _environment_path_is_valid(
    settings_cls: type[BaseSettings],
    path: tuple[str, ...],
    *,
    normalize: Callable[[str], str],
) -> bool:
    annotations: tuple[Any, ...] = (settings_cls,)
    for segment in path:
        next_annotations: list[Any] = []
        for annotation in annotations:
            if _allows_arbitrary_nested_keys(annotation):
                return True
            for model in _model_types(annotation):
                fields = {normalize(name): field for name, field in model.model_fields.items()}
                field = fields.get(segment)
                if field is not None:
                    next_annotations.append(field.annotation)
        if not next_annotations:
            return False
        annotations = tuple(next_annotations)
    return True


def _allows_arbitrary_nested_keys(annotation: Any) -> bool:
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    mapping_candidate = origin or annotation
    if isinstance(mapping_candidate, type) and issubclass(mapping_candidate, Mapping):
        return True
    return any(_allows_arbitrary_nested_keys(argument) for argument in get_args(annotation))


def _model_types(annotation: Any) -> tuple[type[BaseModel], ...]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return (annotation,)
    return tuple(model for argument in get_args(annotation) for model in _model_types(argument))


class _ServerEnvironmentSource(_ServerSettingsSource):
    pass


class _ServerDotenvSource(_ServerSettingsSource):
    keep_only_prefixed_model_fields = True
    source_description = "the dotenv file"


class _ServerSecretsSource(_ServerSettingsSource):
    """Interpret prefixed secret filenames with the environment nesting contract."""

    source_description = "file secrets"

    def __init__(
        self,
        source: SecretsSettingsSource,
        environment_source: EnvSettingsSource,
    ) -> None:
        self._file_secret_source = source
        environment_options: dict[str, Any] = {
            "case_sensitive": environment_source.case_sensitive,
            "env_prefix": environment_source.env_prefix,
            "env_nested_delimiter": environment_source.env_nested_delimiter,
            "env_ignore_empty": environment_source.env_ignore_empty,
            "env_parse_none_str": environment_source.env_parse_none_str,
            "env_parse_enums": environment_source.env_parse_enums,
        }
        for option in ("env_prefix_target", "env_nested_max_split"):
            if hasattr(environment_source, option):
                environment_options[option] = getattr(environment_source, option)
        super().__init__(
            EnvSettingsSource(
                source.settings_cls,
                **environment_options,
            )
        )

    def __call__(self) -> dict[str, Any]:
        # Let the standard source validate configured directories and preserve
        # its missing-directory warnings before adapting the discovered files.
        source_validated = False
        try:
            self._file_secret_source()
        except (SettingsError, UnicodeError):
            pass
        else:
            source_validated = True
        if not source_validated:
            _raise_redacted_settings_parse_error(self.source_description)
        self._source.env_vars = self._load_prefixed_secret_files()
        return super().__call__()

    def _load_prefixed_secret_files(self) -> dict[str, str]:
        case_sensitive = self._source.case_sensitive

        def normalize(name: str) -> str:
            return name if case_sensitive else name.lower()

        prefix = normalize(self._source.env_prefix)
        values: dict[str, str] = {}
        for directory in getattr(self._file_secret_source, "secrets_paths", ()):
            directory_values: dict[str, str] = {}
            for entry in sorted(directory.iterdir(), key=lambda path: path.name):
                normalized_name = normalize(entry.name)
                if not normalized_name.startswith(prefix):
                    continue
                if not entry.is_file():
                    raise ValueError(
                        f"Prefixed secret entry {entry.name!r} must be a regular file."
                    )
                if normalized_name in directory_values:
                    raise ValueError(
                        "Secret filenames must be unique under case-insensitive matching: "
                        f"{entry.name!r}."
                    )
                contents = ""
                contents_read = False
                try:
                    contents = entry.read_text()
                except UnicodeError:
                    pass
                else:
                    contents_read = True
                if not contents_read:
                    _raise_redacted_settings_parse_error(self.source_description)
                directory_values[normalized_name] = contents.strip()
            values.update(directory_values)
        return values


class ServerSettings(BaseSettings):
    """Load conventional Cayu server settings from init values, env, or dotenv.

    The resulting object remains source-specific. Call :meth:`to_config` once
    during startup to obtain the source-agnostic ``ServerConfig`` consumed by
    ``create_server``.
    """

    model_config = SettingsConfigDict(
        env_prefix="CAYU_SERVER_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    @wraps(BaseSettings.__init__)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        dotenv_decode_failed = False
        try:
            super().__init__(*args, **kwargs)
        except UnicodeError:
            dotenv_decode_failed = True
        if dotenv_decode_failed:
            # Raise outside the exception handler so the decoder exception and
            # its secret-bearing byte buffer are not retained as __context__.
            # Pydantic Settings versions may decode dotenv files before
            # settings_customise_sources() can install Cayu's redacting wrapper.
            _raise_redacted_settings_parse_error("the dotenv file")

    deployment_name: str = DEFAULT_SERVER_DEPLOYMENT_NAME
    title: str = DEFAULT_SERVER_TITLE
    access: ServerAccessSettings | None = None
    api: ServerApiSettings = Field(default_factory=ServerApiSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    docs: DocsSettings = Field(default_factory=DocsSettings)
    cors: CorsSettings = Field(default_factory=CorsSettings)
    lifecycle: ServerLifecycleSettings = Field(default_factory=ServerLifecycleSettings)

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        return _model_validate_json_with_redaction(
            cls,
            super().model_validate_json,
            json_data,
            strict=strict,
            extra=extra,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @model_validator(mode="wrap")
    @classmethod
    def redact_validation_inputs(cls, value: Any, handler: Any) -> ServerSettings:
        return _validate_settings_model(
            cls,
            value,
            handler,
            sensitive_context_prefixes=(
                ("access",),
                ("dashboard", "access"),
                ("dashboard", "runtime_config"),
            ),
        )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        if (
            not isinstance(env_settings, EnvSettingsSource)
            or not isinstance(dotenv_settings, DotEnvSettingsSource)
            or not isinstance(file_secret_settings, SecretsSettingsSource)
        ):
            raise TypeError(
                "ServerSettings requires standard environment, dotenv, and file-secret sources."
            )
        return (
            init_settings,
            _ServerEnvironmentSource(env_settings),
            _ServerDotenvSource(dotenv_settings),
            _ServerSecretsSource(file_secret_settings, env_settings),
        )

    def to_config(
        self,
        *,
        access: ServerAccessConfig | None = None,
        dashboard_access: ServerAccessConfig | None = None,
    ) -> ServerConfig:
        """Resolve source settings and optional external auth into ``ServerConfig``."""

        resolved_access = _resolve_access(
            self.access,
            explicit=access,
            field_name="access",
        )
        if resolved_access is None:  # defensive invariant; required resolution fails above
            raise RuntimeError("Resolved server access cannot be empty.")
        resolved_dashboard_access = _resolve_access(
            self.dashboard.access,
            explicit=dashboard_access,
            field_name="dashboard.access",
            allow_inherit=True,
        )
        return ServerConfig(
            deployment_name=self.deployment_name,
            title=self.title,
            access=resolved_access,
            api=ServerApiConfig(
                enabled=self.api.enabled,
                path=self.api.path,
            ),
            dashboard=DashboardConfig(
                enabled=self.dashboard.enabled,
                path=self.dashboard.path,
                directory=self.dashboard.directory,
                access=resolved_dashboard_access,
                runtime_config=self.dashboard.runtime_config,
            ),
            docs=DocsConfig(enabled=self.docs.enabled),
            cors=CorsConfig(
                allowed_origins=self.cors.allowed_origins,
                allow_methods=self.cors.allow_methods,
                allow_headers=self.cors.allow_headers,
                allow_credentials=self.cors.allow_credentials,
            ),
            lifecycle=ServerLifecycleConfig(
                replay_idle_timeout_s=self.lifecycle.replay_idle_timeout_s,
                startup_recovery_statuses=self.lifecycle.startup_recovery_statuses,
                recovery_inactive_after_seconds=(self.lifecycle.recovery_inactive_after_seconds),
                event_side_effect_startup_timeout_seconds=(
                    self.lifecycle.event_side_effect_startup_timeout_seconds
                ),
                interruption_shutdown_grace_seconds=(
                    self.lifecycle.interruption_shutdown_grace_seconds
                ),
            ),
        )


def _resolve_access(
    settings: ServerAccessSettings | None,
    *,
    explicit: ServerAccessConfig | None,
    field_name: str,
    allow_inherit: bool = False,
) -> ServerAccessConfig | None:
    if explicit is not None:
        if settings is not None and settings.mode != "external":
            raise ValueError(
                f"Explicit {field_name} conflicts with settings mode {settings.mode!r}. "
                "Use mode='external' when injecting an application auth policy."
            )
        if settings is not None:
            _reject_basic_fields(settings, field_name=field_name)
        return explicit
    if settings is None:
        if allow_inherit:
            return None
        raise ValueError(
            "Server settings do not select an access policy. Configure "
            "CAYU_SERVER_ACCESS__MODE or pass an explicit access policy to to_config()."
        )
    if settings.mode == "external":
        raise ValueError(
            f"{field_name} mode 'external' requires an explicit access policy in to_config()."
        )
    if settings.mode == "open":
        _reject_basic_fields(settings, field_name=field_name)
        return OpenAccess()
    if settings.username is None or settings.password is None:
        raise ValueError("Basic server access requires username and password.")
    if not settings.password.get_secret_value().strip():
        raise ValueError("Basic server access password cannot be blank.")
    dependency = BasicAuth(
        username=settings.username,
        password=settings.password.get_secret_value(),
        realm=settings.realm or DEFAULT_SERVER_TITLE,
        subject=settings.subject,
        tenant=settings.tenant,
    )
    return AuthenticatedAccess(dependency=dependency)


def _reject_basic_fields(settings: ServerAccessSettings, *, field_name: str) -> None:
    if any(
        value is not None
        for value in (
            settings.username,
            settings.password,
            settings.realm,
            settings.subject,
            settings.tenant,
        )
    ):
        raise ValueError(
            f"{field_name} mode {settings.mode!r} cannot include Basic authentication fields."
        )
