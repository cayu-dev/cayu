"""Settings an application's parts declare so Cayu Cloud can manage them.

A part declares its settings where it is registered: ``register_agent(...,
settings=...)``, ``register_provider(..., settings=...)``,
``register_environment_factory(..., settings=...)`` and
``CayuApp.declare_settings`` for the application itself. Chosen ``CayuConfig``
fields and keyed application budget limits are declared by the framework. The
code's value is always the default; a value set in Cloud overrides it.

The assembled list is the Release's ``cayu.managed-settings-schema/v1``
document (see ``docs/managed-settings.md``). It is part of ``describe_app`` and
is published to Cloud in the agent registry registration when the Release
starts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from math import isfinite
from typing import Annotated, Any, Literal, get_args
from urllib.parse import quote
from uuid import uuid4

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    WithJsonSchema,
    model_validator,
)

from cayu.context.thinking import ThinkingEffort

MANAGED_SETTINGS_SCHEMA_CONTRACT = "cayu.managed-settings-schema/v1"
# Limits of the published schema (contract section 2). Assembly refuses to exceed them.
MAX_MANAGED_SETTINGS = 500
MAX_MANAGED_SETTINGS_SCHEMA_BYTES = 256 * 1024
MAX_KEY_LENGTH = 128
MAX_OWNER_LENGTH = 128
MAX_LABEL_LENGTH = 80
MAX_DESCRIPTION_LENGTH = 500
MAX_STRING_LENGTH = 4096
MAX_CHOICES = 100

ManagedSettingType = Literal["bool", "int", "number", "string", "enum", "secret_ref"]
ManagedSettingApplies = Literal["restart", "next_session", "live"]
ManagedSettingValue = Annotated[
    StrictBool | StrictInt | StrictFloat | StrictStr | None,
    # Spelled out: the generated order of union members varies across Python versions.
    WithJsonSchema({"type": ["boolean", "integer", "number", "string", "null"]}),
]
_Bound = Annotated[StrictInt | StrictFloat | None, WithJsonSchema({"type": ["number", "null"]})]

KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_-]+)+$")
# A part's own key, before the part's namespace is put in front of it.
_RELATIVE_KEY_PATTERN = re.compile(r"^[a-z0-9_-]+(\.[a-z0-9_-]+)*$")
# One namespace segment: an agent, provider or environment name, or a budget key.
_SEGMENT_PATTERN = re.compile(r"^[a-z0-9_-]+$")
SECRET_REF_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_THINKING_EFFORTS: tuple[str, ...] = get_args(ThinkingEffort)
_THINKING_EFFORT_LABELS = {"xhigh": "Extra high"}


class ManagedSettingChoice(BaseModel):
    """One allowed value of an ``enum`` setting."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    value: StrictStr = Field(min_length=1, max_length=256)
    label: StrictStr = Field(min_length=1, max_length=MAX_LABEL_LENGTH)


def _absent(value: object) -> bool:
    return value is None


class ManagedSetting(BaseModel):
    """One setting descriptor (``docs/managed-settings.md``, section 1).

    Parts declare settings with a key relative to themselves (``"max_steps"``);
    registration puts the part's namespace in front (``"agent.reviewer.max_steps"``)
    and sets ``owner``. ``default`` is the code's value, or ``None`` when unset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    key: StrictStr = Field(min_length=1, max_length=MAX_KEY_LENGTH)
    type: ManagedSettingType
    label: StrictStr = Field(min_length=1, max_length=MAX_LABEL_LENGTH)
    applies: ManagedSettingApplies = "next_session"
    default: ManagedSettingValue = None
    owner: StrictStr | None = Field(
        default=None, min_length=1, max_length=MAX_OWNER_LENGTH, exclude_if=_absent
    )
    description: StrictStr | None = Field(
        default=None, max_length=MAX_DESCRIPTION_LENGTH, exclude_if=_absent
    )
    minimum: _Bound = Field(default=None, exclude_if=_absent)
    maximum: _Bound = Field(default=None, exclude_if=_absent)
    max_length: StrictInt | None = Field(
        default=None, ge=1, le=MAX_STRING_LENGTH, exclude_if=_absent
    )
    choices: tuple[ManagedSettingChoice, ...] | None = Field(default=None, exclude_if=_absent)

    @model_validator(mode="after")
    def _validate_shape(self) -> ManagedSetting:
        if not (KEY_PATTERN.fullmatch(self.key) or _RELATIVE_KEY_PATTERN.fullmatch(self.key)):
            raise ValueError(
                "key must be dotted lowercase segments of letters, digits, '_' and '-'."
            )
        numeric = self.type in {"int", "number"}
        if not numeric and (self.minimum is not None or self.maximum is not None):
            raise ValueError("minimum and maximum apply only to int and number settings.")
        if self.type == "int" and any(
            type(bound) is not int for bound in (self.minimum, self.maximum) if bound is not None
        ):
            raise ValueError("int bounds must be integers.")
        for bound in (self.minimum, self.maximum):
            if type(bound) is float and not isfinite(bound):
                raise ValueError("bounds must be finite.")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must not be greater than maximum.")
        if (self.type == "string") != (self.max_length is not None):
            raise ValueError("max_length is required for string settings and only for them.")
        if (self.type == "enum") != (self.choices is not None):
            raise ValueError("choices are required for enum settings and only for them.")
        if self.choices is not None:
            if not 1 <= len(self.choices) <= MAX_CHOICES:
                raise ValueError(f"enum settings need 1 to {MAX_CHOICES} choices.")
            values = [choice.value for choice in self.choices]
            if len(set(values)) != len(values):
                raise ValueError("enum choices must have distinct values.")
        if self.default is not None:
            self.validate_value(self.default)
        return self

    def validate_value(self, value: object) -> ManagedSettingValue:
        """Return ``value`` if it is valid for this setting, else raise ``ValueError``."""

        if value is None:
            return None
        if self.type == "bool":
            if type(value) is not bool:
                raise ValueError("expected true or false")
            return value
        if self.type == "int":
            if type(value) is not int:
                raise ValueError("expected an integer")
            return self._within_bounds(value)
        if self.type == "number":
            if not isinstance(value, int | float) or type(value) is bool or not isfinite(value):
                raise ValueError("expected a finite number")
            return self._within_bounds(value)
        if type(value) is not str:
            raise ValueError("expected a string")
        if self.type == "string":
            assert self.max_length is not None
            if len(value) > self.max_length:
                raise ValueError(f"longer than {self.max_length} characters")
            return value
        if self.type == "enum":
            assert self.choices is not None
            if value not in {choice.value for choice in self.choices}:
                raise ValueError("not one of the allowed choices")
            return value
        if not SECRET_REF_PATTERN.fullmatch(value):
            raise ValueError("expected the name of an environment variable")
        return value

    def _within_bounds(self, value: int | float) -> int | float:
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"less than the minimum {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"greater than the maximum {self.maximum}")
        return value

    def descriptor(self) -> dict[str, Any]:
        """Return the contract's JSON descriptor."""

        return self.model_dump(mode="json")


def _choices(values: Iterable[str | tuple[str, str] | ManagedSettingChoice]) -> tuple:
    result = []
    for item in values:
        if isinstance(item, ManagedSettingChoice):
            result.append(item)
        elif isinstance(item, tuple):
            result.append(ManagedSettingChoice(value=item[0], label=item[1]))
        else:
            result.append(ManagedSettingChoice(value=item, label=item))
    return tuple(result)


def model_setting(
    *,
    label: str = "Model",
    description: str | None = "The model new sessions use.",
    choices: Iterable[str | tuple[str, str] | ManagedSettingChoice] | None = None,
) -> ManagedSetting:
    """Declare an agent's model; pass it to ``register_agent(settings=...)``.

    New sessions use the value set in Cloud; running and resumed sessions keep
    the model they started with. The default is the agent's own model. Agents
    whose model the Gateway's model policy manages leave it out of the schema.
    """

    if choices is None:
        return ManagedSetting(
            key="model", type="string", max_length=256, label=label, description=description
        )
    return ManagedSetting(
        key="model", type="enum", choices=_choices(choices), label=label, description=description
    )


def thinking_effort_choices(
    efforts: Iterable[str] = _THINKING_EFFORTS,
) -> tuple[ManagedSettingChoice, ...]:
    """Choices for a thinking-effort setting, labelled for people."""

    return tuple(
        ManagedSettingChoice(
            value=effort, label=_THINKING_EFFORT_LABELS.get(effort, effort.capitalize())
        )
        for effort in efforts
    )


def thinking_effort_setting(
    *,
    label: str = "Thinking effort",
    description: str | None = "How much the model reasons before it answers.",
    choices: Iterable[str] = _THINKING_EFFORTS,
) -> ManagedSetting:
    """Declare an agent's thinking effort; it applies when the service restarts."""

    return ManagedSetting(
        key="thinking.effort",
        type="enum",
        choices=thinking_effort_choices(choices),
        applies="restart",
        label=label,
        description=description,
    )


def provider_option_setting(
    name: str,
    *,
    type: Literal["bool", "int", "number", "string", "enum"],
    label: str,
    description: str | None = None,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    max_length: int | None = None,
    choices: Iterable[str | tuple[str, str] | ManagedSettingChoice] | None = None,
) -> ManagedSetting:
    """Declare one of an agent's provider options; it applies when the service restarts."""

    return ManagedSetting(
        key=f"provider_options.{name}",
        type=type,
        applies="restart",
        label=label,
        description=description,
        minimum=minimum,
        maximum=maximum,
        max_length=max_length,
        choices=None if choices is None else _choices(choices),
    )


def namespace_segment(name: str, *, part: str) -> str:
    """Return ``name`` if it can be one segment of a setting key."""

    if not _SEGMENT_PATTERN.fullmatch(name):
        raise ValueError(
            f"{part} {name!r} cannot declare managed settings: its name must use only "
            "lowercase letters, digits, '_' and '-'."
        )
    return name


def namespaced_settings(
    settings: Iterable[ManagedSetting] | None,
    *,
    prefix: str,
    owner: str,
    defaults: Mapping[str, Any] | None = None,
) -> tuple[ManagedSetting, ...]:
    """Return a part's settings with full keys, its owner and its code defaults."""

    if settings is None:
        return ()
    if isinstance(settings, ManagedSetting | str | bytes):
        raise TypeError("settings must be an iterable of ManagedSetting.")
    result: list[ManagedSetting] = []
    for setting in settings:
        if type(setting) is not ManagedSetting:
            raise TypeError("settings must contain only ManagedSetting values.")
        if not _RELATIVE_KEY_PATTERN.fullmatch(setting.key):
            raise ValueError(f"Setting key {setting.key!r} must be relative to its part.")
        if setting.owner is not None and setting.owner != owner:
            raise ValueError(
                f"Setting {setting.key!r} belongs to {owner!r}, not {setting.owner!r}."
            )
        update: dict[str, Any] = {"key": f"{prefix}.{setting.key}", "owner": owner}
        if defaults is not None and setting.key in defaults:
            update["default"] = defaults[setting.key]
        data = {**setting.model_dump(mode="python"), **update}
        try:
            full = ManagedSetting.model_validate(data)
        except ValueError as exc:
            raise ValueError(f"Setting {update['key']!r} is invalid: {exc}") from None
        if not KEY_PATTERN.fullmatch(full.key):
            raise ValueError(f"Setting key {full.key!r} is not a valid managed setting key.")
        result.append(full)
    keys = [item.key for item in result]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{owner} declares the same setting more than once.")
    return tuple(result)


def agent_setting_defaults(
    settings: Sequence[ManagedSetting],
    *,
    model: str,
    thinking_effort: str | None,
    provider_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Check an agent's settings that Cayu applies, and return their code defaults."""

    defaults: dict[str, Any] = {}
    for setting in settings:
        key = setting.key
        if key == "model":
            if setting.type not in {"string", "enum"} or setting.applies == "live":
                raise ValueError(
                    "An agent's 'model' setting is a string or enum that applies at "
                    "next_session or restart."
                )
            defaults[key] = model
        elif key.startswith("thinking."):
            if key != "thinking.effort" or setting.type != "enum":
                raise ValueError("The only thinking setting is the enum 'thinking.effort'.")
            assert setting.choices is not None
            if not {choice.value for choice in setting.choices} <= set(_THINKING_EFFORTS):
                raise ValueError(
                    "thinking.effort choices must be among: " + ", ".join(_THINKING_EFFORTS)
                )
            defaults[key] = thinking_effort
        elif key.startswith("provider_options."):
            name = key.removeprefix("provider_options.")
            if "." in name or setting.type == "secret_ref":
                raise ValueError(f"{key!r} must name one scalar provider option.")
            value = provider_options.get(name)
            defaults[key] = value if type(value) in {bool, int, float, str} else None
        else:
            continue
        if key != "model" and setting.applies != "restart":
            # Execution profiles fingerprint them, so a resumed session would no
            # longer match its profile if they changed between sessions.
            raise ValueError(f"An agent's {key!r} setting applies only at restart.")
    return defaults


def canonical_settings_json(settings: Sequence[Mapping[str, Any]]) -> bytes:
    """Canonical JSON of a ``settings`` list: sorted keys, no spaces, UTF-8."""

    return json.dumps(
        list(settings), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def settings_digest(settings: Sequence[Mapping[str, Any]]) -> str:
    """The schema digest: SHA-256 of the canonical JSON of ``settings``."""

    return "sha256:" + hashlib.sha256(canonical_settings_json(settings)).hexdigest()


class ManagedSettingsLimitExceeded(ValueError):
    """The assembled schema would exceed the contract's limits."""


def managed_settings_schema_document(settings: Iterable[ManagedSetting]) -> dict[str, Any]:
    """Assemble the sorted ``cayu.managed-settings-schema/v1`` document, within its limits."""

    ordered = sorted(settings, key=lambda item: item.key)
    counts = Counter(item.key for item in ordered)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError("Managed settings are declared more than once: " + ", ".join(duplicates))
    missing_owner = [item.key for item in ordered if item.owner is None]
    if missing_owner:
        raise ValueError("Managed settings have no owner: " + ", ".join(missing_owner))
    if len(ordered) > MAX_MANAGED_SETTINGS:
        raise ManagedSettingsLimitExceeded(
            f"The application declares {len(ordered)} managed settings; "
            f"the limit is {MAX_MANAGED_SETTINGS}."
        )
    descriptors = [item.descriptor() for item in ordered]
    document = {
        "contract": MANAGED_SETTINGS_SCHEMA_CONTRACT,
        "digest": settings_digest(descriptors),
        "settings": descriptors,
    }
    size = len(json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    if size > MAX_MANAGED_SETTINGS_SCHEMA_BYTES:
        raise ManagedSettingsLimitExceeded(
            f"The managed settings schema is {size} bytes; "
            f"the limit is {MAX_MANAGED_SETTINGS_SCHEMA_BYTES}."
        )
    return document


# Reading Cloud's values (contract sections 3 and 4).

MANAGED_SETTINGS_VALUES_CONTRACT = "cayu.managed-settings-values/v1"
MANAGED_SETTINGS_REPORT_CONTRACT = "cayu.managed-settings-report/v1"
DEFAULT_POLL_INTERVAL_SECONDS = 25.0
# Cloud treats a process silent for 10 minutes as gone; report well inside that.
_REPORT_HEARTBEAT_SECONDS = 300.0
_BUILD_TIMEOUT_SECONDS = 5.0
_REQUEST_TIMEOUT_SECONDS = 10.0
_MAX_RESPONSE_BYTES = 1024 * 1024
MAX_REPORT_BYTES = 64 * 1024
MAX_REASON_LENGTH = 200
ManagedSettingsRole = Literal["web", "worker"]
ManagedSettingsState = Literal["applied", "stale", "unreported"]

logger = logging.getLogger(__name__)


class ManagedSettingsConflict(BaseModel):
    """A value Cloud holds that doesn't fit this Release's schema."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    key: StrictStr
    reason: StrictStr


class ManagedSettingsSnapshot(BaseModel):
    """One ``cayu.managed-settings-values/v1`` document from Cloud."""

    model_config = ConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True)

    contract: Literal["cayu.managed-settings-values/v1"]
    application_id: StrictStr
    release_id: StrictStr
    schema_digest: StrictStr
    revision: StrictInt = Field(ge=0)
    valid_until: datetime
    values: dict[StrictStr, StrictBool | StrictInt | StrictFloat | StrictStr]
    conflicts: tuple[ManagedSettingsConflict, ...] = ()

    @model_validator(mode="after")
    def _aware(self) -> ManagedSettingsSnapshot:
        if self.valid_until.tzinfo is None:
            raise ValueError("valid_until must carry a time zone.")
        return self


class ManagedSettingsUnavailable(Exception):
    """Cloud could not be reached or gave an unusable answer."""


class ManagedSettingsSchemaMissing(Exception):
    """Cloud has no schema for this Release yet (409 ``settings_schema_missing``)."""


class ManagedSettingsChannel(ABC):
    """How the controller reaches Cloud; ``HttpManagedSettingsChannel`` is Cloud's own."""

    @property
    @abstractmethod
    def release_id(self) -> str: ...

    @abstractmethod
    async def read(self) -> ManagedSettingsSnapshot:
        """Return the effective values, or raise ``ManagedSettingsSchemaMissing`` or
        ``ManagedSettingsUnavailable``."""

    @abstractmethod
    async def report(self, report: Mapping[str, Any]) -> None:
        """Send one ``cayu.managed-settings-report/v1`` document."""

    def read_at_build(self) -> ManagedSettingsSnapshot:
        """Read once, synchronously, while the application is built."""

        raise ManagedSettingsUnavailable("This channel doesn't read at build time.")

    async def aclose(self) -> None:
        return None


def _error_code(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    code = body.get("code")
    error = body.get("error")
    if code is None and isinstance(error, dict):
        code = error.get("code")
    return code if isinstance(code, str) else None


class HttpManagedSettingsChannel(ManagedSettingsChannel):
    """Cloud's settings routes, with the Agent's registry credential.

    Requests ignore proxy variables, never follow redirects and read at most
    1 MiB. Nothing about a failure is logged except that it happened.
    """

    def __init__(
        self,
        *,
        api_url: str,
        application_id: str,
        release_id: str,
        token: str,
        transport: httpx.AsyncBaseTransport | None = None,
        build_transport: httpx.BaseTransport | None = None,
    ) -> None:
        from cayu.cloud_inventory import CloudInventoryConfig

        # Same destination rules as the registry publisher, which uses this credential.
        config = CloudInventoryConfig(
            api_url=api_url.strip().rstrip("/"),
            application_id=application_id,
            release_id=release_id,
            token=SecretStr(token),
        )
        self._config = config
        self._transport = transport
        self._build_transport = build_transport
        self._client: httpx.AsyncClient | None = None

    @property
    def release_id(self) -> str:
        return self._config.release_id

    def _url(self, suffix: str) -> str:
        return (
            f"{self._config.api_url}/v1/applications/"
            f"{quote(self._config.application_id, safe='')}/settings/{suffix}"
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._config.token.get_secret_value()}"}

    def _parse(self, status: int, raw: bytes) -> ManagedSettingsSnapshot:
        try:
            body = json.loads(raw) if len(raw) <= _MAX_RESPONSE_BYTES else None
        except ValueError:
            body = None
        if status == 409 and _error_code(body) == "settings_schema_missing":
            raise ManagedSettingsSchemaMissing
        if not 200 <= status < 300 or body is None:
            raise ManagedSettingsUnavailable(f"Cayu Cloud answered {status}.")
        try:
            snapshot = ManagedSettingsSnapshot.model_validate(body)
        except ValueError:
            raise ManagedSettingsUnavailable("Cayu Cloud sent unusable settings.") from None
        if (
            snapshot.application_id != self._config.application_id
            or snapshot.release_id != self._config.release_id
        ):
            raise ManagedSettingsUnavailable("Cayu Cloud sent another Agent's settings.")
        return snapshot

    def read_at_build(self) -> ManagedSettingsSnapshot:
        with httpx.Client(
            timeout=_BUILD_TIMEOUT_SECONDS, trust_env=False, transport=self._build_transport
        ) as client:
            try:
                with client.stream(
                    "GET",
                    self._url("effective"),
                    params={"release": self._config.release_id},
                    headers=self._headers(),
                    follow_redirects=False,
                ) as response:
                    raw = bytearray()
                    for chunk in response.iter_bytes():
                        raw.extend(chunk)
                        if len(raw) > _MAX_RESPONSE_BYTES:
                            break
            except httpx.HTTPError:
                raise ManagedSettingsUnavailable("Cayu Cloud could not be reached.") from None
        return self._parse(response.status_code, bytes(raw))

    def _async_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=_REQUEST_TIMEOUT_SECONDS, trust_env=False, transport=self._transport
            )
        return self._client

    async def read(self) -> ManagedSettingsSnapshot:
        try:
            async with self._async_client().stream(
                "GET",
                self._url("effective"),
                params={"release": self._config.release_id},
                headers=self._headers(),
                follow_redirects=False,
            ) as response:
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > _MAX_RESPONSE_BYTES:
                        break
        except httpx.HTTPError:
            raise ManagedSettingsUnavailable("Cayu Cloud could not be reached.") from None
        return self._parse(response.status_code, bytes(raw))

    async def report(self, report: Mapping[str, Any]) -> None:
        try:
            response = await self._async_client().post(
                self._url("reports"),
                content=json.dumps(report, separators=(",", ":")).encode(),
                headers={**self._headers(), "Content-Type": "application/json"},
                follow_redirects=False,
            )
        except httpx.HTTPError:
            raise ManagedSettingsUnavailable("Cayu Cloud could not be reached.") from None
        if not response.is_success:
            raise ManagedSettingsUnavailable(f"Cayu Cloud answered {response.status_code}.")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fit_report_settings(
    settings: Sequence[Mapping[str, Any]], *, overhead: int = 512
) -> list[dict[str, Any]]:
    """Keep a report's settings within 500 entries and 64 KiB.

    Settings at their default go first, then reasons are shortened, then the
    list is cut; adopted and refused settings are what Cloud counts.
    """

    entries = [dict(item) for item in settings][:MAX_MANAGED_SETTINGS]

    def size(items: list[dict[str, Any]]) -> int:
        return len(json.dumps(items, separators=(",", ":"), ensure_ascii=False).encode()) + overhead

    if size(entries) <= MAX_REPORT_BYTES:
        return entries
    entries = [item for item in entries if item["outcome"] != "default"]
    for limit in (80, 0):
        if size(entries) <= MAX_REPORT_BYTES:
            return entries
        for item in entries:
            if "reason" in item:
                item["reason"] = item["reason"][:limit] or "refused"
    while entries and size(entries) > MAX_REPORT_BYTES:
        entries.pop()
    return entries


class ManagedSettings:
    """Reads an Agent's settings from Cayu Cloud and reports what this process applied.

    It keeps the last good snapshot: when Cloud can't be reached, the application
    keeps using it (and reports ``stale`` once it is past ``valid_until``); when it
    never got one, the code defaults apply and it reports ``unreported``. The
    application decides what each value does; see ``docs/managed-settings.md``.
    """

    def __init__(
        self,
        channel: ManagedSettingsChannel,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(channel, ManagedSettingsChannel):
            raise TypeError("channel must be a ManagedSettingsChannel.")
        if type(poll_interval) not in {int, float} or not 0 < poll_interval <= 30:
            raise ValueError("poll_interval must be more than 0 and at most 30 seconds.")
        self._channel = channel
        self.poll_interval = float(poll_interval)
        self._clock = clock or (lambda: datetime.now(UTC))
        self.process_id = uuid4().hex
        self.role: ManagedSettingsRole = "web"
        self._snapshot: ManagedSettingsSnapshot | None = None
        self._build_snapshot: ManagedSettingsSnapshot | None = None
        self._built = False
        self._sequence = 0
        self._last_report: tuple[Any, ...] | None = None
        self._last_report_at: datetime | None = None
        self._outcomes: Callable[[], list[dict[str, Any]]] = list
        self._on_snapshot: Callable[[], None] = lambda: None
        self._task: asyncio.Task[None] | None = None
        self.status: Literal["stopped", "ready", "unavailable", "schema_missing"] = "stopped"

    @classmethod
    def from_environment(
        cls, *, release_id: str | None, environ: Mapping[str, str] | None = None
    ) -> ManagedSettings | None:
        """The Cloud controller when Cloud bound this Agent's registry credential, else None."""

        source = os.environ if environ is None else environ
        token = source.get("CAYU_CLOUD_REGISTRY_TOKEN")
        application_id = source.get("CAYU_CLOUD_APPLICATION_ID")
        if not token or not application_id:
            return None
        try:
            channel = HttpManagedSettingsChannel(
                api_url=source.get("CAYU_CLOUD_API_URL", ""),
                application_id=application_id,
                release_id=release_id or "",
                token=token,
            )
        except ValueError:
            logger.warning(
                "Cayu Cloud settings are off: the Cloud API URL, Agent id or Release id is "
                "missing or invalid, so the code's defaults apply."
            )
            return None
        return cls(channel)

    @property
    def release_id(self) -> str:
        return self._channel.release_id

    @property
    def snapshot(self) -> ManagedSettingsSnapshot | None:
        """The last good snapshot; ``next_session`` and ``live`` settings read it."""

        return self._snapshot

    @property
    def build_snapshot(self) -> ManagedSettingsSnapshot | None:
        """The snapshot read when the application was built; ``restart`` settings read it."""

        return self._build_snapshot

    def state(self) -> ManagedSettingsState:
        snapshot = self._snapshot
        if snapshot is None:
            return "unreported"
        return "stale" if self._clock() > snapshot.valid_until else "applied"

    def bind(
        self,
        *,
        outcomes: Callable[[], list[dict[str, Any]]],
        on_snapshot: Callable[[], None],
    ) -> None:
        """Connect the application that applies the values (``CayuApp`` does this)."""

        self._outcomes = outcomes
        self._on_snapshot = on_snapshot

    def read_at_build(self) -> None:
        """Read once while the application is built, for its ``restart`` settings."""

        if self._built:
            return
        self._built = True
        try:
            self._snapshot = self._channel.read_at_build()
        except ManagedSettingsSchemaMissing:
            self.status = "schema_missing"
        except Exception:
            # Never log credentials, response bodies or URLs.
            logger.warning(
                "Cayu Cloud settings could not be read while the app was built; the "
                "code's defaults apply to settings that apply at restart."
            )
        self._build_snapshot = self._snapshot

    async def poll_once(self) -> None:
        """Read Cloud's values, apply a changed snapshot, and report when needed."""

        try:
            snapshot = await self._channel.read()
        except ManagedSettingsSchemaMissing:
            # The Release hasn't published its schema yet; keep what we have.
            self.status = "schema_missing"
        except Exception:
            self.status = "unavailable"
        else:
            previous = self._snapshot
            # valid_until moves on every answer; only a new revision or new values matter.
            changed = previous is None or (previous.revision, previous.values) != (
                snapshot.revision,
                snapshot.values,
            )
            self._snapshot = snapshot
            self.status = "ready"
            if changed:
                self._on_snapshot()
        await self.report_once()

    def report_document(self) -> dict[str, Any]:
        """The ``cayu.managed-settings-report/v1`` document this process would send now."""

        state = self.state()
        snapshot = self._snapshot
        return {
            "contract": MANAGED_SETTINGS_REPORT_CONTRACT,
            "release_id": self._channel.release_id,
            "process": {"id": self.process_id, "role": self.role},
            "sequence": self._sequence + 1,
            "revision": None if snapshot is None else snapshot.revision,
            "state": state,
            "settings": fit_report_settings(self._outcomes()),
            "reported_at": _timestamp(self._clock()),
        }

    async def report_once(self, *, force: bool = False) -> bool:
        """Send a report when what this process applied changed, or as a heartbeat."""

        document = self.report_document()
        key = (document["revision"], document["state"], json.dumps(document["settings"]))
        now = self._clock()
        due = (
            self._last_report_at is None
            or (now - self._last_report_at).total_seconds() >= _REPORT_HEARTBEAT_SECONDS
        )
        if not force and key == self._last_report and not due:
            return False
        try:
            await self._channel.report(document)
        except Exception:
            logger.warning("Cayu Cloud settings report failed; retrying on the next poll.")
            return False
        self._sequence = document["sequence"]
        self._last_report = key
        self._last_report_at = now
        return True

    async def start(self, *, role: ManagedSettingsRole) -> None:
        """Poll now, then every ``poll_interval`` seconds until ``close``."""

        if role not in {"web", "worker"}:
            raise ValueError("role must be 'web' or 'worker'.")
        if self._task is not None:
            return
        self.role = role
        await self.poll_once()
        self._task = asyncio.create_task(self._run(), name="cayu-managed-settings")

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.poll_interval)
            try:
                await self.poll_once()
            except Exception:
                logger.warning("Cayu Cloud settings polling failed; retrying.")

    async def close(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await self._channel.aclose()
        self.status = "stopped"


__all__ = [
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "MANAGED_SETTINGS_REPORT_CONTRACT",
    "MANAGED_SETTINGS_SCHEMA_CONTRACT",
    "MANAGED_SETTINGS_VALUES_CONTRACT",
    "MAX_MANAGED_SETTINGS",
    "MAX_MANAGED_SETTINGS_SCHEMA_BYTES",
    "HttpManagedSettingsChannel",
    "ManagedSetting",
    "ManagedSettingApplies",
    "ManagedSettingChoice",
    "ManagedSettingType",
    "ManagedSettingValue",
    "ManagedSettings",
    "ManagedSettingsChannel",
    "ManagedSettingsConflict",
    "ManagedSettingsLimitExceeded",
    "ManagedSettingsSchemaMissing",
    "ManagedSettingsSnapshot",
    "ManagedSettingsUnavailable",
    "managed_settings_schema_document",
    "model_setting",
    "provider_option_setting",
    "settings_digest",
    "thinking_effort_choices",
    "thinking_effort_setting",
]
