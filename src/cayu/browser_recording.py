"""Application-owned browser recording authority and private playback metadata.

Recording is independent of model visual publication, operator viewing, and
profile checkpoints. No recording authority is inferred from any of those.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class _RecordingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class BrowserRecordingPolicy(_RecordingModel):
    """Finite consent for sampled active-page video in fresh Docker browsers.

    The application must explicitly authorize the session scope and origins.
    Cookies, imported profiles, sensitive takeover, child frames and opaque
    surfaces are excluded. Limits stop recording without retrying browser input.
    """

    schema_version: Literal[1] = 1
    scope: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    allowed_origins: tuple[str, ...] = Field(min_length=1, max_length=64)
    allowed_profile_contexts: tuple[Literal["fresh_temporary"], ...] = ("fresh_temporary",)
    allow_authenticated_pages: Literal[False] = False
    capture_during_sensitive_entry: Literal[False] = False
    capture: Literal["active_page_viewport"] = "active_page_viewport"
    max_duration_seconds: int = Field(default=300, strict=True, ge=1, le=3600)
    max_bytes: int = Field(default=32 * 1024 * 1024, strict=True, ge=4096, le=256 * 1024 * 1024)
    max_width: int = Field(default=1280, strict=True, ge=16, le=1920)
    max_height: int = Field(default=720, strict=True, ge=16, le=1080)
    frames_per_second: int = Field(default=2, strict=True, ge=1, le=5)
    retention_seconds: int = Field(strict=True, ge=60, le=30 * 86400)

    @field_validator("allowed_origins")
    @classmethod
    def origins(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        from cayu.tools.browser_visual import BrowserVisualPolicy

        return BrowserVisualPolicy.validate_origins(value)

    @field_validator("allow_authenticated_pages", "capture_during_sensitive_entry", mode="before")
    @classmethod
    def exclusion(cls, value: object) -> bool:
        if value is not False:
            raise ValueError("Authenticated and sensitive recording are unsupported.")
        return False

    @field_validator("allowed_profile_contexts")
    @classmethod
    def contexts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != ("fresh_temporary",):
            raise ValueError("Recording requires a fresh temporary context.")
        return value


class BrowserRecordingConfig(_RecordingModel):
    """Application-only capture credential; excluded from ordinary serialization."""

    policy: BrowserRecordingPolicy
    session_id: str = Field(min_length=1, max_length=128)
    guest_endpoint: str = Field(repr=False)
    credential: SecretStr = Field(repr=False, exclude=True)

    @field_validator("guest_endpoint")
    @classmethod
    def endpoint(cls, value: str) -> str:
        from cayu.tools._browser_control_transport import validate_control_endpoint

        return validate_control_endpoint(value)

    @field_validator("credential")
    @classmethod
    def token(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value()
        if len(raw) != 64 or any(c not in "0123456789abcdef" for c in raw):
            raise ValueError("Recording requires a dedicated 32-byte hexadecimal credential.")
        return value

    def private_guest_configuration(self) -> dict[str, object]:
        """Only for Runtime's private runner transport, never a tool argument."""
        return {
            "policy": self.policy.model_dump(mode="json"),
            "endpoint": self.guest_endpoint,
            "credential": self.credential.get_secret_value(),
        }


class BrowserRecordingCapability(_RecordingModel):
    backend: str
    supported: bool
    reason: Literal["docker_sampled_active_page", "unsupported_backend"]


def browser_recording_capability(backend: str) -> BrowserRecordingCapability:
    return BrowserRecordingCapability(
        backend=backend,
        supported=backend == "docker",
        reason="docker_sampled_active_page" if backend == "docker" else "unsupported_backend",
    )


class BrowserRecordingIdentity(_RecordingModel):
    session_id: str = Field(min_length=1, max_length=128)
    session_instance_id: str = Field(min_length=1, max_length=128)
    allocation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    browser_id: str = Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9_-]+$")
    worker_instance: str = Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9_-]+$")


class BrowserRecordingSegment(_RecordingModel):
    sequence: int = Field(strict=True, ge=0, le=18000)
    page_id: str = Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9_-]+$")
    elapsed_ms: int = Field(strict=True, ge=0, le=3600000)
    duration_ms: int = Field(strict=True, ge=1, le=1000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(strict=True, ge=1)
    media_type: Literal["video/webm"] = "video/webm"


class BrowserRecordingGap(_RecordingModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    reason: Literal["capture_gap", "worker_lost"] = "capture_gap"


class BrowserRecordingManifest(_RecordingModel):
    schema_version: Literal[1] = 1
    recording_id: str
    scope: str
    browser_id: str
    worker_instance: str
    identity: BrowserRecordingIdentity
    gaps: tuple[BrowserRecordingGap, ...] = ()
    status: Literal["recording", "complete", "partial", "unavailable", "failed"]
    reason: Literal[
        "normal_close",
        "capture_gap",
        "limit_exhausted",
        "worker_lost",
        "storage_failure",
        "recording",
    ]
    video_sha256: str | None = None
    video_size_bytes: int | None = None
    media_type: Literal["video/webm"] = "video/webm"
    started_at_ms: int
    expires_at_ms: int
    segments: tuple[BrowserRecordingSegment, ...] = ()
