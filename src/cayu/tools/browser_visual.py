"""Application authority and bounded public evidence for visual browser interaction.

These models never contain guest node handles or selectors. A serialized visual
bundle is evidence, not authority to recreate a target in a different allocation.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self, get_args
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import require_durable_clean_nonblank, require_durable_text

VISUAL_OPERATIONS = frozenset({"observe_visual", "click_visual_target", "click_visual_point"})
VISUAL_ACTIONS = VISUAL_OPERATIONS - {"observe_visual"}
VISUAL_POINT_PRECISION = 1_000_000
MAX_VISUAL_TARGETS = 256
MAX_VISUAL_CAPTURES = 256

VisualFailureCode = Literal[
    "visual_mode_disabled",
    "visual_publication_denied",
    "unstable_visual_observation",
    "visual_evidence_expired",
    "unknown_visual_target",
    "visual_viewport_mismatch",
    "visual_page_mismatch",
    "visual_cross_frame_refused",
    "visual_hit_test_mismatch",
    "visual_target_not_actionable",
    "visual_point_outside_viewport",
    "unsupported_visual_surface",
]

VISUAL_FAILURE_CODES = frozenset(get_args(VisualFailureCode))

_Opaque = Annotated[str, Field(min_length=1, max_length=96)]
_Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_Counter = Annotated[int, Field(strict=True, ge=1, le=(1 << 31) - 1)]
_Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class _VisualModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, validate_default=True
    )


class BrowserVisualPolicy(_VisualModel):
    """Explicit application-owned permission to capture and publish browser pixels.

    Retention is session-scoped and application-managed: the application owns
    deletion from the named artifact store. This is not a TTL or secure erasure
    promise. ``publish_to_model`` permits pixels to reach the models selected by
    the application's execution profile, through normal attachment admission.
    """

    artifact_store_id: str = Field(min_length=1, max_length=256)
    allowed_origins: tuple[str, ...] = Field(min_length=1, max_length=64)
    retention: Literal["application_managed"]
    publish_to_model: bool = Field(strict=True)
    allow_coordinate_fallback: bool = Field(default=False, strict=True)
    allowed_profile_contexts: tuple[Literal["fresh_temporary"], ...] = ("fresh_temporary",)
    allow_authenticated_pages: Literal[False] = False
    capture_during_sensitive_takeover: Literal[False] = False
    max_width: int = Field(default=1280, strict=True, ge=1, le=4096)
    max_height: int = Field(default=1024, strict=True, ge=1, le=4096)
    max_pixels: int = Field(default=1_310_720, strict=True, ge=1, le=8_000_000)
    max_bytes: int = Field(default=2 * 1024 * 1024, strict=True, ge=1, le=8 * 1024 * 1024)
    max_targets: int = Field(default=64, strict=True, ge=1, le=MAX_VISUAL_TARGETS)
    max_label_bytes: int = Field(default=128, strict=True, ge=0, le=512)
    max_hit_tests: int = Field(default=256, strict=True, ge=1, le=1024)
    max_frame_depth: int = Field(default=8, strict=True, ge=0, le=16)
    max_processing_ms: int = Field(default=5000, strict=True, ge=1, le=30_000)
    max_captures: int = Field(default=8, strict=True, ge=1, le=MAX_VISUAL_CAPTURES)
    evidence_lifetime_ms: int = Field(default=30_000, strict=True, ge=1, le=120_000)

    @field_validator(
        "capture_during_sensitive_takeover", "allow_authenticated_pages", mode="before"
    )
    @classmethod
    def validate_takeover_exclusion(cls, value: object) -> bool:
        if value is not False:
            raise ValueError(
                "Visual capture in an authenticated profile or sensitive takeover is not supported."
            )
        return False

    @field_validator("allowed_profile_contexts")
    @classmethod
    def validate_profile_contexts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != ("fresh_temporary",):
            raise ValueError(
                "Only the fresh temporary browser profile is admitted for visual capture."
            )
        return value

    @field_validator("artifact_store_id")
    @classmethod
    def validate_store_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "artifact_store_id")

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        origins: set[str] = set()
        for value in values:
            require_durable_clean_nonblank(value, "allowed_origins")
            if len(value.encode("utf-8")) > 2048:
                raise ValueError("Visual origin exceeds its byte limit.")
            parsed = urlsplit(value)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.port not in {None, 443}
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("Visual origins must be HTTPS origins without credentials.")
            origins.add(f"https://{parsed.hostname.lower()}")
        return tuple(sorted(origins))

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.max_width * self.max_height > self.max_pixels:
            raise ValueError("Visual viewport dimensions exceed the pixel limit.")
        if self.max_targets > self.max_hit_tests:
            raise ValueError("Visual target limit exceeds the hit-test limit.")
        return self


class BrowserVisualGeometry(_VisualModel):
    x: _Unit
    y: _Unit
    width: _Unit
    height: _Unit

    @field_validator("x", "y", "width", "height", mode="before")
    @classmethod
    def validate_number(cls, value: object) -> float:
        if (type(value) is not int and type(value) is not float) or not 0 <= value <= 1:
            raise ValueError("Visual geometry must contain finite numbers.")
        return float(value)

    @model_validator(mode="after")
    def validate_rectangle(self) -> Self:
        if (
            self.width <= 0
            or self.height <= 0
            or self.x + self.width > 1.000001
            or self.y + self.height > 1.000001
        ):
            raise ValueError("Visual target must fit inside the captured viewport.")
        return self


class BrowserVisualTarget(_VisualModel):
    ref: Annotated[str, Field(pattern=r"^vt_[0-9a-f]{32}$")]
    geometry: BrowserVisualGeometry
    visible: bool = Field(strict=True)
    occluded: bool = Field(strict=True)
    actionable: bool = Field(strict=True)
    label: str = Field(default="", max_length=512)
    role: str = Field(default="", max_length=128)
    frame_depth: int = Field(default=0, strict=True, ge=0, le=16)

    @field_validator("label", "role")
    @classmethod
    def validate_hint(cls, value: str) -> str:
        value = require_durable_text(value, "visual_hint")
        if len(value.encode("utf-8")) > 512:
            raise ValueError("Visual hint exceeds its byte limit.")
        return value


class BrowserVisualObservation(_VisualModel):
    session_id: _Opaque
    page_id: _Opaque
    page_revision: _Opaque
    control_epoch: _Counter
    worker_instance: Annotated[str, Field(pattern=r"^vw_[0-9a-f]{32}$")]
    visual_revision: Annotated[str, Field(pattern=r"^vr_[0-9a-f]{32}$")]
    screenshot_sha256: _Digest
    viewport_width: int = Field(strict=True, ge=1, le=4096)
    viewport_height: int = Field(strict=True, ge=1, le=4096)
    device_scale: float = Field(gt=0, le=4, allow_inf_nan=False)
    scroll_x: float = Field(ge=0, le=16_777_216, allow_inf_nan=False)
    scroll_y: float = Field(ge=0, le=16_777_216, allow_inf_nan=False)
    targets: tuple[BrowserVisualTarget, ...] = Field(max_length=MAX_VISUAL_TARGETS)
    truncation_reasons: tuple[Literal["targets", "labels", "hit_tests"], ...] = Field(
        default=(), max_length=3
    )
    unsupported_reasons: tuple[
        Literal["cross_origin_frame", "embedded_surface", "closed_shadow_root", "opaque_surface"],
        ...,
    ] = Field(default=(), max_length=4)

    @field_validator("session_id", "page_id", "page_revision")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "visual_identity")

    @field_validator("device_scale", "scroll_x", "scroll_y", mode="before")
    @classmethod
    def validate_number(cls, value: object) -> float:
        if (type(value) is not int and type(value) is not float) or not 0 <= value <= 16_777_216:
            raise ValueError("Visual viewport must contain finite numbers.")
        return float(value)

    @model_validator(mode="after")
    def validate_targets(self) -> Self:
        if len({target.ref for target in self.targets}) != len(self.targets):
            raise ValueError("Visual target refs must be unique.")
        return self


class BrowserVisualAuthority(_VisualModel):
    """Content-free host continuity; the live guest retains the actual target map."""

    visual_revision: Annotated[str, Field(pattern=r"^vr_[0-9a-f]{32}$")]
    screenshot_sha256: _Digest
    worker_instance: Annotated[str, Field(pattern=r"^vw_[0-9a-f]{32}$")]
    refs: tuple[Annotated[str, Field(pattern=r"^vt_[0-9a-f]{32}$")], ...] = Field(
        max_length=MAX_VISUAL_TARGETS
    )

    @classmethod
    def from_observation(cls, value: BrowserVisualObservation) -> Self:
        return cls(
            visual_revision=value.visual_revision,
            screenshot_sha256=value.screenshot_sha256,
            worker_instance=value.worker_instance,
            refs=tuple(sorted(t.ref for t in value.targets)),
        )

    @field_validator("refs")
    @classmethod
    def validate_refs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != tuple(sorted(set(values))):
            raise ValueError("Visual refs must be unique and sorted.")
        return values


def canonical_visual_point(value: object) -> float:
    """One bounded request identity for a normalized, model-selected point."""
    if (type(value) is not int and type(value) is not float) or not 0 <= value < 1:
        raise ValueError("Visual point must be inside the captured viewport.")
    canonical = round(value * VISUAL_POINT_PRECISION) / VISUAL_POINT_PRECISION
    if not 0 <= canonical < 1:
        raise ValueError("Visual point rounds outside the captured viewport.")
    return canonical
