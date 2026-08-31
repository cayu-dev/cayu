"""Framework-owned project context for zero-code Control Plane assembly."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from cayu._validation import require_clean_nonblank, require_unicode_scalar_text
from cayu.evals.store import EvalStore
from cayu.runtime.app import CayuApp
from cayu.runtime.costs import PriceBook, copy_price_book

_CANONICAL_POSITIVE_DECIMAL_RE = re.compile(
    r"(?:0|[1-9]\d*)(?:\.\d*[1-9])?\Z",
    re.ASCII,
)


class ProjectControlPlaneAccess(StrEnum):
    """Bootstrap authority proven by the project-serving boundary."""

    AUTHENTICATED_PRODUCTION = "authenticated_production"
    TRUSTED_LOCAL_DEVELOPMENT = "trusted_local_development"


_PROJECT_CONTEXT_ASSEMBLY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class ProjectEvalJudgeConfiguration:
    """Secret-free declarative authority for one generated-project judge."""

    provider_name: str
    model: str
    privacy_policy: Literal["public-only", "public-and-transcript"]
    allow_same_model: bool
    timeout_seconds: int = 120
    max_input_tokens: int = 32_768
    max_output_tokens: int = 4_096
    max_total_tokens: int = 36_864
    max_estimated_cost: str | None = None
    cost_currency: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("provider_name", "model"):
            value = require_clean_nonblank(getattr(self, field_name), field_name)
            require_unicode_scalar_text(value, field_name)
            if len(value) > 256:
                raise ValueError(f"{field_name} cannot exceed 256 characters.")
            object.__setattr__(self, field_name, value)
        if not isinstance(self.privacy_policy, str) or self.privacy_policy not in {
            "public-only",
            "public-and-transcript",
        }:
            raise ValueError("privacy_policy must be public-only or public-and-transcript.")
        if type(self.allow_same_model) is not bool:
            raise TypeError("allow_same_model must be a bool.")
        for field_name, maximum in (
            ("timeout_seconds", 3_600),
            ("max_input_tokens", 1_000_000),
            ("max_output_tokens", 1_000_000),
            ("max_total_tokens", 1_000_000),
        ):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an integer.")
            if value < 1 or value > maximum:
                raise ValueError(f"{field_name} must be between 1 and {maximum}.")
        if self.max_total_tokens < max(self.max_input_tokens, self.max_output_tokens):
            raise ValueError("max_total_tokens cannot be below an individual token ceiling.")
        if (self.max_estimated_cost is None) != (self.cost_currency is None):
            raise ValueError(
                "max_estimated_cost and cost_currency must either both be configured or omitted."
            )
        if self.max_estimated_cost is not None:
            value = require_clean_nonblank(self.max_estimated_cost, "max_estimated_cost")
            require_unicode_scalar_text(value, "max_estimated_cost")
            try:
                decimal_value = Decimal(value)
            except InvalidOperation:
                decimal_value = Decimal(0)
            if (
                len(value) > 64
                or _CANONICAL_POSITIVE_DECIMAL_RE.fullmatch(value) is None
                or not decimal_value.is_finite()
                or decimal_value <= 0
            ):
                raise ValueError(
                    "max_estimated_cost must be a canonical positive decimal with at most "
                    "64 characters."
                )
            object.__setattr__(self, "max_estimated_cost", value)
            currency_source = self.cost_currency
            if currency_source is None:
                raise ValueError("cost_currency is required with max_estimated_cost.")
            currency = require_clean_nonblank(currency_source, "cost_currency")
            require_unicode_scalar_text(currency, "cost_currency")
            if (
                len(currency) > 16
                or not currency[0].isalpha()
                or not currency.isascii()
                or not all(
                    character.isupper() or character.isdigit() or character in "._-"
                    for character in currency
                )
            ):
                raise ValueError("cost_currency must be a portable uppercase identifier.")
            object.__setattr__(self, "cost_currency", currency)


class ProjectControlPlaneContext:
    """Opaque project authority assembled by ``cayu serve`` and ``cayu check``.

    Application factories may pass this object through to Cayu's maintained
    service assembler, but cannot construct or rewrite it. Runtime objects and
    filesystem locations are deliberately absent from its representation.
    """

    __slots__ = (
        "_access",
        "_assembly_token",
        "_close_lock",
        "_closed",
        "_configured_release_id",
        "_eval_judge_configuration",
        "_eval_price_book",
        "_eval_store",
        "_project_id",
        "_project_root",
        "_store_backend",
        "_store_source",
    )

    def __init__(
        self,
        *,
        project_root: Path,
        project_id: str | None,
        configured_release_id: str | None,
        eval_judge_configuration: ProjectEvalJudgeConfiguration | None,
        eval_price_book: PriceBook | None,
        eval_store: EvalStore | None,
        store_backend: Literal["sqlite", "postgres"] | None,
        store_source: str | None,
        access: ProjectControlPlaneAccess,
        _assembly_token: object | None = None,
    ) -> None:
        if _assembly_token is not _PROJECT_CONTEXT_ASSEMBLY_TOKEN:
            raise TypeError(
                "ProjectControlPlaneContext instances are created only by Cayu project bootstrap."
            )
        if not isinstance(project_root, Path) or not project_root.is_absolute():
            raise TypeError("project_root must be an absolute Path.")
        if project_id is not None:
            project_id = require_clean_nonblank(project_id, "project_id")
            require_unicode_scalar_text(project_id, "project_id")
        if configured_release_id is not None:
            configured_release_id = require_clean_nonblank(
                configured_release_id,
                "configured_release_id",
            )
            require_unicode_scalar_text(configured_release_id, "configured_release_id")
        if (
            eval_judge_configuration is not None
            and type(eval_judge_configuration) is not ProjectEvalJudgeConfiguration
        ):
            raise TypeError(
                "eval_judge_configuration must be an exact ProjectEvalJudgeConfiguration."
            )
        if eval_price_book is not None and type(eval_price_book) is not PriceBook:
            raise TypeError("eval_price_book must be an exact PriceBook or None.")
        if eval_store is not None:
            if not isinstance(eval_store, EvalStore) or not eval_store.durable:
                raise TypeError("eval_store must be a durable EvalStore.")
            if store_backend not in {"sqlite", "postgres"} or store_source is None:
                raise ValueError("A configured EvalStore requires backend and source evidence.")
        elif store_backend is not None or store_source is not None:
            raise ValueError("Store evidence requires a configured EvalStore.")
        if store_source is not None:
            store_source = require_clean_nonblank(store_source, "store_source")
        if not isinstance(access, ProjectControlPlaneAccess):
            raise TypeError("access must be a ProjectControlPlaneAccess.")

        self._project_root = project_root
        self._project_id = project_id
        self._configured_release_id = configured_release_id
        self._eval_judge_configuration = eval_judge_configuration
        self._eval_price_book = (
            None if eval_price_book is None else copy_price_book(eval_price_book)
        )
        self._eval_store = eval_store
        self._store_backend = store_backend
        self._store_source = store_source
        self._access = access
        self._assembly_token = _assembly_token
        self._close_lock = threading.Lock()
        self._closed = False

    def __repr__(self) -> str:
        return (
            "ProjectControlPlaneContext("
            f"project_identity_configured={self.project_identity_configured!r}, "
            f"eval_store_configured={self.eval_store_configured!r}, "
            f"eval_judge_configured={self.eval_judge_configured!r}, "
            f"eval_pricing_configured={self.eval_pricing_configured!r}, "
            f"access={self._access.value!r})"
        )

    @property
    def project_identity_configured(self) -> bool:
        return self._project_id is not None

    @property
    def eval_store_configured(self) -> bool:
        return self._eval_store is not None

    @property
    def eval_judge_configured(self) -> bool:
        return self._eval_judge_configuration is not None

    @property
    def eval_pricing_configured(self) -> bool:
        return self._eval_price_book is not None

    @property
    def access(self) -> ProjectControlPlaneAccess:
        return self._access

    async def close(self) -> None:
        """Close the framework-owned store at most once."""

        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        if self._eval_store is not None:
            await self._eval_store.close()

    def _resolve(self, app: CayuApp) -> ResolvedProjectControlPlaneContext:
        if not isinstance(app, CayuApp):
            raise TypeError("Project Control Plane context requires a CayuApp.")
        with self._close_lock:
            if self._closed:
                raise RuntimeError("Project Control Plane context is already closed.")
        manifest = app.describe(project_root=self._project_root)
        release_id = self._configured_release_id or f"manifest-{manifest.fingerprint}"
        public_identity: dict[str, str] = {
            "application_release_id": release_id,
            "app_manifest_fingerprint": manifest.fingerprint,
        }
        if self._project_id is not None:
            public_identity["project_id"] = self._project_id
        try:
            redacted_identity = app.redact_json(public_identity)
        except Exception as exc:
            raise ValueError(
                "Project Control Plane identity could not cross the application redaction boundary."
            ) from exc
        if redacted_identity != public_identity:
            raise ValueError("Project Control Plane identity contains a workload secret.")
        return ResolvedProjectControlPlaneContext(
            project_id=self._project_id,
            application_release_id=release_id,
            app_manifest_fingerprint=manifest.fingerprint,
            app_manifest_project_root=self._project_root,
            eval_judge_configuration=self._eval_judge_configuration,
            eval_price_book=(
                None if self._eval_price_book is None else copy_price_book(self._eval_price_book)
            ),
            eval_store=self._eval_store,
            store_backend=self._store_backend,
            store_source=self._store_source,
            access=self._access,
            owner=self,
            _assembly_token=_PROJECT_CONTEXT_ASSEMBLY_TOKEN,
        )


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedProjectControlPlaneContext:
    """Application-bound project context retained only inside server assembly."""

    project_id: str | None
    application_release_id: str
    app_manifest_fingerprint: str
    app_manifest_project_root: Path
    eval_judge_configuration: ProjectEvalJudgeConfiguration | None
    eval_price_book: PriceBook | None
    eval_store: EvalStore | None
    store_backend: Literal["sqlite", "postgres"] | None
    store_source: str | None
    access: ProjectControlPlaneAccess
    owner: ProjectControlPlaneContext
    _assembly_token: object

    def __post_init__(self) -> None:
        if self._assembly_token is not _PROJECT_CONTEXT_ASSEMBLY_TOKEN:
            raise TypeError("Resolved project context must originate from Cayu project bootstrap.")
        if (
            not isinstance(self.app_manifest_project_root, Path)
            or not self.app_manifest_project_root.is_absolute()
        ):
            raise TypeError("app_manifest_project_root must be an absolute Path.")
        if self.eval_price_book is not None and type(self.eval_price_book) is not PriceBook:
            raise TypeError("eval_price_book must be an exact PriceBook or None.")

    @property
    def trusted_local_development(self) -> bool:
        return self.access is ProjectControlPlaneAccess.TRUSTED_LOCAL_DEVELOPMENT

    def safe_summary(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "application_release_id": self.application_release_id,
            "app_manifest_fingerprint": self.app_manifest_fingerprint,
            "access": self.access.value,
            "eval_judge": {"configured": self.eval_judge_configuration is not None},
            "eval_pricing": {"configured": self.eval_price_book is not None},
            "eval_store": {
                "configured": self.eval_store is not None,
                "backend": self.store_backend,
                "source": self.store_source,
            },
        }


def _create_project_control_plane_context(
    *,
    project_root: Path,
    project_id: str | None,
    configured_release_id: str | None,
    eval_store: EvalStore | None,
    store_backend: Literal["sqlite", "postgres"] | None,
    store_source: str | None,
    access: ProjectControlPlaneAccess,
    eval_judge_configuration: ProjectEvalJudgeConfiguration | None = None,
    eval_price_book: PriceBook | None = None,
) -> ProjectControlPlaneContext:
    return ProjectControlPlaneContext(
        project_root=project_root,
        project_id=project_id,
        configured_release_id=configured_release_id,
        eval_judge_configuration=eval_judge_configuration,
        eval_price_book=eval_price_book,
        eval_store=eval_store,
        store_backend=store_backend,
        store_source=store_source,
        access=access,
        _assembly_token=_PROJECT_CONTEXT_ASSEMBLY_TOKEN,
    )


def resolve_project_control_plane_context(
    context: ProjectControlPlaneContext | None,
    app: CayuApp,
) -> ResolvedProjectControlPlaneContext | None:
    if context is None:
        return None
    if type(context) is not ProjectControlPlaneContext:
        raise TypeError("project_context must be a framework-owned ProjectControlPlaneContext.")
    return context._resolve(app)
