"""Framework-owned project context for zero-code Control Plane assembly."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from cayu._validation import require_clean_nonblank, require_unicode_scalar_text
from cayu.evals.store import EvalStore
from cayu.runtime.app import CayuApp


class ProjectControlPlaneAccess(StrEnum):
    """Bootstrap authority proven by the project-serving boundary."""

    AUTHENTICATED_PRODUCTION = "authenticated_production"
    TRUSTED_LOCAL_DEVELOPMENT = "trusted_local_development"


_PROJECT_CONTEXT_ASSEMBLY_TOKEN = object()


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
            f"access={self._access.value!r})"
        )

    @property
    def project_identity_configured(self) -> bool:
        return self._project_id is not None

    @property
    def eval_store_configured(self) -> bool:
        return self._eval_store is not None

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

    @property
    def trusted_local_development(self) -> bool:
        return self.access is ProjectControlPlaneAccess.TRUSTED_LOCAL_DEVELOPMENT

    def safe_summary(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "application_release_id": self.application_release_id,
            "app_manifest_fingerprint": self.app_manifest_fingerprint,
            "access": self.access.value,
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
) -> ProjectControlPlaneContext:
    return ProjectControlPlaneContext(
        project_root=project_root,
        project_id=project_id,
        configured_release_id=configured_release_id,
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
