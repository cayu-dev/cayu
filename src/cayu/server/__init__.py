"""Cayu server — FastAPI-based control plane.

Usage::

    import os

    from cayu import CayuApp
    from cayu.server import BasicAuth, create_server

    app = CayuApp(session_store=..., task_store=..., knowledge_store=...)
    app.register_agent(...)

    server = create_server(
        app,
        auth=BasicAuth(username="admin", password=os.environ["CAYU_SERVER_PASSWORD"]),
    )

    # Run with: uvicorn my_module:server --port 8000
    # The bundled CAYU runtime dashboard is served at /cayu by default.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Any

from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import IncompleteSessionsRecoveryRequest, SessionStatus

try:
    from cayu.server.auth import AuthContext, AuthDependency, BasicAuth
    from cayu.server.contracts import SERVER_API_PREFIX
    from cayu.server.routes import create_router
    from cayu.server.sse import event_to_sse_data
    from cayu.server.static import DashboardStaticFiles
except ModuleNotFoundError as exc:
    # Only remap to the optional-dependency hint when one of the server extra's
    # packages is absent. A deeper/unrelated missing module (a corrupt install,
    # a missing transitive dep) must surface as its own error instead of being
    # masked.
    if (exc.name or "").partition(".")[0] not in {
        "fastapi",
        "starlette",
        "sse_starlette",
        "uvicorn",
    }:
        raise
    raise RuntimeError(
        "Cayu's server requires the optional server packages. "
        'Install them with `pip install "cayu[server]"`.'
    ) from exc

__all__ = [
    "AuthContext",
    "AuthDependency",
    "BasicAuth",
    "DashboardStaticFiles",
    "create_router",
    "create_server",
    "event_to_sse_data",
    "mount_cayu",
    "mount_dashboard",
]

logger = logging.getLogger(__name__)


def create_server(
    app: CayuApp,
    *,
    title: str = "Cayu",
    cors_origins: list[str] | None = None,
    dashboard_dir: str | Path | None = None,
    dashboard_path: str | None = "/cayu",
    api_path: str = SERVER_API_PREFIX,
    auth: AuthDependency | None = None,
    dev: bool = False,
    expose_docs: bool | None = None,
    dashboard_config: dict[str, Any] | None = None,
    replay_idle_timeout_s: float = 300.0,
    startup_recovery_statuses: set[SessionStatus] | None = None,
    recovery_inactive_after_seconds: int = 300,
    interruption_shutdown_grace_seconds: float = 10.0,
    **fastapi_kwargs: Any,
) -> Any:
    """Create a FastAPI server wired to a CayuApp.

    Returns a FastAPI application with standard cayu API routes
    and optional static dashboard serving.

    Args:
        app: The CayuApp instance with registered agents/providers.
        title: FastAPI app title.
        cors_origins: Allowed CORS origins. Defaults to localhost:5173 (Vite dev).
        dashboard_dir: Path to pre-built dashboard static files. Defaults to the
            bundled CAYU dashboard build.
        dashboard_path: URL path for the bundled dashboard. Defaults to
            ``/cayu``. Pass ``None`` to disable dashboard mounting.
        api_path: URL path prefix for the CAYU control plane. Defaults to
            ``/api``. Use ``/cayu/api`` when embedding dashboard and API under
            one product path.
        auth: FastAPI-compatible dependency guarding the CAYU control plane. It
            must accept ``Request`` and return ``AuthContext``; raise
            ``fastapi.HTTPException`` (401/403) inside it to reject a request.
            It protects every control-plane route that can start, change,
            inspect, or reveal runtime state; only the health route stays open
            for load balancers.
        dev: Explicit opt-in for unauthenticated local/dev servers. Without
            ``auth`` or ``dev=True``, ``create_server`` raises instead of
            accidentally exposing the control plane.
        expose_docs: Whether to expose FastAPI's generated ``/openapi.json``,
            ``/docs``, and ``/redoc`` routes. Defaults to ``True`` in dev mode
            and ``False`` in protected deployments. Set this explicitly for
            production deployments that intentionally expose generated docs.
        dashboard_config: Optional JSON-serializable runtime config injected
            into the bundled dashboard shell. ``basePath`` and ``apiBaseUrl``
            are owned by the server mount and cannot be overridden here.
        replay_idle_timeout_s: Maximum time a replay stream waits without a
            persisted event before emitting an error and closing.
        startup_recovery_statuses: Explicit statuses for a bounded recovery sweep
            during server startup. ``None`` disables it. Include ``pending`` or
            ``running`` only when the deployment boundary proves they are abandoned.
        recovery_inactive_after_seconds: Minimum inactivity required before the
            startup sweep atomically fences and recovers a session.
        interruption_shutdown_grace_seconds: Maximum server-shutdown wait for
            accepted background interruption cascades.
        **fastapi_kwargs: Additional kwargs passed to FastAPI().
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    if auth is None and not dev:
        raise ValueError(
            "create_server requires auth=... for protected deployments. "
            "Pass dev=True only for local development or tests."
        )
    if type(recovery_inactive_after_seconds) is not int or recovery_inactive_after_seconds < 0:
        raise ValueError("recovery_inactive_after_seconds must be a non-negative integer.")
    interruption_shutdown_grace_seconds = _validate_interruption_shutdown_grace_seconds(
        interruption_shutdown_grace_seconds
    )

    docs_enabled = dev if expose_docs is None else expose_docs
    fastapi_options = dict(fastapi_kwargs)
    user_lifespan = fastapi_options.pop("lifespan", None)
    recovery_statuses = None
    if startup_recovery_statuses is not None:
        recovery_statuses = IncompleteSessionsRecoveryRequest(
            statuses=startup_recovery_statuses
        ).statuses

    async def recover_incomplete_sessions() -> None:
        if recovery_statuses is not None:
            await app.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(
                    statuses=recovery_statuses,
                    inactive_before=datetime.now(UTC)
                    - timedelta(seconds=recovery_inactive_after_seconds),
                    reason="server_startup_recovery",
                    metadata={"source": "create_server"},
                )
            )
        await app.resume_pending_interruption_cascades(
            interrupting_inactive_before=datetime.now(UTC)
            - timedelta(seconds=recovery_inactive_after_seconds)
        )

    @asynccontextmanager
    async def cayu_lifespan(server):
        if user_lifespan is None:
            try:
                await recover_incomplete_sessions()
                yield
            finally:
                await _drain_background_interruptions(
                    app,
                    timeout_s=interruption_shutdown_grace_seconds,
                )
            return
        async with user_lifespan(server) as state:
            try:
                await recover_incomplete_sessions()
                yield state
            finally:
                await _drain_background_interruptions(
                    app,
                    timeout_s=interruption_shutdown_grace_seconds,
                )

    fastapi_options["lifespan"] = cayu_lifespan
    if docs_enabled:
        fastapi_options.setdefault("docs_url", "/docs")
        fastapi_options.setdefault("redoc_url", "/redoc")
        fastapi_options.setdefault("openapi_url", "/openapi.json")
    else:
        fastapi_options["docs_url"] = None
        fastapi_options["redoc_url"] = None
        fastapi_options["openapi_url"] = None

    server = FastAPI(title=title, **fastapi_options)

    origins = cors_origins or ["http://localhost:5173"]
    server.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    session_store = app.session_store
    task_store = app.task_store
    knowledge_store = app.knowledge_store
    control_plane_path = _normalize_api_path(api_path)
    dashboard_mount_path = (
        _normalize_dashboard_path(dashboard_path, api_path=control_plane_path)
        if dashboard_path is not None
        else None
    )

    router = create_router(
        cayu_app=app,
        session_store=session_store,
        task_store=task_store,
        knowledge_store=knowledge_store,
        knowledge_review_namespace=app.knowledge_review_namespace,
        knowledge_review_labels=app.knowledge_review_labels,
        auth=auth,
        api_path=control_plane_path,
        openapi_url=server.openapi_url,
        replay_idle_timeout_s=replay_idle_timeout_s,
    )
    server.include_router(router)

    mount_dashboard(
        server,
        dashboard_dir=dashboard_dir,
        dashboard_path=dashboard_mount_path,
        auth=auth,
        api_base_url=control_plane_path,
        dashboard_config=dashboard_config,
    )

    return server


def mount_cayu(
    server: Any,
    app: CayuApp,
    *,
    path: str = "/cayu",
    dashboard: bool = True,
    dashboard_dir: str | Path | None = None,
    auth: AuthDependency | None = None,
    dev: bool = False,
    dashboard_config: dict[str, Any] | None = None,
    replay_idle_timeout_s: float = 300.0,
    interruption_shutdown_grace_seconds: float = 10.0,
    interruption_recovery_inactive_after_seconds: int = 300,
    name: str = "cayu-dashboard",
) -> None:
    """Mount CAYU's control plane and dashboard into an existing FastAPI app.

    This is the high-level adapter for product apps that already own their
    server. It mounts API routes at ``{path}/api`` and the dashboard shell at
    ``{path}``, so the browser can use one same-origin product path. Its
    composed lifespan recovers cascade parents inactive for at least
    ``interruption_recovery_inactive_after_seconds`` and drains accepted
    background interruption cascades for up to
    ``interruption_shutdown_grace_seconds`` before the host shuts down.
    """
    if auth is None and not dev:
        raise ValueError(
            "mount_cayu requires auth=... for protected deployments. "
            "Pass dev=True only for local development or tests."
        )
    interruption_shutdown_grace_seconds = _validate_interruption_shutdown_grace_seconds(
        interruption_shutdown_grace_seconds
    )
    if (
        type(interruption_recovery_inactive_after_seconds) is not int
        or interruption_recovery_inactive_after_seconds < 0
    ):
        raise ValueError(
            "interruption_recovery_inactive_after_seconds must be a non-negative integer."
        )
    _compose_interruption_drain_lifespan(
        server,
        app,
        timeout_s=interruption_shutdown_grace_seconds,
        recovery_inactive_after_seconds=interruption_recovery_inactive_after_seconds,
    )

    mount_path = _normalize_dashboard_path(path, api_path=None)
    api_path = _join_public_paths(mount_path, "api")
    router = create_router(
        cayu_app=app,
        session_store=app.session_store,
        task_store=app.task_store,
        knowledge_store=app.knowledge_store,
        knowledge_review_namespace=app.knowledge_review_namespace,
        knowledge_review_labels=app.knowledge_review_labels,
        auth=auth,
        api_path=api_path,
        openapi_url=getattr(server, "openapi_url", None),
        replay_idle_timeout_s=replay_idle_timeout_s,
    )
    server.include_router(router)
    if dashboard:
        mount_dashboard(
            server,
            dashboard_dir=dashboard_dir,
            dashboard_path=mount_path,
            auth=auth,
            api_base_url=api_path,
            dashboard_config=dashboard_config,
            name=name,
        )


def mount_dashboard(
    server: Any,
    *,
    dashboard_dir: str | Path | None = None,
    dashboard_path: str | None = "/cayu",
    auth: AuthDependency | None = None,
    api_base_url: str = SERVER_API_PREFIX,
    dashboard_config: dict[str, Any] | None = None,
    name: str = "dashboard",
) -> bool:
    """Mount the built CAYU dashboard on a FastAPI app.

    Product apps that compose CAYU with ``create_router`` can call this helper
    after registering their API routes. It serves one subpath-safe dashboard
    build under ``dashboard_path`` and injects runtime config so the dashboard
    uses the selected mount path.

    Returns ``True`` when a dashboard was mounted and ``False`` when mounting
    was disabled or the dashboard directory is absent.
    """
    if dashboard_path is None:
        return False

    mount_path = _normalize_dashboard_path(dashboard_path, api_path=None)
    api_url = _normalize_api_base_url(api_base_url)
    dist_path = (
        Path(dashboard_dir) if dashboard_dir is not None else Path(__file__).parent / "dashboard"
    )
    if not dist_path.exists():
        return False

    server.mount(
        mount_path,
        DashboardStaticFiles(
            directory=str(dist_path),
            html=True,
            auth=auth,
            base_path=mount_path,
            api_base_url=api_url,
            dashboard_config=dashboard_config,
        ),
        name=name,
    )
    return True


def _validate_interruption_shutdown_grace_seconds(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError("interruption_shutdown_grace_seconds must be a finite positive number.")
    return float(value)


async def _drain_background_interruptions(app: CayuApp, *, timeout_s: float) -> None:
    try:
        drained = await app.drain_background_interruptions(timeout_s=timeout_s)
    except Exception:
        logger.exception("Failed to drain background interruption cascades during shutdown.")
        return
    if not drained:
        logger.warning(
            "Background interruption cascades exceeded the %.3fs shutdown grace period.",
            timeout_s,
        )


def _compose_interruption_drain_lifespan(
    server: Any,
    app: CayuApp,
    *,
    timeout_s: float,
    recovery_inactive_after_seconds: int,
) -> None:
    existing_lifespan = server.router.lifespan_context

    @asynccontextmanager
    async def lifespan(server_app):
        async with existing_lifespan(server_app) as state:
            try:
                await app.resume_pending_interruption_cascades(
                    interrupting_inactive_before=datetime.now(UTC)
                    - timedelta(seconds=recovery_inactive_after_seconds)
                )
                yield state
            finally:
                await _drain_background_interruptions(app, timeout_s=timeout_s)

    server.router.lifespan_context = lifespan


def _normalize_dashboard_path(path: str, *, api_path: str | None) -> str:
    value = path.strip()
    if not value:
        raise ValueError("dashboard_path must not be blank.")
    if "?" in value or "#" in value:
        raise ValueError("dashboard_path must be a URL path, not a URL.")
    value = "/" + value.strip("/")
    normalized = "/" if value == "/" else value
    if api_path is not None and _is_same_or_child_path(normalized, api_path):
        raise ValueError("dashboard_path must not live inside the CAYU api_path.")
    return normalized


def _normalize_api_path(path: str) -> str:
    value = path.strip()
    if not value:
        raise ValueError("api_path must not be blank.")
    if "?" in value or "#" in value or "://" in value:
        raise ValueError("api_path must be a URL path, not a URL.")
    value = "/" + value.strip("/")
    if value == "/":
        raise ValueError("api_path must not be the site root.")
    return value


def _normalize_api_base_url(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("api_base_url must not be blank.")
    if _is_absolute_url(stripped):
        return stripped.rstrip("/")
    return _normalize_api_path(stripped)


def _join_public_paths(base: str, child: str) -> str:
    suffix = child.strip("/")
    if not suffix:
        return base
    return f"/{suffix}" if base == "/" else f"{base}/{suffix}"


def _is_same_or_child_path(path: str, parent: str) -> bool:
    if path == parent:
        return True
    if parent == "/":
        return True
    return path.startswith(f"{parent}/")


def _is_absolute_url(value: str) -> bool:
    return "://" in value
