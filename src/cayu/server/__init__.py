"""Cayu server — FastAPI-based control plane.

Usage::

    import os

    from cayu import CayuApp
    from cayu.server import BasicAuth, ServerConfig, create_server

    app = CayuApp(session_store=..., task_store=..., knowledge_store=...)
    app.register_agent(...)

    server = create_server(
        app,
        config=ServerConfig.protected(
            BasicAuth(username="admin", password=os.environ["CAYU_SERVER_PASSWORD"]),
            deployment_name="production",
        ),
    )

    # Run with: uvicorn my_module:server --port 8000
    # The bundled CAYU runtime dashboard is served at /cayu by default.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from inspect import Parameter, signature
from math import isfinite
from pathlib import Path
from typing import Any

from cayu._validation import require_clean_nonblank, thaw_json_value
from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import IncompleteSessionsRecoveryRequest

try:
    from fastapi import Request  # noqa: TC002 - FastAPI resolves endpoint annotations at runtime
    from starlette.responses import RedirectResponse

    from cayu.server.auth import AuthContext, AuthDependency, BasicAuth
    from cayu.server.config import (
        DEFAULT_EVENT_SIDE_EFFECT_STARTUP_TIMEOUT_SECONDS,
        DEFAULT_INTERRUPTION_SHUTDOWN_GRACE_SECONDS,
        DEFAULT_RECOVERY_INACTIVE_AFTER_SECONDS,
        DEFAULT_REPLAY_IDLE_TIMEOUT_SECONDS,
        AuthenticatedAccess,
        CorsConfig,
        DashboardConfig,
        DocsConfig,
        OpenAccess,
        ServerAccessConfig,
        ServerApiConfig,
        ServerConfig,
        ServerLifecycleConfig,
        auth_dependency_for,
        normalize_api_path,
        normalize_dashboard_path,
    )
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
    "AuthenticatedAccess",
    "BasicAuth",
    "CorsConfig",
    "DashboardConfig",
    "DashboardStaticFiles",
    "DocsConfig",
    "OpenAccess",
    "ServerAccessConfig",
    "ServerApiConfig",
    "ServerConfig",
    "ServerLifecycleConfig",
    "create_router",
    "create_server",
    "event_to_sse_data",
    "mount_cayu",
    "mount_dashboard",
]

logger = logging.getLogger(__name__)

_PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_INTERVAL_SECONDS = 30.0
_PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_BATCH_SIZE = 1000
_ALLOWED_FASTAPI_OPTIONS = frozenset(
    {
        "contact",
        "description",
        "license_info",
        "lifespan",
        "openapi_tags",
        "root_path",
        "root_path_in_servers",
        "servers",
        "summary",
        "terms_of_service",
        "version",
    }
)


def create_server(
    app: CayuApp,
    *,
    config: ServerConfig,
    fastapi_options: Mapping[str, Any] | None = None,
) -> Any:
    """Create a FastAPI server wired to a CayuApp.

    Returns a FastAPI application with standard cayu API routes
    and optional static dashboard serving.

    Args:
        app: The CayuApp instance with registered agents/providers.
        config: Fully resolved server identity and policy. Construct it directly
            after resolving any external settings or secrets. An authenticated
            policy's dependency returns ``AuthContext``; its ``tenant`` value is
            operator-action provenance only and does not filter Cayu data or
            provide tenant isolation.
        fastapi_options: Optional non-policy FastAPI constructor options. Cayu
            validates the option names and retains ownership of its title,
            documentation routes, and lifespan composition.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    if type(config) is not ServerConfig:
        raise TypeError("config must be a resolved ServerConfig instance.")
    resolved_config = config
    api_auth = auth_dependency_for(resolved_config.access)
    dashboard_auth = auth_dependency_for(resolved_config.dashboard.access or resolved_config.access)
    lifecycle = resolved_config.lifecycle
    resolved_fastapi_options = _validate_fastapi_options(FastAPI, fastapi_options)
    if resolved_config.dashboard.enabled:
        _require_dashboard_assets(
            resolved_config.dashboard.directory,
            disable_with="dashboard.enabled=False",
        )
    user_lifespan = resolved_fastapi_options.pop("lifespan", None)
    recovery_statuses = lifecycle.startup_recovery_statuses

    async def recover_startup_state() -> None:
        await _recover_persisted_event_side_effects_during_startup(
            app,
            timeout_s=lifecycle.event_side_effect_startup_timeout_seconds,
        )
        if recovery_statuses is not None:
            await app.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(
                    statuses=set(recovery_statuses),
                    inactive_before=datetime.now(UTC)
                    - timedelta(seconds=lifecycle.recovery_inactive_after_seconds),
                    reason="server_startup_recovery",
                    metadata={"source": "create_server"},
                )
            )
        await app.resume_pending_interruption_cascades(
            interrupting_inactive_before=datetime.now(UTC)
            - timedelta(seconds=lifecycle.recovery_inactive_after_seconds)
        )

    @asynccontextmanager
    async def cayu_lifespan(server):
        side_effect_recovery_task: asyncio.Task[None] | None = None
        if user_lifespan is None:
            try:
                await recover_startup_state()
                side_effect_recovery_task = _start_persisted_event_side_effect_recovery(app)
                yield
            finally:
                await _stop_persisted_event_side_effect_recovery(side_effect_recovery_task)
                await _drain_background_interruptions(
                    app,
                    timeout_s=lifecycle.interruption_shutdown_grace_seconds,
                )
            return
        async with user_lifespan(server) as state:
            try:
                await recover_startup_state()
                side_effect_recovery_task = _start_persisted_event_side_effect_recovery(app)
                yield state
            finally:
                await _stop_persisted_event_side_effect_recovery(side_effect_recovery_task)
                await _drain_background_interruptions(
                    app,
                    timeout_s=lifecycle.interruption_shutdown_grace_seconds,
                )

    resolved_fastapi_options["lifespan"] = cayu_lifespan
    resolved_fastapi_options["debug"] = False
    if resolved_config.docs.enabled:
        resolved_fastapi_options["docs_url"] = "/docs"
        resolved_fastapi_options["redoc_url"] = "/redoc"
        resolved_fastapi_options["openapi_url"] = "/openapi.json"
    else:
        resolved_fastapi_options["docs_url"] = None
        resolved_fastapi_options["redoc_url"] = None
        resolved_fastapi_options["openapi_url"] = None

    server = FastAPI(title=resolved_config.title, **resolved_fastapi_options)
    server.state.cayu_server_config = resolved_config
    server.state.cayu_server_config_summary = resolved_config.safe_summary()

    if resolved_config.cors.allowed_origins:
        server.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_config.cors.allowed_origins),
            allow_methods=list(resolved_config.cors.allow_methods),
            allow_headers=list(resolved_config.cors.allow_headers),
            allow_credentials=resolved_config.cors.allow_credentials,
        )

    session_store = app.session_store
    task_store = app.task_store
    knowledge_store = app.knowledge_store
    control_plane_path = resolved_config.api.path
    if resolved_config.api.enabled:
        router = create_router(
            cayu_app=app,
            session_store=session_store,
            task_store=task_store,
            knowledge_store=knowledge_store,
            knowledge_review_namespace=app.knowledge_review_namespace,
            knowledge_review_labels=app.knowledge_review_labels,
            auth=api_auth,
            api_path=control_plane_path,
            openapi_url=server.openapi_url,
            replay_idle_timeout_s=lifecycle.replay_idle_timeout_s,
        )
        server.include_router(router)

    if resolved_config.dashboard.enabled:
        dashboard_mounted = mount_dashboard(
            server,
            dashboard_dir=resolved_config.dashboard.directory,
            dashboard_path=resolved_config.dashboard.path,
            auth=dashboard_auth,
            api_base_url=control_plane_path,
            dashboard_config=dict(thaw_json_value(resolved_config.dashboard.runtime_config)),
        )
        if not dashboard_mounted:
            raise _dashboard_assets_unavailable_error(
                resolved_config.dashboard.directory,
                disable_with="dashboard.enabled=False",
            )

    return server


def _validate_fastapi_options(
    fastapi_cls: type[Any],
    options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("fastapi_options must be a mapping of FastAPI constructor options.")
    if any(not isinstance(name, str) for name in options):
        raise TypeError("fastapi_options keys must be strings.")

    resolved_options = dict(options)
    disallowed = sorted(resolved_options.keys() - _ALLOWED_FASTAPI_OPTIONS)
    if disallowed:
        raise ValueError(
            "Unsupported or policy-bearing FastAPI options: "
            f"{', '.join(disallowed)}. Use ServerConfig for Cayu policy or "
            "mount_cayu() when the host application must own broader FastAPI behavior."
        )

    supported_options = {
        name
        for name, parameter in signature(fastapi_cls).parameters.items()
        if parameter.kind is not Parameter.VAR_KEYWORD
    }
    unsupported = sorted(resolved_options.keys() - supported_options)
    if unsupported:
        raise ValueError(f"Unsupported FastAPI options: {', '.join(unsupported)}.")
    return resolved_options


def mount_cayu(
    server: Any,
    app: CayuApp,
    *,
    path: str = "/cayu",
    dashboard: bool = True,
    dashboard_dir: str | Path | None = None,
    access: ServerAccessConfig,
    dashboard_config: dict[str, Any] | None = None,
    replay_idle_timeout_s: float = DEFAULT_REPLAY_IDLE_TIMEOUT_SECONDS,
    event_side_effect_startup_timeout_seconds: float = (
        DEFAULT_EVENT_SIDE_EFFECT_STARTUP_TIMEOUT_SECONDS
    ),
    interruption_shutdown_grace_seconds: float = DEFAULT_INTERRUPTION_SHUTDOWN_GRACE_SECONDS,
    interruption_recovery_inactive_after_seconds: int = DEFAULT_RECOVERY_INACTIVE_AFTER_SECONDS,
    name: str = "cayu-dashboard",
) -> None:
    """Mount CAYU's control plane and dashboard into an existing FastAPI app.

    This is the high-level adapter for product apps that already own their
    server. It mounts API routes at ``{path}/api`` and the dashboard shell at
    ``{path}``, so the browser can use one same-origin product path. Its composed
    lifespan recovers persisted event side effects, then cascade parents
    inactive for at least ``interruption_recovery_inactive_after_seconds``, and
    drains accepted background interruption cascades for up to
    ``interruption_shutdown_grace_seconds`` before the host shuts down.

    ``access`` explicitly selects authenticated or deliberately open access to
    the complete mounted surface. ``AuthenticatedAccess`` wraps the existing
    auth dependency contract. Authentication does not make the surface
    tenant-scoped: ``AuthContext.tenant`` is operator-action provenance only,
    and Cayu's built-in reads are not filtered by it. Treat the mount as an
    operator surface unless the host application enforces end-to-end tenant
    authorization and storage isolation.

    ``dashboard_config`` must be a JSON object. Cayu validates and copies it
    before modifying the host application.
    """
    auth = auth_dependency_for(access)
    mount_path = normalize_dashboard_path(path, field_name="path")
    api_path = _join_public_paths(mount_path, "api")
    prepared_dashboard: tuple[str, DashboardStaticFiles] | None = None
    if dashboard:
        prepared_dashboard = _prepare_dashboard_mount(
            dashboard_dir=dashboard_dir,
            dashboard_path=mount_path,
            auth=auth,
            api_base_url=api_path,
            dashboard_config=dashboard_config,
        )
        if prepared_dashboard is None:
            raise _dashboard_assets_unavailable_error(
                dashboard_dir,
                disable_with="dashboard=False",
            )
    event_side_effect_startup_timeout_seconds = _validate_positive_seconds(
        event_side_effect_startup_timeout_seconds,
        "event_side_effect_startup_timeout_seconds",
    )
    interruption_shutdown_grace_seconds = _validate_positive_seconds(
        interruption_shutdown_grace_seconds,
        "interruption_shutdown_grace_seconds",
    )
    if (
        type(interruption_recovery_inactive_after_seconds) is not int
        or interruption_recovery_inactive_after_seconds < 0
    ):
        raise ValueError(
            "interruption_recovery_inactive_after_seconds must be a non-negative integer."
        )
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

    # All caller-controlled values and route construction are validated before
    # changing the host application. A rejected mount must leave it reusable.
    _compose_interruption_drain_lifespan(
        server,
        app,
        timeout_s=interruption_shutdown_grace_seconds,
        recovery_inactive_after_seconds=interruption_recovery_inactive_after_seconds,
        side_effect_startup_timeout_s=event_side_effect_startup_timeout_seconds,
    )
    server.include_router(router)
    if prepared_dashboard is not None:
        prepared_mount_path, dashboard_app = prepared_dashboard
        _attach_dashboard_mount(
            server,
            mount_path=prepared_mount_path,
            dashboard_app=dashboard_app,
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

    ``auth`` protects only the dashboard shell and static assets; the API at
    ``api_base_url`` must have its own equivalent protection. A returned
    ``AuthContext.tenant`` is provenance only and does not filter dashboard or
    API data. Treat this helper as an operator surface unless the host proves
    end-to-end tenant authorization and storage isolation.

    ``dashboard_config`` must be a JSON object. Cayu validates and takes an
    owned copy before registering any dashboard route.

    Returns ``True`` when a dashboard was mounted and ``False`` when mounting
    was disabled, the dashboard directory is absent, or its ``index.html``
    entrypoint is unavailable.
    """
    prepared_dashboard = _prepare_dashboard_mount(
        dashboard_dir=dashboard_dir,
        dashboard_path=dashboard_path,
        auth=auth,
        api_base_url=api_base_url,
        dashboard_config=dashboard_config,
    )
    if prepared_dashboard is None:
        return False

    mount_path, dashboard_app = prepared_dashboard
    _attach_dashboard_mount(
        server,
        mount_path=mount_path,
        dashboard_app=dashboard_app,
        name=name,
    )
    return True


def _prepare_dashboard_mount(
    *,
    dashboard_dir: str | Path | None,
    dashboard_path: str | None,
    auth: AuthDependency | None,
    api_base_url: str,
    dashboard_config: dict[str, Any] | None,
) -> tuple[str, DashboardStaticFiles] | None:
    if dashboard_path is None:
        return None

    mount_path = normalize_dashboard_path(dashboard_path, field_name="dashboard_path")
    api_url = _normalize_api_base_url(api_base_url)
    dist_path = _dashboard_assets_path(dashboard_dir)
    if not dist_path.is_dir() or not (dist_path / "index.html").is_file():
        return None

    dashboard_app = DashboardStaticFiles(
        directory=str(dist_path),
        html=True,
        auth=auth,
        base_path=mount_path,
        api_base_url=api_url,
        dashboard_config=dashboard_config,
    )
    return mount_path, dashboard_app


def _attach_dashboard_mount(
    server: Any,
    *,
    mount_path: str,
    dashboard_app: DashboardStaticFiles,
    name: str,
) -> None:

    if mount_path != "/":

        async def redirect_to_dashboard(request: Request) -> RedirectResponse:
            canonical_url = request.url.replace(path=f"{request.url.path}/")
            return RedirectResponse(canonical_url, status_code=307)

        server.add_api_route(
            mount_path,
            redirect_to_dashboard,
            methods=["GET", "HEAD"],
            include_in_schema=False,
            name=f"{name}-slash-redirect",
        )

    server.mount(
        mount_path,
        dashboard_app,
        name=name,
    )


def _dashboard_assets_path(dashboard_dir: str | Path | None) -> Path:
    if isinstance(dashboard_dir, str):
        dashboard_dir = require_clean_nonblank(dashboard_dir, "dashboard_dir")
    return Path(dashboard_dir) if dashboard_dir is not None else Path(__file__).parent / "dashboard"


def _dashboard_assets_unavailable_error(
    dashboard_dir: str | Path | None,
    *,
    disable_with: str,
) -> RuntimeError:
    return RuntimeError(
        "Dashboard assets are unavailable at "
        f"{_dashboard_assets_path(dashboard_dir)}. The directory must contain index.html. "
        f"Set {disable_with} when the dashboard is intentionally omitted."
    )


def _require_dashboard_assets(
    dashboard_dir: str | Path | None,
    *,
    disable_with: str,
) -> None:
    dashboard_path = _dashboard_assets_path(dashboard_dir)
    if not dashboard_path.is_dir() or not (dashboard_path / "index.html").is_file():
        raise _dashboard_assets_unavailable_error(
            dashboard_dir,
            disable_with=disable_with,
        )


def _validate_positive_seconds(value: float, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a finite positive number.")
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


async def _recover_persisted_event_side_effects_until_idle(app: CayuApp) -> None:
    while True:
        recovered = await app.recover_persisted_event_side_effects(
            limit=_PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_BATCH_SIZE
        )
        # Failed deliveries receive a durable retry deadline and are omitted
        # from the result. Check the store once more before declaring the
        # backlog idle so a full page of poison events cannot strand later work.
        if recovered:
            continue
        remaining = await app.session_store.list_persisted_event_side_effect_deliveries(
            claimable_only=True,
            limit=1,
        )
        if not remaining:
            return


async def _recover_persisted_event_side_effects_during_startup(
    app: CayuApp,
    *,
    timeout_s: float,
) -> None:
    timeout = asyncio.timeout(timeout_s)
    try:
        async with timeout:
            await _recover_persisted_event_side_effects_until_idle(app)
    except TimeoutError:
        if not timeout.expired():
            raise
        logger.warning(
            "Persisted event side-effect startup recovery exceeded %.3fs; "
            "unfinished durable handoffs will remain eligible for background recovery.",
            timeout_s,
        )


async def _recover_persisted_event_side_effects_forever(app: CayuApp) -> None:
    while True:
        await asyncio.sleep(_PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_INTERVAL_SECONDS)
        try:
            await _recover_persisted_event_side_effects_until_idle(app)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to recover persisted event side effects.")


def _start_persisted_event_side_effect_recovery(app: CayuApp) -> asyncio.Task[None]:
    return asyncio.create_task(
        _recover_persisted_event_side_effects_forever(app),
        name="cayu-persisted-event-side-effect-recovery",
    )


async def _stop_persisted_event_side_effect_recovery(
    task: asyncio.Task[None] | None,
) -> None:
    if task is None:
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def _compose_interruption_drain_lifespan(
    server: Any,
    app: CayuApp,
    *,
    timeout_s: float,
    recovery_inactive_after_seconds: int,
    side_effect_startup_timeout_s: float,
) -> None:
    existing_lifespan = server.router.lifespan_context

    @asynccontextmanager
    async def lifespan(server_app):
        async with existing_lifespan(server_app) as state:
            side_effect_recovery_task: asyncio.Task[None] | None = None
            try:
                await _recover_persisted_event_side_effects_during_startup(
                    app,
                    timeout_s=side_effect_startup_timeout_s,
                )
                await app.resume_pending_interruption_cascades(
                    interrupting_inactive_before=datetime.now(UTC)
                    - timedelta(seconds=recovery_inactive_after_seconds)
                )
                side_effect_recovery_task = _start_persisted_event_side_effect_recovery(app)
                yield state
            finally:
                await _stop_persisted_event_side_effect_recovery(side_effect_recovery_task)
                await _drain_background_interruptions(app, timeout_s=timeout_s)

    server.router.lifespan_context = lifespan


def _normalize_api_base_url(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("api_base_url must not be blank.")
    if _is_absolute_url(stripped):
        return stripped.rstrip("/")
    return normalize_api_path(stripped, field_name="api_base_url")


def _join_public_paths(base: str, child: str) -> str:
    suffix = child.strip("/")
    if not suffix:
        return base
    return f"/{suffix}" if base == "/" else f"{base}/{suffix}"


def _is_absolute_url(value: str) -> bool:
    return "://" in value
