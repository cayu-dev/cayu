"""Cayu server — FastAPI-based control plane.

Usage::

    from cayu import CayuApp
    from cayu.server import create_server

    app = CayuApp(session_store=..., task_store=..., knowledge_store=...)
    app.register_agent(...)

    server = create_server(app)

    # Run with: uvicorn my_module:server --port 8000
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cayu.runtime.app import CayuApp
from cayu.server.routes import create_router
from cayu.server.sse import event_to_sse_data
from cayu.server.static import DashboardStaticFiles

__all__ = [
    "create_server",
    "event_to_sse_data",
]


def create_server(
    app: CayuApp,
    *,
    title: str = "Cayu",
    cors_origins: list[str] | None = None,
    dashboard_dir: str | Path | None = None,
    **fastapi_kwargs: Any,
) -> Any:
    """Create a FastAPI server wired to a CayuApp.

    Returns a FastAPI application with standard cayu API routes
    and optional static dashboard serving.

    Args:
        app: The CayuApp instance with registered agents/providers.
        title: FastAPI app title.
        cors_origins: Allowed CORS origins. Defaults to localhost:5173 (Vite dev).
        dashboard_dir: Path to pre-built dashboard static files. Served at ``/``.
        **fastapi_kwargs: Additional kwargs passed to FastAPI().
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    server = FastAPI(title=title, **fastapi_kwargs)

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

    router = create_router(
        cayu_app=app,
        session_store=session_store,
        task_store=task_store,
        knowledge_store=knowledge_store,
        knowledge_review_namespace=app.knowledge_review_namespace,
        knowledge_review_labels=app.knowledge_review_labels,
    )
    server.include_router(router)

    # Serve pre-built dashboard if available
    if dashboard_dir is not None:
        dist_path = Path(dashboard_dir)
        if dist_path.exists():
            server.mount(
                "/",
                DashboardStaticFiles(directory=str(dist_path), html=True),
                name="dashboard",
            )
    else:
        builtin_dashboard = Path(__file__).parent / "dashboard"
        if builtin_dashboard.exists():
            server.mount(
                "/",
                DashboardStaticFiles(directory=str(builtin_dashboard), html=True),
                name="dashboard",
            )

    return server
