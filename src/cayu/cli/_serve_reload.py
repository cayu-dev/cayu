"""Uvicorn reload supervision without constructing a project in the parent."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from socket import socket
from typing import TYPE_CHECKING, Any

from uvicorn import Server
from uvicorn.supervisors.watchfilesreload import WatchFilesReload

from cayu.cli.project import project_context
from cayu.cli.project_control_plane import build_project_control_plane_context
from cayu.cli.serve import _build_serve_app, _resolve_serve_project, parse_serve_arguments

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

_REQUEST_DRAIN_SECONDS = 5
_RELOAD_ENV = "_CAYU_SERVE_RELOAD"
_RELOAD_WARNING = (
    "WARNING: cayu serve --reload is for development. On every restart, "
    "in-memory state is lost and active runs can be interrupted. "
    "Persistent stores retain data, but do not keep active runs uninterrupted."
)


def run_reload(uvicorn: Any, args: argparse.Namespace, root: Path) -> None:
    """Pass only CLI options and the project root to fresh spawned workers."""
    arguments = ["--host", args.host, "--port", str(args.port)]
    if args.dev:
        arguments.append("--dev")
    elif args.auth is not None:
        arguments.extend(("--auth", args.auth))
    previous = os.environ.get(_RELOAD_ENV)
    os.environ[_RELOAD_ENV] = json.dumps({"root": str(root), "arguments": arguments})
    try:
        config = uvicorn.Config(
            "cayu.cli._serve_reload:create_reload_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[str(root)],
            lifespan="on",
            timeout_graceful_shutdown=_REQUEST_DRAIN_SECONDS,
        )
        server = _ReloadServer(config)
        with config.bind_socket() as listener:
            WatchFilesReload(config, target=server.run, sockets=[listener]).run()
    finally:
        if previous is None:
            os.environ.pop(_RELOAD_ENV, None)
        else:
            os.environ[_RELOAD_ENV] = previous


class _ReloadServer(Server):
    def run(self, sockets: list[socket] | None = None) -> None:
        # This target runs only in the spawned child, before Server.run starts
        # an event loop. Synchronous project factories may use asyncio.run.
        try:
            app = create_reload_app()
        except (Exception, SystemExit):
            logging.getLogger("uvicorn.error").exception("Application startup failed.")
            return
        self.config.app = app
        self.config.factory = False
        try:
            super().run(sockets=sockets)
        except BaseException as primary:
            try:
                app.close_sync()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Reload server and project storage cleanup failed", [primary, cleanup]
                ) from None
            raise
        else:
            # Covers configuration failures before ASGI lifespan starts too.
            app.close_sync()


def create_reload_app() -> _ReloadApp:
    """Construct a worker synchronously, transferring cleanup to its lifespan."""
    snapshot = json.loads(os.environ[_RELOAD_ENV])
    return _ReloadApp(Path(snapshot["root"]), parse_serve_arguments(snapshot["arguments"]))


class _ReloadApp:
    def __init__(self, root: Path, args: argparse.Namespace) -> None:
        self._scope = ExitStack()
        self._context = None
        self._server: ASGIApp | None = None
        print(_RELOAD_WARNING, file=sys.stderr, flush=True)
        try:
            project, settings = _resolve_serve_project(args, start=root)
            self._scope.enter_context(project_context(project.root))
            self._context = build_project_control_plane_context(
                project.root, mode="development" if args.dev else "production"
            )
            self._server = _build_serve_app(args, project, settings, self._context)
        except BaseException as primary:
            try:
                self.close_sync()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Reload worker and project storage cleanup failed", [primary, cleanup]
                ) from None
            raise

    def close_sync(self) -> None:
        try:
            if self._context is not None:
                asyncio.run(self._context.close())
        finally:
            self._scope.close()
            self._server = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "lifespan":
            if self._server is None:
                raise RuntimeError("Reload worker has not completed startup.")
            await self._server(scope, receive, send)
            return

        assert self._context is not None
        assert self._server is not None
        context = self._context
        terminal: Message | None = None

        async def send_lifespan(message: Message) -> None:
            nonlocal terminal
            if message["type"] in {
                "lifespan.startup.failed",
                "lifespan.shutdown.complete",
                "lifespan.shutdown.failed",
            }:
                if terminal is not None:
                    raise RuntimeError(
                        "Reload worker published more than one terminal lifespan event."
                    )
                terminal = dict(message)
            else:
                await send(message)

        try:
            await self._server(scope, receive, send_lifespan)
        except BaseException as primary:
            try:
                await context.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Reload worker and project storage cleanup failed", [primary, cleanup]
                ) from None
            raise
        else:
            try:
                await context.close()
            except BaseException as cleanup:
                if terminal is not None and terminal["type"].endswith(".failed"):
                    # ASGI applications may report failure without raising an
                    # exception. Retain that diagnostic alongside cleanup.
                    reported = RuntimeError(
                        terminal.get("message") or "Application lifespan reported failure."
                    )
                    raise BaseExceptionGroup(
                        "Reload worker and project storage cleanup failed", [reported, cleanup]
                    ) from None
                raise
        finally:
            self._scope.close()
            self._server = None
        # Uvicorn releases its lifecycle waiter at this publication. All owned
        # cleanup (including restoring the project context) must precede it.
        if terminal is not None:
            await send(terminal)
