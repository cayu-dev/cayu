"""Run one configured Cayu control-plane process."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any, cast

from cayu.cli._targets import TargetResolutionError, load_target
from cayu.cli.project import (
    build_project_app,
    discover_cayu_project_configuration,
    project_context,
    resolve_project,
)
from cayu.runtime.sessions import SessionStatus


class ServeError(ValueError):
    """The configured server process cannot be started safely."""


@dataclass(frozen=True)
class _ServeSettings:
    auth_target: str | None = None
    startup_recovery_statuses: frozenset[SessionStatus] | None = None
    recovery_inactive_after_seconds: int = 300


def add_serve_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "serve",
        help="Serve a configured Cayu project's control plane.",
        description=(
            "Start one process-scoped Cayu control plane. Select --dev explicitly "
            "for unauthenticated local development or configure authentication "
            "for deployment."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface passed to uvicorn (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=8000,
        help="TCP port passed to uvicorn (default: 8000).",
    )
    access = parser.add_mutually_exclusive_group()
    access.add_argument(
        "--dev",
        action="store_true",
        help="Explicitly allow an unauthenticated local-development server.",
    )
    access.add_argument(
        "--auth",
        metavar="MODULE:ATTRIBUTE",
        help="Override [tool.cayu.serve].auth with a protected auth dependency target.",
    )


def run_serve(args: argparse.Namespace) -> int:
    try:
        if args.dev:
            _require_loopback_dev_host(args.host)
        project = resolve_project(
            command="cayu serve",
            suggest_explicit_target=False,
        )
        settings = _serve_settings(project.root)
        auth_target = settings.auth_target if args.auth is None else args.auth
        if not args.dev and auth_target is None:
            raise ServeError(
                "Refusing to start an unauthenticated server. Pass --dev only for "
                "trusted local development or configure an authentication target."
            )
        with project_context(project.root):
            try:
                server_module = importlib.import_module("cayu.server")
                try:
                    uvicorn = importlib.import_module("uvicorn")
                except ModuleNotFoundError as exc:
                    if exc.name != "uvicorn":
                        raise
                    raise ServeError(
                        "Cayu serve requires the server extra. "
                        'Install it with: pip install "cayu[server]"'
                    ) from exc
                auth = None if args.dev else _load_auth(auth_target)
                app = build_project_app(project.target, command="Serve")
                lifecycle = server_module.ServerLifecycleConfig(
                    startup_recovery_statuses=settings.startup_recovery_statuses,
                    recovery_inactive_after_seconds=settings.recovery_inactive_after_seconds,
                )
                config = (
                    server_module.ServerConfig.local_development(lifecycle=lifecycle)
                    if args.dev
                    else server_module.ServerConfig.protected(auth, lifecycle=lifecycle)
                )
                server = server_module.create_server(app, config=config)
            except SystemExit as exc:
                status = exc.code if type(exc.code) is int else 1
                raise ServeError(
                    f"Serve project startup raised SystemExit with status {status}."
                ) from exc
            try:
                print(
                    "Cayu control plane: "
                    f"{_control_plane_url(args.host, args.port, config.dashboard.path)}"
                )
                uvicorn.run(server, host=args.host, port=args.port)
            except SystemExit as exc:
                return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _require_loopback_dev_host(host: str) -> None:
    try:
        is_loopback = ip_address(host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise ServeError(
            "Refusing to expose an unauthenticated control plane on a non-loopback host. "
            "Use --dev with a loopback host such as 127.0.0.1 or ::1, or configure "
            "authentication before binding publicly."
        )


def _serve_settings(root: Path) -> _ServeSettings:
    discovered = discover_cayu_project_configuration(
        discovery_keys=("factory",),
        start=root,
    )
    if discovered is None:
        raise ServeError("No configured Cayu project was found.")
    value = discovered.config.get("serve")
    if value is None:
        return _ServeSettings()
    if not isinstance(value, dict):
        raise ServeError(f"{discovered.pyproject}: [tool.cayu].serve must be a table.")
    configured = cast("dict[str, Any]", value)
    unexpected = set(configured) - {
        "auth",
        "startup_recovery_statuses",
        "recovery_inactive_after_seconds",
    }
    if unexpected:
        raise ServeError(
            f"{discovered.pyproject}: [tool.cayu.serve] has unsupported keys: "
            f"{', '.join(sorted(unexpected))}."
        )
    auth = configured.get("auth")
    if auth is not None and (not isinstance(auth, str) or not auth.strip()):
        raise ServeError(
            f"{discovered.pyproject}: [tool.cayu.serve].auth must be a non-empty string."
        )
    statuses_value = configured.get("startup_recovery_statuses")
    if statuses_value is None:
        statuses = None
    else:
        if (
            not isinstance(statuses_value, list)
            or not statuses_value
            or any(not isinstance(status, str) or not status.strip() for status in statuses_value)
        ):
            raise ServeError(
                f"{discovered.pyproject}: [tool.cayu.serve].startup_recovery_statuses "
                "must be a non-empty array of status strings."
            )
        try:
            statuses = frozenset(SessionStatus(status.strip()) for status in statuses_value)
        except ValueError as exc:
            supported = ", ".join(status.value for status in SessionStatus)
            raise ServeError(
                f"{discovered.pyproject}: "
                "[tool.cayu.serve].startup_recovery_statuses contains an unknown "
                f"status. Known values are: {supported}."
            ) from exc
    inactivity = configured.get("recovery_inactive_after_seconds", 300)
    if type(inactivity) is not int or inactivity < 0:
        raise ServeError(
            f"{discovered.pyproject}: "
            "[tool.cayu.serve].recovery_inactive_after_seconds must be "
            "a non-negative integer."
        )
    if statuses is None and "recovery_inactive_after_seconds" in configured:
        raise ServeError(
            f"{discovered.pyproject}: "
            "[tool.cayu.serve].recovery_inactive_after_seconds requires "
            "startup_recovery_statuses."
        )
    if statuses is not None and "recovery_inactive_after_seconds" not in configured:
        raise ServeError(
            f"{discovered.pyproject}: "
            "[tool.cayu.serve].startup_recovery_statuses requires "
            "recovery_inactive_after_seconds."
        )
    return _ServeSettings(
        auth_target=None if auth is None else auth.strip(),
        startup_recovery_statuses=statuses,
        recovery_inactive_after_seconds=inactivity,
    )


def _load_auth(target: str | None) -> Any:
    if target is None:
        raise AssertionError("Protected serve mode requires an authentication target.")
    try:
        auth = load_target(
            target,
            label="Serve authentication target",
            normalize_errors=True,
        )
    except TargetResolutionError as exc:
        raise ServeError(str(exc)) from exc
    if not callable(auth):
        raise ServeError("Serve authentication target must be callable.")
    return auth


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("port must be an integer.") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535.")
    return port


def _control_plane_url(host: str, port: int, path: str) -> str:
    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    stripped_path = path.strip("/")
    url_path = "/" if not stripped_path else f"/{stripped_path}/"
    return f"http://{url_host}:{port}{url_path}"
