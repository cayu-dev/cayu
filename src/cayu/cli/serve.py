"""Run one configured Cayu control-plane process."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any, NoReturn, cast

from cayu._operator_credentials import (
    ENVIRONMENT_OPERATOR_AUTH_TARGET,
    OPERATOR_PASSWORD_VARIABLE,
    OPERATOR_USERNAME_VARIABLE,
)
from cayu.cli._targets import TargetResolutionError, load_target
from cayu.cli.project import (
    CayuProject,
    build_project_app,
    build_project_service,
    discover_cayu_project_configuration,
    project_context,
    resolve_project,
)
from cayu.cli.project_control_plane import (
    build_project_control_plane_context,
    close_project_control_plane_context,
)
from cayu.project_control_plane import ProjectControlPlaneContext
from cayu.sessions.records import SessionStatus


class ServeError(ValueError):
    """The configured server process cannot be started safely."""


@dataclass(frozen=True)
class _ServeSettings:
    auth_target: str | None = None
    price_book: Path | None = None
    startup_recovery_statuses: frozenset[SessionStatus] | None = None
    recovery_inactive_after_seconds: int = 300
    # The project's own routes: `module:function` targets, each called with the app and
    # `development=` and returning a FastAPI router served next to Cayu's routes.
    routers: tuple[str, ...] = ()


def add_serve_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "serve",
        help="Serve a configured Cayu project's API and Developer dashboard.",
        description=(
            "Start one process-scoped Cayu server and its Developer dashboard. Select "
            "--dev explicitly for unauthenticated local development or configure "
            "authentication for deployment."
        ),
    )
    add_serve_arguments(parser)


class _ServeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ServeError(f"cayu serve: {message}")

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        raise ServeError(message or f"cayu serve exited with status {status}.")


def parse_serve_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    """Parse the arguments after `cayu serve` exactly as the command does.

    Raises `ServeError` instead of exiting, so a tool that only reads a `cayu serve`
    command line, such as a deploy check, sees the same options and errors.
    """

    parser = _ServeArgumentParser(prog="cayu serve", add_help=False)
    add_serve_arguments(parser)
    return parser.parse_args(list(arguments))


def serve_auth_target(args: argparse.Namespace, configured: str | None) -> str | None:
    """The auth target `cayu serve` loads outside `--dev`: `--auth` overrides the config."""

    return configured if args.auth is None else args.auth


def add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the `cayu serve` options to `parser`."""

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
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Restart on Python source changes for development; in-memory state is lost.",
    )
    parser.add_argument(
        "--price-book",
        type=Path,
        metavar="PATH",
        help="Dashboard PriceBook JSON file; overrides [tool.cayu.serve].price_book.",
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


def _resolve_serve_project(
    args: argparse.Namespace, *, start: Path | None = None
) -> tuple[CayuProject, _ServeSettings]:
    if args.dev:
        require_loopback_dev_host(args.host)
    project = resolve_project(command="cayu serve", suggest_explicit_target=False, start=start)
    settings = _serve_settings(project.root)
    auth_target = serve_auth_target(args, settings.auth_target)
    if args.price_book is not None:
        # Resolve before project_context changes the working directory or reload spawns.
        args.price_book = args.price_book.resolve()
    if project.service_target is not None and (
        args.price_book is not None or settings.price_book is not None
    ):
        raise ServeError(
            "A maintained public service owns its dashboard configuration; "
            "configure its DashboardConfig.runtime_config priceBook in the service factory."
        )
    if project.service_target is not None and settings.routers:
        raise ServeError(
            "[tool.cayu.serve].routers is for projects served by their factory; a "
            "maintained public service declares its product routes in its service_factory."
        )
    if project.service_target is None and not args.dev and auth_target is None:
        raise ServeError(
            "Refusing to start an unauthenticated server. Pass --dev only for "
            "trusted local development or configure an authentication target, "
            "for example [tool.cayu.serve] auth = "
            f'"{ENVIRONMENT_OPERATOR_AUTH_TARGET}", which reads '
            f"{OPERATOR_USERNAME_VARIABLE} and {OPERATOR_PASSWORD_VARIABLE}."
        )
    return project, settings


def _build_serve_app(
    args: argparse.Namespace,
    project: CayuProject,
    settings: _ServeSettings,
    project_control_plane: ProjectControlPlaneContext,
):
    """Assemble either supported server with the same access and lifecycle policy."""
    server_module = importlib.import_module("cayu.server")
    service_mode = "development" if args.dev else "production"
    auth_target = serve_auth_target(args, settings.auth_target)
    if project.service_target is not None:
        if args.auth is not None or settings.auth_target is not None:
            raise ServeError(
                "A maintained public service owns its separate product and Developer "
                "dashboard access policies; do not configure "
                "[tool.cayu.serve].auth or --auth."
            )
        service = build_project_service(
            project.service_target,
            mode=service_mode,
            command="Serve",
            project_context=project_control_plane,
        )
        if not args.dev:
            from cayu.runtime.checks import check_public_service_deployment

            findings = check_public_service_deployment(service.manifest)
            if findings:
                codes = ", ".join(item.code for item in findings)
                raise ServeError(
                    "Refusing to start an unsafe public service: "
                    f"{codes}. Run `cayu check --deploy --fail-on warning --json`."
                )
        server = service.asgi_app
        print(
            "Cayu product service: "
            f"{_service_endpoint_url(args.host, args.port, service.manifest.product_api_path)}"
            "/operations"
        )
        print(
            "Cayu Developer dashboard: "
            f"{_control_plane_url(args.host, args.port, service.manifest.control_plane_path)}"
        )
    else:
        auth = None if args.dev else _load_auth(auth_target)
        price_book = args.price_book or settings.price_book
        runtime_config = {}
        if price_book is not None:
            from cayu.cli._json_inputs import json_object_file

            try:
                runtime_config["priceBook"] = json_object_file(price_book)
                dashboard = server_module.DashboardConfig(runtime_config=runtime_config)
            except (ValueError, TypeError):
                raise ServeError(
                    "Dashboard price book must be a valid PriceBook JSON file of at most 1 MiB."
                ) from None
        else:
            dashboard = server_module.DashboardConfig()
        app = build_project_app(project.target, command="Serve")
        lifecycle = server_module.ServerLifecycleConfig(
            startup_recovery_statuses=settings.startup_recovery_statuses,
            recovery_inactive_after_seconds=settings.recovery_inactive_after_seconds,
        )
        # Request timing is on for every `cayu serve` process so
        # `cayu diagnostics requests` and support bundles can read it.
        request_timing = server_module.RequestTimingConfig()
        config = (
            server_module.ServerConfig.local_development(
                lifecycle=lifecycle,
                request_timing=request_timing,
                dashboard=dashboard,
            )
            if args.dev
            else server_module.ServerConfig.protected(
                auth,
                lifecycle=lifecycle,
                request_timing=request_timing,
                dashboard=dashboard,
            )
        )
        server = server_module.create_server(
            app,
            config=config,
            project_context=project_control_plane,
        )
        _include_project_routers(server, app, settings.routers, development=args.dev)
        print(
            "Cayu Developer dashboard: "
            f"{_control_plane_url(args.host, args.port, config.dashboard.path)}"
        )
        if settings.routers:
            print("Project routes: " + ", ".join(settings.routers))
    return server


def _include_project_routers(
    server: Any, app: Any, targets: tuple[str, ...], *, development: bool
) -> None:
    """Serve the project's own routes next to Cayu's, never in place of one.

    They authenticate their own callers: the Developer dashboard's access policy doesn't
    cover them.
    """

    from fastapi import APIRouter

    reserved = {
        "/" + path.split("/")[1]
        for path in (getattr(route, "path", "") for route in server.routes)
        if path.startswith("/")
    }
    for target in targets:
        try:
            build = load_target(target, label="Serve router target", normalize_errors=True)
        except TargetResolutionError as exc:
            raise ServeError(str(exc)) from exc
        if not callable(build):
            raise ServeError(f"Serve router target {target} must be callable.")
        router = build(app, development=development)
        if not isinstance(router, APIRouter):
            raise ServeError(f"Serve router target {target} must return a fastapi.APIRouter.")
        for route in router.routes:
            path = getattr(route, "path", "")
            first = path.split("/")[1] if path.startswith("/") else ""
            # A parameter or an empty first segment would also match Cayu's own paths.
            if not first or "{" in first:
                raise ServeError(
                    f"Serve router target {target}: {path or '/'} must start with a fixed "
                    "path segment, such as /app."
                )
            prefix = "/" + first
            if prefix in reserved:
                raise ServeError(
                    f"Serve router target {target}: {path} collides with Cayu's own "
                    f"{prefix} routes; serve the project's routes under another prefix."
                )
        server.include_router(router)


def run_serve(args: argparse.Namespace) -> int:
    project_control_plane: ProjectControlPlaneContext | None = None
    exit_code = 0
    try:
        project, settings = _resolve_serve_project(args)
        try:
            uvicorn = importlib.import_module("uvicorn")
        except ModuleNotFoundError as exc:
            if exc.name != "uvicorn":
                raise
            raise ServeError(
                'Cayu serve requires the server extra. Install it with: pip install "cayu[server]"'
            ) from exc
        if args.reload:
            from cayu.cli._serve_reload import run_reload

            run_reload(uvicorn, args, project.root)
        else:
            with project_context(project.root):
                try:
                    project_control_plane = build_project_control_plane_context(
                        project.root, mode="development" if args.dev else "production"
                    )
                    server = _build_serve_app(args, project, settings, project_control_plane)
                except SystemExit as exc:
                    status = exc.code if type(exc.code) is int else 1
                    raise ServeError(
                        f"Serve project startup raised SystemExit with status {status}."
                    ) from exc
                uvicorn.run(server, host=args.host, port=args.port)
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        try:
            close_project_control_plane_context(project_control_plane)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


def require_loopback_dev_host(host: str) -> None:
    """Refuse `--dev` on any host but loopback, as `cayu serve` does."""

    try:
        is_loopback = ip_address(host).is_loopback
    except ValueError:
        is_loopback = host.casefold() == "localhost"
    if not is_loopback:
        raise ServeError(
            "Refusing to expose an unauthenticated API and Developer dashboard on a "
            "non-loopback host. "
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
        "price_book",
        "startup_recovery_statuses",
        "recovery_inactive_after_seconds",
        "routers",
    }
    if unexpected:
        raise ServeError(
            f"{discovered.pyproject}: [tool.cayu.serve] has unsupported keys: "
            f"{', '.join(sorted(unexpected))}."
        )
    price_book = configured.get("price_book")
    if price_book is not None and (not isinstance(price_book, str) or not price_book.strip()):
        raise ServeError("[tool.cayu.serve].price_book must be a non-empty file path.")
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
    routers = configured.get("routers")
    if routers is not None and (
        not isinstance(routers, list)
        or not routers
        or any(not isinstance(target, str) or not target.strip() for target in routers)
    ):
        raise ServeError(
            f"{discovered.pyproject}: [tool.cayu.serve].routers must be a non-empty array "
            "of MODULE:FUNCTION strings."
        )
    return _ServeSettings(
        auth_target=None if auth is None else auth.strip(),
        price_book=None if price_book is None else (root / price_book).resolve(),
        startup_recovery_statuses=statuses,
        recovery_inactive_after_seconds=inactivity,
        routers=tuple(target.strip() for target in routers or ()),
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
    stripped_path = path.strip("/")
    url_path = "/" if not stripped_path else f"/{stripped_path}/"
    return _service_endpoint_url(host, port, url_path)


def _service_endpoint_url(host: str, port: int, path: str) -> str:
    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{url_host}:{port}{path}"
