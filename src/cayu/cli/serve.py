"""Run one configured Cayu control-plane process."""

from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import importlib
import inspect
import math
import os
import secrets
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any, NoReturn, cast

from cayu._development_people import (
    DEVELOPMENT_ROLES,
    DevelopmentPerson,
    parse_identity_proxy_table,
)
from cayu._operator_credentials import (
    ENVIRONMENT_OPERATOR_AUTH_TARGET,
    OPERATOR_PASSWORD_VARIABLE,
    OPERATOR_USERNAME_VARIABLE,
)
from cayu.cli._project_evals import load_project_eval_targets
from cayu.cli._targets import TargetResolutionError, load_target
from cayu.cli.project import (
    CayuProject,
    build_project_app,
    build_project_service,
    close_project_app,
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


# `cayu.server.proxy_identity`'s variables, named here so parsing needs no server extra.
_PROXY_IDENTITY_VARIABLES = (
    "CAYU_PROXY_IDENTITY_ISSUER",
    "CAYU_PROXY_IDENTITY_AUDIENCE",
    "CAYU_PROXY_IDENTITY_JWKS_URL",
    "CAYU_PROXY_IDENTITY_HEADER",
)
_IDENTITY_PERSON_OPTIONS = (
    ("identity_org", "--org", "org"),
    ("identity_role", "--role", "role"),
    ("identity_name", "--name", "name"),
    ("identity_email", "--email", "email"),
)
#: Every option that describes the `--identity-proxy` person, in the order reload passes them.
_IDENTITY_OPTIONS = (
    ("identity_as", "--as"),
    *((attribute, option) for attribute, option, _ in _IDENTITY_PERSON_OPTIONS),
)


@dataclass(frozen=True)
class _ServeSettings:
    auth_target: str | None = None
    price_book: Path | None = None
    startup_recovery_statuses: frozenset[SessionStatus] | None = None
    recovery_inactive_after_seconds: int = 300
    # The project's own routes: `module:function` targets, each called with the app and
    # `development=` and returning a FastAPI router served next to Cayu's routes. A
    # function that also takes `auth=` gets the auth dependency `cayu serve` resolved
    # (`--auth` over `[tool.cayu.serve].auth`, as a dependency returning an
    # `AuthContext`; `None` under `--dev`), so its routes can use the same auth as
    # Cayu's own. Functions without it get only the app and `development=`.
    routers: tuple[str, ...] = ()
    # The project's built web app, served at / for every path no route answers.
    ui: Path | None = None
    # `RequestTimingConfig` fields the project sets, such as `long_request_seconds`.
    request_timing: tuple[tuple[str, Any], ...] = ()
    # A `module:function` target called with the app and returning the
    # `cayu.server.inbox.InboxAccess` rule `create_server` applies on every inbox route.
    inbox_access: str | None = None
    # `[tool.cayu.serve.identity_proxy]`: who `--identity-proxy` can sign in as.
    identity_people: tuple[DevelopmentPerson, ...] = ()
    identity_default: str | None = None


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
        "--deployment-kind",
        choices=("local", "cloud", "embedded"),
        help="Explicit hosting declaration for System diagnostics; no effect on access policy.",
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
    identity = parser.add_argument_group(
        "development identity proxy",
        "Sign requests in as a chosen person, as Cayu Cloud's router does, with a key "
        "that lives only in this process. Loopback only; serves with the project's real "
        "auth, so the app takes its deployed sign-in path. Pick a person at "
        "/.cayu/sign-in or per request with an X-Cayu-Dev-Person header.",
    )
    identity.add_argument(
        "--identity-proxy",
        action="store_true",
        help="Run the development identity proxy in front of the server. Not with --dev.",
    )
    identity.add_argument(
        "--as",
        dest="identity_as",
        metavar="SUBJECT",
        help="The person signed in until another is picked; overrides "
        "[tool.cayu.serve.identity_proxy].default.",
    )
    identity.add_argument(
        "--org", dest="identity_org", metavar="ORG", help="The --as person's org."
    )
    identity.add_argument(
        "--role", dest="identity_role", choices=DEVELOPMENT_ROLES, help="The --as person's role."
    )
    identity.add_argument("--name", dest="identity_name", metavar="NAME", help="Their name.")
    identity.add_argument("--email", dest="identity_email", metavar="EMAIL", help="Their email.")


def _resolve_serve_project(
    args: argparse.Namespace, *, start: Path | None = None
) -> tuple[CayuProject, _ServeSettings]:
    if args.dev:
        require_loopback_dev_host(args.host)
    _check_identity_proxy_arguments(args)
    project = resolve_project(command="cayu serve", suggest_explicit_target=False, start=start)
    settings = _serve_settings(project.root)
    if args.identity_proxy:
        identity_proxy_default(args, settings)
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
    if project.service_target is not None and settings.request_timing:
        raise ServeError(
            "[tool.cayu.serve].long_request_seconds is for projects served by their "
            "factory; a maintained public service keeps the default request log."
        )
    if project.service_target is not None and settings.routers:
        raise ServeError(
            "[tool.cayu.serve].routers is for projects served by their factory; a "
            "maintained public service declares its product routes in its service_factory."
        )
    if project.service_target is not None and settings.inbox_access is not None:
        raise ServeError(
            "[tool.cayu.serve].inbox_access is for projects served by their factory; a "
            "maintained public service passes inbox_access in its service_factory."
        )
    if project.service_target is not None and settings.ui is not None:
        raise ServeError(
            "[tool.cayu.serve].ui is for projects served by their factory; a maintained "
            "public service serves its own pages."
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


def serve_project_control_plane(
    args: argparse.Namespace,
    project: CayuProject,
) -> ProjectControlPlaneContext:
    """The project context `cayu serve` serves, with the eval targets it declares.

    Call it inside ``project_context(project.root)``. A declared eval target that
    can't be loaded is reported and left out; the Agent still serves.
    """

    def report(reference: str, reason: str) -> None:
        print(
            f"warning: eval target {reference} is not served: {reason}",
            file=sys.stderr,
            flush=True,
        )

    return build_project_control_plane_context(
        project.root,
        mode="development" if args.dev else "production",
        eval_targets=load_project_eval_targets(project.root, on_error=report),
        deployment_kind=args.deployment_kind,
    )


def _build_serve_app(
    args: argparse.Namespace,
    project: CayuProject,
    settings: _ServeSettings,
    project_control_plane: ProjectControlPlaneContext,
    *,
    identity_proxy: Any,
):
    """Assemble either supported server with the same access and lifecycle policy.

    ``identity_proxy`` is what ``start_serve_identity_proxy(args)`` returned, before
    any project code was imported.
    """
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
        if args.deployment_kind is not None and service.deployment_kind != args.deployment_kind:
            mismatch = ServeError(
                "The service deployment kind does not match --deployment-kind. "
                "Forward project_context to create_agent_service(), or configure its "
                "deployment_kind explicitly to match."
            )
            try:
                asyncio.run(close_project_app(service.cayu_app))
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Deployment declaration rejected and runtime cleanup failed",
                    [mismatch, cleanup],
                ) from None
            raise mismatch
        if not args.dev:
            from cayu.runtime.checks import check_public_service_deployment

            findings = check_public_service_deployment(service.manifest)
            if findings:
                codes = ", ".join(item.code for item in findings)
                raise ServeError(
                    "Refusing to start an unsafe public service: "
                    f"{codes}. Run `cayu check --deploy --fail-on warning --json`."
                )
        hosting = service.cayu_app._external_wait_hosting
        if (
            hosting is not None
            and hosting.delivery_path is not None
            and not hosting.serves_delivery(service.asgi_app)
        ):
            raise ServeError(
                "The service_factory's app hosts external-wait webhooks that its service "
                "doesn't serve: call app.host_external_waits(...) before "
                "create_agent_service(...), which then mounts the delivery route."
            )
        server = service.asgi_app
        if identity_proxy is not None:
            server = _with_identity_proxy(server, identity_proxy, args, settings)
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
        if identity_proxy is not None:
            from cayu.server.proxy_identity import ProxyIdentityAuth

            if not isinstance(auth, ProxyIdentityAuth):
                print(
                    f"warning: --identity-proxy changes nothing for /api: its auth target "
                    f"{auth_target} isn't a ProxyIdentityAuth. Use "
                    '"cayu.server.environment_auth:OPERATOR_ONLY_AUTH" or '
                    "ProxyIdentityAuth.from_environment(...) to trust it.",
                    file=sys.stderr,
                    flush=True,
                )
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
        if args.dev:
            register_demo_provider(app)
        lifecycle = server_module.ServerLifecycleConfig(
            startup_recovery_statuses=settings.startup_recovery_statuses,
            recovery_inactive_after_seconds=settings.recovery_inactive_after_seconds,
        )
        # Request timing is on for every `cayu serve` process so
        # `cayu diagnostics requests` and support bundles can read it, and its
        # request log replaces uvicorn's access log (`serve_uvicorn_options`).
        request_timing = server_module.RequestTimingConfig(**dict(settings.request_timing))
        config = (
            server_module.ServerConfig.local_development(
                lifecycle=lifecycle,
                deployment_kind=project_control_plane.deployment_kind or "local",
                request_timing=request_timing,
                dashboard=dashboard,
            )
            if args.dev
            else server_module.ServerConfig.protected(
                auth,
                lifecycle=lifecycle,
                deployment_kind=project_control_plane.deployment_kind or "local",
                request_timing=request_timing,
                dashboard=dashboard,
            )
        )
        server = server_module.create_server(
            app,
            config=config,
            project_context=project_control_plane,
            inbox_access=_load_inbox_access(settings.inbox_access, app, development=args.dev),
        )
        _include_project_routers(server, app, settings.routers, development=args.dev, auth=auth)
        if settings.ui is not None:
            # Last, so every route above answers first.
            from cayu.server import WebAppError, serve_web_app

            try:
                serve_web_app(server, settings.ui)
            except WebAppError as exc:
                raise ServeError(f"[tool.cayu.serve].ui: {exc}") from exc
        if identity_proxy is not None:
            server = _with_identity_proxy(server, identity_proxy, args, settings)
        hosting = app._external_wait_hosting
        if hosting is not None and hosting.delivery_path is not None:
            print(f"External-wait delivery: POST {hosting.delivery_path}/{{source}}")
        print(
            "Cayu Developer dashboard: "
            f"{_control_plane_url(args.host, args.port, config.dashboard.path)}"
        )
        if settings.routers:
            print("Project routes: " + ", ".join(settings.routers))
        if settings.ui is not None:
            print(f"Web app: {_control_plane_url(args.host, args.port, '/')}")
    return server


def register_demo_provider(app: Any) -> bool:
    """Let a development server continue sessions started under the demo provider.

    ``cayu serve --dev`` registers ``DemoModelProvider`` as ``demo`` (never the
    default), so a demonstration session started by a script or `cayu session
    start` can be answered and continued here with canned, marked replies. An app
    that registers its own ``demo`` provider keeps it. Returns whether it registered.
    """

    from cayu.providers.demo import DEMO_PROVIDER_NAME, DemoModelProvider

    if DEMO_PROVIDER_NAME in app.list_providers():
        return False
    # Only sessions that name `demo` use it: it never becomes the default, even in an
    # application that registers no other provider.
    app._provider_registry.register(DemoModelProvider(), claim_default=False)
    return True


def request_log_covers(server: Any) -> bool:
    """Whether Cayu's request log has a line for every request ``server`` answers."""

    # Only --identity-proxy imports this module, and then it wraps ``server``.
    development_identity = sys.modules.get("cayu.server.development_identity")
    if development_identity is not None and isinstance(
        server, development_identity.DevelopmentIdentityMiddleware
    ):
        # Its sign-in pages aside, every request reaches the server it wraps.
        server = server.app
    return getattr(getattr(server, "state", None), "cayu_request_log_covers_server", False) is True


def serve_uvicorn_options(uvicorn: Any, *, covered: bool) -> dict[str, Any]:
    """Uvicorn's logging for `cayu serve`: Cayu's request log beside uvicorn's own.

    Cayu's request log (``cayu.server.request``) goes to stdout, where uvicorn writes
    its access log, as ``INFO:     GET /api/sessions/{session_id} 200 18 ms``. When it
    has a line for every request (``covered``), uvicorn's access log is left out: its
    lines have no duration and carry raw paths and query strings. The configuration
    extends uvicorn's own default (``uvicorn.config.LOGGING_CONFIG``); a runner without
    one is left on its defaults.
    """

    from cayu.server._request_timing import REQUEST_LOGGER_NAME

    defaults = getattr(getattr(uvicorn, "config", None), "LOGGING_CONFIG", None)
    if not isinstance(defaults, dict):
        return {}
    log_config = copy.deepcopy(defaults)
    log_config["handlers"]["cayu_request"] = {
        "formatter": "default",
        "class": "logging.StreamHandler",
        "stream": "ext://sys.stdout",
    }
    log_config["loggers"][REQUEST_LOGGER_NAME] = {
        "handlers": ["cayu_request"],
        "level": "INFO",
        "propagate": False,
    }
    return {"log_config": log_config, "access_log": not covered}


def _include_project_routers(
    server: Any,
    app: Any,
    targets: tuple[str, ...],
    *,
    development: bool,
    auth: Any,
) -> None:
    """Serve the project's own routes next to Cayu's, never in place of one.

    These routes, like the delivery route of ``app.host_external_waits`` served
    first, are outside ``--auth`` and the Developer dashboard's access policy: they
    authenticate their own callers (each ``ExternalEventSource`` does for its route).
    A router function that takes ``auth`` gets the dependency ``cayu serve`` resolved,
    adapted as for Cayu's routes to return an ``AuthContext`` (``None`` under
    ``--dev``), so it can protect its routes the same way.
    """

    from fastapi import APIRouter

    reserved = {
        "/" + path.split("/")[1]
        for path in (getattr(route, "path", "") for route in server.routes)
        if path.startswith("/")
    }
    hosting = app._external_wait_hosting
    delivery_path = None if hosting is None else hosting.delivery_path
    if delivery_path is not None:
        _include_router(
            server, hosting.delivery_router(), reserved, None, label="External-wait delivery"
        )
    for target in targets:
        try:
            build = load_target(target, label="Serve router target", normalize_errors=True)
        except TargetResolutionError as exc:
            raise ServeError(str(exc)) from exc
        if not callable(build):
            raise ServeError(f"Serve router target {target} must be callable.")
        options: dict[str, Any] = {"development": development}
        if _accepts_keyword(build, "auth"):
            from cayu.server.auth import server_auth_dependency

            # The same adapter as Cayu's own routes: it takes the request and
            # returns a validated AuthContext, whatever the target's signature.
            options["auth"] = None if auth is None else server_auth_dependency(auth)
        router = build(app, **options)
        if not isinstance(router, APIRouter):
            raise ServeError(f"Serve router target {target} must return a fastapi.APIRouter.")
        _include_router(
            server, router, reserved, delivery_path, label=f"Serve router target {target}"
        )


def _accepts_keyword(function: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        (
            parameter.name == name
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _include_router(
    server: Any,
    router: Any,
    reserved: set[str],
    delivery_path: str | None,
    *,
    label: str,
) -> None:
    for route in router.routes:
        path = getattr(route, "path", "")
        first = path.split("/")[1] if path.startswith("/") else ""
        # A parameter or an empty first segment would also match Cayu's own paths.
        if not first or "{" in first:
            raise ServeError(
                f"{label}: {path or '/'} must start with a fixed path segment, such as /app."
            )
        prefix = "/" + first
        if prefix in reserved:
            raise ServeError(
                f"{label}: {path} collides with Cayu's own {prefix} routes; serve the "
                "project's routes under another prefix."
            )
        if delivery_path is not None and (
            path == delivery_path or path.startswith(delivery_path + "/")
        ):
            raise ServeError(
                f"{label}: {path} collides with the external-wait delivery route under "
                f"{delivery_path}; serve it under another path."
            )
    server.include_router(router)


def run_serve(args: argparse.Namespace) -> int:
    project_control_plane: ProjectControlPlaneContext | None = None
    exit_code = 0
    try:
        project, settings = _resolve_serve_project(args)
        if args.identity_proxy:
            # Before --reload spawns, so every worker keeps the same login.
            _ensure_operator_login()
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
                    identity_proxy = start_serve_identity_proxy(args)
                    project_control_plane = serve_project_control_plane(args, project)
                    server = _build_serve_app(
                        args,
                        project,
                        settings,
                        project_control_plane,
                        identity_proxy=identity_proxy,
                    )
                except SystemExit as exc:
                    status = exc.code if type(exc.code) is int else 1
                    raise ServeError(
                        f"Serve project startup raised SystemExit with status {status}."
                    ) from exc
                uvicorn.run(
                    server,
                    host=args.host,
                    port=args.port,
                    **serve_uvicorn_options(uvicorn, covered=request_log_covers(server)),
                )
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


def _is_loopback_bind_host(host: str) -> bool:
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return host.casefold() == "localhost"


def require_loopback_dev_host(host: str) -> None:
    """Refuse `--dev` on any host but loopback, as `cayu serve` does."""

    if not _is_loopback_bind_host(host):
        raise ServeError(
            "Refusing to expose an unauthenticated API and Developer dashboard on a "
            "non-loopback host. "
            "Use --dev with a loopback host such as 127.0.0.1 or ::1, or configure "
            "authentication before binding publicly."
        )


def require_identity_proxy_for_person_options(args: argparse.Namespace) -> None:
    """Refuse `--as`, `--org`, `--role`, `--name` and `--email` without `--identity-proxy`."""

    if args.identity_proxy:
        return
    person_options = [
        option for attribute, option in _IDENTITY_OPTIONS if getattr(args, attribute) is not None
    ]
    if person_options:
        raise ServeError(f"{', '.join(person_options)} only apply with --identity-proxy.")


def _check_identity_proxy_arguments(args: argparse.Namespace) -> None:
    require_identity_proxy_for_person_options(args)
    if not args.identity_proxy:
        return
    if args.dev:
        raise ServeError(
            "--identity-proxy can't be combined with --dev: it serves with the project's "
            "real auth, so the app takes its deployed sign-in path."
        )
    if not _is_loopback_bind_host(args.host):
        raise ServeError(
            "Refusing to run the development identity proxy on a non-loopback host: "
            "anyone who can reach it could sign in as anybody. Use a loopback host such "
            "as 127.0.0.1 or ::1."
        )
    already = [name for name in _PROXY_IDENTITY_VARIABLES if os.environ.get(name, "").strip()]
    if already:
        raise ServeError(
            f"--identity-proxy sets the proxy identity variables itself, but "
            f"{', '.join(already)} {'is' if len(already) == 1 else 'are'} already set. "
            "Unset them, or serve without --identity-proxy to trust that proxy."
        )


def identity_proxy_default(
    args: argparse.Namespace, settings: _ServeSettings
) -> DevelopmentPerson | None:
    """The person `--identity-proxy` signs in until another is picked.

    `--as` over `[tool.cayu.serve.identity_proxy].default`; `--org`, `--role`,
    `--name` and `--email` replace that person's fields.
    """

    subject = args.identity_as if args.identity_as is not None else settings.identity_default
    changes = {
        field: getattr(args, attribute)
        for attribute, _, field in _IDENTITY_PERSON_OPTIONS
        if getattr(args, attribute) is not None
    }
    if subject is None:
        if changes:
            raise ServeError(
                "--org, --role, --name and --email describe the --as person; pass --as."
            )
        return None
    base = next((person for person in settings.identity_people if person.subject == subject), None)
    try:
        if base is None:
            return DevelopmentPerson(subject=subject, **changes)
        return dataclasses.replace(base, **changes)
    except ValueError as exc:
        raise ServeError(f"--identity-proxy person: {exc}") from None


def _ensure_operator_login() -> None:
    """Set a random Developer dashboard login for this process when none is set."""

    generated = {}
    for name, value in (
        (OPERATOR_USERNAME_VARIABLE, "operator"),
        (OPERATOR_PASSWORD_VARIABLE, secrets.token_urlsafe(18)),
    ):
        if not os.environ.get(name, "").strip():
            os.environ[name] = generated[name] = value
    if OPERATOR_PASSWORD_VARIABLE in generated:
        print(
            "Developer dashboard login for this process: "
            f"{os.environ[OPERATOR_USERNAME_VARIABLE]} / {os.environ[OPERATOR_PASSWORD_VARIABLE]}",
            flush=True,
        )
    elif generated:
        # Never print a password this process didn't make.
        print(
            "Developer dashboard username for this process: "
            f"{os.environ[OPERATOR_USERNAME_VARIABLE]}",
            flush=True,
        )


def start_serve_identity_proxy(args: argparse.Namespace) -> Any:
    """With `--identity-proxy`, register a new development identity proxy and name it in
    this process's environment; otherwise ``None``.

    Call it before anything imports project code, eval targets included, so
    `ProxyIdentityAuth.from_environment()` in the project, or in
    `cayu.server.environment_auth` (whose ready-made targets are built on import),
    trusts it. Each process makes its own key: a `--reload` worker gets a new one.
    """

    if not args.identity_proxy:
        return None
    from cayu.server.development_identity import DevelopmentIdentityProxy

    proxy = DevelopmentIdentityProxy()
    proxy.register()
    os.environ.update(proxy.environment())
    return proxy


def _with_identity_proxy(
    server: Any, proxy: Any, args: argparse.Namespace, settings: _ServeSettings
) -> Any:
    """``server`` behind the development identity proxy, as behind Cayu Cloud's router.

    It wraps the assembled ASGI app rather than adding middleware, which a
    maintained service has already built by the time it is returned.
    """

    default = identity_proxy_default(args, settings)
    people = list(settings.identity_people)
    if default is not None:
        people = [default, *(person for person in people if person.subject != default.subject)]
    wrapped = proxy.middleware(server, people=people, default=default)
    who = (
        f"signed in as {default.subject} ({default.role} in {default.org}) until another is picked"
        if default is not None
        else "nobody is signed in until a person is picked"
    )
    print(
        "Development identity proxy: "
        f"{_service_endpoint_url(args.host, args.port, '/.cayu/sign-in')}; {who}"
    )
    return wrapped


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
        "ui",
        "long_request_seconds",
        "identity_proxy",
        "inbox_access",
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
    ui = configured.get("ui")
    if ui is not None and (not isinstance(ui, str) or not ui.strip()):
        raise ServeError(
            f"{discovered.pyproject}: [tool.cayu.serve].ui must be the path of a directory "
            'holding a built web app, such as "web/static".'
        )
    request_timing: tuple[tuple[str, Any], ...] = ()
    if "long_request_seconds" in configured:
        long_request = configured["long_request_seconds"]
        if long_request is False:
            request_timing = (("long_request_seconds", None),)
        elif (
            isinstance(long_request, bool)
            or not isinstance(long_request, int | float)
            or not math.isfinite(long_request)
            or not 0 < long_request <= 86_400
        ):
            raise ServeError(
                f"{discovered.pyproject}: [tool.cayu.serve].long_request_seconds must be "
                "a number of seconds above 0 and at most 86400, or false to turn the "
                "still-running line off."
            )
        else:
            request_timing = (("long_request_seconds", float(long_request)),)
    inbox_access = configured.get("inbox_access")
    if inbox_access is not None and (not isinstance(inbox_access, str) or not inbox_access.strip()):
        raise ServeError(
            f"{discovered.pyproject}: [tool.cayu.serve].inbox_access must be a "
            "MODULE:FUNCTION string."
        )
    identity_people: tuple[DevelopmentPerson, ...] = ()
    identity_default: str | None = None
    if "identity_proxy" in configured:
        try:
            identity_people, identity_default = parse_identity_proxy_table(
                configured["identity_proxy"]
            )
        except ValueError as exc:
            raise ServeError(f"{discovered.pyproject}: {exc}") from None
    return _ServeSettings(
        inbox_access=None if inbox_access is None else inbox_access.strip(),
        identity_people=identity_people,
        identity_default=identity_default,
        auth_target=None if auth is None else auth.strip(),
        price_book=None if price_book is None else (root / price_book).resolve(),
        startup_recovery_statuses=statuses,
        recovery_inactive_after_seconds=inactivity,
        routers=tuple(target.strip() for target in routers or ()),
        ui=None if ui is None else (root / ui.strip()),
        request_timing=request_timing,
    )


def _load_inbox_access(target: str | None, app: Any, *, development: bool) -> Any:
    """The `InboxAccess` rule `[tool.cayu.serve].inbox_access` builds for `app`.

    Under `--dev` there is no auth, so no caller to apply a rule to: it isn't
    loaded, and serve says so.
    """

    if target is None:
        return None
    if development:
        print(
            f"note: [tool.cayu.serve].inbox_access ({target}) isn't applied under --dev, "
            "which has no authenticated caller. Run `cayu serve --identity-proxy` to see "
            "the inbox as a signed-in person.",
            file=sys.stderr,
            flush=True,
        )
        return None
    try:
        build = load_target(target, label="Serve inbox access target", normalize_errors=True)
    except TargetResolutionError as exc:
        raise ServeError(str(exc)) from exc
    if not callable(build):
        raise ServeError(f"Serve inbox access target {target} must be callable.")
    try:
        rule = build(app)
    except ValueError as exc:
        raise ServeError(f"Serve inbox access target {target}: {exc}") from exc
    if not callable(rule):
        raise ServeError(
            f"Serve inbox access target {target} must return an InboxAccess rule, a "
            "function of (AuthContext, InboxItemScope) returning an awaitable bool."
        )
    return rule


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
