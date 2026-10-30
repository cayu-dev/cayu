from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from cayu._validation import require_durable_clean_nonblank
from cayu.budgets.pricing import PriceBook, default_price_book
from cayu.cli._json_inputs import json_object_file
from cayu.cli._output import add_output_options, output_destination, terminal_safe
from cayu.cli.project import (
    CayuProjectConfiguration,
    ProjectError,
    build_project_app,
    build_project_service,
    close_project_app,
    discover_cayu_project_configuration,
    project_context,
    resolve_project,
)
from cayu.runtime._model_check import (
    MODEL_CHECK_SCHEMA_VERSION,
    MODEL_CHECK_TIMEOUT_SECONDS,
    ModelCheckResult,
    ModelCheckRouteError,
    PriceBookSource,
    check_model,
    resolve_model_check_route,
)

_NOT_RUN_CODE = "MODEL_CHECK_NOT_RUN"


def add_providers_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "providers",
        help="Check the model providers a Cayu project registers.",
        description=(
            "Check the model providers a Cayu project registers. Run `cayu guide "
            "providers#check-that-a-model-answers` for the output and exit codes."
        ),
    )
    commands = parser.add_subparsers(dest="providers_command", required=True)
    check = commands.add_parser(
        "check",
        help="Check that a model answers through the project's provider and is priced.",
        description=(
            "Build the project's application and send one bounded request for MODEL "
            "through its registered provider and credentials: a fixed short prompt, the "
            "smallest output cap, no tools and a short timeout. Report the outcome, "
            "latency, route, usage and pricing, never the prompt or the answer. Exits 1 "
            "when the model does not answer."
        ),
    )
    check.add_argument("model", metavar="MODEL", type=model_argument, help="Model to check.")
    check.add_argument(
        "--provider",
        metavar="NAME",
        help=(
            "Registered provider to send the request through (default: the provider an "
            "agent using MODEL resolves to: a matching model pattern, else the default "
            "provider)."
        ),
    )
    check.add_argument(
        "--target",
        help="Override project discovery with a module:factory target.",
    )
    add_output_options(check)


def model_argument(value: str) -> str:
    try:
        return require_durable_clean_nonblank(value, "model")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def run_providers(args: argparse.Namespace) -> int:
    return run_model_check(
        target=args.target,
        model=args.model,
        provider_name=args.provider,
        output_format=args.output_format,
        output=args.output,
    )


def run_model_check(
    *,
    target: str | None,
    model: str,
    provider_name: str | None,
    output_format: str,
    output: str | None,
) -> int:
    """Run the check and render it; exit 0 when the model answered, 1 when not."""

    try:
        result = asyncio.run(_check(target, model=model, provider_name=provider_name))
    except (ProjectError, ModelCheckRouteError) as exc:
        _render_not_run(str(exc), as_json=output_format == "json")
        return 2
    try:
        with output_destination(output):
            if output_format == "json":
                print(result.model_dump_json(indent=2))
            else:
                print(_render_human(result))
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 2
    return 0 if result.ok else 1


async def _check(target: str | None, *, model: str, provider_name: str | None) -> ModelCheckResult:
    project = resolve_project(target, command="cayu providers check")
    pricing, price_book_source = _price_book(
        None
        if target is not None
        else discover_cayu_project_configuration(discovery_keys=("factory",), start=project.root)
    )
    with project_context(project.root):
        try:
            # A maintained service's app is the one `cayu serve` runs, with its providers.
            service = (
                None
                if project.service_target is None
                else build_project_service(
                    project.service_target, mode="production", command="Providers check"
                )
            )
            app = (
                build_project_app(project.target, command="Providers check")
                if service is None
                else service.cayu_app
            )
        except ProjectError:
            raise
        except Exception as exc:
            raise ProjectError(f"Application factory failed ({type(exc).__name__}): {exc}") from exc
        try:
            registered, resolution = resolve_model_check_route(
                app, model=model, provider_name=provider_name
            )
            return await check_model(
                registered,
                model=model,
                resolution=resolution,
                pricing=pricing,
                price_book_source=price_book_source,
                redactor=app._secret_redactor,
                timeout_seconds=MODEL_CHECK_TIMEOUT_SECONDS,
            )
        finally:
            await close_project_app(app)


def _price_book(
    configured: CayuProjectConfiguration | None,
) -> tuple[PriceBook, PriceBookSource]:
    """The project's `[tool.cayu.serve].price_book`, else Cayu's bundled book."""

    serve = None if configured is None else configured.config.get("serve")
    path = serve.get("price_book") if isinstance(serve, dict) else None
    if configured is None or path is None:
        return default_price_book(), "bundled"
    message = (
        f"{configured.pyproject}: [tool.cayu.serve].price_book must name a valid "
        "PriceBook JSON file of at most 1 MiB."
    )
    if not isinstance(path, str) or not path.strip():
        raise ProjectError(message)
    try:
        return PriceBook.model_validate(json_object_file(configured.root / path)), "project"
    except ValueError:
        raise ProjectError(message) from None


def _render_not_run(message: str, *, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "schema_version": MODEL_CHECK_SCHEMA_VERSION,
                    "error": {"code": _NOT_RUN_CODE, "message": message},
                },
                sort_keys=True,
            )
        )
    else:
        print(f"error: {message}", file=sys.stderr)


def _render_human(result: ModelCheckResult) -> str:
    route = f"{result.route.provider}/{result.route.model}"
    latency = "" if result.latency_ms is None else f" in {result.latency_ms} ms"
    if result.answer is not None:
        lines = [f"OK {route} answered{latency} (finish: {result.answer.finish_reason})."]
    else:
        error = result.error
        if error is None:  # pragma: no cover - a failed check always carries its error
            raise AssertionError("A failed model check has no error.")
        status = "" if error.status_code is None else f" HTTP {error.status_code}"
        lines = [f"FAILED {route}{latency}: {error.category.value}{status}. {error.message}"]
        if error.detail is not None:
            lines.append(f"  Detail: {error.detail}")
    if result.usage is not None:
        lines.append(
            f"Usage: {result.usage.input_tokens} input, {result.usage.output_tokens} output tokens."
        )
    pricing = result.pricing
    book = f"{pricing.price_book.source} price book {pricing.price_book.version}"
    if not pricing.priced:
        lines.append(f"Pricing: not priced by the {book} ({pricing.unpriced_reason}).")
    elif pricing.estimated_cost is not None:
        lines.append(
            f"Pricing: priced by the {book}; this check cost about "
            f"{pricing.estimated_cost} {pricing.currency}."
        )
    else:
        lines.append(f"Pricing: priced by the {book}.")
    return "\n".join(terminal_safe(line) for line in lines)
