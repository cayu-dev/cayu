from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from cayu._version import package_version


def _version() -> str:
    return package_version()


def _run_version(_args: argparse.Namespace) -> int:
    print(f"cayu {_version()}")
    return 0


def _command_specs() -> tuple[
    tuple[
        str,
        Callable[[argparse._SubParsersAction], None],
        Callable[[argparse.Namespace], int],
    ],
    ...,
]:
    from cayu.cli.auth import add_auth_parser, run_auth
    from cayu.cli.check import add_check_parser, run_check
    from cayu.cli.cloud import add_cloud_parser, run_cloud
    from cayu.cli.console import add_console_parser, run_console
    from cayu.cli.dashboard import add_dashboard_parser, run_dashboard
    from cayu.cli.evals import add_eval_parser, run_eval_command
    from cayu.cli.generate import add_generate_parser, run_generate
    from cayu.cli.guide import add_guide_parser, run_guide
    from cayu.cli.inspect import add_inspect_parser, run_inspect
    from cayu.cli.lambda_microvm import add_lambda_microvm_parser, run_lambda_microvm
    from cayu.cli.scaffold import add_new_parser, run_new
    from cayu.cli.serve import add_serve_parser, run_serve
    from cayu.cli.session import add_session_parser, run_session
    from cayu.cli.storage import add_storage_parser, run_storage
    from cayu.cli.worker import add_worker_parser, run_worker

    return (
        ("auth", add_auth_parser, run_auth),
        ("check", add_check_parser, run_check),
        ("cloud", add_cloud_parser, run_cloud),
        ("console", add_console_parser, run_console),
        ("dashboard", add_dashboard_parser, run_dashboard),
        ("eval", add_eval_parser, run_eval_command),
        ("generate", add_generate_parser, run_generate),
        ("guide", add_guide_parser, run_guide),
        ("inspect", add_inspect_parser, run_inspect),
        ("lambda-microvm", add_lambda_microvm_parser, run_lambda_microvm),
        ("new", add_new_parser, run_new),
        ("serve", add_serve_parser, run_serve),
        ("session", add_session_parser, run_session),
        ("storage", add_storage_parser, run_storage),
        ("worker", add_worker_parser, run_worker),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cayu",
        description="Developer/admin CLI for Cayu agent projects.",
    )
    parser.add_argument("--version", action="version", version=f"cayu {_version()}")
    subparsers = parser.add_subparsers(dest="command")

    version_parser = subparsers.add_parser(
        "version",
        help="Print the Cayu version.",
        description="Print the installed Cayu version, then return to `cayu --help`.",
    )
    version_parser.set_defaults(_runner=_run_version)

    for name, add_parser, runner in _command_specs():
        add_parser(subparsers)
        subparsers.choices[name].set_defaults(_runner=runner)

    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "cloud":
        from cayu.cli.cloud import run_cloud_cli

        return run_cloud_cli(arguments[1:])

    parser = _build_parser()
    args = parser.parse_args(arguments)
    runner = getattr(args, "_runner", None)
    if runner is not None:
        return runner(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
