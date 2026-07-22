from __future__ import annotations

import argparse

from cayu._version import package_version


def _version() -> str:
    return package_version()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cayu",
        description="Developer/admin CLI for Cayu agent projects.",
    )
    parser.add_argument("--version", action="version", version=f"cayu {_version()}")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "version",
        help="Print the Cayu version.",
        description="Print the installed Cayu version, then return to `cayu --help`.",
    )

    from cayu.cli.auth import add_auth_parser
    from cayu.cli.check import add_check_parser
    from cayu.cli.console import add_console_parser
    from cayu.cli.evals import add_eval_parser
    from cayu.cli.generate import add_generate_parser
    from cayu.cli.guide import add_guide_parser
    from cayu.cli.inspect import add_inspect_parser
    from cayu.cli.lambda_microvm import add_lambda_microvm_parser
    from cayu.cli.scaffold import add_new_parser
    from cayu.cli.session import add_session_parser
    from cayu.cli.storage import add_storage_parser

    add_auth_parser(subparsers)
    add_check_parser(subparsers)
    add_console_parser(subparsers)
    add_eval_parser(subparsers)
    add_generate_parser(subparsers)
    add_guide_parser(subparsers)
    add_inspect_parser(subparsers)
    add_lambda_microvm_parser(subparsers)
    add_new_parser(subparsers)
    add_session_parser(subparsers)
    add_storage_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()

    args = parser.parse_args(argv)

    from cayu.cli.auth import run_auth
    from cayu.cli.check import run_check
    from cayu.cli.console import run_console
    from cayu.cli.evals import run_eval_command
    from cayu.cli.generate import run_generate
    from cayu.cli.guide import run_guide
    from cayu.cli.inspect import run_inspect
    from cayu.cli.lambda_microvm import run_lambda_microvm
    from cayu.cli.scaffold import run_new
    from cayu.cli.session import run_session
    from cayu.cli.storage import run_storage

    if args.command == "version":
        print(f"cayu {_version()}")
        return 0

    if args.command == "auth":
        return run_auth(args)

    if args.command == "new":
        return run_new(args)

    if args.command == "storage":
        return run_storage(args)

    if args.command == "session":
        return run_session(args)

    if args.command == "console":
        return run_console(args)

    if args.command == "check":
        return run_check(args)

    if args.command == "eval":
        return run_eval_command(args)

    if args.command == "generate":
        return run_generate(args)

    if args.command == "guide":
        return run_guide(args)

    if args.command == "inspect":
        return run_inspect(args)

    if args.command == "lambda-microvm":
        return run_lambda_microvm(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
