from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version


def _version() -> str:
    try:
        return version("cayu")
    except PackageNotFoundError:
        return "0.1.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cayu",
        description="Developer/admin CLI for Cayu agent projects.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="Print the Cayu version.")
    subparsers.add_parser("validate", help="Validate an agent project.")
    subparsers.add_parser("serve", help="Start the agent runtime server.")

    from cayu.cli.evals import add_eval_parser, run_eval_command
    from cayu.cli.storage import add_storage_parser, run_storage

    add_eval_parser(subparsers)
    add_storage_parser(subparsers)

    args = parser.parse_args(argv)

    if args.command == "version":
        print(f"cayu {_version()}")
        return 0

    if args.command == "storage":
        return run_storage(args)

    if args.command == "eval":
        return run_eval_command(args)

    if args.command in {"validate", "serve"}:
        parser.error(f"'{args.command}' is not implemented yet")

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
