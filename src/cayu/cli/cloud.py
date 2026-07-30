from __future__ import annotations

import argparse


def add_cloud_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "cloud",
        help="Manage Cayu Cloud.",
        description=("Manage Cayu Cloud. Deployment commands are not available in this release."),
    )
    parser.set_defaults(_cloud_parser=parser)


def run_cloud(args: argparse.Namespace) -> int:
    parser: argparse.ArgumentParser = args._cloud_parser
    parser.print_help()
    return 0
