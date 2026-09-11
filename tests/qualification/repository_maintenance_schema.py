"""Explicit application reservation schema commands, separate from Runtime migrations."""

import argparse
import asyncio
import os
import sys

from tests.qualification.repository_maintenance_runs import PostgresMaintenanceRunStore


async def _run_schema(action, dsn):
    store = PostgresMaintenanceRunStore(dsn)
    operation = {"initialize": store.initialize, "check": store.check_ready}[action]
    await operation()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manage the application reservation schema only.")
    parser.add_argument("action", choices=("initialize", "check"))
    args = parser.parse_args(argv)
    try:
        dsn = os.environ.get("CAYU_DATABASE_URL")
        if type(dsn) is not str or not dsn.strip():
            raise ValueError
        asyncio.run(_run_schema(args.action, dsn))
    except Exception:
        # A commit may have succeeded; do not print DSNs or claim rollback.
        print("Maintenance reservation schema unavailable.", file=sys.stderr)
        return 1
    print(
        "Maintenance reservation schema "
        + ("initialized." if args.action == "initialize" else "ready.")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
