"""Validate the bundled model catalog before tests and release builds."""

from __future__ import annotations

import argparse
from datetime import date

from cayu import default_model_catalog
from maintenance.model_catalog.policy import CATALOG_MAX_AGE_DAYS, validate_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--today", type=date.fromisoformat, default=None)
    parser.add_argument("--max-age-days", type=int, default=CATALOG_MAX_AGE_DAYS)
    parser.add_argument(
        "--skip-staleness",
        action="store_true",
        help="run deterministic structural/source checks without a wall-clock freshness gate",
    )
    args = parser.parse_args(argv)
    catalog = default_model_catalog()
    max_age_days = None if args.skip_staleness else args.max_age_days
    validate_catalog(catalog, today=args.today, max_age_days=max_age_days)
    print(
        f"validated {len(catalog.models)} bundled models "
        f"(catalog {catalog.catalog_version}, "
        f"max age {f'{args.max_age_days}d' if max_age_days is not None else 'not checked'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
