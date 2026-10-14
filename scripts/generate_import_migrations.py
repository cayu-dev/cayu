"""Generate the compact installed import map from docs/public-api-imports.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(replacements: dict[str, str]) -> str:
    grouped: dict[str, dict[str, list[str]]] = {}
    for old, new in sorted(replacements.items()):
        package, name = old.rsplit(".", 1)
        module, replacement_name = new.rsplit(".", 1)
        if package not in {"cayu", "cayu.runtime"} or name != replacement_name:
            raise ValueError(f"Unsupported import migration: {old} -> {new}")
        grouped.setdefault(package, {}).setdefault(module, []).append(name)
    return json.dumps(grouped, sort_keys=True, separators=(",", ":")) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without rewriting the map")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    replacements = json.loads((root / "docs/public-api-imports.json").read_text())
    target = root / "src/cayu/data/public_api_imports.json"
    expected = render(replacements).encode()
    if args.check:
        if not target.is_file() or target.read_bytes() != expected:
            print("Import map is stale; run python scripts/generate_import_migrations.py")
            return 1
        print(f"All {len(replacements)} installed import replacements are current.")
        return 0
    target.write_bytes(expected)
    print(f"Generated {len(replacements)} import replacements ({len(expected)} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
