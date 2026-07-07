from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = REPO_ROOT / "dashboard"
SRC_ROOT = REPO_ROOT / "src"
GENERATED_DIR = DASHBOARD_ROOT / "src" / "lib" / "generated" / "server-api"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate dashboard TypeScript API types.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if committed generated API types are stale.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(SRC_ROOT))

    with tempfile.TemporaryDirectory(prefix="cayu-openapi-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        schema_path = temp_dir / "openapi.json"
        output_dir = temp_dir / "server-api" if args.check else GENERATED_DIR

        schema_path.write_text(_openapi_json(), encoding="utf-8")
        if not args.check and output_dir.exists():
            shutil.rmtree(output_dir)
        _run_generator(schema_path=schema_path, output_dir=output_dir)

        if args.check:
            _assert_generated_tree_matches(expected_dir=output_dir, actual_dir=GENERATED_DIR)

    return 0


def _openapi_json() -> str:
    from cayu import CayuApp
    from cayu.server import create_server

    server = create_server(CayuApp(), dev=True)
    return json.dumps(server.openapi(), indent=2, sort_keys=True) + "\n"


def _run_generator(*, schema_path: Path, output_dir: Path) -> None:
    subprocess.run(
        [
            "npm",
            "exec",
            "openapi-ts",
            "--",
            "--input",
            str(schema_path),
            "--output",
            str(output_dir),
            "--plugins",
            "@hey-api/typescript",
            "--silent",
        ],
        cwd=DASHBOARD_ROOT,
        check=True,
    )


def _assert_generated_tree_matches(*, expected_dir: Path, actual_dir: Path) -> None:
    if not actual_dir.exists():
        raise SystemExit(f"Generated API types are missing: {actual_dir}")

    expected_files = _relative_files(expected_dir)
    actual_files = _relative_files(actual_dir)
    if expected_files != actual_files:
        missing = sorted(str(path) for path in expected_files - actual_files)
        extra = sorted(str(path) for path in actual_files - expected_files)
        details = []
        if missing:
            details.append(f"missing committed files: {missing}")
        if extra:
            details.append(f"unexpected committed files: {extra}")
        raise SystemExit("Generated API types are stale; " + "; ".join(details))

    changed = [
        path
        for path in sorted(expected_files)
        if not filecmp.cmp(expected_dir / path, actual_dir / path, shallow=False)
    ]
    if changed:
        changed_files = ", ".join(str(path) for path in changed)
        raise SystemExit(
            "Generated API types are stale. Run `cd dashboard && npm run generate:api`. "
            f"Changed files: {changed_files}"
        )


def _relative_files(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    }


if __name__ == "__main__":
    raise SystemExit(main())
