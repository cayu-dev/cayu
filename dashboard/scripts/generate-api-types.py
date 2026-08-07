"""Generate or verify the dashboard API baseline against the installed Cayu package."""

from __future__ import annotations

import argparse
import asyncio
import filecmp
import importlib.metadata
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = DASHBOARD_ROOT / "src" / "lib" / "generated" / "server-api"
OPENAPI_BASELINE = DASHBOARD_ROOT / "server-openapi.json"
RELEASE_METADATA = DASHBOARD_ROOT / "src" / "lib" / "release-metadata.ts"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed API baseline or generated types are stale.",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="cayu-dashboard-openapi-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        schema_path = temp_dir / "server-openapi.json" if args.check else OPENAPI_BASELINE
        output_dir = temp_dir / "server-api" if args.check else GENERATED_DIR
        release_metadata = _release_metadata()
        schema_path.write_text(_openapi_json(), encoding="utf-8", newline="\n")
        if not args.check and output_dir.exists():
            shutil.rmtree(output_dir)
        _run_generator(schema_path=schema_path, output_dir=output_dir)
        if args.check:
            if RELEASE_METADATA.read_bytes() != release_metadata:
                raise SystemExit(
                    "Dashboard release metadata is stale; run `npm run generate:api` "
                    "with the intended Cayu Python environment."
                )
            _assert_file_matches(expected=schema_path, actual=OPENAPI_BASELINE)
            _assert_generated_tree_matches(expected_dir=output_dir, actual_dir=GENERATED_DIR)
        else:
            RELEASE_METADATA.write_bytes(release_metadata)
    return 0


def _openapi_json() -> str:
    from cayu import AgentSpec, CayuApp, CorpusTarget, RunRequest, SQLiteEvalStore
    from cayu.server import EvalsConfig, EvaluationPromotionConfig, ServerConfig, create_server

    app = CayuApp()
    app.register_agent(AgentSpec(name="assistant", model="schema-only"))

    def schema_auth(_request):
        return {"subject": "schema-generator"}

    with tempfile.TemporaryDirectory(prefix="cayu-openapi-evals-") as evals_temp_dir:
        eval_store = SQLiteEvalStore(Path(evals_temp_dir) / "evals.db")
        try:
            server = create_server(
                app,
                config=ServerConfig.protected(
                    schema_auth,
                    evaluation_promotion=EvaluationPromotionConfig(
                        target_key="schema",
                        source_agent_name="assistant",
                        application_release_id="schema",
                    ),
                    evals=EvalsConfig(
                        target=CorpusTarget(
                            key="schema",
                            app=app,
                            request_base=RunRequest(agent_name="assistant", messages=[]),
                            application_release_id="schema",
                        ),
                        store=eval_store,
                    ),
                ),
            )
            schema = server.openapi()
        finally:
            asyncio.run(eval_store.close())
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def _release_metadata() -> bytes:
    from cayu._server_contract_version import SERVER_CONTRACT_VERSION

    cayu_version = importlib.metadata.version("cayu")
    return (
        f'export const DASHBOARD_SOURCE_CAYU_VERSION = "{cayu_version}"\n'
        f'export const SUPPORTED_SERVER_CONTRACT_VERSION = "{SERVER_CONTRACT_VERSION}"\n'
    ).encode()


def _run_generator(*, schema_path: Path, output_dir: Path) -> None:
    generator = DASHBOARD_ROOT / "node_modules" / "@hey-api" / "openapi-ts" / "bin" / "run.js"
    subprocess.run(
        [
            "node",
            str(generator),
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


def _assert_file_matches(*, expected: Path, actual: Path) -> None:
    if not actual.is_file() or not filecmp.cmp(expected, actual, shallow=False):
        raise SystemExit(
            "Dashboard OpenAPI baseline is stale; run `npm run generate:api` "
            "with the intended Cayu Python environment."
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
        raise SystemExit(
            "Generated API types are stale; changed files: "
            + ", ".join(str(path) for path in changed)
        )


def _relative_files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


if __name__ == "__main__":
    raise SystemExit(main())
