"""Generate or verify the dashboard API baseline against the installed Cayu package."""

from __future__ import annotations

import argparse
import asyncio
import filecmp
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from cayu._dashboard_artifacts import (
    API_TYPE_FILES,
    API_TYPES_ROOT,
    EVAL_TEXT_LIMITS_FILE,
    GENERATED_ROOT,
    LABEL_LIMITS_FILE,
    OPENAPI_FILE,
    RELEASE_METADATA_FILE,
    render_release_metadata,
    validate_generated_source_inventory,
)

DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
GENERATED_SOURCE_ROOT = DASHBOARD_ROOT / GENERATED_ROOT
GENERATED_DIR = DASHBOARD_ROOT / API_TYPES_ROOT
OPENAPI_BASELINE = DASHBOARD_ROOT / OPENAPI_FILE
RELEASE_METADATA = DASHBOARD_ROOT / RELEASE_METADATA_FILE
EVAL_TEXT_LIMITS = DASHBOARD_ROOT / EVAL_TEXT_LIMITS_FILE
LABEL_LIMITS = DASHBOARD_ROOT / LABEL_LIMITS_FILE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed API baseline or generated types are stale.",
    )
    args = parser.parse_args(argv)
    if args.check:
        _assert_expected_cayu_import()

    with tempfile.TemporaryDirectory(prefix="cayu-dashboard-openapi-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        schema_path = temp_dir / "server-openapi.json" if args.check else OPENAPI_BASELINE
        output_dir = temp_dir / "server-api" if args.check else GENERATED_DIR
        release_metadata = _release_metadata()
        eval_text_limits = _eval_text_limits()
        label_limits = _label_limits()
        schema_path.write_text(_openapi_json(), encoding="utf-8", newline="\n")
        if not args.check and output_dir.exists():
            shutil.rmtree(output_dir)
        _run_generator(schema_path=schema_path, output_dir=output_dir)
        expected_api_files = {Path(path).relative_to(API_TYPES_ROOT) for path in API_TYPE_FILES}
        actual_api_files = _relative_files(output_dir)
        if actual_api_files != expected_api_files:
            raise SystemExit(
                "Dashboard API generator outputs do not match Cayu's declared files; "
                f"missing: {sorted(map(str, expected_api_files - actual_api_files))}; "
                f"unexpected: {sorted(map(str, actual_api_files - expected_api_files))}"
            )
        if args.check:
            if not RELEASE_METADATA.is_file() or RELEASE_METADATA.read_bytes() != release_metadata:
                raise SystemExit(
                    "Dashboard release metadata is stale; run `npm run generate:api` "
                    "with the intended Cayu Python environment."
                )
            _assert_file_matches(expected=schema_path, actual=OPENAPI_BASELINE)
            _assert_generated_tree_matches(expected_dir=output_dir, actual_dir=GENERATED_DIR)
            if not EVAL_TEXT_LIMITS.is_file() or EVAL_TEXT_LIMITS.read_bytes() != eval_text_limits:
                raise SystemExit(
                    "Dashboard evaluation text limits are stale; run `npm run generate:api` "
                    "with the intended Cayu Python environment."
                )
            if not LABEL_LIMITS.is_file() or LABEL_LIMITS.read_bytes() != label_limits:
                raise SystemExit(
                    "Dashboard label limits are stale; run `npm run generate:api` "
                    "with the intended Cayu Python environment."
                )
        else:
            RELEASE_METADATA.write_bytes(release_metadata)
            EVAL_TEXT_LIMITS.parent.mkdir(parents=True, exist_ok=True)
            EVAL_TEXT_LIMITS.write_bytes(eval_text_limits)
            LABEL_LIMITS.write_bytes(label_limits)
        try:
            validate_generated_source_inventory(
                [
                    OPENAPI_FILE,
                    RELEASE_METADATA_FILE,
                    *(
                        f"{GENERATED_ROOT}/{path.as_posix()}"
                        for path in _relative_files(GENERATED_SOURCE_ROOT)
                    ),
                ]
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
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
    return render_release_metadata(
        cayu_version=cayu_version, server_contract_version=SERVER_CONTRACT_VERSION
    )


def _eval_text_limits() -> bytes:
    from cayu.evals.corpus import (
        EVAL_CORPUS_MAX_DESCRIPTION_CHARS,
        EVAL_CORPUS_MAX_NAME_CHARS,
        EVAL_CORPUS_TEXT_LENGTH_UNIT,
    )

    return (
        "// Generated from cayu.evals.corpus by npm run generate:api.\n"
        "export const EVAL_TEXT_LIMITS = {\n"
        f'  unit: "{EVAL_CORPUS_TEXT_LENGTH_UNIT}",\n'
        f"  name: {EVAL_CORPUS_MAX_NAME_CHARS},\n"
        f"  description: {EVAL_CORPUS_MAX_DESCRIPTION_CHARS},\n"
        "} as const\n"
    ).encode()


def _label_limits() -> bytes:
    from cayu._validation import LABEL_MAX_KEY_CHARS, LABEL_MAX_VALUE_CHARS, LABEL_TEXT_LENGTH_UNIT
    from cayu.sessions.queries import (
        MAX_AGGREGATE_LABEL_FILTERS,
        MAX_AGGREGATE_LABEL_SELECTOR_VALUES,
        MAX_AGGREGATE_LABEL_SELECTORS,
    )

    return (
        "// Generated from cayu._validation and cayu.sessions.queries by npm run generate:api.\n"
        "export const LABEL_LIMITS = {\n"
        f'  unit: "{LABEL_TEXT_LENGTH_UNIT}",\n'
        f"  key: {LABEL_MAX_KEY_CHARS},\n"
        f"  value: {LABEL_MAX_VALUE_CHARS},\n"
        f"  filters: {MAX_AGGREGATE_LABEL_FILTERS},\n"
        f"  selectors: {MAX_AGGREGATE_LABEL_SELECTORS},\n"
        f"  selectorValues: {MAX_AGGREGATE_LABEL_SELECTOR_VALUES},\n"
        "} as const\n"
    ).encode()


def _assert_expected_cayu_import() -> None:
    import cayu

    if not RELEASE_METADATA.is_file():
        raise SystemExit(
            "Dashboard release metadata is missing; run `npm run generate:api` "
            "with the intended Cayu Python environment."
        )
    committed = RELEASE_METADATA.read_text(encoding="utf-8")
    match = re.search(r'DASHBOARD_SOURCE_CAYU_VERSION = "([^"]+)"', committed)
    if match is None:
        raise SystemExit(f"Dashboard release metadata has no Cayu version: {RELEASE_METADATA}")
    expected = match.group(1)
    observed = importlib.metadata.version("cayu")
    if observed != expected:
        raise SystemExit(
            "Dashboard Cayu import version mismatch: "
            f"expected={expected!r}, observed={observed!r}, "
            f"python={sys.executable!r}, cayu={cayu.__file__!r}."
        )


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
