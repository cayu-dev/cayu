"""Exercise the complete generated-application contract from an installed Cayu CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

_REQUIRED_HOMES = {
    "configuration",
    "agents",
    "prompts",
    "tools",
    "policies",
    "environments",
    "workflows",
    "operations",
    "knowledge",
    "memory",
    "domain",
    "integrations",
    "evals",
    "observability",
    "tests",
    "data",
}
_POSTGRES_PLACEHOLDER = "postgresql://cayu-unconfigured@127.0.0.1/cayu"

_POSTGRES_PROOF = r"""\
import asyncio

from cayu import PostgresSessionStore, PostgresTaskStore
from cayu.storage.migrations import SchemaMode

from app import build_app


async def main() -> None:
    creator = PostgresSessionStore(
        __import__("os").environ["CAYU_DATABASE_URL"],
        schema_mode=SchemaMode.CREATE,
    )
    try:
        await creator.ensure_schema()
    finally:
        await creator.close()

    application = build_app()
    assert isinstance(application.session_store, PostgresSessionStore)
    assert isinstance(application.task_store, PostgresTaskStore)
    stores = [application.session_store, application.task_store]
    if application.knowledge_store is not None:
        from cayu import PostgresKnowledgeStore

        assert isinstance(application.knowledge_store, PostgresKnowledgeStore)
        stores.append(application.knowledge_store)
    try:
        for store in stores:
            await store.ensure_schema()
    finally:
        for store in stores:
            await store.close()


asyncio.run(main())
"""


def _run(
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    capture: bool = True,
    allowed_exit_codes: frozenset[int] = frozenset({0}),
) -> str:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=capture,
        check=False,
    )
    if completed.returncode not in allowed_exit_codes:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {argv!r}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed.stdout


def _create(
    cayu: Path,
    root: Path,
    name: str,
    *options: str,
    environment: dict[str, str],
) -> Path:
    _run(
        [str(cayu), "new", name, "--dir", str(root), *options, "--json"],
        cwd=root,
        environment=environment,
        capture=True,
    )
    return root / name


def _verify_agent(
    *,
    cayu: Path,
    python: Path,
    project: Path,
    environment: dict[str, str],
    postgres: bool = False,
    postgres_dsn: str | None = None,
) -> None:
    assert _REQUIRED_HOMES.issubset({path.name for path in project.iterdir()})
    assert (project / "CLAUDE.md").read_text(encoding="utf-8") == "@AGENTS.md\n"
    assert "[tool.cayu.scaffold]" in (project / "pyproject.toml").read_text(encoding="utf-8")
    app = (project / "app.py").read_text(encoding="utf-8")
    assert "class " not in app
    assert "def build_app(" in app
    check_environment = dict(environment)
    if postgres:
        check_environment["CAYU_DATABASE_URL"] = _POSTGRES_PLACEHOLDER
    _run([str(cayu), "inspect", "--json"], cwd=project, environment=environment)
    _run(
        [str(cayu), "check", "--fail-on", "warning", "--json"],
        cwd=project,
        environment=check_environment,
    )
    _run([str(python), "-m", "pytest", "-q"], cwd=project, environment=environment)
    _run([str(cayu), "eval", "run"], cwd=project, environment=environment)
    if postgres_dsn is not None:
        live_environment = {**environment, "CAYU_DATABASE_URL": postgres_dsn}
        _run(
            [str(python), "-c", _POSTGRES_PROOF],
            cwd=project,
            environment=live_environment,
        )


def _verify_minimal(
    *,
    cayu: Path,
    python: Path,
    project: Path,
    environment: dict[str, str],
) -> None:
    assert not (project / "configuration").exists()
    assert "minimal = true" in (project / "pyproject.toml").read_text(encoding="utf-8")
    _run([str(cayu), "inspect", "--json"], cwd=project, environment=environment)
    _run(
        [str(cayu), "check", "--fail-on", "warning", "--json"],
        cwd=project,
        environment=environment,
    )
    _run([str(python), "-m", "pytest", "-q"], cwd=project, environment=environment)
    _run([str(cayu), "eval", "run"], cwd=project, environment=environment)


def _verify_coding(
    *,
    cayu: Path,
    python: Path,
    project: Path,
    environment: dict[str, str],
    docker: bool = False,
) -> None:
    assert (project / "operations/coding.py").is_file()
    compatibility_facade = (project / "composition.py").read_text(encoding="utf-8")
    assert "from operations.coding import" in compatibility_facade
    assert ("build_coding_composition" if docker else "build_coding_app") in compatibility_facade
    assert "class " not in compatibility_facade
    assert "def " not in compatibility_facade
    if docker:
        assert (project / "Dockerfile.coding").is_file()
        assert (project / "docker-coding-image.json").is_file()
        image = json.loads((project / "docker-coding-image.json").read_text())
        assert image["content_digest"] is None
        check = subprocess.run(
            [str(cayu), "check", "--fail-on", "warning", "--json"],
            cwd=project,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert check.returncode == 2
        failure = json.loads(check.stdout)
        assert failure["error"]["code"] == "PROJECT_CHECK_FAILED"
        assert "no immutable image ID; run build_coding_image.py" in failure["error"]["message"]
    else:
        _run(
            [str(cayu), "check", "--fail-on", "warning", "--json"],
            cwd=project,
            environment=environment,
        )
    _run(
        [str(python), "-m", "pytest", "-q", "tests/test_coding_composition.py"],
        cwd=project,
        environment=environment,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cayu", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--postgres-dsn",
        help="exercise the generated Postgres application against this disposable DSN",
    )
    args = parser.parse_args(argv)
    cayu = Path(os.path.abspath(args.cayu))
    python = Path(os.path.abspath(args.python))
    if not cayu.is_file():
        parser.error(f"--cayu does not exist: {cayu}")
    if not python.is_file():
        parser.error(f"--python does not exist: {python}")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "PYTHONPATH",
            "CAYU_PROVIDER",
            "CAYU_DATABASE_URL",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
        }
    }
    environment["PYTHONNOUSERSITE"] = "1"

    with tempfile.TemporaryDirectory(prefix="cayu-wheel-applications-") as raw_root:
        root = Path(raw_root)
        topics = json.loads(
            _run(
                [str(cayu), "guide", "--json"],
                cwd=root,
                environment=environment,
                capture=True,
            )
        )
        assert "applications" in {item["topic"] for item in topics["topics"]}
        _run(
            [str(cayu), "guide", "applications#convention", "--json"],
            cwd=root,
            environment=environment,
            capture=True,
        )

        dry = json.loads(
            _run(
                [
                    str(cayu),
                    "new",
                    "planned",
                    "--preset",
                    "agent",
                    "--with",
                    "observability",
                    "--dry-run",
                    "--json",
                    "--dir",
                    str(root),
                ],
                cwd=root,
                environment=environment,
                capture=True,
            )
        )
        assert dry["status"] == "planned"
        assert not (root / "planned").exists()

        agent = _create(cayu, root, "agent", environment=environment)
        _verify_agent(
            cayu=cayu,
            python=python,
            project=agent,
            environment=environment,
        )

        for provider in ("openai", "anthropic", "openrouter", "openai-subscription"):
            selected = _create(
                cayu,
                root,
                f"provider-{provider}",
                "--provider",
                provider,
                environment=environment,
            )
            _verify_agent(
                cayu=cayu,
                python=python,
                project=selected,
                environment=environment,
            )

        minimal = _create(cayu, root, "minimal", "--minimal", environment=environment)
        _verify_minimal(
            cayu=cayu,
            python=python,
            project=minimal,
            environment=environment,
        )

        postgres = _create(
            cayu,
            root,
            "postgres-agent",
            "--database",
            "postgres",
            environment=environment,
        )
        _verify_agent(
            cayu=cayu,
            python=python,
            project=postgres,
            environment=environment,
            postgres=True,
            postgres_dsn=args.postgres_dsn,
        )

        coding = _create(
            cayu,
            root,
            "coding",
            "--preset",
            "coding",
            environment=environment,
        )
        _verify_coding(
            cayu=cayu,
            python=python,
            project=coding,
            environment=environment,
        )

        docker_coding = _create(
            cayu,
            root,
            "docker-coding",
            "--preset",
            "coding",
            "--execution",
            "docker",
            environment=environment,
        )
        _verify_coding(
            cayu=cayu,
            python=python,
            project=docker_coding,
            environment=environment,
            docker=True,
        )

        service_environment = {
            **environment,
            "PRODUCT_AUTH_TOKENS_JSON": (
                '{"customer-token":{"tenant_id":"tenant-a","subject_id":"alice"}}'
            ),
            "CAYU_OPERATOR_BEARER_TOKEN": "operator-token",
        }
        service = _create(
            cayu,
            root,
            "service",
            "--preset",
            "service",
            environment=service_environment,
        )
        _run(
            [
                str(cayu),
                "check",
                "--deploy",
                "--fail-on",
                "warning",
                "--json",
            ],
            cwd=service,
            environment=service_environment,
        )
        _run(
            [str(python), "-m", "pytest", "-q", "tests/test_public_service_security.py"],
            cwd=service,
            environment=service_environment,
        )

        postgres_service = root / "postgres-service"
        rejected = _run(
            [
                str(cayu),
                "new",
                postgres_service.name,
                "--preset",
                "service",
                "--database",
                "postgres",
                "--dir",
                str(root),
                "--json",
            ],
            cwd=root,
            environment=service_environment,
            allowed_exit_codes=frozenset({1}),
        )
        rejected_payload = json.loads(rejected)
        assert rejected_payload["error"]["code"] == "UNSUPPORTED_ADAPTER"
        assert not postgres_service.exists()

    print("built-wheel Cayu application convention smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
