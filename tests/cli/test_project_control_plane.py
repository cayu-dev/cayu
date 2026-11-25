from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import cayu.cli.project_control_plane as project_control_plane_cli
from cayu import CayuApp
from cayu.cli.project import ProjectError
from cayu.cli.project_control_plane import build_project_control_plane_context
from cayu.project_control_plane import resolve_project_control_plane_context
from cayu.releases import RELEASE_ID_ENV, ReleaseIdentity
from cayu.storage._diagnostic_inspection import diagnostic_store_inspection
from cayu.storage.evals_postgres import PostgresEvalStore
from cayu.storage.migrations import LATEST_REVISION, SchemaMode
from cayu.vaults import SecretRedactor


def _project(root: Path, *, name: str | None = "Example.Project", store: str = "") -> None:
    project = "" if name is None else f'[project]\nname = "{name}"\n\n'
    (root / "pyproject.toml").write_text(
        project + '[tool.cayu]\nfactory = "app:build_app"\n' + store,
        encoding="utf-8",
    )


def test_project_context_normalizes_identity_and_uses_manifest_release_fallback(
    tmp_path: Path,
) -> None:
    _project(
        tmp_path,
        store='\n[tool.cayu.session_store]\nbackend = "sqlite"\npath = "data/cayu.db"\n',
    )
    context = build_project_control_plane_context(
        tmp_path,
        mode="production",
        environ={},
    )
    try:
        app = CayuApp(enable_logging=False)
        resolved = resolve_project_control_plane_context(context, app)
        assert resolved is not None
        assert resolved.project_id == "example-project"
        assert resolved.application_release_id == f"manifest-{resolved.app_manifest_fingerprint}"
        assert app.release_identity is None
        assert resolved.store_backend == "sqlite"
        assert resolved.store_source == "project"
        assert context.project_identity_configured is True
        assert context.eval_store_configured is True
        assert "data/cayu.db" not in repr(context)
    finally:
        asyncio.run(context.close())


def test_diagnostic_project_context_does_not_mutate_existing_sqlite_eval_store(
    tmp_path: Path,
) -> None:
    _project(
        tmp_path,
        store='\n[tool.cayu.session_store]\nbackend = "sqlite"\npath = "data/cayu.db"\n',
    )
    created = build_project_control_plane_context(
        tmp_path,
        mode="production",
        environ={},
    )
    asyncio.run(created.close())

    database = tmp_path / "data" / "cayu.db"
    before_bytes = database.read_bytes()
    before_stat = database.stat()
    before_entries = tuple(sorted(path.name for path in database.parent.iterdir()))

    with diagnostic_store_inspection() as inspection:
        inspected = build_project_control_plane_context(
            tmp_path,
            mode="production",
            environ={},
        )
        try:
            assert inspected.eval_store_configured is True
            inspection.verify()
        finally:
            asyncio.run(inspected.close())

    after_stat = database.stat()
    assert database.read_bytes() == before_bytes
    assert (after_stat.st_size, after_stat.st_mtime_ns) == (
        before_stat.st_size,
        before_stat.st_mtime_ns,
    )
    assert tuple(sorted(path.name for path in database.parent.iterdir())) == before_entries


def test_diagnostic_project_context_forces_postgres_eval_store_read_only(
    tmp_path: Path,
) -> None:
    _project(
        tmp_path,
        store=('\n[tool.cayu.session_store]\nbackend = "postgres"\nenv = "PROJECT_DATABASE_URL"\n'),
    )

    with diagnostic_store_inspection() as inspection:
        context = build_project_control_plane_context(
            tmp_path,
            mode="production",
            environ={"PROJECT_DATABASE_URL": "postgresql://example/cayu"},
        )
        try:
            resolved = resolve_project_control_plane_context(
                context,
                CayuApp(enable_logging=False),
            )
            assert resolved is not None
            store = resolved.eval_store
            assert isinstance(store, PostgresEvalStore)
            assert store._read_only is True
            assert store._schema_mode is SchemaMode.VALIDATE
            inspection.verify()
        finally:
            asyncio.run(context.close())


def test_project_context_uses_the_stated_release_without_exposing_store_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _project(
        tmp_path,
        store=('\n[tool.cayu.session_store]\nbackend = "postgres"\nenv = "PROJECT_DATABASE_URL"\n'),
    )
    secret = "postgresql://operator:do-not-print@db.example/cayu"
    monkeypatch.setenv(RELEASE_ID_ENV, "release-42")
    context = build_project_control_plane_context(
        tmp_path,
        mode="production",
        environ={"PROJECT_DATABASE_URL": secret},
    )
    try:
        app = CayuApp(enable_logging=False)
        resolved = resolve_project_control_plane_context(context, app)
        assert resolved is not None
        assert resolved.application_release_id == "release-42"
        assert app.release_identity == ReleaseIdentity(id="release-42")
        assert resolved.store_backend == "postgres"
        assert secret not in repr(context)
        assert secret not in str(resolved.safe_summary())
    finally:
        asyncio.run(context.close())


def test_project_context_uses_only_the_trusted_local_default(
    tmp_path: Path,
) -> None:
    _project(tmp_path)

    production = build_project_control_plane_context(
        tmp_path,
        mode="production",
        environ={},
    )
    development = build_project_control_plane_context(
        tmp_path,
        mode="development",
        environ={},
    )
    try:
        assert production.eval_store_configured is False
        assert development.eval_store_configured is True
        resolved = resolve_project_control_plane_context(
            development,
            CayuApp(enable_logging=False),
        )
        assert resolved is not None
        assert resolved.trusted_local_development is True
        assert resolved.store_source == "local-development-default"
        assert (tmp_path / "data" / "cayu.db").is_file()
    finally:
        asyncio.run(production.close())
        asyncio.run(development.close())


def test_project_context_parses_an_explicit_bounded_default_judge(tmp_path: Path) -> None:
    _project(
        tmp_path,
        store="""

[tool.cayu.evals]
price_book = "bundled-public"

[tool.cayu.evals.default_judge]
provider = "anthropic"
model = "claude-sonnet-4-6"
privacy_policy = "public-and-transcript"
allow_same_model = false
timeout_seconds = 45
max_input_tokens = 4096
max_output_tokens = 1024
max_total_tokens = 5120
max_estimated_cost = "0.1"
cost_currency = "USD"
""",
    )

    context = build_project_control_plane_context(
        tmp_path,
        mode="production",
        environ={},
    )
    try:
        resolved = resolve_project_control_plane_context(
            context,
            CayuApp(enable_logging=False),
        )
        assert resolved is not None
        assert context.eval_judge_configured is True
        assert context.eval_pricing_configured is True
        assert resolved.eval_judge_configuration is not None
        assert resolved.eval_judge_configuration.provider_name == "anthropic"
        assert resolved.eval_judge_configuration.model == "claude-sonnet-4-6"
        assert resolved.eval_judge_configuration.privacy_policy == "public-and-transcript"
        assert resolved.eval_judge_configuration.allow_same_model is False
        assert resolved.eval_judge_configuration.timeout_seconds == 45
        assert resolved.eval_judge_configuration.max_input_tokens == 4096
        assert resolved.eval_judge_configuration.max_output_tokens == 1024
        assert resolved.eval_judge_configuration.max_total_tokens == 5120
        assert resolved.eval_judge_configuration.max_estimated_cost == "0.1"
        assert resolved.eval_judge_configuration.cost_currency == "USD"
        assert resolved.eval_price_book is not None
        assert resolved.safe_summary()["eval_judge"] == {"configured": True}
        assert resolved.safe_summary()["eval_pricing"] == {"configured": True}
        assert "claude-sonnet-4-6" not in repr(context)
    finally:
        asyncio.run(context.close())


@pytest.mark.parametrize(
    "judge",
    [
        "provider = 'openai'\nmodel = 'gpt-5'\nprivacy_policy = 'public-only'\n",
        (
            "provider = 'openai'\nmodel = 'gpt-5'\n"
            "privacy_policy = 'ambient'\nallow_same_model = false\n"
        ),
        (
            "provider = 'openai'\nmodel = 'gpt-5'\n"
            "privacy_policy = ['public-only']\nallow_same_model = false\n"
        ),
        (
            "provider = 'openai'\nmodel = 'gpt-5'\n"
            "privacy_policy = 'public-only'\nallow_same_model = 'yes'\n"
        ),
        (
            "provider = 'openai'\nmodel = 'gpt-5'\n"
            "privacy_policy = 'public-only'\nallow_same_model = false\n"
            "max_total_tokens = 10\nmax_input_tokens = 11\n"
        ),
        (
            "provider = 'openai'\nmodel = 'gpt-5'\n"
            "privacy_policy = 'public-only'\nallow_same_model = false\n"
            "max_estimated_cost = '1e-3'\ncost_currency = 'USD'\n"
        ),
    ],
)
def test_project_context_rejects_incomplete_or_unsafe_default_judge(
    tmp_path: Path,
    judge: str,
) -> None:
    _project(
        tmp_path,
        store=f"\n[tool.cayu.evals.default_judge]\n{judge}",
    )

    with pytest.raises(ProjectError, match=r"\[tool\.cayu\.evals\.default_judge\]"):
        build_project_control_plane_context(tmp_path, mode="production", environ={})


def test_project_context_rejects_an_unknown_eval_price_book(tmp_path: Path) -> None:
    _project(
        tmp_path,
        store='\n[tool.cayu.evals]\nprice_book = "ambient"\n',
    )

    with pytest.raises(ProjectError, match=r'price_book must be "bundled-public"'):
        build_project_control_plane_context(tmp_path, mode="production", environ={})


def test_project_context_rejects_unknown_eval_configuration_keys(tmp_path: Path) -> None:
    _project(
        tmp_path,
        store='\n[tool.cayu.evals]\nprice_bok = "bundled-public"\n',
    )

    with pytest.raises(ProjectError, match=r"unsupported keys: price_bok"):
        build_project_control_plane_context(tmp_path, mode="production", environ={})


def test_project_eval_targets_are_read_in_order_without_importing_them(tmp_path: Path) -> None:
    _project(
        tmp_path,
        store=(
            "\n[tool.cayu.evals]\n"
            'targets = ["evals.questions:build_corpus_eval", "evals.refunds:plan.build"]\n'
        ),
    )

    assert project_control_plane_cli.project_eval_target_references(tmp_path) == (
        "evals.questions:build_corpus_eval",
        "evals.refunds:plan.build",
    )
    context = build_project_control_plane_context(tmp_path, mode="production", environ={})
    try:
        assert context.eval_targets == ()
    finally:
        asyncio.run(context.close())


def test_a_project_without_eval_targets_declares_none(tmp_path: Path) -> None:
    _project(tmp_path)

    assert project_control_plane_cli.project_eval_target_references(tmp_path) == ()


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        ('"evals:build"', r"must be a list"),
        ('["evals.build"]', r"'evals.build' must be \"module:function\""),
        ('["evals:build()"]', r"must be \"module:function\""),
        ('["evals:build", "evals:build"]', r"lists an entry twice"),
        ("[" + ", ".join(f'"evals:build_{index}"' for index in range(17)) + "]", r"at most 16"),
    ],
)
def test_project_context_rejects_malformed_eval_targets(
    tmp_path: Path, targets: str, message: str
) -> None:
    _project(tmp_path, store=f"\n[tool.cayu.evals]\ntargets = {targets}\n")

    with pytest.raises(ProjectError, match=message):
        build_project_control_plane_context(tmp_path, mode="production", environ={})


def test_the_eval_worker_must_be_a_declared_worker_with_targets_to_run(tmp_path: Path) -> None:
    workers = '\n[tool.cayu.workers]\nevals = "workers:evals"\n'
    _project(
        tmp_path,
        store=workers + '\n[tool.cayu.evals]\ntargets = ["evals:plan"]\nworker = "evals"\n',
    )
    assert project_control_plane_cli.project_eval_worker(tmp_path) == "evals"
    context = build_project_control_plane_context(tmp_path, mode="production", environ={})
    try:
        assert context.eval_worker == "evals"
    finally:
        asyncio.run(context.close())

    _project(tmp_path, store=workers + '\n[tool.cayu.evals]\nworker = "evals"\n')
    with pytest.raises(ProjectError, match=r"declare them in \[tool.cayu.evals\].targets"):
        build_project_control_plane_context(tmp_path, mode="production", environ={})

    _project(
        tmp_path,
        store=workers + '\n[tool.cayu.evals]\ntargets = ["evals:plan"]\nworker = "timers"\n',
    )
    with pytest.raises(ProjectError, match=r"names 'timers', which \[tool.cayu.workers\]"):
        build_project_control_plane_context(tmp_path, mode="production", environ={})


def test_project_context_requires_pricing_for_a_judge_cost_ceiling(tmp_path: Path) -> None:
    _project(
        tmp_path,
        store="""

[tool.cayu.evals.default_judge]
provider = "openai"
model = "gpt-5"
privacy_policy = "public-only"
allow_same_model = false
max_estimated_cost = "0.1"
cost_currency = "USD"
""",
    )

    with pytest.raises(ProjectError, match="cost ceiling requires"):
        build_project_control_plane_context(tmp_path, mode="production", environ={})


def test_project_context_reports_missing_project_identity_without_guessing(
    tmp_path: Path,
) -> None:
    _project(tmp_path, name=None)

    context = build_project_control_plane_context(
        tmp_path,
        mode="production",
        environ={},
    )
    try:
        assert context.project_identity_configured is False
    finally:
        asyncio.run(context.close())


def test_project_context_uses_the_release_the_app_records_on_its_runs(tmp_path: Path) -> None:
    _project(tmp_path)
    context = build_project_control_plane_context(tmp_path, mode="production", environ={})
    try:
        app = CayuApp(
            enable_logging=False,
            release_identity=ReleaseIdentity(id="release-configured"),
        )
        resolved = resolve_project_control_plane_context(context, app)
        assert resolved is not None
        assert resolved.application_release_id == "release-configured"
    finally:
        asyncio.run(context.close())


def test_project_context_redacts_store_initialization_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _project(
        tmp_path,
        store='\n[tool.cayu.session_store]\nbackend = "sqlite"\npath = "data/cayu.db"\n',
    )
    secret = "do-not-print-store-detail"

    def fail_store(_target):
        raise RuntimeError(secret)

    monkeypatch.setattr(project_control_plane_cli, "_create_eval_store", fail_store)

    with pytest.raises(ProjectError, match="Could not initialize project Evals storage") as excinfo:
        build_project_control_plane_context(tmp_path, mode="production", environ={})

    assert secret not in str(excinfo.value)


_SQLITE_STORE = '\n[tool.cayu.session_store]\nbackend = "sqlite"\npath = "data/cayu.db"\n'


def test_project_context_explains_an_unreadable_local_store(tmp_path: Path) -> None:
    _project(tmp_path, store=_SQLITE_STORE)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "cayu.db").write_bytes(b"not a sqlite database" * 10)

    with pytest.raises(ProjectError) as excinfo:
        build_project_control_plane_context(tmp_path, mode="production", environ={})

    message = str(excinfo.value)
    assert "Could not initialize project Evals storage (DatabaseError: " in message
    assert "file is not a database" in message
    assert "move data/cayu.db and any data/cayu.db-wal and data/cayu.db-shm files" in message
    assert "discards its local sessions and Evals results" in message


def test_project_context_reports_a_store_from_a_newer_schema(tmp_path: Path) -> None:
    _project(tmp_path, store=_SQLITE_STORE)
    context = build_project_control_plane_context(tmp_path, mode="production", environ={})
    asyncio.run(context.close())
    with closing(sqlite3.connect(tmp_path / "data" / "cayu.db")) as connection:
        connection.execute(
            "UPDATE cayu_schema_migrations SET compatible_from = ? "
            "WHERE revision = (SELECT MAX(revision) FROM cayu_schema_migrations)",
            (LATEST_REVISION + 1,),
        )
        connection.commit()

    with pytest.raises(ProjectError) as excinfo:
        build_project_control_plane_context(tmp_path, mode="production", environ={})

    message = str(excinfo.value)
    assert "(SchemaTooNew: Database is at schema revision" in message
    assert "Upgrade the application." in message
    assert "move data/cayu.db" in message


def test_project_context_gives_no_reset_hint_for_a_locked_store(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _project(tmp_path, store=_SQLITE_STORE)
    context = build_project_control_plane_context(tmp_path, mode="production", environ={})
    asyncio.run(context.close())

    def locked_store(_target):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(project_control_plane_cli, "_create_eval_store", locked_store)

    with pytest.raises(ProjectError) as excinfo:
        build_project_control_plane_context(tmp_path, mode="production", environ={})

    assert str(excinfo.value) == (
        "Could not initialize project Evals storage (OperationalError: database is locked)."
    )


def test_project_context_shows_why_a_sqlite_store_is_refused(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _project(tmp_path, store=_SQLITE_STORE)
    context = build_project_control_plane_context(tmp_path, mode="production", environ={})
    asyncio.run(context.close())
    monkeypatch.setenv("CAYU_REQUIRE_POSTGRES", "1")

    with pytest.raises(ProjectError) as excinfo:
        build_project_control_plane_context(tmp_path, mode="production", environ={})

    message = str(excinfo.value)
    assert "(PostgresRequiredError: SQLiteEvalStore cannot open because" in message
    assert "move data/cayu.db" not in message


def test_project_context_keeps_the_failure_when_the_store_path_cannot_be_checked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _project(tmp_path, store=_SQLITE_STORE)

    def unreadable_store(_target):
        raise sqlite3.DatabaseError("file is not a database")

    def refuse_stat(_self, *args, **kwargs):
        raise PermissionError("Permission denied")

    monkeypatch.setattr(project_control_plane_cli, "_create_eval_store", unreadable_store)
    monkeypatch.setattr(Path, "exists", refuse_stat)

    with pytest.raises(ProjectError) as excinfo:
        build_project_control_plane_context(tmp_path, mode="production", environ={})

    assert str(excinfo.value) == (
        "Could not initialize project Evals storage (DatabaseError: file is not a database)."
    )


def test_project_context_keeps_unknown_postgres_failure_text_out(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _project(tmp_path)
    dsn = "postgresql://cayu:do-not-print-password@db.example/cayu"

    def fail_store(_target):
        raise ValueError(f"invalid dsn: {dsn}")

    monkeypatch.setattr(project_control_plane_cli, "_create_eval_store", fail_store)

    with pytest.raises(ProjectError) as excinfo:
        build_project_control_plane_context(
            tmp_path, mode="production", environ={"CAYU_DATABASE_URL": dsn}
        )

    assert str(excinfo.value) == "Could not initialize project Evals storage (ValueError)."


def test_project_context_rejects_identity_colliding_with_a_workload_secret(
    tmp_path: Path,
) -> None:
    _project(tmp_path, name="secret-project-value")
    context = build_project_control_plane_context(
        tmp_path,
        mode="production",
        environ={},
    )
    try:
        app = CayuApp(
            enable_logging=False,
            secret_redactor=SecretRedactor("secret-project-value"),
        )
        with pytest.raises(ValueError, match="contains a workload secret"):
            resolve_project_control_plane_context(context, app)
    finally:
        asyncio.run(context.close())


def test_project_context_close_is_idempotent_and_fences_reuse(tmp_path: Path) -> None:
    _project(tmp_path)
    context = build_project_control_plane_context(
        tmp_path,
        mode="development",
        environ={},
    )

    asyncio.run(context.close())
    asyncio.run(context.close())

    with pytest.raises(RuntimeError, match="already closed"):
        resolve_project_control_plane_context(context, CayuApp(enable_logging=False))
