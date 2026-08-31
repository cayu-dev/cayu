from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import cayu.cli.project_control_plane as project_control_plane_cli
from cayu import CayuApp
from cayu.cli.project import ProjectError
from cayu.cli.project_control_plane import build_project_control_plane_context
from cayu.project_control_plane import resolve_project_control_plane_context
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
        resolved = resolve_project_control_plane_context(
            context,
            CayuApp(enable_logging=False),
        )
        assert resolved is not None
        assert resolved.project_id == "example-project"
        assert resolved.application_release_id == f"manifest-{resolved.app_manifest_fingerprint}"
        assert resolved.store_backend == "sqlite"
        assert resolved.store_source == "project"
        assert context.project_identity_configured is True
        assert context.eval_store_configured is True
        assert "data/cayu.db" not in repr(context)
    finally:
        asyncio.run(context.close())


def test_project_context_uses_explicit_release_without_exposing_store_credentials(
    tmp_path: Path,
) -> None:
    _project(
        tmp_path,
        store=('\n[tool.cayu.session_store]\nbackend = "postgres"\nenv = "PROJECT_DATABASE_URL"\n'),
    )
    secret = "postgresql://operator:do-not-print@db.example/cayu"
    context = build_project_control_plane_context(
        tmp_path,
        mode="production",
        environ={
            "PROJECT_DATABASE_URL": secret,
            "CAYU_RELEASE_ID": "release-42",
        },
    )
    try:
        resolved = resolve_project_control_plane_context(
            context,
            CayuApp(enable_logging=False),
        )
        assert resolved is not None
        assert resolved.application_release_id == "release-42"
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


def test_project_context_rejects_invalid_release_without_echoing_it(tmp_path: Path) -> None:
    _project(tmp_path)
    secret = "do-not-print\n"

    with pytest.raises(ProjectError) as excinfo:
        build_project_control_plane_context(
            tmp_path,
            mode="production",
            environ={"CAYU_RELEASE_ID": secret},
        )

    assert secret not in str(excinfo.value)


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


def test_project_context_rejects_identity_colliding_with_a_workload_secret(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    context = build_project_control_plane_context(
        tmp_path,
        mode="production",
        environ={"CAYU_RELEASE_ID": "secret-release-value"},
    )
    try:
        app = CayuApp(
            enable_logging=False,
            secret_redactor=SecretRedactor("secret-release-value"),
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
