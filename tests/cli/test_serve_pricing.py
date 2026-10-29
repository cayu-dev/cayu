"""Serve pricing is validated once and follows project/CLI/reload precedence."""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from types import ModuleType

import pytest

from cayu.budgets.pricing import ModelPrice, PriceBook, dump_price_book
from cayu.cli import main


def book(version):
    return dump_price_book(
        PriceBook(
            price_book_version=version,
            prices=(
                ModelPrice.fixed(
                    provider_name="demo",
                    model="demo",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                    match="exact",
                ),
            ),
        )
    )


def project(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "priced_agent:build_app"\n[tool.cayu.serve]\nprice_book = "prices.json"\n'
    )
    (tmp_path / "priced_agent.py").write_text(
        'from cayu import CayuApp\nfrom pathlib import Path\ndef build_app():\n Path("built").touch()\n return CayuApp(enable_logging=False)\n'
    )
    (tmp_path / "prices.json").write_text(book("project"))
    (tmp_path / "nested").mkdir()


@pytest.mark.parametrize("override", [False, True])
def test_serve_project_relative_price_book_and_cli_override(
    tmp_path, monkeypatch, capsys, override
):
    project(tmp_path)
    nested = tmp_path / "nested"
    (nested / "override.json").write_text(book("override"))
    monkeypatch.chdir(nested)
    monkeypatch.delitem(sys.modules, "priced_agent", raising=False)
    launched = []
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda server, **kwargs: launched.append(server)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    assert main(["serve", "--dev", *(["--price-book", "override.json"] if override else [])]) == 0
    config = launched[0].state.cayu_server_config
    assert config.dashboard.runtime_config["priceBook"]["price_book_version"] == (
        "override" if override else "project"
    )
    assert config.access.kind == "open"
    assert (tmp_path / "built").exists()
    capsys.readouterr()


@pytest.mark.parametrize("value", ['{"secret-canary": true}', "[]", "{broken", '{"prices": [NaN]}'])
def test_invalid_price_book_refused_without_factory_or_secret_echo(
    tmp_path, monkeypatch, capsys, value
):
    project(tmp_path)
    (tmp_path / "prices.json").write_text(value)
    monkeypatch.chdir(tmp_path)
    assert main(["serve", "--dev"]) == 1
    assert not (tmp_path / "built").exists()
    assert "secret-canary" not in capsys.readouterr().err


def test_price_book_reload_snapshot_survives_new_worker(tmp_path, monkeypatch):
    import os

    from cayu.cli import _serve_reload
    from cayu.cli.serve import parse_serve_arguments

    args = parse_serve_arguments(["--dev", "--price-book", str(tmp_path / "prices.json")])
    observed = []

    class Config:
        def __init__(self, *a, **kw):
            observed.append(json.loads(os.environ[_serve_reload._RELOAD_ENV]))

        def bind_socket(self):
            from contextlib import nullcontext

            return nullcontext(None)

    class Supervisor:
        def __init__(self, *a, **kw):
            pass

        def run(self):
            pass

    class Server:
        def __init__(self, config):
            pass

        def run(self, sockets=None):
            pass

    monkeypatch.setattr(_serve_reload, "WatchFilesReload", Supervisor)
    monkeypatch.setattr(_serve_reload, "_ReloadServer", Server)
    uvicorn = ModuleType("uvicorn")
    uvicorn.Config = Config
    _serve_reload.run_reload(uvicorn, args, tmp_path)
    restored = parse_serve_arguments(observed[0]["arguments"])
    assert restored.price_book == args.price_book
    assert restored.dev is True


def test_service_factory_rejects_price_override_before_loading_service(
    tmp_path, monkeypatch, capsys
):
    project(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(
            "[tool.cayu.serve]",
            'service_factory = "must_not_load:build_service"\n[tool.cayu.serve]',
        )
    )
    monkeypatch.chdir(tmp_path)
    assert main(["serve", "--dev"]) == 1
    assert "must_not_load" not in sys.modules
    assert not (tmp_path / "built").exists()
    assert "owns its dashboard configuration" in capsys.readouterr().err
