from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Top-level packages that only optional extras (or the dev extra) install.
EXTRA_ONLY = (
    "IPython",
    "PIL",
    "boto3",
    "botocore",
    "e2b",
    "fastapi",
    "google",
    "httpx2",
    "mcp",
    "microsandbox",
    "opentelemetry",
    "playwright",
    "psycopg",
    "psycopg_pool",
    "pydantic_settings",
    "pypdf",
    "sse_starlette",
    "starlette",
    "uvicorn",
    "websockets",
)
BLOCK = f"""
import importlib.abc, sys
class _WithoutExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {EXTRA_ONLY!r}:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
sys.meta_path.insert(0, _WithoutExtras())
"""


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", BLOCK + code],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_wildcard_imports_work_without_optional_extras():
    packages = json.loads((ROOT / "docs/public-api-packages.json").read_text())
    code = "\n".join(f"exec('from {name} import *', {{}})" for name in packages)

    completed = _run(code)

    assert completed.returncode == 0, completed.stderr


def test_optional_extra_names_stay_importable_by_name_with_a_hint():
    completed = _run(
        """
import importlib
import cayu

assert 'SQLiteSessionStore' in cayu.__all__
for package_name, name, extra in (
    ('cayu', 'PostgresSessionStore', 'postgres'),
    ('cayu.storage', 'PostgresTaskStore', 'postgres'),
    ('cayu.collaboration', 'PostgresCollaborationStore', 'postgres'),
    ('cayu', 'PostgresEvalStore', 'postgres'),
    ('cayu', 'PostgresProductOperationStore', 'postgres'),
    ('cayu.storage', 'SQLiteProductOperationStore', 'server'),
):
    package = importlib.import_module(package_name)
    assert name not in package.__all__, (package_name, name)
    assert name in dir(package), (package_name, name)
    try:
        exec(f'from {package_name} import {name}', {})
    except RuntimeError as exc:
        assert f'pip install "cayu[{extra}]"' in str(exc), str(exc)
    else:
        raise AssertionError(f'{package_name}.{name} imported without its extra')
"""
    )

    assert completed.returncode == 0, completed.stderr
