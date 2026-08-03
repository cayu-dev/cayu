"""Run model-catalog maintenance with the latest public Cayu package.

This bootstrapper intentionally does not import :mod:`cayu`. It creates a disposable
environment, runs ``pip install --upgrade cayu``, and invokes the checkout's repository-only
maintenance code with that installed runtime and the checkout's catalog data.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPOSITORY_ROOT / "src/cayu/data/default_model_catalog.json"
PRICE_BOOK_PATH = REPOSITORY_ROOT / "src/cayu/data/default_price_book.json"
_PUBLIC_RUNTIME_PROBE = (
    "import cayu; print(f'Using public Cayu {cayu.__version__} from {cayu.__file__}')"
)


def _commands(
    *,
    uv_executable: str,
    environment_dir: Path,
    passthrough: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    public_python = environment_dir / "bin/python"
    return (
        (
            uv_executable,
            "venv",
            "--python",
            "3.12",
            "--seed",
            str(environment_dir),
        ),
        (
            str(public_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "cayu",
        ),
        (str(public_python), "-c", _PUBLIC_RUNTIME_PROBE),
        (
            str(public_python),
            "-m",
            "maintenance.model_catalog.refresh",
            "--openai-subscription",
            "--source-catalog",
            str(CATALOG_PATH),
            "--source-price-book",
            str(PRICE_BOOK_PATH),
            *passthrough,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise RuntimeError("uv is required to create the disposable refresh environment")
    if shutil.which("agent-browser") is None:
        raise RuntimeError("agent-browser is required for model-catalog verification")

    clean_env = os.environ.copy()
    clean_env.pop("PYTHONHOME", None)
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("VIRTUAL_ENV", None)
    passthrough = tuple(sys.argv[1:] if argv is None else argv)
    with TemporaryDirectory(prefix="cayu-model-catalog-refresh-") as temporary_directory:
        commands = _commands(
            uv_executable=uv_executable,
            environment_dir=Path(temporary_directory) / "venv",
            passthrough=passthrough,
        )
        for command in commands:
            subprocess.run(command, cwd=REPOSITORY_ROOT, env=clean_env, check=True)


if __name__ == "__main__":
    main()
