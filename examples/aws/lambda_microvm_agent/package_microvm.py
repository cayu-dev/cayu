"""Build the Lambda MicroVM image code artifact expected by infra.yaml."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
SIDECAR = EXAMPLES / "lambda_microvm_sidecar"


def package(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(HERE / "microvm.Dockerfile", "Dockerfile")
        archive.write(HERE / "microvm-entrypoint.sh", "entrypoint.sh")
        for name in ("__init__.py", "app.py", "supervisor.py", "requirements.txt"):
            archive.write(SIDECAR / name, name)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(package(args.output))


if __name__ == "__main__":
    main()
