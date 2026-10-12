"""Run the recorded storage selection against source or an installed wheel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


class ImportGuard:
    def __init__(self, package: Path):
        self.package = package.resolve()

    def check(self):
        import cayu

        assert Path(cayu.__file__).resolve().is_relative_to(self.package)
        count = 0
        for name, module in tuple(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if (name == "cayu" or name.startswith("cayu.")) and filename:
                assert Path(filename).resolve().is_relative_to(self.package), (name, filename)
                count += 1
        return count

    def pytest_sessionstart(self, session):
        self.check()

    def pytest_sessionfinish(self, session, exitstatus):
        print(f"\nImport origin guard: {self.check()} Cayu modules from selected package.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-package", type=Path)
    args, pytest_args = parser.parse_known_args()
    sys.path.insert(0, str(ROOT))
    package = args.installed_package or ROOT / "src/cayu"
    if args.installed_package is None:
        sys.path.insert(0, str(ROOT / "src"))
    import pytest

    selection = json.loads((HERE / "test-selection.json").read_text())
    return pytest.main(
        ["-c", str(ROOT / "pyproject.toml"), "-o", "pythonpath=", "-q"]
        + [str(ROOT / item) for item in selection]
        + pytest_args,
        plugins=[ImportGuard(package)],
    )


if __name__ == "__main__":
    raise SystemExit(main())
