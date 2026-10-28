"""The parts of a ``pyproject.toml`` that decide what a uv sync installs.

A coding image installs its environment with ``uv sync --frozen`` from
``pyproject.toml`` and ``uv.lock``. Tool settings such as ``[tool.pytest]`` or
``[tool.ruff]`` are read from the workspace copy at run time and never change
what was installed, so a dependency input with the ``pyproject_environment``
digest scope hashes only these parts:

- ``[project]``: ``name``, ``requires-python``, ``dependencies``,
  ``optional-dependencies`` and ``dynamic``;
- ``[build-system]``, ``[dependency-groups]`` and ``[tool.uv]`` in full.
"""

from __future__ import annotations

import datetime
import json
import tomllib
from hashlib import sha256

PYPROJECT_ENVIRONMENT_DIGEST_SCHEMA = "cayu.pyproject_environment.v1"
_PROJECT_KEYS = ("name", "requires-python", "dependencies", "optional-dependencies", "dynamic")
_WHOLE_TABLES = ("build-system", "dependency-groups")


def pyproject_environment_sha256(content: bytes) -> str:
    """Return the digest of the environment-relevant parts of ``content``.

    Raises ``ValueError`` when ``content`` is not UTF-8 TOML.
    """

    if type(content) is not bytes:
        raise TypeError("pyproject.toml content must be bytes.")
    try:
        document = tomllib.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, RecursionError):
        # The parser recurses, so very deeply nested arrays exhaust the stack.
        raise ValueError("pyproject.toml is not valid UTF-8 TOML.") from None
    selected: dict[str, object] = {"schema": PYPROJECT_ENVIRONMENT_DIGEST_SCHEMA}
    project = document.get("project")
    if isinstance(project, dict):
        selected["project"] = {key: project[key] for key in _PROJECT_KEYS if key in project}
    elif project is not None:
        selected["project"] = project
    for table in _WHOLE_TABLES:
        if table in document:
            selected[table] = document[table]
    tool = document.get("tool")
    if isinstance(tool, dict) and "uv" in tool:
        selected["tool.uv"] = tool["uv"]
    try:
        encoded = json.dumps(
            selected,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=_toml_temporal,
        ).encode("utf-8")
    except RecursionError:
        raise ValueError("pyproject.toml nests values too deeply.") from None
    return "sha256:" + sha256(encoded).hexdigest()


def _toml_temporal(value: object) -> object:
    # TOML dates and times are the only parsed values JSON has no form for.
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return {"toml_" + type(value).__name__: value.isoformat()}
    raise TypeError(f"Unsupported TOML value {type(value).__name__}.")
