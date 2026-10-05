"""Immutable project resolution for `cayu cloud deploy`."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import posixpath
import re
import stat
import subprocess
import tarfile
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import cast
from urllib.parse import quote, urlsplit

import httpx

from cayu._operator_credentials import (
    ENVIRONMENT_OPERATOR_AUTH_TARGET,
    LAST_RELEASE_WITHOUT_ENVIRONMENT_AUTH,
    OPERATOR_PASSWORD_VARIABLE,
    OPERATOR_USERNAME_VARIABLE,
)
from cayu.cli._cloud_api import CloudApiError
from cayu.cli._serve_readiness import ServeSetupPlan, plan_serve_setup

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_APPLICATION = re.compile(r"[a-z0-9][a-z0-9-]{6,61}[a-z0-9]\Z")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,127}\Z")
_CAPABILITY = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_POLICY_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PROCESS_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\Z")
_GITHUB_OWNER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")
_GITHUB_REPOSITORY = re.compile(r"[A-Za-z0-9._-]{1,100}\Z")
_SLUG_SEPARATOR = re.compile(r"[^a-z0-9]+")
_SOURCE_BUNDLE_PREFIX = "cayu-cloud://source-bundles/sha256/"
_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
_SAFE_ENV_TEMPLATES = frozenset({".env.example", ".env.sample", ".env.template"})


def is_application_slug(value: str) -> bool:
    return _APPLICATION.fullmatch(value) is not None


@dataclass(frozen=True)
class CloudProcess:
    command: str
    cpu_millis: int | None = field(default=None, kw_only=True)
    memory_mb: int | None = field(default=None, kw_only=True)


# Cayu Cloud's web readiness probe defaults and bounds; keep them identical to Cloud.
_READY_TIMEOUT_SECONDS = 2
_READY_TIMEOUT_RANGE = (1, 30)
_READY_START_PERIOD_SECONDS = 180
_READY_START_PERIOD_RANGE = (0, 300)


def _ready_path(value: object) -> str:
    # The probe's urllib refuses spaces, control characters and non-ASCII, so such
    # a path would fail every check and restart the web process in a loop.
    if (
        type(value) is not str
        or not 1 <= len(value) <= 1024
        or not value.startswith("/")
        or value.startswith("//")
        or any(not 33 <= ord(char) <= 126 for char in value)
    ):
        raise CloudApiError(
            "manifest_invalid",
            "Runtime web ready_path must be a local absolute HTTP path of printable ASCII "
            "without spaces; percent-encode anything else.",
        )
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or "#" in value:
        raise CloudApiError(
            "manifest_invalid", "Runtime web ready_path must not contain an origin or fragment."
        )
    return value


def _ready_seconds(value: object, *, name: str, bounds: tuple[int, int]) -> int:
    low, high = bounds
    if type(value) is not int or not low <= value <= high:
        raise CloudApiError("manifest_invalid", f"Runtime web {name} must be {low}-{high} seconds.")
    return value


@dataclass(frozen=True)
class CloudWebProcess(CloudProcess):
    port: int
    idle_timeout_seconds: int | None = None
    ready_path: str = "/"
    ready_timeout_seconds: int = _READY_TIMEOUT_SECONDS
    ready_start_period_seconds: int = _READY_START_PERIOD_SECONDS


@dataclass(frozen=True)
class CloudSchedule:
    name: str
    command: str
    expression: str
    cpu_millis: int | None = None
    memory_mb: int | None = None


@dataclass(frozen=True)
class CloudLocalFile:
    """A `[storage] local_files` acknowledgement of a rebuildable file under /data."""

    path: str
    reason: str


_LOCAL_FILES_LIMIT = 50
_STORAGE_TABLE_ERROR = "[storage] must contain only local_files, an array of at most 50 tables."
_LOCAL_FILE_ENTRY_ERROR = (
    "Each [storage] local_files entry needs a path (1-256 characters) "
    "and a reason (1-500 characters)."
)


def _local_files(storage: object) -> tuple[CloudLocalFile, ...]:
    # Mirrors Cayu Cloud's source admission so an invalid acknowledgement fails here instead
    # of as storage_acknowledgement_invalid after upload.
    if storage is None:
        return ()
    if not isinstance(storage, dict) or set(storage) - {"local_files"}:
        raise CloudApiError("manifest_invalid", _STORAGE_TABLE_ERROR)
    local_files = cast("dict[str, object]", storage).get("local_files", [])
    if not isinstance(local_files, list) or len(local_files) > _LOCAL_FILES_LIMIT:
        raise CloudApiError("manifest_invalid", _STORAGE_TABLE_ERROR)
    entries = []
    for item in local_files:
        if not isinstance(item, dict) or set(item) - {"path", "reason"}:
            raise CloudApiError("manifest_invalid", _LOCAL_FILE_ENTRY_ERROR)
        entry = cast("dict[str, object]", item)
        entries.append(_local_file(entry.get("path"), entry.get("reason")))
    return tuple(entries)


def _local_file(path: object, reason: object) -> CloudLocalFile:
    if (
        not isinstance(path, str)
        or not 1 <= len(path.strip()) <= 256
        or not isinstance(reason, str)
        or not 1 <= len(reason.strip()) <= 500
    ):
        raise CloudApiError("manifest_invalid", _LOCAL_FILE_ENTRY_ERROR)
    return CloudLocalFile(path=path.strip(), reason=reason.strip())


def _process_resources(payload: dict[str, object]) -> dict[str, int]:
    resources = {}
    for name, minimum, maximum in (("cpu_millis", 100, 16_000), ("memory_mb", 128, 131_072)):
        value = payload.get(name)
        if value is not None:
            if type(value) is not int or not minimum <= value <= maximum:
                raise CloudApiError("manifest_invalid", f"Runtime process {name} is invalid.")
            resources[name] = value
    return resources


@dataclass(frozen=True)
class CloudProjectManifest:
    application: str
    name: str
    version: str
    entrypoint: str
    capabilities: tuple[str, ...]
    cpu_millis: int
    memory_mb: int
    timeout_seconds: int
    environment: str
    compatibility: str
    policy_version: str
    web: CloudWebProcess | None = None
    worker: CloudProcess | None = None
    schedules: tuple[CloudSchedule, ...] = ()
    runtime_environment: dict[str, str] | None = None
    local_files: tuple[CloudLocalFile, ...] = ()

    @classmethod
    def load(cls, path: Path) -> CloudProjectManifest:
        try:
            content = path.read_text()
        except OSError as exc:
            raise CloudApiError(
                "manifest_unavailable",
                f"Could not read Cayu Cloud manifest: {path}",
            ) from exc
        return cls.loads(content)

    @classmethod
    def loads(cls, content: str) -> CloudProjectManifest:
        try:
            payload = tomllib.loads(content)
        except tomllib.TOMLDecodeError as exc:
            raise CloudApiError("manifest_invalid", f"Invalid Cayu Cloud TOML: {exc}") from exc
        allowed = {
            "application",
            "capabilities",
            "compatibility",
            "cpu_millis",
            "entrypoint",
            "env",
            "environment",
            "memory_mb",
            "name",
            "policy_version",
            "schema_version",
            "timeout_seconds",
            "version",
            "web",
            "worker",
            "schedules",
            "storage",
        }
        schema_version = payload.get("schema_version")
        if set(payload) - allowed or schema_version not in {1, 2}:
            raise CloudApiError(
                "manifest_invalid",
                "cayu-cloud.toml must use schema_version 1 or 2 and supported fields only.",
            )
        if schema_version == 1 and any(
            key in payload for key in ("env", "schedules", "web", "worker")
        ):
            raise CloudApiError(
                "manifest_invalid",
                "Application processes require cayu-cloud.toml schema_version 2.",
            )
        application = payload.get("application")
        if type(application) is not str:
            raise CloudApiError(
                "manifest_invalid",
                "cayu-cloud.toml application must be a string.",
            )
        try:
            web = payload.get("web")
            worker = payload.get("worker")
            schedules = payload.get("schedules", [])
            runtime_environment = payload.get("env", {})
            if web is not None and not isinstance(web, dict):
                raise TypeError("web must be a table")
            if worker is not None and not isinstance(worker, dict):
                raise TypeError("worker must be a table")
            if not isinstance(schedules, list) or not all(
                isinstance(item, dict) for item in schedules
            ):
                raise TypeError("schedules must be an array of tables")
            if not isinstance(runtime_environment, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in runtime_environment.items()
            ):
                raise TypeError("env must be a string table")
            schedule_tables = [item for item in schedules if isinstance(item, dict)]
            for table_name, table, model in (
                ("web", web, CloudWebProcess),
                ("worker", worker, CloudProcess),
                *(
                    (
                        f'schedule "{item["name"]}"'
                        if type(item.get("name")) is str and item["name"]
                        else f"schedules[{index}]",
                        item,
                        CloudSchedule,
                    )
                    for index, item in enumerate(schedule_tables)
                ),
            ):
                if table is not None:
                    unknown = set(table) - {item.name for item in fields(model)}
                    if unknown:
                        raise CloudApiError(
                            "manifest_invalid",
                            f"{table_name} contains unsupported fields: {', '.join(sorted(unknown))}.",
                        )
            manifest = cls(
                application=application,
                name=str(payload["name"]),
                version=str(payload["version"]),
                entrypoint=str(payload["entrypoint"]),
                capabilities=tuple(payload["capabilities"]),
                cpu_millis=payload["cpu_millis"],
                memory_mb=payload["memory_mb"],
                timeout_seconds=payload["timeout_seconds"],
                environment=str(payload["environment"]),
                compatibility=str(payload["compatibility"]),
                policy_version=str(payload["policy_version"]),
                web=(
                    None
                    if web is None
                    else CloudWebProcess(
                        command=str(web["command"]),
                        **_process_resources(web),
                        port=web["port"],
                        ready_path=_ready_path(web.get("ready_path", "/")),
                        ready_timeout_seconds=_ready_seconds(
                            web.get("ready_timeout_seconds", _READY_TIMEOUT_SECONDS),
                            name="ready_timeout_seconds",
                            bounds=_READY_TIMEOUT_RANGE,
                        ),
                        ready_start_period_seconds=_ready_seconds(
                            web.get("ready_start_period_seconds", _READY_START_PERIOD_SECONDS),
                            name="ready_start_period_seconds",
                            bounds=_READY_START_PERIOD_RANGE,
                        ),
                        idle_timeout_seconds=(
                            None
                            if web.get("idle_timeout_seconds") is None
                            else web["idle_timeout_seconds"]
                        ),
                    )
                ),
                worker=(
                    None
                    if worker is None
                    else CloudProcess(command=str(worker["command"]), **_process_resources(worker))
                ),
                schedules=tuple(
                    CloudSchedule(
                        name=str(item["name"]),
                        command=str(item["command"]),
                        expression=str(item["expression"]),
                        **_process_resources(item),
                    )
                    for item in schedules
                ),
                runtime_environment=dict(runtime_environment),
                local_files=_local_files(payload.get("storage")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CloudApiError(
                "manifest_invalid",
                "cayu-cloud.toml is missing a required field or contains the wrong type.",
            ) from exc
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not is_application_slug(self.application):
            raise CloudApiError(
                "manifest_invalid",
                "Manifest application slug must be 8-63 lowercase letters, numbers, "
                "or interior hyphens.",
            )
        if not self.name or len(self.name) > 128:
            raise CloudApiError("manifest_invalid", "Manifest application name is invalid.")
        if _VERSION.fullmatch(self.version) is None:
            raise CloudApiError("manifest_invalid", "Manifest version is invalid.")
        if not self.entrypoint or len(self.entrypoint) > 512 or "\x00" in self.entrypoint:
            raise CloudApiError("manifest_invalid", "Manifest entrypoint is invalid.")
        if (
            not self.capabilities
            or len(self.capabilities) > 128
            or len(set(self.capabilities)) != len(self.capabilities)
            or not all(
                isinstance(capability, str) and _CAPABILITY.fullmatch(capability)
                for capability in self.capabilities
            )
        ):
            raise CloudApiError("manifest_invalid", "Manifest capabilities are invalid.")
        if type(self.cpu_millis) is not int or not 100 <= self.cpu_millis <= 16_000:
            raise CloudApiError("manifest_invalid", "Manifest cpu_millis is invalid.")
        if type(self.memory_mb) is not int or not 128 <= self.memory_mb <= 131_072:
            raise CloudApiError("manifest_invalid", "Manifest memory_mb is invalid.")
        if type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= 86_400:
            raise CloudApiError("manifest_invalid", "Manifest timeout_seconds is invalid.")
        if not self.environment or len(self.environment) > 128:
            raise CloudApiError("manifest_invalid", "Manifest environment is invalid.")
        if not self.compatibility or len(self.compatibility) > 256:
            raise CloudApiError("manifest_invalid", "Manifest compatibility is invalid.")
        if _POLICY_VERSION.fullmatch(self.policy_version) is None:
            raise CloudApiError("manifest_invalid", "Manifest policy_version is invalid.")
        for process in (self.web, self.worker):
            if process is not None:
                _process_resources(
                    {"cpu_millis": process.cpu_millis, "memory_mb": process.memory_mb}
                )
            if process is not None and (
                not process.command or len(process.command) > 2048 or "\x00" in process.command
            ):
                raise CloudApiError("manifest_invalid", "Runtime process command is invalid.")
        if self.web is not None and (
            type(self.web.port) is not int or not 1 <= self.web.port <= 65_535
        ):
            raise CloudApiError("manifest_invalid", "Runtime web port is invalid.")
        if self.web is not None:
            _ready_path(self.web.ready_path)
            _ready_seconds(
                self.web.ready_timeout_seconds,
                name="ready_timeout_seconds",
                bounds=_READY_TIMEOUT_RANGE,
            )
            _ready_seconds(
                self.web.ready_start_period_seconds,
                name="ready_start_period_seconds",
                bounds=_READY_START_PERIOD_RANGE,
            )
        if (
            self.web is not None
            and self.web.idle_timeout_seconds is not None
            and (
                type(self.web.idle_timeout_seconds) is not int
                or not 60 <= self.web.idle_timeout_seconds <= 86_400
            )
        ):
            raise CloudApiError(
                "manifest_invalid",
                "Runtime web idle_timeout_seconds is invalid.",
            )
        if len(self.schedules) > 20:
            raise CloudApiError("manifest_invalid", "Too many runtime schedules.")
        for schedule in self.schedules:
            _process_resources({"cpu_millis": schedule.cpu_millis, "memory_mb": schedule.memory_mb})
            if (
                _PROCESS_NAME.fullmatch(schedule.name) is None
                or not schedule.command
                or len(schedule.command) > 2048
                or not schedule.expression
                or len(schedule.expression) > 256
            ):
                raise CloudApiError("manifest_invalid", "Runtime schedule is invalid.")
        if len(set(item.name for item in self.schedules)) != len(self.schedules):
            raise CloudApiError("manifest_invalid", "Runtime schedule names must be unique.")
        if len(self.local_files) > _LOCAL_FILES_LIMIT:
            raise CloudApiError("manifest_invalid", _STORAGE_TABLE_ERROR)
        for entry in self.local_files:
            _local_file(entry.path, entry.reason)

    def deployment_payload(self, *, repository: str, revision: str) -> dict[str, object]:
        return {
            "version": f"{self.version}+{revision[:12]}",
            "manifest": {
                "schema_version": 1,
                "source": {"repository": repository, "revision": revision},
                "entrypoint": self.entrypoint,
                "capabilities": list(self.capabilities),
                "resources": {
                    "cpu_millis": self.cpu_millis,
                    "memory_mb": self.memory_mb,
                },
                "runtime": self.runtime_payload(),
                "timeout_seconds": self.timeout_seconds,
                "environment": self.environment,
                "compatibility": self.compatibility,
            },
            "policy_version": self.policy_version,
        }

    def runtime_payload(self) -> dict[str, object] | None:
        if self.web is None and self.worker is None and not self.schedules:
            return None
        return {
            "environment": dict(sorted((self.runtime_environment or {}).items())),
            "schedules": [
                {
                    "command": item.command,
                    "expression": item.expression,
                    "name": item.name,
                    **_process_resources(
                        {"cpu_millis": item.cpu_millis, "memory_mb": item.memory_mb}
                    ),
                }
                for item in self.schedules
            ],
            "web": (
                None
                if self.web is None
                else {
                    "command": self.web.command,
                    "port": self.web.port,
                    **_process_resources(
                        {"cpu_millis": self.web.cpu_millis, "memory_mb": self.web.memory_mb}
                    ),
                    # Defaults are omitted so older Cloud builds accept the payload.
                    **({} if self.web.ready_path == "/" else {"ready_path": self.web.ready_path}),
                    **(
                        {}
                        if self.web.ready_timeout_seconds == _READY_TIMEOUT_SECONDS
                        else {"ready_timeout_seconds": self.web.ready_timeout_seconds}
                    ),
                    **(
                        {}
                        if self.web.ready_start_period_seconds == _READY_START_PERIOD_SECONDS
                        else {"ready_start_period_seconds": self.web.ready_start_period_seconds}
                    ),
                    **(
                        {}
                        if self.web.idle_timeout_seconds is None
                        else {"idle_timeout_seconds": self.web.idle_timeout_seconds}
                    ),
                }
            ),
            "worker": (
                None
                if self.worker is None
                else {
                    "command": self.worker.command,
                    **_process_resources(
                        {"cpu_millis": self.worker.cpu_millis, "memory_mb": self.worker.memory_mb}
                    ),
                }
            ),
        }


@dataclass(frozen=True)
class ResolvedCloudProject:
    root: Path | None
    manifest_path: Path
    manifest: CloudProjectManifest
    repository: str
    revision: str
    bundle: bytes | None = None
    content_digest: str | None = None


@dataclass(frozen=True)
class InitializedCloudProject:
    application: str
    manifest_path: Path
    name: str
    runtime: str
    serve: dict[str, object] | None = None
    command: str | None = None


def initialize_project(path: Path, *, force: bool = False) -> InitializedCloudProject:
    """Create a small deployment descriptor from conventional Python project metadata."""

    root = path.expanduser().resolve()
    pyproject_path = root / "pyproject.toml"
    manifest_path = root / "cayu-cloud.toml"
    if not root.is_dir() or not pyproject_path.is_file():
        raise CloudApiError(
            "project_unavailable",
            f"Cayu Cloud init requires a Python project with pyproject.toml: {root}",
        )
    if manifest_path.exists() and not force:
        raise CloudApiError(
            "manifest_exists",
            "cayu-cloud.toml already exists; pass --force to replace it.",
        )
    try:
        payload = tomllib.loads(pyproject_path.read_text())
        project = payload["project"]
        raw_name = project["name"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise CloudApiError(
            "project_invalid",
            "pyproject.toml must contain a valid [project] name.",
        ) from exc
    if not isinstance(project, dict) or not isinstance(raw_name, str):
        raise CloudApiError(
            "project_invalid",
            "pyproject.toml must contain a valid [project] name.",
        )
    application = _application_slug(raw_name)
    name = " ".join(part.capitalize() for part in re.split(r"[-_.]+", raw_name) if part)
    raw_version = project.get("version", "0.1.0")
    version = raw_version if isinstance(raw_version, str) else "0.1.0"
    tool = payload.get("tool", {})
    cayu = tool.get("cayu", {}) if isinstance(tool, dict) else {}
    cayu = cayu if isinstance(cayu, dict) else {}
    workers = cayu.get("workers", {})
    scripts = project.get("scripts", {})
    serve_setup: ServeSetupPlan | None = None
    writes_auth_module = False
    if "serve" in cayu or isinstance(cayu.get("factory"), str):
        runtime = "web"
        command = "cayu serve --host 0.0.0.0 --port 8000"
        serve_setup = _plan_init_serve_setup(pyproject_path)
        writes_auth_module = _auth_module_needs_writing(root, serve_setup)
    elif isinstance(workers, dict) and len(workers) == 1:
        runtime = "worker"
        command = f"cayu worker {next(iter(workers))}"
    elif isinstance(scripts, dict) and len(scripts) == 1:
        runtime = "worker"
        command = str(next(iter(scripts)))
    else:
        runtime = "worker"
        command = "python -m " + application.replace("-", "_")
    content = _render_initial_manifest(
        application=application,
        name=name or application,
        version=version,
        command=command,
        runtime=runtime,
    )
    CloudProjectManifest.loads(content)
    if writes_auth_module:
        assert serve_setup is not None and serve_setup.auth_module is not None
        module_path = root / serve_setup.auth_module.path
        try:
            # Exclusive creation: never replace a file that appeared since planning.
            with module_path.open("x", encoding="utf-8") as module_file:
                module_file.write(serve_setup.auth_module.content)
        except OSError as exc:
            raise CloudApiError(
                "project_unavailable",
                f"Could not create {module_path} for `cayu serve`.",
            ) from exc
    if serve_setup is not None and serve_setup.updated_text is not None:
        try:
            pyproject_path.write_text(serve_setup.updated_text)
        except OSError as exc:
            raise CloudApiError(
                "project_unavailable",
                f"Could not update {pyproject_path} for `cayu serve`.",
            ) from exc
    try:
        manifest_path.write_text(content)
    except OSError as exc:
        raise CloudApiError(
            "manifest_unavailable",
            f"Could not write Cayu Cloud manifest: {manifest_path}",
        ) from exc
    return InitializedCloudProject(
        application=application,
        manifest_path=manifest_path,
        name=name or application,
        runtime=runtime,
        serve=None
        if serve_setup is None
        else _serve_setup_report(serve_setup, created_auth_module=writes_auth_module),
        command=command,
    )


def _plan_init_serve_setup(pyproject_path: Path) -> ServeSetupPlan:
    """Make sure the `cayu serve` web process init writes can start on Cloud."""

    setup = plan_serve_setup(pyproject_path.read_text(), root=pyproject_path.parent)
    if setup.manual_edits:
        edits = "\n".join(f"- {edit}" for edit in setup.manual_edits)
        raise CloudApiError(
            "serve_setup_required",
            "cayu cloud init writes a `cayu serve` web process, which needs the "
            "`server` extra of cayu and an authentication target to start outside "
            f"--dev. It could not safely edit {pyproject_path}; make these edits, "
            f"run `uv lock`, and rerun `cayu cloud init`:\n{edits}",
        )
    return setup


def _auth_module_needs_writing(root: Path, setup: ServeSetupPlan) -> bool:
    """Whether init must create the auth module; refuse if another file holds its name."""

    module = setup.auth_module
    if module is None:
        return False
    path = root / module.path
    package = root / module.path.removesuffix(".py")
    if not path.exists() and not path.is_symlink() and not package.exists():
        return True
    try:
        identical = (
            not path.is_symlink()
            and path.is_file()
            and not package.exists()
            and path.read_text(encoding="utf-8") == module.content
        )
    except (OSError, UnicodeError):
        identical = False
    if identical:
        return False
    raise CloudApiError(
        "serve_setup_required",
        "cayu cloud init writes a `cayu serve` web process, which needs an "
        "authentication target to start outside --dev. The declared cayu requirement "
        f"allows cayu {LAST_RELEASE_WITHOUT_ENVIRONMENT_AUTH} or older, which lack "
        f"{ENVIRONMENT_OPERATOR_AUTH_TARGET}, so init would create {module.path} for "
        f"auth = {json.dumps(module.target)}, but {path if path.exists() else package} "
        "already exists and init never replaces it. Either set [tool.cayu.serve].auth "
        "to your own auth dependency, or make these edits, run `uv lock`, and set auth = "
        f"{json.dumps(ENVIRONMENT_OPERATOR_AUTH_TARGET)}:\n"
        + "\n".join(f"- {edit}" for edit in setup.environment_auth_upgrade),
    )


def _serve_setup_report(setup: ServeSetupPlan, *, created_auth_module: bool) -> dict[str, object]:
    extra_added = setup.server_extra.status == "missing_extra"
    auth_added = setup.auth.status == "missing"
    module = setup.auth_module
    report: dict[str, object] = {
        "server_extra": {
            "status": "added" if extra_added else "present",
            "requirement": setup.server_extra.replacement
            if extra_added
            else setup.server_extra.requirement,
        },
        "auth": {
            "status": "added"
            if auth_added
            else "service_factory"
            if setup.auth.status == "service_factory"
            else "kept",
            "target": setup.auth_target if auth_added else setup.auth.target,
        },
        "pyproject_changes": list(setup.changes),
        "next_steps": ["uv lock"] if extra_added else [],
    }
    if module is not None:
        report["auth_module"] = {
            "path": module.path,
            "status": "created" if created_auth_module else "present",
        }
    notes: list[str] = []
    if extra_added:
        notes.append(
            "Added the server extra to the cayu dependency. Run `uv lock` and commit "
            "pyproject.toml and uv.lock; Cayu Cloud installs from uv.lock."
        )
    if auth_added and module is not None:
        notes.append(
            f"Set [tool.cayu.serve].auth to {module.target}, defined in {module.path}, "
            "which builds BasicAuth from "
            f"{OPERATOR_USERNAME_VARIABLE} and {OPERATOR_PASSWORD_VARIABLE}. The declared "
            f"cayu requirement allows cayu {LAST_RELEASE_WITHOUT_ENVIRONMENT_AUTH} or "
            f"older, which lack {ENVIRONMENT_OPERATOR_AUTH_TARGET}; this module works on "
            f"every release. Commit {module.path}. Cayu Cloud provides both variables to "
            "each Agent; read them with `cayu cloud service credentials --application APP`. "
            "Run `cayu serve --dev` locally without them."
        )
    elif auth_added:
        notes.append(
            f"Set [tool.cayu.serve].auth to {ENVIRONMENT_OPERATOR_AUTH_TARGET}. Cayu "
            f"Cloud provides {OPERATOR_USERNAME_VARIABLE} and {OPERATOR_PASSWORD_VARIABLE} "
            "to each Agent; read them with `cayu cloud service credentials "
            "--application APP`. Run `cayu serve --dev` locally without them."
        )
    elif setup.auth.target == ENVIRONMENT_OPERATOR_AUTH_TARGET and not (
        setup.ships_environment_auth
    ):
        notes.append(
            f"[tool.cayu.serve].auth names {ENVIRONMENT_OPERATOR_AUTH_TARGET}, but the "
            f"declared cayu requirement allows cayu {LAST_RELEASE_WITHOUT_ENVIRONMENT_AUTH} "
            "or older, which lack it, so `cayu serve` would not start. "
            + " ".join(setup.environment_auth_upgrade)
            + " Then run `uv lock`."
        )
    elif setup.auth.target == ENVIRONMENT_OPERATOR_AUTH_TARGET:
        notes.append(
            f"[tool.cayu.serve].auth already reads {OPERATOR_USERNAME_VARIABLE} and "
            f"{OPERATOR_PASSWORD_VARIABLE}, which Cayu Cloud provides to each Agent; read "
            "them with `cayu cloud service credentials --application APP`."
        )
    elif setup.auth.status == "configured":
        notes.append(
            f"Left the existing [tool.cayu.serve].auth target {setup.auth.target} "
            f"unchanged. Cayu Cloud's {OPERATOR_USERNAME_VARIABLE} and "
            f"{OPERATOR_PASSWORD_VARIABLE} are unused unless that target reads them."
        )
    else:
        notes.append(
            "The project's service_factory owns product and operator access; "
            "[tool.cayu.serve].auth does not apply."
        )
    report["notes"] = notes
    return report


def _application_slug(value: str) -> str:
    slug = _SLUG_SEPARATOR.sub("-", value.strip().lower()).strip("-")[:63].rstrip("-")
    if not is_application_slug(slug):
        raise CloudApiError(
            "project_invalid",
            "The project name must produce an 8-63 character lowercase Cayu Cloud "
            "application slug containing only letters, numbers, and interior hyphens.",
        )
    return slug


def _render_initial_manifest(
    *,
    application: str,
    name: str,
    version: str,
    command: str,
    runtime: str,
) -> str:
    values = [
        "schema_version = 2",
        f"application = {json.dumps(application)}",
        f"name = {json.dumps(name)}",
        f"version = {json.dumps(version)}",
        f"entrypoint = {json.dumps(command)}",
        'capabilities = ["model.generate"]',
        "cpu_millis = 1000",
        "memory_mb = 2048",
        "timeout_seconds = 900",
        'environment = "python"',
        'compatibility = "cayu>=0.1,<1"',
        'policy_version = "cayu-egress-v1"',
        "",
    ]
    if runtime == "web":
        values.extend(("[web]", f"command = {json.dumps(command)}", "port = 8000", ""))
    else:
        values.extend(("[worker]", f"command = {json.dumps(command)}", ""))
    return "\n".join(values)


def resolve_project(
    source: str,
    *,
    manifest_path: Path | None,
    revision: str | None,
) -> ResolvedCloudProject:
    candidate = Path(source).expanduser()
    if candidate.exists():
        project_root = candidate.resolve()
        if not project_root.is_dir():
            raise CloudApiError(
                "source_invalid",
                "Local deployment source must be a directory.",
            )
        if revision is not None:
            raise CloudApiError(
                "source_invalid",
                "--revision is only valid for a remote repository source.",
            )
        selected_manifest = manifest_path or Path("cayu-cloud.toml")
        if not selected_manifest.is_absolute():
            selected_manifest = project_root / selected_manifest
        selected_manifest = selected_manifest.resolve()
        try:
            selected_manifest.relative_to(project_root)
        except ValueError as exc:
            raise CloudApiError(
                "manifest_invalid",
                "Local manifest must be inside the project directory.",
            ) from exc
        manifest = CloudProjectManifest.load(selected_manifest)
        _validate_storage_manifest_path(manifest, selected_manifest.relative_to(project_root))
        bundle = _archive_project(project_root, required_file=selected_manifest)
        content_digest = "sha256:" + hashlib.sha256(bundle).hexdigest()
        digest = content_digest.removeprefix("sha256:")
        return ResolvedCloudProject(
            root=project_root,
            manifest_path=selected_manifest,
            manifest=manifest,
            repository=_SOURCE_BUNDLE_PREFIX + digest,
            revision=digest[:40],
            bundle=bundle,
            content_digest=content_digest,
        )
    repository = _canonical_github_repository(source)
    if revision is None or _COMMIT.fullmatch(revision) is None:
        raise CloudApiError(
            "source_revision_required",
            "Repository deployment requires an exact 40-character --revision.",
        )
    selected_manifest = manifest_path or Path("cayu-cloud.toml")
    if (
        selected_manifest.is_absolute()
        or not selected_manifest.parts
        or ".." in selected_manifest.parts
    ):
        raise CloudApiError(
            "manifest_invalid",
            "Repository manifest must be a relative path inside the exact source commit.",
        )
    manifest = CloudProjectManifest.loads(
        _github_file(repository, revision=revision, path=selected_manifest.as_posix())
    )
    _validate_storage_manifest_path(manifest, selected_manifest)
    return ResolvedCloudProject(
        root=None,
        manifest_path=selected_manifest,
        manifest=manifest,
        repository=repository,
        revision=revision,
    )


def _validate_storage_manifest_path(manifest: CloudProjectManifest, path: Path) -> None:
    # Source admission reads only the root file, not the selected CLI manifest path.
    if manifest.local_files and path != Path("cayu-cloud.toml"):
        raise CloudApiError(
            "manifest_invalid",
            "Storage local_files requires cayu-cloud.toml at the deployment root. "
            "Move the manifest there as a regular file and deploy without --manifest; "
            "Cloud reads storage acknowledgements only from that file.",
        )


class CloudSourceInputsError(CloudApiError):
    """A required hosted Python input is absent from the selected upload."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        self.hint = (
            "Run uv lock in the deployment root and include pyproject.toml and uv.lock "
            "in the uploaded source. Check the selected directory and Git ignore rules."
        )
        super().__init__(
            "source_build_inputs_invalid", f"Required build input {path} is {reason}. {self.hint}"
        )


def _validate_python_build_inputs(root: Path, bundle: bytes) -> None:
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        files = {member.name.removeprefix("source/"): member for member in archive}
    for name in ("pyproject.toml", "uv.lock"):
        if name not in files:
            reason = "excluded from the upload" if os.path.lexists(root / name) else "missing"
            raise CloudSourceInputsError(name, reason)
        target = name
        seen: set[str] = set()
        while target not in seen:
            seen.add(target)
            member = files.get(target)
            if member is None:
                break
            if member.isfile() and member.size > 0:
                break
            if not member.issym() or member.linkname.startswith("/"):
                break
            target = posixpath.normpath(posixpath.join(posixpath.dirname(target), member.linkname))
            if target == ".." or target.startswith("../"):
                break
        else:
            member = None
        if member is None or not member.isfile() or member.size <= 0:
            raise CloudSourceInputsError(name, "not a usable file in the upload")


def _archive_project(root: Path, *, required_file: Path) -> bytes:
    files = set(_listed_project_files(root))
    files.add(required_file.relative_to(root))
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for relative in sorted(files, key=lambda item: item.as_posix()):
            path = root / relative
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise CloudApiError(
                    "source_unavailable",
                    f"Could not read local deployment source: {relative.as_posix()}",
                ) from exc
            info = tarfile.TarInfo(f"source/{relative.as_posix()}")
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            if stat.S_ISREG(metadata.st_mode):
                try:
                    content = path.read_bytes()
                except OSError as exc:
                    raise CloudApiError(
                        "source_unavailable",
                        f"Could not read local deployment source: {relative.as_posix()}",
                    ) from exc
                info.type = tarfile.REGTYPE
                info.mode = 0o755 if metadata.st_mode & 0o111 else 0o644
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            elif stat.S_ISLNK(metadata.st_mode):
                info.type = tarfile.SYMTYPE
                info.mode = 0o777
                info.linkname = os.readlink(path)
                archive.addfile(info)
            else:
                raise CloudApiError(
                    "source_invalid",
                    f"Local deployment source contains an unsupported file: {relative.as_posix()}",
                )
    bundle = buffer.getvalue()
    _validate_python_build_inputs(root, bundle)
    return bundle


def _listed_project_files(root: Path) -> tuple[Path, ...]:
    git_files = _git_project_files(root)
    if git_files is not None:
        return tuple(
            path
            for path in git_files
            if not _source_path_excluded(path)
            and ((root / path).is_file() or (root / path).is_symlink())
        )
    return tuple(
        path.relative_to(root)
        for path in root.rglob("*")
        if (path.is_file() or path.is_symlink())
        and not _source_path_excluded(path.relative_to(root))
    )


def _git_project_files(root: Path) -> tuple[Path, ...] | None:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
            env=_source_command_environment(),
            stdin=subprocess.DEVNULL,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise CloudApiError(
            "source_git_failed",
            "Git could not enumerate the local deployment source.",
        ) from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        if _git_metadata_present(root):
            raise CloudApiError(
                "source_git_failed",
                "Git could not enumerate the local deployment source.",
            ) from exc
        return None
    return tuple(Path(os.fsdecode(item)) for item in completed.stdout.split(b"\0") if item)


def _git_metadata_present(root: Path) -> bool:
    return any(os.path.lexists(directory / ".git") for directory in (root, *root.parents))


def _source_path_excluded(path: Path) -> bool:
    if any(part in _EXCLUDED_DIRECTORIES for part in path.parts):
        return True
    name = path.name
    if name in {".DS_Store", ".env"} or name.endswith((".pyc", ".pyo")):
        return True
    return name.startswith(".env.") and name not in _SAFE_ENV_TEMPLATES


def _github_file(repository: str, *, revision: str, path: str) -> str:
    owner, name = urlsplit(repository).path.strip("/").split("/", maxsplit=1)
    environment = _source_command_environment()
    authentication_available = _github_authentication_available(environment)
    if not authentication_available:
        return _anonymous_github_file(
            owner=owner,
            repository=name,
            revision=revision,
            path=path,
        )
    try:
        completed = subprocess.run(
            [
                "gh",
                "api",
                "-H",
                "Accept: application/vnd.github.raw+json",
                f"repos/{owner}/{name}/contents/{path}?ref={revision}",
            ],
            check=True,
            capture_output=True,
            env=environment,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise CloudApiError(
            "manifest_unavailable",
            "Could not read the Cayu Cloud manifest from the exact GitHub revision.",
        ) from exc
    if not completed.stdout:
        raise CloudApiError(
            "manifest_unavailable",
            "The exact GitHub revision returned an empty Cayu Cloud manifest.",
        )
    return completed.stdout


def _anonymous_github_file(
    *,
    owner: str,
    repository: str,
    revision: str,
    path: str,
) -> str:
    repository_url = f"https://api.github.com/repos/{owner}/{repository}"
    content_url = f"{repository_url}/contents/{quote(path, safe='/')}"
    repository_response: httpx.Response | None = None
    try:
        with httpx.Client(follow_redirects=False, timeout=30.0) as client:
            response = client.get(
                content_url,
                headers={"Accept": "application/vnd.github.raw+json"},
                params={"ref": revision},
            )
            if response.status_code == 404:
                repository_response = client.get(
                    repository_url,
                    headers={"Accept": "application/vnd.github+json"},
                    params={},
                )
    except httpx.RequestError as exc:
        raise CloudApiError(
            "source_unavailable",
            "GitHub was unavailable while resolving the deployment source.",
        ) from exc

    if 200 <= response.status_code < 300:
        try:
            content = response.content.decode("utf-8")
        except UnicodeDecodeError:
            content = ""
        if content:
            return content
        raise CloudApiError(
            "manifest_unavailable",
            "The exact GitHub revision returned an empty Cayu Cloud manifest.",
        )
    if response.status_code == 404 and repository_response is not None:
        if 200 <= repository_response.status_code < 300:
            raise CloudApiError(
                "manifest_unavailable",
                "Could not read the Cayu Cloud manifest from the exact GitHub revision.",
            )
        if repository_response.status_code == 404:
            raise CloudApiError(
                "source_auth_unavailable",
                "GitHub authentication is unavailable. Run `gh auth login` or set "
                "`GH_TOKEN` for noninteractive use.",
            )
    raise CloudApiError(
        "source_unavailable",
        "GitHub was unavailable while resolving the deployment source.",
    )


def _github_authentication_available(environment: dict[str, str]) -> bool:
    try:
        subprocess.run(
            ["gh", "auth", "status", "--hostname", "github.com"],
            check=True,
            capture_output=True,
            env=environment,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _source_command_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("CAYU_CLOUD_")
    }
    environment.update(
        {
            "GCM_INTERACTIVE": "Never",
            "GH_PROMPT_DISABLED": "1",
            "GIT_SSH_COMMAND": "ssh -oBatchMode=yes",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _canonical_github_repository(value: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise CloudApiError(
            "source_repository_invalid",
            "Deployment source must be a canonical HTTPS GitHub repository.",
        )
    stripped = value.strip()
    if stripped.startswith("git@github.com:"):
        stripped = "https://github.com/" + stripped.removeprefix("git@github.com:")
    parsed = urlsplit(stripped)
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise CloudApiError(
            "source_repository_invalid",
            "Deployment source must be a canonical HTTPS GitHub repository.",
        )
    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = path.removeprefix("/").split("/")
    if (
        len(parts) != 2
        or path != f"/{parts[0]}/{parts[1]}"
        or "%" in path
        or _GITHUB_OWNER.fullmatch(parts[0]) is None
        or _GITHUB_REPOSITORY.fullmatch(parts[1]) is None
        or parts[1] in {".", ".."}
    ):
        raise CloudApiError(
            "source_repository_invalid",
            "Deployment source must be a canonical HTTPS GitHub repository.",
        )
    return f"https://github.com/{parts[0]}/{parts[1]}"
