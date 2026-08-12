"""Immutable project resolution for `cayu cloud deploy`."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx

from cayu.cli._cloud_api import CloudApiError

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_APPLICATION = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\Z")
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


@dataclass(frozen=True)
class CloudWebProcess(CloudProcess):
    port: int
    idle_timeout_seconds: int | None = None


@dataclass(frozen=True)
class CloudSchedule:
    name: str
    command: str
    expression: str


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
            manifest = cls(
                application=str(payload["application"]),
                name=str(payload["name"]),
                version=str(payload["version"]),
                entrypoint=str(payload["entrypoint"]),
                capabilities=tuple(payload["capabilities"]),
                cpu_millis=int(payload["cpu_millis"]),
                memory_mb=int(payload["memory_mb"]),
                timeout_seconds=int(payload["timeout_seconds"]),
                environment=str(payload["environment"]),
                compatibility=str(payload["compatibility"]),
                policy_version=str(payload["policy_version"]),
                web=(
                    None
                    if web is None
                    else CloudWebProcess(
                        command=str(web["command"]),
                        port=int(web["port"]),
                        idle_timeout_seconds=(
                            None
                            if web.get("idle_timeout_seconds") is None
                            else int(web["idle_timeout_seconds"])
                        ),
                    )
                ),
                worker=(None if worker is None else CloudProcess(command=str(worker["command"]))),
                schedules=tuple(
                    CloudSchedule(
                        name=str(item["name"]),
                        command=str(item["command"]),
                        expression=str(item["expression"]),
                    )
                    for item in schedules
                ),
                runtime_environment=dict(runtime_environment),
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
            raise CloudApiError("manifest_invalid", "Manifest application slug is invalid.")
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
        if not 100 <= self.cpu_millis <= 16_000:
            raise CloudApiError("manifest_invalid", "Manifest cpu_millis is invalid.")
        if not 128 <= self.memory_mb <= 131_072:
            raise CloudApiError("manifest_invalid", "Manifest memory_mb is invalid.")
        if not 1 <= self.timeout_seconds <= 86_400:
            raise CloudApiError("manifest_invalid", "Manifest timeout_seconds is invalid.")
        if not self.environment or len(self.environment) > 128:
            raise CloudApiError("manifest_invalid", "Manifest environment is invalid.")
        if not self.compatibility or len(self.compatibility) > 256:
            raise CloudApiError("manifest_invalid", "Manifest compatibility is invalid.")
        if _POLICY_VERSION.fullmatch(self.policy_version) is None:
            raise CloudApiError("manifest_invalid", "Manifest policy_version is invalid.")
        for process in (self.web, self.worker):
            if process is not None and (
                not process.command or len(process.command) > 2048 or "\x00" in process.command
            ):
                raise CloudApiError("manifest_invalid", "Runtime process command is invalid.")
        if self.web is not None and not 1 <= self.web.port <= 65_535:
            raise CloudApiError("manifest_invalid", "Runtime web port is invalid.")
        if (
            self.web is not None
            and self.web.idle_timeout_seconds is not None
            and not 60 <= self.web.idle_timeout_seconds <= 86_400
        ):
            raise CloudApiError(
                "manifest_invalid",
                "Runtime web idle_timeout_seconds is invalid.",
            )
        if len(self.schedules) > 20:
            raise CloudApiError("manifest_invalid", "Too many runtime schedules.")
        for schedule in self.schedules:
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
                }
                for item in self.schedules
            ],
            "web": (
                None
                if self.web is None
                else {
                    "command": self.web.command,
                    "port": self.web.port,
                    **(
                        {}
                        if self.web.idle_timeout_seconds is None
                        else {"idle_timeout_seconds": self.web.idle_timeout_seconds}
                    ),
                }
            ),
            "worker": (None if self.worker is None else {"command": self.worker.command}),
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
    if "serve" in cayu or isinstance(cayu.get("factory"), str):
        runtime = "web"
        command = "cayu serve --host 0.0.0.0 --port 8000"
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
    )


def _application_slug(value: str) -> str:
    slug = _SLUG_SEPARATOR.sub("-", value.strip().lower()).strip("-")[:63].rstrip("-")
    if not is_application_slug(slug):
        raise CloudApiError(
            "project_invalid",
            "The project name cannot be converted to a Cayu Cloud application slug.",
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
    return ResolvedCloudProject(
        root=None,
        manifest_path=selected_manifest,
        manifest=manifest,
        repository=repository,
        revision=revision,
    )


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
    return buffer.getvalue()


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
