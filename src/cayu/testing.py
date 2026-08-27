"""Explicit, bounded deployment-readiness checks for Cayu tools."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, field_validator, model_validator

from cayu._validation import (
    copy_durable_json_object,
    copy_json_value,
    require_clean_nonblank,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    require_durable_text,
    require_nonblank,
)
from cayu.core.isolated_tools import ProcessIsolatedTool
from cayu.core.tools import ToolContext, ToolEffect, ToolResult
from cayu.runners.base import ExecCommand, Runner
from cayu.runners.local import LocalRunner
from cayu.runtime.app import CayuApp
from cayu.workspaces import LocalWorkspace

_DEFAULT_MAX_FILES = 1_000
_DEFAULT_MAX_ENTRIES = 2_000
_DEFAULT_MAX_FILE_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 30.0
_HASH_CHUNK_BYTES = 64 * 1024
_PROVIDER_CREDENTIAL_PROBE_OUTPUT_LIMIT_BYTES = 1024 * 1024
_PROVIDER_CREDENTIAL_DETECTOR_CONTROL_ENV = "CAYU_PROVIDER_CREDENTIAL_PROBE_DETECTOR_CONTROL"
_PROVIDER_CREDENTIAL_DETECTOR_CONTROL_VALUE = "cayu-provider-credential-detector-control-0123456789"
_PROVIDER_CREDENTIAL_PROJECTIONS = (
    "artifacts",
    "auth_paths",
    "environment",
    "stderr",
    "stdout",
)
_PROVIDER_CREDENTIAL_AUTH_PATHS = (
    "/root/.cayu/auth.json",
    "/home/cayu/.cayu/auth.json",
    "/home/user/.cayu/auth.json",
    "/workspace/.cayu/auth.json",
)
_PROVIDER_CREDENTIAL_AUTH_SEARCH_LABELS = (
    "current_working_directory",
    "guest_home",
    "workspace_root",
)
_PROVIDER_CREDENTIAL_AUTH_SCAN_MAX_DIRECTORIES = 10_000
_PROVIDER_CREDENTIAL_ENV_NAMES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_CONFIG_FILE",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "CAYU_HOME",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "OPENAI_API_KEY",
        "OPENAI_AUTHORIZATION",
    }
)
_ENVIRONMENT_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_EVIDENCE_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]*\Z")

_BOUNDARY_NAME = "isolated_workspace"
_BASE_UNOBSERVED_SYSTEMS = (
    "artifact_stores",
    "databases_outside_workspace",
    "host_filesystem_outside_workspace",
    "network_and_external_services",
    "process_and_tool_instance_state",
    "runner_execution",
)
_LIMITATIONS = (
    "Evidence covers only files visible through the isolated workspace supplied to the tool.",
    "Empty directories, symlinks, non-regular entries, permissions, timestamps, and other filesystem metadata are not observed as mutations.",
    "The tool runs in the current Python process; this is an observation boundary, not a security sandbox.",
    "Execution deadlines use cooperative asyncio cancellation; a hard stop requires a killable process boundary.",
    "Tool policy, approvals, hooks, events, and the model loop are not evaluated.",
    "Work scheduled by the tool after run() returns is outside the before/after snapshot.",
    "No observed mutation is scoped evidence, not universal proof of purity.",
)


class ProviderCredentialIsolationViolation(RuntimeError):
    """A provider credential canary was observed across the guest boundary.

    The exception intentionally retains only the caller-defined canary label and
    the projection where it was observed. The canary value is never stored in
    the exception or included in its message.
    """

    def __init__(self, *, adapter: str, canary_label: str, projection: str) -> None:
        self.adapter = require_clean_nonblank(adapter, "adapter")
        self.canary_label = require_clean_nonblank(canary_label, "canary_label")
        self.projection = require_clean_nonblank(projection, "projection")
        super().__init__(
            "Provider credential isolation verification failed: "
            f"adapter={self.adapter}, canary={self.canary_label}, "
            f"projection={self.projection}."
        )


class ProviderCredentialIsolationVerification(BaseModel):
    """Content-free evidence from one provider-credential boundary probe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["cayu.provider_credential_isolation.v1"] = (
        "cayu.provider_credential_isolation.v1"
    )
    status: Literal["verified", "environment_minimized"]
    adapter: str
    scope: Literal["isolated_guest", "local_environment"]
    canary_labels: tuple[str, ...]
    auth_search_labels: tuple[str, ...] = ()
    projections: tuple[
        Literal["artifacts", "auth_paths", "environment", "stderr", "stdout"], ...
    ] = _PROVIDER_CREDENTIAL_PROJECTIONS
    positive_controls: tuple[str, ...]

    @field_validator("adapter")
    @classmethod
    def validate_adapter(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "adapter")

    @field_validator("canary_labels", "positive_controls")
    @classmethod
    def normalize_names(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        names = tuple(require_durable_clean_nonblank(item, info.field_name) for item in value)
        if not names:
            raise ValueError(f"{info.field_name} must not be empty")
        if len(names) != len(set(names)):
            raise ValueError(f"{info.field_name} entries must be unique")
        return tuple(sorted(names))

    @field_validator("auth_search_labels")
    @classmethod
    def normalize_optional_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        names = tuple(require_durable_clean_nonblank(item, "auth_search_labels") for item in value)
        if len(names) != len(set(names)):
            raise ValueError("auth_search_labels entries must be unique")
        return tuple(sorted(names))

    @field_validator("projections")
    @classmethod
    def validate_projections(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(value) != _PROVIDER_CREDENTIAL_PROJECTIONS:
            raise ValueError("projections must enumerate the complete probe boundary")
        return tuple(value)

    @model_validator(mode="after")
    def validate_status_scope(self) -> ProviderCredentialIsolationVerification:
        expected = "verified" if self.scope == "isolated_guest" else "environment_minimized"
        if self.status != expected:
            raise ValueError("status must match the verification scope")
        if self.scope == "isolated_guest" and not self.auth_search_labels:
            raise ValueError("isolated_guest evidence requires auth search labels")
        if self.scope == "local_environment" and self.auth_search_labels:
            raise ValueError("local_environment evidence cannot claim auth path searches")
        return self


@dataclass(frozen=True, slots=True)
class _ProviderCredentialIsolationFailure:
    kind: Literal["cancelled", "runtime_error", "type_error", "value_error", "violation"]
    message: str = ""
    adapter: str = ""
    canary_label: str = ""
    projection: str = ""


async def verify_provider_credential_isolation(
    runner: Runner,
    *,
    adapter: str,
    scope: Literal["isolated_guest", "local_environment"],
    provider_canaries: Mapping[str, str],
    operational_env: Mapping[str, str],
    workload_env: Mapping[str, str] | None = None,
    guest_cwd: str | None = None,
    guest_auth_search_paths: Mapping[str, str] | None = None,
    timeout_s: int = 30,
) -> ProviderCredentialIsolationVerification:
    """Probe that provider credentials are absent from an execution boundary.

    Provider canaries are comparison-only trusted inputs: they are never passed
    to the runner. ``operational_env`` and ``workload_env`` are positive
    controls which must be observable in the guest so an empty or broken probe
    cannot produce a false success. Raw stdout, stderr, artifacts, the complete
    guest environment, and the presence of Cayu auth files are inspected before
    content-free evidence is returned. Auth-file contents are never read.

    ``local_environment`` proves only default environment minimization. It does
    not claim filesystem or process isolation for :class:`LocalRunner`.
    """

    outcome: ProviderCredentialIsolationVerification | _ProviderCredentialIsolationFailure
    try:
        outcome = await _verify_provider_credential_isolation(
            runner,
            adapter=adapter,
            scope=scope,
            provider_canaries=provider_canaries,
            operational_env=operational_env,
            workload_env=workload_env,
            guest_cwd=guest_cwd,
            guest_auth_search_paths=guest_auth_search_paths,
            timeout_s=timeout_s,
        )
    except asyncio.CancelledError:
        outcome = _ProviderCredentialIsolationFailure(kind="cancelled")
    except ProviderCredentialIsolationViolation as exc:
        outcome = _ProviderCredentialIsolationFailure(
            kind="violation",
            adapter=exc.adapter,
            canary_label=exc.canary_label,
            projection=exc.projection,
        )
    except TypeError as exc:
        outcome = _ProviderCredentialIsolationFailure(
            kind="type_error",
            message=_safe_probe_failure_message(exc, provider_canaries),
        )
    except ValueError as exc:
        outcome = _ProviderCredentialIsolationFailure(
            kind="value_error",
            message=_safe_probe_failure_message(exc, provider_canaries),
        )
    except RuntimeError as exc:
        outcome = _ProviderCredentialIsolationFailure(
            kind="runtime_error",
            message=_safe_probe_failure_message(exc, provider_canaries),
        )
    except Exception:
        outcome = _ProviderCredentialIsolationFailure(
            kind="runtime_error",
            message="provider credential isolation verification failed unexpectedly",
        )

    # The public traceback frame must not retain trusted values even when a
    # capture-locals formatter walks it after a failed proof.
    del (
        runner,
        adapter,
        scope,
        provider_canaries,
        operational_env,
        workload_env,
        guest_cwd,
        guest_auth_search_paths,
        timeout_s,
    )
    return _resolve_provider_credential_isolation_outcome(outcome)


async def _verify_provider_credential_isolation(
    runner: Runner,
    *,
    adapter: str,
    scope: Literal["isolated_guest", "local_environment"],
    provider_canaries: Mapping[str, str],
    operational_env: Mapping[str, str],
    workload_env: Mapping[str, str] | None,
    guest_cwd: str | None,
    guest_auth_search_paths: Mapping[str, str] | None,
    timeout_s: int,
) -> ProviderCredentialIsolationVerification:
    if not isinstance(runner, Runner):
        raise TypeError("runner must implement Runner")
    adapter = require_clean_nonblank(adapter, "adapter")
    if scope not in {"isolated_guest", "local_environment"}:
        raise ValueError("scope must be isolated_guest or local_environment")
    if isinstance(runner, LocalRunner) and scope == "isolated_guest":
        raise ValueError("LocalRunner cannot prove isolated_guest credential isolation")
    canaries = _copy_secret_canaries(provider_canaries)
    if any(canary in adapter for canary in canaries.values()):
        raise ValueError("adapter must not contain provider credential canaries")
    operational_controls = _copy_probe_environment(operational_env, "operational_env")
    workload_controls = _copy_probe_environment(workload_env or {}, "workload_env")
    auth_search_paths = _copy_guest_auth_search_paths(
        guest_auth_search_paths or {},
        canaries=canaries,
    )
    if scope == "local_environment" and auth_search_paths:
        raise ValueError("local_environment cannot claim filesystem-level guest auth path searches")
    if guest_cwd is not None:
        guest_cwd = require_nonblank(guest_cwd, "guest_cwd")
        if "\x00" in guest_cwd or "\n" in guest_cwd or "\r" in guest_cwd:
            raise ValueError("guest_cwd must be a single filesystem path")
        if any(canary in guest_cwd for canary in canaries.values()):
            raise ValueError("guest_cwd must not contain provider credential canaries")
    duplicate_controls = set(operational_controls).intersection(workload_controls)
    if duplicate_controls:
        raise ValueError("operational_env and workload_env names must be distinct")
    controls = {**operational_controls, **workload_controls}
    if _PROVIDER_CREDENTIAL_DETECTOR_CONTROL_ENV in controls:
        raise ValueError(
            "positive controls cannot use the reserved provider credential detector name"
        )
    if not controls:
        raise ValueError("at least one positive control is required")
    probe_environment = {
        **controls,
        _PROVIDER_CREDENTIAL_DETECTOR_CONTROL_ENV: (_PROVIDER_CREDENTIAL_DETECTOR_CONTROL_VALUE),
    }
    rendered_controls = _stable_probe_text(probe_environment)
    if any(canary in rendered_controls for canary in canaries.values()):
        raise ValueError("positive controls must not contain provider credential canaries")
    if type(timeout_s) is not int:
        raise TypeError("timeout_s must be an integer")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be greater than zero")

    result = None
    cancellation: asyncio.CancelledError | None = None
    try:
        result = await runner.exec(
            _provider_credential_probe_command(
                inspect_auth_paths=scope == "isolated_guest",
                auth_search_paths=tuple(auth_search_paths.values()),
                canary_descriptors=tuple(
                    (label, len(canary), hashlib.sha256(canary.encode()).hexdigest())
                    for label, canary in canaries.items()
                ),
            ),
            cwd=guest_cwd,
            env=probe_environment,
            timeout_s=timeout_s,
            output_limit_bytes=_PROVIDER_CREDENTIAL_PROBE_OUTPUT_LIMIT_BYTES,
        )
    except asyncio.CancelledError as exc:
        cancellation = _sanitize_probe_cancellation(exc, canaries)
    except Exception:
        pass
    if cancellation is not None:
        # Keep the public traceback frame content-free as well as the exception:
        # capture-locals formatters must not recover trusted canary inputs.
        del runner, provider_canaries, operational_env, workload_env, canaries, controls, result
        raise cancellation
    if result is None:
        raise RuntimeError("provider credential isolation probe execution failed")

    observed = _parse_provider_credential_probe(result.stdout)
    environment = observed.get("environment")
    auth_paths = observed.get("auth_paths")
    auth_scan_complete = observed.get("auth_scan_complete")
    canary_matches = observed.get("provider_canary_matches")
    detector_control_match = observed.get("detector_control_match")
    if isinstance(environment, Mapping):
        _raise_on_provider_canary(
            environment,
            projection="environment",
            adapter=adapter,
            canaries=canaries,
        )
    if isinstance(auth_paths, Mapping):
        _raise_on_provider_canary(
            auth_paths,
            projection="auth_paths",
            adapter=adapter,
            canaries=canaries,
        )

    _raise_on_provider_canary(
        result.stdout,
        projection="stdout",
        adapter=adapter,
        canaries=canaries,
    )
    _raise_on_provider_canary(
        result.stderr,
        projection="stderr",
        adapter=adapter,
        canaries=canaries,
    )
    _raise_on_provider_canary(
        result.artifacts,
        projection="artifacts",
        adapter=adapter,
        canaries=canaries,
    )

    if (
        type(environment) is not dict
        or type(auth_paths) is not dict
        or type(auth_scan_complete) is not bool
        or type(canary_matches) is not list
        or any(type(label) is not str or label not in canaries for label in canary_matches)
        or canary_matches != sorted(set(canary_matches))
        or type(detector_control_match) is not bool
    ):
        raise RuntimeError("provider credential isolation probe returned malformed output")
    if not detector_control_match:
        raise RuntimeError(
            "provider credential isolation detector positive control was not observed"
        )
    if canary_matches:
        raise ProviderCredentialIsolationViolation(
            adapter=adapter,
            canary_label=sorted(canary_matches)[0],
            projection="environment",
        )
    if scope == "isolated_guest" and not auth_scan_complete:
        raise RuntimeError("provider credential isolation guest auth path scan was incomplete")
    unexpected_provider_env = sorted(
        name
        for name in _PROVIDER_CREDENTIAL_ENV_NAMES.intersection(environment)
        if name not in controls
    )
    if unexpected_provider_env:
        raise RuntimeError(
            "provider credential isolation probe observed an undeclared provider environment"
        )
    if scope == "isolated_guest" and auth_paths:
        raise RuntimeError(
            "provider credential isolation probe observed an unexpected guest auth path"
        )

    if (
        result.exit_code != 0
        or result.timed_out
        or result.cancelled
        or result.stdout_truncated
        or result.stderr_truncated
    ):
        raise RuntimeError("provider credential isolation probe did not complete cleanly")

    missing_controls = [name for name, value in controls.items() if environment.get(name) != value]
    if missing_controls:
        raise RuntimeError("provider credential isolation positive control was not observed")

    return ProviderCredentialIsolationVerification(
        status="verified" if scope == "isolated_guest" else "environment_minimized",
        adapter=adapter,
        scope=scope,
        canary_labels=tuple(canaries),
        auth_search_labels=(
            tuple(
                (
                    *_PROVIDER_CREDENTIAL_AUTH_SEARCH_LABELS,
                    *(("configured_working_directory",) if guest_cwd is not None else ()),
                    *auth_search_paths,
                )
            )
            if scope == "isolated_guest"
            else ()
        ),
        positive_controls=tuple(controls),
    )


def _safe_probe_failure_message(
    exc: Exception,
    provider_canaries: object,
) -> str:
    message = str(exc)
    if isinstance(provider_canaries, Mapping):
        for raw_canary in provider_canaries.values():
            if isinstance(raw_canary, str) and raw_canary:
                message = message.replace(raw_canary, "[provider credential]")
    return message or "provider credential isolation verification failed"


def _resolve_provider_credential_isolation_outcome(
    outcome: ProviderCredentialIsolationVerification | _ProviderCredentialIsolationFailure,
) -> ProviderCredentialIsolationVerification:
    if isinstance(outcome, ProviderCredentialIsolationVerification):
        return outcome
    if outcome.kind == "cancelled":
        cancellation = asyncio.CancelledError("provider credential isolation probe cancelled")
        vars(cancellation)["artifacts"] = []
        raise cancellation from None
    if outcome.kind == "violation":
        raise ProviderCredentialIsolationViolation(
            adapter=outcome.adapter,
            canary_label=outcome.canary_label,
            projection=outcome.projection,
        ) from None
    if outcome.kind == "type_error":
        raise TypeError(outcome.message) from None
    if outcome.kind == "value_error":
        raise ValueError(outcome.message) from None
    raise RuntimeError(outcome.message) from None


def _copy_secret_canaries(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("provider_canaries must be a mapping")
    copied: dict[str, str] = {}
    seen_values: set[str] = set()
    for raw_label, raw_canary in value.items():
        label = require_clean_nonblank(raw_label, "provider_canaries label")
        if _EVIDENCE_NAME_PATTERN.fullmatch(label) is None:
            raise ValueError("provider credential canary labels must be safe identifiers")
        canary = require_nonblank(raw_canary, "provider_canaries value")
        if len(canary) < 16:
            raise ValueError("provider credential canaries must contain at least 16 characters")
        if canary in seen_values:
            raise ValueError("provider credential canary values must be unique")
        copied[label] = canary
        seen_values.add(canary)
    if not copied:
        raise ValueError("provider_canaries must not be empty")
    if any(canary in label for label in copied for canary in copied.values()):
        raise ValueError(
            "provider credential canary labels must not contain provider credential canaries"
        )
    return dict(sorted(copied.items()))


def _copy_probe_environment(value: Mapping[str, str], field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    copied: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        name = require_clean_nonblank(raw_name, f"{field_name} name")
        if _ENVIRONMENT_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(f"{field_name} names must be portable environment names")
        copied[name] = require_nonblank(raw_value, f"{field_name} value")
    return copied


def _copy_guest_auth_search_paths(
    value: Mapping[str, str],
    *,
    canaries: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("guest_auth_search_paths must be a mapping")
    copied: dict[str, str] = {}
    for raw_label, raw_path in value.items():
        label = require_clean_nonblank(raw_label, "guest_auth_search_paths label")
        if _EVIDENCE_NAME_PATTERN.fullmatch(label) is None:
            raise ValueError("guest auth search labels must be safe identifiers")
        path = require_nonblank(raw_path, f"guest_auth_search_paths.{label}")
        if "\x00" in path or "\n" in path or "\r" in path:
            raise ValueError("guest auth search paths must be single filesystem paths")
        if any(canary in path for canary in canaries.values()):
            raise ValueError(
                "guest auth search paths must not contain provider credential canaries"
            )
        copied[label] = path
    if len(copied.values()) != len(set(copied.values())):
        raise ValueError("guest auth search paths must be unique")
    return dict(sorted(copied.items()))


def _provider_credential_probe_command(
    *,
    inspect_auth_paths: bool,
    auth_search_paths: tuple[str, ...] = (),
    canary_descriptors: tuple[tuple[str, int, str], ...],
) -> ExecCommand:
    auth_probe = "auth_paths={}\nauth_scan_complete=True\n"
    if inspect_auth_paths:
        auth_probe = (
            f"paths={_PROVIDER_CREDENTIAL_AUTH_PATHS!r}\n"
            f"configured_roots={auth_search_paths!r}\n"
            "home=pathlib.Path.home()/'.cayu'/'auth.json'\n"
            "workspace=pathlib.Path.cwd()/'.cayu'/'auth.json'\n"
            "configured=os.environ.get('CAYU_HOME')\n"
            "candidates=set(paths+(str(home),str(workspace)))\n"
            "if configured: candidates.add(str(pathlib.Path(configured)/'auth.json'))\n"
            "configured_root_paths={pathlib.Path(raw) for raw in configured_roots}\n"
            "search_roots={pathlib.Path.home(),pathlib.Path.cwd(),pathlib.Path('/workspace')}\n"
            "search_roots.update(configured_root_paths)\n"
            "auth_scan_complete=True\n"
            "scanned_directories=0\n"
            "scanned_directory_ids=set()\n"
            "def scan_error(_error):\n"
            " global auth_scan_complete\n"
            " auth_scan_complete=False\n"
            "for root in sorted(search_roots,key=str):\n"
            " candidates.add(str(root/'.cayu'/'auth.json'))\n"
            " candidates.add(str(root/'.aws'/'credentials'))\n"
            " candidates.add(str(root/'.aws'/'config'))\n"
            " candidates.add(str(root/'.config'/'gcloud'/'application_default_credentials.json'))\n"
            " try:\n"
            "  root_is_dir=root.is_dir()\n"
            " except OSError:\n"
            "  auth_scan_complete=False\n"
            "  continue\n"
            " if not root_is_dir:\n"
            "  if root in configured_root_paths: auth_scan_complete=False\n"
            "  continue\n"
            " for current,dirs,files in os.walk("
            "root,topdown=True,onerror=scan_error,followlinks=True):\n"
            "  scanned_directories+=1\n"
            f"  if scanned_directories>{_PROVIDER_CREDENTIAL_AUTH_SCAN_MAX_DIRECTORIES}:\n"
            "   auth_scan_complete=False\n"
            "   dirs[:]=[]\n"
            "   break\n"
            "  current_path=pathlib.Path(current)\n"
            "  try:\n"
            "   current_stat=current_path.stat()\n"
            "  except OSError:\n"
            "   auth_scan_complete=False\n"
            "   dirs[:]=[]\n"
            "   continue\n"
            "  current_id=(current_stat.st_dev,current_stat.st_ino)\n"
            "  if current_id in scanned_directory_ids:\n"
            "   dirs[:]=[]\n"
            "   continue\n"
            "  scanned_directory_ids.add(current_id)\n"
            "  traversable_dirs=[]\n"
            "  for name in sorted(dirs):\n"
            "   child=current_path/name\n"
            "   if child.is_symlink():\n"
            "    try:\n"
            "     if not child.resolve(strict=True).is_dir():\n"
            "      auth_scan_complete=False\n"
            "      continue\n"
            "    except OSError:\n"
            "     auth_scan_complete=False\n"
            "     continue\n"
            "   traversable_dirs.append(name)\n"
            "  dirs[:]=traversable_dirs\n"
            "  if 'auth.json' in files:\n"
            "   candidates.add(str(current_path/'auth.json'))\n"
            "  if current_path.name=='.aws':\n"
            "   for name in ('credentials','config'):\n"
            "    if name in files: candidates.add(str(current_path/name))\n"
            "  if 'application_default_credentials.json' in files:\n"
            "   candidates.add(str(current_path/'application_default_credentials.json'))\n"
            "auth_paths={}\n"
            "for raw in sorted(candidates):\n"
            " try:\n"
            "  if pathlib.Path(raw).is_file(): auth_paths[raw]='present'\n"
            " except OSError:\n"
            "  auth_paths[raw]='unreadable'\n"
        )
    script = (
        "import hashlib,json,os,pathlib\n"
        f"canary_descriptors={canary_descriptors!r}\n"
        "environment=dict(os.environ)\n"
        "provider_canary_matches=sorted("
        "label for label,length,fingerprint in canary_descriptors "
        "if any(hashlib.sha256(value[start:start+length].encode()).hexdigest()==fingerprint "
        "for value in environment.values() "
        "for start in range(max(0,len(value)-length+1))))\n"
        "detector_control_match=("
        "hashlib.sha256(environment.get("
        f"{_PROVIDER_CREDENTIAL_DETECTOR_CONTROL_ENV!r},'').encode()).hexdigest()=="
        f"{hashlib.sha256(_PROVIDER_CREDENTIAL_DETECTOR_CONTROL_VALUE.encode()).hexdigest()!r}"
        ")\n"
        f"{auth_probe}"
        "print(json.dumps({'environment':environment,'auth_paths':auth_paths,"
        "'auth_scan_complete':auth_scan_complete,"
        "'provider_canary_matches':provider_canary_matches,"
        "'detector_control_match':detector_control_match},sort_keys=True))\n"
    )
    return ExecCommand.process("python3", "-c", script)


def _parse_provider_credential_probe(stdout: str) -> dict[str, Any]:
    try:
        parsed = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _stable_probe_text(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return repr(value)


def _raise_on_provider_canary(
    value: object,
    *,
    projection: str,
    adapter: str,
    canaries: Mapping[str, str],
) -> None:
    rendered = _stable_probe_text(value)
    for label, canary in canaries.items():
        if canary in rendered:
            raise ProviderCredentialIsolationViolation(
                adapter=adapter,
                canary_label=label,
                projection=projection,
            )


def _sanitize_probe_cancellation(
    exc: asyncio.CancelledError,
    canaries: Mapping[str, str],
) -> asyncio.CancelledError:
    del canaries
    safe = asyncio.CancelledError("provider credential isolation probe cancelled")
    if hasattr(exc, "artifacts"):
        vars(safe)["artifacts"] = []
    safe.__cause__ = None
    safe.__context__ = None
    return safe


class ToolEffectVerificationStatus(StrEnum):
    """Outcome of one scoped tool-effect verification run."""

    CONSISTENT = "consistent"
    MISMATCH = "mismatch"
    OBSERVED = "observed"
    EXECUTION_FAILED = "execution_failed"


class ToolEffectVerification(BaseModel):
    """Content-free evidence from one isolated-workspace tool invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    status: ToolEffectVerificationStatus
    agent_name: str
    tool_name: str
    declared_effect: ToolEffect
    observation_boundary: Literal["isolated_workspace"] = _BOUNDARY_NAME
    created_paths: tuple[str, ...] = ()
    updated_paths: tuple[str, ...] = ()
    deleted_paths: tuple[str, ...] = ()
    observed_mutation: StrictBool
    execution_succeeded: StrictBool
    result_is_error: StrictBool | None = None
    exception_type: str | None = None
    timeout_seconds: float
    workspace_max_entries: int
    workspace_max_files: int
    workspace_max_file_bytes: int
    workspace_max_total_bytes: int
    unobserved_systems: tuple[str, ...]
    limitations: tuple[str, ...] = _LIMITATIONS

    @field_validator("agent_name", "tool_name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("created_paths", "updated_paths", "deleted_paths")
    @classmethod
    def normalize_paths(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        paths = tuple(require_durable_nonblank(path, info.field_name) for path in value)
        if len(paths) != len(set(paths)):
            raise ValueError(f"{info.field_name} entries must be unique")
        return tuple(sorted(paths))

    @field_validator("unobserved_systems")
    @classmethod
    def normalize_unobserved_systems(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        systems = tuple(
            require_durable_clean_nonblank(item, "unobserved_systems") for item in value
        )
        if len(systems) != len(set(systems)):
            raise ValueError("unobserved_systems entries must be unique")
        return tuple(sorted(systems))

    @field_validator("exception_type")
    @classmethod
    def validate_exception_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, "exception_type")

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(require_durable_text(item, "limitations") for item in value)

    @model_validator(mode="after")
    def validate_evidence(self) -> ToolEffectVerification:
        changed = bool(self.created_paths or self.updated_paths or self.deleted_paths)
        if self.observed_mutation is not changed:
            raise ValueError("observed_mutation must match the reported workspace changes")
        if self.execution_succeeded:
            if self.result_is_error is not False or self.exception_type is not None:
                raise ValueError("successful execution requires a non-error ToolResult")
        else:
            has_error_result = self.result_is_error is True
            has_exception = self.exception_type is not None
            if has_error_result == has_exception:
                raise ValueError(
                    "failed execution requires either an error ToolResult or an exception type"
                )
        _positive_float(self.timeout_seconds, "timeout_seconds")
        _positive_int(self.workspace_max_entries, "workspace_max_entries")
        _positive_int(self.workspace_max_files, "workspace_max_files")
        _positive_int(self.workspace_max_file_bytes, "workspace_max_file_bytes")
        _positive_int(self.workspace_max_total_bytes, "workspace_max_total_bytes")

        if self.status is ToolEffectVerificationStatus.CONSISTENT:
            valid = (
                self.declared_effect is ToolEffect.NONE
                and self.execution_succeeded
                and not self.observed_mutation
            )
        elif self.status is ToolEffectVerificationStatus.MISMATCH:
            valid = self.declared_effect is ToolEffect.NONE and self.observed_mutation
        elif self.status is ToolEffectVerificationStatus.OBSERVED:
            valid = self.declared_effect is not ToolEffect.NONE and self.execution_succeeded
        else:
            valid = not self.execution_succeeded and not (
                self.declared_effect is ToolEffect.NONE and self.observed_mutation
            )
        if not valid:
            raise ValueError("status does not match the verification evidence")
        return self


async def verify_tool_effect(
    app: CayuApp,
    *,
    agent_name: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    workspace_files: Mapping[str, bytes] | None = None,
    unobserved_systems: Iterable[str] = (),
    session_id: str = "tool-effect-verification",
    idempotency_key: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    allow_effectful_execution: bool = False,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    max_files: int = _DEFAULT_MAX_FILES,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES,
) -> ToolEffectVerification:
    """Invoke one registered tool against a bounded, isolated temporary workspace.

    ``NONE`` receives a scoped consistency verdict. ``IDEMPOTENT`` and
    ``EXTERNAL`` require ``allow_effectful_execution=True`` and are observed
    once without a replay-safety verdict. The tool runs directly: runtime
    policy, hooks, events, and the model loop are intentionally outside this
    deployment-readiness seam. One cooperative asyncio deadline covers seeding,
    both workspace snapshots, tool execution, and cleanup. Expiration raises
    ``TimeoutError`` without returning a verdict. A blocking tool or filesystem
    operation can delay that failure; a hard stop requires a process boundary.
    ``ProcessIsolatedTool`` is rejected because invoking it directly would bypass
    its runtime-owned process and recovery boundary.
    Arguments are validated and copied as portable durable JSON before the tool
    can be invoked.
    """

    verification_task = asyncio.current_task()
    if verification_task is None:
        raise RuntimeError("verify_tool_effect must run in an asyncio task")
    initial_cancellation_requests = verification_task.cancelling()
    if not isinstance(app, CayuApp):
        raise TypeError("app must be a CayuApp")
    agent_name = require_clean_nonblank(agent_name, "agent_name")
    tool_name = require_clean_nonblank(tool_name, "tool_name")
    if not isinstance(arguments, Mapping):
        raise TypeError("arguments must be a mapping")
    copied_arguments = copy_durable_json_object(dict(arguments), "arguments")
    copied_metadata = _copy_metadata(metadata)
    seeded_files = _copy_workspace_files(workspace_files)
    declared_unobserved = _copy_names(unobserved_systems, "unobserved_systems")
    session_id = require_clean_nonblank(session_id, "session_id")
    if idempotency_key is not None:
        idempotency_key = require_clean_nonblank(idempotency_key, "idempotency_key")
    if type(allow_effectful_execution) is not bool:
        raise TypeError("allow_effectful_execution must be a bool")
    timeout_seconds = _positive_float(timeout_seconds, "timeout_seconds")
    max_entries = _positive_int(max_entries, "max_entries")
    max_files = _positive_int(max_files, "max_files")
    max_file_bytes = _positive_int(max_file_bytes, "max_file_bytes")
    max_total_bytes = _positive_int(max_total_bytes, "max_total_bytes")
    _validate_seed_bounds(
        seeded_files,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )

    registered_agent = app.get_agent(agent_name)
    try:
        registered_tool = registered_agent.tools[tool_name]
    except KeyError as exc:
        raise KeyError(f"Tool not registered for agent {agent_name}: {tool_name}") from exc
    if type(registered_tool.tool) is ProcessIsolatedTool:
        raise ValueError(
            "verify_tool_effect does not support ProcessIsolatedTool; isolated tools require "
            "the runtime-owned process execution boundary."
        )
    effect = registered_tool.effect
    if effect is not ToolEffect.NONE and not allow_effectful_execution:
        raise ValueError(
            "IDEMPOTENT and EXTERNAL tools require allow_effectful_execution=True; "
            "the verifier executes them once and does not claim replay safety"
        )

    effective_metadata = {
        **copied_metadata,
        "tool_call_id": "tool-effect-verification",
        "tool_effect": effect.value,
    }
    if idempotency_key is not None:
        effective_metadata["idempotency_key"] = idempotency_key

    result_is_error: bool | None = None
    exception_type: str | None = None
    result: object | None = None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    async with asyncio.timeout_at(deadline):
        with tempfile.TemporaryDirectory(prefix="cayu-tool-effect-") as directory:
            workspace = LocalWorkspace(directory, workspace_id=_BOUNDARY_NAME)
            await _seed_workspace(
                workspace,
                seeded_files,
                clock=loop.time,
                deadline=deadline,
            )
            before = _capture_workspace(
                workspace,
                max_entries=max_entries,
                max_files=max_files,
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
                clock=loop.time,
                deadline=deadline,
            )
            context = ToolContext(
                session_id=session_id,
                agent_name=agent_name,
                environment_name=_BOUNDARY_NAME,
                workspace_id=workspace.id,
                idempotency_key=idempotency_key,
                workspace=workspace,
                metadata=effective_metadata,
            )
            tool_task = asyncio.create_task(
                registered_tool.tool.run(context, copied_arguments),
                name=f"cayu-tool-effect-verification:{agent_name}:{tool_name}",
            )
            try:
                result = await tool_task
            except Exception as exc:
                exception_type = type(exc).__name__
            if verification_task.cancelling() > initial_cancellation_requests:
                raise asyncio.CancelledError
            _raise_if_deadline_exceeded(loop.time, deadline)
            if exception_type is None:
                if type(result) is ToolResult:
                    result_is_error = result.is_error
                else:
                    exception_type = "InvalidToolResult"
            after = _capture_workspace(
                workspace,
                max_entries=max_entries,
                max_files=max_files,
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
                clock=loop.time,
                deadline=deadline,
            )
        _raise_if_deadline_exceeded(loop.time, deadline)

    created, updated, deleted = _compare_snapshots(before, after)
    observed_mutation = bool(created or updated or deleted)
    execution_succeeded = exception_type is None and result_is_error is False
    if effect is ToolEffect.NONE and observed_mutation:
        status = ToolEffectVerificationStatus.MISMATCH
    elif not execution_succeeded:
        status = ToolEffectVerificationStatus.EXECUTION_FAILED
    elif effect is ToolEffect.NONE:
        status = ToolEffectVerificationStatus.CONSISTENT
    else:
        status = ToolEffectVerificationStatus.OBSERVED

    return ToolEffectVerification(
        status=status,
        agent_name=agent_name,
        tool_name=tool_name,
        declared_effect=effect,
        created_paths=created,
        updated_paths=updated,
        deleted_paths=deleted,
        observed_mutation=observed_mutation,
        execution_succeeded=execution_succeeded,
        result_is_error=result_is_error,
        exception_type=exception_type,
        timeout_seconds=timeout_seconds,
        workspace_max_entries=max_entries,
        workspace_max_files=max_files,
        workspace_max_file_bytes=max_file_bytes,
        workspace_max_total_bytes=max_total_bytes,
        unobserved_systems=tuple(sorted(set(_BASE_UNOBSERVED_SYSTEMS) | set(declared_unobserved))),
    )


def _copy_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    return copy_json_value(dict(value), "metadata")


def _copy_workspace_files(value: Mapping[str, bytes] | None) -> tuple[tuple[str, bytes], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError("workspace_files must be a mapping")
    files: list[tuple[str, bytes]] = []
    for path, content in value.items():
        workspace_path = require_nonblank(path, "workspace_files path")
        if type(content) is not bytes:
            raise TypeError("workspace_files values must be bytes")
        files.append((workspace_path, bytes(content)))
    return tuple(sorted(files))


def _copy_names(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of strings")
    try:
        names = tuple(require_clean_nonblank(value, field_name) for value in values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of strings") from exc
    if len(names) != len(set(names)):
        raise ValueError(f"{field_name} entries must be unique")
    return tuple(sorted(names))


def _positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return value


def _positive_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be a number")
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field_name} must be finite and greater than zero")
    return normalized


def _validate_seed_bounds(
    files: tuple[tuple[str, bytes], ...],
    *,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
) -> None:
    if len(files) > max_files:
        raise ValueError("isolated workspace observation exceeds max_files")
    total_bytes = 0
    for path, content in files:
        if len(content) > max_file_bytes:
            raise ValueError(f"isolated workspace file exceeds max_file_bytes: {path}")
        total_bytes += len(content)
        if total_bytes > max_total_bytes:
            raise ValueError("isolated workspace observation exceeds max_total_bytes")


async def _seed_workspace(
    workspace: LocalWorkspace,
    files: tuple[tuple[str, bytes], ...],
    *,
    clock: Callable[[], float],
    deadline: float,
) -> None:
    for path, content in files:
        _raise_if_deadline_exceeded(clock, deadline)
        write_task = asyncio.create_task(
            workspace.write_bytes(path, content),
            name="cayu-tool-effect-verification:seed-workspace",
        )
        try:
            await asyncio.shield(write_task)
        except asyncio.CancelledError:
            try:
                await write_task
            finally:
                raise
        _raise_if_deadline_exceeded(clock, deadline)


def _capture_workspace(
    workspace: LocalWorkspace,
    *,
    max_entries: int,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
    clock: Callable[[], float],
    deadline: float,
) -> dict[str, str]:
    paths = _bounded_workspace_files(
        workspace.root,
        max_entries=max_entries,
        max_files=max_files,
        clock=clock,
        deadline=deadline,
    )
    snapshot: dict[str, str] = {}
    total_bytes = 0
    for path in paths:
        _raise_if_deadline_exceeded(clock, deadline)
        remaining_total_bytes = max_total_bytes - total_bytes
        observed_bytes, digest = _hash_workspace_file(
            workspace.resolve(path),
            relative_path=path,
            max_file_bytes=max_file_bytes,
            remaining_total_bytes=remaining_total_bytes,
            clock=clock,
            deadline=deadline,
        )
        total_bytes += observed_bytes
        snapshot[path] = digest
    return snapshot


def _bounded_workspace_files(
    root: str | os.PathLike[str],
    *,
    max_entries: int,
    max_files: int,
    clock: Callable[[], float],
    deadline: float,
) -> tuple[str, ...]:
    directories: list[tuple[str | os.PathLike[str], str]] = [(root, "")]
    files: list[str] = []
    entry_count = 0
    while directories:
        directory, prefix = directories.pop()
        _raise_if_deadline_exceeded(clock, deadline)
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_count += 1
                if entry_count > max_entries:
                    raise ValueError("isolated workspace observation exceeds max_entries")
                relative_path = f"{prefix}/{entry.name}" if prefix else entry.name
                if entry.is_symlink():
                    _raise_if_deadline_exceeded(clock, deadline)
                    continue
                if entry.is_dir(follow_symlinks=False):
                    directories.append((entry.path, relative_path))
                elif entry.is_file(follow_symlinks=False):
                    files.append(relative_path)
                    if len(files) > max_files:
                        raise ValueError("isolated workspace observation exceeds max_files")
                _raise_if_deadline_exceeded(clock, deadline)
        _raise_if_deadline_exceeded(clock, deadline)
    return tuple(sorted(files))


def _hash_workspace_file(
    path: os.PathLike[str],
    *,
    relative_path: str,
    max_file_bytes: int,
    remaining_total_bytes: int,
    clock: Callable[[], float],
    deadline: float,
) -> tuple[int, str]:
    digest = hashlib.sha256(b"file\0")
    observed_bytes = 0
    with open(path, "rb") as file:
        initial_size = os.fstat(file.fileno()).st_size
        if initial_size > max_file_bytes:
            raise ValueError(f"isolated workspace file exceeds max_file_bytes: {relative_path}")
        if initial_size > remaining_total_bytes:
            raise ValueError("isolated workspace observation exceeds max_total_bytes")
        while True:
            _raise_if_deadline_exceeded(clock, deadline)
            chunk = file.read(
                min(
                    _HASH_CHUNK_BYTES,
                    max_file_bytes - observed_bytes + 1,
                    remaining_total_bytes - observed_bytes + 1,
                )
            )
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > max_file_bytes:
                raise ValueError(f"isolated workspace file exceeds max_file_bytes: {relative_path}")
            if observed_bytes > remaining_total_bytes:
                raise ValueError("isolated workspace observation exceeds max_total_bytes")
            digest.update(chunk)
    _raise_if_deadline_exceeded(clock, deadline)
    return observed_bytes, digest.hexdigest()


def _raise_if_deadline_exceeded(clock: Callable[[], float], deadline: float) -> None:
    if clock() >= deadline:
        raise TimeoutError


def _compare_snapshots(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    created = tuple(sorted(after.keys() - before.keys()))
    deleted = tuple(sorted(before.keys() - after.keys()))
    updated = tuple(
        sorted(path for path in before.keys() & after.keys() if before[path] != after[path])
    )
    return created, updated, deleted


__all__ = [
    "ProviderCredentialIsolationVerification",
    "ProviderCredentialIsolationViolation",
    "ToolEffectVerification",
    "ToolEffectVerificationStatus",
    "verify_provider_credential_isolation",
    "verify_tool_effect",
]
