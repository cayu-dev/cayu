"""Static checks and safe edits that let a deployed `cayu serve` start.

A deployed `cayu serve` needs Cayu's server dependencies installed and, unless a
maintained service factory owns access, a `[tool.cayu.serve].auth` target. These
helpers read `pyproject.toml` without importing the project or `cayu.server`, so
they work where the server extra is not installed locally.

The ready-made target `cayu.server.environment_auth:OPERATOR_BASIC_AUTH` exists
only in releases newer than 0.8.1. It is chosen only when the declared `cayu`
requirement guarantees such a release; otherwise a small project module composes
`BasicAuth` from the same variables, which every released version supports.
"""

from __future__ import annotations

import copy
import json
import re
import shlex
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from cayu._operator_credentials import (
    ENVIRONMENT_OPERATOR_AUTH_TARGET,
    LAST_RELEASE_WITHOUT_ENVIRONMENT_AUTH,
    OPERATOR_PASSWORD_VARIABLE,
    OPERATOR_USERNAME_VARIABLE,
)

_REQUIREMENT = re.compile(
    r"\s*(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*"
    r"(?:\[(?P<extras>[^\]]*)\])?(?P<rest>.*)\Z",
    re.DOTALL,
)
_SPECIFIER = re.compile(r"\s*(?P<op>~=|===|==|!=|<=|>=|<|>)\s*(?P<version>[^\s,;()]+)\s*\Z")
_VERSION = re.compile(r"v?(?:(?P<epoch>\d+)!)?(?P<release>\d+(?:\.\d+)*)(?P<suffix>.*)\Z", re.I)
_SERVE_HEADER = re.compile(r"^\[\s*tool\s*\.\s*cayu\s*\.\s*serve\s*\][ \t]*(?:#[^\n]*)?$", re.M)
# Each of these runtime extras includes every dependency in the server extra.
_SERVER_RUNTIME_EXTRAS = frozenset({"server", "server-settings", "all", "oidc"})

ServerExtraStatus = Literal[
    "present",
    "missing_extra",
    "no_cayu_dependency",
    "ambiguous",
    "unsupported_project",
]
AuthStatus = Literal["configured", "missing", "service_factory", "invalid"]


@dataclass(frozen=True)
class ServerExtraState:
    """Whether `[project].dependencies` installs Cayu's server dependencies."""

    status: ServerExtraStatus
    requirement: str | None = None
    replacement: str | None = None


@dataclass(frozen=True)
class ServeAuthState:
    """Whether `cayu serve` has an authentication owner outside `--dev`."""

    status: AuthStatus
    target: str | None = None


@dataclass(frozen=True)
class GeneratedAuthModule:
    """A project module that serves as `[tool.cayu.serve].auth` on any Cayu release."""

    path: str
    target: str
    content: str


GENERATED_AUTH_MODULE = GeneratedAuthModule(
    path="server_auth.py",
    target="server_auth:OPERATOR_BASIC_AUTH",
    content=f'''"""Operator authentication for `cayu serve`, written by `cayu cloud init`.

`[tool.cayu.serve].auth` names OPERATOR_BASIC_AUTH. It reads HTTP Basic
credentials from {OPERATOR_USERNAME_VARIABLE} and {OPERATOR_PASSWORD_VARIABLE}
when `cayu serve` starts, and the server refuses to start if either is unset
or empty. Cayu Cloud provides both to each Agent.

Cayu releases newer than {LAST_RELEASE_WITHOUT_ENVIRONMENT_AUTH} ship the same dependency as
{ENVIRONMENT_OPERATOR_AUTH_TARGET}. Once this project
requires one of them, you can point [tool.cayu.serve].auth there and delete
this file.
"""

import os

from cayu.server import BasicAuth

_VARIABLES = ("{OPERATOR_USERNAME_VARIABLE}", "{OPERATOR_PASSWORD_VARIABLE}")
_missing = [name for name in _VARIABLES if not os.environ.get(name, "").strip()]
if _missing:
    raise RuntimeError(
        "Basic authentication is not configured; unset or empty: "
        + ", ".join(_missing)
        + ". Set both {OPERATOR_USERNAME_VARIABLE} and {OPERATOR_PASSWORD_VARIABLE} "
        "before starting the server; it does not fall back to open access. "
        "Use `cayu serve --dev` for local development."
    )

OPERATOR_BASIC_AUTH = BasicAuth(
    username=os.environ["{OPERATOR_USERNAME_VARIABLE}"],
    password=os.environ["{OPERATOR_PASSWORD_VARIABLE}"],
)
''',
)


@dataclass(frozen=True)
class ServeSetupPlan:
    """Edits `cayu cloud init` makes, or the manual edits it asks for instead.

    `auth_module` is the project module the added auth target names, when the
    declared runtime does not guarantee the ready-made target. The caller writes
    it and must refuse rather than overwrite a different existing file.
    `ships_environment_auth` says whether the declared runtime guarantees the
    ready-made target, and `environment_auth_upgrade` lists the requirement edits
    it needs otherwise.
    """

    server_extra: ServerExtraState
    auth: ServeAuthState
    changes: tuple[str, ...]
    updated_text: str | None
    manual_edits: tuple[str, ...]
    auth_target: str | None = None
    auth_module: GeneratedAuthModule | None = None
    ships_environment_auth: bool = True
    environment_auth_upgrade: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.changes and not self.manual_edits


def is_cayu_serve_command(command: str | None) -> bool:
    try:
        words = shlex.split(command or "")
    except ValueError:
        words = (command or "").split()
    return words[:2] == ["cayu", "serve"]


def server_extra_state(document: Mapping[str, Any]) -> ServerExtraState:
    project = document.get("project")
    if not isinstance(project, Mapping):
        return ServerExtraState("unsupported_project")
    dynamic = project.get("dynamic", ())
    dependencies = project.get("dependencies")
    if (
        (isinstance(dynamic, list) and "dependencies" in dynamic)
        or not isinstance(dependencies, list)
        or any(not isinstance(item, str) for item in dependencies)
    ):
        return ServerExtraState("unsupported_project")
    matches = [item for item in dependencies if _requirement_name(item) == "cayu"]
    if not matches:
        return ServerExtraState("no_cayu_dependency")
    with_server = [
        item for item in matches if _SERVER_RUNTIME_EXTRAS.intersection(_requirement_extras(item))
    ]
    if with_server:
        return ServerExtraState("present", requirement=with_server[0])
    if len(matches) != 1:
        return ServerExtraState("ambiguous", requirement=matches[0])
    requirement = matches[0]
    return ServerExtraState(
        "missing_extra",
        requirement=requirement,
        replacement=_with_server_extra(requirement),
    )


def serve_auth_state(document: Mapping[str, Any]) -> ServeAuthState:
    tool = document.get("tool")
    cayu = tool.get("cayu") if isinstance(tool, Mapping) else None
    cayu = cayu if isinstance(cayu, Mapping) else {}
    if isinstance(cayu.get("service_factory"), str):
        return ServeAuthState("service_factory")
    serve = cayu.get("serve")
    if serve is None:
        return ServeAuthState("missing")
    if not isinstance(serve, Mapping):
        return ServeAuthState("invalid")
    target = serve.get("auth")
    if target is None:
        return ServeAuthState("missing")
    if not isinstance(target, str) or not target.strip():
        return ServeAuthState("invalid")
    return ServeAuthState("configured", target=target.strip())


def plan_serve_setup(text: str, *, root: Path | None = None) -> ServeSetupPlan:
    """Plan the smallest safe `pyproject.toml` edit that lets `cayu serve` start.

    An existing auth target is never replaced. A missing one becomes the
    ready-made environment target when the declared `cayu` requirement guarantees
    a release that ships it, and otherwise the generated `server_auth.py` module,
    which composes `BasicAuth` and works on every release. Edits are made only
    when the re-parsed result equals the original document plus the intended
    values; anything else becomes a manual edit for the caller to report.
    """

    document = tomllib.loads(text)
    extra = server_extra_state(document)
    auth = serve_auth_state(document)
    changes: list[str] = []
    manual: list[str] = []
    updated = text
    expected = copy.deepcopy(document)

    if extra.status == "missing_extra":
        assert extra.requirement is not None and extra.replacement is not None
        replaced = _replace_string_literal(updated, extra.requirement, extra.replacement)
        if replaced is None:
            manual.append(server_extra_edit(extra))
        else:
            updated = replaced
            dependencies = expected["project"]["dependencies"]
            dependencies[dependencies.index(extra.requirement)] = extra.replacement
            changes.append(
                f"[project].dependencies: {json.dumps(extra.requirement)} -> "
                f"{json.dumps(extra.replacement)}"
            )
    elif extra.status != "present":
        manual.append(server_extra_edit(extra))

    # The ready-made target is named only when every release the project may
    # install ships it; otherwise a project module composes BasicAuth directly.
    ships_environment_auth = declared_runtime_ships_environment_auth(document, root=root)
    auth_module = None if ships_environment_auth else GENERATED_AUTH_MODULE
    auth_target = ENVIRONMENT_OPERATOR_AUTH_TARGET if auth_module is None else auth_module.target
    if auth.status == "missing":
        added = _add_auth_target(updated, auth_target)
        if added is None:
            manual.append(serve_auth_edit(document, ships_environment_auth))
        else:
            updated = added
            cayu = expected.setdefault("tool", {}).setdefault("cayu", {})
            cayu.setdefault("serve", {})["auth"] = auth_target
            changes.append(f"[tool.cayu.serve].auth = {json.dumps(auth_target)}")
    elif auth.status == "invalid":
        manual.append(serve_auth_edit(document, ships_environment_auth))

    if changes and not manual:
        try:
            verified = tomllib.loads(updated) == expected
        except tomllib.TOMLDecodeError:
            verified = False
    else:
        verified = False
    if changes and not verified:
        # Describe every needed edit rather than writing a partial change.
        manual = []
        if extra.status != "present":
            manual.append(server_extra_edit(extra))
        if auth.status in {"missing", "invalid"}:
            manual.append(serve_auth_edit(document, ships_environment_auth))
        changes = []
    adds_auth = auth.status == "missing" and bool(changes)
    return ServeSetupPlan(
        server_extra=extra,
        auth=auth,
        changes=tuple(changes),
        updated_text=updated if changes else None,
        manual_edits=tuple(manual),
        auth_target=auth_target if adds_auth else None,
        auth_module=auth_module if adds_auth else None,
        ships_environment_auth=ships_environment_auth,
        environment_auth_upgrade=(
            () if ships_environment_auth else environment_auth_upgrade_edits(document)
        ),
    )


def requirement_ships_environment_auth(requirement: str) -> bool:
    """Whether every version a `cayu` requirement allows ships the ready-made target.

    Only version specifiers are read. A direct URL, a requirement without a lower
    bound, or one that still allows 0.8.1 or older returns False.
    """

    return any(
        _guarantees_newer(operator, version)
        for operator, version in _version_specifiers(requirement) or ()
    )


def declared_runtime_ships_environment_auth(
    document: Mapping[str, Any], *, root: Path | None = None
) -> bool:
    """Whether the declared `cayu` runtime guarantees `cayu.server.environment_auth`.

    The `cayu` requirements in `[project].dependencies`, or the uv overrides that
    replace them, must exclude every release up to and including 0.8.1. When
    `[tool.uv.sources]` installs `cayu` from elsewhere, only a local path source
    under `root` that contains the module counts.
    """

    project = document.get("project")
    dependencies = project.get("dependencies") if isinstance(project, Mapping) else None
    requirements = _cayu_requirements(dependencies)
    if not requirements:
        return False
    uv = _uv_settings(document)
    source = _cayu_source(uv.get("sources"))
    if source is not None:
        return _path_source_ships_environment_auth(source, root)
    effective = _cayu_requirements(uv.get("override-dependencies")) or requirements
    unconditional = [item for item in effective if not _requirement_marker(item)]
    # Requirements combine, so one unconditional guarantee is enough; conditional
    # ones guarantee it only when all of them do.
    return any(requirement_ships_environment_auth(item) for item in unconditional) or all(
        requirement_ships_environment_auth(item) for item in effective
    )


def environment_auth_upgrade_edits(document: Mapping[str, Any]) -> tuple[str, ...]:
    """The exact `pyproject.toml` edits that make the ready-made target available.

    Lists the runtime `cayu` requirement (or uv override) that does not guarantee a
    release newer than 0.8.1, and any other `cayu` requirement, such as a
    development pin, that would hold the lock at or below it.
    """

    minimum = f">{LAST_RELEASE_WITHOUT_ENVIRONMENT_AUTH}"
    project = document.get("project")
    project = project if isinstance(project, Mapping) else {}
    uv = _uv_settings(document)
    locations: list[tuple[str, object, bool]] = [
        ("[project].dependencies", project.get("dependencies"), True),
    ]
    for table, values in (
        ("project.optional-dependencies", project.get("optional-dependencies")),
        ("dependency-groups", document.get("dependency-groups")),
    ):
        if isinstance(values, Mapping):
            locations.extend((f"[{table}].{name}", items, False) for name, items in values.items())
    for key in ("dev-dependencies", "constraint-dependencies", "override-dependencies"):
        locations.append((f"[tool.uv].{key}", uv.get(key), key == "override-dependencies"))
    edits: list[str] = []
    for location, items, runtime in locations:
        for requirement in _cayu_requirements(items):
            if requirement_ships_environment_auth(requirement) or (
                not runtime and not _caps_at_last_release(requirement)
            ):
                continue
            replacement = _with_specifier(requirement, minimum)
            edits.append(
                f"In {location}, change {json.dumps(requirement)} to {json.dumps(replacement)}."
            )
    if _cayu_source(uv.get("sources")) is not None:
        edits.append(
            "Remove the cayu entry from [tool.uv.sources] so a released cayu is installed."
        )
    return tuple(edits)


def _requirement_name(requirement: str) -> str | None:
    match = _REQUIREMENT.match(requirement)
    if match is None:
        return None
    return re.sub(r"[-_.]+", "-", match.group("name")).lower()


def _requirement_extras(requirement: str) -> tuple[str, ...]:
    match = _REQUIREMENT.match(requirement)
    if match is None or match.group("extras") is None:
        return ()
    return tuple(
        re.sub(r"[-_.]+", "-", item.strip()).lower()
        for item in match.group("extras").split(",")
        if item.strip()
    )


def _with_server_extra(requirement: str) -> str:
    match = _REQUIREMENT.match(requirement)
    assert match is not None
    extras = [item.strip() for item in (match.group("extras") or "").split(",") if item.strip()]
    extras.append("server")
    return f"{match.group('name')}[{','.join(extras)}]{match.group('rest')}"


def _replace_string_literal(text: str, value: str, replacement: str) -> str | None:
    for quote in ('"', "'"):
        if quote == "'" and ("'" in value or "\n" in value):
            continue
        literal = json.dumps(value) if quote == '"' else f"'{value}'"
        new_literal = json.dumps(replacement) if quote == '"' else f"'{replacement}'"
        if text.count(literal) == 1:
            return text.replace(literal, new_literal, 1)
    return None


def _add_auth_target(text: str, target: str) -> str | None:
    line = f"auth = {json.dumps(target)}\n"
    headers = list(_SERVE_HEADER.finditer(text))
    if len(headers) > 1:
        return None
    if headers:
        end = headers[0].end()
        separator = "" if text[end : end + 1] == "\n" else "\n"
        insert_at = end + 1 if separator == "" else end
        return f"{text[:insert_at]}{separator}{line}{text[insert_at:]}"
    prefix = text if text.endswith("\n") or not text else f"{text}\n"
    return f"{prefix}\n[tool.cayu.serve]\n{line}"


def server_extra_edit(extra: ServerExtraState) -> str:
    if extra.status in {"missing_extra", "ambiguous"} and extra.requirement is not None:
        replacement = extra.replacement or _with_server_extra(extra.requirement)
        return (
            f"In [project].dependencies, change {json.dumps(extra.requirement)} to "
            f"{json.dumps(replacement)}."
        )
    if extra.status == "unsupported_project":
        return (
            "Declare a static [project].dependencies list in pyproject.toml that includes "
            '"cayu[server]" (with any other extras you use, for example '
            '"cayu[postgres,server]").'
        )
    return (
        'Add "cayu[server]" (with any other extras you use, for example '
        '"cayu[postgres,server]") to [project].dependencies.'
    )


def serve_auth_edit(document: Mapping[str, Any], ships_environment_auth: bool) -> str:
    if ships_environment_auth:
        return (
            "Set [tool.cayu.serve].auth to an authentication target, for example:\n"
            "[tool.cayu.serve]\n"
            f"auth = {json.dumps(ENVIRONMENT_OPERATOR_AUTH_TARGET)}"
        )
    module = GENERATED_AUTH_MODULE
    upgrade = " ".join(environment_auth_upgrade_edits(document))
    return (
        "Set [tool.cayu.serve].auth to an authentication target. The declared cayu "
        f"requirement allows cayu {LAST_RELEASE_WITHOUT_ENVIRONMENT_AUTH} or older, which "
        f"lack {ENVIRONMENT_OPERATOR_AUTH_TARGET}, so either add {module.path} with "
        f"`OPERATOR_BASIC_AUTH = BasicAuth(username=..., password=...)` built from "
        f"{OPERATOR_USERNAME_VARIABLE} and {OPERATOR_PASSWORD_VARIABLE} and set:\n"
        "[tool.cayu.serve]\n"
        f"auth = {json.dumps(module.target)}\n"
        f"or require a newer cayu ({upgrade}) and set:\n"
        "[tool.cayu.serve]\n"
        f"auth = {json.dumps(ENVIRONMENT_OPERATOR_AUTH_TARGET)}"
    )


def lock_installs_server_extra(lock_text: str, project_name: str) -> bool | None:
    """Whether `uv.lock` installs Cayu's server dependencies for the project's own package.

    Cayu Cloud installs with `uv sync --frozen`, which trusts the lock over
    `pyproject.toml`. Returns ``None`` when the lock does not show the answer.
    """

    try:
        lock = tomllib.loads(lock_text)
    except tomllib.TOMLDecodeError:
        return None
    packages = lock.get("package")
    if not isinstance(packages, list):
        return None
    normalized = _normalize_name(project_name)
    for package in packages:
        if not isinstance(package, Mapping) or _normalize_name(package.get("name")) != normalized:
            continue
        source = package.get("source")
        if not isinstance(source, Mapping) or "." not in (
            source.get("editable"),
            source.get("virtual"),
        ):
            continue
        dependencies = package.get("dependencies")
        if not isinstance(dependencies, list):
            return None
        entries = [
            item
            for item in dependencies
            if isinstance(item, Mapping) and _normalize_name(item.get("name")) == "cayu"
        ]
        if not entries:
            return None
        return any(
            _normalize_name(extra) in _SERVER_RUNTIME_EXTRAS
            for item in entries
            for extra in item.get("extra", ())
        )
    return None


def _normalize_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return re.sub(r"[-_.]+", "-", value).lower()


def _uv_settings(document: Mapping[str, Any]) -> Mapping[str, Any]:
    tool = document.get("tool")
    uv = tool.get("uv") if isinstance(tool, Mapping) else None
    return uv if isinstance(uv, Mapping) else {}


def _cayu_source(table: object) -> object | None:
    if not isinstance(table, Mapping):
        return None
    for key, value in table.items():
        if isinstance(key, str) and re.sub(r"[-_.]+", "-", key).lower() == "cayu":
            return value
    return None


def _path_source_ships_environment_auth(source: object, root: Path | None) -> bool:
    path = cast("Mapping[str, object]", source).get("path") if isinstance(source, Mapping) else None
    if root is None or not isinstance(path, str):
        return False
    checkout = root / path
    return any(
        (checkout / prefix / "cayu" / "server" / "environment_auth.py").is_file()
        for prefix in ("src", ".")
    )


def _cayu_requirements(items: object) -> list[str]:
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, str) and _requirement_name(item) == "cayu"]


def _requirement_marker(requirement: str) -> str:
    match = _REQUIREMENT.match(requirement)
    return "" if match is None else match.group("rest").partition(";")[2].strip()


def _version_specifiers(requirement: str) -> tuple[tuple[str, str], ...] | None:
    """The `(operator, version)` pairs of a requirement, or None for a URL or bad text."""

    match = _REQUIREMENT.match(requirement)
    if match is None:
        return None
    text = match.group("rest").partition(";")[0].strip()
    if text.startswith("@"):
        return None
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if not text:
        return ()
    specifiers: list[tuple[str, str]] = []
    for part in text.split(","):
        specifier = _SPECIFIER.match(part)
        if specifier is None:
            return None
        specifiers.append((specifier.group("op"), specifier.group("version")))
    return tuple(specifiers)


def _release(version: str) -> tuple[tuple[int, ...], bool] | None:
    """The release segment of a PEP 440 version, and whether nothing follows it."""

    match = _VERSION.match(version.strip())
    if match is None or int(match.group("epoch") or 0):
        return None
    release = tuple(int(part) for part in match.group("release").split("."))
    return release, not match.group("suffix")


def _compare(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    width = max(len(left), len(right))
    padded_left = left + (0,) * (width - len(left))
    padded_right = right + (0,) * (width - len(right))
    return (padded_left > padded_right) - (padded_left < padded_right)


def _last_release() -> tuple[int, ...]:
    parsed = _release(LAST_RELEASE_WITHOUT_ENVIRONMENT_AUTH)
    assert parsed is not None
    return parsed[0]


def _guarantees_newer(operator: str, version: str) -> bool:
    """Whether one specifier excludes every release up to the last one without the target."""

    last = _last_release()
    if version.endswith(".*"):
        prefix = _release(version[:-2])
        return operator == "==" and prefix is not None and _compare(prefix[0], last) > 0
    parsed = _release(version)
    if parsed is None:
        return False
    release, plain = parsed
    if operator in {">=", "~=", "==", "==="}:
        return _compare(release, last) > 0
    if operator == ">":
        # `>V` excludes V's post-releases only when V is itself a plain release.
        order = _compare(release, last)
        return order > 0 or (order == 0 and plain)
    return False


def _caps_at_last_release(requirement: str) -> bool:
    """Whether a requirement excludes every release newer than the last one without it."""

    last = _last_release()
    next_patch = (*last[:-1], last[-1] + 1)
    for operator, version in _version_specifiers(requirement) or ():
        wildcard = version.endswith(".*")
        parsed = _release(version[:-2] if wildcard else version)
        if parsed is None:
            continue
        release = parsed[0]
        if operator == "~=":
            if len(release) < 2:
                continue
            # `~=X.Y.Z` allows `==X.Y.*`.
            release, wildcard = release[:-1], True
        if wildcard:
            caps = operator in {"==", "~="} and (
                _compare(release, last) < 0 and last[: len(release)] != release
            )
        elif operator == "<":
            caps = _compare(release, next_patch) <= 0
        else:
            caps = operator in {"==", "===", "<="} and _compare(release, last) <= 0
        if caps:
            return True
    return False


def _with_specifier(requirement: str, specifier: str) -> str:
    match = _REQUIREMENT.match(requirement)
    assert match is not None
    extras = [item.strip() for item in (match.group("extras") or "").split(",") if item.strip()]
    marker = _requirement_marker(requirement)
    return (
        match.group("name")
        + (f"[{','.join(extras)}]" if extras else "")
        + specifier
        + (f"; {marker}" if marker else "")
    )
