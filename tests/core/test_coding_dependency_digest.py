"""Dependency inputs that cover only what a pyproject.toml installs, and stale-input errors."""

from __future__ import annotations

import asyncio
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from cayu.environments import (
    CodingCommandAuthority,
    CodingDependencyInput,
    CodingToolchainError,
    CodingToolchainProfile,
    DockerCodingToolchainProfile,
    verify_coding_toolchain_dependencies,
    verify_local_coding_toolchain_dependencies,
)
from cayu.runners import DockerImageIdentity
from cayu.workspaces import LocalWorkspace

_PYPROJECT = b"""[project]
name = "demo"
version = "0.1.0"
description = "Demo"
requires-python = ">=3.12"
dependencies = ["cayu>=0.16"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
exclude-newer = 2026-01-01T00:00:00Z

[tool.pytest.ini_options]
markers = ["slow: slow tests"]
"""
_REBUILD = "uv run --no-sync python build_coding_image.py"


def _profile(*inputs: CodingDependencyInput, rebuild_command: str | None = None):
    return CodingToolchainProfile(
        profile_id="digest-scope",
        revision="1",
        command_authorities=(
            CodingCommandAuthority(
                selector="test",
                revision="1",
                description="Run tests.",
                exposure="named_check",
                executable="/usr/bin/true",
                max_arguments=0,
            ),
        ),
        dependency_inputs=inputs,
        rebuild_command=rebuild_command,
    )


def _verify_both(profile: CodingToolchainProfile, root: Path) -> None:
    verify_local_coding_toolchain_dependencies(profile, root)
    asyncio.run(verify_coding_toolchain_dependencies(profile, LocalWorkspace(root)))


def _changed_paths(profile: CodingToolchainProfile, root: Path) -> tuple[str, ...]:
    with pytest.raises(CodingToolchainError) as local:
        verify_local_coding_toolchain_dependencies(profile, root)
    with pytest.raises(CodingToolchainError) as workspace:
        asyncio.run(verify_coding_toolchain_dependencies(profile, LocalWorkspace(root)))
    assert local.value.code == workspace.value.code == "dependency_inputs_changed"
    assert str(local.value) == str(workspace.value)
    return local.value.paths


def test_tool_settings_do_not_make_a_pyproject_environment_input_stale(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_bytes(_PYPROJECT)
    profile = _profile(
        CodingDependencyInput.from_content(
            "pyproject.toml", _PYPROJECT, digest_scope="pyproject_environment"
        )
    )
    _verify_both(profile, tmp_path)

    edited = _PYPROJECT.replace(
        b'markers = ["slow: slow tests"]',
        b'markers = ["slow: slow tests", "docker: needs the image"]\n\n[tool.ruff]\nline-length = 99',
    ).replace(b'version = "0.1.0"', b'version = "0.2.0"')
    edited = edited.replace(b'description = "Demo"', b'description = "Renamed demo"')
    (tmp_path / "pyproject.toml").write_bytes(b"# reformatted\n" + edited)
    _verify_both(profile, tmp_path)


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (b'dependencies = ["cayu>=0.16"]', b'dependencies = ["cayu>=0.16", "httpx"]'),
        (b'requires-python = ">=3.12"', b'requires-python = ">=3.13"'),
        (b'dev = ["pytest>=8"]', b'dev = ["pytest>=9"]'),
        (b'name = "demo"', b'name = "demo-two"'),
        (b'requires = ["hatchling"]', b'requires = ["setuptools"]'),
        (b"exclude-newer = 2026-01-01T00:00:00Z", b"exclude-newer = 2026-02-01T00:00:00Z"),
        (b"exclude-newer = 2026-01-01T00:00:00Z", b'exclude-newer = "2026-01-01T00:00:00Z"'),
        (b"[tool.pytest.ini_options]", b'[dependency-groups]\nlint = ["ruff"]\n\n[tool.pytest]'),
        (b"[project]", b"[project"),
    ),
)
def test_what_a_pyproject_installs_still_makes_its_input_stale(
    tmp_path: Path, old: bytes, new: bytes
) -> None:
    profile = _profile(
        CodingDependencyInput.from_content(
            "pyproject.toml", _PYPROJECT, digest_scope="pyproject_environment"
        )
    )
    assert old in _PYPROJECT
    (tmp_path / "pyproject.toml").write_bytes(_PYPROJECT.replace(old, new))

    assert _changed_paths(profile, tmp_path) == ("pyproject.toml",)


def test_the_default_file_scope_keeps_hashing_exact_bytes_and_old_dumps() -> None:
    recorded = CodingDependencyInput.from_content("uv.lock", b"locked\n")
    assert recorded == CodingDependencyInput(
        path="uv.lock", content_sha256="sha256:" + sha256(b"locked\n").hexdigest()
    )
    # An input recorded before digest_scope existed dumps, and so fingerprints, the same.
    assert recorded.model_dump(mode="json") == {
        "path": "uv.lock",
        "content_sha256": recorded.content_sha256,
        "max_bytes": recorded.max_bytes,
    }
    assert recorded.matches(b"locked\n") and not recorded.matches(b"locked \n")

    scoped = CodingDependencyInput.from_content(
        "targets/app/pyproject.toml", _PYPROJECT, digest_scope="pyproject_environment"
    )
    assert scoped.model_dump(mode="json")["digest_scope"] == "pyproject_environment"
    assert CodingDependencyInput.model_validate(scoped.model_dump(mode="json")) == scoped
    assert scoped.content_sha256 != "sha256:" + sha256(_PYPROJECT).hexdigest()

    with pytest.raises(ValidationError, match="needs a pyproject.toml path"):
        CodingDependencyInput(
            path="uv.lock",
            content_sha256=recorded.content_sha256,
            digest_scope="pyproject_environment",
        )
    with pytest.raises(ValueError, match="not valid UTF-8 TOML"):
        CodingDependencyInput.from_content(
            "pyproject.toml", b"[project", digest_scope="pyproject_environment"
        )


def test_the_stale_error_names_the_changed_inputs_and_the_rebuild_command(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_bytes(_PYPROJECT)
    (tmp_path / "uv.lock").write_bytes(b"changed\n")
    inputs = (
        CodingDependencyInput.from_content(
            "pyproject.toml", _PYPROJECT, digest_scope="pyproject_environment"
        ),
        CodingDependencyInput.from_content("uv.lock", b"locked\n"),
    )

    with pytest.raises(CodingToolchainError) as guided:
        verify_local_coding_toolchain_dependencies(
            _profile(*inputs, rebuild_command=_REBUILD), tmp_path
        )
    assert str(guided.value) == (
        "Toolchain dependency inputs changed since the toolchain was built (1 of 2 declared "
        f"inputs: uv.lock); rebuild the toolchain with `{_REBUILD}`."
    )
    # Structured evidence still carries no paths.
    assert guided.value.diagnostic() == {
        "code": "dependency_inputs_changed",
        "path_count": 1,
        "paths_fingerprint": guided.value.paths_fingerprint,
    }

    with pytest.raises(CodingToolchainError) as unguided:
        verify_local_coding_toolchain_dependencies(_profile(*inputs), tmp_path)
    assert str(unguided.value).endswith(
        "inputs: uv.lock); a new admitted profile or build is required."
    )


def test_the_message_names_at_most_five_printable_inputs(tmp_path: Path) -> None:
    names = [f"inputs/{index}.lock" for index in range(6)] + ["inputs/" + "x" * 250 + ".lock"]
    (tmp_path / "inputs").mkdir()
    for name in names:
        (tmp_path / name).write_bytes(b"changed\n")
    profile = _profile(
        *(CodingDependencyInput.from_content(name, b"locked\n") for name in sorted(names))
    )

    with pytest.raises(CodingToolchainError) as changed:
        verify_local_coding_toolchain_dependencies(profile, tmp_path)
    message = str(changed.value)
    assert "(7 of 7 declared inputs: inputs/0.lock, inputs/1.lock, " in message
    assert "inputs/4.lock and 2 more)" in message
    assert "x" * 250 not in message


def test_rebuild_command_is_guidance_outside_the_fingerprint() -> None:
    lock = CodingDependencyInput.from_content("uv.lock", b"locked\n")
    plain = _profile(lock)
    guided = _profile(lock, rebuild_command=_REBUILD)

    assert guided.fingerprint == plain.fingerprint
    assert guided.dependency_identity == plain.dependency_identity
    assert "rebuild_command" not in plain.model_dump()
    docker = DockerCodingToolchainProfile.from_toolchain(
        guided,
        image_identity=DockerImageIdentity(
            reference="digest-scope:local", content_digest="sha256:" + "a" * 64
        ),
        platform_architecture="amd64",
    )
    assert docker.rebuild_command == _REBUILD
    assert docker.coding_toolchain().rebuild_command == _REBUILD
    for invalid in ("make `image`", "build\nimage", "  "):
        with pytest.raises(ValidationError):
            _profile(lock, rebuild_command=invalid)


def test_from_content_refuses_content_over_its_byte_limit() -> None:
    with pytest.raises(ValueError, match="exceeds its 4-byte limit"):
        CodingDependencyInput.from_content("uv.lock", b"locked\n", max_bytes=4)


def test_a_deeply_nested_pyproject_counts_as_changed_instead_of_crashing(tmp_path: Path) -> None:
    profile = _profile(
        CodingDependencyInput.from_content(
            "pyproject.toml", _PYPROJECT, digest_scope="pyproject_environment"
        )
    )
    depth = 5000
    nested = b'[project]\nname = "demo"\ndependencies = ' + b"[" * depth + b"]" * depth + b"\n"
    (tmp_path / "pyproject.toml").write_bytes(nested)

    assert _changed_paths(profile, tmp_path) == ("pyproject.toml",)
    with pytest.raises(ValueError):
        CodingDependencyInput.from_content(
            "pyproject.toml", nested, digest_scope="pyproject_environment"
        )
