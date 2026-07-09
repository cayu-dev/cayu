from __future__ import annotations

import tempfile

import pytest

from cayu.egress import CredentialMode
from cayu.runners.docker import DockerRunner
from cayu.runners.local import LocalRunner
from cayu.vaults import SecretRef, StaticVault

_SECRET_ENV = {"API_KEY": SecretRef(name="api_key")}


def _vault() -> StaticVault:
    return StaticVault({"api_key": "sk-secret"})


def test_local_default_mode_allows_raw_secret_env() -> None:
    with tempfile.TemporaryDirectory() as root:
        runner = LocalRunner(root, secret_env=_SECRET_ENV, secret_resolver=_vault())
    assert runner.credential_mode is CredentialMode.RAW_ENV
    assert runner.secret_env["API_KEY"].name == "api_key"


def test_local_virtual_egress_refuses_raw_secret_env() -> None:
    with tempfile.TemporaryDirectory() as root, pytest.raises(ValueError, match="virtual_egress"):
        LocalRunner(
            root,
            secret_env=_SECRET_ENV,
            secret_resolver=_vault(),
            credential_mode=CredentialMode.VIRTUAL_EGRESS,
        )


def test_local_virtual_egress_string_refuses_raw_secret_env() -> None:
    with tempfile.TemporaryDirectory() as root, pytest.raises(ValueError, match="virtual_egress"):
        LocalRunner(
            root,
            secret_env=_SECRET_ENV,
            secret_resolver=_vault(),
            credential_mode="virtual_egress",
        )


def test_local_trusted_tool_refuses_raw_secret_env() -> None:
    with tempfile.TemporaryDirectory() as root, pytest.raises(ValueError, match="trusted_tool"):
        LocalRunner(
            root,
            secret_env=_SECRET_ENV,
            secret_resolver=_vault(),
            credential_mode=CredentialMode.TRUSTED_TOOL,
        )


def test_local_trusted_tool_string_refuses_raw_secret_env() -> None:
    with tempfile.TemporaryDirectory() as root, pytest.raises(ValueError, match="trusted_tool"):
        LocalRunner(
            root,
            secret_env=_SECRET_ENV,
            secret_resolver=_vault(),
            credential_mode="trusted_tool",
        )


def test_local_opt_out_refuses_raw_secret_env() -> None:
    with (
        tempfile.TemporaryDirectory() as root,
        pytest.raises(ValueError, match="allow_raw_secret_env"),
    ):
        LocalRunner(
            root,
            secret_env=_SECRET_ENV,
            secret_resolver=_vault(),
            allow_raw_secret_env=False,
        )


def test_virtual_egress_without_secret_env_is_fine() -> None:
    # A virtual_egress runner carries the virtual cred as a plain env, not secret_env.
    with tempfile.TemporaryDirectory() as root:
        runner = LocalRunner(root, credential_mode="virtual_egress")
    assert runner.credential_mode is CredentialMode.VIRTUAL_EGRESS
    assert runner.secret_env == {}


def test_docker_virtual_egress_refuses_raw_secret_env() -> None:
    with pytest.raises(ValueError, match="virtual_egress"):
        DockerRunner(
            "agent",
            docker_path="/usr/bin/docker",
            secret_env=_SECRET_ENV,
            secret_resolver=_vault(),
            credential_mode="virtual_egress",
        )


def test_docker_trusted_tool_refuses_raw_secret_env() -> None:
    with pytest.raises(ValueError, match="trusted_tool"):
        DockerRunner(
            "agent",
            docker_path="/usr/bin/docker",
            secret_env=_SECRET_ENV,
            secret_resolver=_vault(),
            credential_mode="trusted_tool",
        )


def test_docker_default_stores_mode() -> None:
    runner = DockerRunner("agent", docker_path="/usr/bin/docker")
    assert runner.credential_mode is CredentialMode.RAW_ENV
