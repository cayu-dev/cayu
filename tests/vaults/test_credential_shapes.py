from __future__ import annotations

import base64
import subprocess

import pytest

from cayu.egress._docker_diagnostics import docker_setup_failure
from cayu.runtime._diagnostics import exception_diagnostic
from cayu.sessions._call_inspection import redact_inspection_value
from cayu.vaults import REDACTED_SECRET, SecretRedactor
from cayu.vaults._credential_shapes import mask_credential_shapes

# A GitHub App installation token in the newer form, a JWT after the prefix.
_INSTALLATION_TOKEN = "ghs_5486236_eyJhbGciOiJFUzI1NiJ9.eyJhdWQiOiJ0ZXN0In0.c2lnbmF0dXJl"
_BASIC = base64.b64encode(f"x-access-token:{_INSTALLATION_TOKEN}".encode()).decode()


def _git_timeout() -> subprocess.TimeoutExpired:
    return subprocess.TimeoutExpired(
        [
            "git",
            "-c",
            f"http.extraheader=AUTHORIZATION: basic {_BASIC}",
            "clone",
            "--quiet",
            "https://github.com/owner/repository.git",
        ],
        120,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"Authorization: Basic {_BASIC}", "Authorization: [M]"),
        (f"AUTHORIZATION: basic {_BASIC}", "AUTHORIZATION: [M]"),
        ("authorization: Bearer abc.def.ghi", "authorization: [M]"),
        ("Authorization: token gho_abcdefgh12345678", "Authorization: [M]"),
        ("Authorization: bare-credential-1234", "Authorization: [M]"),
        (f"http.extraheader=AUTHORIZATION: basic {_BASIC}", "http.extraheader=AUTHORIZATION: [M]"),
        ("authorization=Bearer abc123", "authorization=[M]"),
        ('{"Authorization": "Bearer abc123", "x": 1}', '{"Authorization": "[M]", "x": 1}'),
        ("{'authorization': 'Basic abc123'}", "{'authorization': '[M]'}"),
        ("Authorization:\nBasic abc123", "Authorization:\n[M]"),
    ],
)
def test_an_authorization_value_is_masked_with_its_scheme(text: str, expected: str) -> None:
    assert mask_credential_shapes(text, "[M]") == expected


def test_git_access_tokens_are_masked_plain_and_base64() -> None:
    assert mask_credential_shapes(f"credential {_BASIC} end", "[M]") == "credential [M] end"
    assert (
        mask_credential_shapes(
            f"https://x-access-token:{_INSTALLATION_TOKEN}@github.com/o/r.git", "[M]"
        )
        == "https://[M]@github.com/o/r.git"
    )


@pytest.mark.parametrize(
    "token",
    [
        _INSTALLATION_TOKEN,
        "ghp_abcdefgh12345678",
        "github_pat_11AB_cdefgh12345678",
        "sk-abcdefgh1234",
    ],
)
def test_prefixed_tokens_are_masked_whole(token: str) -> None:
    assert mask_credential_shapes(f"token {token} rejected", "[M]") == "token [M] rejected"


def test_text_without_credentials_is_unchanged() -> None:
    text = "unauthorized: authentication required; the task's checkout failed"
    assert mask_credential_shapes(text, "[M]") == text


def test_a_durable_exception_diagnostic_drops_a_command_lines_authorization_header() -> None:
    message = exception_diagnostic(_git_timeout()).message

    assert _BASIC not in message and _INSTALLATION_TOKEN not in message
    assert f"AUTHORIZATION: {REDACTED_SECRET}'" in message
    assert message.endswith("timed out after 120 seconds")


def test_inspection_masks_the_whole_authorization_value() -> None:
    shown = redact_inspection_value(str(_git_timeout()))

    assert _BASIC not in shown and "basic" not in shown
    assert "'http.extraheader=AUTHORIZATION: [REDACTED]', 'clone'" in shown


def test_docker_diagnostics_mask_an_extraheader_authorization() -> None:
    failure = docker_setup_failure(
        ["run"], 1, f"fatal: config http.extraheader=Authorization: Basic {_BASIC} rejected"
    )

    assert _BASIC not in failure and "Basic" not in failure


def test_a_registered_secret_is_removed_whole_before_shapes_are_masked() -> None:
    secret = "sk-abcdefgh!ReallySecretPassword"

    message = exception_diagnostic(
        RuntimeError(f"Login failed for {secret}"), redactor=SecretRedactor(secret)
    ).message

    assert message == f"Login failed for {REDACTED_SECRET}"


@pytest.mark.parametrize(
    "text",
    [
        "password='hunter2-unterminated",
        'password=\\"hunter2\\"',
        "password=hun'ter2secret",
        'secret=abc"def123',
    ],
)
def test_inspection_masks_awkwardly_quoted_assignments_whole(text: str) -> None:
    shown = redact_inspection_value(text)

    assert "hunter2" not in shown and "ter2secret" not in shown and "def123" not in shown


@pytest.mark.parametrize(
    "text",
    [
        'Authorization: Bearer "tok12345"',
        "Bearer\ntok12345",
        "Authorization: Basic abc'tok12345",
    ],
)
def test_quoted_and_wrapped_bearer_credentials_are_masked(text: str) -> None:
    assert "tok12345" not in mask_credential_shapes(text, "[M]")


@pytest.mark.parametrize(
    "stderr",
    [
        "HTTP_AUTHORIZATION: Basic dXNlcjpzZWNyZXQ=",
        "proxy_authorization: Bearer dXNlcjpzZWNyZXQ=",
        'Authorization: Basic "dXNlcjpzZWNyZXQ="',
    ],
)
def test_docker_diagnostics_mask_suffixed_names_and_quoted_credentials(stderr: str) -> None:
    assert "dXNlcjpzZWNyZXQ" not in docker_setup_failure(["run"], 1, stderr)
