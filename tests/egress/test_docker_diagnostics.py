from __future__ import annotations

import asyncio
import json

import pytest

from cayu.egress._docker_diagnostics import docker_setup_failure
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.egress.errors import UnsupportedEgressError
from cayu.vaults import SecretRedactor


@pytest.mark.parametrize(
    "stderr,expected",
    [
        ("", "[unavailable]"),
        (" \n\t", "[unavailable]"),
        ("daemon said something new", "daemon said something new"),
        ("permission denied " * 500, "...[truncated]"),
        ("x" * 70000, "...[truncated]"),
    ],
)
def test_stderr_is_shown_bounded_and_explicit(stderr, expected):
    message = docker_setup_failure(["network", "create", "private-name"], 17, stderr)
    assert expected in message
    assert "exit_code=17" in message
    assert len(message.split("stderr: ", 1)[1].encode()) <= 1024 + len("...[truncated]")
    # Command arguments are never copied into the message.
    assert "private-name" not in message


@pytest.mark.parametrize("secret_start", [0, 1000, 1010, 1020])
def test_known_secrets_are_removed_before_bounding(secret_start):
    secret = "transport-token-ABCDEFGHIJKLMNOP"
    stderr = "x" * secret_start + secret + " tail"
    message = docker_setup_failure(
        ["network", "create"], 1, stderr, redactor=SecretRedactor([secret])
    )
    assert not any(secret[:size] in message for size in range(8, len(secret) + 1))


def test_credentials_in_stderr_and_argv_never_escape():
    stderr = (
        "Error response from daemon: permission denied\n"
        "https://username-canary:password-canary@example.com?token=token-canary\n"
        "Authorization: Bearer token-canary\n"
        "ENV=environment-canary /path-canary\n"
        "-----BEGIN PRIVATE KEY-----\nmount-canary\n-----END PRIVATE KEY-----\n"
        "opaque-canary argument-canary\x1b[31m\x00"
    )
    message = docker_setup_failure(
        ["exec", "argument-canary", "cat", "/path-canary"],
        1,
        stderr,
        redactor=SecretRedactor(["opaque-canary"]),
    )
    assert "permission denied" in message
    assert "[REDACTED]" in message
    # Registered secrets and credential-shaped values are removed; plain text
    # such as names and paths is shown (#1974).
    for value in ("password-canary", "token-canary", "mount-canary", "opaque-canary"):
        assert value not in message
    assert "username-canary" not in message
    assert "/path-canary" in message
    assert "\x1b" not in message and "\x00" not in message and "\n" not in message
    assert "argument-canary" not in docker_setup_failure(["argument-canary"], 1, "")
    assert "argument-canary" not in docker_setup_failure(["network", "argument-canary"], 1, "")


@pytest.mark.parametrize(
    "operation,stderr",
    [
        (
            "create",
            "Error response from daemon: all predefined address pools have been fully subnetted",
        ),
        (
            "connect",
            "Error response from daemon: endpoint with name private-name already exists in network private-network",
        ),
        (
            "probe",
            "cayu-broker-gateway=172.30.0.1\nconnect(172.30.0.1): Network unreachable",
        ),
    ],
)
def test_setup_failure_survives_sqlite_environment_and_session_evidence(
    tmp_path, monkeypatch, operation, stderr
):
    from cayu.agents import AgentSpec
    from cayu.applications import CayuApp
    from cayu.egress.policy import PublicWebEgressPolicy
    from cayu.egress.runtime import VirtualEgressEnvironmentFactory
    from cayu.environments import EnvironmentSpec
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.events import EventType
    from cayu.messages import Message
    from cayu.runtime import RunRequest
    from cayu.storage.sqlite import SQLiteSessionStore

    calls = []

    async def execute(argv):
        calls.append(list(argv))
        if list(argv[:2]) == ["network", operation] or (
            operation == "probe" and argv[-1] == "probe"
        ):
            return 1, stderr + " AUTH_TOKEN=secret-canary"
        return 0, ""

    async def start(_self):
        return 8123

    monkeypatch.setattr("cayu.egress.docker_adapter.TransparentEgressProxyServer.start", start)

    async def scenario():
        path = tmp_path / "sessions.db"
        store = SQLiteSessionStore(path)
        adapter = DockerEgressAdapter(docker_exec=execute, proxy_host="127.0.0.1")
        factory = VirtualEgressEnvironmentFactory(
            policies={"web": PublicWebEgressPolicy(name="web")},
            public_web_policy="web",
            adapter=adapter,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider([]), default=True)
        app.register_environment_factory(EnvironmentSpec(name="docker"), factory, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake"))
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="docker-failure",
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        await store.close()
        reopened = SQLiteSessionStore(path)
        persisted = await reopened.load_events("docker-failure")
        await reopened.close()
        for kind in (EventType.ENVIRONMENT_FACTORY_FAILED, EventType.SESSION_FAILED):
            live = next(event for event in events if event.type == kind)
            durable = next(event for event in persisted if event.type == kind)
            assert live.payload["error"] == durable.payload["error"]
            payload = json.dumps(durable.payload)
            if operation == "probe":
                assert "egress_sidecar_unreachable" in payload
                assert "172.30.0.1" in payload
                assert "Network unreachable" in payload
            else:
                assert f"docker network {operation}" in payload
                assert (
                    "fully subnetted" if operation == "create" else "already exists in network"
                ) in payload
            assert "exit_code=1" in payload
            assert "secret-canary" not in payload
        assert any(argv[:2] == ["network", "rm"] for argv in calls)
        assert any(argv[0] == "rm" for argv in calls)
        assert not adapter._preparation_cleanups

    asyncio.run(scenario())


def test_setup_exception_contains_safe_diagnostic():
    async def execute(_argv):
        return 9, "Pool overlaps with other one on this address space; TOKEN=secret"

    adapter = DockerEgressAdapter(docker_exec=execute)
    with pytest.raises(UnsupportedEgressError, match="network create.*exit_code=9") as raised:
        asyncio.run(adapter._run(["network", "create", "private-network"]))
    assert "Pool overlaps" in str(raised.value)
    assert "secret" not in str(raised.value)


def test_each_preparation_redacts_only_its_own_transport_token(monkeypatch):
    import cayu.egress.docker_adapter as adapter_module
    from cayu.egress import TransparentEgressBroker, VirtualCredentialRegistry
    from cayu.vaults import StaticVault

    seen: list[tuple[str, ...]] = []
    real_failure = adapter_module.docker_setup_failure

    def recording_failure(argv, exit_code, stderr, *, redactor=None):
        seen.append(() if redactor is None else redactor._values)
        return real_failure(argv, exit_code, stderr, redactor=redactor)

    async def execute(argv):
        if list(argv[:2]) == ["network", "create"]:
            return 1, "network create failed"
        return 0, ""

    async def start(_self):
        return 8123

    monkeypatch.setattr(adapter_module, "docker_setup_failure", recording_failure)
    monkeypatch.setattr(adapter_module.TransparentEgressProxyServer, "start", start)

    async def scenario():
        adapter = DockerEgressAdapter(docker_exec=execute, proxy_host="127.0.0.1")
        for session in ("first", "second", "third"):
            broker = TransparentEgressBroker(
                registry=VirtualCredentialRegistry(), resolver=StaticVault({}), policies={}
            )
            with pytest.raises(UnsupportedEgressError):
                await adapter.prepare(session_id=session, broker=broker, grants=())

    asyncio.run(scenario())

    # One token per preparation, never the tokens of earlier preparations.
    assert len(seen) == 3
    assert all(len(values) == 1 for values in seen)
    assert len({values[0] for values in seen}) == 3


@pytest.mark.parametrize(
    "stderr,kept",
    [
        ('{"Authorization": "Bearer json-canary", "user": "bob"}', '"user": "bob"'),
        ('{"password":"json-canary","registry":"r.example"}', '"registry":"r.example"'),
        ('"api_key": "escaped\\"json-canary"', '"api_key"'),
        ("X-Api-Key: header-canary for registry", "for registry"),
        ("Private-Token: header-canary", "Private-Token:"),
        ("Cookie: a=cookie-canary; b=json-canary", "Cookie:"),
        ("docker login --password flag-canary registry", "registry"),
        ("docker login --token=flag-canary registry", "registry"),
        ("docker login -u bob -p flag-canary registry.example", "registry.example"),
    ],
)
def test_common_credential_shapes_are_masked(stderr, kept):
    message = docker_setup_failure(["run"], 1, stderr)
    assert "canary" not in message
    assert kept in message


def test_non_credential_flags_are_kept():
    message = docker_setup_failure(["run"], 1, "docker run -p 8080:80 --password-stdin image")
    assert "-p 8080:80 --password-stdin image" in message


@pytest.mark.parametrize("offset", [1, 8, 20, 31])
def test_secret_cut_at_the_scan_limit_never_appears(offset):
    # Whitespace collapses after the cut, so the scan's tail lands inside the
    # 1 KiB output; a fragment of a secret cut at the limit must not survive.
    secret = "transport-token-ABCDEFGHIJKLMNOP"
    scan_limit = 65536
    stderr = " " * (scan_limit - offset) + secret + " tail"
    message = docker_setup_failure(["run"], 1, stderr, redactor=SecretRedactor([secret]))
    assert not any(secret[:size] in message for size in range(4, len(secret) + 1))
    unregistered = " " * (scan_limit - offset) + "https://user:pass-canary@host"
    assert "pass-can" not in docker_setup_failure(["run"], 1, unregistered)


@pytest.mark.parametrize(
    "stderr,kept",
    [
        ('docker login --password "quoted canary" registry', "registry"),
        ("docker login --password 'quoted canary' registry", "registry"),
        ('docker login --password="quoted canary" registry', "registry"),
        ("docker login -u bob -pshort-canary registry", "-u bob"),
        ("docker login -u bob -p=short-canary registry", "-u bob"),
        ('docker login -u bob -p "quoted canary" registry', "registry"),
    ],
)
def test_quoted_and_attached_flag_values_are_masked(stderr, kept):
    message = docker_setup_failure(["run"], 1, stderr)
    assert "canary" not in message
    assert kept in message


@pytest.mark.parametrize(
    "stderr",
    [
        "Error: login failed; docker run -p 8080:80 nginx",
        "pull access denied, may require 'docker login' -p 8080:80",
        "docker login registry && docker run -p 8080:80 nginx",
        "docker login registry\ndocker run -p 8080:80 nginx",
        "login: docker run -p 8080:80 nginx",
    ],
)
def test_port_flags_outside_docker_login_are_kept(stderr):
    assert "-p 8080:80" in docker_setup_failure(["run"], 1, stderr)


_ADVERSARIAL_STDERR = {
    "name run": "a-" * 32000,
    "credential words": "key" * 21000,
    "login words": "login " * 10900,
    "docker login": "docker login " * 5000,
    "docker login -p": "docker login -p " * 4000,
    "quotes": '"' * 65000,
    "quoted keys": '"key":' * 10900,
    "unterminated values": '"password": "' * 5000,
    "password flags": "--password " * 5900,
    "assignments": "x=" * 32000,
    "url userinfo": "://a:" * 13000,
    "authorization": "Authorization: " * 4300,
    "pem": "-----BEGIN " * 5900,
    "separators": ";" * 65000,
    "unclosed quoted values": '--password "a ' * 5000,
    "closed quoted values": 'docker login -p "a b" ' * 2900,
    "global options": "docker --config x " * 3600,
    "escaped json keys": '\\"a\\":\\"' * 7000,
    "python repr keys": "'a': '" * 10000,
    "mapping lines": "a: b\n" * 13000,
    "credential lines": "password: x\n" * 5400,
    "assignment chains": "--env=" * 10800,
    "quoted credential values": "PASSWORD=a'b " * 5000,
    "separated words": "x|" * 32000,
    "login segments": "x;login -p " * 5900,
    "unterminated ansi": "\x1b[1" * 21000,
    "ansi sequences": "\x1b[1;31m--password\x1b[0m " * 2000,
    "controls": "\x00\u200b" * 32000,
}


@pytest.mark.parametrize("name", sorted(_ADVERSARIAL_STDERR))
def test_credential_masking_is_linear_on_adversarial_stderr(name):
    import time

    stderr = _ADVERSARIAL_STDERR[name]
    elapsed = []
    for _ in range(3):
        started = time.perf_counter()
        docker_setup_failure(["run"], 1, stderr, redactor=SecretRedactor(["s3cr3t-canary"]))
        elapsed.append(time.perf_counter() - started)
    # Quadratic patterns took seconds to minutes on these inputs.
    assert min(elapsed) < 0.2


def test_scan_cut_inside_a_quoted_credential_value_never_shows_it():
    stderr = " " * (65536 - 25) + '"password": "frag-canary more" tail'
    message = docker_setup_failure(["run"], 1, stderr)
    assert "frag" not in message


@pytest.mark.parametrize(
    "stderr",
    [
        "Error: can't connect; docker login -p hunter2 registry",
        "Error response from daemon: container doesn't exist --password hunter2",
        "it's broken: --token hunter2",
        'msg "unterminated --password hunter2',
        "docker login -p hun;ter2 registry",
        "docker login -p hun|ter2&x registry",
        "docker --config /tmp/c login -p hunter2",
        "docker --config=/tmp/c -H unix:///run/docker.sock login -p hunter2",
        "/usr/bin/docker login -p hunter2",
        "podman login -p hunter2 registry",
        "login -p hunter2 registry",
        "Error: failed; login -phunter2 registry",
    ],
)
def test_credential_flags_are_masked_around_stray_quotes_and_cli_forms(stderr):
    message = docker_setup_failure(["run"], 1, stderr)
    assert "hunter2" not in message
    assert "hun" not in message.split("stderr: ", 1)[1].replace("[REDACTED]", "")


@pytest.mark.parametrize(
    "stderr",
    [
        "docker run -p 8080:80 nginx",
        "/usr/bin/docker --config /tmp/c run -p 8080:80 nginx",
        "docker login -p hunter2; docker run -p 8080:80 nginx",
        "podman run -p 8080:80 nginx",
    ],
)
def test_port_flags_after_other_subcommands_are_kept(stderr):
    message = docker_setup_failure(["run"], 1, stderr)
    assert "-p 8080:80" in message
    assert "hunter2" not in message


@pytest.mark.parametrize(
    "stderr",
    [
        # Separators leading or inside a word still start a new command.
        "x|docker login -p hunter2",
        "x |docker login -p hunter2",
        "x&&docker login -p hunter2",
        "x;login -p hunter2",
        "x||podman login -p hunter2",
        # An unclosed quote masks to the end of the line.
        '--password "a b hunter2',
        '--token="a b hunter2',
        # A quote opening partway into the value is followed.
        "--password a'b c'hunter2",
        '--password=a"b hunter2"',
        # Every credential-named NAME= in a word, with its whole value run.
        "--env=PASSWORD=hunter2",
        "-e=DB_TOKEN=hunter2",
        "--build-arg=GITHUB_TOKEN=hunter2",
        "env=AWS_SECRET_ACCESS_KEY=hunter2",
        "PASSWORD=abc,hunter2",
        'TOKEN="aa hunter2"',
        "DB_PASS=hunter2",
        "PW=hunter2",
        "mysql pwd=hunter2",
        # Header values after any non-newline whitespace.
        "Authorization: Bearer\rhunter2",
        "Authorization: Bearer\u2003hunter2",
        "Authorization: Bearer\x0bhunter2",
        "X-Api-Key:\rhunter2",
        # Other shapes.
        "https://ghp_hunter2@github.com/org/repo",
        "{'Authorization': 'Bearer hunter2'}",
        "{'password': 'hunter2'}",
        "password: hunter2",
        "  db_password: hunter2",
        '{\\"password\\":\\"hunter2\\"}',
        "-----BEGIN PGP PRIVATE KEY BLOCK-----\nhunter2\n-----END PGP PRIVATE KEY BLOCK-----",
    ],
)
def test_more_credential_shapes_are_masked(stderr):
    message = docker_setup_failure(["run"], 1, stderr)
    assert "hunter2" not in message


@pytest.mark.parametrize(
    "stderr,kept",
    [
        ("unauthorized: authentication required", "authentication required"),
        ("docker run -p 8080:80 nginx", "-p 8080:80"),
        ("docker create -p 8080:80 nginx", "-p 8080:80"),
        ("x|docker run -p 8080:80 nginx", "-p 8080:80"),
        ("passport=ok bypass=ok", "passport=ok bypass=ok"),
        ("X-Api-Key: hunter2 for registry", "for registry"),
    ],
)
def test_ordinary_text_next_to_credential_shapes_is_kept(stderr, kept):
    message = docker_setup_failure(["run"], 1, stderr)
    assert kept in message
    assert "hunter2" not in message


@pytest.mark.parametrize(
    "stderr",
    [
        "--password\x00hunter2",
        "\x1b[1m--password\x1b[0m hunter2",
        "--password\x07hunter2",
        "--password\x7fhunter2",
        "--password\x1bhunter2",
        "password:​hunter2",
        "password:﻿hunter2",
        "--token‌hunter2",
        "--token‍hunter2",
        "--token⁠hunter2",
        "docker login -p\x00hunter2",
    ],
)
def test_invisible_characters_do_not_hide_credentials(stderr):
    message = docker_setup_failure(["run"], 1, stderr)
    assert "hunter2" not in message
    assert "\x1b" not in message and "​" not in message


@pytest.mark.parametrize(
    "secret",
    ["first-canary-line\nsecond-canary-line", "first-canary\r\nsecond", "canary  two \t words"],
)
def test_whitespace_secrets_are_redacted_before_normalization(secret):
    message = docker_setup_failure(
        ["run"], 1, f"failed: {secret} end", redactor=SecretRedactor([secret])
    )
    assert "canary" not in message
    assert "failed:" in message and "end" in message


def test_scan_cut_never_splits_a_registered_multi_word_secret():
    # Backing up to whitespace at the scan limit used to cut inside a secret
    # containing spaces and show its first words.
    secret = "alpha-canary beta-canary gamma-canary"
    # The secret ends inside the scan window but straddles the trimmed margin;
    # the whitespace before it collapses, so its head would reach the output.
    stderr = " " * 65480 + secret + " tail" + " " * 200
    message = docker_setup_failure(["run"], 1, stderr, redactor=SecretRedactor([secret]))
    assert "canary" not in message


def test_scan_cut_drops_a_nested_secret_split_by_the_cut():
    outer = (
        "user=dbadmin_account host=db.internal.example "
        "password=Sup3rS3cretPW_value sslmode=require region=eu-west"
    )
    inner = "Sup3rS3cretPW_value"
    filler = "L" * (65536 - len(outer) + 9)
    stderr = filler + " " + outer + " more"
    message = docker_setup_failure(
        ["run"], 1, stderr, redactor=SecretRedactor([outer, inner, filler])
    )
    assert "dbadmin" not in message and "db.internal" not in message


def test_scan_cut_drops_a_multi_line_credential_blob_split_by_the_cut():
    blob = (
        '{\n  "type": "service_account",\n'
        '  "client_email": "svc-canary@proj.iam.gserviceaccount.com",\n'
        '  "client_secret": "cs-canary-0123456789",\n'
        '  "token_uri": "https://oauth2.example/token"\n}'
    )
    # The cut lands after the registered client_secret but inside the blob.
    stderr = " " * (65536 - len(blob) + 10) + blob + "\nmore"
    message = docker_setup_failure(
        ["run"], 1, stderr, redactor=SecretRedactor([blob, "cs-canary-0123456789"])
    )
    assert "canary" not in message and "service_account" not in message
