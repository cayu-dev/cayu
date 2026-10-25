from __future__ import annotations

import asyncio
import json

import pytest

from cayu.egress._docker_diagnostics import docker_setup_failure, docker_stderr_tail
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
    from cayu.sessions import RunRequest
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
    # No part of the value survives, including the rest after its first letter.
    assert "unter2" not in message
    assert "unter2" not in docker_stderr_tail(stderr)
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


def test_stderr_tail_keeps_the_last_lines_after_masking_the_whole_window():
    key = "\n".join(["-----BEGIN PRIVATE KEY-----", *(["A" * 64] * 8), "-----END PRIVATE KEY-----"])
    stderr = f"first\nsecond\n{key}\nthird\n\n  fourth\twith  tabs  \n"
    redactor = SecretRedactor(["registered-probe-secret"])

    tail = docker_stderr_tail(stderr, max_lines=3, redactor=redactor)

    # The key is masked before lines are picked, so no body line survives.
    assert tail == "[REDACTED] | third | fourth with tabs"
    assert docker_stderr_tail(" \n\t", redactor=redactor) == "[unavailable]"
    assert docker_stderr_tail("x registered-probe-secret y\n", redactor=redactor) == (
        "x [REDACTED_SECRET] y"
    )
    long = docker_stderr_tail("permission denied\n" * 5000)
    assert long.startswith("...[truncated] permission denied | ")
    assert len(long.encode()) <= 1024 + len("...[truncated] ")


def test_stderr_tail_of_long_stderr_ends_with_its_last_line():
    stderr = "".join(f"pull progress line {i}\n" for i in range(5000))
    stderr += "docker: Error response from daemon: real cause\n"
    assert len(stderr) > 65536

    tail = docker_stderr_tail(stderr)

    assert tail == (
        "...[truncated] pull progress line 4996 | pull progress line 4997 | "
        "pull progress line 4998 | pull progress line 4999 | "
        "docker: Error response from daemon: real cause"
    )


def test_stderr_tail_drops_earlier_lines_to_keep_the_last_within_the_bound():
    tail = docker_stderr_tail("x" * 900 + "\n" + "y" * 600 + "\nfinal error\n")
    assert tail == "y" * 600 + " | final error"


def test_stderr_tail_cut_drops_a_private_key_whose_header_fell_before_it():
    body = "\n".join(["Q" * 64] * 40)
    key = f"-----BEGIN PRIVATE KEY-----\n{body}\n-----END PRIVATE KEY-----"
    # The window's start lands inside the key body.
    stderr = "early\n" + key + "\n" + "z" * (65536 - len(body) // 2) + "\nfinal error\n"

    tail = docker_stderr_tail(stderr, max_lines=10)

    assert "QQQQ" not in tail
    assert tail.endswith("final error")


def test_stderr_tail_cut_drops_a_registered_secret_split_by_it():
    secret = "split-canary-" + "s" * 40
    tail_text = "\nfinal error\n"
    # The window starts 20 characters into the secret.
    stderr = "x" * 100 + secret + " " * (65536 - len(secret) + 20 - len(tail_text)) + tail_text

    tail = docker_stderr_tail(stderr, redactor=SecretRedactor([secret]))

    assert "canary" not in tail and "sssss" not in tail
    assert tail == "...[truncated] final error"


@pytest.mark.parametrize(
    "begin",
    ["-----BEGIN PGP PRIVATE KEY BLOCK-----", "-----BEGIN PRIVATE KEY-----"],
)
def test_stderr_tail_masks_an_unterminated_private_key_open_at_the_window(begin):
    # The window starts inside the key body, with neither armor line in it.
    stderr = begin + "\n" + ("A" * 64 + "\n") * 1100
    assert len(stderr) > 65536

    tail = docker_stderr_tail(stderr)

    assert tail == "...[truncated] [REDACTED]"
    assert docker_stderr_tail(stderr + "late line\n") == "...[truncated] [REDACTED]"


def test_stderr_tail_keeps_lines_after_a_key_that_closed_before_the_window():
    key = "-----BEGIN PRIVATE KEY-----\n" + "Q" * 64 + "\n-----END PRIVATE KEY-----\n"
    stderr = key + "progress\n" * 8000 + "final error\n"

    tail = docker_stderr_tail(stderr, max_lines=2)

    assert tail == "...[truncated] progress | final error"


def test_stderr_tail_key_scan_is_linear_on_many_key_markers():
    import time

    keys = "-----BEGIN PRIVATE KEY-----\n-----END PRIVATE KEY-----\n" * 20000
    stderr = keys + "-----BEGIN RSA PRIVATE KEY-----\n" * 3000 + "x\n" * 40000
    elapsed = []
    for _ in range(3):
        started = time.perf_counter()
        tail = docker_stderr_tail(stderr, redactor=SecretRedactor(["s3cr3t-canary"]))
        elapsed.append(time.perf_counter() - started)
    assert tail == "...[truncated] [REDACTED]"
    assert min(elapsed) < 0.5


_CANARY_BODY = ("SYNTHETIC_KEY_CANARY_" + "A" * 44 + "\n") * 1100


@pytest.mark.parametrize(
    "header",
    [
        "-----BEGIN \x1b[31mPRIVATE KEY-----",
        "-----BEGIN \x1b[1;31mPGP PRIVATE\x1b[0m KEY BLOCK-----",
        "-----BEGIN \x9b31mRSA PRIVATE KEY-----",
        "-----BEGIN \x1b]0;title\x07PRIVATE KEY-----",
        "-----BEGIN \x1b]8;;https://example.invalid\x1b\\PRIVATE KEY-----",
    ],
)
def test_stderr_tail_masks_a_key_with_escape_formatted_armor_before_the_window(header):
    unterminated = header + "\n" + _CANARY_BODY
    assert docker_stderr_tail(unterminated) == "...[truncated] [REDACTED]"
    # The same text in a setup failure is masked too.
    assert "SYNTHETIC_KEY_CANARY" not in docker_setup_failure(["run"], 1, unterminated)

    # A key that ends inside the window, with its END line formatted as well.
    ended = unterminated + "-----END \x1b[31mPRIVATE KEY-----\n" + "final error\n"
    assert docker_stderr_tail(ended) == "...[truncated] [REDACTED] | final error"


@pytest.mark.parametrize("into_header", [1, 9, 20, 27])
def test_stderr_tail_masks_a_formatted_armor_line_across_the_window_start(into_header):
    header = "-----BEGIN \x1b[31mPRIVATE KEY-----\n"
    shown = len("-----BEGIN PRIVATE KEY-----\n")
    assert into_header < shown
    # The window holds the last 65,536 characters after the escape is
    # dropped, so it starts *into_header* characters into the BEGIN line.
    body = _CANARY_BODY[: 65536 - (shown - into_header)]
    stderr = "x" * 70000 + "\n" + header + body

    assert docker_stderr_tail(stderr) == "...[truncated] [REDACTED]"
    ended = stderr + "\n-----END PRIVATE KEY-----\nfinal error\n"
    assert docker_stderr_tail(ended) == "...[truncated] [REDACTED] | final error"


def test_stderr_tail_still_redacts_registered_secrets_after_normalizing():
    secret = "registered-tail-secret-0123456789"
    redactor = SecretRedactor([secret])
    stderr = "x" * 70000 + "\n" + f"\x1b[31mfailed with {secret}\x1b[0m\n"
    assert docker_stderr_tail(stderr, redactor=redactor) == (
        "...[truncated] failed with [REDACTED_SECRET]"
    )
    # A registered secret split by the window start leaves no fragment.
    split = "y" * 10 + secret + "z" * (65536 - len(secret) + 20) + "\nlast\n"
    tail = docker_stderr_tail(split, redactor=redactor)
    assert "tail-secret" not in tail and "0123456789" not in tail


@pytest.mark.parametrize(
    ("stderr", "leaked"),
    [
        ("\x1bpassword=hunter2-secret-value", "hunter2"),
        ("\x1bAuthorization: Bearer tok-0123456789abcdef", "0123456789"),
        ("--password\x1bhunter2 rest", "unter2"),
        ("--token\x1bxyzsecret123", "secret123"),
        ("\x1b(B\x1b[m password: hunter2-secret\n", "hunter2"),
        ("\x1b(B--password\x1b(B hunter2", "unter2"),
        ("\x1b(Authorization: Bearer tok-0123456789abcdef", "0123456789"),
        ("\x1b(Bearer tok-0123456789abcdef\nAuthorization: \x1b(Bearer tok-1", "tok-1"),
        (
            "-----BEGIN \x1b(BPRIVATE KEY-----\nSYNTHETIC_PRIVATE_KEY_BODY\n-----END PRIVATE KEY-----",
            "SYNTHETIC_PRIVATE_KEY_BODY",
        ),
        ("--password \x1b]0;title\x07 hunter2", "unter2"),
        ("TOKEN=\x1b]8;;https://example.invalid\x1b\\secret-value", "secret-value"),
        (
            "\x1b-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANSECRETBODY\n-----END PRIVATE KEY-----",
            "SECRETBODY",
        ),
    ],
)
def test_escape_bytes_never_join_or_hide_a_credential(stderr, leaked):
    # A lone ESC becomes a space; only CSI and OSC sequences are dropped whole.
    assert leaked not in docker_setup_failure(["run"], 1, stderr)
    assert leaked not in docker_stderr_tail(stderr)
    long = "x" * 70000 + "\n" + stderr + "\n"
    assert leaked not in docker_stderr_tail(long)
