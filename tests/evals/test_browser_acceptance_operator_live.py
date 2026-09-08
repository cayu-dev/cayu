"""Opt-in canonical handoff through a colocated protected server and real Docker."""

import ipaddress
import json
import os
import secrets
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.evals._browser_acceptance_operator_live import PRIVATE_CANARY


@pytest.mark.skipif(
    os.environ.get("CAYU_BROWSER_ACCEPTANCE_OPERATOR_LIVE") != "1",
    reason="Requires existing controller image, Docker CLI and WebSocket dependencies.",
)
def test_canonical_operator_case_through_real_docker(tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    repo = Path(__file__).resolve().parents[2]
    deps = Path(os.environ["CAYU_OPERATOR_CHANNEL_DEPS"]).resolve()
    cli = Path(os.environ["CAYU_OPERATOR_DOCKER_CLI"]).resolve()
    proof = tmp_path.resolve()
    (proof / "trust").mkdir()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "cayu-control")])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("cayu-control"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    (proof / "trust" / "control.crt").write_bytes(
        certificate.public_bytes(serialization.Encoding.PEM)
    )
    (proof / "server.key").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    (proof / "server.key").chmod(0o600)
    name = "cayu-1273-operator-" + secrets.token_hex(8)
    full_corpus = os.environ.get("CAYU_BROWSER_ACCEPTANCE_FULL_CORPUS") == "1"

    def docker(*arguments, check=True, timeout=30):
        result = subprocess.run(
            ["docker", *arguments], capture_output=True, text=True, timeout=timeout
        )
        if check and result.returncode:
            raise RuntimeError("Operator Docker fixture command failed: " + result.stderr)
        return (result.stdout + (result.stderr if arguments[0] == "logs" else "")).strip()

    try:
        identifier = docker(
            "run",
            "-d",
            "--name",
            name,
            "--user",
            "0",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=16m",
            "--mount",
            f"type=bind,src={repo},dst={repo},readonly",
            "--mount",
            f"type=bind,src={proof},dst={proof}",
            "--mount",
            f"type=bind,src={deps},dst=/deps,readonly",
            "--mount",
            f"type=bind,src={cli},dst=/usr/local/bin/docker,readonly",
            "--mount",
            "type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock",
            "--env",
            f"PYTHONPATH={repo / 'src'}:{repo}:/deps",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONPYCACHEPREFIX=/tmp/cayu-acceptance-pycache",
            "--env",
            f"CAYU_BROWSER_ACCEPTANCE_FULL_CORPUS={int(full_corpus)}",
            "--env",
            f"CAYU_OPERATOR_PROOF={proof}",
            "--env",
            f"TMPDIR={proof}",
            "--entrypoint",
            "python",
            "cayu-review-agent:local",
            str(Path(__file__).with_name("_browser_acceptance_operator_live.py").resolve()),
            "controller",
        )
        (proof / "controller-id").write_text(identifier)
        status = docker("wait", identifier, timeout=1300 if full_corpus else 210)
        logs = docker("logs", identifier)
        assert PRIVATE_CANARY not in logs
        assert status == "0", logs
        report = json.loads((proof / "report.json").read_text())
        operator = next(
            row for row in report["rows"] if row["case_id"] == "operator-private-handoff"
        )
        assert operator["diagnostic"]["operator"]["state"] == "closed"
    finally:
        # Only networks attached to this exact dedicated application container
        # are candidates, and only positively labelled egress allocations.
        raw = docker("inspect", name, check=False)
        if raw:
            networks = json.loads(raw)[0]["NetworkSettings"]["Networks"]
            for network in networks:
                detail = json.loads(docker("network", "inspect", network))[0]
                session = (detail.get("Labels") or {}).get("cayu.egress.session")
                if session:
                    for child in docker(
                        "ps", "-aq", "--filter", "label=cayu.egress.session=" + session
                    ).split():
                        docker("rm", "-f", child)
                    docker("network", "disconnect", "-f", network, name, check=False)
                    docker("network", "rm", network, check=False)
        docker("rm", "-f", name, check=False)
