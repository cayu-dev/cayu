"""Guest audit program executed with ``python3 -c`` inside a Lambda MicroVM."""

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import pathlib
import socket
import ssl
import sys
from urllib.parse import urlparse

fingerprint_key = base64.b64decode(sys.argv[1], validate=True)
if len(fingerprint_key) != 32:
    raise SystemExit("candidate fingerprint key must be 32 bytes")
mode = sys.argv[2]
MAX_CANDIDATE_FINGERPRINTS = 8192


def candidate_values(data):
    candidates = {data, data.strip().strip(b"\"'")}
    normalized = data.replace(b"\0", b"\n").replace(b"\r", b"\n")
    for record in normalized.split(b"\n"):
        record = record.strip()
        if not record:
            continue
        candidates.add(record.strip(b"\"'"))
        for separator in (b"=", b":"):
            if separator in record:
                candidates.add(record.split(separator, 1)[1].strip().strip(b"\"'"))
    return candidates


def fingerprint_candidates(sources):
    fingerprints = set()
    for source in sources:
        for candidate in candidate_values(source):
            if not candidate:
                continue
            fingerprint = hmac.new(fingerprint_key, candidate, hashlib.sha256).hexdigest()
            if fingerprint in fingerprints:
                continue
            if len(fingerprints) >= MAX_CANDIDATE_FINGERPRINTS:
                return sorted(fingerprints), True
            fingerprints.add(fingerprint)
    return sorted(fingerprints), False


def read_bounded(path, limit=262144):
    try:
        with open(path, "rb") as handle:
            return handle.read(limit)
    except OSError:
        return b""


def read_init_network_namespace():
    try:
        return os.readlink("/proc/1/ns/net"), "readable"
    except PermissionError:
        return None, "permission-denied"
    except FileNotFoundError:
        return None, "missing"
    except OSError:
        return None, "os-error"


def process_status():
    fields = {}
    for line in read_bounded("/proc/self/status").decode("utf-8", errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key] = value.strip()
    return fields


def network_routes():
    routes = set()
    text = read_bounded("/proc/net/route").decode("ascii", errors="replace")
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 8:
            continue
        try:
            destination = socket.inet_ntoa(bytes.fromhex(fields[1])[::-1])
            mask = socket.inet_ntoa(bytes.fromhex(fields[7])[::-1])
            routes.add(str(ipaddress.ip_network(f"{destination}/{mask}", strict=False)))
        except (OSError, ValueError):
            continue
    return sorted(routes)


def response_status(head):
    try:
        return int(head.split(b" ", 2)[1])
    except (IndexError, ValueError):
        return None


def metadata_request(method, path, headers=None):
    try:
        sock = socket.create_connection(("169.254.169.254", 80), timeout=2)
    except OSError:
        return False, None, b""
    lines = [f"{method} {path} HTTP/1.1", "Host: 169.254.169.254", "Connection: close"]
    for key, value in (headers or {}).items():
        lines.append(f"{key}: {value}")
    response = b""
    try:
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        while len(response) < 65536:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
    except OSError:
        pass
    finally:
        sock.close()
    head, _, body = response.partition(b"\r\n\r\n")
    return True, response_status(head), body


def proxy_request():
    proxy = urlparse(os.environ["HTTPS_PROXY"])
    sock = socket.create_connection((proxy.hostname, proxy.port), timeout=5)
    sock.sendall(b"CONNECT receiver.internal:443 HTTP/1.1\r\nHost: receiver.internal:443\r\n\r\n")
    head = b""
    while b"\r\n\r\n" not in head and len(head) < 8192:
        head += sock.recv(1024)
    if response_status(head) != 200:
        return response_status(head)
    context = ssl.create_default_context(cafile="/etc/cayu/ca.pem")
    tls = context.wrap_socket(sock, server_hostname="receiver.internal")
    body = b'{"action":"metadata-isolation-live"}'
    token = os.environ.get("INTERNAL_SERVICE_TOKEN", "")
    request = (
        b"POST /v1/actions HTTP/1.1\r\n"
        b"Host: receiver.internal\r\n"
        + f"Authorization: Bearer {token}\r\n".encode()
        + b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"Connection: close\r\n\r\n"
        + body
    )
    tls.sendall(request)
    response = b""
    while b"\r\n\r\n" not in response and len(response) < 8192:
        chunk = tls.recv(1024)
        if not chunk:
            break
        response += chunk
    tls.close()
    return response_status(response)


if mode == "revoked":
    print(json.dumps({"proxy_status": proxy_request()}, sort_keys=True))
    raise SystemExit(0)


status = process_status()
network_namespace = os.readlink("/proc/self/ns/net")
init_network_namespace, init_network_namespace_access = read_init_network_namespace()


process_environment_paths = []
process_command_paths = []
for process_dir in pathlib.Path("/proc").glob("[0-9]*"):
    process_environment_paths.append(process_dir / "environ")
    process_command_paths.append(process_dir / "cmdline")

credential_paths = {
    pathlib.Path.home() / ".aws" / "credentials",
    pathlib.Path.home() / ".aws" / "config",
    pathlib.Path("/root/.aws/credentials"),
    pathlib.Path("/root/.aws/config"),
    pathlib.Path("/etc/aws/credentials"),
    pathlib.Path("/etc/aws/config"),
    pathlib.Path("/var/run/secrets/eks.amazonaws.com/serviceaccount/token"),
}
for name in ("AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE", "AWS_WEB_IDENTITY_TOKEN_FILE"):
    value = os.environ.get(name)
    if value:
        credential_paths.add(pathlib.Path(value))

environment_source = b"\0".join(f"{key}={value}".encode() for key, value in os.environ.items())
process_environment_sources = [read_bounded(path) for path in process_environment_paths]
process_command_sources = [read_bounded(path) for path in process_command_paths]
credential_file_sources = [read_bounded(path) for path in credential_paths]
sources = [
    environment_source,
    *process_environment_sources,
    *process_command_sources,
    *credential_file_sources,
]

filesystem_files = []
for root in ("/workspace", "/tmp", "/home", "/root/.aws", "/etc/aws", "/var/run/secrets"):
    root_path = pathlib.Path(root)
    try:
        if not root_path.exists():
            continue
        if root_path.is_file():
            filesystem_files.append(root_path)
            continue
    except OSError:
        continue
    try:
        for path in root_path.rglob("*"):
            if len(filesystem_files) >= 128:
                break
            if path.is_file() and not path.is_symlink() and path.stat().st_size <= 65536:
                filesystem_files.append(path)
    except OSError:
        continue
filesystem_sources = []
remaining_filesystem_bytes = 2 * 1024 * 1024
for path in filesystem_files:
    if remaining_filesystem_bytes <= 0:
        break
    data = read_bounded(path, min(65536, remaining_filesystem_bytes))
    filesystem_sources.append(data)
    remaining_filesystem_bytes -= len(data)
sources.extend(filesystem_sources)
candidate_fingerprints, candidate_fingerprint_overflow = fingerprint_candidates(sources)

token_reachable, _, token_body = metadata_request(
    "PUT",
    "/latest/api/token",
    {"X-aws-ec2-metadata-token-ttl-seconds": "60"},
)
metadata_reachable = token_reachable
metadata_bodies = [token_body]
headers = {}
if token_body:
    headers["X-aws-ec2-metadata-token"] = token_body.decode("utf-8", errors="ignore").strip()
roles_reachable, roles_status, roles_body = metadata_request(
    "GET", "/latest/meta-data/iam/security-credentials/", headers
)
metadata_reachable = metadata_reachable or roles_reachable
metadata_bodies.append(roles_body)
if roles_status is not None and 200 <= roles_status < 300 and roles_body.strip():
    role = roles_body.decode("utf-8", errors="ignore").strip().splitlines()[0]
    credentials_reachable, _, credentials_body = metadata_request(
        "GET", "/latest/meta-data/iam/security-credentials/" + role, headers
    )
    metadata_reachable = metadata_reachable or credentials_reachable
    metadata_bodies.append(credentials_body)

aws_env_names = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
}
aws_markers = (
    b"accesskeyid",
    b"secretaccesskey",
    b"aws_access_key_id",
    b"aws_secret_access_key",
    b"aws_session_token",
)
credential_material_sources = [
    environment_source,
    *process_environment_sources,
    *credential_file_sources,
]
aws_credentials_present = any(os.environ.get(name) for name in aws_env_names) or any(
    marker in source.lower() for marker in aws_markers for source in credential_material_sources
)
metadata_credentials_present = any(
    marker in body.lower() for marker in aws_markers for body in metadata_bodies
)
try:
    direct = socket.create_connection(("1.1.1.1", 443), timeout=2)
except OSError:
    direct_public_reachable = False
else:
    direct.close()
    direct_public_reachable = True
try:
    sidecar = socket.create_connection(("192.0.2.1", 8080), timeout=2)
except OSError:
    sidecar_api_reachable = False
else:
    sidecar.close()
    sidecar_api_reachable = True

virtual_credentials = [
    (name, value)
    for name, value in os.environ.items()
    if value.startswith("cayu_vc_") or value.startswith("sk_test_cayu_vc_")
]
expected_virtual_credential = [
    item for item in virtual_credentials if item[0] == "INTERNAL_SERVICE_TOKEN"
]

print(
    json.dumps(
        {
            "aws_credentials_present": bool(aws_credentials_present),
            "candidate_fingerprint_overflow": candidate_fingerprint_overflow,
            "candidate_fingerprints": candidate_fingerprints,
            "cap_ambient": int(status.get("CapAmb", "-1"), 16),
            "cap_bounding": int(status.get("CapBnd", "-1"), 16),
            "cap_effective": int(status.get("CapEff", "-1"), 16),
            "cap_inheritable": int(status.get("CapInh", "-1"), 16),
            "credential_paths_checked": len(credential_paths),
            "direct_public_reachable": direct_public_reachable,
            "effective_gid": os.getegid(),
            "effective_uid": os.geteuid(),
            "filesystem_files_inspected": len(filesystem_sources),
            "init_network_namespace": init_network_namespace,
            "init_network_namespace_access": init_network_namespace_access,
            "metadata_credentials_present": metadata_credentials_present,
            "metadata_network_reachable": metadata_reachable,
            "network_namespace": network_namespace,
            "network_routes": network_routes(),
            "no_new_privs": status.get("NoNewPrivs") == "1",
            "processes_inspected": len(process_environment_paths),
            "proxy_status": proxy_request(),
            "sidecar_api_reachable": sidecar_api_reachable,
            "unexpected_virtual_credentials": len(virtual_credentials) != 1,
            "virtual_credential_count": len(virtual_credentials),
            "virtual_credential_present": len(expected_virtual_credential) == 1,
        },
        sort_keys=True,
    )
)
