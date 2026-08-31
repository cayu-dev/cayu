#!/usr/bin/env bash
set -euo pipefail

repository_root="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must name the checked-out repository}"
proof_root="$(mktemp -d "$RUNNER_TEMP/sidecar.XXXXXX")"
cd "$proof_root"
unset PYTHONPATH

"$repository_root/.release-venv/bin/cayu" \
  lambda-microvm sidecar export "$proof_root/wheel"
"$repository_root/.release-sdist-venv/bin/cayu" \
  lambda-microvm sidecar export "$proof_root/sdist"
diff -ru "$proof_root/wheel" "$proof_root/sdist"

"$repository_root/.release-venv/bin/python" - <<'PY'
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from cayu.runners.aws_lambda_microvm import LAMBDA_MICROVM_PROTOCOL_VERSION

root = Path("wheel")
manifest = json.loads(
    (root / "cayu-lambda-microvm-sidecar-manifest.json").read_text()
)
assert manifest["schema_version"] == 1
assert manifest["artifact_version"] == 1
assert manifest["cayu_version"] == importlib.metadata.version("cayu")
assert manifest["protocol_version"] == LAMBDA_MICROVM_PROTOCOL_VERSION
expected_paths = {
    "cayu-lambda-microvm-sidecar-manifest.json",
    *(item["path"] for item in manifest["files"]),
}
assert {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file()
} == expected_paths
for item in manifest["files"]:
    content = (root / item["path"]).read_bytes()
    assert len(content) == item["size"]
    assert "sha256:" + hashlib.sha256(content).hexdigest() == item["sha256"]
canonical_files = json.dumps(
    manifest["files"],
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode()
assert (
    "sha256:" + hashlib.sha256(canonical_files).hexdigest()
    == manifest["content_digest"]
)
assert importlib.util.find_spec("boto3") is None
assert importlib.util.find_spec("fastapi") is None
PY

docker build --platform linux/arm64 \
  -t cayu-lambda-microvm-sidecar-proof "$proof_root/wheel"
docker run --rm --platform linux/arm64 --entrypoint python3.11 \
  cayu-lambda-microvm-sidecar-proof -c \
  'import asyncio, json; from pathlib import Path; from lambda_microvm_sidecar.app import health; manifest = json.loads(Path("/opt/cayu/lambda_microvm_sidecar/cayu-lambda-microvm-sidecar-manifest.json").read_text()); result = asyncio.run(health()); assert result == {"status": "ok", "protocol_version": manifest["protocol_version"]}'

"$repository_root/.release-sdist-venv/bin/python" - <<'PY'
from importlib.resources import files

resource = files("cayu.data").joinpath("lambda_microvm_sidecar", "app.py")
resource.write_bytes(resource.read_bytes() + b"\n# tampered\n")
PY
if "$repository_root/.release-sdist-venv/bin/cayu" \
  lambda-microvm sidecar export "$proof_root/tampered"; then
  echo "tampered installed sidecar unexpectedly exported"
  exit 1
fi
