"""`cayu cloud deployment sbom`: download the SBOM a Release's image build recorded, verify it,
and save it to a new file."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from cayu.cli import _cloud_api, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient

_APP = "/v1/applications/my-agent"
_SBOM_ROUTE = ("GET", f"{_APP}/deployments/dep_live/sbom")
_AGENTS = {"items": [{"id": "my-agent", "name": "My Agent", "current_deployment_id": "dep_live"}]}
_RELEASE = {"id": "dep_live", "application_id": "my-agent", "status": "failed"}
_SBOM = json.dumps(
    {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "components": [
            {"type": "library", "name": "pypdf", "version": "6.21.2"},
            {"type": "library", "name": "httpx", "version": "0.28.1"},
        ],
    }
).encode()


def _served(body: bytes, *, digest: str | None = None) -> httpx.Response:
    headers = {
        "Content-Type": "application/vnd.cyclonedx+json",
        "Content-Disposition": 'attachment; filename="dep_live.cdx.json"',
        "X-Cayu-Content-Digest": (
            digest if digest is not None else "sha256:" + hashlib.sha256(body).hexdigest()
        ),
    }
    return httpx.Response(200, headers=headers, content=body)


def _refused(http_status: int, code: str, message: str, **extra: str) -> httpx.Response:
    return httpx.Response(http_status, json={"detail": {"code": code, "message": message, **extra}})


class _Cloud:
    def __init__(self, routes: dict[tuple[str, str], httpx.Response]) -> None:
        self.routes = {
            ("GET", "/v1/applications?lifecycle=all"): httpx.Response(200, json=_AGENTS),
            ("GET", "/v1/deployments/dep_live"): httpx.Response(200, json=_RELEASE),
            **routes,
        }
        self.requests: list[str] = []
        self.accept: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.raw_path.decode())
        self.requests.append(" ".join(key))
        self.accept.append(request.headers.get("Accept", ""))
        return self.routes.get(key) or httpx.Response(404, json={"detail": "Not Found"})


@pytest.fixture
def cloud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[dict[tuple[str, str], httpx.Response]], _Cloud]:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)

    def install(routes: dict[tuple[str, str], httpx.Response]) -> _Cloud:
        fake = _Cloud(routes)
        real_client = httpx.Client
        monkeypatch.setattr(
            _cloud_api.httpx,
            "Client",
            lambda **kwargs: real_client(transport=httpx.MockTransport(fake), **kwargs),
        )
        monkeypatch.setattr(
            cloud_cli,
            "_cloud_client",
            lambda *_args, **_kwargs: CloudApiClient(
                api_url="https://cloud.example", api_key="login-token"
            ),
        )
        return fake

    return install


def _error(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)["error"]


def _leftovers(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if path.name.startswith(".cayu-"))


def test_saves_the_verified_sbom_to_the_named_file(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({_SBOM_ROUTE: _served(_SBOM)})
    output = tmp_path / "evidence" / "sbom.cdx.json"

    assert main(["cloud", "deployment", "sbom", "dep_live", "--output", str(output)]) == 0

    assert output.read_bytes() == _SBOM
    out = capsys.readouterr().out
    assert (
        f"Wrote the SBOM of release dep_live to {output} ({len(_SBOM)} bytes, 2 components, "
        "CycloneDX 1.5)." in out
    )
    assert f"Verified digest sha256:{hashlib.sha256(_SBOM).hexdigest()}." in out
    assert _leftovers(output.parent) == []
    # The Agent came from the release id alone, and the download asked for CycloneDX.
    assert "GET /v1/applications?lifecycle=all" not in fake.requests
    assert fake.accept[-1].startswith("application/vnd.cyclonedx+json")


def test_json_reports_the_metadata_and_the_default_name_is_the_release(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloud({_SBOM_ROUTE: _served(_SBOM)})
    monkeypatch.chdir(tmp_path)

    arguments = ["cloud", "deployment", "sbom", "dep_live", "--application", "my-agent"]
    assert main([*arguments, "--json"]) == 0

    answer = json.loads(capsys.readouterr().out)
    assert answer["operation"] == "deployment.sbom"
    assert answer["result"] == {
        "application_id": "my-agent",
        "deployment_id": "dep_live",
        "output": str(tmp_path / "dep_live.cdx.json"),
        "format": "CycloneDX",
        "spec_version": "1.5",
        "components": 2,
        "digest": "sha256:" + hashlib.sha256(_SBOM).hexdigest(),
        "bytes": len(_SBOM),
    }
    assert (tmp_path / "dep_live.cdx.json").read_bytes() == _SBOM


def test_an_existing_file_is_never_replaced_and_is_refused_before_downloading(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({_SBOM_ROUTE: _served(_SBOM)})
    output = tmp_path / "sbom.cdx.json"
    output.write_text("mine")

    arguments = ["cloud", "deployment", "sbom", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    assert _error(capsys)["category"] == "output_exists"
    assert output.read_text() == "mine"
    assert " ".join(_SBOM_ROUTE) not in fake.requests

    assert (
        main(["cloud", "deployment", "sbom", "dep_live", "--output", str(tmp_path), "--json"]) == 2
    )
    assert "is a directory" in str(_error(capsys)["message"])


@pytest.mark.parametrize("digest", ["sha256:" + "0" * 64, ""])
def test_a_digest_mismatch_or_a_missing_digest_writes_nothing(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    digest: str,
) -> None:
    cloud({_SBOM_ROUTE: _served(_SBOM, digest=digest)})
    output = tmp_path / "sbom.cdx.json"

    arguments = ["cloud", "deployment", "sbom", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    expected = "sbom_digest_mismatch" if digest else "api_response_invalid"
    assert _error(capsys)["category"] == expected
    assert not output.exists()
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        b'{"bomFormat": "SPDX"}',
        b"\xff\xfe",
        # Nested past the interpreter's recursion limit.
        b'{"bomFormat": "CycloneDX", "x": ' + b"[" * 200_000 + b"]" * 200_000 + b"}",
    ],
)
def test_a_body_that_is_not_a_cyclonedx_document_writes_nothing(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str], body: bytes
) -> None:
    cloud({_SBOM_ROUTE: _served(body)})
    output = tmp_path / "sbom.cdx.json"

    arguments = ["cloud", "deployment", "sbom", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    assert _error(capsys)["category"] == "sbom_invalid"
    assert not output.exists()
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (404, "release_not_found"),
        (404, "release_sbom_not_found"),
        (410, "release_sbom_removed"),
        (503, "release_sboms_unavailable"),
        (403, "agent_not_yours"),
        (403, "key_scope_required"),
    ],
)
def test_cloud_refusals_carry_their_code_and_message(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: int,
    code: str,
) -> None:
    cloud({_SBOM_ROUTE: _refused(status, code, "Cloud says why.")})
    output = tmp_path / "sbom.cdx.json"

    arguments = ["cloud", "deployment", "sbom", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    error = _error(capsys)
    assert error["code"] == code
    assert error["message"] == f"Cayu Cloud API returned HTTP {status}: Cloud says why."
    assert not output.exists()
    assert _leftovers(tmp_path) == []


def test_a_release_whose_image_was_never_built_says_so_in_text(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    message = (
        "Release dep_live has no SBOM: Cloud records one when it builds the Release's image, "
        "and this Release's image was never built."
    )
    cloud({_SBOM_ROUTE: _refused(404, "release_sbom_not_found", message, status="failed")})

    output = tmp_path / "sbom.cdx.json"
    assert main(["cloud", "deployment", "sbom", "dep_live", "--output", str(output)]) == 2
    assert message in capsys.readouterr().err

    assert main(["cloud", "deployment", "sbom", "dep_live", "--output", str(output), "--json"]) == 2
    error = _error(capsys)
    assert (error["code"], error["status"]) == ("release_sbom_not_found", "failed")
    assert not output.exists()


def test_a_cloud_without_the_route_says_it_cannot_download_sboms_yet(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({})
    output = tmp_path / "sbom.cdx.json"

    assert main(["cloud", "deployment", "sbom", "dep_live", "--output", str(output)]) == 2
    assert capsys.readouterr().err.strip() == (
        "Error: This Cayu Cloud can't download a release's SBOM yet."
    )
    assert _leftovers(tmp_path) == []


def test_a_release_name_that_is_not_an_id_needs_the_agent(
    cloud: Callable[..., _Cloud], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({})

    arguments = ["cloud", "deployment", "sbom", "latest", "--output", str(tmp_path / "s.json")]
    assert main([*arguments, "--json"]) == 2
    assert "--application" in str(_error(capsys)["message"])


def test_an_oversized_answer_is_refused(
    cloud: Callable[..., _Cloud],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import _cloud_release_sbom

    monkeypatch.setattr(_cloud_release_sbom, "SBOM_MAX_BYTES", 16)
    cloud({_SBOM_ROUTE: _served(_SBOM)})
    output = tmp_path / "sbom.cdx.json"

    arguments = ["cloud", "deployment", "sbom", "dep_live", "--output", str(output), "--json"]
    assert main(arguments) == 2
    assert _error(capsys)["category"] == "sbom_invalid"
    assert not output.exists()
    assert _leftovers(tmp_path) == []
