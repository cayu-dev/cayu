"""`home` and `[sign_in]` in cayu-cloud.toml, and what `cayu cloud init` writes for them."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from cayu.cli import _cloud_project as cloud_project
from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_project import CloudProjectManifest, CloudSignIn

_BASE = """
schema_version = 2
application = "closing-agent"
name = "Closing Agent"
version = "1.0.0"
entrypoint = "cayu serve --host 0.0.0.0 --port 8000"
capabilities = ["model.generate"]
cpu_millis = 1000
memory_mb = 2048
timeout_seconds = 900
environment = "python"
compatibility = "cayu>=0.1,<1"
policy_version = "cayu-egress-v1"
"""
_WEB = '\n[web]\ncommand = "cayu serve --host 0.0.0.0 --port 8000"\nport = 8000\n'
_PATH_ERROR = "must be a path on the Agent's own host"


def _load(top: str = "", tables: str = "") -> CloudProjectManifest:
    return CloudProjectManifest.loads(_BASE + top + _WEB + tables)


def _refused(top: str = "", tables: str = "", *, schema: int = 2) -> str:
    content = _BASE.replace("schema_version = 2", f"schema_version = {schema}")
    with pytest.raises(CloudApiError) as raised:
        CloudProjectManifest.loads(content + top + (_WEB if schema == 2 else "") + tables)
    assert raised.value.category == "manifest_invalid"
    return str(raised.value)


def test_an_agent_that_declares_nothing_keeps_its_own_sign_in() -> None:
    manifest = _load()

    assert manifest.home is None
    assert manifest.sign_in == CloudSignIn(mode="own", public_paths=())


def test_cloud_sign_in_with_home_and_public_paths() -> None:
    manifest = _load(
        'home = "/close"\n',
        '\n[sign_in]\nmode = "cloud"\npublic_paths = ["/vendor/", "/webhooks/stripe", "/"]\n',
    )

    assert manifest.home == "/close"
    assert manifest.sign_in == CloudSignIn(
        mode="cloud", public_paths=("/vendor/", "/webhooks/stripe", "/")
    )
    manifest.validate()


@pytest.mark.parametrize(
    "tables",
    ['\n[sign_in]\nmode = "own"\n', '\n[sign_in]\nmode = "cloud"\npublic_paths = []\n'],
)
def test_sign_in_without_public_paths(tables: str) -> None:
    manifest = _load(tables=tables)

    assert manifest.sign_in.public_paths == ()
    assert manifest.sign_in.mode == tomllib.loads(tables)["sign_in"]["mode"]


@pytest.mark.parametrize("home", ["/", "/close", "/app/close%20out/", "/a/./b", "/%7Euser"])
def test_home_accepts_same_host_paths(home: str) -> None:
    assert _load(f'home = "{home}"\n').home == home


@pytest.mark.parametrize(
    "path",
    [
        "",
        "close",
        "https://example.com/close",
        "//example.com/close",
        "/close?tab=1",
        "/close#top",
        "/a/../b",
        "/..",
        "/a/%2e%2e/b",
        "/a/%2E%2E",
        "/\\\\example.com",
        "/%5Cexample.com",
        "/close out",
        "/clöse",
        "/" + "a" * 1024,
    ],
)
def test_home_and_public_paths_refuse_anything_but_a_same_host_path(path: str) -> None:
    value = f'"{path}"'
    home_message = _refused(f"home = {value}\n")
    public_message = _refused(tables=f'\n[sign_in]\nmode = "cloud"\npublic_paths = [{value}]\n')

    assert home_message.startswith("cayu-cloud.toml home " + _PATH_ERROR)
    assert public_message.startswith("Each [sign_in] public_paths entry " + _PATH_ERROR)
    assert '".." segment' in home_message and "query, fragment" in home_message


def test_home_must_be_a_string() -> None:
    assert _PATH_ERROR in _refused("home = 1\n")


@pytest.mark.parametrize(
    "tables",
    [
        "\n[sign_in]\n",
        '\n[sign_in]\nmode = "Cloud"\n',
        '\n[sign_in]\nmode = "workos"\n',
        "\n[sign_in]\nmode = true\n",
        '\n[sign_in]\npublic_paths = ["/vendor/"]\n',
    ],
)
def test_sign_in_mode_must_be_cloud_or_own(tables: str) -> None:
    assert _refused(tables=tables) == (
        '[sign_in] mode must be "cloud" (Cayu Cloud signs people in) or "own" '
        "(the Agent's own login)."
    )


@pytest.mark.parametrize("public_paths", ['["/vendor/"]', "[]"])
def test_public_paths_are_refused_with_own_sign_in(public_paths: str) -> None:
    message = _refused(tables=f'\n[sign_in]\nmode = "own"\npublic_paths = {public_paths}\n')

    assert message.startswith('[sign_in] public_paths is allowed only with mode = "cloud"')


@pytest.mark.parametrize(
    "public_paths",
    ['"/vendor/"', '["/vendor/", "/vendor/"]', "[" + ", ".join(['"/a"'] * 51) + "]"],
)
def test_public_paths_must_be_a_short_array_of_different_prefixes(public_paths: str) -> None:
    message = _refused(tables=f'\n[sign_in]\nmode = "cloud"\npublic_paths = {public_paths}\n')

    assert message == (
        "[sign_in] public_paths must be an array of at most 50 different path prefixes."
    )


def test_public_paths_accept_fifty_prefixes() -> None:
    paths = ", ".join(f'"/p{index}/"' for index in range(50))
    manifest = _load(tables=f'\n[sign_in]\nmode = "cloud"\npublic_paths = [{paths}]\n')

    assert len(manifest.sign_in.public_paths) == 50


def test_sign_in_refuses_unknown_fields_like_other_tables() -> None:
    message = _refused(tables='\n[sign_in]\nmode = "cloud"\npublic_path = ["/vendor/"]\n')

    assert message == "sign_in contains unsupported fields: public_path."


def test_sign_in_must_be_a_table() -> None:
    message = _refused('sign_in = "cloud"\n')

    assert message == "cayu-cloud.toml is missing a required field or contains the wrong type."


@pytest.mark.parametrize("declaration", ['home = "/close"\n', '[sign_in]\nmode = "cloud"\n'])
def test_home_and_sign_in_require_schema_version_2(declaration: str) -> None:
    message = _refused(declaration, schema=1)

    assert message == "home and [sign_in] require cayu-cloud.toml schema_version 2."


def test_home_and_sign_in_are_not_sent_in_the_deployment_request() -> None:
    plain = _load()
    declared = _load('home = "/close"\n', '\n[sign_in]\nmode = "cloud"\npublic_paths = ["/v/"]\n')

    def payload(manifest: CloudProjectManifest) -> dict[str, object]:
        return manifest.deployment_payload(
            repository="https://github.com/example/agent", revision="a" * 40
        )

    assert payload(declared) == payload(plain)


def _write_source(root: Path, manifest: str, *, name: str = "cayu-cloud.toml") -> None:
    (root / "pyproject.toml").write_text('[project]\nname = "closing-agent"\nversion = "1.0"\n')
    (root / "uv.lock").write_text("version = 1\n")
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(manifest)


@pytest.mark.parametrize(
    "declaration",
    ['home = "/close"\n' + _WEB, _WEB + '[sign_in]\nmode = "cloud"\n'],
)
def test_custom_manifest_path_refuses_home_and_cloud_sign_in(
    tmp_path: Path, declaration: str
) -> None:
    _write_source(tmp_path, _BASE + declaration, name="deploy/cloud.toml")

    with pytest.raises(CloudApiError, match=re.escape("cayu-cloud.toml at the deployment root")):
        cloud_project.resolve_project(
            str(tmp_path), manifest_path=Path("deploy/cloud.toml"), revision=None
        )


def test_custom_manifest_path_accepts_own_sign_in(tmp_path: Path) -> None:
    _write_source(tmp_path, _BASE + _WEB + '[sign_in]\nmode = "own"\n', name="cloud.toml")

    project = cloud_project.resolve_project(
        str(tmp_path), manifest_path=Path("cloud.toml"), revision=None
    )

    assert project.manifest.sign_in == CloudSignIn()


_ROOT_DECLARES = "The deployment root's cayu-cloud.toml declares"


@pytest.mark.parametrize(
    "root_declaration",
    [
        _WEB + '[sign_in]\nmode = "cloud"\n',
        _WEB + '[sign_in]\nmode = "cloud"\npublic_paths = ["/webhooks/"]\n',
        'home = "/close"\n' + _WEB,
        'home = "/close"\n' + _WEB + '[sign_in]\nmode = "own"\n',
    ],
    ids=["cloud", "cloud-public-paths", "home", "home-own"],
)
@pytest.mark.parametrize("custom_sign_in", ["", '[sign_in]\nmode = "own"\n'], ids=["none", "own"])
def test_custom_manifest_path_refuses_a_root_manifest_that_declares_home_or_cloud_sign_in(
    tmp_path: Path, root_declaration: str, custom_sign_in: str
) -> None:
    # Cloud reads sign-in and home only from the uploaded root file, so deploying the
    # custom manifest would silently use the root file's settings instead.
    _write_source(tmp_path, _BASE + root_declaration)
    _write_source(tmp_path, _BASE + _WEB + custom_sign_in, name="deploy/own.toml")

    with pytest.raises(CloudApiError, match=re.escape(_ROOT_DECLARES)) as raised:
        cloud_project.resolve_project(
            str(tmp_path), manifest_path=Path("deploy/own.toml"), revision=None
        )
    assert raised.value.category == "manifest_invalid"
    assert "deploy/own.toml" in str(raised.value)


@pytest.mark.parametrize("root_sign_in", ["", '[sign_in]\nmode = "own"\n'], ids=["none", "own"])
def test_custom_manifest_path_accepts_a_root_manifest_with_own_sign_in(
    tmp_path: Path, root_sign_in: str
) -> None:
    _write_source(tmp_path, _BASE + _WEB + root_sign_in)
    _write_source(tmp_path, _BASE + _WEB, name="deploy/own.toml")

    project = cloud_project.resolve_project(
        str(tmp_path), manifest_path=Path("deploy/own.toml"), revision=None
    )

    assert project.manifest.sign_in == CloudSignIn()


def test_custom_manifest_path_refuses_an_unreadable_root_manifest(tmp_path: Path) -> None:
    _write_source(tmp_path, "schema_version = [")
    _write_source(tmp_path, _BASE + _WEB, name="deploy/own.toml")

    with pytest.raises(CloudApiError, match="Could not read the deployment root") as raised:
        cloud_project.resolve_project(
            str(tmp_path), manifest_path=Path("deploy/own.toml"), revision=None
        )
    assert raised.value.category == "manifest_invalid"


def test_custom_manifest_path_ignores_a_root_manifest_left_out_of_the_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_source(tmp_path, _BASE + _WEB + '[sign_in]\nmode = "cloud"\n')
    _write_source(tmp_path, _BASE + _WEB, name="deploy/own.toml")
    listed = (Path("pyproject.toml"), Path("uv.lock"), Path("deploy/own.toml"))
    monkeypatch.setattr(cloud_project, "_listed_project_files", lambda _root: listed)

    project = cloud_project.resolve_project(
        str(tmp_path), manifest_path=Path("deploy/own.toml"), revision=None
    )

    assert project.manifest.sign_in == CloudSignIn()


@pytest.mark.parametrize(
    ("uploaded", "edited", "refused"),
    [("cloud", "own", True), ("own", "cloud", False)],
    ids=["uploaded-cloud", "uploaded-own"],
)
def test_custom_manifest_path_checks_the_uploaded_root_manifest_not_a_later_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uploaded: str, edited: str, refused: bool
) -> None:
    # Cloud reads the root file in the bundle; an edit after archiving must not change
    # what the check sees.
    _write_source(tmp_path, _BASE + _WEB + f'[sign_in]\nmode = "{uploaded}"\n')
    _write_source(tmp_path, _BASE + _WEB, name="deploy/own.toml")
    archive = cloud_project._archive_project

    def archive_then_edit(root: Path, *, required_file: Path) -> bytes:
        bundle = archive(root, required_file=required_file)
        (root / "cayu-cloud.toml").write_text(_BASE + _WEB + f'[sign_in]\nmode = "{edited}"\n')
        return bundle

    monkeypatch.setattr(cloud_project, "_archive_project", archive_then_edit)

    def deploy() -> object:
        return cloud_project.resolve_project(
            str(tmp_path), manifest_path=Path("deploy/own.toml"), revision=None
        )

    if refused:
        with pytest.raises(CloudApiError, match=re.escape(_ROOT_DECLARES)):
            deploy()
    else:
        deploy()


@pytest.mark.parametrize(
    ("target", "refused"),
    [("deploy/own.toml", False), ("deploy/cloud.toml", True)],
    ids=["own", "cloud"],
)
def test_a_symlinked_root_manifest_is_checked_through_the_uploaded_target(
    tmp_path: Path, target: str, refused: bool
) -> None:
    _write_source(tmp_path, _BASE + _WEB, name="deploy/own.toml")
    _write_source(tmp_path, _BASE + _WEB + '[sign_in]\nmode = "cloud"\n', name="deploy/cloud.toml")
    (tmp_path / "cayu-cloud.toml").symlink_to(target)

    def deploy() -> object:
        return cloud_project.resolve_project(
            str(tmp_path), manifest_path=Path("deploy/own.toml"), revision=None
        )

    if refused:
        with pytest.raises(CloudApiError, match=re.escape(_ROOT_DECLARES)):
            deploy()
    else:
        deploy()


@pytest.mark.parametrize(
    ("root_manifest", "refused"),
    [
        (_BASE + _WEB + '[sign_in]\nmode = "cloud"\n', True),
        (_BASE + _WEB + '[sign_in]\nmode = "own"\n', False),
        (None, False),
    ],
    ids=["cloud", "own", "absent"],
)
def test_repository_custom_manifest_path_checks_the_root_manifest_sign_in(
    monkeypatch: pytest.MonkeyPatch, root_manifest: str | None, refused: bool
) -> None:
    source, revision = "https://github.com/example/agent", "a" * 40
    reads: list[str] = []

    def github_file(repository: str, *, revision: str, path: str) -> str:
        assert (repository, revision, path) == (source, "a" * 40, "deploy/own.toml")
        return _BASE + _WEB

    def optional_github_file(repository: str, *, revision: str, path: str) -> str | None:
        assert (repository, revision) == (source, "a" * 40)
        reads.append(path)
        return root_manifest

    monkeypatch.setattr(cloud_project, "_github_file", github_file)
    monkeypatch.setattr(cloud_project, "_optional_github_file", optional_github_file)

    def resolve() -> cloud_project.ResolvedCloudProject:
        return cloud_project.resolve_project(
            source, manifest_path=Path("deploy/own.toml"), revision=revision
        )

    if refused:
        with pytest.raises(CloudApiError, match=re.escape(_ROOT_DECLARES)) as raised:
            resolve()
        assert raised.value.category == "manifest_invalid"
    else:
        assert resolve().manifest.sign_in == CloudSignIn()
    assert reads == ["cayu-cloud.toml"]


def test_root_manifest_carries_home_and_cloud_sign_in(tmp_path: Path) -> None:
    _write_source(tmp_path, _BASE + 'home = "/close"\n' + _WEB + '[sign_in]\nmode = "cloud"\n')

    project = cloud_project.resolve_project(str(tmp_path), manifest_path=None, revision=None)

    assert project.manifest.home == "/close"
    assert project.manifest.sign_in == CloudSignIn(mode="cloud")


def test_init_writes_cloud_sign_in_and_no_home_for_a_worker(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "mail-agent"\nversion = "0.4.0"\n\n'
        '[project.scripts]\nmail-agent = "mail_agent.cli:main"\n'
    )

    cloud_project.initialize_project(tmp_path)

    assert (tmp_path / "cayu-cloud.toml").read_text() == (
        "schema_version = 2\n"
        'application = "mail-agent"\n'
        'name = "Mail Agent"\n'
        'version = "0.4.0"\n'
        'entrypoint = "mail-agent"\n'
        'capabilities = ["model.generate"]\n'
        "cpu_millis = 1000\n"
        "memory_mb = 2048\n"
        "timeout_seconds = 900\n"
        'environment = "python"\n'
        'compatibility = "cayu>=0.1,<1"\n'
        'policy_version = "cayu-egress-v1"\n'
        "\n"
        "[worker]\n"
        'command = "mail-agent"\n'
        "\n"
        "[sign_in]\n"
        'mode = "cloud"\n'
    )


def test_init_writes_cloud_sign_in_and_no_home_for_cayu_serve(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "research-agent"\nversion = "1.2.3"\n'
        'dependencies = ["cayu[postgres,server]==0.13.0"]\n\n'
        '[tool.cayu]\nfactory = "research_agent:build_app"\n\n'
        '[tool.cayu.serve]\nauth = "cayu.server.environment_auth:OPERATOR_BASIC_AUTH"\n'
    )

    cloud_project.initialize_project(tmp_path)

    content = (tmp_path / "cayu-cloud.toml").read_text()
    assert content.endswith(
        '\n[web]\ncommand = "cayu serve --host 0.0.0.0 --port 8000"\nport = 8000\n'
        '\n[sign_in]\nmode = "cloud"\n'
    )
    assert "home" not in tomllib.loads(content)
    manifest = CloudProjectManifest.loads(content)
    assert manifest.home is None
    assert manifest.sign_in == CloudSignIn(mode="cloud")
