"""Source-owned consent stays exact through CLI validation, packaging, and init."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cayu.cli import _cloud_project as project
from cayu.cli import main
from cayu.cli._cloud_api import CloudApiError

_BASE = """schema_version = 2
application = "filing-agent"
name = "Filing Agent"
version = "1.0.0"
entrypoint = "python -m agent"
capabilities = ["model.generate"]
cpu_millis = 1000
memory_mb = 2048
timeout_seconds = 900
environment = "python"
compatibility = "cayu>=0.1"
policy_version = "v1"
"""
_RULE = {"agent_name": "issue_filer", "tool_name": "file_issue", "reason": "May lose continuation."}


def _tables(*rules: dict[str, object]) -> str:
    return "".join(
        "\n[[session_replacement.approvals]]\n"
        + "\n".join(
            f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in rule.items()
        )
        + "\n"
        for rule in rules
    )


def _source(root: Path, content: str, name: str = "cayu-cloud.toml") -> None:
    (root / "pyproject.toml").write_text('[project]\nname = "filing-agent"\nversion = "1.0.0"\n')
    (root / "uv.lock").write_text("version = 1\n")
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_absence_and_canonical_exact_rules() -> None:
    assert project.CloudProjectManifest.loads(_BASE).session_replacement is None
    manifest = project.CloudProjectManifest.loads(
        _BASE + _tables(_RULE, {**_RULE, "agent_name": "A_1.x:y-z", "reason": "Unicode café"})
    )
    assert manifest.session_replacement is not None
    assert [
        (rule.agent_name, rule.tool_name, rule.reason) for rule in manifest.session_replacement
    ] == [
        ("A_1.x:y-z", "file_issue", "Unicode café"),
        ("issue_filer", "file_issue", "May lose continuation."),
    ]
    assert manifest.deployment_payload(repository="https://github.com/a/b", revision="a" * 40) == (
        project.CloudProjectManifest.loads(_BASE).deployment_payload(
            repository="https://github.com/a/b", revision="a" * 40
        )
    )


@pytest.mark.parametrize("field", ["agent_name", "tool_name"])
@pytest.mark.parametrize(
    "value", ["*", "a*", "", " a", "a ", "a\nb", "a\tb", "x" * 129, True, 1, []]
)
def test_rejects_ambiguous_selector(field: str, value: object) -> None:
    with pytest.raises(CloudApiError, match="session_replacement"):
        project.CloudProjectManifest.loads(_BASE + _tables({**_RULE, field: value}))


@pytest.mark.parametrize(
    "reason", ["", " ", " padded", "padded ", "line\nbreak", "tab\t", "a" * 513, True, 1, []]
)
def test_rejects_invalid_reason_without_rendering_value(reason: object) -> None:
    with pytest.raises(CloudApiError, match="session_replacement"):
        project.CloudProjectManifest.loads(_BASE + _tables({**_RULE, "reason": reason}))


@pytest.mark.parametrize(
    "tables",
    [
        "\n[session_replacement]\n",
        "\n[session_replacement]\napprovals = []\n",
        "\n[session_replacement]\napprovals = true\n",
        "\n[session_replacement]\napprovals = [true]\n",
        "\nsession_replacement = true\n",
        _tables({**_RULE, "wildcard": True}),
        _tables({"agent_name": "issue_filer", "tool_name": "file_issue"}),
        _tables(_RULE, {**_RULE, "reason": "Conflicting reason."}),
        _tables(_RULE) + "\n[session_replacement]\nunknown = true\n",
    ],
)
def test_rejects_unknown_empty_wrong_type_and_duplicate_rules(tables: str) -> None:
    with pytest.raises(CloudApiError, match="session_replacement"):
        project.CloudProjectManifest.loads(_BASE + tables)


@pytest.mark.parametrize("schema", ["1", "true", "[]"])
def test_requires_integer_schema_2(schema: str) -> None:
    with pytest.raises(CloudApiError):
        project.CloudProjectManifest.loads(
            _BASE.replace("schema_version = 2", f"schema_version = {schema}") + _tables(_RULE)
        )


def test_maximum_bounds_and_over_limit() -> None:
    rules = [
        {**_RULE, "agent_name": str(i) + "a" * 126, "tool_name": "t" * 128, "reason": "r" * 512}
        for i in range(20)
    ]
    assert (
        len(project.CloudProjectManifest.loads(_BASE + _tables(*rules)).session_replacement or ())
        == 20
    )
    with pytest.raises(CloudApiError, match="session_replacement"):
        project.CloudProjectManifest.loads(_BASE + _tables(*rules, _RULE))


def test_source_bundle_roundtrip_retains_declaration(tmp_path: Path) -> None:
    content = _BASE + _tables(_RULE)
    _source(tmp_path, content)
    resolved = project.resolve_project(str(tmp_path), manifest_path=None, revision=None)
    uploaded = project._uploaded_root_manifest(resolved.bundle or b"")
    assert uploaded == content
    assert (
        project.CloudProjectManifest.loads(uploaded).session_replacement
        == resolved.manifest.session_replacement
    )


@pytest.mark.parametrize("where", ["root", "selected"])
def test_custom_manifest_cannot_hide_or_supply_consent(tmp_path: Path, where: str) -> None:
    _source(tmp_path, _BASE + (_tables(_RULE) if where == "root" else ""))
    _source(tmp_path, _BASE + (_tables(_RULE) if where == "selected" else ""), "deploy.toml")
    with pytest.raises(CloudApiError, match="session_replacement"):
        project.resolve_project(str(tmp_path), manifest_path=Path("deploy.toml"), revision=None)


def test_repository_custom_manifest_cannot_hide_root_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(project, "_github_file", lambda *args, **kwargs: _BASE)
    monkeypatch.setattr(
        project, "_optional_github_file", lambda *args, **kwargs: _BASE + _tables(_RULE)
    )
    with pytest.raises(CloudApiError, match="session_replacement"):
        project.resolve_project(
            "https://github.com/example/agent", manifest_path=Path("deploy.toml"), revision="a" * 40
        )


def test_init_force_preserves_canonical_consent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _source(tmp_path, _BASE + _tables(_RULE, {**_RULE, "agent_name": "a"}))
    before = project.CloudProjectManifest.load(tmp_path / "cayu-cloud.toml").session_replacement
    assert main(["cloud", "init", str(tmp_path), "--force"]) == 0
    capsys.readouterr()
    content = (tmp_path / "cayu-cloud.toml").read_text()
    assert content.index('agent_name = "a"') < content.index('agent_name = "issue_filer"')
    assert project.CloudProjectManifest.loads(content).session_replacement == before


def test_init_does_not_invent_consent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _source(tmp_path, _BASE)
    (tmp_path / "cayu-cloud.toml").unlink()
    assert main(["cloud", "init", str(tmp_path)]) == 0
    capsys.readouterr()
    assert (
        project.CloudProjectManifest.load(tmp_path / "cayu-cloud.toml").session_replacement is None
    )


def test_init_force_rejects_ambiguous_consent_before_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content = _BASE + _tables({**_RULE, "tool_name": "*"})
    _source(tmp_path, content)
    original_project = (tmp_path / "pyproject.toml").read_bytes()
    assert main(["cloud", "init", str(tmp_path), "--force"]) != 0
    capsys.readouterr()
    assert (tmp_path / "cayu-cloud.toml").read_text() == content
    assert (tmp_path / "pyproject.toml").read_bytes() == original_project


@pytest.mark.parametrize(
    "name", ["_private", "café", "internal space", "robot🤖", ".a", "a/b", "e\u0301", "a?b"]
)
def test_supported_exact_unicode_and_punctuation_names(name: str) -> None:
    manifest = project.CloudProjectManifest.loads(
        _BASE + _tables({**_RULE, "agent_name": name, "tool_name": name})
    )
    assert manifest.session_replacement is not None
    assert manifest.session_replacement[0].agent_name == name
    assert manifest.session_replacement[0].tool_name == name


@pytest.mark.parametrize("bad_first", [False, True])
def test_ambiguous_rule_does_not_get_hidden_by_valid_sibling(bad_first: bool) -> None:
    rules = [_RULE, {**_RULE, "tool_name": True}]
    if bad_first:
        rules.reverse()
    with pytest.raises(CloudApiError, match="session_replacement"):
        project.CloudProjectManifest.loads(_BASE + _tables(*rules))


def test_exact_identity_is_not_unicode_normalized() -> None:
    declared = project.CloudProjectManifest.loads(
        _BASE + _tables({**_RULE, "agent_name": "é"}, {**_RULE, "agent_name": "e\u0301"})
    )
    assert declared.session_replacement is not None
    assert len(declared.session_replacement) == 2


def test_custom_manifest_checks_archived_consent_not_later_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source(tmp_path, _BASE + _tables(_RULE))
    _source(tmp_path, _BASE, "deploy.toml")
    archive = project._archive_project

    def archive_then_edit(root: Path, *, required_file: Path) -> bytes:
        bundle = archive(root, required_file=required_file)
        (root / "cayu-cloud.toml").write_text(_BASE)
        return bundle

    monkeypatch.setattr(project, "_archive_project", archive_then_edit)
    with pytest.raises(CloudApiError, match="session_replacement"):
        project.resolve_project(str(tmp_path), manifest_path=Path("deploy.toml"), revision=None)


def test_symlink_root_cannot_hide_declaration(tmp_path: Path) -> None:
    _source(tmp_path, _BASE + _tables(_RULE), "consent.toml")
    _source(tmp_path, _BASE, "deploy.toml")
    (tmp_path / "cayu-cloud.toml").symlink_to("consent.toml")
    with pytest.raises(CloudApiError, match="session_replacement"):
        project.resolve_project(str(tmp_path), manifest_path=Path("deploy.toml"), revision=None)


def test_repository_root_accepts_exact_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    content = _BASE + _tables(_RULE)
    monkeypatch.setattr(project, "_github_file", lambda *args, **kwargs: content)
    resolved = project.resolve_project(
        "https://github.com/example/agent", manifest_path=None, revision="a" * 40
    )
    assert (
        resolved.manifest.session_replacement
        == project.CloudProjectManifest.loads(content).session_replacement
    )


def test_post_construction_wrong_type_is_diagnostically_safe(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    recwarn: pytest.WarningsRecorder,
) -> None:
    from dataclasses import replace

    manifest = project.CloudProjectManifest.loads(_BASE + _tables(_RULE))
    assert manifest.session_replacement is not None

    class Secret:
        def __repr__(self) -> str:
            return "SECRET-CANARY"

        __str__ = __repr__

    rule = manifest.session_replacement[0]
    object.__setattr__(rule, "tool_name", Secret())
    with pytest.raises(CloudApiError) as raised:
        replace(manifest, session_replacement=(rule,)).validate()
    output = capsys.readouterr()
    assert (
        "SECRET-CANARY"
        not in str(raised.value) + repr(raised.value) + output.out + output.err + caplog.text
    )
    assert not recwarn.list


def test_init_preserves_non_bmp_names_and_escaped_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rule = {
        "agent_name": "robot🤖",
        "tool_name": "工具🛠",
        "reason": 'May lose "approval" \\ context 🤖.',
    }
    _source(tmp_path, _BASE + _tables(rule))
    original = project.CloudProjectManifest.load(tmp_path / "cayu-cloud.toml").session_replacement
    assert main(["cloud", "init", str(tmp_path), "--force"]) == 0
    capsys.readouterr()
    assert (
        project.CloudProjectManifest.load(tmp_path / "cayu-cloud.toml").session_replacement
        == original
    )


@pytest.mark.parametrize(
    "descriptor",
    [
        'schema_version = 2\napplication = "x"\n',
        _BASE.replace('application = "filing-agent"', "application = true"),
        _BASE + "unknown_future_field = true\n",
        _BASE.replace("schema_version = 2", "schema_version = 999"),
    ],
)
@pytest.mark.parametrize("consent", [False, True])
def test_init_force_regenerates_invalid_descriptor_preserving_only_consent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], descriptor: str, consent: bool
) -> None:
    _source(tmp_path, descriptor + (_tables(_RULE) if consent else ""))
    assert main(["cloud", "init", str(tmp_path), "--force"]) == 0
    capsys.readouterr()
    manifest = project.CloudProjectManifest.load(tmp_path / "cayu-cloud.toml")
    assert manifest.application == "filing-agent"
    expected = project.CloudProjectManifest.loads(_BASE + (_tables(_RULE) if consent else ""))
    assert manifest.session_replacement == expected.session_replacement


@pytest.mark.parametrize(
    "content",
    ["schema_version = [\n", _BASE + "\n[[session_replacement.approvals]\n"],
)
def test_init_force_unparseable_consent_is_actionable_without_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    _source(tmp_path, content)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert main(["cloud", "init", str(tmp_path), "--force"]) == 2
    captured = capsys.readouterr()
    assert "consent" in captured.out + captured.err
    assert "repair" in captured.out + captured.err
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize(
    "tables",
    [_tables(_RULE, _RULE), _tables({**_RULE, "tool_name": True}), "[session_replacement]\n"],
)
def test_init_force_validates_consent_even_when_other_fields_are_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], tables: str
) -> None:
    content = "schema_version = 999\n" + tables
    _source(tmp_path, content)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert main(["cloud", "init", str(tmp_path), "--force"]) == 2
    captured = capsys.readouterr()
    assert "session_replacement" in captured.out + captured.err
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
