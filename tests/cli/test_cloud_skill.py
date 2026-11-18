"""The Cayu Cloud skill and the Claude Code and Codex plugin that ship in the package."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

import pytest

from cayu.cli import _build_parser, main
from cayu.cli.guide import _CLOUD_SKILL, _GUIDES, plugin_marketplace_dir

_PLUGIN = "cayu"


def _skill_text() -> str:
    return (plugin_marketplace_dir() / _PLUGIN / "skills" / "cayu-cloud" / "SKILL.md").read_text(
        encoding="utf-8"
    )


def _front_matter(text: str) -> dict[str, str]:
    assert text.startswith("---\n")
    block = text[4 : text.index("\n---\n", 4)]
    fields = {}
    for line in block.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_skill_is_a_plugin_skill_and_the_cloud_guide() -> None:
    assert _GUIDES["cloud"][0] == _CLOUD_SKILL
    fields = _front_matter(_skill_text())

    # Claude Code and Codex both need these two; the folder name is the skill's name.
    assert set(fields) == {"name", "description"}
    assert fields["name"] == "cayu-cloud"
    assert 0 < len(fields["description"]) <= 1024


def test_cloud_guide_prints_the_skill_without_its_front_matter(capsys) -> None:
    assert main(["guide", "cloud"]) == 0
    printed = capsys.readouterr().out

    body = _skill_text().split("\n---\n", 1)[1].lstrip("\n")
    assert printed == body
    assert printed.startswith("# Cayu Cloud from the command line\n")

    assert main(["guide", "cloud#2-know-where-a-command-acts"]) == 0
    assert "--organization ORG_ID" in capsys.readouterr().out


def test_plugin_dir_is_a_claude_code_and_codex_marketplace_of_the_skill(capsys) -> None:
    assert main(["guide", "--plugin-dir"]) == 0
    root = Path(capsys.readouterr().out.strip())
    assert root == plugin_marketplace_dir()

    assert main(["guide", "--plugin-dir", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["plugin_dir"] == str(root)

    claude_marketplace = _json(root / ".claude-plugin" / "marketplace.json")
    assert claude_marketplace["name"] == _PLUGIN
    assert claude_marketplace["owner"]["name"]
    [claude_entry] = claude_marketplace["plugins"]
    assert claude_entry == {**claude_entry, "name": _PLUGIN, "source": f"./{_PLUGIN}"}

    codex_marketplace = _json(root / ".agents" / "plugins" / "marketplace.json")
    [codex_entry] = codex_marketplace["plugins"]
    assert codex_entry["name"] == _PLUGIN
    assert codex_entry["source"] == {"source": "local", "path": f"./{_PLUGIN}"}

    plugin = root / _PLUGIN
    claude_manifest = _json(plugin / ".claude-plugin" / "plugin.json")
    codex_manifest = _json(plugin / ".codex-plugin" / "plugin.json")
    assert claude_manifest["name"] == codex_manifest["name"] == _PLUGIN
    # A fixed version would pin installs to it; the package version is the version.
    assert "version" not in claude_manifest and "version" not in codex_manifest
    assert codex_manifest["skills"] == "./skills/"
    skills = sorted(path.parent.name for path in (plugin / "skills").glob("*/SKILL.md"))
    assert skills == ["cayu-cloud"]


def test_plugin_dir_takes_no_topic(capsys) -> None:
    assert main(["guide", "cloud", "--plugin-dir"]) == 2
    assert "--plugin-dir takes no topic" in capsys.readouterr().err


def _commands(text: str) -> list[str]:
    """Every `cayu ...` command line the skill shows, in console blocks and inline code."""

    commands = []
    for block in re.findall(r"```console\n(.*?)```", text, re.DOTALL):
        for line in block.splitlines():
            line = line.split("  #", 1)[0].strip()
            if line.startswith("cayu "):
                commands.append(line)
    for inline in re.findall(r"`([^`\n]+)`", text):
        if inline.startswith("cayu "):
            commands.append(inline)
    # A command substitution inside another tool's command line.
    commands += re.findall(r"\$\((?:uv run )?(cayu [^)]+)\)", text)
    return commands


def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices.items())
    return {}


def _check(command: str) -> argparse.ArgumentParser:
    """Walk ``command`` down the real parser: each subcommand and each option must exist.

    Returns the parser of the command it names.
    """

    tokens = shlex.split(command)[1:]
    parser = _build_parser()
    path = ["cayu"]
    index = 0
    walking = True
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if token.startswith("--"):
            flag = token.split("=", 1)[0]
            action = parser._option_string_actions.get(flag)
            assert action is not None, f"{' '.join(path)} has no {flag}: {command}"
            if "=" not in token and action.nargs != 0:
                index += 1
            continue
        choices = _subcommands(parser)
        if walking and choices:
            assert token in choices, f"{' '.join(path)} has no command {token}: {command}"
            parser = choices[token]
            path.append(token)
        else:
            walking = False
    assert len(path) > 1, command
    return parser


def _example_lines(text: str) -> list[str]:
    """The `cayu ...` lines of the skill's console examples."""

    return [
        line.split("  #", 1)[0].strip()
        for block in re.findall(r"```console\n(.*?)```", text, re.DOTALL)
        for line in block.splitlines()
        if line.strip().startswith("cayu ")
    ]


def test_every_example_change_that_can_name_its_organization_does() -> None:
    # The `cayu cloud` commands that change something are the ones that take
    # `--organization`; the skill tells the agent to pass it on each of them.
    examples = _example_lines(_skill_text())
    changes = [line for line in examples if "--organization" in _check(line)._option_string_actions]

    assert any(line.startswith("cayu cloud --context PATH rollback") for line in changes)
    for line in changes:
        assert "--organization ORG_ID" in line, line


def test_commands_without_an_organization_option_are_not_told_to_take_one() -> None:
    for command in ("cayu inbox approve ITEM_ID", "cayu session close SESSION_ID"):
        assert "--organization" not in _check(command)._option_string_actions
    assert "It also works on reads" not in _skill_text()


def test_every_command_the_skill_names_exists_in_the_cli() -> None:
    commands = _commands(_skill_text())

    assert "cayu cloud --context PATH deploy . --organization ORG_ID" in commands
    assert any(command.startswith("cayu inbox approve") for command in commands)
    for command in commands:
        _check(command)


@pytest.mark.parametrize(
    "command",
    [
        "cayu cloud deploy . --context PATH",
        "cayu cloud deployment rewind DEPLOYMENT_ID",
        "cayu inbox approve ITEM_ID --organization ORG_ID",
    ],
)
def test_the_command_check_catches_a_wrong_command(command: str) -> None:
    with pytest.raises(AssertionError):
        _check(command)


def test_the_skill_uses_only_public_names_and_neutral_examples() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(plugin_marketplace_dir().rglob("*"))
        if path.is_file()
    )

    assert not re.search(r"(?<![\w&])#\d+\b", text)
    assert not re.search(r"\b(?:org|dep|app)_[0-9A-Za-z]{6,}", text)
    for url in re.findall(r"https?://[^\s\"'`)>]+", text):
        assert url.startswith(("https://cloud.cayu.dev", "https://github.com/example/")), url
