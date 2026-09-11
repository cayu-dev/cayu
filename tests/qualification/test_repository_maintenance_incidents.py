"""Runbook emission and real CLI parser conformance, not an operator live trial."""

import argparse
import re
import shlex

import pytest

from cayu.cli.recovery import add_recovery_parser
from tests.qualification.repository_maintenance_application import maintenance_project_files


@pytest.mark.parametrize("database", ["sqlite", "postgres"])
def test_operator_runbook_uses_existing_recovery_commands(database):
    files = maintenance_project_files(database=database)
    guide = files["operations/maintenance-incidents.md"]
    parser = argparse.ArgumentParser()
    add_recovery_parser(parser.add_subparsers(dest="command", required=True))
    commands = []
    for block in re.findall(r"```sh\n(.*?)```", guide, re.DOTALL):
        if "cayu recovery " not in block:
            continue
        words = shlex.split(block.replace("\\\n", ""))
        argv = words[words.index("cayu") + 1 : words.index(">")]
        commands.append(parser.parse_args(argv))
    assert len(commands) == 2
    plan, execute = commands
    assert plan.recovery_command == "plan" and plan.limit == 1
    assert plan.session_ids == ["$CODING_SESSION_ID"]
    assert execute.recovery_command == "execute"
    assert execute.plan_file == "/operator-evidence/plan.json"
    assert execute.decisions == "/operator-evidence/decisions.json"
    assert execute.execution_id == "$RECOVERY_EXECUTION_ID"
    assert "does not\nship a general task-reset or unattended recovery worker" in guide
    assert "unknown cleanup\noutcome fails production qualification" in guide
    if database == "postgres":
        assert "operations/maintenance-incidents.md" in files["deployment/README.md"]
