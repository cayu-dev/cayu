"""Keep the documented operator policy action family exhaustive."""

import re
from pathlib import Path
from typing import get_args

from cayu.runtime.browser_control import BrowserControlAction


def test_documented_browser_policy_actions_match_runtime():
    documentation = (
        Path(__file__).resolve().parents[2] / "docs" / "browser-session.md"
    ).read_text()
    actions = documentation.split("The action\nset is ", 1)[1].split(
        ". Decisions do not receive", 1
    )[0]
    names = re.findall(r"`([a-z_]+)`", actions)
    assert len(names) == len(set(names))
    assert set(names) == set(get_args(BrowserControlAction))
