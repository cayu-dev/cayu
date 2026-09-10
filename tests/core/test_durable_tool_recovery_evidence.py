from __future__ import annotations

import pytest

from cayu.core.tools import DurableToolRecoveryEvidence, ToolResult


@pytest.mark.parametrize("disposition", ["confirmed", "not_started", "unresolved"])
def test_native_evidence_detaches_the_result_without_inferring_disposition(disposition):
    result = ToolResult(content="diagnostic", structured={"status": "completed"})
    evidence = DurableToolRecoveryEvidence(disposition, result)
    assert evidence.disposition == disposition
    assert evidence.result == result
    assert evidence.result is not result
    object.__setattr__(result, "content", "changed by extension")
    assert evidence.result.content == "diagnostic"


@pytest.mark.parametrize("disposition", [None, True, 1, "", "future", "completed"])
def test_native_evidence_rejects_unknown_dispositions(disposition):
    with pytest.raises(ValueError, match="invalid evidence disposition"):
        DurableToolRecoveryEvidence(disposition, ToolResult())


def test_native_evidence_requires_an_exact_result_type():
    class ResultSubclass(ToolResult):
        pass

    for value in (None, {"content": "untrusted"}, ResultSubclass()):
        with pytest.raises(TypeError, match="exact ToolResult"):
            DurableToolRecoveryEvidence("confirmed", value)
